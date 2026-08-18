import re
from pathlib import Path

import streamlit as st
from google import genai
from google.genai import types
from pypdf import PdfReader

st.set_page_config(page_title="ROMUS.IA", page_icon="🔥", layout="centered")

BASE_DIR = Path(__file__).parent
KNOWLEDGE_DIR = BASE_DIR / "base_conhecimento"
MODEL = "gemini-3.5-flash-lite"
MAX_CHUNKS = 5
MIN_SCORE = 2
LIMITE_INSUFICIENTE = "__BASE_INSUFICIENTE__"

SYSTEM_PROMPT = """
Você é o ROMUS.IA, assistente técnico e objetivo.
Responda em português do Brasil.
Não invente informações.
Quando usar a base local, use somente o conteúdo fornecido por ela.
Não complete lacunas com conhecimento próprio.
Quando usar pesquisa externa, deixe isso claro e priorize fontes oficiais.
Em legislação e normas, seja rigoroso com números, datas, itens e redação.
"""


def normalizar(texto: str) -> list[str]:
    texto = texto.lower()
    texto = re.sub(r"[^\w\s]", " ", texto, flags=re.UNICODE)
    return [p for p in texto.split() if len(p) > 2]


def ler_arquivo(caminho: Path) -> str:
    if caminho.suffix.lower() in {".txt", ".md"}:
        return caminho.read_text(encoding="utf-8", errors="ignore")
    if caminho.suffix.lower() == ".pdf":
        reader = PdfReader(str(caminho))
        return "\n".join((pagina.extract_text() or "") for pagina in reader.pages)
    return ""


def carregar_documentos() -> list[dict]:
    KNOWLEDGE_DIR.mkdir(parents=True, exist_ok=True)
    documentos = []
    for caminho in KNOWLEDGE_DIR.rglob("*"):
        if caminho.is_file() and caminho.suffix.lower() in {".txt", ".md", ".pdf"}:
            try:
                texto = ler_arquivo(caminho).strip()
                if texto:
                    documentos.append({
                        "arquivo": caminho.relative_to(KNOWLEDGE_DIR).as_posix(),
                        "texto": texto,
                    })
            except Exception:
                continue
    return documentos


def dividir_em_blocos(texto: str, tamanho: int = 1800, sobreposicao: int = 250) -> list[str]:
    texto = re.sub(r"\n{3,}", "\n\n", texto).strip()
    if len(texto) <= tamanho:
        return [texto]
    blocos = []
    inicio = 0
    while inicio < len(texto):
        fim = min(inicio + tamanho, len(texto))
        bloco = texto[inicio:fim].strip()
        if bloco:
            blocos.append(bloco)
        if fim >= len(texto):
            break
        inicio = max(fim - sobreposicao, inicio + 1)
    return blocos


@st.cache_data(show_spinner=False)
def construir_indice() -> list[dict]:
    indice = []
    for documento in carregar_documentos():
        for bloco in dividir_em_blocos(documento["texto"]):
            indice.append({
                "arquivo": documento["arquivo"],
                "texto": bloco,
                "termos": set(normalizar(bloco)),
            })
    return indice


def buscar_base(pergunta: str) -> list[dict]:
    termos = set(normalizar(pergunta))
    if not termos:
        return []
    resultados = []
    for item in construir_indice():
        score = len(termos.intersection(item["termos"]))
        if score >= MIN_SCORE:
            resultados.append({**item, "score": score})
    resultados.sort(key=lambda x: x["score"], reverse=True)
    return resultados[:MAX_CHUNKS]


def extrair_numero_item(pergunta: str) -> str | None:
    m = re.search(r"\b(?:item\s+)?(\d+(?:\.\d+){2,})\b", pergunta.lower())
    return m.group(1) if m else None


def localizar_documento_para_item(pergunta: str) -> dict | None:
    candidatos = carregar_documentos()
    numeros_it = re.findall(r"\bit\s*(?:n[ºo°]?\s*)?(\d{1,2})\s*[-/]\s*25\b", pergunta.lower())
    if not numeros_it:
        numeros_it = re.findall(r"\bit\s*(?:n[ºo°]?\s*)?(\d{1,2})\s*[/ -]?\s*2025\b", pergunta.lower())
    if numeros_it:
        numero = int(numeros_it[0])
        for doc in candidatos:
            if re.search(rf"\bit\s*(?:n[ºo°]?\s*)?0?{numero}\s*[-/]\s*25\b", doc["arquivo"].lower()):
                return doc
    termos = set(normalizar(pergunta))
    melhor = None
    melhor_score = 0
    for doc in candidatos:
        score = len(termos.intersection(set(normalizar(doc["arquivo"]))))
        if score > melhor_score:
            melhor, melhor_score = doc, score
    return melhor


def extrair_item_literal(texto: str, numero_item: str) -> str | None:
    numero = re.escape(numero_item)
    padrao = re.compile(rf"(?ms)^\s*{numero}\b.*?(?=^\s*\d+(?:\.\d+){{2,}}\b|\Z)")
    encontrado = padrao.search(texto)
    if not encontrado:
        padrao = re.compile(rf"(?ms)(?<!\d){numero}\b.*?(?=\n\s*\d+(?:\.\d+){{2,}}\b|\Z)")
        encontrado = padrao.search(texto)
    return encontrado.group(0).strip() if encontrado else None


def pedido_literal(pergunta: str) -> bool:
    termos = set(normalizar(pergunta))
    return bool(termos.intersection({"transcreva", "transcrever", "literalmente", "literal", "exatamente", "exato"}))


def responder_local_sem_gemini(pergunta: str, resultados: list[dict]) -> str | None:
    if pedido_literal(pergunta):
        numero = extrair_numero_item(pergunta)
        if numero:
            doc = localizar_documento_para_item(pergunta)
            if doc:
                trecho = extrair_item_literal(doc["texto"], numero)
                if trecho:
                    return trecho

    termos = set(normalizar(pergunta))
    linhas = []
    vistos = set()
    for item in resultados:
        for linha in item["texto"].splitlines():
            linha = linha.strip()
            if not linha or linha in vistos:
                continue
            score = len(termos.intersection(set(normalizar(linha))))
            if score > 0:
                linhas.append((score, linha, item["arquivo"]))
                vistos.add(linha)
    linhas.sort(key=lambda x: x[0], reverse=True)
    if not linhas:
        return None
    melhores = linhas[:12]
    return "\n".join(f"{linha}" for _, linha, _ in melhores)


def gerar_resposta_local(client, pergunta: str, resultados: list[dict]):
    contexto = "\n\n".join(f"[FONTE LOCAL: {i['arquivo']}]\n{i['texto']}" for i in resultados)
    prompt = f"""
{SYSTEM_PROMPT}

MODO: BASE LOCAL.
Responda exclusivamente com o conteúdo abaixo.
Se não houver conteúdo suficiente, responda somente {LIMITE_INSUFICIENTE}.
Não use internet neste modo.
Quando houver fórmula ou critério de dimensionamento na base, faça o cálculo solicitado usando somente esses dados.

CONTEÚDO DA BASE:
{contexto}

PERGUNTA:
{pergunta}
"""
    return client.models.generate_content(model=MODEL, contents=prompt)


def gerar_resposta_web(client, pergunta: str):
    prompt = f"""
{SYSTEM_PROMPT}
MODO: PESQUISA NA INTERNET.
A base local não apresentou conteúdo suficiente.
Pesquise na web e priorize fontes oficiais e primárias.
Informe que a resposta veio de pesquisa externa.

PERGUNTA:
{pergunta}
"""
    ferramenta = types.Tool(google_search=types.GoogleSearch())
    return client.models.generate_content(model=MODEL, contents=prompt, config=types.GenerateContentConfig(tools=[ferramenta]))


def erro_api(ex: Exception) -> bool:
    texto = str(ex).upper()
    return any(x in texto for x in ("429", "404", "RESOURCE_EXHAUSTED", "QUOTA", "NOT_FOUND"))


def mostrar_fontes_web(resposta):
    try:
        metadata = resposta.candidates[0].grounding_metadata
        fontes = getattr(metadata, "grounding_chunks", None) or []
        urls = []
        for fonte in fontes:
            web = getattr(fonte, "web", None)
            if web and getattr(web, "uri", None) and web.uri not in [u[1] for u in urls]:
                urls.append((getattr(web, "title", None) or web.uri, web.uri))
        if urls:
            st.markdown("### Fontes consultadas")
            for titulo, url in urls[:8]:
                st.markdown(f"- [{titulo}]({url})")
    except Exception:
        pass


def mostrar_resultados(resultados: list[dict]):
    with st.expander("Documentos encontrados na base"):
        for item in resultados:
            st.write(f"**{item['arquivo']}** — relevância {item['score']}")


def mostrar_fallback_local(pergunta: str, resultados: list[dict]):
    resposta = responder_local_sem_gemini(pergunta, resultados)
    if resposta:
        st.markdown("### ROMUS.IA")
        st.write(resposta)
        mostrar_resultados(resultados)
    else:
        st.info("A base encontrou documentos relacionados, mas não foi possível extrair uma resposta objetiva sem o mecanismo de síntese. Nenhuma informação externa foi inventada.")
        mostrar_resultados(resultados)


st.title("ROMUS.IA")
st.subheader("Inteligência artificial técnica e objetiva.")
pergunta = st.text_area("Digite sua pergunta:", placeholder="Pergunte qualquer coisa...", height=120)

col1, col2 = st.columns(2)
with col1:
    pesquisar_web_se_necessario = st.checkbox("Pesquisar na web se a base não responder", value=True)
with col2:
    if st.button("Recarregar base"):
        st.cache_data.clear()
        st.rerun()

if st.button("Perguntar", type="primary"):
    if not pergunta.strip():
        st.warning("Digite uma pergunta.")
    else:
        resultados = buscar_base(pergunta)

        # PRIMEIRO: base local. Esta etapa não depende do Gemini.
        if resultados:
            st.caption("Fonte: base local do ROMUS.IA")

            # Transcrição literal: extração direta do PDF, sem Gemini.
            if pedido_literal(pergunta):
                direta = responder_local_sem_gemini(pergunta, resultados)
                if direta:
                    st.markdown("### ROMUS.IA")
                    st.write(direta)
                    mostrar_resultados(resultados)
                    st.stop()

            # Tenta síntese pelo Gemini apenas depois de localizar a base.
            try:
                client = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])
                resposta = gerar_resposta_local(client, pergunta, resultados)
                texto = resposta.text or ""
                if LIMITE_INSUFICIENTE not in texto:
                    st.markdown("### ROMUS.IA")
                    st.write(texto)
                    mostrar_resultados(resultados)
                    st.stop()
            except Exception as ex:
                if not erro_api(ex):
                    st.error(f"Erro no mecanismo de síntese: {ex}")
                    st.stop()

            # Gemini indisponível: a base continua funcionando de forma extrativa.
            st.warning("Gemini indisponível ou sem cota. O ROMUS continua consultando a base local sem inventar conteúdo.")
            mostrar_fallback_local(pergunta, resultados)
            st.stop()

        # SEGUNDO: só chega aqui se a base não encontrou conteúdo suficiente.
        if pesquisar_web_se_necessario:
            st.caption("Fonte: pesquisa na internet — base local sem conteúdo suficiente")
            try:
                client = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])
                resposta = gerar_resposta_web(client, pergunta)
                st.markdown("### ROMUS.IA")
                st.write(resposta.text)
                mostrar_fontes_web(resposta)
            except Exception as ex:
                if erro_api(ex):
                    st.error("A base local não encontrou conteúdo suficiente e a API Gemini está indisponível ou sem cota para realizar a pesquisa web.")
                else:
                    st.error(f"Erro na pesquisa web: {ex}")
        else:
            st.info("A base local não contém informação suficiente e a pesquisa na web está desativada.")
