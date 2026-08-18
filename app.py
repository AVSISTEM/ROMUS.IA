import re
from pathlib import Path

import streamlit as st
from google import genai
from google.genai import types
from pypdf import PdfReader

st.set_page_config(page_title="ROMUS.IA", page_icon="🔥", layout="centered")

BASE_DIR = Path(__file__).parent
KNOWLEDGE_DIR = BASE_DIR / "base_conhecimento"

# Gemini NÃO participa da busca local. Só é chamado quando a base encontrou
# material, mas a resposta realmente exige síntese.
MODEL = "gemini-3.5-flash-lite"
MAX_DOCUMENTOS = 5
MAX_BLOCOS = 8
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
    texto = texto.replace("º", " ").replace("°", " ")
    texto = re.sub(r"[^\w\s]", " ", texto, flags=re.UNICODE)
    return [p for p in texto.split() if len(p) > 1]


def texto_normalizado(texto: str) -> str:
    return " ".join(normalizar(texto))


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
    vistos = set()

    for caminho in sorted(KNOWLEDGE_DIR.rglob("*")):
        if not caminho.is_file() or caminho.suffix.lower() not in {".txt", ".md", ".pdf"}:
            continue
        try:
            texto = ler_arquivo(caminho).strip()
        except Exception:
            continue
        if not texto:
            continue

        relativo = caminho.relative_to(KNOWLEDGE_DIR).as_posix()
        chave = (relativo.lower(), len(texto))
        if chave in vistos:
            continue
        vistos.add(chave)
        documentos.append({"arquivo": relativo, "nome": caminho.name, "texto": texto})

    return documentos


def dividir_em_blocos(texto: str, tamanho: int = 2200, sobreposicao: int = 250) -> list[str]:
    texto = re.sub(r"\n{3,}", "\n\n", texto).strip()
    if len(texto) <= tamanho:
        return [texto]

    blocos = []
    inicio = 0
    while inicio < len(texto):
        fim = min(inicio + tamanho, len(texto))
        corte = texto.rfind("\n", inicio + tamanho // 2, fim)
        if corte > inicio:
            fim = corte
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
        for numero, bloco in enumerate(dividir_em_blocos(documento["texto"])):
            indice.append({
                "arquivo": documento["arquivo"],
                "texto": bloco,
                "termos": set(normalizar(bloco)),
                "numero_bloco": numero,
            })
    return indice


def remover_duplicados_itens(itens: list[dict]) -> list[dict]:
    resultado = []
    vistos = set()
    for item in itens:
        chave = (item["arquivo"].lower(), re.sub(r"\s+", " ", item["texto"]).strip().lower())
        if chave in vistos:
            continue
        vistos.add(chave)
        resultado.append(item)
    return resultado


def extrair_referencias(pergunta: str) -> dict:
    q = pergunta.lower()
    referencias = {"it": [], "item": [], "decreto": [], "ano": []}

    for numero in re.findall(r"\bit\s*(?:n[ºo°]?\s*)?(\d{1,2})\s*[-/]\s*(20\d{2})\b", q):
        referencias["it"].append((int(numero[0]), int(numero[1])))

    for numero in re.findall(r"\bit\s*(?:n[ºo°]?\s*)?(\d{1,2})\s*[-/]\s*25\b", q):
        par = (int(numero), 2025)
        if par not in referencias["it"]:
            referencias["it"].append(par)

    referencias["item"] = re.findall(r"\b(?:item|subitem|alínea|alinea)\s*(\d+(?:\.\d+){1,4})\b", q)
    referencias["decreto"] = re.findall(r"\bdecreto\s*(?:estadual\s*)?(?:n[ºo°]?\s*)?(\d{2,6}(?:[./-]\d{2,4})?)", q)
    referencias["ano"] = re.findall(r"\b20\d{2}\b", q)
    return referencias


def score_documento(pergunta: str, documento: dict) -> int:
    q = texto_normalizado(pergunta)
    termos = set(normalizar(pergunta))
    nome = texto_normalizado(documento["nome"])
    texto_inicio = texto_normalizado(documento["texto"][:12000])
    score = 0
    refs = extrair_referencias(pergunta)

    for numero, ano in refs["it"]:
        padrao = rf"\bit\s*0?{numero}\s*[-/]\s*{ano}\b"
        if re.search(padrao, nome, flags=re.I):
            score += 100
        elif f"it {numero} 25" in nome or f"it 0{numero} 25" in nome:
            score += 90

    for decreto in refs["decreto"]:
        if decreto.replace(".", "").replace("/", "").replace("-", "") in re.sub(r"[./-]", "", nome):
            score += 100

    nome_termos = set(normalizar(documento["nome"]))
    score += 6 * len(termos.intersection(nome_termos))
    score += len(termos.intersection(set(normalizar(documento["texto"][:12000]))))

    if q and len(q) >= 12 and q in texto_inicio:
        score += 80
    return score


def selecionar_documentos(pergunta: str) -> list[dict]:
    documentos = carregar_documentos()
    avaliados = []
    for documento in documentos:
        score = score_documento(pergunta, documento)
        if score >= MIN_SCORE:
            avaliados.append({**documento, "score_documento": score})

    avaliados.sort(key=lambda x: (x["score_documento"], len(x["texto"])), reverse=True)
    refs = extrair_referencias(pergunta)

    if refs["it"]:
        fortes = [
            d for d in avaliados
            if any(re.search(rf"\bit\s*0?{numero}\s*[-/]\s*{ano}\b", d["nome"], flags=re.I) for numero, ano in refs["it"])
        ]
        if fortes:
            return fortes[:MAX_DOCUMENTOS]

    return avaliados[:MAX_DOCUMENTOS]


def score_bloco(pergunta: str, bloco: dict, documento_score: int = 0) -> int:
    termos = set(normalizar(pergunta))
    texto = bloco["texto"]
    bloco_termos = set(normalizar(texto))
    score = documento_score // 10
    score += 3 * len(termos.intersection(bloco_termos))
    q = texto_normalizado(pergunta)
    nt = texto_normalizado(texto)

    if q and len(q) >= 12 and q in nt:
        score += 100

    palavras = normalizar(pergunta)
    for tamanho in (5, 4, 3):
        for i in range(max(0, len(palavras) - tamanho + 1)):
            trecho = " ".join(palavras[i:i + tamanho])
            if trecho and trecho in nt:
                score += tamanho * 8
    return score


def buscar_base(pergunta: str) -> list[dict]:
    documentos = selecionar_documentos(pergunta)
    if not documentos:
        return []

    indice = construir_indice()
    nomes_prioritarios = {d["arquivo"]: d for d in documentos}
    resultados = []

    for item in indice:
        if item["arquivo"] not in nomes_prioritarios:
            continue
        doc = nomes_prioritarios[item["arquivo"]]
        score = score_bloco(pergunta, item, doc["score_documento"])
        if score >= MIN_SCORE:
            resultados.append({**item, "score": score})

    resultados.sort(key=lambda x: x["score"], reverse=True)
    resultados = remover_duplicados_itens(resultados)

    # Impede que muitos blocos do mesmo documento dominem a resposta.
    finais = []
    por_documento = {}
    for item in resultados:
        quantidade = por_documento.get(item["arquivo"], 0)
        if quantidade >= MAX_BLOCOS:
            continue
        por_documento[item["arquivo"]] = quantidade + 1
        finais.append(item)
    return finais[:MAX_BLOCOS]


def extrair_numero_item(pergunta: str) -> str | None:
    m = re.search(r"\b(?:item|subitem)\s*(\d+(?:\.\d+){1,4})\b", pergunta.lower())
    return m.group(1) if m else None


def extrair_item_literal(texto: str, numero_item: str) -> str | None:
    numero = re.escape(numero_item)
    padroes = [
        rf"(?ms)^\s*{numero}\s+.*?(?=^\s*\d+(?:\.\d+){{1,4}}\s+|\Z)",
        rf"(?ms)(?<!\d){numero}\s+.*?(?=\n\s*\d+(?:\.\d+){{1,4}}\s+|\Z)",
    ]
    for padrao in padroes:
        encontrado = re.search(padrao, texto)
        if encontrado:
            return encontrado.group(0).strip()
    return None


def localizar_documento_para_item(pergunta: str) -> dict | None:
    candidatos = selecionar_documentos(pergunta)
    numero_item = extrair_numero_item(pergunta)
    if numero_item:
        for doc in candidatos:
            if extrair_item_literal(doc["texto"], numero_item):
                return doc
    return candidatos[0] if candidatos else None


def pedido_literal(pergunta: str) -> bool:
    termos = set(normalizar(pergunta))
    return bool(termos.intersection({"transcreva", "transcrever", "literalmente", "literal", "exatamente", "exato", "texto", "redação"}))


def pergunta_pede_calculo(pergunta: str) -> bool:
    termos = set(normalizar(pergunta))
    return bool(termos.intersection({"calcule", "calcular", "cálculo", "calculo", "dimensione", "dimensionar"}))


def extrair_resposta_explicita(pergunta: str, resultados: list[dict]) -> str | None:
    """Responde diretamente quando a informação está explícita na base, sem Gemini."""
    if not resultados:
        return None

    termos = set(normalizar(pergunta))
    candidatos = []
    vistos = set()

    for item in resultados:
        for linha in item["texto"].splitlines():
            linha = re.sub(r"\s+", " ", linha).strip()
            if len(linha) < 12:
                continue
            chave = linha.lower()
            if chave in vistos:
                continue
            vistos.add(chave)

            linha_termos = set(normalizar(linha))
            coincidencias = len(termos.intersection(linha_termos))
            if coincidencias == 0:
                continue

            bonus = 0
            if re.search(r"\d", linha):
                bonus += 4
            if re.search(r"\b(item|it|decreto|largura|mínim|minim|altura|número|unidades|pessoas)\b", linha.lower()):
                bonus += 3

            score = coincidencias * 3 + bonus + item["score"]
            candidatos.append((score, linha))

    candidatos.sort(key=lambda x: x[0], reverse=True)
    if not candidatos:
        return None

    melhor_score, melhor = candidatos[0]
    if melhor_score >= 14:
        return melhor

    fortes = [linha for score, linha in candidatos[:3] if score >= 10]
    if fortes:
        return "\n".join(dict.fromkeys(fortes))
    return None


def responder_local_sem_gemini(pergunta: str, resultados: list[dict]) -> str | None:
    # Pedido literal: extração direta do PDF.
    if pedido_literal(pergunta):
        numero = extrair_numero_item(pergunta)
        if numero:
            doc = localizar_documento_para_item(pergunta)
            if doc:
                trecho = extrair_item_literal(doc["texto"], numero)
                if trecho:
                    return trecho

    # Informação explícita: não chama Gemini.
    explicita = extrair_resposta_explicita(pergunta, resultados)
    if explicita and not pergunta_pede_calculo(pergunta):
        return explicita
    return None


def gerar_resposta_local(client, pergunta: str, resultados: list[dict]):
    contexto = "\n\n".join(f"[FONTE LOCAL: {i['arquivo']}]\n{i['texto']}" for i in resultados)
    prompt = f"""
{SYSTEM_PROMPT}

MODO: SÍNTESE DA BASE LOCAL.
A busca documental já foi feita pelo ROMUS.IA.
Use exclusivamente o conteúdo abaixo.
Não pesquise na internet.
Não acrescente conhecimento externo.
Se o conteúdo não permitir responder com segurança, responda somente {LIMITE_INSUFICIENTE}.
Quando houver cálculo solicitado, use somente os critérios, fórmulas e números presentes na base.
Quando houver mais de uma regra, indique claramente de qual trecho vem cada uma.

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
A base local foi consultada e não apresentou conteúdo suficiente.
Pesquise na web e priorize fontes oficiais e primárias.
Informe claramente que a resposta veio de pesquisa externa.

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
        vistos = set()
        for item in resultados:
            if item["arquivo"] in vistos:
                continue
            vistos.add(item["arquivo"])
            st.write(f"**{item['arquivo']}** — relevância {item['score']}")


def mostrar_fallback_local(pergunta: str, resultados: list[dict]):
    resposta = responder_local_sem_gemini(pergunta, resultados)
    if resposta:
        st.markdown("### ROMUS.IA")
        st.write(resposta)
        mostrar_resultados(resultados)
    else:
        st.info("A base encontrou conteúdo relacionado, mas a resposta exige síntese ou cálculo. O Gemini está indisponível no momento; nenhuma informação externa foi inventada.")
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
        # 1. BASE LOCAL — não depende do Gemini.
        resultados = buscar_base(pergunta)

        if resultados:
            st.caption("Fonte: base local do ROMUS.IA")

            # 2. RESPOSTA DIRETA — literal ou explícita, zero Gemini.
            direta = responder_local_sem_gemini(pergunta, resultados)
            if direta:
                st.markdown("### ROMUS.IA")
                st.write(direta)
                mostrar_resultados(resultados)
                st.stop()

            # 3. SÍNTESE — Gemini só quando realmente necessária.
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

            # Gemini indisponível: a base continua sendo utilizada.
            st.warning("Gemini indisponível ou sem cota. O ROMUS continua consultando a base local.")
            mostrar_fallback_local(pergunta, resultados)
            st.stop()

        # 4. WEB — só chega aqui quando a base realmente não encontrou conteúdo.
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
