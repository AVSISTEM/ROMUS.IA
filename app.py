import hashlib
import os
import re
from pathlib import Path

import streamlit as st
from google import genai
from google.genai import types
from pypdf import PdfReader

st.set_page_config(page_title="ROMUS.IA", page_icon="🔥", layout="centered")

BASE_DIR = Path(__file__).parent
KNOWLEDGE_DIR = BASE_DIR / "base_conhecimento"

# Modelo atual. Gemini só entra depois da busca local e somente quando
# a resposta não puder ser extraída diretamente.
MODEL = "gemini-3.6-flash"
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


def arquivo_signature() -> tuple:
    """Assinatura barata dos arquivos. Permite recarregar a base quando ela muda."""
    KNOWLEDGE_DIR.mkdir(parents=True, exist_ok=True)
    itens = []
    for caminho in sorted(KNOWLEDGE_DIR.rglob("*")):
        if caminho.is_file() and caminho.suffix.lower() in {".txt", ".md", ".pdf"}:
            try:
                stat = caminho.stat()
                itens.append((caminho.relative_to(KNOWLEDGE_DIR).as_posix(), stat.st_size, stat.st_mtime_ns))
            except OSError:
                pass
    return tuple(itens)


def ler_arquivo(caminho: Path) -> str:
    if caminho.suffix.lower() in {".txt", ".md"}:
        return caminho.read_text(encoding="utf-8", errors="ignore")
    if caminho.suffix.lower() == ".pdf":
        reader = PdfReader(str(caminho))
        paginas = []
        for pagina in reader.pages:
            paginas.append(pagina.extract_text() or "")
        return "\n".join(paginas)
    return ""


@st.cache_data(show_spinner=False)
def carregar_documentos(signature: tuple) -> list[dict]:
    """Lê cada PDF uma única vez por versão da base."""
    del signature
    documentos = []
    hashes = set()

    for caminho in sorted(KNOWLEDGE_DIR.rglob("*")):
        if not caminho.is_file() or caminho.suffix.lower() not in {".txt", ".md", ".pdf"}:
            continue
        try:
            texto = ler_arquivo(caminho).strip()
        except Exception:
            continue
        if not texto:
            continue

        # Duplicidade real: mesmo conteúdo, mesmo que o nome do PDF seja diferente.
        digest = hashlib.sha256(texto.encode("utf-8", errors="ignore")).hexdigest()
        if digest in hashes:
            continue
        hashes.add(digest)

        relativo = caminho.relative_to(KNOWLEDGE_DIR).as_posix()
        documentos.append({
            "arquivo": relativo,
            "nome": caminho.name,
            "texto": texto,
            "texto_norm": texto_normalizado(texto[:30000]),
            "nome_norm": texto_normalizado(caminho.name),
            "hash": digest,
        })

    return documentos


def dividir_em_blocos(texto: str, tamanho: int = 2400, sobreposicao: int = 180) -> list[str]:
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
def construir_indice(signature: tuple) -> list[dict]:
    documentos = carregar_documentos(signature)
    indice = []
    for documento in documentos:
        for numero, bloco in enumerate(dividir_em_blocos(documento["texto"])):
            indice.append({
                "arquivo": documento["arquivo"],
                "nome": documento["nome"],
                "texto": bloco,
                "texto_norm": texto_normalizado(bloco),
                "termos": set(normalizar(bloco)),
                "numero_bloco": numero,
                "hash": documento["hash"],
            })
    return indice


def extrair_referencias(pergunta: str) -> dict:
    q = pergunta.lower()
    referencias = {"it": [], "item": [], "decreto": [], "ano": []}

    for numero, ano in re.findall(r"\bit\s*(?:n[ºo°]?\s*)?(\d{1,2})\s*[-/]\s*(20\d{2})\b", q):
        par = (int(numero), int(ano))
        if par not in referencias["it"]:
            referencias["it"].append(par)

    for numero in re.findall(r"\bit\s*(?:n[ºo°]?\s*)?(\d{1,2})\s*[-/]\s*25\b", q):
        par = (int(numero), 2025)
        if par not in referencias["it"]:
            referencias["it"].append(par)

    referencias["item"] = re.findall(r"\b(?:item|subitem|alínea|alinea)\s*(\d+(?:\.\d+){1,4})\b", q)
    referencias["decreto"] = re.findall(r"\bdecreto\s*(?:estadual\s*)?(?:n[ºo°]?\s*)?(\d{2,6}(?:[./-]\d{2,4})?)", q)
    referencias["ano"] = re.findall(r"\b20\d{2}\b", q)
    return referencias


def intencao_documental(pergunta: str) -> str:
    q = texto_normalizado(pergunta)
    if "decreto" in q:
        return "decreto"
    if re.search(r"\bit\b", q):
        return "it"
    return "geral"


def score_documento(pergunta: str, documento: dict) -> int:
    termos = set(normalizar(pergunta))
    nome = documento["nome_norm"]
    texto_inicio = documento["texto_norm"][:14000]
    refs = extrair_referencias(pergunta)
    score = 0
    intencao = intencao_documental(pergunta)

    # Referência normativa explícita tem prioridade absoluta.
    for numero, ano in refs["it"]:
        if re.search(rf"\bit\s*0?{numero}\s*[-/]\s*{ano}\b", nome, flags=re.I):
            score += 500
        elif f"it {numero} 25" in nome or f"it 0{numero} 25" in nome:
            score += 450

    for decreto in refs["decreto"]:
        limpo = re.sub(r"[./-]", "", decreto)
        if limpo and limpo in re.sub(r"[./-]", "", nome):
            score += 500

    # Pergunta sobre decreto sem número: arquivo do decreto vence ITs de terminologia.
    if intencao == "decreto" and "decreto" in nome:
        score += 350

    if intencao == "it" and "it " in nome:
        score += 40

    nome_termos = set(normalizar(documento["nome"]))
    score += 8 * len(termos.intersection(nome_termos))

    # Termos relevantes no começo do documento ajudam sem deixar um documento
    # genérico ganhar apenas porque é grande.
    score += 2 * len(termos.intersection(set(normalizar(documento["texto"][:12000]))))

    return score


def selecionar_documentos(pergunta: str, documentos: list[dict]) -> list[dict]:
    avaliados = []
    for documento in documentos:
        score = score_documento(pergunta, documento)
        if score >= MIN_SCORE:
            avaliados.append({**documento, "score_documento": score})

    avaliados.sort(key=lambda x: (x["score_documento"], -len(x["texto"])), reverse=True)
    refs = extrair_referencias(pergunta)

    if refs["it"]:
        fortes = [
            d for d in avaliados
            if any(re.search(rf"\bit\s*0?{numero}\s*[-/]\s*{ano}\b", d["nome"], flags=re.I) for numero, ano in refs["it"])
        ]
        if fortes:
            return fortes[:MAX_DOCUMENTOS]

    if intencao_documental(pergunta) == "decreto":
        decretos = [d for d in avaliados if "decreto" in d["nome_norm"]]
        if decretos:
            return decretos[:MAX_DOCUMENTOS]

    return avaliados[:MAX_DOCUMENTOS]


def score_bloco(pergunta: str, bloco: dict, documento_score: int = 0) -> int:
    termos = set(normalizar(pergunta))
    bloco_termos = bloco["termos"]
    score = documento_score // 10
    score += 3 * len(termos.intersection(bloco_termos))
    nt = bloco["texto_norm"]
    q = texto_normalizado(pergunta)

    if q and len(q) >= 12 and q in nt:
        score += 180

    palavras = normalizar(pergunta)
    for tamanho in (5, 4, 3):
        for i in range(max(0, len(palavras) - tamanho + 1)):
            trecho = " ".join(palavras[i:i + tamanho])
            if trecho and trecho in nt:
                score += tamanho * 10
    return score


def buscar_base(pergunta: str, indice: list[dict], documentos: list[dict]) -> list[dict]:
    selecionados = selecionar_documentos(pergunta, documentos)
    if not selecionados:
        return []

    prioritarios = {d["arquivo"]: d for d in selecionados}
    resultados = []
    for bloco in indice:
        if bloco["arquivo"] not in prioritarios:
            continue
        doc = prioritarios[bloco["arquivo"]]
        score = score_bloco(pergunta, bloco, doc["score_documento"])
        if score >= MIN_SCORE:
            resultados.append({**bloco, "score": score})

    resultados.sort(key=lambda x: x["score"], reverse=True)

    # Remove blocos idênticos e, principalmente, não apresenta o mesmo documento
    # várias vezes na interface.
    vistos_conteudo = set()
    unicos = []
    for item in resultados:
        chave = hashlib.sha1(re.sub(r"\s+", " ", item["texto"]).strip().lower().encode("utf-8", errors="ignore")).hexdigest()
        if chave in vistos_conteudo:
            continue
        vistos_conteudo.add(chave)
        unicos.append(item)

    finais = []
    por_documento = {}
    for item in unicos:
        quantidade = por_documento.get(item["arquivo"], 0)
        if quantidade >= 3:
            continue
        por_documento[item["arquivo"]] = quantidade + 1
        finais.append(item)
        if len(finais) >= MAX_BLOCOS:
            break
    return finais


def pedido_literal(pergunta: str) -> bool:
    termos = set(normalizar(pergunta))
    return bool(termos.intersection({"transcreva", "transcrever", "literalmente", "literal", "exatamente", "exato", "redação"}))


def pergunta_pede_calculo(pergunta: str) -> bool:
    termos = set(normalizar(pergunta))
    return bool(termos.intersection({"calcule", "calcular", "cálculo", "calculo", "dimensione", "dimensionar"}))


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


def melhor_linha_explicita(pergunta: str, resultados: list[dict]) -> str | None:
    """Extrai uma resposta objetiva da própria base, sem modelo generativo."""
    if not resultados:
        return None

    q = texto_normalizado(pergunta)
    termos = set(normalizar(pergunta))
    intencao = intencao_documental(pergunta)
    refs = extrair_referencias(pergunta)
    candidatos = []
    vistos = set()

    # Para perguntas sobre decreto, primeiro procurar linhas que realmente tratem
    # do decreto, e não qualquer linha de uma IT que mencione segurança contra incêndio.
    if intencao == "decreto":
        palavras_prioritarias = {"decreto", "regulamento", "institui", "seguranca", "incendio"}
    elif refs["it"]:
        palavras_prioritarias = {"saidas", "emergencia", "largura", "unidade", "passagem", "porta", "minima", "minimo"}
    else:
        palavras_prioritarias = termos

    for item in resultados:
        linhas = re.split(r"\n+", item["texto"])
        for linha in linhas:
            linha = re.sub(r"\s+", " ", linha).strip(" -\t")
            if len(linha) < 15:
                continue
            chave = linha.lower()
            if chave in vistos:
                continue
            vistos.add(chave)

            lt = set(normalizar(linha))
            coincidencias = len(termos.intersection(lt))
            prioridades = len(palavras_prioritarias.intersection(lt))
            score = item["score"] + coincidencias * 4 + prioridades * 7

            if re.search(r"\d", linha):
                score += 5
            if intencao == "decreto" and "decreto" in lt:
                score += 80
            if refs["it"] and ("largura" in lt or "saida" in lt or "emergencia" in lt):
                score += 35
            if q and q in texto_normalizado(linha):
                score += 100

            if prioridades >= 1 or coincidencias >= 2:
                candidatos.append((score, linha))

    candidatos.sort(key=lambda x: x[0], reverse=True)
    if not candidatos:
        return None

    # Evita retornar lixo de cabeçalho só porque contém muitas palavras comuns.
    melhor_score, melhor = candidatos[0]
    if melhor_score < 30:
        return None
    return melhor


def resposta_local_deterministica(pergunta: str, resultados: list[dict]) -> str | None:
    if not resultados:
        return None

    if pedido_literal(pergunta):
        numero = extrair_numero_item(pergunta)
        if numero:
            for item in resultados:
                trecho = extrair_item_literal(item["texto"], numero)
                if trecho:
                    return trecho

    if not pergunta_pede_calculo(pergunta):
        return melhor_linha_explicita(pergunta, resultados)

    return None


def obter_client():
    chave = st.secrets.get("GEMINI_API_KEY", os.getenv("GEMINI_API_KEY", ""))
    if not chave:
        return None
    return genai.Client(api_key=chave)


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
Quando houver cálculo solicitado, use somente critérios, fórmulas e números presentes na base.
Quando houver mais de uma regra, indique claramente de qual fonte vem cada uma.

CONTEÚDO DA BASE:
{contexto}

PERGUNTA:
{pergunta}
"""
    return client.models.generate_content(
        model=MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(temperature=0, max_output_tokens=700),
    )


def gerar_resposta_web(client, pergunta: str):
    prompt = f"""
{SYSTEM_PROMPT}

A base documental local foi consultada e não contém informação suficiente para responder com segurança.
Agora faça pesquisa na internet usando somente fontes confiáveis, dando prioridade a legislação, órgãos públicos e fontes oficiais.
Deixe explícito que a resposta veio da pesquisa externa.

PERGUNTA:
{pergunta}
"""
    config = types.GenerateContentConfig(
        temperature=0,
        max_output_tokens=800,
        tools=[types.Tool(google_search=types.GoogleSearch())],
    )
    return client.models.generate_content(model=MODEL, contents=prompt, config=config)


def mostrar_fontes(resultados: list[dict]):
    vistos = set()
    fontes = []
    for item in resultados:
        if item["arquivo"] in vistos:
            continue
        vistos.add(item["arquivo"])
        fontes.append(item["arquivo"])
    if fontes:
        with st.expander("Documentos encontrados na base"):
            for fonte in fontes:
                st.write(f"• {fonte}")


st.title("ROMUS.IA")
st.caption("Inteligência artificial técnica e objetiva.")

pergunta = st.text_area("Digite sua pergunta:", height=110)
web_habilitada = st.checkbox("Pesquisar na web se a base não responder", value=True)

col1, col2 = st.columns([1, 1])
with col1:
    perguntar = st.button("Perguntar", type="primary")
with col2:
    recarregar = st.button("Recarregar base")

if recarregar:
    st.cache_data.clear()
    st.rerun()

if perguntar and pergunta.strip():
    assinatura = arquivo_signature()
    with st.spinner("Consultando a base local..."):
        documentos = carregar_documentos(assinatura)
        indice = construir_indice(assinatura)
        resultados = buscar_base(pergunta.strip(), indice, documentos)

    st.caption("Fonte: base local do ROMUS.IA")

    # CAMADA 1: resposta direta, sem Gemini.
    resposta_direta = resposta_local_deterministica(pergunta.strip(), resultados)
    if resposta_direta:
        st.markdown("### ROMUS.IA")
        st.write(resposta_direta)
        mostrar_fontes(resultados)
        st.stop()

    # CAMADA 2: Gemini somente quando a pergunta exige síntese/cálculo.
    client = obter_client()
    if resultados and client is not None:
        with st.spinner("Sintetizando a resposta com base no documento..."):
            try:
                resposta = gerar_resposta_local(client, pergunta.strip(), resultados)
                texto = (resposta.text or "").strip()
            except Exception as exc:
                texto = f"ERRO_GEMINI: {exc}"

        if texto and texto != LIMITE_INSUFICIENTE and not texto.startswith("ERRO_GEMINI:"):
            st.markdown("### ROMUS.IA")
            st.write(texto)
            mostrar_fontes(resultados)
            st.stop()

    # CAMADA 3: internet somente quando a base não foi suficiente.
    if web_habilitada and client is not None:
        with st.spinner("A base não foi suficiente. Pesquisando na internet..."):
            try:
                resposta = gerar_resposta_web(client, pergunta.strip())
                texto = (resposta.text or "").strip()
            except Exception as exc:
                texto = f"ERRO_WEB: {exc}"
        if texto:
            st.caption("Fonte: pesquisa na internet (base local insuficiente)")
            st.markdown("### ROMUS.IA")
            st.write(texto)
            mostrar_fontes(resultados)
            st.stop()

    st.markdown("### ROMUS.IA")
    st.warning("A base local não contém informação suficiente para responder com segurança.")
    mostrar_fontes(resultados)
