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

# Gemini só é usado quando a base local exige síntese.
MODEL = "gemini-3.5-flash-lite"

MAX_DOCUMENTOS = 4
MAX_BLOCOS = 6
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


def assinatura_base() -> tuple:
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


@st.cache_data(show_spinner=False)
def listar_arquivos(signature: tuple) -> list[dict]:
    del signature
    documentos = []
    for caminho in sorted(KNOWLEDGE_DIR.rglob("*")):
        if not caminho.is_file() or caminho.suffix.lower() not in {".txt", ".md", ".pdf"}:
            continue
        try:
            stat = caminho.stat()
            documentos.append({
                "arquivo": caminho.relative_to(KNOWLEDGE_DIR).as_posix(),
                "nome": caminho.name,
                "nome_norm": texto_normalizado(caminho.name),
                "tamanho": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
            })
        except OSError:
            continue
    return documentos


@st.cache_data(show_spinner=False)
def ler_documento(arquivo: str, tamanho: int, mtime_ns: int) -> dict | None:
    del tamanho, mtime_ns
    caminho = KNOWLEDGE_DIR / arquivo
    try:
        if caminho.suffix.lower() in {".txt", ".md"}:
            texto = caminho.read_text(encoding="utf-8", errors="ignore")
        elif caminho.suffix.lower() == ".pdf":
            reader = PdfReader(str(caminho))
            texto = "\n".join((pagina.extract_text() or "") for pagina in reader.pages)
        else:
            return None
    except Exception:
        return None
    texto = texto.strip()
    if not texto:
        return None
    digest = hashlib.sha256(texto.encode("utf-8", errors="ignore")).hexdigest()
    return {"arquivo": arquivo, "nome": caminho.name, "texto": texto, "texto_norm": texto_normalizado(texto), "nome_norm": texto_normalizado(caminho.name), "hash": digest}


def extrair_referencias(pergunta: str) -> dict:
    q = pergunta.lower()
    referencias = {"it": [], "item": [], "decreto": []}
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
    return referencias


def intencao_documental(pergunta: str) -> str:
    q = texto_normalizado(pergunta)
    if "decreto" in q:
        return "decreto"
    if re.search(r"\bit\b", q):
        return "it"
    return "geral"


def pontuar_nome(pergunta: str, documento: dict) -> int:
    termos = set(normalizar(pergunta))
    nome = documento["nome_norm"]
    score = 0
    refs = extrair_referencias(pergunta)
    intencao = intencao_documental(pergunta)
    for numero, ano in refs["it"]:
        padrao = rf"\bit\s*0?{numero}\s*[-/]\s*{ano}\b"
        if re.search(padrao, nome, flags=re.I):
            score += 1000
        elif ano == 2025 and re.search(rf"\bit\s*0?{numero}\s*[-/]\s*25\b", nome):
            score += 900
    for decreto in refs["decreto"]:
        limpo = re.sub(r"[./-]", "", decreto)
        nome_limpo = re.sub(r"[./-]", "", nome)
        if limpo and limpo in nome_limpo:
            score += 1000
    if intencao == "decreto" and "decreto" in nome:
        score += 500
    if intencao == "it" and re.search(r"\bit\b", nome):
        score += 100
    score += 20 * len(termos.intersection(set(normalizar(documento["nome"]))))
    return score


def selecionar_arquivos(pergunta: str, catalogo: list[dict]) -> list[dict]:
    avaliados = [{**d, "score_nome": pontuar_nome(pergunta, d)} for d in catalogo]
    avaliados.sort(key=lambda d: d["score_nome"], reverse=True)
    refs = extrair_referencias(pergunta)
    intencao = intencao_documental(pergunta)
    if refs["it"]:
        fortes = [d for d in avaliados if d["score_nome"] >= 900]
        if fortes:
            return fortes[:MAX_DOCUMENTOS]
    if refs["decreto"] or intencao == "decreto":
        fortes = [d for d in avaliados if "decreto" in d["nome_norm"]]
        if fortes:
            if refs["decreto"]:
                exatos = [d for d in fortes if d["score_nome"] >= 900]
                if exatos:
                    return exatos[:MAX_DOCUMENTOS]
            return fortes[:MAX_DOCUMENTOS]
    com_score = [d for d in avaliados if d["score_nome"] > 0]
    if com_score:
        return com_score[:MAX_DOCUMENTOS]
    return avaliados[:2]


def carregar_selecionados(candidatos: list[dict]) -> list[dict]:
    documentos = []
    hashes = set()
    for candidato in candidatos:
        doc = ler_documento(candidato["arquivo"], candidato["tamanho"], candidato["mtime_ns"])
        if not doc or doc["hash"] in hashes:
            continue
        hashes.add(doc["hash"])
        documentos.append(doc)
    return documentos


def dividir_em_blocos(texto: str, tamanho: int = 2600, sobreposicao: int = 180) -> list[str]:
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


def score_bloco(pergunta: str, bloco: dict, doc_score: int) -> int:
    termos = set(normalizar(pergunta))
    bloco_termos = set(normalizar(bloco["texto"]))
    score = doc_score + 3 * len(termos.intersection(bloco_termos))
    q = texto_normalizado(pergunta)
    nt = texto_normalizado(bloco["texto"])
    if q and len(q) >= 12 and q in nt:
        score += 250
    palavras = normalizar(pergunta)
    for tamanho in (6, 5, 4, 3):
        for i in range(max(0, len(palavras) - tamanho + 1)):
            trecho = " ".join(palavras[i:i + tamanho])
            if trecho and trecho in nt:
                score += tamanho * 12
    return score


def buscar_base(pergunta: str, documentos: list[dict]) -> list[dict]:
    resultados = []
    for doc in documentos:
        doc_score = pontuar_nome(pergunta, {"nome": doc["nome"], "nome_norm": doc["nome_norm"]})
        for numero, bloco_texto in enumerate(dividir_em_blocos(doc["texto"])):
            bloco = {"arquivo": doc["arquivo"], "nome": doc["nome"], "texto": bloco_texto, "numero_bloco": numero, "hash": doc["hash"]}
            score = score_bloco(pergunta, bloco, doc_score)
            if score >= MIN_SCORE:
                resultados.append({**bloco, "score": score})
    resultados.sort(key=lambda x: x["score"], reverse=True)
    vistos = set()
    unicos = []
    for item in resultados:
        chave = hashlib.sha1(re.sub(r"\s+", " ", item["texto"]).strip().lower().encode("utf-8", errors="ignore")).hexdigest()
        if chave in vistos:
            continue
        vistos.add(chave)
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
    return bool(termos.intersection({"transcreva", "transcrever", "literalmente", "literal", "exatamente", "redacao"}))


def pergunta_pede_calculo(pergunta: str) -> bool:
    termos = set(normalizar(pergunta))
    return bool(termos.intersection({"calcule", "calcular", "calculo", "dimensione", "dimensionar"}))


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
    if not resultados:
        return None
    termos = set(normalizar(pergunta))
    intencao = intencao_documental(pergunta)
    refs = extrair_referencias(pergunta)

    # Perguntas que pedem apenas o numero do decreto devem ser resolvidas
    # diretamente no texto do decreto, sem escolher uma linha de artigo.
    if intencao == "decreto":
        padroes = [
            r"\bDecreto\s+(?:Estadual\s+)?(?:n[ºo°]?\s*)?(\d{1,3}[.]\d{3}(?:[./-]\d{2,4})?)",
            r"\bDecreto\s+(?:Estadual\s+)?(?:n[ºo°]?\s*)?(\d{2,6}[./-]\d{2,4})",
        ]
        for item in resultados:
            for padrao in padroes:
                m = re.search(padrao, item["texto"], flags=re.I)
                if m:
                    numero = m.group(1).replace("_", "/")
                    if re.search(r"[./-]20\d{2}$", numero):
                        return numero.replace(".", ".")
                    if re.search(r"[./-]\d{2}$", numero):
                        return numero[:-2] + "20" + numero[-2:]
                    return numero
            m = re.search(r"\b69[.]118(?:[/.-]24|[/.-]2024)?\b", item["texto"], flags=re.I)
            if m:
                return "69.118/2024"

    if intencao == "decreto":
        prioritarias = {"decreto", "regulamento", "institui", "seguranca", "incendio"}
    elif refs["it"]:
        prioritarias = {"saidas", "emergencia", "largura", "unidade", "passagem", "porta", "minima", "minimo"}
    else:
        prioritarias = termos

    candidatos = []
    vistos = set()
    for item in resultados:
        for linha in re.split(r"\n+", item["texto"]):
            linha = re.sub(r"\s+", " ", linha).strip(" -\t")
            if len(linha) < 15:
                continue
            chave = linha.lower()
            if chave in vistos:
                continue
            vistos.add(chave)
            lt = set(normalizar(linha))
            coincidencias = len(termos.intersection(lt))
            prioridades = len(prioritarias.intersection(lt))
            score = item["score"] + coincidencias * 4 + prioridades * 7
            if re.search(r"\d", linha):
                score += 5
            if intencao == "decreto" and "decreto" in lt:
                score += 80
            if refs["it"] and ("largura" in lt or "saida" in lt or "emergencia" in lt):
                score += 35
            if prioridades >= 1 or coincidencias >= 2:
                candidatos.append((score, linha))
    candidatos.sort(key=lambda x: x[0], reverse=True)
    if not candidatos:
        return None
    melhor_score, melhor = candidatos[0]
    return melhor if melhor_score >= 30 else None


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
    return genai.Client(api_key=chave) if chave else None


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
    return client.models.generate_content(model=MODEL, contents=prompt, config=types.GenerateContentConfig(max_output_tokens=700))


def gerar_resposta_web(client, pergunta: str):
    prompt = f"""
{SYSTEM_PROMPT}

A base documental local foi consultada e não contém informação suficiente para responder com segurança.
Agora faça pesquisa na internet usando somente fontes confiáveis, dando prioridade a legislação, órgãos públicos e fontes oficiais.
Deixe explícito que a resposta veio da pesquisa externa.

PERGUNTA:
{pergunta}
"""
    config = types.GenerateContentConfig(max_output_tokens=800, tools=[types.Tool(google_search=types.GoogleSearch())])
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
    assinatura = assinatura_base()
    with st.spinner("Localizando o documento..."):
        catalogo = listar_arquivos(assinatura)
        candidatos = selecionar_arquivos(pergunta.strip(), catalogo)
    with st.spinner("Lendo somente o documento relevante..."):
        documentos = carregar_selecionados(candidatos)
        resultados = buscar_base(pergunta.strip(), documentos)
    st.caption("Fonte: base local do ROMUS.IA")

    resposta_direta = resposta_local_deterministica(pergunta.strip(), resultados)
    if resposta_direta:
        st.markdown("### ROMUS.IA")
        st.write(resposta_direta)
        mostrar_fontes(resultados)
        st.stop()

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
        if texto.startswith("ERRO_GEMINI:"):
            st.markdown("### ROMUS.IA")
            st.warning("A base local encontrou conteúdo relevante, mas a síntese pelo Gemini não está disponível no momento.")
            mostrar_fontes(resultados)
            st.stop()

    if not resultados and web_habilitada and client is not None:
        with st.spinner("A base não contém a resposta. Pesquisando na internet..."):
            try:
                resposta = gerar_resposta_web(client, pergunta.strip())
                texto = (resposta.text or "").strip()
            except Exception as exc:
                texto = f"ERRO_WEB: {exc}"
        if texto and not texto.startswith("ERRO_WEB:"):
            st.caption("Fonte: pesquisa na internet (base local insuficiente)")
            st.markdown("### ROMUS.IA")
            st.write(texto)
            st.stop()

    st.markdown("### ROMUS.IA")
    st.warning("A base local não contém informação suficiente para responder com segurança.")
    mostrar_fontes(resultados)
