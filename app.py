import hashlib
import os
import re
import unicodedata
from pathlib import Path

import streamlit as st
from google import genai
from google.genai import types
from pypdf import PdfReader

st.set_page_config(page_title="ROMUS.IA", page_icon="🔥", layout="centered")

BASE_DIR = Path(__file__).parent
KNOWLEDGE_DIR = BASE_DIR / "base_conhecimento"
MODEL = "gemini-3.5-flash"
ABSTAIN = "A base local não contém informação suficiente para responder com segurança."

PROMPT = """Você é o ROMUS.IA, assistente técnico de segurança contra incêndio.

Use SOMENTE as evidências fornecidas em <contexto> para responder.
Não use conhecimento externo e não invente informações.

REGRAS OBRIGATÓRIAS:
1. Responda diretamente à pergunta.
2. Em tabelas, localize primeiro a linha/código exato solicitado e depois a coluna correspondente.
3. Se a pergunta mencionar um grupo de ocupação, use somente esse grupo. Não troque F-11 por E-5, E-6 ou outro grupo.
4. Preserve exatamente números, unidades, itens, datas e referências encontrados nas evidências.
5. Faça cálculos somente quando todos os valores necessários estiverem nas evidências.
6. Se as evidências contiverem a resposta, responda. Não diga que falta informação.
7. Se as evidências realmente não contiverem a resposta, escreva exatamente: A base local não contém informação suficiente para responder com segurança.
8. Não mostre raciocínio, análise, rascunho ou comentários internos.
9. Responda sempre em português do Brasil.
10. Ao final, informe de forma curta o documento e a página utilizados.

<contexto>
{context}
</contexto>

<pergunta>
{question}
</pergunta>

RESPOSTA DIRETA:
"""

WEB_PROMPT = """Responda à pergunta usando fontes oficiais ou técnicas confiáveis encontradas na Pesquisa Google.
Não invente informações. Seja objetivo, em português do Brasil.
Quando houver norma, decreto ou instrução técnica, dê preferência ao documento oficial e informe as fontes."""


def normalizar(texto):
    texto = unicodedata.normalize("NFKD", str(texto).lower())
    texto = "".join(c for c in texto if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", texto)).strip()


def tokens(texto):
    return {x for x in normalizar(texto).split() if len(x) > 1}


def grupos(texto):
    encontrados = re.findall(r"\b([A-Z]{1,3})\s*[-–—]?\s*(\d{1,3})\b", str(texto).upper())
    return {f"{a}-{b}" for a, b in encontrados if a != "IT"}


def it_referenciada(question):
    q = normalizar(question)
    m = re.search(r"\bit\s*(?:n|no)?\s*0?(\d{1,2})\s*[-/ ]\s*(20\d{2}|\d{2})\b", q)
    if not m:
        return None
    ano = int(m.group(2))
    if ano < 100:
        ano += 2000
    return int(m.group(1)), ano


@st.cache_data(show_spinner=False)
def assinatura_base():
    KNOWLEDGE_DIR.mkdir(parents=True, exist_ok=True)
    dados = []
    for p in sorted(KNOWLEDGE_DIR.rglob("*")):
        if p.is_file() and p.suffix.lower() in {".pdf", ".txt", ".md"}:
            try:
                s = p.stat()
                dados.append((str(p.relative_to(KNOWLEDGE_DIR)), s.st_size, s.st_mtime_ns))
            except OSError:
                pass
    return tuple(dados)


@st.cache_data(show_spinner=False)
def catalogo(sig):
    del sig
    saida = []
    for p in sorted(KNOWLEDGE_DIR.rglob("*")):
        if p.is_file() and p.suffix.lower() in {".pdf", ".txt", ".md"}:
            s = p.stat()
            saida.append({
                "arquivo": str(p.relative_to(KNOWLEDGE_DIR)),
                "nome": p.name,
                "nome_norm": normalizar(p.name),
                "tamanho": s.st_size,
                "mtime": s.st_mtime_ns,
            })
    return saida


@st.cache_data(show_spinner=False)
def ler(arquivo, tamanho, mtime):
    del tamanho, mtime
    p = KNOWLEDGE_DIR / arquivo
    try:
        if p.suffix.lower() == ".pdf":
            paginas = []
            for numero, page in enumerate(PdfReader(str(p)).pages, start=1):
                texto = page.extract_text() or ""
                texto = texto.replace("\u00a0", " ")
                texto = re.sub(r"[ \t]+", " ", texto)
                texto = re.sub(r"\n{3,}", "\n\n", texto).strip()
                if texto:
                    paginas.append((numero, texto))
        else:
            texto = p.read_text(encoding="utf-8", errors="ignore").strip()
            paginas = [(1, texto)] if texto else []
    except Exception:
        return None
    if not paginas:
        return None
    total = "\n\n".join(t for _, t in paginas)
    return {"arquivo": arquivo, "nome": p.name, "paginas": paginas, "hash": hashlib.sha256(total.encode()).hexdigest()}


def escolher_documentos(question, catalogo_docs):
    ref = it_referenciada(question)
    if ref:
        numero, ano = ref
        candidatos = [
            d for d in catalogo_docs
            if re.search(rf"\bit\s*(?:n|no)?\s*0?{numero}\s*(?:-|/|\s)*{ano}\b", d["nome_norm"])
        ]
        if candidatos:
            return candidatos

    q = tokens(question)
    if {"unidade", "passagem"} & q or {"saida", "saidas", "emergencia"} & q:
        candidatos = [
            d for d in catalogo_docs
            if re.search(r"\bit\s*(?:n|no)?\s*0?11\s*(?:-|/|\s)*(2025|25)\b", d["nome_norm"])
        ]
        if candidatos:
            return candidatos

    ranked = sorted(catalogo_docs, key=lambda d: len(q & set(d["nome_norm"].split())), reverse=True)
    return ranked[:5]


def pontuar_pagina(question, texto):
    q = tokens(question)
    n = normalizar(texto)
    score = 4 * len(q & tokens(texto))
    frases = {
        "unidade de passagem": 100,
        "unidades de passagem": 100,
        "largura das saidas": 100,
        "largura minima": 80,
        "dimensionamento das saidas": 80,
        "capacidade da unidade de passagem": 70,
        "calculo da largura": 80,
        "largura da saida": 80,
        "saida de emergencia": 90,
    }
    for frase, peso in frases.items():
        if frase in n:
            score += peso
    grupos_q = grupos(question)
    grupos_p = grupos(texto)
    if grupos_q:
        score += 500 if grupos_q & grupos_p else -40
    nums_q = set(re.findall(r"\b\d+(?:[.,]\d+)?\b", normalizar(question)))
    nums_p = set(re.findall(r"\b\d+(?:[.,]\d+)?\b", n))
    score += 10 * len(nums_q & nums_p)
    return score


def trecho_relevante(texto, question):
    linhas = [x.strip() for x in str(texto).splitlines() if x.strip()]
    if len(linhas) <= 50:
        return "\n".join(linhas)
    grupos_q = grupos(question)
    palavras = tokens(question)
    indices = []
    for i, linha in enumerate(linhas):
        nl = normalizar(linha)
        gp = grupos(linha)
        relevancia = len(palavras & tokens(linha))
        if grupos_q and grupos_q & gp:
            relevancia += 100
        if any(p in nl for p in (
            "unidade de passagem", "largura das saidas", "largura minima",
            "dimensionamento das saidas", "saida de emergencia", "largura da saida"
        )):
            relevancia += 50
        if relevancia > 0:
            indices.append((relevancia, i))
    if not indices:
        return "\n".join(linhas[:100])
    indices.sort(reverse=True)
    escolhidos = set()
    for _, i in indices[:15]:
        for j in range(max(0, i - 3), min(len(linhas), i + 5)):
            escolhidos.add(j)
    return "\n".join(linhas[i] for i in sorted(escolhidos))


def recuperar(question, docs):
    candidatos = []
    for doc in docs:
        for pagina, texto in doc["paginas"]:
            score = pontuar_pagina(question, texto)
            if score > 0:
                candidatos.append({"arquivo": doc["arquivo"], "pagina": pagina, "texto": texto, "score": score})
    candidatos.sort(key=lambda x: x["score"], reverse=True)
    saida = []
    vistos = set()
    for item in candidatos:
        chave = (item["arquivo"], item["pagina"])
        if chave in vistos:
            continue
        vistos.add(chave)
        saida.append({**item, "texto": trecho_relevante(item["texto"], question)})
        if len(saida) >= 8:
            break
    return saida


def contexto(passagens):
    return "\n\n".join(
        f"[EVIDÊNCIA {i}]\nDocumento: {p['arquivo']}\nPágina: {p['pagina']}\n{p['texto']}"
        for i, p in enumerate(passagens, start=1)
    )


def cliente():
    chave = st.secrets.get("GEMINI_API_KEY", os.getenv("GEMINI_API_KEY", ""))
    return genai.Client(api_key=chave) if chave else None


def gerar_local(c, question, passagens):
    resposta = c.models.generate_content(
        model=MODEL,
        contents=PROMPT.format(question=question, context=contexto(passagens)),
        config=types.GenerateContentConfig(temperature=0, max_output_tokens=900),
    )
    return (resposta.text or "").strip()


def validar(texto, question):
    if not texto or ABSTAIN in texto:
        return None
    baixo = texto.lower()
    proibidos = (
        "wait,", "let's look", "i need to", "let me", "reasoning:",
        "analyzing", "vamos analisar", "vou analisar", "preciso verificar",
        "raciocínio:", "raciocinio:"
    )
    if any(x in baixo for x in proibidos):
        return None

    # Não exigir que o modelo repita o código do grupo na resposta.
    # Isso causava falso negativo em respostas corretas como:
    # "São necessárias 2 unidades de passagem...".
    grupos_q = grupos(question)
    if grupos_q:
        grupos_resposta = grupos(texto)
        grupos_contraditorios = grupos_resposta - grupos_q
        if grupos_contraditorios:
            return None
    return texto


def pesquisar_web(c, question):
    resposta = c.models.generate_content(
        model=MODEL,
        contents=WEB_PROMPT + "\n\nPERGUNTA:\n" + question,
        config=types.GenerateContentConfig(
            temperature=0,
            max_output_tokens=900,
            tools=[types.Tool(google_search=types.GoogleSearch())],
        ),
    )
    return (resposta.text or "").strip()


def mostrar_fontes(passagens):
    if not passagens:
        return
    with st.expander("Documentos encontrados na base"):
        for p in passagens:
            st.write(f"• {p['arquivo']} — página {p['pagina']}")


st.title("ROMUS.IA")
st.caption("Inteligência artificial técnica e objetiva.")
question = st.text_area("Digite sua pergunta:", height=110)
web_ok = st.checkbox("Pesquisar na web se a base não responder", value=True)
c1, c2 = st.columns(2)
with c1:
    perguntar = st.button("Perguntar", type="primary")
with c2:
    recarregar = st.button("Recarregar base")

if recarregar:
    st.cache_data.clear()
    st.rerun()

if perguntar and question.strip():
    question = question.strip()

    with st.spinner("Localizando documentos..."):
        cat = catalogo(assinatura_base())
        docs = []
        hashes = set()
        for item in escolher_documentos(question, cat):
            doc = ler(item["arquivo"], item["tamanho"], item["mtime"])
            if doc and doc["hash"] not in hashes:
                docs.append(doc)
                hashes.add(doc["hash"])

    with st.spinner("Localizando as páginas relevantes..."):
        passagens = recuperar(question, docs)

    st.markdown("### ROMUS.IA")
    c = cliente()
    resposta_local = None
    erro_local = None

    if c and passagens:
        with st.spinner("Consultando a evidência local..."):
            try:
                resposta_local = validar(gerar_local(c, question, passagens), question)
            except Exception as e:
                erro_local = str(e)

    if resposta_local:
        st.caption("Fonte: base local do ROMUS.IA")
        st.write(resposta_local)
        mostrar_fontes(passagens)
        st.stop()

    if web_ok and c:
        with st.spinner("Base local não produziu resposta. Consultando fontes técnicas na web..."):
            try:
                web_text = pesquisar_web(c, question)
            except Exception as e:
                web_text = ""
                erro_web = str(e)
            else:
                erro_web = None
        if web_text:
            st.caption("Fonte: pesquisa na internet — base local sem resposta válida")
            st.write(web_text)
            mostrar_fontes(passagens)
            st.stop()
    else:
        erro_web = None

    if erro_local or erro_web:
        with st.expander("Diagnóstico técnico"):
            if erro_local:
                st.code("LOCAL: " + erro_local)
            if erro_web:
                st.code("WEB: " + erro_web)

    if passagens:
        st.error("A evidência foi localizada, mas o mecanismo de resposta não conseguiu gerar uma resposta válida.")
    else:
        st.warning(ABSTAIN)
    mostrar_fontes(passagens)
