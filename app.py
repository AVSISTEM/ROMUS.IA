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

# Prompt adaptado de padrões públicos de RAG com resposta grounded e citações verificáveis.
RAG_PROMPT = """Você é o ROMUS.IA, assistente técnico de segurança contra incêndio.

RESPONDA SOMENTE COM BASE NAS PASSAGENS DA BASE LOCAL.

REGRAS OBRIGATÓRIAS:
1. Não use conhecimento externo, memória do modelo ou suposições.
2. Não invente números, unidades, artigos, itens, tabelas, fórmulas ou exceções.
3. Preserve exatamente os números e unidades existentes nas passagens.
4. Se a resposta não estiver explicitamente nas passagens, responda EXATAMENTE:
A base local não contém informação suficiente para responder com segurança.
5. Cada frase da resposta deve terminar com uma citação no formato [P1], [P2] etc.
6. Use somente os identificadores de passagem fornecidos.
7. Se a pergunta pedir cálculo, só calcule se todos os valores e a regra de cálculo estiverem nas passagens.
8. Se houver conflito entre passagens, não escolha por conta própria; informe o conflito e cite as passagens.
9. Seja direto. Para pergunta objetiva, dê a resposta objetiva e a referência normativa.

PERGUNTA:
{question}

PASSAGENS DA BASE LOCAL:
{context}
"""

WEB_PROMPT = """A base local foi consultada e não contém evidência suficiente.
Pesquise fontes oficiais ou técnicas confiáveis. Não invente.
Separe claramente a informação encontrada na internet da base local.
Priorize legislação, normas e documentos oficiais.
"""

STOP = {
    "para", "uma", "com", "qual", "quais", "quanto", "quantas", "deve", "ser", "das", "dos",
    "pessoas", "edificacao", "edificacoes", "populacao", "numero", "minima", "minimo",
    "necessarias", "necessarios", "conforme", "sobre", "segundo", "como", "que", "um", "o", "a", "e", "de"
}


def normalizar(texto):
    texto = unicodedata.normalize("NFKD", str(texto).lower())
    texto = "".join(c for c in texto if not unicodedata.combining(c))
    return [x for x in re.sub(r"[^\w\s]", " ", texto).split() if len(x) > 1]


def norm(texto):
    return " ".join(normalizar(texto))


def refs(question):
    q = norm(question)
    padrao = r"\bit\s*(?:(?:n|no)\s*)?0?(\d{1,2})\s*[-/ ]\s*(20\d{2}|\d{2})\b"
    return [
        (int(n), int(a) if len(a) == 4 else 2000 + int(a))
        for n, a in re.findall(padrao, q)
    ]


def eh_it(nome, numero, ano=2025):
    n = norm(nome)
    return bool(
        re.search(rf"\bit\s*(?:(?:n|no)\s*)?0?{numero}\s*(?:-|/|\s)*{ano}\b", n, re.I)
        or (
            ano == 2025
            and re.search(rf"\bit\s*(?:(?:n|no)\s*)?0?{numero}\s*(?:-|/|\s)*25\b", n, re.I)
        )
    )


def assinatura_base():
    KNOWLEDGE_DIR.mkdir(parents=True, exist_ok=True)
    out = []
    for p in sorted(KNOWLEDGE_DIR.rglob("*")):
        if p.is_file() and p.suffix.lower() in {".pdf", ".txt", ".md"}:
            try:
                s = p.stat()
                out.append((str(p.relative_to(KNOWLEDGE_DIR)), s.st_size, s.st_mtime_ns))
            except OSError:
                pass
    return tuple(out)


@st.cache_data(show_spinner=False)
def catalogo(sig):
    del sig
    out = []
    for p in sorted(KNOWLEDGE_DIR.rglob("*")):
        if p.is_file() and p.suffix.lower() in {".pdf", ".txt", ".md"}:
            s = p.stat()
            out.append({
                "arquivo": str(p.relative_to(KNOWLEDGE_DIR)),
                "nome": p.name,
                "norm": norm(p.name),
                "tamanho": s.st_size,
                "mtime": s.st_mtime_ns,
            })
    return out


@st.cache_data(show_spinner=False)
def ler(arquivo, tamanho, mtime):
    del tamanho, mtime
    p = KNOWLEDGE_DIR / arquivo
    try:
        if p.suffix.lower() == ".pdf":
            paginas = []
            for numero, page in enumerate(PdfReader(str(p)).pages, start=1):
                texto = re.sub(r"[ \t]+", " ", (page.extract_text() or "")).strip()
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
    return {
        "arquivo": arquivo,
        "nome": p.name,
        "paginas": paginas,
        "hash": hashlib.sha256(total.encode()).hexdigest(),
    }


def selecionar(question, cat):
    r = refs(question)
    for numero, ano in r:
        docs = [d for d in cat if eh_it(d["nome"], numero, ano)]
        if docs:
            return docs[:1]

    qn = set(normalizar(question))
    if {"unidade", "passagem"} <= qn or "saida" in qn or "saidas" in qn or "emergencia" in qn:
        docs = [d for d in cat if eh_it(d["nome"], 11)]
        if docs:
            return docs[:1]

    if "decreto" in qn or "regulamento" in qn:
        docs = [
            d for d in cat
            if "decreto" in d["norm"] and ("69 118" in d["norm"] or "regulamento" in d["norm"])
        ]
        if docs:
            return docs[:1]

    ranked = sorted(cat, key=lambda d: len(qn & set(d["norm"].split())), reverse=True)
    return [d for d in ranked[:3] if len(qn & set(d["norm"].split())) >= 2]


def pagina_score(question, page_text, filename):
    q = set(normalizar(question)) - STOP
    b = set(normalizar(page_text))
    bn = norm(page_text)
    score = 5 * len(q & b)

    phrases = [
        "unidade de passagem", "unidades de passagem", "largura das saidas", "largura minima",
        "larguras minimas", "dimensionamento das saidas", "capacidade da unidade de passagem",
    ]
    score += 35 * sum(1 for p in phrases if p in bn)

    for numero, ano in refs(question):
        if eh_it(filename, numero, ano):
            score += 100

    nq = set(re.findall(r"\b\d+(?:[.,]\d+)?\b", norm(question)))
    nb = set(re.findall(r"\b\d+(?:[.,]\d+)?\b", bn))
    score += 8 * len(nq & nb)
    return score


def recuperar(question, docs):
    candidatos = []
    for d in docs:
        for pagina, texto in d["paginas"]:
            score = pagina_score(question, texto, d["nome"])
            if score >= 10:
                candidatos.append({
                    "id": None,
                    "arquivo": d["arquivo"],
                    "pagina": pagina,
                    "texto": texto,
                    "score": score,
                })

    candidatos.sort(key=lambda x: x["score"], reverse=True)

    finais = []
    vistos = set()
    for item in candidatos:
        chave = (item["arquivo"], item["pagina"])
        if chave in vistos:
            continue
        vistos.add(chave)
        item["id"] = f"P{len(finais) + 1}"
        finais.append(item)
        if len(finais) >= 3:
            break
    return finais


def cliente():
    key = st.secrets.get("GEMINI_API_KEY", os.getenv("GEMINI_API_KEY", ""))
    return genai.Client(api_key=key) if key else None


def gerar_resposta(c, question, passages):
    context = "\n\n".join(
        f"[{p['id']}] FONTE: {p['arquivo']} | PÁGINA: {p['pagina']}\n{p['texto']}"
        for p in passages
    )
    prompt = RAG_PROMPT.format(question=question, context=context)
    return c.models.generate_content(
        model=MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0,
            max_output_tokens=700,
        ),
    )


def validar_resposta(texto, passages):
    texto = (texto or "").strip()
    if not texto:
        return None
    if ABSTAIN in texto:
        return ABSTAIN
    ids = {f"[{p['id']}]" for p in passages}
    citacoes = set(re.findall(r"\[P\d+\]", texto))
    if not citacoes or not citacoes <= ids:
        return None
    return texto


def pesquisar_web(c, question):
    prompt = WEB_PROMPT + "\n\nPERGUNTA:\n" + question
    return c.models.generate_content(
        model=MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            max_output_tokens=800,
            tools=[types.Tool(google_search=types.GoogleSearch())],
        ),
    )


def fontes(passages):
    if not passages:
        return
    with st.expander("Documentos encontrados na base"):
        for p in passages:
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

    with st.spinner("Localizando o documento correto..."):
        cat = catalogo(assinatura_base())
        docs = []
        hashes = set()
        for d in selecionar(question, cat):
            x = ler(d["arquivo"], d["tamanho"], d["mtime"])
            if x and x["hash"] not in hashes:
                docs.append(x)
                hashes.add(x["hash"])

    with st.spinner("Localizando as páginas relevantes..."):
        passages = recuperar(question, docs)

    st.caption("Fonte: base local do ROMUS.IA")

    c = cliente()
    resposta = None
    if passages and c:
        with st.spinner("Verificando a resposta na evidência local..."):
            try:
                resposta = validar_resposta(gerar_resposta(c, question, passages), passages)
            except Exception:
                resposta = None

    if resposta and resposta != ABSTAIN:
        st.markdown("### ROMUS.IA")
        st.write(resposta)
        fontes(passages)
        st.stop()

    if web_ok and c:
        try:
            web_text = (pesquisar_web(c, question).text or "").strip()
        except Exception:
            web_text = ""
        if web_text:
            st.caption("Fonte: pesquisa na internet — base local insuficiente")
            st.markdown("### ROMUS.IA")
            st.write(web_text)
            fontes(passages)
            st.stop()

    st.markdown("### ROMUS.IA")
    st.warning(ABSTAIN)
    fontes(passages)
