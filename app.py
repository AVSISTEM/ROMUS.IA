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

RAG_PROMPT = """Você é o ROMUS.IA, assistente técnico de segurança contra incêndio.

FONTE ÚNICA: use somente a BASE LOCAL fornecida nesta mensagem.

REGRAS ABSOLUTAS:
1. Entregue SOMENTE a resposta final. NÃO mostre raciocínio, análise, rascunho, dúvidas, pensamentos ou comentários como "Wait", "Let's look", "preciso verificar" ou equivalentes.
2. Não use conhecimento próprio ou externo.
3. Não invente números, unidades, artigos, itens, tabelas, classificações ou conclusões.
4. Preserve exatamente números, unidades, datas e referências normativas existentes na base.
5. Se a pergunta informar uma classificação/código (por exemplo, F-11), use SOMENTE a linha/regra correspondente a esse código. NÃO substitua por E-5, E-6 ou qualquer outra classificação.
6. Se a pergunta trouxer população, faça o cálculo somente se a base fornecer expressamente todos os valores necessários.
7. Se faltar qualquer informação necessária, responda exatamente: A base local não contém informação suficiente para responder com segurança.
8. Não misture linhas de tabelas diferentes para fabricar uma resposta.
9. Se houver tabela, procure primeiro a linha que corresponda exatamente à classificação solicitada e depois a coluna aplicável à pergunta.
10. Para pergunta objetiva, responda primeiro de forma objetiva e curta. Depois, se necessário, indique documento, item e página.
11. Nunca cite informação que não esteja efetivamente nas passagens fornecidas.

PERGUNTA:
{question}

BASE LOCAL:
{context}

RESPOSTA FINAL:
"""

EXTRACTIVE_PROMPT = """Você é um extrator literal de norma técnica.

Use EXCLUSIVAMENTE a BASE LOCAL abaixo.

SAÍDA OBRIGATÓRIA:
- Escreva somente a resposta final em português.
- Não mostre raciocínio, análise, rascunho, dúvidas ou comentários internos.
- Não escreva frases como "Wait", "Let's look", "I need", "talvez", "acho que" ou equivalentes.
- Não reproduza linhas de tabela que não correspondam à classificação perguntada.
- Se a pergunta contiver uma classificação/código, como F-11, considere somente F-11.
- Preserve literalmente números, unidades e referências.
- Faça cálculo somente quando todos os dados necessários estiverem comprovados na base.
- Se a resposta não estiver comprovada, responda exatamente:
A base local não contém informação suficiente para responder com segurança.

PERGUNTA:
{question}

BASE LOCAL:
{context}

RESPOSTA FINAL:
"""

WEB_PROMPT = """A base local foi consultada e não contém evidência suficiente para responder.
Pesquise somente fontes oficiais ou técnicas confiáveis. Não invente informação.
Responda diretamente à pergunta e diferencie a informação encontrada na internet da base local."""

STOP = {
    "para", "uma", "com", "qual", "quais", "quanto", "quantas", "deve", "ser", "das", "dos",
    "pessoas", "edificacao", "edificacoes", "populacao", "numero", "minima", "minimo",
    "necessarias", "necessarios", "conforme", "sobre", "segundo", "como", "que", "um", "o",
    "a", "e", "de", "da", "do", "na", "no", "grupo"
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
    return [(int(n), int(a) if len(a) == 4 else 2000 + int(a)) for n, a in re.findall(padrao, q)]


def grupos(question):
    # Não usar norm() aqui: ele remove o hífen de códigos como F-11.
    return set(re.findall(r"\b[A-Z]{1,3}-\d{1,3}\b", str(question).upper()))


def eh_it(nome, numero, ano=2025):
    n = norm(nome)
    if re.search(rf"\bit\s*(?:(?:n|no)\s*)?0?{numero}\s*(?:-|/|\s)*{ano}\b", n, re.I):
        return True
    return ano == 2025 and bool(re.search(rf"\bit\s*(?:(?:n|no)\s*)?0?{numero}\s*(?:-|/|\s)*25\b", n, re.I))


@st.cache_data(show_spinner=False)
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
            out.append({"arquivo": str(p.relative_to(KNOWLEDGE_DIR)), "nome": p.name, "norm": norm(p.name), "tamanho": s.st_size, "mtime": s.st_mtime_ns})
    return out


@st.cache_data(show_spinner=False)
def ler(arquivo, tamanho, mtime):
    del tamanho, mtime
    p = KNOWLEDGE_DIR / arquivo
    try:
        if p.suffix.lower() == ".pdf":
            paginas = []
            for numero, page in enumerate(PdfReader(str(p)).pages, start=1):
                bruto = page.extract_text() or ""
                texto = bruto.replace("\u00a0", " ")
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


def selecionar(question, cat):
    for numero, ano in refs(question):
        docs = [d for d in cat if eh_it(d["nome"], numero, ano)]
        if docs:
            return docs[:1]
    qn = set(normalizar(question))
    if {"unidade", "passagem"} <= qn or bool({"saida", "saidas", "emergencia"} & qn):
        docs = [d for d in cat if eh_it(d["nome"], 11, 2025)]
        if docs:
            return docs[:1]
    if "decreto" in qn or "regulamento" in qn:
        docs = [d for d in cat if "decreto" in d["norm"] and ("69 118" in d["norm"] or "regulamento" in d["norm"])]
        if docs:
            return docs[:1]
    ranked = sorted(cat, key=lambda d: len(qn & set(d["norm"].split())), reverse=True)
    return [d for d in ranked[:3] if len(qn & set(d["norm"].split())) >= 2]


def pagina_score(question, page_text, filename):
    q = set(normalizar(question)) - STOP
    b = set(normalizar(page_text))
    bn = norm(page_text)
    score = 5 * len(q & b)

    grupos_q = grupos(question)
    if grupos_q:
        grupos_p = grupos(page_text)
        if grupos_q & grupos_p:
            score += 500
        else:
            score -= 250

    for phrase, weight in [
        ("unidade de passagem", 60),
        ("unidades de passagem", 60),
        ("largura das saidas", 60),
        ("largura minima", 50),
        ("larguras minimas", 50),
        ("dimensionamento das saidas", 50),
        ("capacidade da unidade de passagem", 45),
        ("calculo da largura", 55),
        ("calculo das larguras", 55),
        ("largura da saida", 55),
    ]:
        if phrase in bn:
            score += weight

    for numero, ano in refs(question):
        if eh_it(filename, numero, ano):
            score += 100

    nq = set(re.findall(r"\b\d+(?:[.,]\d+)?\b", norm(question)))
    nb = set(re.findall(r"\b\d+(?:[.,]\d+)?\b", bn))
    score += 8 * len(nq & nb)
    return score


def recuperar(question, docs):
    candidatos = []
    grupos_q = grupos(question)
    for d in docs:
        for pagina, texto in d["paginas"]:
            score = pagina_score(question, texto, d["nome"])
            if grupos_q and score < 0:
                continue
            if score > 0:
                candidatos.append({"arquivo": d["arquivo"], "pagina": pagina, "texto": texto, "score": score})
    candidatos.sort(key=lambda x: x["score"], reverse=True)

    por_doc = {d["arquivo"]: d for d in docs}
    escolhidos = []
    vistos = set()
    limite = 6 if grupos_q else 10

    for item in candidatos[:6]:
        d = por_doc.get(item["arquivo"])
        if not d:
            continue
        mapa = {p: t for p, t in d["paginas"]}
        numeros = [item["pagina"]]
        if not grupos_q:
            numeros += [item["pagina"] - 1, item["pagina"] + 1]

        for pagina in numeros:
            if pagina not in mapa:
                continue
            chave = (item["arquivo"], pagina)
            if chave in vistos:
                continue
            vistos.add(chave)
            escolhidos.append({"arquivo": item["arquivo"], "pagina": pagina, "texto": mapa[pagina], "score": item["score"] if pagina == item["pagina"] else max(1, item["score"] - 5)})
            if len(escolhidos) >= limite:
                break
        if len(escolhidos) >= limite:
            break

    return escolhidos


def cliente():
    key = st.secrets.get("GEMINI_API_KEY", os.getenv("GEMINI_API_KEY", ""))
    return genai.Client(api_key=key) if key else None


def contexto(passages):
    return "\n\n".join(
        f"[DOCUMENTO {i}] {p['arquivo']} | PÁGINA {p['pagina']}\n{p['texto']}"
        for i, p in enumerate(passages, 1)
    )


def gerar(c, prompt):
    return c.models.generate_content(
        model=MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(temperature=0, max_output_tokens=700),
    )


def resposta_valida(texto, question=""):
    texto = (texto or "").strip()
    if not texto or ABSTAIN in texto:
        return None

    proibidos = [
        "wait,", "let's look", "i need to", "let me", "first,", "reasoning:",
        "analyzing", "vamos analisar", "vou analisar", "preciso verificar", "raciocinio:"
    ]
    baixo = texto.lower()
    if any(x in baixo for x in proibidos):
        return None

    if re.search(r"\b(for|wait|let's|look|text|answer)\b", baixo) and re.search(r"\b(e-\d+|f-\d+)\b", baixo):
        return None

    qgrupos = grupos(question)
    if qgrupos:
        outros = set(re.findall(r"\b[A-Z]{1,3}-\d{1,3}\b", texto.upper())) - qgrupos
        if outros:
            return None

    return texto


def gerar_resposta(c, question, passages):
    ctx = contexto(passages)
    primeira = gerar(c, RAG_PROMPT.format(question=question, context=ctx))
    texto = resposta_valida(primeira.text, question)
    if texto:
        return texto
    segunda = gerar(c, EXTRACTIVE_PROMPT.format(question=question, context=ctx))
    return resposta_valida(segunda.text, question)


def pesquisar_web(c, question):
    return c.models.generate_content(
        model=MODEL,
        contents=WEB_PROMPT + "\n\nPERGUNTA:\n" + question,
        config=types.GenerateContentConfig(max_output_tokens=800, tools=[types.Tool(google_search=types.GoogleSearch())]),
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
        with st.spinner("Consultando a evidência local..."):
            try:
                resposta = gerar_resposta(c, question, passages)
            except Exception:
                resposta = None

    if resposta:
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
