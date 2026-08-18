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

Sua única fonte de verdade nesta resposta são as passagens da BASE LOCAL fornecidas abaixo.

REGRAS OBRIGATÓRIAS:
1. Responda SOMENTE com informação que esteja nas passagens.
2. Não complete lacunas com conhecimento próprio.
3. Não invente números, unidades, artigos, itens, tabelas ou conclusões.
4. Preserve exatamente números, unidades, datas e referências normativas encontradas.
5. Se a pergunta pedir um cálculo, só faça o cálculo quando todos os valores necessários estiverem nas passagens.
6. Não misture documentos diferentes para criar uma regra que não esteja expressamente sustentada.
7. Se as passagens não permitirem responder com segurança, responda exatamente com a frase de insuficiência.
8. Para perguntas objetivas, responda primeiro com a resposta objetiva e depois, se necessário, indique o item/página que a sustenta.
9. Nunca diga que uma informação está na base sem que ela esteja efetivamente nas passagens.

PERGUNTA:
{question}

BASE LOCAL:
{context}

RESPOSTA:
"""

EXTRACTIVE_PROMPT = """Atue como um extrator rigoroso de informação normativa.

Use exclusivamente o texto da BASE LOCAL abaixo. A tarefa é encontrar a resposta para a pergunta, não explicar conhecimento geral.

REGRAS:
- Copie literalmente os trechos necessários da base quando eles contiverem a resposta.
- Preserve números, unidades, sinais, artigos, itens e referências.
- Se houver uma fórmula ou regra de cálculo na base, reproduza-a e aplique-a somente se todos os valores necessários estiverem presentes.
- Não invente informação e não use conhecimento externo.
- Se a resposta não estiver comprovada no texto fornecido, responda exatamente:
A base local não contém informação suficiente para responder com segurança.
- Seja curto e objetivo.

PERGUNTA:
{question}

BASE LOCAL:
{context}

RESPOSTA:
"""

WEB_PROMPT = """A base local foi consultada e não contém evidência suficiente para responder.
Pesquise somente fontes oficiais ou técnicas confiáveis. Não invente informação.
Responda diretamente à pergunta e diferencie a informação encontrada na internet da base local."""

STOP = {"para","uma","com","qual","quais","quanto","quantas","deve","ser","das","dos","pessoas","edificacao","edificacoes","populacao","numero","minima","minimo","necessarias","necessarios","conforme","sobre","segundo","como","que","um","o","a","e","de","da","do","na","no"}


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
    for d in docs:
        for pagina, texto in d["paginas"]:
            score = pagina_score(question, texto, d["nome"])
            if score > 0:
                candidatos.append({"arquivo": d["arquivo"], "pagina": pagina, "texto": texto, "score": score})
    candidatos.sort(key=lambda x: x["score"], reverse=True)

    # Não basta pegar páginas isoladas: normas frequentemente quebram uma tabela,
    # fórmula ou regra entre duas páginas consecutivas. Mantemos as páginas
    # vizinhas dos melhores resultados para preservar o contexto normativo.
    por_doc = {d["arquivo"]: d for d in docs}
    escolhidos = []
    vistos = set()
    for item in candidatos[:6]:
        d = por_doc.get(item["arquivo"])
        if not d:
            continue
        numeros = [item["pagina"] - 1, item["pagina"], item["pagina"] + 1]
        mapa = {p: t for p, t in d["paginas"]}
        for pagina in numeros:
            if pagina not in mapa:
                continue
            chave = (item["arquivo"], pagina)
            if chave in vistos:
                continue
            vistos.add(chave)
            escolhidos.append({"arquivo": item["arquivo"], "pagina": pagina, "texto": mapa[pagina], "score": item["score"] if pagina == item["pagina"] else max(1, item["score"] - 5)})
            if len(escolhidos) >= 10:
                break
        if len(escolhidos) >= 10:
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
        config=types.GenerateContentConfig(temperature=0, max_output_tokens=900),
    )


def gerar_resposta(c, question, passages):
    ctx = contexto(passages)
    primeira = gerar(c, RAG_PROMPT.format(question=question, context=ctx))
    texto = resposta_valida(primeira.text)
    if texto:
        return texto

    # Segunda passagem deliberadamente extrativa. Não é troca de "personalidade":
    # é uma validação contra o próprio texto recuperado quando o modelo se abstém.
    segunda = gerar(c, EXTRACTIVE_PROMPT.format(question=question, context=ctx))
    return resposta_valida(segunda.text)


def resposta_valida(texto):
    texto = (texto or "").strip()
    if not texto:
        return None
    if ABSTAIN in texto:
        return None
    return texto


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
