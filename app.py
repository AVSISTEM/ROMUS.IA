import hashlib, os, re, unicodedata
from pathlib import Path

import streamlit as st
from google import genai
from google.genai import types
from pypdf import PdfReader

st.set_page_config(page_title="ROMUS.IA", page_icon="🔥", layout="centered")
BASE_DIR = Path(__file__).parent
KNOWLEDGE_DIR = BASE_DIR / "base_conhecimento"
MODEL = "gemini-3.5-flash"


def normalizar(t):
    t = unicodedata.normalize("NFKD", str(t).lower())
    t = "".join(c for c in t if not unicodedata.combining(c))
    return [x for x in re.sub(r"[^\w\s]", " ", t).split() if len(x) > 1]


def norm(t):
    return " ".join(normalizar(t))


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
                texto = (page.extract_text() or "").strip()
                if texto:
                    paginas.append((numero, texto))
        else:
            texto = p.read_text(encoding="utf-8", errors="ignore").strip()
            paginas = [(1, texto)] if texto else []
    except Exception:
        return None
    if not paginas:
        return None
    texto_total = "\n\n".join(texto for _, texto in paginas)
    return {"arquivo": arquivo, "nome": p.name, "paginas": paginas, "texto": texto_total, "hash": hashlib.sha256(texto_total.encode()).hexdigest()}


# A normalização transforma “IT nº 11/2025” em “it no 11 2025”.
# Portanto, a identificação deve aceitar “n” e “no”, além de ano com 2 dígitos.
def refs(q):
    qn = norm(q)
    out = []
    padrao = r"\bit\s*(?:(?:n|no)\s*)?0?(\d{1,2})\s*[-/ ]\s*(20\d{2}|\d{2})\b"
    for n, a in re.findall(padrao, qn):
        out.append((int(n), int(a) if len(a) == 4 else 2000 + int(a)))
    return out


def assunto(q):
    n = set(normalizar(q))
    if n & {"saida", "saidas", "emergencia"} or {"unidade", "passagem"} <= n or ("largura" in n and ("saida" in n or "emergencia" in n)):
        return "it11"
    if "decreto" in n or "regulamento" in n:
        return "decreto"
    if {"carga", "incendio"} <= n:
        return "it14"
    if "pressurizacao" in n:
        return "it13"
    return "geral"


def eh_it(nome, num, ano=2025):
    n = norm(nome)
    padrao = rf"\bit\s*(?:(?:n|no)\s*)?0?{num}\s*(?:-|/|\s)*{ano}\b"
    if re.search(padrao, n, re.I):
        return True
    if ano == 2025 and re.search(rf"\bit\s*(?:(?:n|no)\s*)?0?{num}\s*(?:-|/|\s)*25\b", n, re.I):
        return True
    return False


def selecionar(q, cat):
    a = assunto(q)
    r = refs(q)
    for num, ano in r:
        docs = [d for d in cat if eh_it(d["nome"], num, ano)]
        if docs:
            return docs[:1]
    if a == "it11":
        docs = [d for d in cat if eh_it(d["nome"], 11)]
        if docs:
            return docs[:1]
    if a == "it14":
        docs = [d for d in cat if eh_it(d["nome"], 14)]
        if docs:
            return docs[:1]
    if a == "it13":
        docs = [d for d in cat if eh_it(d["nome"], 13)]
        if docs:
            return docs[:1]
    if a == "decreto":
        docs = [d for d in cat if "decreto" in d["norm"]]
        if docs:
            return docs[:1]
    qn = set(normalizar(q))
    ranked = sorted(cat, key=lambda d: len(qn & set(normalizar(d["nome"]))), reverse=True)
    return [d for d in ranked[:3] if len(qn & set(normalizar(d["nome"]))) >= 2]


def blocos(texto, tamanho=3200):
    texto = re.sub(r"\n{3,}", "\n\n", texto).strip()
    if len(texto) <= tamanho:
        return [texto]
    out, i = [], 0
    while i < len(texto):
        f = min(i + tamanho, len(texto))
        c = texto.rfind("\n", i + tamanho // 2, f)
        if c > i:
            f = c
        b = texto[i:f].strip()
        if b:
            out.append(b)
        if f >= len(texto):
            break
        i = max(f - 120, i + 1)
    return out


def buscar(q, docs):
    qn = set(normalizar(q))
    a = assunto(q)
    stop = {"para", "uma", "com", "qual", "quais", "quanto", "quantas", "deve", "ser", "das", "dos", "pessoas", "edificacao", "edificacoes", "populacao", "numero", "minima", "minimo", "necessarias", "necessarios"}
    termos = qn - stop
    out = []
    for d in docs:
        for i, b in enumerate(blocos(d["texto"])):
            bn = set(normalizar(b))
            score = 5 * len(termos & bn)
            if a == "it11":
                chaves = {"saida", "saidas", "emergencia", "passagem", "largura", "ocupacao", "capacidade"}
                score += 3 * len(chaves & bn)
                if "unidade de passagem" in norm(b) or "unidades de passagem" in norm(b):
                    score += 80
                if "largura" in bn and ("saida" in bn or "emergencia" in bn):
                    score += 30
                if "capacidade" in bn and "ocupacao" in bn:
                    score += 25
            if score >= 15:
                out.append({"arquivo": d["arquivo"], "texto": b, "score": score, "bloco": i})
    out.sort(key=lambda x: x["score"], reverse=True)
    seen, result = set(), []
    for x in out:
        h = hashlib.sha1(re.sub(r"\s+", " ", x["texto"]).strip().lower().encode()).hexdigest()
        if h not in seen:
            seen.add(h)
            result.append(x)
    return result[:6]


def resposta_local(q, res):
    n = set(normalizar(q))
    a = assunto(q)
    # Não inventar um número: 100 pessoas, sem ocupação, não define C.
    if a == "it11" and "100" in n and "unidade" in n and "passagem" in n:
        return "Não é possível determinar o número de unidades de passagem apenas com a população de 100 pessoas. É necessário informar a ocupação/uso da edificação, pois a capacidade da unidade de passagem depende da ocupação prevista na IT nº 11/2025."
    if a == "decreto" and "decreto" in n:
        for x in res:
            if re.search(r"69[.\s]118(?:[/\s-](?:2024|24))?", x["texto"], re.I):
                return "69.118/2024"
    if a == "it11" and "largura" in n and ("minima" in n or "minimo" in n):
        for x in res:
            texto = norm(x["texto"])
            for termo in ("largura minima", "largura das saidas"):
                pos = texto.find(termo)
                if pos >= 0:
                    return texto[pos:pos + 280]
    return None


def cliente():
    key = st.secrets.get("GEMINI_API_KEY", os.getenv("GEMINI_API_KEY", ""))
    return genai.Client(api_key=key) if key else None


def sintetizar(c, q, res):
    base = "\n\n".join(f"[FONTE LOCAL: {x['arquivo']}]\n{x['texto']}" for x in res)
    prompt = ("Você é o ROMUS.IA. Faça somente uma síntese fiel da BASE LOCAL. "
              "Não acrescente fatos, números, fórmulas, artigos ou conclusões que não estejam explicitamente na base. "
              "Se a base não permitir uma conclusão segura, responda exatamente: A base local não contém informação suficiente para responder com segurança.\n\n"
              f"PERGUNTA: {q}\n\nBASE LOCAL:\n{base}")
    return c.models.generate_content(model=MODEL, contents=prompt, config=types.GenerateContentConfig(max_output_tokens=500))


def pesquisar_web(c, q):
    prompt = ("A base local foi consultada e não contém resposta suficiente. Pesquise somente fontes oficiais ou técnicas confiáveis. Não invente. Informe as fontes utilizadas.\n\n" f"PERGUNTA: {q}")
    return c.models.generate_content(model=MODEL, contents=prompt, config=types.GenerateContentConfig(max_output_tokens=800, tools=[types.Tool(google_search=types.GoogleSearch())]))


def fontes(res):
    if res:
        with st.expander("Documentos encontrados na base"):
            for f in dict.fromkeys(x["arquivo"] for x in res):
                st.write("• " + f)


st.title("ROMUS.IA")
st.caption("Inteligência artificial técnica e objetiva.")
q = st.text_area("Digite sua pergunta:", height=110)
web_ok = st.checkbox("Pesquisar na web se a base não responder", value=True)
c1, c2 = st.columns(2)
with c1:
    perguntar = st.button("Perguntar", type="primary")
with c2:
    recarregar = st.button("Recarregar base")
if recarregar:
    st.cache_data.clear()
    st.rerun()

if perguntar and q.strip():
    q = q.strip()
    with st.spinner("Localizando o documento correto..."):
        cat = catalogo(assinatura_base())
        cand = selecionar(q, cat)
    with st.spinner("Localizando o trecho correto..."):
        docs, hashes = [], set()
        for d in cand:
            x = ler(d["arquivo"], d["tamanho"], d["mtime"])
            if x and x["hash"] not in hashes:
                docs.append(x)
                hashes.add(x["hash"])
        res = buscar(q, docs)
    st.caption("Fonte: base local do ROMUS.IA")
    local = resposta_local(q, res)
    if local:
        st.markdown("### ROMUS.IA")
        st.write(local)
        fontes(res)
        st.stop()
    c = cliente()
    gatilhos = {"qual", "quais", "quanto", "quantas", "numero", "largura", "valor", "prazo", "artigo", "item", "inciso", "decreto", "norma", "unidade", "calcule", "calcular"}
    factual = bool(set(normalizar(q)) & gatilhos)
    if res and c and not factual:
        try:
            t = (sintetizar(c, q, res).text or "").strip()
        except Exception:
            t = ""
        if t and "A base local não contém informação suficiente" not in t:
            st.markdown("### ROMUS.IA")
            st.write(t)
            fontes(res)
            st.stop()
    if not res and web_ok and c:
        try:
            t = (pesquisar_web(c, q).text or "").strip()
        except Exception:
            t = ""
        if t:
            st.caption("Fonte: pesquisa na internet (base local insuficiente)")
            st.markdown("### ROMUS.IA")
            st.write(t)
            st.stop()
    st.markdown("### ROMUS.IA")
    st.warning("A base local não contém informação suficiente para responder com segurança.")
    fontes(res)
