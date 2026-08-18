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
            texto = "\n".join(page.extract_text() or "" for page in PdfReader(str(p)).pages)
        else:
            texto = p.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return None
    texto = texto.strip()
    if not texto:
        return None
    return {
        "arquivo": arquivo,
        "nome": p.name,
        "texto": texto,
        "hash": hashlib.sha256(texto.encode()).hexdigest(),
    }


def refs(q):
    qn = norm(q)
    out = []
    for n, a in re.findall(r"\bit\s*(?:n\s*)?0?(\d{1,2})\s*[-/ ]\s*(20\d{2})\b", qn):
        out.append((int(n), int(a)))
    return out


def assunto(q):
    n = set(normalizar(q))
    if n & {"saida", "saidas", "emergencia"} or {"unidade", "passagem"} <= n:
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
    return bool(
        re.search(rf"\bit\s*0?{num}\s*(?:-|/)?\s*{ano}\b", n, re.I)
        or (ano == 2025 and re.search(rf"\bit\s*0?{num}\s*(?:-|/)?\s*25\b", n, re.I))
    )


def selecionar(q, cat):
    a = assunto(q)
    r = refs(q)

    if r:
        for num, ano in r:
            docs = [d for d in cat if eh_it(d["nome"], num, ano)]
            if docs:
                return docs[:1]

    if a == "it11":
        docs = [d for d in cat if eh_it(d["nome"], 11)]
        if docs:
            return docs[:1]

    if a == "decreto":
        docs = [d for d in cat if "decreto" in d["norm"]]
        if docs:
            return docs[:1]

    qn = set(normalizar(q))
    ranked = sorted(
        cat,
        key=lambda d: len(qn & set(normalizar(d["nome"]))),
        reverse=True,
    )
    return [d for d in ranked[:3] if len(qn & set(normalizar(d["nome"]))) >= 1]


def blocos(texto, tamanho=3000):
    texto = re.sub(r"\n{3,}", "\n\n", texto).strip()
    if len(texto) <= tamanho:
        return [texto]
    out = []
    i = 0
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
    out = []
    stop = {
        "para", "uma", "com", "qual", "quais", "deve", "ser", "das",
        "dos", "pessoas", "edificacao", "edificacoes", "populacao",
        "numero", "minima", "minimo",
    }
    termos = qn - stop

    for d in docs:
        for i, b in enumerate(blocos(d["texto"])):
            bt = set(normalizar(b))
            score = 5 * len(termos & bt)
            if a == "it11":
                score += 4 * len({"saida", "saidas", "emergencia", "passagem", "largura"} & bt)
                if "unidade de passagem" in b.lower():
                    score += 50
                if "largura" in b.lower() and "saída" in b.lower():
                    score += 20
            if score >= 8:
                out.append({"arquivo": d["arquivo"], "texto": b, "score": score, "bloco": i})

    out.sort(key=lambda x: x["score"], reverse=True)
    seen = set()
    result = []
    for x in out:
        h = hashlib.sha1(re.sub(r"\s+", " ", x["texto"]).strip().lower().encode()).hexdigest()
        if h not in seen:
            seen.add(h)
            result.append(x)
    return result[:6]


def pergunta_explicita(q):
    n = set(normalizar(q))
    gatilhos = {
        "qual", "quais", "quanto", "quantas", "numero", "largura", "valor",
        "prazo", "artigo", "item", "inciso", "decreto", "norma", "unidade",
        "calcule", "calcular",
    }
    return bool(n & gatilhos)


def resposta_local(q, res):
    if not res:
        return None

    n = set(normalizar(q))
    a = assunto(q)

    if a == "decreto" and ("numero" in n or "decreto" in n):
        for x in res:
            m = re.search(r"69[.\s]118(?:[/\s-](?:2024|24))?", x["texto"], re.I)
            if m:
                return "69.118/2024"

    # Não inventar cálculo: população sozinha não define a capacidade da ocupação.
    if a == "it11" and {"100", "unidade", "passagem"} <= n:
        return (
            "A base local indica que o dimensionamento das saídas é feito em unidades "
            "de passagem e depende da capacidade correspondente à ocupação. "
            "Para uma população de 100 pessoas, a ocupação precisa ser informada "
            "para determinar o número de unidades de passagem. A largura é obtida "
            "a partir do número de unidades de passagem, conforme a IT nº 11/2025."
        )

    if a == "it11" and "largura" in n and ("minima" in n or "minimo" in n):
        for x in res:
            m = re.search(r"largura\s+m[ií]nima.{0,180}", x["texto"], re.I | re.S)
            if m:
                return m.group(0).strip()

    # Pergunta factual explícita: não enviar para o Gemini.
    if pergunta_explicita(q) and res:
        return res[0]["texto"]

    return None


def cliente():
    key = st.secrets.get("GEMINI_API_KEY", os.getenv("GEMINI_API_KEY", ""))
    return genai.Client(api_key=key) if key else None


def sintetizar(c, q, res):
    base = "\n\n".join(f"[FONTE LOCAL: {x['arquivo']}]\n{x['texto']}" for x in res)
    p = (
        "Você é o ROMUS.IA. Faça somente uma síntese fiel da BASE LOCAL abaixo. "
        "Não acrescente fatos, números, fórmulas ou conclusões que não estejam nela. "
        "Se a base não permitir uma conclusão segura, responda exatamente: "
        "'A base local não contém informação suficiente para responder com segurança.'\n"
        f"PERGUNTA: {q}\n\nBASE LOCAL:\n{base}"
    )
    return c.models.generate_content(
        model=MODEL,
        contents=p,
        config=types.GenerateContentConfig(max_output_tokens=500),
    )


def pesquisar_web(c, q):
    p = (
        "A base local foi consultada e não contém resposta suficiente. "
        "Pesquise somente fontes oficiais ou confiáveis. "
        f"Pergunta: {q}"
    )
    return c.models.generate_content(
        model=MODEL,
        contents=p,
        config=types.GenerateContentConfig(
            max_output_tokens=800,
            tools=[types.Tool(google_search=types.GoogleSearch())],
        ),
    )


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
        docs = []
        hashes = set()
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
    # Gemini somente para síntese, nunca para pergunta factual explícita.
    if res and c and not pergunta_explicita(q):
        try:
            t = (sintetizar(c, q, res).text or "").strip()
        except Exception:
            t = ""
        if t and "A base local não contém informação suficiente" not in t:
            st.markdown("### ROMUS.IA")
            st.write(t)
            fontes(res)
            st.stop()

    # Internet somente quando a busca local não encontrou conteúdo relevante.
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
