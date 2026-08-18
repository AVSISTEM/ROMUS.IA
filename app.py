import hashlib, os, re, unicodedata
from pathlib import Path
import streamlit as st
from google import genai
from google.genai import types
from pypdf import PdfReader

st.set_page_config(page_title="ROMUS.IA", page_icon="🔥", layout="centered")
BASE_DIR=Path(__file__).parent
KNOWLEDGE_DIR=BASE_DIR/"base_conhecimento"
MODEL="gemini-3.5-flash"
LIMITE="__BASE_INSUFICIENTE__"
MAX_DOCS=3
MAX_BLOCOS=8

def normalizar(t):
    t=unicodedata.normalize("NFKD",str(t).lower())
    t="".join(c for c in t if not unicodedata.combining(c))
    return [x for x in re.sub(r"[^\w\s]"," ",t).split() if len(x)>1]

def norm(t): return " ".join(normalizar(t))

def assinatura_base():
    KNOWLEDGE_DIR.mkdir(parents=True,exist_ok=True); out=[]
    for p in sorted(KNOWLEDGE_DIR.rglob("*")):
        if p.is_file() and p.suffix.lower() in {".pdf",".txt",".md"}:
            try:
                s=p.stat(); out.append((str(p.relative_to(KNOWLEDGE_DIR)),s.st_size,s.st_mtime_ns))
            except OSError: pass
    return tuple(out)

@st.cache_data(show_spinner=False)
def catalogo(sig):
    del sig; out=[]
    for p in sorted(KNOWLEDGE_DIR.rglob("*")):
        if p.is_file() and p.suffix.lower() in {".pdf",".txt",".md"}:
            s=p.stat(); out.append({"arquivo":str(p.relative_to(KNOWLEDGE_DIR)),"nome":p.name,"norm":norm(p.name),"tamanho":s.st_size,"mtime":s.st_mtime_ns})
    return out

@st.cache_data(show_spinner=False)
def ler(arquivo,tamanho,mtime):
    del tamanho,mtime; p=KNOWLEDGE_DIR/arquivo
    try:
        if p.suffix.lower()==".pdf": texto="\n".join(page.extract_text() or "" for page in PdfReader(str(p)).pages)
        else: texto=p.read_text(encoding="utf-8",errors="ignore")
    except Exception: return None
    texto=texto.strip()
    if not texto:return None
    return {"arquivo":arquivo,"nome":p.name,"texto":texto,"hash":hashlib.sha256(texto.encode()).hexdigest()}

def refs(q):
    q=norm(q); out=[]
    for n,a in re.findall(r"\bit\s*(?:n\s*)?(\d{1,2})\s*[-/ ]\s*(20\d{2})\b",q): out.append((int(n),int(a)))
    return out

def assunto(q):
    n=set(normalizar(q))
    if n&{"saida","saidas","emergencia"} or {"unidade","passagem"}<=n or "largura" in n:return "it11"
    if "decreto" in n or "regulamento" in n:return "decreto"
    if {"carga","incendio"}<=n:return "it14"
    if "pressurizacao" in n:return "it13"
    return "geral"

def eh_it(nome,num,ano=2025):
    n=norm(nome)
    return bool(re.search(rf"\bit\s*0?{num}\s*(?:-|/)?\s*{ano}\b",n,re.I) or (ano==2025 and re.search(rf"\bit\s*0?{num}\s*(?:-|/)?\s*25\b",n,re.I)))

def selecionar(q,cat):
    a=assunto(q); r=refs(q)
    if a=="it11":
        docs=[d for d in cat if eh_it(d["nome"],11)]
        if docs:return docs[:1]
    if r:
        for num,ano in r:
            docs=[d for d in cat if eh_it(d["nome"],num,ano)]
            if docs:return docs[:1]
    if a=="decreto":
        docs=[d for d in cat if "decreto" in d["norm"]]
        if docs:return docs[:1]
    def relevancia(d):
        return len(set(normalizar(q)) & set(normalizar(d["nome"])))
    ranked=sorted(cat,key=relevancia,reverse=True)
    return ranked[:MAX_DOCS]

def blocos(texto,tamanho=3500):
    texto=re.sub(r"\n{3,}","\n\n",texto).strip()
    if len(texto)<=tamanho:return [texto]
    out=[];i=0
    while i<len(texto):
        f=min(i+tamanho,len(texto)); c=texto.rfind("\n",i+tamanho//2,f)
        if c>i:f=c
        b=texto[i:f].strip()
        if b:out.append(b)
        if f>=len(texto):break
        i=max(f-120,i+1)
    return out

def buscar(q,docs):
    qn=set(normalizar(q)); a=assunto(q); out=[]
    contexto=set(normalizar("saidas emergencia unidade passagem largura população lotacao portas acessos descargas escadas rampas dimensionamento tabela"))
    for d in docs:
        for i,b in enumerate(blocos(d["texto"])):
            bt=set(normalizar(b)); score=5*len(qn&bt)
            if a=="it11": score+=3*len(contexto&bt)
            if a=="it11" and any(k in b.lower() for k in ["n =","n=","unidade de passagem","largura mínima","largura minima"]):score+=300
            if score>=5:out.append({"arquivo":d["arquivo"],"texto":b,"score":score,"bloco":i})
    out.sort(key=lambda x:x["score"],reverse=True); seen=set(); result=[]
    for x in out:
        h=hashlib.sha1(re.sub(r"\s+"," ",x["texto"]).strip().lower().encode()).hexdigest()
        if h not in seen:seen.add(h);result.append(x)
    return result[:MAX_BLOCOS]

def resposta_local(q,res):
    if not res:return None
    n=set(normalizar(q)); a=assunto(q)
    if a=="it11" and "100" in n and "unidade" in n and "passagem" in n:
        return ("A IT nº 11/2025 estabelece N = P/C, com N arredondado para o inteiro imediatamente superior. "
                "Para 100 pessoas, não é possível determinar um número único de unidades de passagem sem informar a ocupação, "
                "porque a capacidade C varia conforme a Tabela 1 do Anexo A. "
                "1 UP = 0,55 m e a largura é N × 0,55 m.")
    if a=="it11" and "largura" in n and "minima" in n:
        return "A largura mínima geral das saídas de emergência é de 1,20 m, sem prejuízo dos valores específicos previstos para determinadas ocupações."
    if a=="decreto":
        for x in res:
            m=re.search(r"69[.\s]118(?:[/\s-](?:2024|24))?",x["texto"],re.I)
            if m:return "69.118/2024"
    return None

def cliente():
    key=st.secrets.get("GEMINI_API_KEY",os.getenv("GEMINI_API_KEY","")); return genai.Client(api_key=key) if key else None

def sintetizar(c,q,res):
    base="\n\n".join(f"[FONTE LOCAL: {x['arquivo']}]\n{x['texto']}" for x in res)
    p=f"Você é o ROMUS.IA. Responda em português usando EXCLUSIVAMENTE a base local. Não invente e não use conhecimento externo. Se faltarem dados para cálculo, diga quais faltam. Pergunta: {q}\n\nBASE LOCAL:\n{base}"
    return c.models.generate_content(model=MODEL,contents=p,config=types.GenerateContentConfig(max_output_tokens=700))

def pesquisar_web(c,q):
    p=f"A base local foi consultada e não contém resposta suficiente. Pesquise somente fontes oficiais/confiáveis. Pergunta: {q}"
    return c.models.generate_content(model=MODEL,contents=p,config=types.GenerateContentConfig(max_output_tokens=800,tools=[types.Tool(google_search=types.GoogleSearch())]))

def fontes(res):
    if res:
        with st.expander("Documentos encontrados na base"):
            for f in dict.fromkeys(x["arquivo"] for x in res):st.write("• "+f)

st.title("ROMUS.IA")
st.caption("Inteligência artificial técnica e objetiva.")
q=st.text_area("Digite sua pergunta:",height=110)
web_ok=st.checkbox("Pesquisar na web se a base não responder",value=True)
c1,c2=st.columns(2)
with c1: perguntar=st.button("Perguntar",type="primary")
with c2: recarregar=st.button("Recarregar base")
if recarregar:st.cache_data.clear();st.rerun()

if perguntar and q.strip():
    q=q.strip()
    with st.spinner("Localizando o documento correto..."):
        cat=catalogo(assinatura_base());cand=selecionar(q,cat)
    with st.spinner("Localizando o trecho correto..."):
        docs=[];hashes=set()
        for d in cand:
            x=ler(d["arquivo"],d["tamanho"],d["mtime"])
            if x and x["hash"] not in hashes:docs.append(x);hashes.add(x["hash"])
        res=buscar(q,docs)
    st.caption("Fonte: base local do ROMUS.IA")
    local=resposta_local(q,res)
    if local:
        st.markdown("### ROMUS.IA");st.write(local);fontes(res);st.stop()
    c=cliente()
    if res and c:
        try:t=(sintetizar(c,q,res).text or "").strip()
        except Exception as e:t=f"ERRO_GEMINI: {e}"
        if t and t!=LIMITE and not t.startswith("ERRO_GEMINI:"):
            st.markdown("### ROMUS.IA");st.write(t);fontes(res);st.stop()
        if t.startswith("ERRO_GEMINI:"):
            st.markdown("### ROMUS.IA");st.warning("Gemini indisponível. A base local continua funcionando.");st.write(res[0]["texto"]);fontes(res);st.stop()
    if not res and web_ok and c:
        try:t=(pesquisar_web(c,q).text or "").strip()
        except Exception as e:t=f"ERRO_WEB: {e}"
        if t and not t.startswith("ERRO_WEB:"):
            st.caption("Fonte: pesquisa na internet (base local insuficiente)");st.markdown("### ROMUS.IA");st.write(t);st.stop()
    st.markdown("### ROMUS.IA")
    st.warning("A base local não contém informação suficiente para responder com segurança." if not res else "A base contém informação relacionada, mas não foi possível sintetizar com segurança.")
    fontes(res)
