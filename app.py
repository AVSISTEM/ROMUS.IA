import hashlib, os, re
from pathlib import Path
import streamlit as st
from google import genai
from google.genai import types
from pypdf import PdfReader

st.set_page_config(page_title="ROMUS.IA", page_icon="🔥", layout="centered")
BASE_DIR=Path(__file__).parent
KNOWLEDGE_DIR=BASE_DIR/"base_conhecimento"
MODEL="gemini-3.5-flash-lite"
LIMITE_INSUFICIENTE="__BASE_INSUFICIENTE__"
MAX_DOCUMENTOS=4
MAX_BLOCOS=6


def normalizar(t):
    t=t.lower().replace("º"," ").replace("°"," ")
    return [x for x in re.sub(r"[^\w\s]"," ",t,flags=re.UNICODE).split() if len(x)>1]


def texto_normalizado(t): return " ".join(normalizar(t))


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
        if not p.is_file() or p.suffix.lower() not in {".pdf",".txt",".md"}: continue
        try:
            s=p.stat(); out.append({"arquivo":str(p.relative_to(KNOWLEDGE_DIR)),"nome":p.name,"nome_norm":texto_normalizado(p.name),"tamanho":s.st_size,"mtime":s.st_mtime_ns})
        except OSError: pass
    return out


@st.cache_data(show_spinner=False)
def ler(arquivo,tamanho,mtime):
    del tamanho,mtime; p=KNOWLEDGE_DIR/arquivo
    try:
        if p.suffix.lower()==".pdf": texto="\n".join((x.extract_text() or "") for x in PdfReader(str(p)).pages)
        else: texto=p.read_text(encoding="utf-8",errors="ignore")
    except Exception: return None
    texto=texto.strip()
    if not texto:return None
    return {"arquivo":arquivo,"nome":p.name,"texto":texto,"norm":texto_normalizado(texto),"nome_norm":texto_normalizado(p.name),"hash":hashlib.sha256(texto.encode()).hexdigest()}


def refs(q):
    q=q.lower(); its=[]
    for n,a in re.findall(r"\bit\s*(?:n[ºo°]?\s*)?(\d{1,2})\s*[-/]\s*(20\d{2})\b",q): its.append((int(n),int(a)))
    for n in re.findall(r"\bit\s*(?:n[ºo°]?\s*)?(\d{1,2})\s*[-/]\s*25\b",q):
        if (int(n),2025) not in its: its.append((int(n),2025))
    return {"it":its,"item":re.findall(r"\b(?:item|subitem)\s*(\d+(?:\.\d+){1,4})\b",q)}


def assunto(q):
    n=set(normalizar(q));
    if {"saída","saida","saídas","saidas","emergência","emergencia"}&n or {"unidade","passagem"}&n or {"largura","saída","saida"}&n:
        return "it11"
    if "decreto" in n or "regulamento" in n:return "decreto"
    if {"carga","incêndio","incendio"}&n:return "carga_incendio"
    if {"pressurização","pressurizacao","escada"}&n:return "pressurizacao"
    return "geral"


def tipo(q):
    n=texto_normalizado(q)
    if "decreto" in n:return "decreto"
    if re.search(r"\bit\b",n): return "it"
    return "geral"


def score_nome(q,d):
    n=d["nome_norm"]; s=0; r=refs(q); a=assunto(q)
    for num,ano in r["it"]:
        if re.search(rf"\bit\s*0?{num}\s*[-/]\s*{ano}\b",n,re.I): s+=5000
        elif ano==2025 and re.search(rf"\bit\s*0?{num}\s*[-/]\s*25\b",n,re.I): s+=4900
    if a=="it11":
        if re.search(r"\bit\s*0?11\s*[-/]\s*(?:2025|25)\b",n,re.I): s+=4500
        if any(x in n for x in ["saidas de emergencia","saidas","emergencia"]): s+=1200
        if "conceitos basicos" in n:s-=300
        if "terminologia" in n:s-=300
    if a=="decreto" and "decreto" in n:s+=3000
    if a=="carga_incendio" and "carga de incendio" in n:s+=3000
    if a=="pressurizacao" and "pressurizacao" in n:s+=3000
    s+=12*len(set(normalizar(q))&set(normalizar(d["nome"])))
    return s


def selecionar(q,cat):
    a=sorted([{**d,"score_nome":score_nome(q,d)} for d in cat],key=lambda x:x["score_nome"],reverse=True); r=refs(q); a2=assunto(q)
    fortes=[d for d in a if d["score_nome"]>=4800]
    if r["it"] and fortes:return fortes[:MAX_DOCUMENTOS]
    if a2=="it11":
        it11=[d for d in a if re.search(r"\bit\s*0?11\s*[-/]\s*(?:2025|25)\b",d["nome_norm"],re.I)]
        if it11:return it11[:MAX_DOCUMENTOS]
    if a2=="decreto":
        fortes=[d for d in a if "decreto" in d["nome_norm"]]
        if fortes:return fortes[:MAX_DOCUMENTOS]
    return [d for d in a if d["score_nome"]>0][:MAX_DOCUMENTOS] or a[:MAX_DOCUMENTOS]


def blocos(texto,tamanho=3200):
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
    termos=set(normalizar(q)); out=[]; a=assunto(q); r=refs(q)
    # Termos de contexto evitam que uma palavra isolada faça o ROMUS escolher trecho irrelevante.
    contexto=set()
    if a=="it11": contexto.update(normalizar("saídas de emergência unidade de passagem largura população lotação portas dimensionamento"))
    elif a=="decreto": contexto.update(normalizar("institui regulamento segurança contra incêndios edificações áreas de risco"))
    elif a=="carga_incendio": contexto.update(normalizar("carga de incêndio densidade tabela classificação"))
    for d in docs:
        ds=score_nome(q,d)
        for no,b in enumerate(blocos(d["texto"])):
            bt=set(normalizar(b)); s=ds+5*len(termos&bt)+2*len(contexto&bt); nq=texto_normalizado(q); nb=texto_normalizado(b)
            if nq and nq in nb:s+=800
            if r["item"]:
                for it in r["item"]:
                    if re.search(rf"(?m)^\s*{re.escape(it)}\b",b):s+=900
            if a=="it11" and len(contexto&bt)>=3:s+=300
            if s>=3:out.append({"arquivo":d["arquivo"],"texto":b,"score":s,"bloco":no})
    out.sort(key=lambda x:x["score"],reverse=True); un=[];seen=set()
    for x in out:
        h=hashlib.sha1(re.sub(r"\s+"," ",x["texto"]).strip().lower().encode()).hexdigest()
        if h in seen:continue
        seen.add(h);un.append(x)
    return un[:MAX_BLOCOS]


def literal(q): return bool(set(normalizar(q))&{"transcreva","transcrever","literalmente","literal","exatamente","redacao"})

def calculo(q): return bool(set(normalizar(q))&{"calcule","calcular","calculo","dimensione","dimensionar","determine","quantas","quanto"})

def item_num(q):
    m=re.search(r"\b(?:item|subitem)\s*(\d+(?:\.\d+){1,4})\b",q.lower()); return m.group(1) if m else None


def item_literal(texto,num):
    p=re.escape(num); m=re.search(rf"(?ms)^\s*{p}\s+.*?(?=^\s*\d+(?:\.\d+){{1,4}}\s+|\Z)",texto); return m.group(0).strip() if m else None


def explicita(q,res):
    if not res:return None
    if tipo(q)=="decreto":
        for x in res:
            m=re.search(r"\b(?:Decreto\s+(?:Estadual\s+)?)?(?:n[ºo°]?\s*)?(\d{1,3}[.]\d{3}(?:[./-]\d{2,4})?)",x["texto"],re.I)
            if m:return m.group(1)
    if literal(q):
        n=item_num(q)
        if n:
            for x in res:
                t=item_literal(x["texto"],n)
                if t:return t
    termos=set(normalizar(q)); a=assunto(q)
    if r:=refs(q):
        if r["it"]: termos|={"largura","saidas","saida","emergencia","unidade","passagem","porta","minima","minimo"}
    cand=[];seen=set()
    for x in res:
        ls=[re.sub(r"\s+"," ",z).strip(" -\t") for z in re.split(r"\n+",x["texto"])]
        for i,l in enumerate(ls):
            if len(l)<12 or l.lower() in seen:continue
            seen.add(l.lower()); lt=set(normalizar(l)); c=len(termos&lt)
            if a=="it11": c+=min(4,len(contexto_linha:=set(normalizar("saídas emergência unidade passagem largura população lotação"))&lt))
            if c>=2 or (c>=1 and re.search(r"\d",l)): cand.append((x["score"]+c*12+(8 if re.search(r"\d",l) else 0),ls,i))
    if not cand:return None
    _,ls,i=max(cand,key=lambda z:z[0]); return " ".join(z for z in ls[max(0,i-1):min(len(ls),i+3)] if z)


def cliente():
    chave=st.secrets.get("GEMINI_API_KEY",os.getenv("GEMINI_API_KEY","")); return genai.Client(api_key=chave) if chave else None


def sintetizar(c,q,res):
    base="\n\n".join(f"[FONTE LOCAL: {x['arquivo']}]\n{x['texto']}" for x in res)
    p=f"Você é o ROMUS.IA. Responda em português. Use exclusivamente a base abaixo. Não invente e não use conhecimento externo. Para perguntas de cálculo, mostre somente as premissas encontradas na base e o cálculo decorrente delas. Se houver conflito entre trechos, informe o conflito. Se não houver elementos suficientes, responda somente {LIMITE_INSUFICIENTE}. Pergunta: {q}\n\nBASE:\n{base}"
    return c.models.generate_content(model=MODEL,contents=p,config=types.GenerateContentConfig(max_output_tokens=700))


def pesquisar_web(c,q):
    p=f"A base local foi consultada e não contém resposta suficiente. Pesquise somente fontes confiáveis e oficiais. Pergunta: {q}"
    return c.models.generate_content(model=MODEL,contents=p,config=types.GenerateContentConfig(max_output_tokens=800,tools=[types.Tool(google_search=types.GoogleSearch())]))


def fontes(res):
    if res:
        with st.expander("Documentos encontrados na base"):
            for x in dict.fromkeys(z["arquivo"] for z in res):st.write("• "+x)

st.title("ROMUS.IA")
st.caption("Inteligência artificial técnica e objetiva.")
q=st.text_area("Digite sua pergunta:",height=110)
web_ok=st.checkbox("Pesquisar na web se a base não responder",value=True)
c1,c2=st.columns(2)
with c1: perguntar=st.button("Perguntar",type="primary")
with c2: recarregar=st.button("Recarregar base")
if recarregar: st.cache_data.clear();st.rerun()

if perguntar and q.strip():
    q=q.strip()
    with st.spinner("Localizando o documento correto..."):
        cat=catalogo(assinatura_base()); cand=selecionar(q,cat)
    with st.spinner("Localizando o trecho correto..."):
        docs=[];hashes=set()
        for d in cand:
            x=ler(d["arquivo"],d["tamanho"],d["mtime"])
            if x and x["hash"] not in hashes:docs.append(x);hashes.add(x["hash"])
        res=buscar(q,docs)
    st.caption("Fonte: base local do ROMUS.IA")
    direta=explicita(q,res)
    if direta and not calculo(q):
        st.markdown("### ROMUS.IA");st.write(direta);fontes(res);st.stop()
    c=cliente()
    if res and c:
        try:t=(sintetizar(c,q,res).text or "").strip()
        except Exception as e:t=f"ERRO_GEMINI: {e}"
        if t and t!=LIMITE_INSUFICIENTE and not t.startswith("ERRO_GEMINI:"):
            st.markdown("### ROMUS.IA");st.write(t);fontes(res);st.stop()
        if t.startswith("ERRO_GEMINI:"):
            st.markdown("### ROMUS.IA");st.warning("Gemini indisponível. A base local continua funcionando.");st.write(direta or res[0]["texto"]);fontes(res);st.stop()
        if t==LIMITE_INSUFICIENTE:res=[]
    if not res and web_ok and c:
        try:t=(pesquisar_web(c,q).text or "").strip()
        except Exception as e:t=f"ERRO_WEB: {e}"
        if t and not t.startswith("ERRO_WEB:"):
            st.caption("Fonte: pesquisa na internet (base local insuficiente)");st.markdown("### ROMUS.IA");st.write(t);st.stop()
    st.markdown("### ROMUS.IA")
    st.warning("A base local não contém informação suficiente para responder com segurança." if not res else "A base contém informação relacionada, mas não foi possível sintetizar com segurança.")
    fontes(res)
