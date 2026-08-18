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
BASE_DIR=Path(__file__).parent
KNOWLEDGE_DIR=BASE_DIR/"base_conhecimento"
MODEL="gemini-3.5-flash"
ABSTAIN="A base local não contém informação suficiente para responder com segurança."
PROMPT="""Você é o ROMUS.IA, assistente técnico de segurança contra incêndio.
Use EXCLUSIVAMENTE a EVIDÊNCIA LOCAL abaixo.

REGRAS:
1. Responda somente em português e somente com a resposta final.
2. Não mostre raciocínio, rascunho, dúvidas ou comentários internos.
3. Não invente números, unidades, artigos, itens, tabelas ou classificações.
4. Se a pergunta indicar uma classificação, como F-11, use somente essa classificação.
5. Não use outra classificação para completar ou substituir a solicitada.
6. Faça cálculos somente com valores expressamente presentes na evidência.
7. Preserve números, unidades e referências normativas conforme aparecem na evidência.
8. Se a evidência não comprovar a resposta, responda exatamente: A base local não contém informação suficiente para responder com segurança.
9. Responda de forma curta e objetiva.

PERGUNTA:
{question}

EVIDÊNCIA LOCAL:
{context}

RESPOSTA FINAL:
"""
WEB_PROMPT="""A base local não forneceu evidência suficiente. Pesquise somente fontes oficiais ou técnicas confiáveis. Não invente informação. Responda diretamente à pergunta e informe a fonte encontrada."""
STOP={"para","uma","com","qual","quais","quanto","quantas","deve","ser","das","dos","pessoas","edificacao","edificacoes","populacao","numero","minima","minimo","necessarias","necessarios","conforme","sobre","segundo","como","que","um","o","a","e","de","da","do","na","no","grupo"}


def normalizar(texto):
    texto=unicodedata.normalize("NFKD",str(texto).lower())
    texto="".join(c for c in texto if not unicodedata.combining(c))
    return [x for x in re.sub(r"[^\w\s]"," ",texto).split() if len(x)>1]


def norm(texto): return " ".join(normalizar(texto))


def refs(question):
    q=norm(question); p=r"\bit\s*(?:(?:n|no)\s*)?0?(\d{1,2})\s*[-/ ]\s*(20\d{2}|\d{2})\b"
    return [(int(n),int(a) if len(a)==4 else 2000+int(a)) for n,a in re.findall(p,q)]


def _codigos(texto):
    xs=re.findall(r"\b[A-Z]{1,3}\s*[-–—]?\s*\d{1,3}\b",str(texto).upper())
    return {x for x in xs if not x.strip().startswith("IT ")}


def grupos(question): return _codigos(question)

def grupo_normalizado(codigo): return re.sub(r"[^A-Z0-9]","",codigo.upper())

def grupos_na_pagina(texto): return {grupo_normalizado(x) for x in _codigos(texto)}


def eh_it(nome,numero,ano=2025):
    return bool(re.search(rf"\bit\s*(?:(?:n|no)\s*)?0?{numero}\s*(?:-|/|\s)*({ano}|25)\b",norm(nome),re.I))


@st.cache_data(show_spinner=False)
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
        if p.suffix.lower()==".pdf":
            paginas=[]
            for n,page in enumerate(PdfReader(str(p)).pages,1):
                t=page.extract_text() or ""; t=t.replace("\u00a0"," "); t=re.sub(r"[ \t]+"," ",t); t=re.sub(r"\n{3,}","\n\n",t).strip()
                if t: paginas.append((n,t))
        else:
            t=p.read_text(encoding="utf-8",errors="ignore").strip(); paginas=[(1,t)] if t else []
    except Exception: return None
    if not paginas: return None
    total="\n\n".join(t for _,t in paginas)
    return {"arquivo":arquivo,"nome":p.name,"paginas":paginas,"hash":hashlib.sha256(total.encode()).hexdigest()}


def selecionar(question,cat):
    for numero,ano in refs(question):
        docs=[d for d in cat if eh_it(d["nome"],numero,ano)]
        if docs:return docs[:1]
    qn=set(normalizar(question))
    if {"unidade","passagem"}<=qn or bool({"saida","saidas","emergencia"}&qn):
        docs=[d for d in cat if eh_it(d["nome"],11,2025)]
        if docs:return docs[:1]
    if "decreto" in qn or "regulamento" in qn:
        docs=[d for d in cat if "decreto" in d["norm"] and ("69 118" in d["norm"] or "regulamento" in d["norm"])]
        if docs:return docs[:1]
    ranked=sorted(cat,key=lambda d:len(qn&set(d["norm"].split())),reverse=True)
    return [d for d in ranked[:3] if len(qn&set(d["norm"].split()))>=2]


def pagina_score(question,texto,nome):
    q=set(normalizar(question))-STOP; b=set(normalizar(texto)); bn=norm(texto); score=5*len(q&b)
    qg={grupo_normalizado(x) for x in grupos(question)}
    if qg: score+=1000 if qg&grupos_na_pagina(texto) else -1000
    for phrase,weight in [("unidade de passagem",60),("unidades de passagem",60),("largura das saidas",60),("largura minima",50),("dimensionamento das saidas",50),("capacidade da unidade de passagem",45),("calculo da largura",55),("largura da saida",55)]:
        if phrase in bn:score+=weight
    for numero,ano in refs(question):
        if eh_it(nome,numero,ano):score+=100
    return score


def trecho_relevante(texto,question):
    qg={grupo_normalizado(x) for x in grupos(question)}
    if not qg:return texto
    linhas=str(texto).splitlines(); achados=[]
    for i,linha in enumerate(linhas):
        if qg&grupos_na_pagina(linha):achados.extend(linhas[max(0,i-3):min(len(linhas),i+5)])
    if not achados:return texto
    vistos=set();out=[]
    for linha in achados:
        x=linha.strip()
        if x and x not in vistos:vistos.add(x);out.append(x)
    return "\n".join(out)


def recuperar(question,docs):
    qg={grupo_normalizado(x) for x in grupos(question)}; candidatos=[]
    for d in docs:
        for pagina,texto in d["paginas"]:
            if qg and not(qg&grupos_na_pagina(texto)):continue
            score=pagina_score(question,texto,d["nome"])
            if score>0:candidatos.append({"arquivo":d["arquivo"],"pagina":pagina,"texto":texto,"score":score})
    candidatos.sort(key=lambda x:x["score"],reverse=True); escolhidos=[]; vistos=set(); limite=3 if qg else 6
    for x in candidatos:
        k=(x["arquivo"],x["pagina"])
        if k in vistos:continue
        vistos.add(k); escolhidos.append({**x,"texto":trecho_relevante(x["texto"],question)})
        if len(escolhidos)>=limite:break
    return escolhidos


def cliente():
    key=st.secrets.get("GEMINI_API_KEY",os.getenv("GEMINI_API_KEY",""))
    return genai.Client(api_key=key) if key else None

def contexto(passages):
    return "\n\n".join(f"[DOCUMENTO {i}] {p['arquivo']} | PÁGINA {p['pagina']}\n{p['texto']}" for i,p in enumerate(passages,1))

def gerar(c,question,passages):
    return c.models.generate_content(model=MODEL,contents=PROMPT.format(question=question,context=contexto(passages)),config=types.GenerateContentConfig(temperature=0,max_output_tokens=700))

def resposta_valida(texto,question=""):
    texto=(texto or "").strip()
    if not texto or ABSTAIN in texto:return None
    baixo=texto.lower()
    if any(x in baixo for x in ("wait,","let's look","i need to","let me","reasoning:","analyzing","vamos analisar","vou analisar","preciso verificar","raciocinio:","raciocínio:")):return None
    qg={grupo_normalizado(x) for x in grupos(question)}
    if qg and (grupos_na_pagina(texto)-qg):return None
    return texto

def pesquisar_web(c,question):
    return c.models.generate_content(model=MODEL,contents=WEB_PROMPT+"\n\nPERGUNTA:\n"+question,config=types.GenerateContentConfig(max_output_tokens=800,tools=[types.Tool(google_search=types.GoogleSearch())]))

def fontes(passages):
    if passages:
        with st.expander("Documentos encontrados na base"):
            for p in passages:st.write(f"• {p['arquivo']} — página {p['pagina']}")

st.title("ROMUS.IA")
st.caption("Inteligência artificial técnica e objetiva.")
question=st.text_area("Digite sua pergunta:",height=110)
web_ok=st.checkbox("Pesquisar na web se a base não responder",value=True)
c1,c2=st.columns(2)
with c1:perguntar=st.button("Perguntar",type="primary")
with c2:recarregar=st.button("Recarregar base")
if recarregar:st.cache_data.clear();st.rerun()

if perguntar and question.strip():
    question=question.strip()
    with st.spinner("Localizando o documento correto..."):
        cat=catalogo(assinatura_base());docs=[];hashes=set()
        for d in selecionar(question,cat):
            x=ler(d["arquivo"],d["tamanho"],d["mtime"])
            if x and x["hash"] not in hashes:docs.append(x);hashes.add(x["hash"])
    with st.spinner("Localizando as páginas relevantes..."):passages=recuperar(question,docs)
    st.caption("Fonte: base local do ROMUS.IA");c=cliente();resposta=None
    if passages and c:
        with st.spinner("Consultando a evidência local..."):
            try:resposta=resposta_valida(gerar(c,question,passages).text,question)
            except Exception:resposta=None
    if resposta:
        st.markdown("### ROMUS.IA");st.write(resposta);fontes(passages);st.stop()
    if web_ok and c and not passages:
        try:web_text=(pesquisar_web(c,question).text or "").strip()
        except Exception:web_text=""
        if web_text:
            st.caption("Fonte: pesquisa na internet — base local insuficiente");st.markdown("### ROMUS.IA");st.write(web_text);st.stop()
    st.markdown("### ROMUS.IA");st.warning(ABSTAIN);fontes(passages)
