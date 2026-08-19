import os
import re
import unicodedata
import streamlit as st

try:
    import pdfplumber
    HAS_PDFPLUMBER = True
except ImportError:
    from pypdf import PdfReader
    HAS_PDFPLUMBER = False

# =========================================================
# 1. CONFIGURAÇÃO DA PÁGINA
# =========================================================
st.set_page_config(
    page_title="ROMANO - Buscador Normativo",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
[data-testid="stHeader"], footer { visibility: hidden; height: 0px; }
html, body, .stApp { background-color: #0e1117 !important; color: #f0f6fc !important; }
.main .block-container { max-width: 1100px; padding-top: 1rem; padding-bottom: 2rem; }
.romano-wrap { text-align: center; margin-top: 0.5rem; margin-bottom: 1.5rem; }
.romano-title { font-size: 38px; font-weight: 900; letter-spacing: 2px; color: #ffffff; margin-bottom: 0.1rem; }
.romano-subtitle { font-size: 16px; font-weight: 600; color: #8b949e; margin-bottom: 0.3rem; }
.romano-slogan { font-size: 12px; color: #6e7681; text-transform: uppercase; letter-spacing: 1px; }
.resultado-card { background-color: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 15px; margin-bottom: 15px; }
.fonte-header { font-weight: bold; color: #58a6ff; margin-bottom: 8px; font-size: 14px; }
.trecho-texto { font-family: monospace; font-size: 13px; white-space: pre-wrap; color: #c9d1d9; background-color: #0d1117; padding: 10px; border-radius: 5px; }
</style>
""", unsafe_allow_html=True)

BASE_CONHECIMENTO_DIR = "base_conhecimento"

# =========================================================
# 2. FUNÇÕES DE TRATAMENTO DE TEXTO
# =========================================================
def normalizar(texto: str) -> str:
    if not texto:
        return ""
    nfkd = unicodedata.normalize('NFD', texto)
    texto_sem_acento = "".join([c for c in nfkd if unicodedata.category(c) != 'Mn'])
    return re.sub(r'[^a-zA-Z0-9\s-]', ' ', texto_sem_acento).lower().strip()

def extrair_texto_pdf(caminho_pdf: str) -> str:
    partes = []
    try:
        if HAS_PDFPLUMBER:
            with pdfplumber.open(caminho_pdf) as pdf:
                for pagina in pdf.pages:
                    txt = pagina.extract_text(layout=True)
                    if txt:
                        partes.append(txt)
        else:
            reader = PdfReader(caminho_pdf)
            for pagina in reader.pages:
                txt = pagina.extract_text()
                if txt:
                    partes.append(txt)
    except Exception:
        return ""
    return "\n".join(partes).strip()

@st.cache_data(show_spinner=False)
def carregar_base_conhecimento():
    base_dados = []
    arquivos_lidos = 0
    
    if not os.path.exists(BASE_CONHECIMENTO_DIR):
        return base_dados, 0

    for raiz, _, arquivos in os.walk(BASE_CONHECIMENTO_DIR):
        for arquivo in arquivos:
            if not arquivo.lower().endswith((".pdf", ".txt")):
                continue
            caminho = os.path.join(raiz, arquivo)
            nome_relativo = os.path.relpath(caminho, BASE_CONHECIMENTO_DIR)

            if arquivo.lower().endswith(".txt"):
                with open(caminho, "r", encoding="utf-8", errors="ignore") as f:
                    texto = f.read()
            else:
                texto = extrair_texto_pdf(caminho)

            if texto:
                arquivos_lidos += 1
                linhas = texto.split("\n")
                bloco_atual = []
                tamanho_atual = 0

                for linha in linhas:
                    bloco_atual.append(linha)
                    tamanho_atual += len(linha)
                    if tamanho_atual >= 800:
                        conteudo_bloco = "\n".join(bloco_atual).strip()
                        if conteudo_bloco:
                            base_dados.append({
                                "arquivo": nome_relativo,
                                "texto": conteudo_bloco,
                                "texto_norm": normalizar(conteudo_bloco)
                            })
                        bloco_atual = []
                        tamanho_atual = 0

                if bloco_atual:
                    conteudo_bloco = "\n".join(bloco_atual).strip()
                    if conteudo_bloco:
                        base_dados.append({
                            "arquivo": nome_relativo,
                            "texto": conteudo_bloco,
                            "texto_norm": normalizar(conteudo_bloco)
                        })

    return base_dados, arquivos_lidos

# =========================================================
# 3. MOTOR DE BUSCA
# =========================================================
def buscar_na_base(termo: str):
    base, _ = carregar_base_conhecimento()
    if not base:
        return []

    termo_norm = normalizar(termo)
    tokens = [t for t in termo_norm.split() if len(t) > 2]
    
    resultados = []
    for item in base:
        score = 0
        texto_norm = item["texto_norm"]

        for token in tokens:
            if token in texto_norm:
                score += texto_norm.count(token) * 10

        if termo_norm in texto_norm:
            score += 100

        if score > 0:
            resultados.append({
                "arquivo": item["arquivo"],
                "texto": item["texto"],
                "score": score
            })

    resultados.sort(key=lambda x: x["score"], reverse=True)
    return resultados[:15]

# =========================================================
# 4. INTERFACE PRINCIPAL
# =========================================================
st.markdown("""
<div class="romano-wrap">
    <div class="romano-title">ROMANO</div>
    <div class="romano-subtitle">BUSCADOR NORMATIVO ALGORÍTMICO</div>
    <div class="romano-slogan">0% API • 100% Local • Resposta Instantânea</div>
</div>
""", unsafe_allow_html=True)

base_indexada, total_pdfs = carregar_base_conhecimento()

with st.sidebar:
    st.subheader("Painel de Diagnóstico")
    st.info(f"Arquivos Lidos: **{total_pdfs}**")
    st.info(f"Blocos Mapeados: **{len(base_indexada)}**")
    if st.button("Recarregar Base", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

# CAMPO DE BUSCA FIXO NO TOPO
query = st.text_input(
    label="Pesquisa Normativa",
    placeholder="Digite os termos da busca (ex: largura escada, F-11, extintores)...",
    label_visibility="collapsed"
)

if query:
    resultados = buscar_na_base(query)

    if not resultados:
        st.error(f"Nenhum trecho localizado para '{query}'. Verifique os arquivos na pasta 'base_conhecimento'.")
    else:
        st.success(f"{len(resultados)} trechos relevantes encontrados.")
        for r in resultados:
            st.markdown(f"""
            <div class="resultado-card">
                <div class="fonte-header">📄 Arquivo: {r['arquivo']} (Pontuação: {r['score']})</div>
                <div class="trecho-texto">{r['texto']}</div>
            </div>
            """, unsafe_allow_html=True)
