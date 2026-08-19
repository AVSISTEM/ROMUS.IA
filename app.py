import os
import re
import streamlit as st

# Tenta usar pdfplumber para preservar formatação de tabelas; fallback para pypdf
try:
    import pdfplumber
    HAS_PDFPLUMBER = True
except ImportError:
    from pypdf import PdfReader
    HAS_PDFPLUMBER = False

# =========================================================
# 1. CONFIGURAÇÃO DA PÁGINA E INTERFACE
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
.stChatInputContainer, div[data-testid="stChatInput"] { background-color: #161b22 !important; border: 1px solid #30363d !important; border-radius: 10px !important; }
.main .block-container { max-width: 1100px; padding-top: 1rem; padding-bottom: 2rem; }
.romano-wrap { text-align: center; margin-top: 0.5rem; margin-bottom: 1.5rem; }
.romano-title { font-size: 42px; font-weight: 900; letter-spacing: 2px; color: #ffffff; margin-bottom: 0.1rem; }
.romano-subtitle { font-size: 18px; font-weight: 600; color: #8b949e; margin-bottom: 0.5rem; }
.romano-slogan { font-size: 13px; color: #6e7681; text-transform: uppercase; letter-spacing: 1px; }
.resultado-card { background-color: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 15px; margin-bottom: 15px; }
.fonte-header { font-weight: bold; color: #58a6ff; margin-bottom: 8px; font-size: 14px; }
.trecho-texto { font-family: monospace; font-size: 13px; white-space: pre-wrap; color: #c9d1d9; background-color: #0d1117; padding: 10px; border-radius: 5px; }
</style>
""", unsafe_allow_html=True)

BASE_CONHECIMENTO_DIR = "base_conhecimento"

# =========================================================
# 2. EXTRATOR DE TEXTO E INDEXADOR LOCAL
# =========================================================
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
    if not os.path.exists(BASE_CONHECIMENTO_DIR):
        return base_dados

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
                # Quebra em blocos por parágrafo ou tabela (quebras duplas)
                blocos = [b.strip() for b in texto.split("\n\n") if len(b.strip()) > 30]
                for idx, bloco in enumerate(blocos):
                    base_dados.append({
                        "arquivo": nome_relativo,
                        "bloco_id": idx + 1,
                        "texto": bloco
                    })
    return base_dados

# =========================================================
# 3. MOTOR DE BUSCA ALGORÍTMICO (SEM IA)
# =========================================================
def buscar_na_base(termo: str):
    base = carregar_base_conhecimento()
    if not base:
        return []

    termo_limpo = termo.lower().strip()
    tokens = re.findall(r'\b\w+\b', termo_limpo)
    
    # Identifica ocupação específica tipo F-11, A-1, C-2
    match_grupo = re.search(r'\b[a-m]-\d+\b', termo_limpo)
    grupo_procurado = match_grupo.group(0) if match_grupo else None

    resultados = []
    for item in base:
        texto_lower = item["texto"].lower()
        score = 0

        # Regra 1: Correspondência exata da ocupação (Ex: F-11)
        if grupo_procurado and grupo_procurado in texto_lower:
            score += 200

        # Regra 2: Contagem de palavras digitadas
        for t in tokens:
            if len(t) > 2 and t in texto_lower:
                score += texto_lower.count(t) * 10

        # Regra 3: Bônus se contiver termos de tabela/norma
        if any(k in termo_limpo for k in ["medidas", "exigencias", "exigências", "m2", "m²", "lotação"]):
            if any(tabelak in texto_lower for tabelak in ["tabela", "medida", "exigência", "existente"]):
                score += 30

        if score > 0:
            resultados.append({
                "arquivo": item["arquivo"],
                "texto": item["texto"],
                "score": score
            })

    # Ordena pelos trechos de maior relevância
    resultados.sort(key=lambda x: x["score"], reverse=True)
    return resultados[:15]

# =========================================================
# 4. INTERFACE GRÁFICA
# =========================================================
st.markdown("""
<div class="romano-wrap">
    <div class="romano-title">ROMANO</div>
    <div class="romano-subtitle">BUSCADOR NORMATIVO ALGORÍTMICO</div>
    <div class="romano-slogan">0% API • 100% Local • Resposta Instantânea</div>
</div>
""", unsafe_allow_html=True)

query = st.chat_input("Digite a ocupação, norma ou parâmetro (ex: F-11, 250m2, tabela)...")

if query:
    st.markdown(f"**Consulta:** `{query}`")
    resultados = buscar_na_base(query)

    if not resultados:
        st.warning("Nenhum trecho correspondente foi localizado na pasta 'base_conhecimento'.")
    else:
        st.success(f"{len(resultados)} trechos relevantes localizados na base local.")
        for r in resultados:
            st.markdown(f"""
            <div class="resultado-card">
                <div class="fonte-header">📄 Arquivo: {r['arquivo']} (Relevância: {r['score']})</div>
                <div class="trecho-texto">{r['texto']}</div>
            </div>
            """, unsafe_allow_html=True)
