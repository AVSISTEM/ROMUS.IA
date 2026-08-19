import os
import re
import time
import streamlit as st
from google import genai
from google.genai import types
import pdfplumber

# =========================================================
# 1. CONFIGURAÇÃO DA PÁGINA E INTERFACE
# =========================================================
st.set_page_config(
    page_title="ROMANO - IA Técnica",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
[data-testid="stHeader"], footer { visibility: hidden; height: 0px; }
[data-testid="chatAvatarIcon-user"], [data-testid="chatAvatarIcon-assistant"],
div[data-testid="stChatMessage"] > div:first-child { display: none !important; }
html, body, .stApp, [data-testid="stAppViewContainer"], [data-testid="stBottom"] {
    background-color: #0e1117 !important; color: #f0f6fc !important;
}
.stChatInputContainer, div[data-testid="stChatInput"] {
    background-color: #161b22 !important; border: 1px solid #30363d !important; border-radius: 10px !important;
}
.main .block-container { max-width: 1100px; padding-top: 1rem; padding-bottom: 2rem; }
.romano-wrap { text-align: center; margin-top: 0.5rem; margin-bottom: 1.5rem; }
.romano-title { font-size: 42px; font-weight: 900; letter-spacing: 2px; color: #ffffff; margin-bottom: 0.1rem; }
.romano-subtitle { font-size: 18px; font-weight: 600; color: #8b949e; margin-bottom: 0.5rem; }
.romano-slogan { font-size: 13px; color: #6e7681; text-transform: uppercase; letter-spacing: 1px; }
.debug-box { border: 1px solid #30363d; border-radius: 8px; padding: 12px; background: #161b22; font-size: 13px; font-family: monospace; white-space: pre-wrap; color: #c9d1d9; }
</style>
""", unsafe_allow_html=True)

# =========================================================
# 2. CONFIGURAÇÕES TÉCNICAS
# =========================================================
BASE_CONHECIMENTO_DIR = "base_conhecimento"
MODELO_UNICO = "gemini-3.6-flash"
TAMANHO_CHUNK = 1500
SOBREPOSICAO_CHUNK = 250
TOP_CHUNKS = 10

PROMPT_SISTEMA = """
Você é o ROMANO, uma inteligência artificial autônoma, técnica e objetiva especializada em Engenharia de Segurança contra Incêndio e Legislação.

DIRETRIZES RÍGIDAS:
- Responda EXCLUSIVAMENTE com base nos documentos locais fornecidos no contexto.
- Se a pergunta solicitar exigências ou medidas de segurança contra incêndio para uma edificação, identifique a ocupação (ex: F-11), a área total e a lotação, cruzando essas informações com as tabelas de exigências presentes nos documentos.
- NUNCA invente decretos ou normas antigas. Cite nominalmente o arquivo do qual extraiu a informação.
- Slogan: "ROMANO não passa pano. ROMANO responde com base. ROMANO não inventa. ROMANO resolve."

ESTRUTURA OBRIGATÓRIA DA RESPOSTA:

RESPOSTA DIRETA:
[Listagem direta e objetiva das medidas de segurança exigidas ou da classificação solicitada]

FUNDAMENTAÇÃO TÉCNICA:
[Nome do arquivo consultado, Tabela, Artigo ou Item específico da norma presente na base local]

GRAU DE CERTEZA TÉCNICA:
[Expressa na Base Local / Obtida via Busca Web]

OBSERVAÇÃO OPERACIONAL:
[Anotações técnicas de aplicação prática, se houver]

BORDÃO OPERACIONAL
ROMANO não passa pano. ROMANO responde com base. ROMANO não inventa. ROMANO resolve.
""".strip()

# =========================================================
# 3. EXTRATOR DE PDF ROBUSTO (PRESERVA TABELAS)
# =========================================================
@st.cache_resource
def criar_cliente():
    api_key = os.getenv("GEMINI_API_KEY") or st.secrets.get("GEMINI_API_KEY", "")
    return genai.Client(api_key=api_key) if api_key else None

def extrair_texto_pdf_estruturado(caminho_pdf: str) -> str:
    partes = []
    try:
        with pdfplumber.open(caminho_pdf) as pdf:
            for pagina in pdf.pages:
                txt = pagina.extract_text(layout=True)
                if txt:
                    partes.append(txt)
    except Exception:
        return ""
    return "\n".join(partes).strip()

@st.cache_data(show_spinner=False)
def carregar_e_indexar_base():
    indice = []
    if not os.path.exists(BASE_CONHECIMENTO_DIR):
        return indice

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
                texto = extrair_texto_pdf_estruturado(caminho)

            if texto:
                # Divisão em Chunks
                inicio = 0
                chunk_id = 1
                while inicio < len(texto):
                    fim = min(len(texto), inicio + TAMANHO_CHUNK)
                    chunk = texto[inicio:fim].strip()
                    if chunk:
                        indice.append({"arquivo": nome_relativo, "chunk_id": chunk_id, "texto": chunk})
                        chunk_id += 1
                    if fim >= len(texto):
                        break
                    inicio += (TAMANHO_CHUNK - SOBREPOSICAO_CHUNK)
    return indice

# =========================================================
# 4. MOTOR DE BUSCA COM RELEVÂNCIA NORMATIVA
# =========================================================
def buscar_contexto(pergunta: str):
    indice = carregar_e_indexar_base()
    if not indice:
        return [], ""

    termo_busca = pergunta.lower()
    tokens = re.findall(r'\b\w+\b', termo_busca)
    
    resultados = []
    for item in indice:
        chunk_lower = item["texto"].lower()
        arq_lower = item["arquivo"].lower()
        score = 0

        # Pontuação por termos
        for t in tokens:
            if len(t) > 2:
                score += chunk_lower.count(t) * 3

        # Impulso para Tabelas e Regulamento quando houver dados numéricos de área/lotação
        if any(k in termo_busca for k in ["m2", "m²", "lotação", "lotacao", "pessoas", "exigências", "exigencias", "medidas"]):
            if "tabela" in chunk_lower or "regulamento" in arq_lower or "decreto" in arq_lower:
                score += 50

        # Busca exata do grupo/divisão (ex: F-11, F-1) sem confusão de substrings
        match_grupo = re.search(r'\b[a-m]-\d+\b', termo_busca)
        if match_grupo:
            grupo_procurado = match_grupo.group(0)
            if grupo_procurado in chunk_lower:
                score += 100

        if score > 0:
            resultados.append({"arquivo": item["arquivo"], "texto": item["texto"], "score": score})

    resultados.sort(key=lambda x: x["score"], reverse=True)
    top_results = resultados[:TOP_CHUNKS]

    contexto_str = "\n\n".join([f"--- FONTE: {r['arquivo']} ---\n{r['texto']}" for r in top_results])
    return top_results, contexto_str

# =========================================================
# 5. PROCESSAMENTO DE PERGUNTAS
# =========================================================
def processar_ordem(pergunta: str, historico: list):
    inicio = time.time()
    trechos, contexto = buscar_contexto(pergunta)
    cliente = criar_cliente()

    if not cliente:
        return {"ok": False, "erro": "Chave GEMINI_API_KEY não encontrada."}

    contents = []
    for msg in historico[:-1]:
        role_api = "user" if msg["role"] == "user" else "model"
        contents.append(types.Content(role=role_api, parts=[types.Part.from_text(text=msg["content"])]))

    prompt_envio = f"ORDEM DO COMANDANTE:\n{pergunta}\n\nDOCUMENTOS CONSULTADOS NA BASE LOCAL:\n{contexto if contexto else 'Nenhum documento retornado.'}"
    contents.append(types.Content(role="user", parts=[types.Part.from_text(text=prompt_envio)]))

    try:
        resposta = cliente.models.generate_content(
            model=MODELO_UNICO,
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=PROMPT_SISTEMA,
                temperature=0.0
            )
        )
        tempo = round(time.time() - inicio, 2)
        return {
            "ok": True,
            "texto": resposta.text.strip(),
            "tempo": tempo,
            "trechos": trechos,
            "modelo": MODELO_UNICO
        }
    except Exception as e:
        return {"ok": False, "erro": str(e)}

# =========================================================
# 6. INTERFACE STREAMLIT
# =========================================================
st.markdown("""
<div class="romano-wrap">
    <div class="romano-title">ROMANO</div>
    <div class="romano-subtitle">A IA QUE NÃO PASSA PANO</div>
    <div class="romano-slogan">Inteligência Técnica • Autonomia Local • Respostas Diretas</div>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.subheader("Painel de Controle")
    mostrar_debug = st.checkbox("Exibir Diagnóstico Técnico", value=True)
    if st.button("Limpar Histórico", use_container_width=True):
        st.session_state.mensagens = []
        st.rerun()

if "mensagens" not in st.session_state:
    st.session_state.mensagens = []

for msg in st.session_state.mensagens:
    with st.chat_message(msg["role"], avatar=None):
        st.markdown(msg["content"])
        if "debug" in msg and mostrar_debug and msg["debug"]:
            with st.expander("Diagnóstico Operacional", expanded=False):
                st.markdown(f'<div class="debug-box">{msg["debug"]}</div>', unsafe_allow_html=True)

pergunta = st.chat_input("Digite sua ordem ou consulta técnica...")

if pergunta:
    st.session_state.mensagens.append({"role": "user", "content": pergunta})
    with st.chat_message("user", avatar=None):
        st.markdown(pergunta)

    with st.chat_message("assistant", avatar=None):
        with st.spinner("ROMANO consultando base local..."):
            res = processar_ordem(pergunta, st.session_state.mensagens)

        if not res["ok"]:
            st.error("Falha na execução.")
            st.code(res["erro"])
        else:
            st.markdown(res["texto"])
            debug_info = (
                f"Trechos Recuperados: {len(res['trechos'])}\n"
                f"Motor: {res['modelo']}\n"
                f"Tempo: {res['tempo']} s"
            )
            if mostrar_debug:
                with st.expander("Diagnóstico Operacional", expanded=False):
                    st.markdown(f'<div class="debug-box">{debug_info}</div>', unsafe_allow_html=True)

            st.session_state.mensagens.append({
                "role": "assistant",
                "content": res["texto"],
                "debug": debug_info
            })
