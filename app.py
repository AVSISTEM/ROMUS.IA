import os
import re
import time
import streamlit as st
from google import genai
from google.genai import types
from pypdf import PdfReader

# =========================================================
# 1. CONFIGURAÇÃO DA PÁGINA E CSS TEMA ESCURO (iOS FIX)
# =========================================================
st.set_page_config(
    page_title="ROMANO - IA Técnica",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
[data-testid="stHeader"], footer {
    visibility: hidden;
    height: 0px;
}

[data-testid="chatAvatarIcon-user"], 
[data-testid="chatAvatarIcon-assistant"],
div[data-testid="stChatMessage"] > div:first-child {
    display: none !important;
}

html, body, .stApp, [data-testid="stAppViewContainer"], [data-testid="stBottom"] {
    background-color: #0e1117 !important;
    color: #f0f6fc !important;
}

.stChatInputContainer, div[data-testid="stChatInput"] {
    background-color: #161b22 !important;
    border: 1px solid #30363d !important;
    border-radius: 10px !important;
}

.main .block-container {
    max-width: 1100px;
    padding-top: 1rem;
    padding-bottom: 2rem;
}

.romano-wrap {
    text-align: center;
    margin-top: 0.5rem;
    margin-bottom: 1.5rem;
}

.romano-title {
    font-size: 42px;
    font-weight: 900;
    letter-spacing: 2px;
    margin-bottom: 0.1rem;
    color: #ffffff;
}

.romano-subtitle {
    font-size: 18px;
    font-weight: 600;
    color: #8b949e;
    margin-bottom: 0.5rem;
}

.romano-slogan {
    font-size: 13px;
    color: #6e7681;
    text-transform: uppercase;
    letter-spacing: 1px;
}

.debug-box {
    border: 1px solid #30363d;
    border-radius: 8px;
    padding: 12px;
    background: #161b22;
    font-size: 13px;
    font-family: monospace;
    white-space: pre-wrap;
    color: #c9d1d9;
}
</style>
""", unsafe_allow_html=True)

# =========================================================
# 2. CONFIGURAÇÕES OTIMIZADAS PARA ECONOMIA DE COTAS
# =========================================================
BASE_CONHECIMENTO_DIR = "base_conhecimento"
ARQUIVOS_SUPORTADOS = (".txt", ".pdf")

# Modelos para rodízio e fallback
MODELOS = ["gemini-3.6-flash", "gemini-2.5-flash", "gemini-1.5-flash"]

# Otimização de tamanho para evitar erro de limite de Tokens/Minuto
TAMANHO_CHUNK = 1200
SOBREPOSICAO_CHUNK = 150
TOP_CHUNKS = 6

PALAVRAS_IGNORADAS = {
    "a", "o", "e", "de", "da", "do", "das", "dos", "um", "uma",
    "em", "por", "para", "com", "sem", "que", "como", "qual",
    "quais", "onde", "quando", "isso", "essa", "esse", "sobre",
    "as", "os", "ao", "aos", "na", "no", "nas", "nos",
    "bom", "boa", "dia", "tarde", "noite", "oi", "ola", "olá"
}

# =========================================================
# 3. PROMPT OTIMIZADO (SISTEMA ROMANO)
# =========================================================
PROMPT_SISTEMA = """
Você é o ROMANO, uma inteligência artificial autônoma, técnica e objetiva.

DIRETRIZES
- Slogan: "A IA que não passa pano. ROMANO responde com base. ROMANO não inventa. ROMANO resolve."
- Responda diretamente ao que foi solicitado, de forma concisa e sem saudações banais.

ESTRUTURA DE RESPOSTA OBRIGATÓRIA

RESPOSTA DIRETA:
[Resposta objetiva]

FUNDAMENTAÇÃO TÉCNICA:
[Norma, artigo, item, tabela ou fonte consultada]

GRAU DE CERTEZA TÉCNICA:
[Expressa na Base Local / Obtida via Busca Web]

OBSERVAÇÃO OPERACIONAL:
[Se necessário]

BORDÃO OPERACIONAL
ROMANO não passa pano. ROMANO responde com base. ROMANO não inventa. ROMANO resolve.
""".strip()

# =========================================================
# 4. INICIALIZAÇÃO DO CLIENTE GEMINI
# =========================================================
@st.cache_resource
def criar_cliente():
    api_key = os.getenv("GEMINI_API_KEY") or st.secrets.get("GEMINI_API_KEY", "")
    if not api_key:
        raise ValueError("A chave GEMINI_API_KEY não foi configurada.")
    return genai.Client(api_key=api_key)

# =========================================================
# 5. BASE LOCAL E BUSCA DE TRECHOS
# =========================================================
def extrair_texto_txt(caminho_txt: str) -> str:
    try:
        with open(caminho_txt, "r", encoding="utf-8", errors="ignore") as f:
            return f.read().strip()
    except Exception:
        return ""

def extrair_texto_pdf(caminho_pdf: str) -> str:
    partes = []
    try:
        reader = PdfReader(caminho_pdf)
        for pagina in reader.pages:
            try:
                txt = pagina.extract_text()
                if txt:
                    partes.append(txt)
            except Exception:
                continue
    except Exception:
        return ""
    return "\n".join(partes).strip()

@st.cache_data(show_spinner=False)
def carregar_base_local():
    base = []
    if not os.path.exists(BASE_CONHECIMENTO_DIR):
        return base

    for raiz, _, arquivos in os.walk(BASE_CONHECIMENTO_DIR):
        for arquivo in arquivos:
            if not arquivo.lower().endswith(ARQUIVOS_SUPORTADOS):
                continue
            caminho = os.path.join(raiz, arquivo)
            nome_relativo = os.path.relpath(caminho, BASE_CONHECIMENTO_DIR)

            texto = extrair_texto_txt(caminho) if arquivo.lower().endswith(".txt") else extrair_texto_pdf(caminho)
            if texto:
                base.append({"arquivo": nome_relativo, "texto": texto})

    return base

def normalizar_termos(texto: str):
    return [t for t in re.findall(r"\w+", (texto or "").lower()) if len(t) >= 2 and t not in PALAVRAS_IGNORADAS]

def dividir_em_chunks(texto: str, tamanho: int = TAMANHO_CHUNK, sobreposicao: int = SOBREPOSICAO_CHUNK):
    if not texto:
        return []
    texto = texto.strip()
    chunks = []
    inicio = 0
    while inicio < len(texto):
        fim = min(len(texto), inicio + tamanho)
        chunk = texto[inicio:fim].strip()
        if chunk:
            chunks.append(chunk)
        if fim >= len(texto):
            break
        inicio = max(0, fim - sobreposicao)
    return chunks

def score_chunk(chunk: str, arquivo: str, pergunta: str) -> int:
    chunk_lower = chunk.lower()
    termos = normalizar_termos(pergunta)
    score = sum(chunk_lower.count(termo) * 5 for termo in termos)
    if "decreto" in pergunta.lower() and "decreto" in arquivo.lower():
        score += 30
    return score

@st.cache_data(show_spinner=False)
def indexar_base_em_chunks():
    base = carregar_base_local()
    indice = []
    for doc in base:
        chunks = dividir_em_chunks(doc["texto"], TAMANHO_CHUNK, SOBREPOSICAO_CHUNK)
        for i, chunk in enumerate(chunks, start=1):
            indice.append({"arquivo": doc["arquivo"], "chunk_id": i, "texto": chunk})
    return indice

def buscar_trechos_relevantes(pergunta: str, top_chunks: int = TOP_CHUNKS):
    indice = indexar_base_em_chunks()
    resultados = []
    for item in indice:
        score = score_chunk(item["texto"], item["arquivo"], pergunta)
        if score > 0:
            resultados.append({
                "arquivo": item["arquivo"],
                "trecho": item["texto"],
                "score": score
            })
    resultados.sort(key=lambda x: x["score"], reverse=True)
    return resultados[:top_chunks]

def montar_contexto_local(trechos):
    if not trechos:
        return ""
    blocos = [f"--- DOC {i} ({item['arquivo']}) ---\n{item['trecho']}" for i, item in enumerate(trechos, start=1)]
    return "\n\n".join(blocos)

# =========================================================
# 6. PROCESSAMENTO COM FALLBACK DE COTAS
# =========================================================
def processar_pergunta(pergunta: str):
    try:
        cliente = criar_cliente()
    except Exception as e:
        return {"ok": False, "texto": "", "tempo": 0, "trechos": [], "erro": str(e)}

    trechos = buscar_trechos_relevantes(pergunta, TOP_CHUNKS)
    contexto = montar_contexto_local(trechos)
    usou_web = False

    if contexto:
        prompt_usuario = f"ORDEM:\n{pergunta}\n\nBASE LOCAL:\n{contexto}"
        ferramentas = None
    else:
        prompt_usuario = f"ORDEM:\n{pergunta}\n\nSem base local. Busque na web."
        ferramentas = [{"google_search": {}}]
        usou_web = True

    inicio = time.time()
    ultimo_erro = ""

    # Tentativas alternando modelos em caso de erro 429
    for modelo in MODELOS:
        try:
            config_args = {
                "system_instruction": PROMPT_SISTEMA,
                "temperature": 0.0
            }
            if ferramentas:
                config_args["tools"] = ferramentas

            resposta = cliente.models.generate_content(
                model=modelo,
                contents=prompt_usuario,
                config=types.GenerateContentConfig(**config_args)
            )
            tempo = round(time.time() - inicio, 2)
            texto = resposta.text.strip() if hasattr(resposta, "text") and resposta.text else ""

            return {
                "ok": True,
                "texto": texto,
                "tempo": tempo,
                "trechos": trechos,
                "usou_web": usou_web,
                "modelo": modelo,
                "erro": ""
            }
        except Exception as e:
            ultimo_erro = str(e)
            if "429" in ultimo_erro or "RESOURCE_EXHAUSTED" in ultimo_erro:
                time.sleep(2)  # Pausa antes do fallback
                continue
            else:
                break

    return {
        "ok": False,
        "texto": "",
        "tempo": round(time.time() - inicio, 2),
        "trechos": trechos,
        "erro": f"Cota temporariamente excedida em todos os modelos. Aguarde 1 minuto. Detalhes: {ultimo_erro}"
    }

# =========================================================
# 7. INTERFACE STREAMLIT
# =========================================================
st.markdown("""
<div class="romano-wrap">
    <div class="romano-title">ROMANO</div>
    <div class="romano-subtitle">A IA QUE NÃO PASSA PANO</div>
    <div class="romano-slogan">Inteligência Técnica • Precisão Normativa • Respostas Diretas</div>
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
        with st.spinner("ROMANO processando resposta..."):
            resultado = processar_pergunta(pergunta)

        if not resultado["ok"]:
            st.error("Erro na geração da resposta.")
            st.code(resultado["erro"])
        else:
            st.markdown(resultado["texto"])

            debug_info = ""
            if mostrar_debug:
                origem = "Busca Web Externa" if resultado.get("usou_web") else "Base Documental Local"
                debug_info = (
                    f"Origem da Fonte: {origem}\n"
                    f"Trechos Analisados: {len(resultado.get('trechos', []))}\n"
                    f"Modelo Utilizado: {resultado.get('modelo')}\n"
                    f"Tempo de Processamento: {resultado.get('tempo', 0)} s"
                )
                with st.expander("Diagnóstico Operacional", expanded=False):
                    st.markdown(f'<div class="debug-box">{debug_info}</div>', unsafe_allow_html=True)

            st.session_state.mensagens.append({
                "role": "assistant",
                "content": resultado["texto"],
                "debug": debug_info
            })
