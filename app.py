import os
import re
import time
import streamlit as st
from google import genai
from google.genai import types
from pypdf import PdfReader

# =========================================================
# 1. CONFIGURAÇÃO DA PÁGINA E ESTILO ESCURO (RESPONSIVO)
# =========================================================
st.set_page_config(
    page_title="ROMANO",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# CSS para garantir exibição correta no mobile e layout escuro
st.markdown("""
<style>
/* Remove o cabeçalho/rodapé padrão do Streamlit */
[data-testid="stHeader"], footer {
    visibility: hidden;
}

/* Oculta totalmente os ícones/avatares do chat */
[data-testid="chatAvatarIcon-user"], 
[data-testid="chatAvatarIcon-assistant"],
div[data-testid="stChatMessage"] > div:first-child {
    display: none !important;
}

/* Força fundo escuro absoluto no app sem quebrar o layout */
html, body, .stApp, [data-testid="stAppViewContainer"] {
    background-color: #0e1117 !important;
    color: #f0f6fc !important;
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
    margin-bottom: 0.1rem;
}

.romano-subtitle {
    font-size: 18px;
    opacity: 0.88;
    margin-bottom: 0.5rem;
}

.romano-slogan {
    font-size: 14px;
    opacity: 0.75;
}

.debug-box {
    border: 1px dashed #555;
    border-radius: 8px;
    padding: 10px;
    background: #161b22;
    font-size: 13px;
    white-space: pre-wrap;
    color: #c9d1d9;
}
</style>
""", unsafe_allow_html=True)

# =========================================================
# 2. CONFIGURAÇÕES GERAIS E MODELOS ATUALIZADOS
# =========================================================
BASE_CONHECIMENTO_DIR = "base_conhecimento"
ARQUIVOS_SUPORTADOS = (".txt", ".pdf")

MODELO_PRINCIPAL = "gemini-3.6-flash"
MODELO_FALLBACK = "gemini-3.6-flash"

TAMANHO_CHUNK = 1800
SOBREPOSICAO_CHUNK = 250
TOP_CHUNKS = 12

PALAVRAS_IGNORADAS = {
    "a", "o", "e", "de", "da", "do", "das", "dos", "um", "uma",
    "em", "por", "para", "com", "sem", "que", "como", "qual",
    "quais", "onde", "quando", "isso", "essa", "esse", "sobre",
    "as", "os", "ao", "aos", "na", "no", "nas", "nos",
    "bom", "boa", "dia", "tarde", "noite", "oi", "ola", "olá"
}

# =========================================================
# 3. PROMPT PRINCIPAL (ROMANO)
# =========================================================
PROMPT_SISTEMA = """
Você é ROMANO, uma inteligência artificial técnica, objetiva e confiável.

IDENTIDADE
Seu nome é ROMANO.
Seu posicionamento é: "A IA que não passa pano."
Seu papel é fornecer respostas diretas, técnicas, jurídicas, administrativas e operacionais, com máxima precisão e sem floreios.
Seu estilo deve ser firme, claro, profissional, disciplinado e eficiente.

MISSÃO
Sua missão é responder prioritariamente com base na base local fornecida pelo sistema.
Quando a informação não estiver na base local, utilize os dados da pesquisa na web para fornecer a resposta exata.

REGRAS ABSOLUTAS
1. Nunca invente leis, artigos, itens, subitens, datas, normas, entendimentos ou fatos.
2. Sempre responda em português do Brasil de forma objetiva e direta.
3. Se a informação for obtida via pesquisa externa, informe no campo FUNDAMENTO que foi utilizada busca na web.

ESTRUTURA PADRÃO DE RESPOSTA
RESPOSTA DIRETA:
[resposta objetiva]

FUNDAMENTO:
[arquivo consultado/artigo ou Fonte da Web]

GRAU DE CERTEZA:
[expresso na base local / obtido via busca web]

OBSERVAÇÃO TÉCNICA:
[apenas se necessário]

BORDÃO OPERACIONAL
ROMANO não passa pano. ROMANO responde com base. ROMANO não inventa. ROMANO resolve.
""".strip()

# =========================================================
# 4. CLIENTE GEMINI
# =========================================================
@st.cache_resource
def criar_cliente():
    api_key = os.getenv("GEMINI_API_KEY") or st.secrets.get("GEMINI_API_KEY", "")
    if not api_key:
        raise ValueError("A chave GEMINI_API_KEY não foi configurada.")
    return genai.Client(api_key=api_key)

# =========================================================
# 5. LEITURA E INDEXAÇÃO DA BASE LOCAL
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

            texto = ""
            if arquivo.lower().endswith(".txt"):
                texto = extrair_texto_txt(caminho)
            elif arquivo.lower().endswith(".pdf"):
                texto = extrair_texto_pdf(caminho)

            if texto:
                base.append({
                    "arquivo": nome_relativo,
                    "texto": texto
                })

    return base

def normalizar_termos(texto: str):
    return [
        t for t in re.findall(r"\w+", (texto or "").lower())
        if len(t) >= 2 and t not in PALAVRAS_IGNORADAS
    ]

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

def extrair_referencia_local(texto: str):
    padroes = [
        r"(art\.?\s*\d+[º°]?)",
        r"(artigo\s+\d+[º°]?)",
        r"(item\s+\d+(\.\d+)*)",
        r"(capítulo\s+[ivxlcdm]+)",
        r"(§\s*\d+[º°]?)"
    ]
    texto_lower = (texto or "").lower()
    for padrao in padroes:
        m = re.search(padrao, texto_lower, re.IGNORECASE)
        if m:
            return m.group(1)
    return ""

def score_chunk(chunk: str, arquivo: str, pergunta: str) -> int:
    chunk_lower = chunk.lower()
    termos = normalizar_termos(pergunta)
    score = sum(chunk_lower.count(termo) * 4 for termo in termos)
    
    if "decreto" in pergunta.lower() and "decreto" in arquivo.lower():
        score += 40
    if "medidas de segurança" in chunk_lower:
        score += 30
    return score

@st.cache_data(show_spinner=False)
def indexar_base_em_chunks():
    base = carregar_base_local()
    indice = []
    for doc in base:
        chunks = dividir_em_chunks(doc["texto"], TAMANHO_CHUNK, SOBREPOSICAO_CHUNK)
        for i, chunk in enumerate(chunks, start=1):
            indice.append({
                "arquivo": doc["arquivo"],
                "chunk_id": i,
                "texto": chunk
            })
    return indice

def buscar_trechos_na_base(pergunta: str, top_chunks: int = TOP_CHUNKS):
    indice = indexar_base_em_chunks()
    resultados = []
    for item in indice:
        score = score_chunk(item["texto"], item["arquivo"], pergunta)
        if score > 0:
            resultados.append({
                "arquivo": item["arquivo"],
                "chunk_id": item["chunk_id"],
                "trecho": item["texto"],
                "score": score,
                "referencia": extrair_referencia_local(item["texto"])
            })
    resultados.sort(key=lambda x: x["score"], reverse=True)
    return resultados[:top_chunks]

def montar_contexto(trechos):
    if not trechos:
        return "Nenhum conteúdo relevante foi localizado na base local."
    blocos = []
    for i, item in enumerate(trechos, start=1):
        bloco = (
            f"[TRECHO {i}]\n"
            f"ARQUIVO: {item['arquivo']}\n"
            f"REFERÊNCIA LOCALIZADA: {item['referencia'] if item['referencia'] else 'N/A'}\n"
            f"TEXTO:\n{item['trecho']}\n"
        )
        blocos.append(bloco)
    return "\n\n".join(blocos)

# =========================================================
# 6. GERAÇÃO DE RESPOSTA COM HIERARQUIA (BASE LOCAL -> WEB)
# =========================================================
def gerar_resposta(pergunta: str):
    try:
        cliente = criar_cliente()
    except Exception as e:
        return {"ok": False, "texto": "", "tempo": 0, "trechos": [], "erro": str(e)}

    trechos = buscar_trechos_na_base(pergunta, TOP_CHUNKS)
    usou_web = False

    if trechos:
        contexto = montar_contexto(trechos)
        prompt_usuario = f"PERGUNTA DO USUÁRIO:\n{pergunta}\n\nBASE LOCAL LOCALIZADA:\n{contexto}"
        ferramentas = []
    else:
        prompt_usuario = f"PERGUNTA DO USUÁRIO:\n{pergunta}\n\nA informação não foi encontrada na base local. Realize uma busca na internet para responder categoricamente."
        ferramentas = [{"google_search": {}}]
        usou_web = True

    inicio = time.time()
    modelos = [MODELO_PRINCIPAL, MODELO_FALLBACK]
    ultimo_erro = ""

    for modelo in modelos:
        for tentativa in range(2):
            try:
                config_params = {
                    "system_instruction": PROMPT_SISTEMA,
                    "temperature": 0.0
                }
                if ferramentas:
                    config_params["tools"] = ferramentas

                resposta = cliente.models.generate_content(
                    model=modelo,
                    contents=prompt_usuario,
                    config=types.GenerateContentConfig(**config_params)
                )
                tempo = round(time.time() - inicio, 2)
                texto = resposta.text.strip() if hasattr(resposta, "text") and resposta.text else ""

                return {
                    "ok": True,
                    "texto": texto if texto else "Não houve resposta textual do modelo.",
                    "tempo": tempo,
                    "trechos": trechos,
                    "usou_web": usou_web,
                    "modelo_usado": modelo,
                    "erro": ""
                }
            except Exception as e:
                ultimo_erro = str(e)
                time.sleep(1)

    return {
        "ok": False,
        "texto": "",
        "tempo": round(time.time() - inicio, 2),
        "trechos": trechos,
        "erro": f"Erro na requisição: {ultimo_erro}"
    }

# =========================================================
# 7. INTERFACE DO USUÁRIO
# =========================================================
st.markdown("""
<div class="romano-wrap">
    <div class="romano-title">ROMANO</div>
    <div class="romano-subtitle">A IA que não passa pano.</div>
    <div class="romano-slogan">Respostas diretas. Soluções reais.</div>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.subheader("Configurações")
    mostrar_debug = st.checkbox("Mostrar diagnóstico técnico", value=True)
    if st.button("Limpar Conversa", use_container_width=True):
        st.session_state.mensagens = []
        st.rerun()

if "mensagens" not in st.session_state:
    st.session_state.mensagens = []

for msg in st.session_state.mensagens:
    with st.chat_message(msg["role"], avatar=None):
        st.markdown(msg["content"])
        if "debug" in msg and mostrar_debug and msg["debug"]:
            with st.expander("Diagnóstico Técnico", expanded=False):
                st.markdown(f'<div class="debug-box">{msg["debug"]}</div>', unsafe_allow_html=True)

pergunta = st.chat_input("Digite sua ordem...")

if pergunta:
    st.session_state.mensagens.append({"role": "user", "content": pergunta})
    with st.chat_message("user", avatar=None):
        st.markdown(pergunta)

    with st.chat_message("assistant", avatar=None):
        with st.spinner("ROMANO processando..."):
            resultado = gerar_resposta(pergunta)

        if not resultado["ok"]:
            st.error("Erro ao gerar resposta.")
            st.code(resultado["erro"])
        else:
            st.markdown(resultado["texto"])

            debug_texto = ""
            if mostrar_debug:
                fonte_usada = "Busca Web (Google)" if resultado.get("usou_web") else "Base Local"
                debug_texto = (
                    f"Origem da resposta: {fonte_usada}\n"
                    f"Trechos locais analisados: {len(resultado.get('trechos', []))}\n"
                    f"Modelo executado: {resultado.get('modelo_usado')}\n"
                    f"Tempo de execução: {resultado.get('tempo', 0)} s"
                )
                with st.expander("Diagnóstico Técnico", expanded=False):
                    st.markdown(f'<div class="debug-box">{debug_texto}</div>', unsafe_allow_html=True)

            st.session_state.mensagens.append({
                "role": "assistant",
                "content": resultado["texto"],
                "debug": debug_texto
            })
