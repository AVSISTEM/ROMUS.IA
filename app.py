import os
import re
import time
import streamlit as st
from google import genai
from google.genai import types
from pypdf import PdfReader

# =========================================================
# 1. CONFIGURAÇÃO DA PÁGINA E CSS ESCURO (OPTIMIZADO iOS)
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
# 2. CONFIGURAÇÃO DE MODELOS E BASE LOCAL
# =========================================================
BASE_CONHECIMENTO_DIR = "base_conhecimento"
ARQUIVOS_SUPORTADOS = (".txt", ".pdf")

MODELO_UNICO = "gemini-3.6-flash"

TAMANHO_CHUNK = 1500
SOBREPOSICAO_CHUNK = 200
TOP_CHUNKS = 8

PALAVRAS_IGNORADAS = {
    "a", "o", "e", "de", "da", "do", "das", "dos", "um", "uma",
    "em", "por", "para", "com", "sem", "que", "como", "qual",
    "quais", "onde", "quando", "isso", "essa", "esse", "sobre",
    "as", "os", "ao", "aos", "na", "no", "nas", "nos"
}

# =========================================================
# 3. BASE DE RESPOSTAS RÁPIDAS LOCAIS
# =========================================================
RESPOSTAS_RAPIDAS = {
    ("oi", "olá", "ola", "bom dia", "boa tarde", "boa noite", "comandante"): 
        "RESPOSTA DIRETA:\nPronto para o serviço, Comandante. Envie a ordem ou consulta técnica.\n\nFUNDAMENTAÇÃO TÉCNICA:\nSistema Operacional ROMANO v3.6.\n\nGRAU DE CERTEZA TÉCNICA:\nAtivo e Local.\n\nBORDÃO OPERACIONAL\nROMANO não passa pano. ROMANO responde com base. ROMANO não inventa. ROMANO resolve.",
    
    ("quem é você", "quem e voce", "quem e você", "o que você faz", "quais suas funcoes"): 
        "RESPOSTA DIRETA:\nSou o ROMANO, uma Inteligência Artificial técnica, objetiva e analítica voltada para legislação, normas e engenharia de segurança.\n\nFUNDAMENTAÇÃO TÉCNICA:\nArquitetura Híbrida de Consulta Local e Raciocínio Normativo.\n\nGRAU DE CERTEZA TÉCNICA:\nMódulo Principal.\n\nBORDÃO OPERACIONAL\nROMANO não passa pano. ROMANO responde com base. ROMANO não inventa. ROMANO resolve.",
    
    ("slogan", "bordao", "bordão", "qual seu bordao"): 
        "RESPOSTA DIRETA:\nROMANO não passa pano. ROMANO responde com base. ROMANO não inventa. ROMANO resolve.\n\nFUNDAMENTAÇÃO TÉCNICA:\nDiretriz de Identidade Corporativa.\n\nGRAU DE CERTEZA TÉCNICA:\nIncondicional."
}

# =========================================================
# 4. PROMPT DO SISTEMA
# =========================================================
PROMPT_SISTEMA = """
Você é o ROMANO, uma inteligência artificial autônoma, técnica e objetiva.

DIRETRIZES
- Slogan: "ROMANO não passa pano. ROMANO responde com base. ROMANO não inventa. ROMANO resolve."
- Responda diretamente ao que foi solicitado, de forma extremamente técnica, concisa e sem enrolação.
- Analise SEMPRE todo o histórico de mensagens da conversa. Se o usuário disser "está errado", identifique imediatamente qual afirmação anterior ele está contestando e reavalie os dados normativos.

ESTRUTURA DE RESPOSTA OBRIGATÓRIA

RESPOSTA DIRETA:
[Resposta objetiva]

FUNDAMENTAÇÃO TÉCNICA:
[Norma, decreto, artigo, item, tabela ou fonte consultada]

GRAU DE CERTEZA TÉCNICA:
[Expressa na Base Local / Obtida via Busca Web]

OBSERVAÇÃO OPERACIONAL:
[Apenas se estritamente necessário]

BORDÃO OPERACIONAL
ROMANO não passa pano. ROMANO responde com base. ROMANO não inventa. ROMANO resolve.
""".strip()

# =========================================================
# 5. INICIALIZAÇÃO DA API E PROCESSAMENTO DE ARQUIVOS
# =========================================================
@st.cache_resource
def criar_cliente():
    api_key = os.getenv("GEMINI_API_KEY") or st.secrets.get("GEMINI_API_KEY", "")
    if not api_key:
        return None
    return genai.Client(api_key=api_key)

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
    if "it" in pergunta.lower() and "it" in arquivo.lower():
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
    blocos = [f"--- DOCUMENTO LOCAL ({item['arquivo']}) ---\n{item['trecho']}" for item in trechos]
    return "\n\n".join(blocos)

# =========================================================
# 6. MONTAGEM DE MENSAGENS COM HISTÓRICO COMPLETO
# =========================================================
def construir_conteudo_com_historico(historico_mensagens, pergunta_atual, contexto_local):
    contents = []
    
    # Adiciona o contexto do histórico de conversa mantido no Streamlit
    for msg in historico_mensagens:
        role_api = "user" if msg["role"] == "user" else "model"
        contents.append(types.Content(
            role=role_api,
            parts=[types.Part.from_text(text=msg["content"])]
        ))

    # Constrói o texto do turno atual com a inclusão de base local, se houver
    if contexto_local:
        prompt_final = f"ORDEM:\n{pergunta_atual}\n\nBASE DOCUMENTAL LOCAL DISPONÍVEL:\n{contexto_local}"
    else:
        prompt_final = f"ORDEM:\n{pergunta_atual}\n\nSem correspondência direta na base local. Realize busca para confirmar norma ou regulamento."

    contents.append(types.Content(
        role="user",
        parts=[types.Part.from_text(text=prompt_final)]
    ))
    
    return contents

# =========================================================
# 7. MOTOR DE PROCESSAMENTO DE PERGUNTAS
# =========================================================
def processar_pergunta(pergunta: str, historico: list):
    p_clean = pergunta.strip().lower()

    # PASSO 1: Resposta Rápida (Apenas para saudações e comandos simples isolados)
    if len(historico) <= 1:
        for gatilhos, resposta_direta in RESPOSTAS_RAPIDAS.items():
            if any(p_clean == g or p_clean.startswith(g) for g in gatilhos):
                return {
                    "ok": True,
                    "texto": resposta_direta,
                    "tempo": 0.01,
                    "trechos": [],
                    "usou_web": False,
                    "modelo": "Cache Interno Local",
                    "erro": ""
                }

    # PASSO 2: Busca Local
    trechos = buscar_trechos_relevantes(pergunta, TOP_CHUNKS)
    contexto = montar_contexto_local(trechos)
    usou_web = False

    cliente = criar_cliente()
    inicio = time.time()

    if cliente:
        try:
            ferramentas = None if contexto else [{"google_search": {}}]
            if not contexto:
                usou_web = True

            config_args = {
                "system_instruction": PROMPT_SISTEMA,
                "temperature": 0.0
            }
            if ferramentas:
                config_args["tools"] = ferramentas

            # Constrói as mensagens enviando TODO o histórico
            contents = construir_conteudo_com_historico(historico[:-1], pergunta, contexto)

            resposta = cliente.models.generate_content(
                model=MODELO_UNICO,
                contents=contents,
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
                "modelo": MODELO_UNICO,
                "erro": ""
            }
        except Exception as e:
            err_msg = str(e)
            if contexto:
                texto_fallback = f"RESPOSTA DIRETA (MODO OFFLINE LOCAL):\n{trechos[0]['trecho'][:1000]}...\n\nFUNDAMENTAÇÃO TÉCNICA:\nArquivo: {trechos[0]['arquivo']}\n\nGRAU DE CERTEZA TÉCNICA:\nExpressa na Base Local\n\nBORDÃO OPERACIONAL\nROMANO não passa pano. ROMANO responde com base. ROMANO não inventa. ROMANO resolve."
                return {
                    "ok": True,
                    "texto": texto_fallback,
                    "tempo": round(time.time() - inicio, 2),
                    "trechos": trechos,
                    "usou_web": False,
                    "modelo": "Banco Local (Contingência)",
                    "erro": ""
                }
            return {
                "ok": False,
                "texto": "",
                "tempo": round(time.time() - inicio, 2),
                "trechos": trechos,
                "erro": f"Erro ou limite de cota atingido. Detalhes: {err_msg}"
            }

    return {
        "ok": False,
        "texto": "",
        "tempo": 0,
        "trechos": [],
        "erro": "Chave GEMINI_API_KEY não localizada nas configurações."
    }

# =========================================================
# 8. INTERFACE STREAMLIT
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
        with st.spinner("ROMANO processando com memória operacional..."):
            resultado = processar_pergunta(pergunta, st.session_state.mensagens)

        if not resultado["ok"]:
            st.error("Erro no processamento.")
            st.code(resultado["erro"])
        else:
            st.markdown(resultado["texto"])

            debug_info = ""
            if mostrar_debug:
                origem = "Busca Web Externa" if resultado.get("usou_web") else "Base Documental Local / Cache"
                debug_info = (
                    f"Origem da Fonte: {origem}\n"
                    f"Trechos Analisados: {len(resultado.get('trechos', []))}\n"
                    f"Motor de Execução: {resultado.get('modelo')}\n"
                    f"Tempo de Processamento: {resultado.get('tempo', 0)} s"
                )
                with st.expander("Diagnóstico Operacional", expanded=False):
                    st.markdown(f'<div class="debug-box">{debug_info}</div>', unsafe_allow_html=True)

            st.session_state.mensagens.append({
                "role": "assistant",
                "content": resultado["texto"],
                "debug": debug_info
            })
