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
/* Remove cabeçalho e rodapé nativos do Streamlit */
[data-testid="stHeader"], footer {
    visibility: hidden;
    height: 0px;
}

/* Oculta avatares do chat */
[data-testid="chatAvatarIcon-user"], 
[data-testid="chatAvatarIcon-assistant"],
div[data-testid="stChatMessage"] > div:first-child {
    display: none !important;
}

/* Garante fundo totalmente escuro no app e na área do iOS */
html, body, .stApp, [data-testid="stAppViewContainer"], [data-testid="stBottom"] {
    background-color: #0e1117 !important;
    color: #f0f6fc !important;
}

/* Ajuste da caixa de entrada do chat */
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
# 2. CONFIGURAÇÕES DA API E MODELO
# =========================================================
BASE_CONHECIMENTO_DIR = "base_conhecimento"
ARQUIVOS_SUPORTADOS = (".txt", ".pdf")

MODELO_PRINCIPAL = "gemini-3.6-flash"

TAMANHO_CHUNK = 2000
SOBREPOSICAO_CHUNK = 300
TOP_CHUNKS = 15

PALAVRAS_IGNORADAS = {
    "a", "o", "e", "de", "da", "do", "das", "dos", "um", "uma",
    "em", "por", "para", "com", "sem", "que", "como", "qual",
    "quais", "onde", "quando", "isso", "essa", "esse", "sobre",
    "as", "os", "ao", "aos", "na", "no", "nas", "nos", "sobre",
    "bom", "boa", "dia", "tarde", "noite", "oi", "ola", "olá",
    "saber", "gostaria", "poderia", "dizer", "qual", "quais"
}

# =========================================================
# 3. PROMPT DO SISTEMA DE ALTA INTELIGÊNCIA LÓGICA (ROMANO)
# =========================================================
PROMPT_SISTEMA = """
Você é o ROMANO, um sistema de inteligência artificial autônomo, técnico, analítico e de alta precisão lógica.

DIRETRIZES DE PERSONALIDADE E ESTILO
- Nome: ROMANO.
- Slogan: "A IA que não passa pano. ROMANO responde com base. ROMANO não inventa. ROMANO resolve."
- Postura: Firme, extremamente lógica, concisa, formal e direta.
- Comunicação: Proibido o uso de saudações banais, enrolação ou introduções vazias. Vá direto ao ponto técnico.

MECANISMO DE RACIOCÍNIO E ANÁLISE (CHAIN OF THOUGHT INTERNO)
Antes de formular qualquer resposta, execute internamente os seguintes passos lógicos:
1. Decomposição do Problema: Identifique os termos técnicos principais, parâmetros numéricos e requisitos legais contidos na pergunta.
2. Mapeamento de Fontes Locais: Analise os trechos de documentos fornecidos e determine se eles contêm a regra específica, artigo, tabela ou item normativo aplicável.
3. Avaliação de Suficiência:
   - Se a base local contiver a resposta: extraia o dado exato com fidelidade absoluta.
   - Se a base local for omissa ou parcial: utilize o conhecimento geral da web via Google Search para complementar a resposta com precisão jurídica/técnica.
4. Validação da Conclusão: Confirme se a resposta responde estritamente ao que foi perguntado, eliminando redundâncias.

REGRAS ABSOLUTAS DE PRECISÃO
1. Fidelidade Normativa: Jamais invente números de artigos, tabelas, distâncias de caminhamento, larguras de saídas, exigências de carga de incêndio ou parâmetros técnicos.
2. Hierarquia Normativa: Leis/Decretos sobressam a Instruções Técnicas/Normas e estas sobressam a manuais genéricos.
3. Objetividade: Apresente cálculos, dados e parâmetros em tópicos estruturados de rápida leitura.

ESTRUTURA DE RESPOSTA OBRIGATÓRIA

RESPOSTA DIRETA:
[Apresentação clara, imediata e objetiva da solução ou do dado técnico solicitado]

FUNDAMENTAÇÃO TÉCNICA:
[Identificação exata do documento, artigo, item, tabela ou fonte consultada. Caso tenha sido obtida via busca externa, explicite que foi consultada a legislação/fonte web]

GRAU DE CERTEZA TÉCNICA:
[Citar se a conclusão é Expressa na Base Local, Derivada da Legislação Web ou Inconclusiva]

OBSERVAÇÃO OPERACIONAL:
[Orientações complementares estritamente necessárias ou ressalvas de aplicação prática]

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
        raise ValueError("A chave GEMINI_API_KEY não foi configurada nos Secrets ou Variáveis de Ambiente.")
    return genai.Client(api_key=api_key)

# =========================================================
# 5. MOTOR DE BUSCA SEMÂNTICA E PROCESSAMENTO DA BASE LOCAL
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
        r"(tabela\s+\d+([a-z])?)",
        r"(capítulo\s+[ivxlcdm]+)",
        r"(§\s*\d+[º°]?)"
    ]
    texto_lower = (texto or "").lower()
    for padrao in padroes:
        m = re.search(padrao, texto_lower, re.IGNORECASE)
        if m:
            return m.group(1).upper()
    return ""

def score_chunk_semantico(chunk: str, arquivo: str, pergunta: str) -> int:
    chunk_lower = chunk.lower()
    pergunta_lower = pergunta.lower()
    termos = normalizar_termos(pergunta)
    
    score = 0
    
    # Ocorrência de termos chave
    for termo in termos:
        contagem = chunk_lower.count(termo)
        score += contagem * 5
    
    # Pesos normativos específicos
    if "it" in pergunta_lower or "instrução técnica" in pergunta_lower:
        if "it-" in arquivo.lower() or "instrução técnica" in chunk_lower:
            score += 50
    if "decreto" in pergunta_lower and "decreto" in arquivo.lower():
        score += 50
    if "tabela" in pergunta_lower and "tabela" in chunk_lower:
        score += 30
    if "saída de emergência" in pergunta_lower and "saída" in chunk_lower:
        score += 25

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

def buscar_trechos_relevantes(pergunta: str, top_chunks: int = TOP_CHUNKS):
    indice = indexar_base_em_chunks()
    resultados = []
    for item in indice:
        score = score_chunk_semantico(item["texto"], item["arquivo"], pergunta)
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

def montar_contexto_local(trechos):
    if not trechos:
        return ""
    blocos = []
    for i, item in enumerate(trechos, start=1):
        bloco = (
            f"--- DOCUMENTO LOCAL {i} ---\n"
            f"ARQUIVO FONTE: {item['arquivo']}\n"
            f"REFERÊNCIA DETECTADA: {item['referencia'] if item['referencia'] else 'N/A'}\n"
            f"CONTEÚDO:\n{item['trecho']}\n"
        )
        blocos.append(bloco)
    return "\n\n".join(blocos)

# =========================================================
# 6. PROCESSAMENTO DE RESPOSTA (LOGICA LOCAL -> SEARCH WEB)
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
        prompt_usuario = (
            f"ORDEM DO USUÁRIO:\n{pergunta}\n\n"
            f"BASE DOCUMENTAL LOCAL DISPONÍVEL:\n{contexto}\n\n"
            f"INSTRUÇÃO DE EXECUÇÃO: Responda fundamentando-se prioritariamente nos documentos locais acima. "
            f"Se a base contiver os dados necessários, não acione fontes externas."
        )
        ferramentas = None
    else:
        prompt_usuario = (
            f"ORDEM DO USUÁRIO:\n{pergunta}\n\n"
            f"INSTRUÇÃO DE EXECUÇÃO: A informação não foi localizada na base local de documentos. "
            f"Realize uma busca completa e atualizada na internet para fundamentar a resposta com precisão."
        )
        ferramentas = [{"google_search": {}}]
        usou_web = True

    inicio = time.time()
    ultimo_erro = ""

    # Tentativas de execução resilientes
    for tentativa in range(3):
        try:
            config_args = {
                "system_instruction": PROMPT_SISTEMA,
                "temperature": 0.0
            }
            if ferramentas:
                config_args["tools"] = ferramentas

            resposta = cliente.models.generate_content(
                model=MODELO_PRINCIPAL,
                contents=prompt_usuario,
                config=types.GenerateContentConfig(**config_args)
            )
            tempo = round(time.time() - inicio, 2)
            texto = resposta.text.strip() if hasattr(resposta, "text") and resposta.text else ""

            return {
                "ok": True,
                "texto": texto if texto else "Não foi possível gerar uma resposta válida.",
                "tempo": tempo,
                "trechos": trechos,
                "usou_web": usou_web,
                "modelo": MODELO_PRINCIPAL,
                "erro": ""
            }
        except Exception as e:
            ultimo_erro = str(e)
            time.sleep(1.5 * (tentativa + 1))

    return {
        "ok": False,
        "texto": "",
        "tempo": round(time.time() - inicio, 2),
        "trechos": trechos,
        "erro": f"Falha na comunicação com o modelo de inteligência: {ultimo_erro}"
    }

# =========================================================
# 7. INTERFACE DO USUÁRIO (STREAMLIT)
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

# Exibição do histórico de mensagens
for msg in st.session_state.mensagens:
    with st.chat_message(msg["role"], avatar=None):
        st.markdown(msg["content"])
        if "debug" in msg and mostrar_debug and msg["debug"]:
            with st.expander("Diagnóstico Operacional", expanded=False):
                st.markdown(f'<div class="debug-box">{msg["debug"]}</div>', unsafe_allow_html=True)

# Campo de entrada
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
                origem = "Busca Web Externa (Google)" if resultado.get("usou_web") else "Base Documental Local"
                debug_info = (
                    f"Origem da Fonte: {origem}\n"
                    f"Trechos Analisados na Base: {len(resultado.get('trechos', []))}\n"
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
