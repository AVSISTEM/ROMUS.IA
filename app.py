import os
import re
import time
import streamlit as st
from google import genai
from google.genai import types
from pypdf import PdfReader

# =========================================================
# 1. CONFIGURAÇÃO DA PÁGINA E ESTILO VISUAL (ESCUDO iOS FIX)
# =========================================================
st.set_page_config(
    page_title="ROMANO",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
/* Remove o cabeçalho/rodapé padrão do Streamlit */
[data-testid="stHeader"], footer {
    visibility: hidden;
}

.main .block-container {
    max-width: 1100px;
    padding-top: 1rem;
    padding-bottom: 5rem;
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

/* Garante fundo escuro na barra de chat inferior no mobile */
.stChatInputContainer, div[data-testid="stChatInput"] {
    background-color: #0e1117 !important;
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
# 2. CONFIGURAÇÕES GERAIS E PARÂMETROS
# =========================================================
BASE_CONHECIMENTO_DIR = "base_conhecimento"
ARQUIVOS_SUPORTADOS = (".txt", ".pdf")
MODELO_GEMINI = "gemini-2.5-flash"

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
Sua missão é responder prioritariamente com base na base local fornecida pelo sistema, com fidelidade documental e rigor técnico.
Você deve buscar a fonte mais central, mais direta e mais específica para responder à pergunta.
Você não deve responder com base em menções periféricas quando houver indício de que existe documento mais adequado.

REGRAS ABSOLUTAS
1. Nunca invente leis, artigos, itens, subitens, datas, normas, entendimentos, citações ou fatos.
2. Nunca afirme que encontrou algo se isso não constar de forma real na base recebida.
3. Nunca trate hipótese como fato.
4. Nunca complete lacunas com suposição disfarçada de certeza.
5. Nunca distorça o conteúdo localizado.
6. Se não houver base suficiente, diga isso claramente.
7. Se houver dúvida relevante, deixe a incerteza explícita.
8. Sempre prefira precisão a velocidade.
9. Sempre prefira resposta exata a texto longo e genérico.
10. Sempre responda em português do Brasil.

HIERARQUIA DE QUALIDADE DA FONTE
1. Documento que define expressamente o conceito perguntado;
2. Documento que liste expressamente os requisitos, medidas, itens ou critérios perguntados;
3. Documento técnico específico do tema;
4. Norma geral relacionada;
5. Documento administrativo, procedimental, formulário ou menção lateral.

FRASES OBRIGATÓRIAS QUANDO NECESSÁRIO
- "Não localizei base suficiente para responder com segurança."
- "A base consultada não trouxe resposta literal para esse ponto."
- "O texto localizado permite apenas conclusão parcial."
- "Não é possível confirmar isso sem extrapolar a base."

ESTRUTURA PADRÃO DE RESPOSTA
RESPOSTA DIRETA:
[resposta objetiva]

FUNDAMENTO:
[arquivo consultado e artigo/item/capítulo se localizados]

GRAU DE CERTEZA:
[expresso na base / conclusão parcial / insuficiente]

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
# 5. LEITURA E INDEXAÇÃO DA BASE
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
# 6. GERAÇÃO DE RESPOSTA
# =========================================================
def gerar_resposta(pergunta: str, modo_estrito: bool = True):
    try:
        cliente = criar_cliente()
    except Exception as e:
        return {"ok": False, "texto": "", "tempo": 0, "trechos": [], "erro": str(e)}

    trechos = buscar_trechos_na_base(pergunta, TOP_CHUNKS)
    contexto = montar_contexto(trechos)

    prompt_usuario = f"PERGUNTA DO USUÁRIO:\n{pergunta}\n\nBASE LOCAL LOCALIZADA:\n{contexto}"

    inicio = time.time()
    try:
        resposta = cliente.models.generate_content(
            model=MODELO_GEMINI,
            contents=prompt_usuario,
            config=types.GenerateContentConfig(
                system_instruction=PROMPT_SISTEMA,
                temperature=0.0
            )
        )
        tempo = round(time.time() - inicio, 2)
        texto = resposta.text.strip() if hasattr(resposta, "text") and resposta.text else ""

        if modo_estrito and not trechos:
            texto = "Não localizei base suficiente para responder com segurança."

        return {
            "ok": True,
            "texto": texto if texto else "Não houve resposta textual do modelo.",
            "tempo": tempo,
            "trechos": trechos,
            "erro": ""
        }
    except Exception as e:
        return {
            "ok": False,
            "texto": "",
            "tempo": round(time.time() - inicio, 2),
            "trechos": trechos,
            "erro": str(e)
        }

# =========================================================
# 7. INTERFACE E EXECUÇÃO
# =========================================================
st.markdown("""
<div class="romano-wrap">
    <div class="romano-title">ROMANO</div>
    <div class="romano-subtitle">A IA que não passa pano.</div>
    <div class="romano-slogan">Respostas diretas. Soluções reais.</div>
</div>
""", unsafe_allow_html=True)

# Painel lateral / Expander para configurações
with st.sidebar:
    st.subheader("Configurações")
    modo_estrito = st.checkbox("Modo estrito (Apenas base local)", value=True)
    mostrar_debug = st.checkbox("Mostrar diagnóstico técnico", value=True)
    if st.button("Limpar Conversa", use_container_width=True):
        st.session_state.mensagens = []
        st.rerun()

# Inicialização do histórico do Chat
if "mensagens" not in st.session_state:
    st.session_state.mensagens = []

# Exibe mensagens anteriores sem ícones
for msg in st.session_state.mensagens:
    with st.chat_message(msg["role"], avatar=""):
        st.markdown(msg["content"])
        if "debug" in msg and mostrar_debug:
            with st.expander("Diagnóstico Técnico", expanded=False):
                st.markdown(f'<div class="debug-box">{msg["debug"]}</div>', unsafe_allow_html=True)

# Entrada fixa na parte inferior
pergunta = st.chat_input("Digite sua ordem...")

if pergunta:
    # Registra e exibe a mensagem do usuário
    st.session_state.mensagens.append({"role": "user", "content": pergunta})
    with st.chat_message("user", avatar=""):
        st.markdown(pergunta)

    # Processa e exibe a resposta do ROMANO
    with st.chat_message("assistant", avatar=""):
        with st.spinner("ROMANO consultando a base..."):
            resultado = gerar_resposta(pergunta, modo_estrito=modo_estrito)

        if not resultado["ok"]:
            st.error("Erro ao gerar resposta.")
            st.code(resultado["erro"])
        else:
            st.markdown(resultado["texto"])

            # Prepara o bloco de debug se ativado
            debug_texto = ""
            if mostrar_debug:
                base_total = carregar_base_local()
                indice_total = indexar_base_em_chunks()
                arquivos_usados = [t["arquivo"] for t in resultado.get("trechos", [])]
                referencias = [t["referencia"] for t in resultado.get("trechos", []) if t.get("referencia")]

                debug_texto = (
                    f"Arquivos na base: {len(base_total)}\n"
                    f"Chunks totais indexados: {len(indice_total)}\n"
                    f"Trechos retornados: {len(resultado.get('trechos', []))}\n"
                    f"Arquivos usados: {arquivos_usados if arquivos_usados else 'Nenhum'}\n"
                    f"Referências: {referencias if referencias else 'Nenhuma'}\n"
                    f"Modelo: {MODELO_GEMINI}\n"
                    f"Tempo de resposta: {resultado.get('tempo', 0)} s"
                )
                with st.expander("Diagnóstico Técnico", expanded=False):
                    st.markdown(f'<div class="debug-box">{debug_texto}</div>', unsafe_allow_html=True)

            # Salva histórico com os dados de debug
            st.session_state.mensagens.append({
                "role": "assistant",
                "content": resultado["texto"],
                "debug": debug_texto
            })
