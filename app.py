import os
import re
import time
import streamlit as st
from google import genai
from pypdf import PdfReader

# =========================================================
# CONFIGURAÇÃO DA PÁGINA
# =========================================================
st.set_page_config(
    page_title="ROMANUS",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# =========================================================
# CONFIGURAÇÕES GERAIS
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
# PROMPT PRINCIPAL
# =========================================================
PROMPT_SISTEMA = """
Você é ROMANUS, uma inteligência artificial técnica, objetiva e confiável.

IDENTIDADE
Seu nome é ROMANUS.
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
Ao analisar a base, siga esta ordem de preferência:
1. documento que define expressamente o conceito perguntado;
2. documento que liste expressamente os requisitos, medidas, itens ou critérios perguntados;
3. documento técnico específico do tema;
4. norma geral relacionada;
5. documento administrativo, procedimental, formulário, anexo de referência, índice, sumário ou menção lateral.

REGRA DE PRIORIDADE DOCUMENTAL
Se a pergunta pedir definição, lista, requisitos, medidas, critérios ou classificação:
- priorize o documento que traga isso expressamente;
- não use como fundamento principal um trecho que apenas menciona o tema;
- não use sumário, índice, anexo citado sem conteúdo, nota lateral ou referência indireta como base principal;
- não trate menção genérica como resposta suficiente;
- se houver apenas menção indireta, diga claramente que a base não trouxe a definição/lista expressa.

USO DA BASE LOCAL
Se o sistema fornecer trechos de documentos:
- trate isso como fonte principal;
- responda com fidelidade ao conteúdo;
- cite o nome do arquivo, se disponível;
- cite artigo, item, capítulo ou parágrafo, se estiverem localizados no trecho;
- diferencie claramente:
  a) texto expresso;
  b) resumo fiel;
  c) interpretação mínima.

QUANDO HOUVER TEXTO EXPRESSO
Se a base trouxer resposta clara e direta:
- responda de forma objetiva;
- preserve o sentido original;
- não acrescente requisito inexistente;
- não amplie o texto além do que ele sustenta.

QUANDO A BASE FOR PARCIAL
Se a base trouxer apenas elementos relacionados:
- diga expressamente que não há resposta literal completa;
- apresente apenas conclusão parcial;
- identifique que se trata de interpretação mínima;
- não extrapole além do que o texto sustenta;
- não transforme referência lateral em fundamento principal.

QUANDO HOUVER CONFLITO ENTRE TRECHOS
Se houver mais de um trecho:
- prefira o mais específico;
- prefira o mais central ao tema;
- prefira o que contenha definição, lista, regra ou critério expresso;
- descarte trechos meramente incidentais como fundamento principal.

FRASES OBRIGATÓRIAS QUANDO NECESSÁRIO
Use estas fórmulas quando forem verdadeiras:
- "Não localizei base suficiente para responder com segurança."
- "A base consultada não trouxe resposta literal para esse ponto."
- "O texto localizado permite apenas conclusão parcial."
- "Não é possível confirmar isso sem extrapolar a base."
- "Não encontrei elemento suficiente para responder com precisão."
- "Os trechos localizados apenas mencionam o tema, mas não trazem a definição/lista expressa."

ESTILO DE RESPOSTA
A resposta deve ser:
- direta;
- técnica;
- clara;
- profissional;
- sem rodeios;
- sem bajulação;
- sem excesso de texto.

ESTRUTURA PADRÃO
Quando houver base útil, use preferencialmente:

RESPOSTA DIRETA:
[resposta objetiva]

FUNDAMENTO:
[arquivo consultado e artigo/item/capítulo se localizados]

GRAU DE CERTEZA:
[expresso na base / conclusão parcial / insuficiente]

OBSERVAÇÃO TÉCNICA:
[apenas se necessário]

INSTRUÇÃO ESPECIAL PARA ENUMERAÇÕES
Se a pergunta pedir lista, rol, definição ou enumeração:
- procure expressões como:
  "constituem", "são", "compreendem", "incluem", "fica instituído", "deverá ser levado em consideração", "classificam-se", "são medidas", "são requisitos";
- se houver enumeração expressa, reproduza a lista fielmente;
- se houver incisos, preserve a estrutura;
- se não houver enumeração expressa, não invente a lista.

PROIBIÇÕES
É proibido:
- criar citações falsas;
- fingir pesquisa;
- omitir incerteza;
- esconder falta de base com texto bonito;
- contradizer a base local sem explicar;
- responder como se tivesse certeza quando não tem;
- usar índice, sumário, título de anexo ou menção indireta como se fossem conteúdo normativo completo.

REGRA DE OURO
Na dúvida, seja honesta.
Melhor admitir limite do que entregar uma resposta errada com aparência bonita.

BORDÃO OPERACIONAL
ROMANUS não passa pano.
ROMANUS responde com base.
ROMANUS não inventa.
ROMANUS resolve.
""".strip()

# =========================================================
# ESTILO VISUAL
# =========================================================
st.markdown("""
<style>
[data-testid="stHeader"] {
    background: transparent !important;
}

.main .block-container {
    max-width: 1100px;
    padding-top: 1.2rem;
    padding-bottom: 3rem;
}

.romanus-wrap {
    text-align: center;
    margin-top: 1rem;
    margin-bottom: 2rem;
}

.romanus-title {
    font-size: 52px;
    font-weight: 900;
    margin-bottom: 0.2rem;
}

.romanus-subtitle {
    font-size: 22px;
    opacity: 0.88;
    margin-bottom: 1rem;
}

.romanus-slogan {
    font-size: 18px;
    opacity: 0.76;
    margin-bottom: 2rem;
}

.bloco-resposta {
    border: 1px solid #d9d9d9;
    border-radius: 12px;
    padding: 18px;
    background: #fafafa;
    white-space: pre-wrap;
    font-size: 18px;
    line-height: 1.6;
}

.debug-box {
    border: 1px dashed #999;
    border-radius: 10px;
    padding: 12px;
    background: #fcfcfc;
    font-size: 14px;
    white-space: pre-wrap;
}
</style>
""", unsafe_allow_html=True)

# =========================================================
# CLIENTE GEMINI
# =========================================================
@st.cache_resource
def criar_cliente():
    api_key = st.secrets.get("GEMINI_API_KEY", "")
    if not api_key:
        raise ValueError("A chave GEMINI_API_KEY não foi encontrada no secrets.")
    return genai.Client(api_key=api_key)

# =========================================================
# LEITURA DOS ARQUIVOS
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
                    "texto": texto,
                    "texto_lower": texto.lower()
                })

    return base

# =========================================================
# FUNÇÕES DE APOIO
# =========================================================
def normalizar_termos(texto: str):
    return [
        t for t in re.findall(r"\w+", (texto or "").lower())
        if len(t) >= 2 and t not in PALAVRAS_IGNORADAS
    ]

def pergunta_pede_lista(pergunta: str) -> bool:
    p = (pergunta or "").lower()
    gatilhos = [
        "quais", "lista", "rol", "enumere", "enumeração",
        "medidas", "requisitos", "itens", "critérios",
        "quais são", "quais as", "defina", "definição"
    ]
    return any(g in p for g in gatilhos)

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
    referencia = ""

    padroes = [
        r"(art\.?\s*\d+[º°]?)",
        r"(artigo\s+\d+[º°]?)",
        r"(item\s+\d+(\.\d+)*)",
        r"(capítulo\s+[ivxlcdm]+)",
        r"(capitulo\s+[ivxlcdm]+)",
        r"(§\s*\d+[º°]?)"
    ]

    texto_lower = (texto or "").lower()

    for padrao in padroes:
        m = re.search(padrao, texto_lower, re.IGNORECASE)
        if m:
            referencia = m.group(1)
            break

    return referencia

def score_nome_arquivo(nome: str, pergunta: str) -> int:
    score = 0
    nome_lower = nome.lower()
    pergunta_lower = pergunta.lower()

    palavras_fortes = [
        "decreto", "lei", "regulamento", "instrução", "it",
        "norma", "anexo", "quadro", "tabela", "medidas",
        "segurança", "incêndio", "capítulo", "artigo"
    ]

    for palavra in palavras_fortes:
        if palavra in nome_lower and palavra in pergunta_lower:
            score += 18

    if pergunta_pede_lista(pergunta):
        for palavra in ["anexo", "quadro", "tabela", "medidas", "regulamento", "decreto"]:
            if palavra in nome_lower:
                score += 16

    return score

def score_chunk(chunk: str, arquivo: str, pergunta: str) -> int:
    chunk_lower = chunk.lower()
    termos = normalizar_termos(pergunta)
    score = 0

    for termo in termos:
        score += chunk_lower.count(termo) * 4

    score += score_nome_arquivo(arquivo, pergunta)

    if pergunta_pede_lista(pergunta):
        gatilhos_lista = [
            "constituem", "incluem", "compreendem", "são medidas",
            "medidas de segurança contra incêndio",
            "deverá ser levado em consideração",
            "i -", "ii -", "iii -", "iv -", "v -", "vi -"
        ]
        for g in gatilhos_lista:
            if g in chunk_lower:
                score += 25

    if "decreto" in pergunta.lower() and "decreto" in arquivo.lower():
        score += 40

    if "medidas de segurança contra incêndio" in chunk_lower:
        score += 35

    if "artigo 20" in chunk_lower or "art. 20" in chunk_lower:
        score += 40

    if "capítulo viii" in chunk_lower or "capitulo viii" in chunk_lower:
        score += 20

    return score

# =========================================================
# INDEXAÇÃO DE TODOS OS ARQUIVOS EM CHUNKS
# =========================================================
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
                "texto": chunk,
                "texto_lower": chunk.lower()
            })

    return indice

# =========================================================
# BUSCA GLOBAL EM TODOS OS TRECHOS DE TODOS OS ARQUIVOS
# =========================================================
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

def montar_contexto(trechos, pergunta: str):
    if not trechos:
        return "Nenhum conteúdo relevante foi localizado na base local."

    alerta = ""
    if pergunta_pede_lista(pergunta):
        alerta = (
            "[ATENÇÃO ESPECIAL]\n"
            "A pergunta pede lista, enumeração, rol ou medidas. "
            "Procure enumeração expressa e, se existir, reproduza fielmente.\n\n"
        )

    blocos = []
    for i, item in enumerate(trechos, start=1):
        bloco = (
            f"[TRECHO {i}]\n"
            f"ARQUIVO: {item['arquivo']}\n"
            f"CHUNK: {item['chunk_id']}\n"
            f"REFERÊNCIA LOCALIZADA: {item['referencia'] if item['referencia'] else 'não localizada'}\n"
            f"TEXTO:\n{item['trecho']}\n"
        )
        blocos.append(bloco)

    return alerta + "\n\n".join(blocos)

# =========================================================
# GERAÇÃO DE RESPOSTA
# =========================================================
def gerar_resposta(pergunta: str, modo_estrito: bool = True):
    cliente = criar_cliente()
    trechos = buscar_trechos_na_base(pergunta, TOP_CHUNKS)
    contexto = montar_contexto(trechos, pergunta)

    prompt_final = f"""
{PROMPT_SISTEMA}

PERGUNTA DO USUÁRIO:
{pergunta}

BASE LOCAL LOCALIZADA:
{contexto}

INSTRUÇÕES FINAIS DE EXECUÇÃO
1. Considere que os trechos acima foram selecionados após consulta global em toda a base local.
2. Priorize os trechos mais específicos e mais centrais ao tema.
3. Cite os arquivos usados e a referência localizada, se houver.
4. Não invente fundamento.
5. Se a pergunta pedir lista, rol, enumeração, medidas ou requisitos, só apresente a lista se ela estiver expressamente visível nos trechos.
6. Se houver enumeração expressa, reproduza a enumeração fielmente.
7. Se os trechos apenas mencionarem o tema, diga isso claramente.
8. Não esconda limitação com texto bonito.
9. Não use sumário, índice ou menção indireta como fundamento principal.
10. Se houver um trecho mais específico que outro, prefira o mais específico.

Se não houver base suficiente, use uma das fórmulas de insuficiência previstas no prompt.
""".strip()

    inicio = time.time()

    try:
        resposta = cliente.models.generate_content(
            model=MODELO_GEMINI,
            contents=prompt_final
        )

        tempo = round(time.time() - inicio, 2)
        texto = ""

        if hasattr(resposta, "text") and resposta.text:
            texto = resposta.text.strip()

        if not texto:
            texto = "Não houve resposta textual do modelo."

        if modo_estrito and not trechos:
            texto = "Não localizei base suficiente para responder com segurança."

        return {
            "ok": True,
            "texto": texto,
            "tempo": tempo,
            "trechos": trechos,
            "erro": ""
        }

    except Exception as e:
        tempo = round(time.time() - inicio, 2)
        return {
            "ok": False,
            "texto": "",
            "tempo": tempo,
            "trechos": trechos,
            "erro": str(e)
        }

# =========================================================
# CABEÇALHO
# =========================================================
st.markdown("""
<div class="romanus-wrap">
    <div class="romanus-title">ROMANUS</div>
    <div class="romanus-subtitle">A IA que não passa pano.</div>
    <div class="romanus-slogan">Respostas diretas. Soluções reais.</div>
</div>
""", unsafe_allow_html=True)

# =========================================================
# CONTROLES
# =========================================================
with st.expander("Configuração", expanded=False):
    modo_estrito = st.checkbox(
        "Modo estrito (responder só com base local, sem improvisar)",
        value=True
    )
    mostrar_debug = st.checkbox(
        "Mostrar diagnóstico técnico",
        value=True
    )

# =========================================================
# ENTRADA
# =========================================================
pergunta = st.text_area(
    "Digite sua ordem:",
    height=140,
    placeholder="O que você precisa resolver hoje?"
)

# =========================================================
# EXECUÇÃO
# =========================================================
if st.button("INICIAR"):
    if not pergunta.strip():
        st.warning("Digite uma pergunta.")
    else:
        with st.spinner("ROMANUS consultando toda a base..."):
            resultado = gerar_resposta(pergunta, modo_estrito=modo_estrito)

        if not resultado["ok"]:
            st.error("Erro ao gerar resposta.")
            st.code(resultado["erro"])
        else:
            st.markdown("### Resposta")
            st.markdown(
                f'<div class="bloco-resposta">{resultado["texto"]}</div>',
                unsafe_allow_html=True
            )

        if mostrar_debug:
            base_total = carregar_base_local()
            indice_total = indexar_base_em_chunks()
            arquivos_usados = [t["arquivo"] for t in resultado.get("trechos", [])]
            referencias = [t["referencia"] for t in resultado.get("trechos", []) if t.get("referencia")]

            debug_texto = (
                f"Arquivos na base: {len(base_total)}\n"
                f"Chunks totais indexados: {len(indice_total)}\n"
                f"Trechos retornados para a resposta: {len(resultado.get('trechos', []))}\n"
                f"Arquivos usados: {arquivos_usados if arquivos_usados else 'Nenhum'}\n"
                f"Referências localizadas: {referencias if referencias else 'Nenhuma'}\n"
                f"Modelo: {MODELO_GEMINI}\n"
                f"Tempo de resposta: {resultado.get('tempo', 0)} s\n"
                f"Modo estrito: {'Ligado' if modo_estrito else 'Desligado'}"
            )

            st.markdown("### Diagnóstico")
            st.markdown(
                f'<div class="debug-box">{debug_texto}</div>',
                unsafe_allow_html=True
            )
