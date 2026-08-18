import re
from pathlib import Path

import streamlit as st
from google import genai
from google.genai import types
from pypdf import PdfReader

st.set_page_config(
    page_title="ROMUS.IA",
    page_icon="🔥",
    layout="centered",
)

BASE_DIR = Path(__file__).parent
KNOWLEDGE_DIR = BASE_DIR / "base_conhecimento"
MODEL = "gemini-3.6-flash"
MAX_CHUNKS = 5
MIN_SCORE = 2

SYSTEM_PROMPT = """
Você é o ROMUS.IA.

IDENTIDADE
- Nome: ROMUS.IA.
- Função: assistente de inteligência artificial técnica e objetiva.
- Idioma principal: português do Brasil.

REGRAS GERAIS
1. Responda de forma direta, clara e objetiva.
2. Não diga que você é o Gemini e não atribua sua identidade ao Google.
3. Não invente informações.
4. Não atribua ao usuário intenções que ele não declarou.
5. Quando a resposta vier da base local, use SOMENTE o conteúdo fornecido da base como fundamento factual. Não complete lacunas com conhecimento próprio.
6. Quando a resposta vier da pesquisa na internet, deixe claro que foi realizada pesquisa externa e priorize fontes oficiais e confiáveis.
7. Em legislação, normas e assuntos técnicos, seja rigoroso com datas, números, artigos, itens e redação.
8. Se a informação não puder ser confirmada, diga isso claramente.
"""


def normalizar(texto: str) -> list[str]:
    texto = texto.lower()
    texto = re.sub(r"[^\w\s]", " ", texto, flags=re.UNICODE)
    return [p for p in texto.split() if len(p) > 2]


def ler_arquivo(caminho: Path) -> str:
    if caminho.suffix.lower() in {".txt", ".md"}:
        return caminho.read_text(encoding="utf-8", errors="ignore")

    if caminho.suffix.lower() == ".pdf":
        reader = PdfReader(str(caminho))
        paginas = []
        for pagina in reader.pages:
            paginas.append(pagina.extract_text() or "")
        return "\n".join(paginas)

    return ""


def carregar_documentos() -> list[dict]:
    KNOWLEDGE_DIR.mkdir(parents=True, exist_ok=True)
    documentos = []

    for caminho in KNOWLEDGE_DIR.rglob("*"):
        if caminho.is_file() and caminho.suffix.lower() in {".txt", ".md", ".pdf"}:
            try:
                texto = ler_arquivo(caminho).strip()
                if texto:
                    documentos.append({"arquivo": caminho.relative_to(KNOWLEDGE_DIR).as_posix(), "texto": texto})
            except Exception:
                continue

    return documentos


def dividir_em_blocos(texto: str, tamanho: int = 1800, sobreposicao: int = 250) -> list[str]:
    texto = re.sub(r"\n{3,}", "\n\n", texto).strip()
    if len(texto) <= tamanho:
        return [texto]

    blocos = []
    inicio = 0
    while inicio < len(texto):
        fim = min(inicio + tamanho, len(texto))
        bloco = texto[inicio:fim].strip()
        if bloco:
            blocos.append(bloco)
        if fim >= len(texto):
            break
        inicio = max(fim - sobreposicao, inicio + 1)
    return blocos


@st.cache_data(show_spinner=False)
def construir_indice() -> list[dict]:
    indice = []
    for documento in carregar_documentos():
        for bloco in dividir_em_blocos(documento["texto"]):
            indice.append(
                {
                    "arquivo": documento["arquivo"],
                    "texto": bloco,
                    "termos": set(normalizar(bloco)),
                }
            )
    return indice


def buscar_base(pergunta: str) -> list[dict]:
    termos = set(normalizar(pergunta))
    if not termos:
        return []

    resultados = []
    for item in construir_indice():
        intersecao = termos.intersection(item["termos"])
        score = len(intersecao)
        if score >= MIN_SCORE:
            resultados.append({**item, "score": score})

    resultados.sort(key=lambda x: x["score"], reverse=True)
    return resultados[:MAX_CHUNKS]


def gerar_resposta_local(client, pergunta: str, resultados: list[dict]):
    contexto = "\n\n".join(
        f"[FONTE LOCAL: {item['arquivo']}]\n{item['texto']}"
        for item in resultados
    )

    prompt = f"""
{SYSTEM_PROMPT}

MODO: BASE LOCAL.

Responda exclusivamente com base no conteúdo abaixo.
Se o conteúdo não permitir responder, diga: "A base local não contém informação suficiente para responder com segurança."
Não use pesquisa na internet neste modo.

CONTEÚDO DA BASE:
{contexto}

PERGUNTA DO USUÁRIO:
{pergunta}
"""

    return client.models.generate_content(
        model=MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(temperature=0.1),
    )


def gerar_resposta_web(client, pergunta: str):
    prompt = f"""
{SYSTEM_PROMPT}

MODO: PESQUISA NA INTERNET.

A base local não apresentou conteúdo suficiente para responder.
Pesquise na web antes de responder.
Priorize fontes oficiais, legislação oficial, órgãos públicos, fabricantes, universidades e documentação técnica primária.
Quando houver conflito entre fontes, informe o conflito e priorize a fonte oficial/primária.
Apresente as fontes utilizadas ao final quando elas estiverem disponíveis.

PERGUNTA DO USUÁRIO:
{pergunta}
"""

    ferramenta_pesquisa = types.Tool(google_search=types.GoogleSearch())
    return client.models.generate_content(
        model=MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            tools=[ferramenta_pesquisa],
            temperature=0.1,
        ),
    )


def mostrar_fontes_web(resposta):
    try:
        metadata = resposta.candidates[0].grounding_metadata
        fontes = getattr(metadata, "grounding_chunks", None) or []
        urls = []
        for fonte in fontes:
            web = getattr(fonte, "web", None)
            if web and getattr(web, "uri", None):
                titulo = getattr(web, "title", None) or web.uri
                if web.uri not in [u[1] for u in urls]:
                    urls.append((titulo, web.uri))

        if urls:
            st.markdown("### Fontes consultadas")
            for titulo, url in urls[:8]:
                st.markdown(f"- [{titulo}]({url})")
    except Exception:
        pass


st.title("ROMUS.IA")
st.subheader("Inteligência artificial técnica e objetiva.")

pergunta = st.text_area(
    "Digite sua pergunta:",
    placeholder="Pergunte qualquer coisa...",
    height=120,
)

col1, col2 = st.columns(2)
with col1:
    pesquisar_web_se_necessario = st.checkbox("Pesquisar na web se a base não responder", value=True)
with col2:
    if st.button("Recarregar base"):
        st.cache_data.clear()
        st.rerun()

if st.button("Perguntar", type="primary"):
    if not pergunta.strip():
        st.warning("Digite uma pergunta.")
    else:
        try:
            client = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])
            resultados = buscar_base(pergunta)

            if resultados:
                st.caption("Fonte: base local do ROMUS.IA")
                resposta = gerar_resposta_local(client, pergunta, resultados)
                st.markdown("### ROMUS.IA")
                st.write(resposta.text)

                with st.expander("Documentos encontrados na base"):
                    for item in resultados:
                        st.write(f"**{item['arquivo']}** — relevância {item['score']}")

            elif pesquisar_web_se_necessario:
                st.caption("Fonte: pesquisa na internet")
                resposta = gerar_resposta_web(client, pergunta)
                st.markdown("### ROMUS.IA")
                st.write(resposta.text)
                mostrar_fontes_web(resposta)

            else:
                st.info("A base local não contém informação suficiente para responder. A pesquisa na web está desativada.")

        except Exception as e:
            st.error(f"Erro ao consultar o sistema: {e}")
