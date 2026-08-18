import streamlit as st
from google import genai
from google.genai import types
import os

# ==============================================================================
# 1. CONFIGURAÇÃO DA PÁGINA STREAMLIT
# ==============================================================================
st.set_page_config(
    page_title="ROMUS.IA",
    page_icon="🔥",
    layout="wide"
)

st.title("ROMUS.IA")
st.caption("Inteligência artificial técnica e objetiva.")

# ==============================================================================
# 2. CONFIGURAÇÃO DA API DO GEMINI
# ==============================================================================
api_key = os.getenv("GEMINI_API_KEY") or st.secrets.get("GEMINI_API_KEY", "")

if not api_key:
    st.error("Chave de API (GEMINI_API_KEY) não configurada. Configure em Secrets do Streamlit ou Variáveis de Ambiente.")
    st.stop()

# Inicializa o cliente oficial da nova SDK
client = genai.Client(api_key=api_key)

# ==============================================================================
# 3. PROMPT DO SISTEMA (ROMUS.IA)
# ==============================================================================
SYSTEM_PROMPT = """Você é o assistente técnico especializado ROMUS.IA, voltado para engenharia de segurança contra incêndio e normas técnicas de edificações.

INSTRUÇÕES DE RESPOSTA:
1. Analise cuidadosamente o contexto e as evidências fornecidas.
2. Seja direto, técnico, objetivo e preciso em suas respostas.
3. Se a pergunta envolver cálculos (ex: Unidades de Passagem - UP, largura mínima de saída, dimensionamento de lotação), apresente o cálculo passo a passo, a fórmula utilizada e os valores finais.
4. Responda fundamentando com base no texto das evidências encontradas.
5. Se o texto da evidência for suficiente para responder parcialmente ou totalmente, gere a resposta completa e detalhada sem emitir mensagens de erro ou de recusa.
6. Nunca decline a resposta se houver dados mínimos no contexto ou se a opção de busca complementar estiver habilitada.
"""

# ==============================================================================
# 4. FUNÇÃO DE BUSCA NA BASE DE DADOS
# ==============================================================================
def buscar_na_base_dados(pergunta):
    evidencias_exemplo = [
        "Instrução Técnica / Norma de Saídas de Emergência em Edificações:",
        "- Grupo F-11: Locais de reunião de público / centros de convenções / recintos de exposições.",
        "- Cálculo de Unidades de Passagem (N): N = P / C, onde P é a população e C é a capacidade do acesso/escada.",
        "- Para saídas e acessos em edifícios do Grupo F: Capacidade de passagem C = 100 pessoas por unidade de passagem (UP).",
        "- Largura da saída: N x 0,55m. A largura mínima recomendada para acessos/portas em saídas de emergência é de 1,10m (2 UPs)."
    ]
    return "\n".join(evidencias_exemplo)

# ==============================================================================
# 5. FUNÇÃO DE GERAÇÃO DE RESPOSTA (MODELO GEMINI-3.6-FLASH)
# ==============================================================================
def gerar_resposta_romus(pergunta, evidencias, pesquisar_web):
    contexto_str = f"EVIDÊNCIAS ENCONTRADAS NA BASE DE DADOS:\n{evidencias}\n\n"
    if pesquisar_web:
        contexto_str += "NOTA: Se as evidências acima forem parciais, utilize também seu conhecimento técnico geral para complementar a resposta com precisão.\n\n"

    prompt_final = f"{contexto_str}PERGUNTA DO USUÁRIO:\n{pergunta}"

    try:
        # Chamada atualizada com o modelo gemini-3.6-flash
        response = client.models.generate_content(
            model="gemini-3.6-flash",
            contents=prompt_final,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                temperature=0.2,
            ),
        )
        if response.text and response.text.strip():
            return response.text
        else:
            return "O mecanismo processou o pedido, mas a resposta retornou vazia. Tente reformular a pergunta."
    except Exception as e:
        return f"Erro na comunicação com a IA: {str(e)}"

# ==============================================================================
# 6. INTERFACE DE USUÁRIO (STREAMLIT UI)
# ==============================================================================
pergunta = st.text_area(
    "Digite sua pergunta:",
    placeholder="Ex: Para uma edificação com população de 100 pessoas, quantas unidades de passagem são necessárias e qual deve ser a largura da saída de emergência? Grupo F-11",
    height=120
)

pesquisar_web = st.checkbox("Pesquisar na web se a base não responder", value=True)

col1, col2 = st.columns([1, 4])
with col1:
    btn_perguntar = st.button("Perguntar", type="primary", use_container_width=True)
with col2:
    btn_recarregar = st.button("Recarregar base", use_container_width=True)

if btn_recarregar:
    st.cache_data.clear()
    st.success("Base recarregada com sucesso!")

if btn_perguntar and pergunta:
    with st.spinner("Buscando evidências e gerando diagnóstico técnico..."):
        evidencias = buscar_na_base_dados(pergunta)
        resposta = gerar_resposta_romus(pergunta, evidencias, pesquisar_web)

        st.markdown("### ROMUS.IA")
        
        with st.expander("Diagnóstico técnico", expanded=True):
            st.write(resposta)

        with st.expander("Documentos encontrados na base", expanded=False):
            st.code(evidencias, language="text")
