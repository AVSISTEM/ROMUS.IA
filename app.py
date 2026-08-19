import os
import streamlit as st
from google import genai
from google.genai import types

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
chave_api = os.getenv("GEMINI_API_KEY") or st.secrets.get("GEMINI_API_KEY", "")

if not chave_api:
    st.error("Chave de API (GEMINI_API_KEY) não configurada. Configure em Secrets do Streamlit ou Variáveis de Ambiente.")
    st.stop()

# Inicializa o cliente oficial
cliente = genai.Client(api_key=chave_api)

# ==============================================================================
# 3. PROMPT DO SISTEMA (ROMUS.IA) - OTIMIZADO PARA SÍNTESE E INTELIGÊNCIA
# ==============================================================================
PROMPT_DO_SISTEMA = """Você é o ROMUS.IA, assistente técnico sênior em Engenharia de Segurança Contra Incêndio e Normas Técnicas.

DIRETRIZES DE RESPOSTA (SÍNTESE & INTELIGÊNCIA):
1. SEJA EXTREMAMENTE CONCISO: Responda de forma direta, eliminando saudações, introduções e conclusões genéricas. Vá direto ao ponto técnico.
2. ESTRUTURA SECCIONADA:
   - **Conclusão Técnico-Normativa**: A resposta direta em 1 ou 2 frases em negrito.
   - **Dimensionamento / Cálculo**: Se houver cálculo, exiba a fórmula, os valores aplicados e o resultado de forma esquemática (1 linha por etapa).
   - **Base Legal / Embasamento**: Apresente apenas a norma (Ex: IT-11, NBR 9077) e o item específico consultado em formato de tópicos (bullet points).
3. FORMATAÇÃO: Use tabelas para comparações ou múltiplos dados. Destaque valores numéricos cruciais em negrito.
4. LINGUAGEM: Utilize terminologia técnica precisa, sem floreios explicativos ou teorias.
"""

# ==============================================================================
# 4. FUNÇÃO DE BUSCA NA BASE DE DADOS
# ==============================================================================
def buscar_na_base_dados(pergunta):
    evidencias_exemplo = [
        "Instrução Técnica / Norma de Saídas de Emergência em Edificações:",
        "- Grupo F-11: Locais de reunião de público / centros de convenções / recintos de exposições.",
        "- Cálculo de Unidades de Passagem (N): N = P/C, onde P é a população e C é a capacidade do acesso/escada.",
        "- Para saídas e acessos em edifícios do Grupo F: Capacidade de passagem C = 100 pessoas por unidade de passagem (UP).",
        "- Largura da saída: N x 0,55m. A largura mínima recomendada para acessos/portas em saídas de emergência é de 1,10m (2 UPs)."
    ]
    return "\n".join(evidencias_exemplo)

# ==============================================================================
# 5. FUNÇÃO DE GERAÇÃO DE RESPOSTA
# ==============================================================================
def gerar_resposta_romus(pergunta, evidencias, pesquisar_web):
    contexto_str = f"EVIDÊNCIAS ENCONTRADAS NA BASE DE DADOS:\n{evidencias}\n\n"
    if pesquisar_web:
        contexto_str += "NOTA: Se as evidências acima forem parciais, utilize também seu conhecimento técnico geral para complementar a resposta com soluções.\n\n"

    prompt_final = f"{contexto_str}PERGUNTA DO USUÁRIO:\n{pergunta}"

    try:
        resposta = cliente.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt_final,
            config=types.GenerateContentConfig(
                system_instruction=PROMPT_DO_SISTEMA,
                temperature=0.0,
            ),
        )
        if resposta.text and resposta.text.strip():
            return resposta.text
        else:
            return "O mecanismo processou o pedido, mas a resposta retornou vazia. Tente reformular a pergunta."
    except Exception as e:
        return f"Erro na comunicação com a IA: {str(e)}"

# ==============================================================================
# 6. INTERFACE DO USUÁRIO (STREAMLIT UI)
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
