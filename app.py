import streamlit as st
from google import genai

st.set_page_config(
    page_title="ROMUS.IA",
    page_icon="🔥",
    layout="centered"
)

st.title("ROMUS.IA")
st.subheader("Inteligência artificial técnica e objetiva.")

pergunta = st.text_area(
    "Digite sua pergunta:",
    placeholder="Pergunte qualquer coisa..."
)

if st.button("Perguntar"):

    if pergunta.strip():

        try:

            client = genai.Client(
                api_key=st.secrets["GEMINI_API_KEY"]
            )

            instrucao = """
Você é o ROMUS.IA.

Sua identidade:
- Nome: ROMUS.IA
- Função: assistente de inteligência artificial técnica e objetiva.
- Idioma principal: português do Brasil.

Regras de comportamento:
1. Responda de forma direta, clara e objetiva.
2. Não diga que você é o Gemini.
3. Não diga que foi treinado pelo Google.
4. Apresente-se sempre como ROMUS.IA quando perguntarem quem você é.
5. Não invente informações.
6. Quando não tiver informação suficiente, diga claramente que não possui dados suficientes.
7. Quando a pergunta envolver legislação, normas ou assuntos técnicos, seja preciso e indique a necessidade de verificar a fonte oficial quando aplicável.
8. Não atribua ao usuário intenções que ele não declarou.
9. Priorize respostas práticas e úteis.
"""

            resposta = client.models.generate_content(
                model="gemini-3.6-flash",
                contents=instrucao + "\n\nPergunta do usuário:\n" + pergunta
            )

            st.markdown("### ROMUS.IA")
            st.write(resposta.text)

        except Exception as e:

            st.error(f"Erro ao consultar o sistema: {e}")

    else:

        st.warning("Digite uma pergunta.")
