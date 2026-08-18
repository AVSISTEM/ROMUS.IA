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

            resposta = client.models.generate_content(
                model="gemini-3.6-flash",
                contents=pergunta
            )

            st.markdown("### ROMUS.IA")
            st.write(resposta.text)

        except Exception as e:
            st.error(f"Erro ao consultar o Gemini: {e}")

    else:
        st.warning("Digite uma pergunta.")
