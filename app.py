import streamlit as st

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
        st.write("ROMUS.IA recebeu sua pergunta:")
        st.write(pergunta)
    else:
        st.warning("Digite uma pergunta.")
