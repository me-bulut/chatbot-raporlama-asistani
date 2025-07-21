import streamlit as st
from features.gemini_query import get_gemini_response

st.title("Chatbot Raporlama Asistanı")

user_input = st.text_input("Bir soru yazın:")

if user_input:
    cevap = get_gemini_response(user_input)
    st.write("Cevap:")
    st.write(cevap)