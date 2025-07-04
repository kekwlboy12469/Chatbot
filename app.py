import streamlit as st
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import fitz  
import io 


prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Use the provided context to answer."),
    ("user", "Context: {context}\n\nQuestion: {question}")
])
st.sidebar.header(" Settings")
model_choice = st.sidebar.selectbox("Choose model", ["llama3", "mistral", "gemma", "llama2"])
temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.7, step=0.1)
max_tokens = st.sidebar.slider("Max Tokens", 64, 2048, 512, step=64)

def extract_text_from_pdf(pdf_file):
    pdf_bytes = pdf_file.read()
    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        return "\n".join([page.get_text() for page in doc])

uploaded_file = st.sidebar.file_uploader(" Upload a PDF", type=["pdf"])
pdf_text = ""
if uploaded_file:
    try:
        pdf_text = extract_text_from_pdf(uploaded_file)
        st.sidebar.success(" PDF content loaded!")
    except Exception as e:
        st.sidebar.error(f" Failed to read PDF: {e}")


def generate_response(question,pdf_context="", engine="llama3", temperature=0.7,max_tokens=512):
    llm = ChatOllama(model=engine, temperature=temperature)
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"question": question, "context": pdf_context})


st.set_page_config(page_title=" Ollama Chatbot", layout="centered")
st.title(" Chatbot (Ollama + LangChain)")



user_input = st.text_input("Ask a question:")


if user_input:
    try:
        response = generate_response(user_input,  pdf_context=pdf_text,engine=model_choice,temperature=temperature,max_tokens=max_tokens)
        st.markdown(f"** Bot:** {response}")
    except Exception as e:
        st.error(f" Error: {e}")
if pdf_text:
    st.sidebar.text_area("PDF Preview", pdf_text[:500], height=150)

