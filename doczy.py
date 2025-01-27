import streamlit as st
import re

from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_ollama import OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM


template = """
You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
Question: {question} 
Context: {context} 
Answer:
"""


pdf_dir = './pdfs/'

embeddings = OllamaEmbeddings(model="deepseek-r1:7b")
vector_store = InMemoryVectorStore(embeddings)

model = OllamaLLM(model="deepseek-r1:7b")


def upload_pdf(file):
    with open(pdf_dir + file.name, "wb") as f:
        f.write(file.getbuffer())

def load_pdf(file_path):
    loader = PDFPlumberLoader(file_path)
    docs = loader.load()

    return docs


def split_doc(docs):
    txt_spliter = RecursiveCharacterTextSplitter(
        chunk_size = 1000,
        chunk_overlap = 200,
        add_start_index = True
    )

    return txt_spliter.split_documents(docs)


def index_doc(docs):
    vector_store.add_documents(docs)


def retireve_doc(query):
    return vector_store.similarity_search(query)


def answer_que(question, docs):
    context = "\n\n".join([doc.page_content for doc in docs])
    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | model

    return chain.invoke({"question": question, "context": context})


st.set_page_config(
    page_title="Doczy", 
    page_icon="📑"
)

st.title("Doczy 📑")
st.markdown("###### Made with DeepSeek-R1:7b & LangChain🦜🔗")
st.markdown("###### by nv33.n")


uploaded_file = st.file_uploader(
    "Unleash your PDF here",
    type="pdf",
    accept_multiple_files=False
)

if uploaded_file:
    upload_pdf(uploaded_file)
    documents = load_pdf(pdf_dir + uploaded_file.name)
    chunked_documents = split_doc(documents)
    index_doc(chunked_documents)

    question = st.chat_input()

    if question:
        
        st.chat_message("user").write(question)
        
        related_documents = retireve_doc(question)
        answer = answer_que(question, related_documents)

        st.chat_message("assistant").write(answer)