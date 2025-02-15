import streamlit as st

import openai
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.embeddings import OllamaEmbeddings

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS

from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain

import os
from dotenv import load_dotenv
load_dotenv()

## Load Groq api key
groq_api_key = os.getenv('GROQ_API_KEY')

## LLM Model
llm = ChatGroq(groq_api_key=groq_api_key, model_name='gemma2-9b-it')
embeddings = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')

## Prompt Template
prompt = ChatPromptTemplate.from_template(
    '''
    Answer the questions based on the provided context only.
    Please provide the most accurate respone based on the question
    <context>
    {context}
    <context>
    Question:{input}
'''
)

def create_vector_embedding ():
    if 'vectors' not in st.session_state:
        st.session_state.embeddings = embeddings
        st.session_state.loader = PyPDFDirectoryLoader('./Documents')
        st.session_state.docs = st.session_state.loader.load()
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        st.session_state.split_docs = st.session_state.text_splitter.split_documents(st.session_state.docs)
        st.session_state.vectors = FAISS.from_documents(st.session_state.split_docs, st.session_state.embeddings)

st.title('RAG Document Q&A with Groq, Gemma & HuggingFace')

user_prompt = st.text_input('Enter your query from the research paper...')

if st.button('Document Embedding...'):
    create_vector_embedding()
    st.write('Vector Database is ready to query from :)')

import time

if user_prompt:
    # if 'vectors' in st.session_state:
    doc_chain = create_stuff_documents_chain(llm, prompt)
    retriever = st.session_state.vectors.as_retriever()
    retriever_chain = create_retrieval_chain(retriever, doc_chain)

    start = time.process_time()
    res = retriever_chain.invoke({'input': user_prompt})
    print(f'Response Time : {time.process_time() - start}')

    st.write(res['answer'])

    with st.expander('Document similarity search...'):
        for i,doc in enumerate(res['context']):
            st.write(doc.page_content)
            st.write('---------------------------')
