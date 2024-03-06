import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

GOOGLE_API_KEY = "Your API Key"

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks


def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")


def get_conversational_chain():
    prompt_template ="""You are a Q&A assistant. Your goal is to answer questions as
        accurately as possible based on the instructions and context provided.\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro",
                                   temperature=0.3)

    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain


def main():
    st.set_page_config("Chat PDF")
    st.header("Chat with PDFs")
    user_question = st.chat_input("Pass your Prompt here")
    if user_question:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

        new_db = FAISS.load_local("faiss_index", embeddings)
        docs = new_db.similarity_search(user_question)

        chain = get_conversational_chain()

        response = chain(
            {"input_documents": docs, "question": user_question}
            , return_only_outputs=True)

        if 'messages' not in st.session_state:
            st.session_state.messages = []

        for message in st.session_state.messages:
            st.chat_message(message['role']).markdown(message['content'])

        if user_question:
            st.chat_message('user').markdown(user_question)
            st.session_state.messages.append({'role': 'user', 'content': user_question})
            st.chat_message('assistant').markdown(response['output_text'])
            st.session_state.messages.append({'role': 'assistant', 'content': response['output_text']})

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button",
                                    accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Done")


if __name__ == "__main__":
    main()
