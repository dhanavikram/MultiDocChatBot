import os
from dotenv import load_dotenv
import tempfile

import streamlit as st
from streamlit_chat import message
from langchain.chains import ConversationalRetrievalChain
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import LlamaCpp
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader


load_dotenv()

def initialize_session_state():
    if "history" not in st.session_state:
        st.session_state["history"] = []
    if "generated" not in st.session_state:
        st.session_state['generated'] = ["Hi there! I am  here to help you with your documents"]
    if 'past' not in st.session_state:
        st.session_state['past'] = ["Hello!"]


def conversation_chat(query, chain, history):
    result = chain({"question": query, "chat_history": history})
    history.append((query, result['answer']))
    return result['answer']


def display_chat_history(chain):
    reply_container = st.container()
    container = st.container()

    with container:

        with st.form(key="my_form", clear_on_submit=True):
            user_input = st.text_input("Question:", placeholder="Enter your question here", key="input")
            submit_button = st.form_submit_button(label='send')

        if submit_button and user_input:
            with st.spinner("Generating response.."):
                output = conversation_chat(user_input, chain, st.session_state['history'])

                st.session_state['past'].append(user_input)
                st.session_state['generated'].append(output)

    if st.session_state['generated']:
        with reply_container:
            for i in range(len(st.session_state['generated'])):
                message(st.session_state['past'][i], is_user=True, key=str(i) + "_user",
                        avatar_style='thumbs')
                message(st.session_state['generated'][i], key=str(i), avatar_style='fun-emoji')


def create_conversational_chain(vector_store):
    load_dotenv()

    # Create the LLM
    llm = LlamaCpp(
        streaming=True,
        model_path="models/mistral-7b-instruct-v0.1.Q3_K_S.gguf",
        temperature=0.75,
        top_p=1,
        verbose=True,
        n_ctx=4096)

    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    chain = ConversationalRetrievalChain.from_llm(llm=llm, chain_type="stuff",
                                                  retriever=vector_store.as_retriever(search_kwargs={"k":2}))

    return chain


def main():

    initialize_session_state()
    st.title("Chat with your docs")
    st.sidebar.title("Upload your docs")
    files = st.sidebar.file_uploader("Upload Files", accept_multiple_files=True)

    if files:
        text = []
        for file in files:
            file_extension = os.path.splitext(file.name)[1]
            with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                temp_file.write(file.read())
                temp_file_path = temp_file.name
            loader = None
            if file_extension == '.pdf':
                loader = PyPDFLoader(temp_file_path)
            elif file_extension == '.docx' or file_extension == '.doc':
                loader = Docx2txtLoader(temp_file_path)
            elif file_extension == '.txt':
                loader = TextLoader(temp_file_path)

            if loader:
                text.extend(loader.load())
                os.remove(temp_file_path)

            text_splitter = CharacterTextSplitter(separator='\n', chunk_size=1000,
                                                  chunk_overlap=100, length_function=len)
            text_chunks = text_splitter.split_documents(text)

            # Create Embeddings
            embeddings = HuggingFaceEmbeddings(model_name = 'sentence-transformers/all-MiniLM-L6-v2',
                                               model_kwargs = {"device": "cpu"})

            # Create a vector store
            vector_store = FAISS.from_documents(text_chunks, embedding=embeddings)

            # Chain object
            chain = create_conversational_chain(vector_store)
            display_chat_history(chain)


if __name__=='__main__':
    main()
