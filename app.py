import streamlit as st
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory 
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.runnables.history import RunnableWithMessageHistory

import os
from dotenv import load_dotenv
load_dotenv()

os.environ["huggingface_api"] = os.getenv("huggingface_api")
groq_api = os.getenv("groq_api")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2") 
def get_session(session:str)->BaseChatMessageHistory:
    if session_id not in st.session_state.store:
        st.session_state.store[session_id] = ChatMessageHistory()
    return st.session_state.store[session_id]  

st.title("Conversational AI")
st.write("upload you PDF file to get started")

llm = ChatGroq(groq_api_key=groq_api,model_name="Llama3-8b-8192")
session_id = st.text_input("Session ID", value="default session")
if 'store' not in st.session_state:
    st.session_state.store = {}

uploaded_file = st.file_uploader("Choose a PDF file", type="pdf", accept_multiple_files=False) 
if uploaded_file:
    documents = []
    if uploaded_file is not None:
        temppdf = f"./temp.pdf"
        with open(temppdf, "wb") as f:
            f.write(uploaded_file.getvalue())
            file_name = uploaded_file.name

        loader = PyPDFLoader(temppdf)
        docs = loader.load()
        
        documents.extend(docs)
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=500)
    splits = text_splitter.split_documents(documents)
    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings) 
    retriever = vectorstore.as_retriever()
    prompt = (
        "Here you have past chat history and current question from the user"
        "Current question may have context from past chat history"
        "Please provide your response"
        "Just reformulate the answer with respect to the question"
    )
    q_and_a_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    history_aware_retriever = create_history_aware_retriever(llm, retriever,q_and_a_prompt)

    system_prompt = (
        "You are an assistant in a chat room. You have to respond to the user's question."
        "Use the following chat history to generate a response."
        "If you dont know the answer just say 'I dont know'"
        "Answer should be in the context of the question and also maximum 20 lines"
        "{context}"
    )

    qanda_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    question_answer_chain = create_stuff_documents_chain(llm, qanda_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)



    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain, get_session,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer"
    )


user_input = st.text_input("Write your question here")
if user_input:
    session_history = get_session(session_id)
    response = conversational_rag_chain.invoke(
           {"input":user_input},
            config={
                "configurable" : {
                    "session_id": session_id
                }
            }
    )  
    st.success("Assistant: "+response["answer"])
else:
    st.warning("Please enter your question")