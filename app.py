import streamlit as st
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.vectorstores import Chroma
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

# Load environment variables
load_dotenv()
os.environ["huggingface_api"] = os.getenv("huggingface_api")
groq_api = os.getenv("groq_api")

# Initialize embeddings and LLM
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
llm = ChatGroq(groq_api_key=groq_api, model_name="Llama3-8b-8192")

def get_session(session_id: str) -> BaseChatMessageHistory:
    """Retrieve or create a session history."""
    if session_id not in st.session_state.store:
        st.session_state.store[session_id] = ChatMessageHistory()
    return st.session_state.store[session_id]

def process_uploaded_file(uploaded_file):
    """Process the uploaded PDF file and return documents."""
    temppdf = f"./temp.pdf"
    with open(temppdf, "wb") as f:
        f.write(uploaded_file.getvalue())
    
    # Ensure the file is properly handled and cleaned up after processing
    loader = PyPDFLoader(temppdf)
    documents = loader.load()
    
    # Remove the temporary file
    os.remove(temppdf)
    return documents

def create_chains(retriever):
    """Create and return the retrieval and question-answer chains."""
    prompt = (
        "Here you have past chat history and the current question from the user. "
        "The current question may have context from past chat history. "
        "Please provide your response, reformulated with respect to the question."
    )
    q_and_a_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    history_aware_retriever = create_history_aware_retriever(llm, retriever, q_and_a_prompt)

    system_prompt = (
        "You are an assistant in a chat room. You must respond to the user's question. "
        "Use the following chat history to generate a response. "
        "If you don't know the answer, just say 'I don't know'. "
        "Answer should be in the context of the question and no more than 20 lines. {context}"
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

    return RunnableWithMessageHistory(
        rag_chain, get_session,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer"
    )

# Streamlit UI
st.title("Conversational AI")
st.write("Upload your PDF file to get started")

# Session state initialization
if 'store' not in st.session_state:
    st.session_state.store = {}

# File uploader
uploaded_file = st.file_uploader("Choose a PDF file", type="pdf", accept_multiple_files=False)
if uploaded_file:
    documents = process_uploaded_file(uploaded_file)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=500)
    splits = text_splitter.split_documents(documents)
    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
    retriever = vectorstore.as_retriever()

    conversational_rag_chain = create_chains(retriever)

    session_id = st.text_input("Session ID", value="default session")

    user_input = st.text_input("Write your question here")
    if user_input:
        session_history = get_session(session_id)
        response = conversational_rag_chain.invoke(
            {"input": user_input},
            config={"configurable": {"session_id": session_id}}
        )
        st.success("Assistant: " + response["answer"])
    else:
        st.warning("Please enter your question")
else:
    st.info("Please upload a PDF file to start.")
