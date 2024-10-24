import traceback
import streamlit as st
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.runnables.history import RunnableWithMessageHistory
import faiss
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
os.environ["huggingface_api"] = os.getenv("huggingface_api")

st.secrets["groq_api_key"]
groq_api_key = os.getenv("groq_api")
# Initialize embeddings and LLM
llm = ChatGroq(groq_api_key=groq_api_key, model_name="Llama3-8b-8192")

def get_session(session_id: str) -> BaseChatMessageHistory:
    """Retrieve or create a session history."""
    if session_id not in st.session_state.store:
        st.session_state.store[session_id] = ChatMessageHistory()
    return st.session_state.store[session_id]

def process_uploaded_file(uploaded_file):
    """Process the uploaded PDF file and return documents."""
    temp_pdf_path = "./temp.pdf"
    with open(temp_pdf_path, "wb") as f:
        f.write(uploaded_file.getvalue())

    loader = PyPDFLoader(temp_pdf_path)
    return loader.load()

def create_chains(retriever):
    """Create and return the retrieval and question-answer chains."""
    prompt = (
        "You have past chat history and the current question from the user. "
        "Use the chat history to formulate a response to the current question."
        "Please provide your response. Just reformulate the answer with respect to the question."
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
        "You are an assistant in a chat room. Respond to the user's question using the provided chat history. "
        "If you don't know the answer, say 'I don't know'. The response should be within 20 lines. {context}"
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
    
    # Embed documents using OllamaEmbeddings
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    db = FAISS.from_documents(splits, embeddings)
    st.write("FAISS vectorstore initialized.")
    
    # Create retriever
    retriever = db.as_retriever()

    # Create the chains using the retriever
    conversational_rag_chain = create_chains(retriever)
    
    # Session ID input
    session_id = st.text_input("Session ID", value="default_session")

    # User input for the question
    user_input = st.text_input("Write your question here")
    if user_input:
        try:
            # Retrieve session history
            session_history = get_session(session_id)
            st.write("Session history:", session_history)

            # Retrieve most relevant documents using similarity_search_with_score
            docs_with_scores = db.similarity_search_with_score(user_input, k=5)
            
            # Display the retrieved documents and their scores
            for i, (doc, score) in enumerate(docs_with_scores):
                st.write(f"Document {i + 1}: (Score: {score:.4f})")
                st.write(doc.page_content[:300] + "...")  # Displaying only part of the content

            # Invoke the chain with correct config
            response = conversational_rag_chain.invoke(
                {"input": user_input},
                config={"configurable": {"session_id": session_id}}  # Passing session_id correctly
            )

            # Ensure 'response' is a dict with an 'answer' key
            if isinstance(response, dict) and "answer" in response:
                st.success("Assistant: " + response["answer"])
            else:
                st.error("Unexpected response format.")
        except Exception as e:
            st.error("An error occurred.")
            st.write(f"Error details: {e}")
            st.write(traceback.format_exc())  # Print detailed traceback
    else:
        st.warning("Please enter your question.")
else:
    st.info("Please upload a PDF file to start.")
