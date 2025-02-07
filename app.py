import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader 
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS 
import os 
import google.generativeai as genai
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain 
from htmlTemplates import css, bot_template, user_template

load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def get_pdf_text(pdf_docs): 
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            if page.extract_text():
                text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator = "\n",
        chunk_size = 1000,
        chunk_overlap = 200,
        length_function = len
    )    
    chunks = text_splitter.split_text(text)
    return chunks 

def get_vectorstore(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=os.getenv("GOOGLE_API_KEY"))
    vectorstore = FAISS.from_texts(text_chunks, embedding=embeddings)
    vectorstore.save_local("vectorstore")
    return vectorstore 
    
def get_conversation_chain(vectorstore):
    llm = ChatGoogleGenerativeAI(model = "gemini-2.0-flash", temperature=1)
    memory = ConversationBufferMemory(
        memory_key="chat_history", 
        return_messages=True
    )
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain

def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history.append((user_question, response["answer"]))
        
def main():
    load_dotenv()
    st.set_page_config(page_title="PDF Talk", page_icon=":books:")
    
    st.write(css, unsafe_allow_html=True)
    
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    st.header("PDF Talk :books:")
    user_question = st.text_input("Ask a question about your documents:")
    if user_question:
        if st.session_state.conversation is None:
            st.error("Please upload your PDFs and click on 'Process' first.")
        else:
            handle_userinput(user_question) 
    
    for message in reversed(st.session_state.chat_history):
        user_msg, bot_msg = message
        st.write(user_template.replace("{{MSG}}", user_msg), unsafe_allow_html=True)
        st.markdown(bot_template.replace("{{MSG}}", f"\n{bot_msg}\n"), unsafe_allow_html=True)
    
    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader("Upload your PDFs and click on 'Process':", accept_multiple_files=True)   
         
        if st.button("Process"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs) 

                if raw_text.strip():
                    text_chunks = get_text_chunks(raw_text)
                    vectorstore = get_vectorstore(text_chunks)
                    st.session_state.conversation = get_conversation_chain(vectorstore)
                    st.success("PDF Processed!")
                else:
                    st.error("No text extracted from PDF. Please check your files.")
                                    
if __name__ == '__main__':
    main()