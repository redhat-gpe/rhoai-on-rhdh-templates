# importing dependencies
from dotenv import load_dotenv
import streamlit as st
import os,glob
from langchain.llms import VLLMOpenAI
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import faiss
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from htmlTemplates import css, bot_template, user_template

# creating custom template to guide llm model
custom_template = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question, in its original language.
Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:"""

CUSTOM_QUESTION_PROMPT = PromptTemplate.from_template(custom_template)
MODEL_NAME = os.getenv('MODEL_NAME')
INFERENCE_SERVER_URL = os.getenv('INFERENCE_SERVER_URL')
MAX_NEW_TOKENS = 512
TOP_P = 0.95
TEMPERATURE = 0.01
PRESENCE_PENALTY = 1.03
LOADED = False

def load_pdfs():
    text=""
    dirname = os.getcwd() + "/docs/*.pdf"
    print(dirname)
    for filename in glob.glob(dirname):
        print(filename)
        if (filename.endswith(".pdf")):
           pdf_file = open(filename, 'rb')
           pdf_reader = PdfReader(pdf_file)
           for page in pdf_reader.pages:
               text+=page.extract_text()
    text_chunks=get_chunks(text)
    vectorstore=get_vectorstore(text_chunks)
    st.session_state.conversation=get_conversationchain(vectorstore)

# extracting text from pdf
def get_pdf_text(docs):
    text=""
    for pdf in docs:
        pdf_reader=PdfReader(pdf)
        for page in pdf_reader.pages:
            text+=page.extract_text()
    return text

# converting text to chunks
def get_chunks(raw_text):
    text_splitter=CharacterTextSplitter(separator="\n",
                                        chunk_size=1000,
                                        chunk_overlap=200,
                                        length_function=len)
    chunks=text_splitter.split_text(raw_text)
    return chunks

# using all-MiniLm embeddings model and faiss to get vectorstore
def get_vectorstore(chunks):
    embeddings=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                     model_kwargs={'device':'cpu'})
    vectorstore=faiss.FAISS.from_texts(texts=chunks,embedding=embeddings)
    return vectorstore

# generating conversation chain
def get_conversationchain(vectorstore):
    llm = VLLMOpenAI(
        openai_api_key="EMPTY",
        openai_api_base= f"{INFERENCE_SERVER_URL}/v1",
        model_name=f"{MODEL_NAME}",
        max_tokens=MAX_NEW_TOKENS,
        top_p=TOP_P,
        temperature=TEMPERATURE,
        presence_penalty=PRESENCE_PENALTY,
        streaming=False,
        verbose=False,
    )
    memory = ConversationBufferMemory(memory_key='chat_history',
                                      return_messages=True,
                                      output_key='answer') # using conversation buffer memory to hold past information
    conversation_chain = ConversationalRetrievalChain.from_llm(
                                llm=llm,
                                retriever=vectorstore.as_retriever(),
                                condense_question_prompt=CUSTOM_QUESTION_PROMPT,
                                memory=memory)
    return conversation_chain

# generating response from user queries and displaying them accordingly
def handle_question(question):
    if st.session_state.conversation == None:
        st.warning("Please upload PDF's to chat with")
        return;
    response=st.session_state.conversation({'question': question})
    st.session_state.chat_history=response["chat_history"]
    for i,msg in enumerate(st.session_state.chat_history):
        if i%2==0:
            st.write(user_template.replace("{{MSG}}",msg.content,),unsafe_allow_html=True)
        else:
            st.write(bot_template.replace("{{MSG}}",msg.content),unsafe_allow_html=True)


def main():
    st.set_page_config(page_title="Chat with your PDF documentation",page_icon=":page_facing_up:")
    st.write(css,unsafe_allow_html=True)
    if "conversation" not in st.session_state:
        st.session_state.conversation=None

    if "chat_history" not in st.session_state:
        st.session_state.chat_history=None

    st.header("Chat with your PDF documentation :page_facing_up:")
    question=st.text_input("Ask a question from your document:")
    if question:
        handle_question(question)
    with st.sidebar:
        st.subheader("Your documents")
        docs=st.file_uploader("Upload your PDF here and click on 'Process'",accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing"):

                #get the pdf
                raw_text=get_pdf_text(docs)

                #get the text chunks
                text_chunks=get_chunks(raw_text)

                #create vectorstore
                vectorstore=get_vectorstore(text_chunks)

                #create conversation chain
                st.session_state.conversation=get_conversationchain(vectorstore)

if __name__ == '__main__':
    main()
