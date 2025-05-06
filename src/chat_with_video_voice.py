# chat_with_video_voice.py

import os
from pinecone import Pinecone
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_pinecone import Pinecone as PineconeVectorStore
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

# ------------------------------------------------------------------------
# config: define embedding model and index name
# ------------------------------------------------------------------------
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
PINECONE_INDEX_NAME = "youtube-video-index"  # should be set externally in production
PINECONE_API_KEY = "<PINECONE_API_KEY>"      # replace with secure source (e.g., .env or secret manager)
OPENAI_API_KEY = "<OPENAI_API_KEY>"          # replace with secure source




# ------------------------------------------------------------------------
# feat: load Pinecone vector store with HuggingFace embeddings
# ------------------------------------------------------------------------
def load_vectorstore(namespace):
    pc = Pinecone(api_key=PINECONE_API_KEY)  # initialize Pinecone client
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)  # use specified embedding model

    vectordb = PineconeVectorStore(
        index_name=PINECONE_INDEX_NAME,
        embedding=embeddings,
        namespace=namespace,
        text_key="text"
    )
    return vectordb  # return ready-to-query vector store




# ------------------------------------------------------------------------
# feat: build a QA chain using LangChain's RetrievalQA wrapper
# ------------------------------------------------------------------------
def build_qa_chain(vectordb):
    llm = ChatOpenAI(
        model="gpt-3.5-turbo",
        temperature=0,
        openai_api_key=OPENAI_API_KEY
    )  # initialize LLM with deterministic response style

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectordb.as_retriever(search_kwargs={"k": 4}),  # retrieve top 4 relevant chunks
        return_source_documents=False
    )
    return qa_chain  # return fully initialized QA pipeline
