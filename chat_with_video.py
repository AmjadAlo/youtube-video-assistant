import os
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.chat_models import ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.memory import ConversationBufferWindowMemory
from langsmith import traceable


# ------------------------------------------------------------------------
# config: set required API keys and project settings using environment variables
# ------------------------------------------------------------------------
os.environ["OPENAI_API_KEY"] = "<OPENAI_API_KEY>"  # OpenAI for LLM responses
os.environ["PINECONE_API_KEY"] = "<PINECONE_API_KEY>"  # Pinecone for vector storage
os.environ["PINECONE_INDEX_NAME"] = "youtube-video-index"  # Index name for Pinecone
os.environ["LANGCHAIN_TRACING_V2"] = "true"  # Enable LangSmith tracing
os.environ["LANGCHAIN_API_KEY"] = "<LANGCHAIN_API_KEY>"  # API key for LangSmith
os.environ["LANGCHAIN_PROJECT"] = "pr-grumpy-simple-26"  # LangSmith project ID

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"  # Pretrained embedding model
PINECONE_INDEX_NAME = os.environ["PINECONE_INDEX_NAME"]  # Dynamic index retrieval




# ------------------------------------------------------------------------
# feat: load Pinecone vector store and prepare embedding model
# ------------------------------------------------------------------------
def load_vectorstore(namespace):
    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])  # Initialize Pinecone client
    index = pc.Index(PINECONE_INDEX_NAME)  # Connect to specified Pinecone index
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)  # Load embedding model

    vectordb = PineconeVectorStore(
        index=index,
        embedding=embeddings,
        namespace=namespace,
        text_key="text"
    )
    return vectordb  # Return ready-to-use vector store




# ------------------------------------------------------------------------
# feat: build LangChain-based QA system with retriever and memory
# ------------------------------------------------------------------------
def build_qa_chain(vectordb):
    retriever = vectordb.as_retriever(search_kwargs={"k": 4})  # Define retriever with top-4 matches

    # Prompt structure to ensure strict use of transcript context
    system_msg = (
        "You are a helpful assistant. Only answer questions using the video transcript provided. "
        "If the answer is not clearly found in the context, say: 'Sorry, I don’t know. That’s not part of the video content.'"
    )

    # Define structured prompt with system and user message formats
    prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(system_msg),
        HumanMessagePromptTemplate.from_template("Context:\n{context}\n\nQuestion:\n{question}")
    ])

    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)  # Load deterministic LLM
    memory = ConversationBufferWindowMemory(k=3, return_messages=True)  # Retain last 3 interactions
    document_chain = create_stuff_documents_chain(llm=llm, prompt=prompt)  # Create chain with prompt and LLM

    # chore: enable LangSmith tracking for the QA function
    @traceable(name="qa_chain")
    def qa_chain(inputs):
        docs = retriever.invoke(inputs["query"])  # Retrieve relevant chunks
        return document_chain.invoke({
            "question": inputs["query"],
            "context": docs,
            "chat_history": memory.load_memory_variables({})["history"]
        })

    return qa_chain  # Return callable QA chain function




# ------------------------------------------------------------------------
# test: basic pipeline execution for QA with test namespace
# ------------------------------------------------------------------------
if __name__ == "__main__":
    vectorstore = load_vectorstore("test_namespace")  # Load data under test namespace
    chain = build_qa_chain(vectorstore)  # Construct QA chain
    print(chain({"query": "What is the video about?"}))  # Run test query
