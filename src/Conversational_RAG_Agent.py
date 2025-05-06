from pinecone import Pinecone
from langchain_community.vectorstores import Pinecone as PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.chat_models import ChatOpenAI
from langchain.agents import initialize_agent, AgentType
from keywords_tool import create_keywords_tool
from quiz_tool import create_quiz_tool
import os




# ------------------------------------------------------------------------
# config: define Pinecone credentials and embedding model
# ------------------------------------------------------------------------
PINECONE_API_KEY = "<PINECONE_API_KEY>"  # Replace with secure source (e.g. env or secret manager)
PINECONE_ENV = "gcp-starter"
PINECONE_INDEX_NAME = "youtube-video-index"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"




# ------------------------------------------------------------------------
# config: load active namespace from file
# ------------------------------------------------------------------------
with open("current_namespace.txt", "r") as f:
    namespace = f.read().strip()  # This defines the scope for vector search




# ------------------------------------------------------------------------
# feat: connect to Pinecone index and prepare retriever
# ------------------------------------------------------------------------
pc = Pinecone(api_key=PINECONE_API_KEY)  # Initialize Pinecone client
index = pc.Index(PINECONE_INDEX_NAME)  # Connect to target index
embedding = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)  # Load embedding model
vectorstore = PineconeVectorStore(index, embedding, text_key="text", namespace=namespace)
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})  # Create retriever from vectorstore




# ------------------------------------------------------------------------
# feat: initialize tools (quiz + keyword extraction)
# ------------------------------------------------------------------------
quiz_tool = create_quiz_tool()
keywords_tool = create_keywords_tool(retriever)
tools = [quiz_tool, keywords_tool]




# ------------------------------------------------------------------------
# feat: initialize agent with tools and LLM
# ------------------------------------------------------------------------
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
agent = initialize_agent(tools, llm, agent=AgentType.OPENAI_FUNCTIONS, verbose=True)




# ------------------------------------------------------------------------
# cli: run interactive console agent
# ------------------------------------------------------------------------
def run_agent_console():
    print("Agent ready! Type your question or 'exit' to quit.")
    while True:
        query = input("\nYou: ").strip()
        if query.lower() in {"exit", "quit"}:
            break
        answer = agent.run(query)
        print(f"\nAgent: {answer}")

if __name__ == "__main__":
    run_agent_console()
