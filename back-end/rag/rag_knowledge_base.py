# rag_knowledge_base.py
import os
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI # Ensure this is imported

def initialize_rag_system():
    try:
        # Load all .md documents from the 'knowledge_base' directory and its subdirectories
        # by specifying glob="**/*.md"
        loader = DirectoryLoader(
            "knowledge_base",
            glob="**/*.md",  # This pattern recursively finds all .md files in subdirectories
            loader_cls=TextLoader, # Use TextLoader for .md files
            recursive=True # Ensure recursive loading
        )
        docs = loader.load()
        print(f"Loaded {len(docs)} documents from knowledge_base and its subfolders.")

        if not docs:
            print("No documents found in the 'knowledge_base' directory or its subfolders.")
            return None, None

        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_documents(docs)
        print(f"Split documents into {len(chunks)} chunks.")

        # Initialize embeddings model
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

        # Create a vector store
        # Use a persistent directory to store the ChromaDB
        persist_directory = './chroma_db'
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=persist_directory
        )
        print(f"Vector store created/loaded at {persist_directory}")

        # Initialize the LLM for RAG
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.5)

        # Create a retriever
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3}) # Retrieve top 3 relevant chunks

        # Define the RAG prompt template
        rag_prompt_template = """
        You are a kind, empathetic, and highly knowledgeable Holistic Metabolic Resilience Agent.
        Your goal is to provide supportive, evidence-based, and personalized guidance on diet, exercise,
        mental well-being, and lifestyle for individuals managing conditions like PCOS, Insulin Resistance,
        and Type 2 Diabetes.

        Use the following retrieved context to answer the user's question.
        If the context does not contain enough information, politely state that you cannot answer from the provided information
        but can offer general guidance or suggest they consult a healthcare professional.

        Always maintain a gentle, encouraging, and positive tone.
        ---
        Context: {context}
        ---
        Question: {input}
        """
        rag_prompt = ChatPromptTemplate.from_template(rag_prompt_template)

        # Create the RAG chain
        rag_chain = (
            {"context": retriever, "input": RunnablePassthrough()}
            | rag_prompt
            | llm
            | StrOutputParser()
        )

        print("RAG system initialized successfully.")
        return rag_chain, vectorstore

    except Exception as e:
        print(f"Error initializing RAG system: {e}")
        return None, None

# This block is for testing the RAG system initialization
if __name__ == "__main__":
    from dotenv import load_dotenv
    import os
    load_dotenv()
    if not os.getenv("GOOGLE_API_KEY"):
        print("GOOGLE_API_KEY not found in .env file. Please set it.")
    else:
        rag_chain, vectorstore = initialize_rag_system()
        if rag_chain:
            print("\nTest query:")
            test_query = "What are good exercises for someone with PCOS?"
            response = rag_chain.invoke(test_query)
            print(f"Agent Response: {response}")

            print("\nTest query for sleep hygiene:")
            test_query_sleep = "Can you give me tips for better sleep?"
            response_sleep = rag_chain.invoke(test_query_sleep)
            print(f"Agent Response: {response_sleep}")

            # Clean up (optional, for development purposes)
            # if os.path.exists(vectorstore.persist_directory):
            #     import shutil
            #     shutil.rmdir(vectorstore.persist_directory)
            #     print(f"Removed persistent directory: {vectorstore.persist_directory}")