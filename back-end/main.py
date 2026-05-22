# main.py
import os
import sys
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)

from zentric.agent.core_agent import create_agent_workflow 

def initialize_rag_system(directory_path="knowledge_base"):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    full_directory_path = os.path.join(current_dir, directory_path)
    
    documents_content = []
    fallback_docs_content = [
        "PCOS often involves insulin resistance, hormonal imbalances, and inflammation. A diet rich in fiber, lean protein, and healthy fats, with limited refined carbs and sugars, can help manage symptoms. Regular, gentle exercise like walking or yoga is beneficial.",
        "Insulin resistance means your body's cells don't respond well to insulin. Focus on whole, unprocessed foods. Strength training and consistent activity can improve insulin sensitivity.",
        "Type 2 Diabetes management involves blood sugar control. Diet and exercise are key. Prioritize complex carbohydrates, limit sugary drinks, and engage in moderate physical activity.",
        "Mindfulness practices, such as deep breathing or meditation for 5-10 minutes daily, can reduce stress and improve well-being for anyone managing chronic conditions. A simple deep breathing exercise is to breathe in for a count of four, hold for a count of four, and exhale for a count of six. This can be done anywhere, anytime.",
        "To manage fatigue and tiredness, ensure you are getting enough sleep (7-9 hours), maintaining a balanced diet, and engaging in light physical activity like walking. It is always best to consult with your healthcare provider to rule out any underlying conditions.",
        "Blood sugar levels are a critical part of managing metabolic health. For individuals without diabetes, a normal fasting glucose level is typically below 100 mg/dL. For individuals with diabetes, the target range is highly personalized and should be determined by a healthcare provider. Never adjust medication or treatment based on general advice. Always follow your doctor's specific recommendations."
    ]

    try:
        found_files = False
        if os.path.exists(full_directory_path):
            for root, _, files in os.walk(full_directory_path):
                for file in files:
                    if file.endswith(".md"):
                        file_path = os.path.join(root, file)
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                            if content.strip():
                                documents_content.append(content)
                                found_files = True
        
        if not found_files:
            print(f"Warning: No markdown files found in directory '{full_directory_path}'.")
            print("Using pre-defined example documents for RAG system.")
            documents_content = fallback_docs_content
        else:
            print(f"Successfully loaded {len(documents_content)} markdown file(s) from the knowledge base.")

    except Exception as e:
        print(f"An error occurred during RAG system initialization: {e}")
        print("Falling back to pre-defined documents for RAG system.")
        documents_content = fallback_docs_content

    if not documents_content:
        raise ValueError("No documents to process for RAG system. The knowledge base is empty and fallback content is missing.")

    text_splitter = CharacterTextSplitter(
        separator="\n\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        is_separator_regex=False,
    )
    
    splits = text_splitter.create_documents(documents_content)
    
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectorstore = Chroma.from_documents(splits, embeddings)
    retriever = vectorstore.as_retriever()

    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.7)

    # REFINED RAG PROMPT
    rag_prompt = ChatPromptTemplate.from_template(
        """You are a gentle, empathetic, and supportive health assistant named zentric. 
        Your goal is to provide helpful, evidence-based guidance.
        Answer the user's question or statement based on the following context. 
        Focus on providing a direct and compassionate response.
        Only if the user is asking for specific medical advice, a diagnosis, or if the context directly mentions it,
        should you include a gentle reminder to consult a healthcare professional.

        Context: {context}

        Question or Statement: {question}
        """
    )

    rag_chain = (
        {"context": retriever, "question": StrOutputParser()}
        | rag_prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain

def run_agent_interactive(agent_app, rag_chain):
    initial_state = {
        "user_id": "user_123",
        "chat_history": [],
        "user_profile": {"conditions": set(), "dietary_preferences": [], "current_mood": "neutral", "initial_setup_complete": False},
        "current_plan": {},
        "rewards_points": 0,
        "plan_generated_today": False,
        "comparison_query": "",
        "glucose_trend_data": {},
        "wellness_sparks": [],
        "zentric_whisper": "",
        "input": "",
        "loop_count": 0,
        "current_stage": "initialize_profile",
        "glucose_level": None,
        "heart_rate": None,
        "steps": None,
        "sleep_hours": None
    }

    print("\n" + "="*50)
    print("Welcome to your Holistic Metabolic Resilience Journey with zentric! 🧘‍♀️")
    print("I'm here to offer gentle, evidence-based support.")
    print("Let's start. You can share your health conditions with me.")
    print("For example: 'I have PCOS and insulin resistance'.")
    print("="*50 + "\n")

    current_state = initial_state

    while True:
        user_input = input("\nYou: ")
        if user_input.lower() in ["exit", "quit"]:
            print("\nIt's been a true pleasure supporting you today! Remember, every small step is a triumph. Take wonderful care of yourself until we chat again! 💖")
            break

        turn_state = current_state.copy()
        turn_state['input'] = user_input
        turn_state['chat_history'] = turn_state['chat_history'] + [HumanMessage(content=user_input)]
        
        final_state = agent_app.invoke(turn_state, config={"recursion_limit": 100})
        current_state = final_state

        full_response_content = "..."
        if "chat_history" in current_state and current_state["chat_history"]:
            last_message = current_state["chat_history"][-1]
            if isinstance(last_message, AIMessage):
                full_response_content = last_message.content
        
        print(f"\nzentric: {full_response_content}")
        
        if current_state.get('heart_rate'):
            print(f"✨ I've recorded your heart rate as {current_state['heart_rate']} bpm.")
        if current_state.get('steps'):
            print(f"✨ I've recorded your steps as {current_state['steps']}.")
        if current_state.get('sleep_hours'):
            print(f"✨ I've recorded your sleep as {current_state['sleep_hours']} hours.")

        print("\nWhat else is on your mind? I'm here to listen. 🎧")
        print("You can also give me your metrics, like 'My heart rate is 75 bpm' or 'I walked 8000 steps today'.")


def main():
    load_dotenv()
    if not os.getenv("GOOGLE_API_KEY"):
        print("Error: GOOGLE_API_KEY not found in .env file. Please set it.")
        sys.exit(1)
        
    print("Initializing RAG system...")
    try:
        rag_chain = initialize_rag_system()
        print("RAG system initialized successfully.")
    except Exception as e:
        print(f"Failed to initialize RAG system. Error: {e}")
        print("Exiting.")
        sys.exit(1)

    print("Creating agent workflow...")
    agent_app = create_agent_workflow(rag_chain)

    print("\nZentric, your Holistic Metabolic Resilience Agent, is ready. Type 'exit' or 'quit' to end.")
    run_agent_interactive(agent_app, rag_chain)

if __name__ == "__main__":
    main()