import os
import requests
from typing import TypedDict, Annotated, List, Optional, Any, Union
import operator
import json
import logging
import re
from pydantic import BaseModel, Field

from langchain_core.agents import AgentAction, AgentFinish, AgentActionMessageLog
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, FunctionMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages

from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv

# --- RAG Specific Imports ---
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.runnables import RunnablePassthrough

# --- Load environment variables FIRST ---
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- RAG Knowledge Base and Setup ---
# This is a small, in-memory knowledge base for demonstration purposes.
# In a real application, you would load this from a file or a persistent database.
MINDFULNESS_KNOWLEDGE_BASE = [
    "Mindfulness is a state of being in the present moment, observing your thoughts and feelings without judgment. For people with Type 2 Diabetes, mindfulness can help reduce stress, which in turn can positively impact blood sugar levels.",
    "Stress can cause your body to release hormones that raise blood sugar. Practicing mindfulness, such as through deep breathing exercises or meditation, can help manage this stress response.",
    "A simple mindfulness practice for diabetes management is 'mindful eating.' This involves paying full attention to the food you are eating, savoring each bite, and listening to your body's hunger and fullness cues. This can help with portion control and healthy eating habits.",
    "Mindful breathing can be done anywhere. Simply sit or stand comfortably, close your eyes, and focus on the sensation of your breath entering and leaving your body. Start with just a few minutes a day and gradually increase the time.",
    "Regular mindfulness practice has been linked to improved emotional well-being and a greater sense of control over one's health, which are beneficial for managing chronic conditions like Type 2 Diabetes."
]

# Initialize RAG components
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
docs = text_splitter.create_documents(MINDFULNESS_KNOWLEDGE_BASE)
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
vectorstore = Chroma.from_documents(docs, embeddings)
retriever = vectorstore.as_retriever()
logging.info("RAG components initialized with in-memory knowledge base.")


# --- Agent State Definition ---
class AgentState(TypedDict):
    """
    Represents the state of our agent.
    """
    user_id: str
    chat_history: Annotated[List[BaseMessage], add_messages]
    user_profile: dict
    current_plan: dict
    rewards_points: int
    plan_generated_today: bool
    comparison_query: str
    glucose_trend_data: dict
    wellness_sparks: list
    zentric_whisper: str
    input: str
    loop_count: int
    current_stage: str
    glucose_level: Optional[float]
    heart_rate: Optional[int]
    steps: Optional[int]
    sleep_hours: Optional[float]
    intermediate_steps: Annotated[List[BaseMessage], operator.add]
    retrieved_context: Optional[str]


# --- LLM and Prompt Configuration ---
llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest", temperature=0.7)

# --- Spoonacular Tool Definition ---
class RecipeSearchInput(BaseModel):
    """Input for searching for recipes."""
    query: str = Field(description="The main ingredient or dish to search for.")
    cuisine: Optional[str] = Field(None, description="The cuisine type (e.g., 'Italian', 'Mexican').")
    diet: Optional[str] = Field(None, description="The diet type (e.g., 'vegetarian', 'gluten free').")

@tool("search_recipes", args_schema=RecipeSearchInput)
def search_recipes(query: str, cuisine: Optional[str] = None, diet: Optional[str] = None) -> str:
    """
    Searches for recipes using the Spoonacular API based on a query, cuisine, and/or diet.
    Returns a list of recipe titles and their IDs.
    """
    api_key = os.getenv("SPOONACULAR_API_KEY", "YOUR_API_KEY")

    if not api_key or api_key == "YOUR_API_KEY":
        return "Error: Spoonacular API key not found. Please set the SPOONACULAR_API_KEY environment variable."

    base_url = "https://api.spoonacular.com/recipes/complexSearch"
    params = {
        "apiKey": api_key,
        "query": query,
        "number": 5, 
        "addRecipeInformation": False,
    }
    if cuisine:
        params["cuisine"] = cuisine
    if diet:
        params["diet"] = diet

    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        data = response.json()
        
        if not data.get("results"):
            return "No recipes found matching the criteria."
        
        formatted_results = "Found the following recipes:\n"
        for recipe in data["results"]:
            formatted_results += f"- {recipe['title']} (ID: {recipe['id']})\n"
        
        return formatted_results
    except requests.exceptions.RequestException as e:
        return f"An error occurred while calling the Spoonacular API: {e}"

# --- Helper Functions ---
def parse_profile(input_text: str) -> dict:
    profile = {}
    if "diabetes" in input_text.lower():
        profile["conditions"] = ["Type 2 Diabetes"]
    if "vegetarian" in input_text.lower():
        profile["dietary_preferences"] = ["vegetarian"]
    return profile

def get_profile_summary(profile: dict) -> str:
    summary = []
    if profile.get("conditions"):
        summary.append(f"Conditions: {', '.join(profile['conditions'])}")
    if profile.get("dietary_preferences"):
        summary.append(f"Dietary Preferences: {', '.join(profile['dietary_preferences'])}")
    if not summary:
        return "No specific health profile information provided yet."
    return "Your current health profile: " + ", ".join(summary)


# --- Graph Nodes ---
def initialize_profile_node(state: AgentState):
    logging.info("Executing initialize_profile_node...")
    user_input = state["input"]
    user_profile = state["user_profile"]

    if not isinstance(user_profile.get("conditions"), set):
        user_profile["conditions"] = set(user_profile.get("conditions", []))
        
    if "diabetes" in user_input.lower():
        user_profile["conditions"].add("Type 2 Diabetes")
    
    return {"user_profile": user_profile}
    
def route_input(state: AgentState) -> str:
    """This function determines which node to execute next based on the state."""
    logging.info("Routing input...")
    user_input_lower = state["input"].lower()
    
    if any(keyword in user_input_lower for keyword in ["glucose", "blood sugar", "sugar level"]):
        return "collect_data_node"
    
    if any(keyword in user_input_lower for keyword in ["recipe", "recipes", "food", "cook", "meal"]):
        return "tool_agent_entry_node"
    
    if any(keyword in user_input_lower for keyword in ["mindfulness", "meditation", "stress", "breathing"]):
        return "retrieve_rag_content"
        
    return "conversational_node"

def conversational_node(state: AgentState):
    logging.info("Executing conversational_node...")
    
    retrieved_context = state.get("retrieved_context")
    context_prompt = ""
    if retrieved_context:
        context_prompt = f"\n\nUse the following retrieved knowledge to answer the user's question:\n{retrieved_context}"
    
    system_prompt = f"""
    You are Zentric, a warm, friendly, and empathetic AI assistant focused on health and wellness. 
    Your primary goal is to provide gentle, helpful, and evidence-based guidance in a conversational tone.
    
    Start the conversation with a friendly greeting, and if the user's health conditions are known,
    gently acknowledge them and offer support. For example: "Hello there! It's so good to connect with you. I see you're managing Type 2 Diabetes, and I'm here to support you on your wellness journey. How can I help you today? 😊"

    You do not have access to real-time tools for recipes. If a user asks for recipes, you must
    direct them to a tool-based search with a prompt like: "I can help with that! Let's search for some recipes. What kind of recipe are you looking for?"
    
    To sound more human and gentle:
    - Use positive and encouraging language.
    - Ask questions to understand the user's feelings and experiences.
    - Acknowledge what the user says with phrases like "That's a great question!" or "Thanks for sharing that."
    - End your responses by inviting further conversation, such as "Is there anything else I can help with?"
    - Avoid sounding overly technical, robotic, or clinical. Your tone should feel like a supportive friend or a wellness coach.
    
    Incorporate relevant emojis into your responses to enhance your friendly and conversational tone.

    When providing lists or structured information, **use Markdown formatting** to make the output easy to read. Use bolding (`**text**`), headings (`# Heading`), and bullet points (`* item`) where appropriate.
    
    Keep the conversation flowing naturally and stay on topic with health and wellness.
    
    {context_prompt}
    """
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}")
    ])
    
    runnable = prompt | llm
    response = runnable.invoke({"chat_history": state["chat_history"], "input": state["input"]})
    
    return {"chat_history": [response], "retrieved_context": None}


def collect_data_node(state: AgentState):
    logging.info("Executing collect_data_node...")
    user_input = state["input"]
    
    match = re.search(r'(\d+(\.\d+)?)', user_input)
    if match:
        try:
            glucose_level = float(match.group(1))
            response_content = f"Thank you! I've logged your glucose level as {glucose_level}. How are you feeling today? 😊"
            return {"glucose_level": glucose_level, "chat_history": [AIMessage(content=response_content)]}
        except (ValueError, TypeError):
            pass

    response_content = "I didn't catch a specific glucose number in your message. Could you please provide just the number? For example: 'My glucose is 120'. 🩺"
    return {"chat_history": [AIMessage(content=response_content)]}


def tool_agent_entry_node(state: AgentState):
    """Entry point for the tool-calling agent sub-graph."""
    logging.info("Executing tool_agent_entry_node...")
    return call_model(state)


def call_model(state: AgentState):
    """
    Invokes the LLM with tools bound to it.
    """
    logging.info("Executing call_model node...")
    
    user_profile = state["user_profile"]
    profile_summary = get_profile_summary(user_profile)
    
    system_prompt = (
        "You are Zentric, a helpful AI assistant. You have access to a tool to find recipes "
        "based on a user's query, cuisine, and diet. Use the tool to respond to questions "
        "about recipes. If the user's request is not about recipes, provide a simple, friendly "
        "response without using the tool.\n\n"
        f"**User Profile:** {profile_summary}\n"
        "**Special Instruction:** If the user's profile lists 'Type 2 Diabetes', you must always "
        "include the `diet='diabetic'` parameter when calling the `search_recipes` tool. "
        "For example, if the user asks for chicken recipes and has diabetes, your tool call should be "
        "`search_recipes(query='chicken', diet='diabetic')`. "
        "Otherwise, use the parameters as requested by the user. If a user asks for recipes but does not specify a cuisine or diet you may not add one."
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}")
    ])
    
    tools = [search_recipes]
    llm_with_tools = llm.bind_tools(tools)
    
    runnable = prompt | llm_with_tools

    messages = state["chat_history"] + [HumanMessage(content=state["input"])]
    response = runnable.invoke({"chat_history": messages, "input": state["input"]})
    
    return {"chat_history": [response]}


def call_tool(state: AgentState):
    """
    Executes the tool call requested by the LLM.
    """
    logging.info("Executing call_tool node...")
    last_message = state['chat_history'][-1]
    tool_calls = last_message.tool_calls
    tool_messages = []
    
    for tool_call in tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]

        if tool_name == "search_recipes":
            try:
                result = search_recipes(**tool_args)
                tool_messages.append(FunctionMessage(content=result, name=tool_name, tool_call_id=tool_call["id"]))
            except Exception as e:
                tool_messages.append(FunctionMessage(content=f"Error executing tool '{tool_name}': {e}", name=tool_name, tool_call_id=tool_call["id"]))
        else:
            tool_messages.append(FunctionMessage(content=f"Tool '{tool_name}' not found.", name=tool_name, tool_call_id=tool_call["id"]))
    
    return {"chat_history": tool_messages}


# --- RAG Retrieval Node ---
def retrieve_rag_content(state: AgentState):
    logging.info("Executing retrieve_rag_content node...")
    user_input = state["input"]
    
    docs = retriever.invoke(user_input)
    
    context = "\n\n".join([doc.page_content for doc in docs])
    
    return {"retrieved_context": context}


def should_continue_tool_agent(state: AgentState) -> str:
    """
    Determines whether to continue the tool-calling agent or finish.
    """
    last_message = state['chat_history'][-1]
    
    if last_message.tool_calls:
        return "continue"
    else:
        return "end"


# --- Flask Web Server Setup ---
def create_agent_workflow():
    logging.info("Starting agent workflow creation...")
    workflow = StateGraph(AgentState)

    # --- CORRECTION: route_input is NOT a node, it's a function for the conditional edge ---
    # We add the other nodes first.
    workflow.add_node("initialize_profile_node", initialize_profile_node)
    workflow.add_node("conversational_node", conversational_node)
    workflow.add_node("collect_data_node", collect_data_node)
    workflow.add_node("tool_agent_entry_node", tool_agent_entry_node)
    workflow.add_node("call_model", call_model)
    workflow.add_node("call_tool", call_tool)
    workflow.add_node("retrieve_rag_content", retrieve_rag_content)

    # Set the entry point of the graph
    workflow.set_entry_point("initialize_profile_node")

    # Define the primary edges for the main flow
    # The initial flow now goes from profile init directly to the conditional router
    workflow.add_conditional_edges(
        "initialize_profile_node",
        route_input,
        {
            "conversational_node": "conversational_node",
            "collect_data_node": "collect_data_node",
            "tool_agent_entry_node": "tool_agent_entry_node",
            "retrieve_rag_content": "retrieve_rag_content"
        }
    )

    # Now, all non-tool nodes, including the conversational one, are terminal states.
    workflow.add_edge("conversational_node", END)
    workflow.add_edge("collect_data_node", END)
    
    # Define the edges for the tool-calling sub-graph
    workflow.add_conditional_edges(
        "tool_agent_entry_node",
        should_continue_tool_agent,
        {
            "continue": "call_tool",
            "end": END,
        }
    )
    workflow.add_edge("call_tool", "call_model")
    workflow.add_conditional_edges(
        "call_model",
        should_continue_tool_agent,
        {
            "continue": "call_tool",
            "end": END,
        }
    )

    # After the RAG retrieval, the flow goes to the conversational node to generate the response, then ends.
    workflow.add_edge("retrieve_rag_content", "conversational_node")
    
    app = workflow.compile()
    logging.info("Agent workflow compiled successfully.")
    return app

flask_app = Flask(__name__)
CORS(flask_app)
agent_app = create_agent_workflow()

@flask_app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json()
        user_input = data.get('input')
        agent_state_from_frontend = data.get('agentState')

        if not user_input:
            return jsonify({'error': 'No input provided'}), 400

        # Construct the initial state for the agent
        turn_state = agent_state_from_frontend or {
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
            "sleep_hours": None,
            "intermediate_steps": [],
            "retrieved_context": None
        }

        if 'user_profile' in turn_state and 'conditions' in turn_state['user_profile']:
            turn_state['user_profile']['conditions'] = set(turn_state['user_profile']['conditions'])

        if 'chat_history' in turn_state and turn_state['chat_history']:
            converted_history = []
            for msg_dict in turn_state['chat_history']:
                if msg_dict['sender'] == 'user':
                    converted_history.append(HumanMessage(content=msg_dict['content']))
                else:
                    converted_history.append(AIMessage(content=msg_dict['content']))
            turn_state['chat_history'] = converted_history

        turn_state['input'] = user_input
        turn_state['chat_history'].append(HumanMessage(content=user_input))

        logging.info(f"Invoking agent with input: {user_input}")
        final_state = agent_app.invoke(turn_state, config={"recursion_limit": 100})
        
        last_message = final_state["chat_history"][-1].content

        serializable_chat_history = []
        for message in final_state['chat_history']:
            if isinstance(message, HumanMessage):
                serializable_chat_history.append({'content': message.content, 'sender': 'user'})
            else:
                serializable_chat_history.append({'content': message.content, 'sender': 'zentric'})
        
        serializable_state = final_state.copy()
        serializable_state['chat_history'] = serializable_chat_history
        serializable_state['user_profile']['conditions'] = list(serializable_state['user_profile']['conditions'])
        
        response_data = {
            "response": last_message,
            "agentState": serializable_state
        }
        
        return jsonify(response_data)

    except Exception as e:
        logging.error(f"--- An error occurred during chat processing ---")
        logging.error(f"Error: {e}")
        logging.error("Traceback:", exc_info=True)
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    flask_app.run(debug=True)
