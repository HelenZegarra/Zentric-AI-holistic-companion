
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

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Agent State Definition ---
class AgentState(TypedDict):
    """
    Represents the state of our agent.

    Attributes:
        user_id: The ID of the current user.
        chat_history: A list of messages exchanged in the conversation.
        user_profile: The user's health profile (e.g., conditions, preferences).
        current_plan: The user's current wellness plan.
        rewards_points: The user's rewards points.
        plan_generated_today: A flag to check if a plan was generated today.
        comparison_query: A query for comparing health data.
        glucose_trend_data: The user's glucose trend data.
        wellness_sparks: A list of wellness tips.
        zentric_whisper: A special prompt for the LLM.
        input: The user's most recent input message.
        loop_count: A counter for loop detection.
        current_stage: The current stage of the conversation.
        glucose_level: The last recorded glucose level.
        heart_rate: The last recorded heart rate.
        steps: The last recorded step count.
        sleep_hours: The last recorded sleep duration.
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
    # Add a field for intermediate steps for the tool calling agent
    intermediate_steps: Annotated[List[BaseMessage], operator.add]


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
    # Replace "YOUR_API_KEY" with your actual Spoonacular API key
    api_key = os.getenv("SPOONACULAR_API_KEY", "YOUR_API_KEY")

    if not api_key or api_key == "YOUR_API_KEY":
        return "Error: Spoonacular API key not found. Please set the SPOONACULAR_API_KEY environment variable."

    base_url = "https://api.spoonacular.com/recipes/complexSearch"
    params = {
        "apiKey": api_key,
        "query": query,
        "number": 5, # Get up to 5 recipes
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

def conversational_node(state: AgentState):
    logging.info("Executing conversational_node...")
    
    system_prompt = """
    You are Zentric, a warm, friendly, and empathetic AI assistant focused on health and wellness. 
    Your primary goal is to provide gentle, helpful, and evidence-based guidance in a conversational tone.
    
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
    """
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}")
    ])
    
    runnable = prompt | llm
    response = runnable.invoke({"chat_history": state["chat_history"], "input": state["input"]})
    
    return {"chat_history": [response]}


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
    # The `call_model` node will handle the initial LLM invocation.
    return call_model(state)


def call_model(state: AgentState):
    """
    Invokes the LLM with tools bound to it.
    """
    logging.info("Executing call_model node...")
    
    # Define a system prompt for the tool-calling agent.
    system_prompt = (
        "You are Zentric, a helpful AI assistant. You have access to a tool to find recipes "
        "based on a user's query, cuisine, and diet. Use the tool to respond to questions "
        "about recipes. If the user's request is not about recipes, provide a simple, friendly "
        "response without using the tool."
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}")
    ])
    
    # Bind the tool to the LLM so it knows how to use it.
    tools = [search_recipes]
    llm_with_tools = llm.bind_tools(tools)
    
    # Combine the prompt and the LLM with tools
    runnable = prompt | llm_with_tools

    # Get all messages including the latest input
    messages = state["chat_history"] + [HumanMessage(content=state["input"])]
    response = runnable.invoke({"chat_history": messages, "input": state["input"]})
    
    # The LLM's response can be a direct answer or a tool call.
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


def should_continue_tool_agent(state: AgentState) -> str:
    """
    Determines whether to continue the tool-calling agent or finish.
    """
    last_message = state['chat_history'][-1]
    
    if last_message.tool_calls:
        return "continue"
    else:
        return "end"


# --- Routing Logic (not a node) ---
def route_to_next_node(state: AgentState) -> str:
    """This function determines which node to execute next based on the state."""
    user_input_lower = state["input"].lower()
    
    if any(keyword in user_input_lower for keyword in ["glucose", "blood sugar", "sugar level"]):
        return "collect_data_node"
    
    if any(keyword in user_input_lower for keyword in ["recipe", "recipes", "food", "cook", "meal"]):
        return "tool_agent_entry_node"
        
    return "conversational_node"

# --- Main Workflow Creation ---
def create_agent_workflow():
    logging.info("Starting agent workflow creation...")
    workflow = StateGraph(AgentState)

    # Add all nodes to the graph
    workflow.add_node("initialize_profile_node", initialize_profile_node)
    workflow.add_node("conversational_node", conversational_node)
    workflow.add_node("collect_data_node", collect_data_node)
    workflow.add_node("tool_agent_entry_node", tool_agent_entry_node)
    workflow.add_node("call_model", call_model)
    workflow.add_node("call_tool", call_tool)

    # Set the entry point of the graph
    workflow.set_entry_point("initialize_profile_node")

    # Define the primary edges for the main flow
    workflow.add_edge("initialize_profile_node", "conversational_node")
    workflow.add_conditional_edges(
        "conversational_node",
        route_to_next_node,
        {
            "conversational_node": "conversational_node",
            "collect_data_node": "collect_data_node",
            "tool_agent_entry_node": "tool_agent_entry_node"
        }
    )

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

    # After non-tool nodes, the turn ends.
    workflow.add_edge("collect_data_node", END)
    workflow.add_edge("conversational_node", END)

    app = workflow.compile()
    logging.info("Agent workflow compiled successfully.")
    return app

# Example usage block
if __name__ == "__main__":
    # The `create_agent_workflow` function no longer needs a rag_chain parameter.
    app = create_agent_workflow()

    # Test the new agent with a recipe query
    print("--- Test 1: Recipe Search ---")
    inputs = {"input": "I want to find a recipe for chicken pasta."}
    response = app.invoke(inputs)
    print(f"Agent's final response: {response['chat_history'][-1].content}\n")
    
    print("--- Test 2: Glucose Data Entry ---")
    inputs = {"input": "My blood sugar is 115."}
    response = app.invoke(inputs)
    print(f"Agent's final response: {response['chat_history'][-1].content}\n")
    
    print("--- Test 3: General Conversation ---")
    inputs = {"input": "Hello, how are you today?"}
    response = app.invoke(inputs)
    print(f"Agent's final response: {response['chat_history'][-1].content}\n")
