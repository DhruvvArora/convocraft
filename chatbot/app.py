# modules for flask backend

from flask import Flask, request, jsonify
from flask_cors import CORS
import os


# modules for langgraph chatbot

from langgraph.prebuilt.chat_agent_executor import AgentState
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage
from langchain.tools import tool
from dotenv import load_dotenv
from pydantic import BaseModel
from typing import Literal
import os
import functools


# loading env varaiable
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
llm = ChatOpenAI(model="gpt-4o-mini", api_key=api_key, max_tokens=150, temperature=0.7, streaming=True)  


# defining the Flask app
app = Flask(__name__)
# Allow all origins in development, specify your frontend URL in production
CORS(app, resources={r"/*": {"origins": "*"}})

port = int(os.environ.get('PORT', 5001))


def agent_node(state, agent, name):
    result = agent.invoke(state)
    
    if "messages" in result and result["messages"]:
        last_message = result["messages"][-1]
        # Only return the agent's response if it's not just echoing the user's input
        if last_message.content.lower() != state["messages"][-1].content.lower():
            return {
                "messages": [HumanMessage(content=last_message.content, name=name)]
            }
    
    # Return empty messages if no valid response
    return {"messages": []}

system_prompt = (
    "You are a medical assistant providing the information related to the medical field."
    "You are also a supervisor tasked with managing a conversation between the following agents: {members}. "
    "Each agent provides specific information related to its expertise area in response to user queries. "
    "For example, MedicineAgent should give medical advice within general advice limitations, MedicalHospitalAgent should suggest hospitals based on location, and MedicalDepartmentAgent should offer relevant departmental options."
)

members = ["GreetingAgent", "FarewellAgent", "MedicineAgent", "MedicalHospitalAgent", "MedicalDepartmentAgent"]

class routeResponse(BaseModel):
    next: Literal["FINISH", "GreetingAgent", "FarewellAgent", "MedicineAgent", "MedicalHospitalAgent", "MedicalDepartmentAgent"]

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="messages"),
        (
            "system",
            "Given the conversation above, who should act next? "
            "Or should we FINISH? Select one of: ['GreetingAgent', 'FarewellAgent', 'MedicineAgent', 'MedicalHospitalAgent', 'MedicalDepartmentAgent', 'FINISH']",
        ),
    ]
).partial(members=", ".join(members))

def UserProxyAgent(state):
    return {"messages": state["messages"]}

def OperatorAgent(state):
    supervisor_chain = prompt | llm.with_structured_output(routeResponse)
    response = supervisor_chain.invoke(state)
    
    if hasattr(response, 'next'):
        # print(f"Operator selected: {response.next}")
        return {"next": response.next, "messages": state.get("messages", [])}
    else:
        print("Error: 'next' not found in operator response.")
        return {"next": "FINISH", "messages": state.get("messages", [])}  



# Helper: always return plain text from LangChain ChatOpenAI
def ask_llm_text(prompt_input) -> str:
    result = llm.invoke(prompt_input)
    # ChatOpenAI.invoke returns an AIMessage
    return result.content if hasattr(result, "content") else str(result)

@tool
def greeting_tool() -> str:
    """Fetches a greeting message from OpenAI."""
    return ask_llm_text("Greet the user warmly and offer assistance.")

@tool
def farewell_tool() -> str:
    """Fetches a farewell message from OpenAI."""
    return ask_llm_text("Say goodbye to the user in a friendly and polite manner.")

@tool
def medicine_tool() -> str:
    """Fetches information about general medicines for common symptoms like fever or headache."""
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", system_instruction),
        ("user", "Provide a clear list of commonly recommended over-the-counter medications for treating fever.")
    ])

    # Option 1: format messages then invoke
    messages = prompt_template.format_messages()
    return ask_llm_text(messages)

@tool
def medical_hospital_tool() -> str:
    """Fetches information about nearby hospitals based on general location data."""
    return ask_llm_text(
        "Provide information about general hospitals that offer a range of medical services."
    )

@tool
def medical_department_tool() -> str:
    """Fetches information about medical departments available in a hospital."""
    return ask_llm_text(
        "List common medical departments in a hospital and briefly describe their primary functions."
    )


GreetingsAgent = create_react_agent(llm, tools=[greeting_tool])
GreetingsNode = functools.partial(agent_node, agent=GreetingsAgent, name="GreetingAgent")

FarewellAgent = create_react_agent(llm, tools=[farewell_tool])
FarewellNode = functools.partial(agent_node, agent=FarewellAgent, name="FarewellAgent")

MedicineAgent = create_react_agent(llm, tools=[medicine_tool])
MedicineNode = functools.partial(agent_node, agent=MedicineAgent, name="MedicineAgent")

MedicalHospitalAgent = create_react_agent(llm, tools=[medical_hospital_tool])
MedicalHospitalNode = functools.partial(agent_node, agent=MedicalHospitalAgent, name="MedicalHospitalAgent")

MedicalDepartmentAgent = create_react_agent(llm, tools=[medical_department_tool])
MedicalDepartmentNode = functools.partial(agent_node, agent=MedicalDepartmentAgent, name="MedicalDepartmentAgent")

workflow = StateGraph(AgentState)
workflow.add_node("UserProxy", UserProxyAgent)
workflow.add_node("GreetingAgent", GreetingsNode)
workflow.add_node("FarewellAgent", FarewellNode)
workflow.add_node("Operator", OperatorAgent)
workflow.add_node("MedicineAgent", MedicineNode)
workflow.add_node("MedicalHospitalAgent", MedicalHospitalNode)
workflow.add_node("MedicalDepartmentAgent", MedicalDepartmentNode)

conditional_map = {
    "GreetingAgent": "GreetingAgent",
    "FarewellAgent": "FarewellAgent",
    "MedicineAgent": "MedicineAgent",
    "MedicalHospitalAgent": "MedicalHospitalAgent",
    "MedicalDepartmentAgent": "MedicalDepartmentAgent",
    "FINISH": END
}
workflow.add_conditional_edges("Operator", lambda x: x["next"], conditional_map)

workflow.add_edge(START, "UserProxy") 
workflow.add_edge("UserProxy", "Operator")

graph = workflow.compile()

user_messages = []


import traceback

@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json(silent=True) or {}
        user_input = (data.get('message') or "").strip()

        if not user_input:
            return jsonify({"response": "Please enter a message."}), 400

        user_messages = [HumanMessage(content=user_input)]
        response_message = None

        for state in graph.stream({"messages": user_messages}):
            if "__end__" not in state:
                for agent_name, agent_response in state.items():
                    if agent_name == "UserProxy":
                        continue

                    if "messages" in agent_response and agent_response["messages"]:
                        last_message = agent_response["messages"][-1]
                        if last_message.content != user_input:
                            response_message = last_message.content
                            user_messages.append(last_message)

        if not response_message:
            response_message = "The chatbot did not provide a response."

        return jsonify({"response": response_message})

    except Exception as e:
        app.logger.exception("Error in /chat")
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=port)
