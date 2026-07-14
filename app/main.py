from typing import TypedDict, Optional
from langgraph.graph import StateGraph, END
from langgraph.types import Command, interrupt
from langgraph.checkpoint.memory import InMemorySaver
from  dotenv import load_dotenv
from app.nodes import retreive_chunks 
from app.schemas.pydantic_schemas import TicketType, ClassificationResult, CriticResult  
import os
import json
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if OPENAI_API_KEY is None:
    raise ValueError("OpenAI API key not found.")

config = {
    "configurable": {
        "thread_id": "user-session-12345"
    }
}

class TicketState(TypedDict):
    ticket_text : str
    category : Optional[str]
    confidence : Optional[float]
    retrieved_context : Optional[str]
    resolution : Optional[str]
    resolved : Optional[bool]
    retry_count : Optional[int]

graph = StateGraph(TicketState)

from openai import OpenAI
from app.main import TicketState

def return_chunk():
    return "chunk"

def classify(state : TicketState):
    llm = OpenAI()
    response = llm.chat.completions.parse(
        model="gpt-4o",
        messages=[{"role": "system", "content": "You are a helpful assistant that classifies support tickets into categories: Technical, Billing, or General Support. Provide the category and a confidence score between 0 and 1."},
                  {"role": "user", "content": f"Classify the following ticket text:{state['ticket_text']}"}
        ],
        response_format=ClassificationResult
    )
    result = response.choices[0].message.content
    print("result:", result)
    dict_result = json.loads(result)
    print("result type:", type(dict_result))
    return_dict = {"category": dict_result["category"], "confidence": dict_result["confidence"]}
    return  ClassificationResult(**return_dict)  # Placeholder confidence

def retrieve_context(state : TicketState):
    llm = OpenAI()
    chunks = retreive_chunks(state['ticket_text'])
    print("Retrieved chunks:", chunks)
    context = ""
    for chunk in chunks[0]:
        print("hi")
        context += chunk + "\n"
    
    print("Retrieved context:", context)
    return {"retrieved_context": context}  # Placeholder context

def resolve(state : TicketState):
    llm = OpenAI()
    # response = llm.response.create(
    #     model="gpt-4o",
    #     input = f"Resolve the following ticket text: {state['ticket_text']}. Use the context: {state.get('retrieved_context', '')}."
    # )
    resolved = interrupt("Please provide a resolution for the ticket.")
    if resolved :
        print("Value of resolved:", resolved)
        return {"resolution": "Issue resolved by human intervention."}
    return {"resolution": "The issue has been resolved by bot."}  # Placeholder resolution

def critic(state: TicketState):
    llm = OpenAI()
    response = llm.chat.completions.parse(
        model="gpt-4o",
        messages=[{"role": "system", "content": "You are a helpful assistant that criticizes the resolution of support tickets. Provide a boolean indicating if the ticket is resolved  and if not a rejection reason."},
                  {"role": "user", "content": f"Critique  the following ticket resolution: {state.get('resolution', '')}"}
        ],
        response_format=CriticResult
    )
    result =json.loads(response.choices[0].message.content)
    
    resolved = result["resolved"]
    rejection_reason = result.get('rejection_reason', None)
    if resolved:
        return {"resolved": True}
    else:
        return {"resolved": False, "rejection_reason": rejection_reason, "retry_count": state.get('retry_count', 0) + 1}

def route_from_critics(state: TicketState):
    if state.get('resolved'):
        return "END"
    elif state.get('retry_count', 0) >= 3:
        return "END"
    else:
        return "resolve"
    

graph.add_node("classify", classify)
graph.add_node("retrieve_context", retrieve_context)
graph.add_node("resolve", resolve)
graph.add_node("critic", critic)

graph.set_entry_point("classify")
graph.add_edge("classify", "retrieve_context")
graph.add_edge("retrieve_context", "resolve")
graph.add_edge("resolve", "critic")
graph.add_conditional_edges("critic", route_from_critics, {
        "resolve": "resolve",
        "END": END,
    })

memory = InMemorySaver()

app = graph.compile(checkpointer=memory)


if __name__ == "__main__":
    result1 = app.invoke({"ticket_text": "The application crashes when I try to upload a file."}, config=config)
    state_snapshot = app.get_state(config)
    print("Current Values:", state_snapshot.values)
    result2 = app.invoke(Command(resume=""), config=config)
    print(result2)