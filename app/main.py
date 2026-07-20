from typing import TypedDict, Optional
from langgraph.graph import StateGraph, END
from langgraph.types import Command, interrupt
from langgraph.checkpoint.memory import InMemorySaver
from  dotenv import load_dotenv
from app.nodes import retreive_chunks 
from typing import Annotated
import operator
from app.schemas.pydantic_schemas import TicketType, ClassificationResult, CriticResult, ResolutionResult
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
    resolution : Annotated[list[str], operator.add]
    resolved : Annotated[list[bool],operator.add]
    rejection_reason : Annotated[list[str], operator.add]
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
    response = llm.chat.completions.parse(
        model="gpt-4o",
        messages=[{"role": "system", "content": "You are a helpful assistant that resolves support tickets. Provide a resolution for the ticket based on the context."},
                  {"role": "user", "content": f"Resolve the following ticket text: {state['ticket_text']}. Use the context: {state.get('retrieved_context', '')}."}
        ],
        response_format=ResolutionResult
    )
    result = json.loads(response.choices[0].message.content)
    if result["confidence"] < 0.5 or state.get('retry_count',0) <= 2:
        resolved = interrupt("Please provide a resolution for the ticket.")
        if resolved:
            print("Value of resolved:", resolved)
            return {"resolution": ["Issue resolved by human intervention."]}
        else:
            return {"resolution": [f"{result['resolution']}"]}
    else:
        return {"resolution": [f"{result['resolution']}"]}  # Placeholder resolution

def critic(state: TicketState):
    llm = OpenAI()
    response = llm.chat.completions.parse(
        model="gpt-4o",
        messages=[{"role": "system", "content": "You are a helpful assistant that criticizes the resolution of support tickets. Provide a boolean indicating if the ticket is resolved  and if not a rejection reason. Check properly wether ticket has really been resolved and were the specifics mentioned."},
                  {"role": "user", "content": f"Critique  the following ticket resolution: {state.get('resolution', '')}. If it is solved by human you should not critique it but if it is not by Human, critique it. If the solution is given, then you can pass it and not crititque."}
        ],
        response_format=CriticResult
    )
    result =json.loads(response.choices[0].message.content)
    
    resolved = result["resolved"]
    rejection_reason = result.get('rejection_reason', None)
    if resolved:
        return {"resolved": [True]}
    else:
        return {"resolved": [False], "rejection_reason": [rejection_reason], "retry_count": state.get('retry_count', 0) + 1}

def route_from_critics(state: TicketState):
    if state.get('resolved')[-1]:
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
    # result1 = app.invoke({"ticket_text": "The application crashes when I try to upload a file."}, config=config)
    for event in app.stream({"ticket_text": "The application crashes when I try to upload a file."}, config):
        print("Graph event:", event)
    state_snapshot = app.get_state(config)
    print(state_snapshot)
    print("Current Values:", state_snapshot.values)
    print("Next Nodes", state_snapshot.next)
    print(type(state_snapshot.next))
    while len(state_snapshot.next):
        result2 = app.invoke(Command(resume="Yes"), config=config)
        state_snapshot = app.get_state(config)
        print(state_snapshot.values)
        print("Next Nodes Final", state_snapshot.next)
    
    print("Final Values:", state_snapshot.values)
    print(result2)