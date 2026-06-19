import os
import re
from typing import TypedDict, Annotated, Literal
from dotenv import load_dotenv
from langchain_core.messages import AnyMessage, HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.types import Command
from langchain_mistralai import ChatMistralAI

load_dotenv()

class StudioState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    game_design: str
    generated_code: str
    bug_report: str
    iteration_count: int

model = ChatMistralAI(model="mistral-small-latest", temperature=0.2)

def route_initial_input(state: StudioState) -> Literal["game_designer_node", "developer_node"]:
    if state.get("bug_report") or state.get("generated_code"):
        return "developer_node"
    return "game_designer_node"

def game_designer_node(state: StudioState):
    system_prompt = (
        "You are an expert game designer. Expand the user's request into a comprehensive "
        "specification list for a 2D browser mini-game. Detail canvas control (Arrow keys), "
        "winning/losing states, and gameplay scoring rules. Do not write code yet."
    )
    messages = [("system", system_prompt)] + state["messages"]
    response = model.invoke(messages)
    return {
        "messages": [response],
        "game_design": response.content,
        "iteration_count": 0
    }

def developer_node(state: StudioState):
    current_iter = state.get("iteration_count", 0)
    system_prompt = (
        "You are an elite Lead Software Engineer. Write a completely self-contained, standalone "
        "HTML file containing inline <style> CSS and <script> JavaScript for the requested game.\n\n"
        "CRITICAL RULES:\n"
        "1. Return ONLY the code inside a standard markdown code block (```html ... ```).\n"
        "2. Ensure all event listeners, logic loops, rendering contexts, and asset variables are explicitly defined.\n"
        "3. Focus 100% on logical completeness, bug-free runtime loops, and state management."
    )
    
    context = state["messages"]
    active_bug = state.get("bug_report")
    
    if active_bug:
        feedback = f"CRITICAL CHANGE/BUG REQUEST TO EXECUTE: \n{active_bug}\n Please completely regenerate or update the file to apply this fix."
        context = context + [HumanMessage(content=feedback)]
    elif state.get("game_design"):
        context = context + [HumanMessage(content="Please build the web source code now based on the designer's specifications above.")]

    messages = [("system", system_prompt)] + context
    response = model.invoke(messages)

    return {
        "messages": [response],
        "generated_code": response.content,
        "iteration_count": current_iter + 1
    }

def ui_designer_node(state: StudioState):
    system_prompt = (
        "You are a Creative Frontend UI/UX Designer specialized in arcade web wrappers.\n"
        "Take the provided standalone HTML code and elevate its aesthetic design.\n"
        "CRITICAL RULES:\n"
        "1. Keep all core game logic, variables, and script states perfectly identical.\n"
        "2. Modernize the styling: use beautiful color palettes, clean fonts, glowing borders, neat side-panels, or modern layouts.\n"
        "3. Output ONLY the resulting fully updated code block wrapped inside standard markdown (```html ... ```)."
    )
    
    code = state["generated_code"]
    messages = [("system", system_prompt), HumanMessage(content=f"Polish the UI layout of this file:\n\n{code}")]
    response = model.invoke(messages)
    
    return {
        "messages": [response],
        "generated_code": response.content
    }

def qa_engineer_node(state: StudioState):
    code = state.get("generated_code", "")

    if "```html" not in code:
        return Command(
            update={"bug_report": "Error: Your output was not encapsulated in a ```html code block."},
            goto="developer_node"
        )

    system_prompt = (
        "You are a Senior QA Automation Engineer. Inspect the provided code string carefully.\n"
        "Verify: Are there open tags? Is the JavaScript event loop complete? Are variables defined correctly?\n"
        "If you find ANY code logic error, reply with a detailed paragraph starting with [BUGFOUND].\n"
        "If the script is pristine and ready to run, reply with exactly: [PASS]."
    )

    verification = model.invoke([
       ("system", system_prompt),
       HumanMessage(content=f"Review this source code:\n\n{code}")
    ])
    
    if state["iteration_count"] >= 4:
        return Command(goto=END)
    
    if "[BUGFOUND]" in verification.content:
        return Command(
            update={"bug_report": verification.content},
            goto="developer_node"
        )
    return Command(goto=END)

studio_graph = StateGraph(StudioState)

studio_graph.add_node("game_designer_node", game_designer_node)
studio_graph.add_node("developer_node", developer_node)
studio_graph.add_node("ui_designer_node", ui_designer_node)
studio_graph.add_node("qa_engineer_node", qa_engineer_node)

studio_graph.add_conditional_edges(START, route_initial_input)

studio_graph.add_edge("game_designer_node", "developer_node")
studio_graph.add_edge("developer_node", "ui_designer_node")
studio_graph.add_edge("ui_designer_node", "qa_engineer_node")

app = studio_graph.compile()

if __name__ == "__main__":
    while True:
        user_command = input("Prompt: ").strip()
        if user_command.lower() == 'exit':
            break
            
        if not user_command:
            continue

        inputs = {}
        
        if user_command.lower() == 'fix':
            if not os.path.exists("mini_game.html"):
                continue
                
            with open("mini_game.html", "r", encoding="utf-8") as f:
                current_code = f.read()
                
            specific_bug = input("Feedback: ").strip()
            
            inputs = {
                "generated_code": current_code,
                "bug_report": specific_bug,
                "messages": [
                    HumanMessage(content=f"Codebase:\n\n{current_code}"),
                    HumanMessage(content=f"Adjustment: {specific_bug}")
                ]
            }
        else:
            inputs = {
                "messages": [HumanMessage(content=user_command)]
            }

        final_state = app.invoke(inputs, config={"recursion_limit": 40})

        raw_response = final_state.get("generated_code", "")
        code_match = re.search(r"```html\s(.*?)\s*```", raw_response, re.DOTALL)

        clean_code = code_match.group(1) if code_match else raw_response

        if clean_code and "<!-- Empty Init File -->" not in clean_code:
            with open("mini_game.html", "w", encoding="utf-8") as f:
                f.write(clean_code)
