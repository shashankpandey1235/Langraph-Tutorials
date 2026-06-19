import os 
import re 
from typing import TypedDict , Annotated
from dotenv import load_dotenv
from langchain_core.messages import AnyMessage , HumanMessage , AIMessage
from langgraph.graph import StateGraph , START , END
from langgraph.graph.message import add_messages
from langgraph.types import Command
from langchain_mistralai import ChatMistralAI
load_dotenv()

class StudioState(TypedDict):
    messages: Annotated[list[AnyMessage] , add_messages]
    game_design: str 
    generated_code: str 
    bug_report: str 
    iteration_count:int

model = ChatMistralAI(model="mistral-small-latest",temperature=0.2)

# Isolated Production worker node
def game_designer_node(state : StudioState):
    """
    Worker 1 : Translate abstract concepts into strict technical specifications.
    """
    print("[Designer]:Architecting game mechanics and structural loops...")

    system_prompt = (
        "You are an expert game designer, Expand the user's request into a comprehensive"
        "specification list for 2D browser mini-game. Detail canvas control (Arrow keys)"
        "winning/losing states , and gameplay scoring rules .Do not write code yet."
    )

    # FIX: Convert dictionary style format to a proper LangChain system tuple
    messages = [("system", system_prompt)] + state["messages"]
    response = model.invoke(messages)

    return {
        "messages": [response],
        "game_design":response.content,
        "iteration_count":0
    }

def developer_node(state: StudioState):
    """
    Worker 2 : Converts design blueprints into production-ready vanilla web code.
    """
    print(f"[Developer]: Writing application source code (Cycle {state['iteration_count'] + 1 })....")

    system_prompt = (
        "You are an elite Lead Software Engineer. Write a completely self-contained, standalone "
        "HTML file containing inline <style> CSS and <script> JavaScript for the requested game.\n\n"
        "CRITICAL RULES:\n"
        "1. Return ONLY the code inside a standard markdown code block (```html ... ```).\n"
        "2. Ensure all event listeners, logic loops, rendering contexts, and asset colors are explicitly defined.\n"
        "3. Ensure the game is visually polished, responsive, and immediate to play."
    )
    
    context = state["messages"]
    
    if state.get("bug_report"):
        feedback = f"CRITICAL BUG REPORT FROM PREVIOUS TRY: \n{state['bug_report']}\n Please re-write the code completely fixing the error"
        context = context + [HumanMessage(content=feedback)]
    else:
        context = context + [HumanMessage(content="Please build the web source code now based on the designer's specifications above.")]

    messages = [("system", system_prompt)] + context
    response = model.invoke(messages)

    return {
        "messages": [response],
        "generated_code":response.content,
        "iteration_count":state["iteration_count"] + 1
    }

def qa_enginner_node(state:StudioState):
    """
    Worker 3 : Reviews code quality, checks formatting constraints, and manage loop traffic.
    """
    print("[QA Enginner] : Reviewing code health and compilation bounds...")

    code = state["generated_code"]
    # 1. Structural Check: Ensure a valid markdown block exists
    if "```html" not in code :
        print("⚠️ [QA Warning]: Missing clean markdown block wrappers. Routing back for formatting...")
        return Command(
            update={"bug_report":"Error: Your output was not encapsulated in a ```html code block."},
            goto="developer_node"
        )
    # 2 . Syntax check : use LLM intelligence to read code logic flaws or missing tags
    system_prompt = (
        "You are a Senior QA Automation Engineer. Inspect the provided code string carefully.\n"
        "Verify: Are there open tags? Is the JavaScript event loop complete? Are variables defined correctly?\n"
        "If you find ANY error, issue, or unhandled crash condition, reply with a detailed paragraph starting with [BUGFOUND].\n"
        "If the script is pristine and ready to run, reply with exactly: [PASS]."
    )

    # FIX: Changed invalid array structure ["system", system_prompt] to proper LangChain tuple sequence
    verification = model.invoke([
       ("system", system_prompt),
       HumanMessage(content=f"Review this source code:\n\n{code}")
    ])
    
    # Infinite loop guard check 
    if state["iteration_count"] >= 3 :
        print("[QA Guardrail] : Max code cycles reached.Exporting asset as-is")
        return Command(goto=END)
    
    if "[BUGFOUND]" in verification.content:
        print("[QA Rejected]: Flaw caught! Feedback sent back to developer.")
        return Command(
            update={"bug_report":verification.content},
            goto="developer_node"
        )
    print("[QA Verified]: Code passed review standard completely!")
    return Command(goto=END)

# 3. Graph Compilation 
studio_graph = StateGraph(StudioState)

studio_graph.add_node("game_designer_node",game_designer_node)
studio_graph.add_node("developer_node",developer_node)
studio_graph.add_node("qa_engineer_node",qa_enginner_node)

studio_graph.add_edge(START , "game_designer_node")
studio_graph.add_edge("game_designer_node","developer_node")
studio_graph.add_edge("developer_node","qa_engineer_node")

app = studio_graph.compile()

# 4 . Runtime & File export Extractor
if __name__ == "__main__":
    # Safety Check: Create a dummy file if mini_game.html doesn't exist yet to prevent FileNotFoundError
    if not os.path.exists("mini_game.html"):
        with open("mini_game.html", "w", encoding="utf-8") as f:
            f.write("<!-- Empty Init File -->")

    with open("mini_game.html" , "r" , encoding="utf-8") as f :
        broken_code = f.read()

    bug_feedback = (
         "The game is rendering nicely, but it hits an immediate infinite game-over loop. "
        "When I press SPACE to restart, it instantly triggers 'GAME OVER!' again. "
        "Fix the restart function: make sure to explicitly reset the snake's direction vector, "
        "reset the coordinates to the dead center of the canvas, clear any active loop timers "
        "using cancelAnimationFrame, and reset the score variables before launching the draw loop."
    )

    inputs = {
        "messages": [
            HumanMessage(content=f"Here is the code to fix:\n\n{broken_code}"),
            HumanMessage(content=bug_feedback)
        ]
    }

    print("\n---Running Self-Correction Optimization Loop ----")
    final_state = app.invoke(inputs , config={"recursion_limit":30})

    # Extract and overwrite with the patched HTML file
    raw_response = final_state["generated_code"]
    code_match = re.search(r"```html\s(.*?)\s*```",raw_response, re.DOTALL)

    if code_match:
        with open("mini_game.html","w",encoding="utf-8") as f :
            f.write(code_match.group(1))
        print("\n Success : The fixed version has been compiled into 'mini_game.html'")
