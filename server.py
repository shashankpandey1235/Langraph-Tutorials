import re
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_core.messages import HumanMessage
# Import your compiled LangGraph workflow instance from game_studio.py
from game_studio import app as game_studio_app

# 1. Initialize the FastAPI Framework
app = FastAPI(
    title="Multi-Agent Game Studio API Engine",
    description="Production backend running an autonomous LangGraph state machine for software generation."
)

# 2. Define Data Validation Models via Pydantic
class GameRequest(BaseModel):
    prompt: str

class GameResponse(BaseModel):
    html_code: str

# 3. Create the Generation API Endpoint
@app.post("/generate-game", response_model=GameResponse)
async def generate_game_endpoint(request: GameRequest):
    """
    Exposes the LangGraph Multi-Agent Team as a web service.
    Accepts text prompts and returns production-validated game code.
    """
    if not request.prompt.strip():
        raise HTTPException(status_code=400, detail="Prompt string cannot be empty.")
        
    try:
        print(f"\n📥 [API Request Received]: {request.prompt}")
        inputs = {"messages": [HumanMessage(content=request.prompt)]}
        
        # 4. Stream or invoke the LangGraph execution block
        final_state = game_studio_app.invoke(inputs, config={"recursion_limit": 30})
        
        # 5. Extract and parse the code asset out of the final graph state
        raw_response = final_state.get("generated_code", "")
        code_match = re.search(r"```html\s*(.*?)\s*```", raw_response, re.DOTALL)
        
        if not code_match:
            raise HTTPException(
                status_code=500, 
                detail="Multi-Agent pipeline executed, but QA node failed to extract valid HTML markdown."
            )
            
        clean_html = code_match.group(1)
        print("📤 [API Response Sent]: Clean game code successfully packaged!")
        return {"html_code": clean_html}
        
    except Exception as e:
        print(f"❌ [API Internal Error]: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Graph Execution Failure: {str(e)}")

# 4. Add a basic health check (Great for production monitors on servers like Render)
@app.get("/health")
async def health_check():
    return {"status": "healthy", "engine": "LangGraph Active"}

if __name__ == "__main__":
    import uvicorn
    # Start the server locally on port 8000
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
