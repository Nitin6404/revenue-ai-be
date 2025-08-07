
import os
from fastapi import FastAPI, HTTPException, Depends
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, Any
import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import agents
from agents.orchestrator import AgentOrchestrator

# Initialize FastAPI app
app = FastAPI(
    title="Power BI Destroyer API",
    description="A powerful analytics API with natural language processing capabilities",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Database configuration
DATABASE_URI = os.getenv("DATABASE_URL", "sqlite:///./test.db")

# Initialize the agent orchestrator
agent_orchestrator = None

def get_agent_orchestrator():
    """Dependency to get the agent orchestrator instance"""
    global agent_orchestrator
    if agent_orchestrator is None:
        agent_orchestrator = AgentOrchestrator(db_uri=DATABASE_URI)
    return agent_orchestrator

# Request models
class QueryRequest(BaseModel):
    query: str
    session_id: Optional[str] = None
    context: Optional[Dict[str, Any]] = None

class ForecastRequest(BaseModel):
    periods: int = 12
    freq: str = "M"
    metric: str = "revenue"
    filters: Optional[Dict[str, Any]] = None

# API Endpoints
@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "ok",
        "message": "Power BI Destroyer API is running",
        "version": "1.0.0"
    }

@app.post("/query")
async def process_query(
    request: QueryRequest,
    orchestrator: AgentOrchestrator = Depends(get_agent_orchestrator)
):
    """
    Process a natural language query and return the result.
    
    This endpoint accepts a natural language question and returns an analysis
    including data, visualizations, and insights.
    """
    try:
        result = await orchestrator.process_query(request.query)
        return {
            "status": "success",
            "data": result
        }
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/forecast")
async def get_forecast(
    request: ForecastRequest,
    orchestrator: AgentOrchestrator = Depends(get_agent_orchestrator)
):
    """
    Generate a revenue forecast for the specified parameters.
    
    This endpoint generates a time series forecast using the Prophet model.
    """
    try:
        forecast = orchestrator._forecast_revenue_tool(json.dumps({
            "periods": request.periods,
            "freq": request.freq,
            "metric": request.metric,
            "filters": request.filters or {}
        }))
        
        return {
            "status": "success",
            "data": json.loads(forecast)
        }
    except Exception as e:
        logger.error(f"Error generating forecast: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={"status": "error", "message": str(exc.detail)}
    )

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"status": "error", "message": "An unexpected error occurred"}
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
