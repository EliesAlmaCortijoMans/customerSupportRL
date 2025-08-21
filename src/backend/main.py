"""
FastAPI Backend for Customer Support Environment
"""

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional, Union
import asyncio
import json
import uuid
from datetime import datetime
import threading
import queue
import time
import logging
import os
from dotenv import load_dotenv

from backend.models import *
from backend.environment_manager import EnvironmentManager
from backend.training_manager import TrainingManager
from backend.websocket_manager import WebSocketManager

# Load environment variables
load_dotenv()

# Setup logging
log_level = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=getattr(logging, log_level))
logger = logging.getLogger(__name__)

# Environment configuration
ENVIRONMENT = os.getenv("ENVIRONMENT", "development")
DEBUG = os.getenv("DEBUG", "true").lower() == "true"

# CORS origins
if ENVIRONMENT == "production":
    allowed_origins = json.loads(os.getenv("ALLOWED_ORIGINS", '["*"]'))
else:
    allowed_origins = [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:3001",
        "*"
    ]

# Create FastAPI app
app = FastAPI(
    title="Customer Support RL Environment API",
    description="API for training and interacting with customer support reinforcement learning environments",
    version="1.0.0",
    debug=DEBUG
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize managers
env_manager = EnvironmentManager()
training_manager = TrainingManager()
websocket_manager = WebSocketManager()

# Store active sessions
active_sessions: Dict[str, Dict[str, Any]] = {}


@app.on_event("startup")
async def startup_event():
    """Initialize the application"""
    logger.info("Starting Customer Support RL Environment API")
    
    # Create default environments for each industry
    for industry in ["bfsi", "retail", "tech", "mixed"]:
        env_id = await env_manager.create_environment(
            CreateEnvironmentRequest(
                industry=industry,
                max_conversation_length=10,
                environment_type="standard"
            )
        )
        logger.info(f"Created default {industry} environment: {env_id}")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down Customer Support RL Environment API")
    await env_manager.cleanup_all()
    await training_manager.cleanup_all()


# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "active_environments": len(env_manager.environments),
        "active_training_sessions": len(training_manager.training_sessions),
        "active_websockets": len(websocket_manager.connections)
    }


# Environment Management Endpoints
@app.post("/environments", response_model=CreateEnvironmentResponse)
async def create_environment(request: CreateEnvironmentRequest):
    """Create a new environment"""
    try:
        env_id = await env_manager.create_environment(request)
        return CreateEnvironmentResponse(
            environment_id=env_id,
            status="created",
            message=f"Environment created successfully for {request.industry} industry"
        )
    except Exception as e:
        logger.error(f"Error creating environment: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/environments", response_model=List[EnvironmentInfo])
async def list_environments():
    """List all available environments"""
    try:
        return await env_manager.list_environments()
    except Exception as e:
        logger.error(f"Error listing environments: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/environments/{env_id}", response_model=EnvironmentInfo)
async def get_environment(env_id: str):
    """Get information about a specific environment"""
    try:
        env_info = await env_manager.get_environment_info(env_id)
        if not env_info:
            raise HTTPException(status_code=404, detail="Environment not found")
        return env_info
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting environment {env_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/environments/{env_id}")
async def delete_environment(env_id: str):
    """Delete an environment"""
    try:
        success = await env_manager.delete_environment(env_id)
        if not success:
            raise HTTPException(status_code=404, detail="Environment not found")
        return {"message": "Environment deleted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting environment {env_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Environment Interaction Endpoints
@app.post("/environments/{env_id}/reset", response_model=ResetResponse)
async def reset_environment(env_id: str):
    """Reset an environment"""
    try:
        result = await env_manager.reset_environment(env_id)
        if not result:
            raise HTTPException(status_code=404, detail="Environment not found")
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error resetting environment {env_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/environments/{env_id}/step", response_model=StepResponse)
async def step_environment(env_id: str, request: StepRequest):
    """Take a step in the environment"""
    try:
        result = await env_manager.step_environment(env_id, request.action)
        if not result:
            raise HTTPException(status_code=404, detail="Environment not found")
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error stepping environment {env_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/environments/{env_id}/metrics", response_model=EnvironmentMetrics)
async def get_environment_metrics(env_id: str):
    """Get metrics for an environment"""
    try:
        metrics = await env_manager.get_metrics(env_id)
        if metrics is None:
            raise HTTPException(status_code=404, detail="Environment not found")
        return metrics
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting metrics for environment {env_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Training Endpoints
@app.post("/training/start", response_model=TrainingResponse)
async def start_training(request: TrainingRequest, background_tasks: BackgroundTasks):
    """Start training a new model"""
    try:
        session_id = await training_manager.start_training(request, background_tasks)
        return TrainingResponse(
            session_id=session_id,
            status="started",
            message="Training started successfully"
        )
    except Exception as e:
        logger.error(f"Error starting training: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/training/sessions", response_model=List[TrainingSessionInfo])
async def list_training_sessions():
    """List all training sessions"""
    try:
        return await training_manager.list_sessions()
    except Exception as e:
        logger.error(f"Error listing training sessions: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/training/sessions/{session_id}", response_model=TrainingSessionInfo)
async def get_training_session(session_id: str):
    """Get information about a training session"""
    try:
        session_info = await training_manager.get_session_info(session_id)
        if not session_info:
            raise HTTPException(status_code=404, detail="Training session not found")
        return session_info
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting training session {session_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/training/sessions/{session_id}/stop")
async def stop_training(session_id: str):
    """Stop a training session"""
    try:
        success = await training_manager.stop_training(session_id)
        if not success:
            raise HTTPException(status_code=404, detail="Training session not found")
        return {"message": "Training stopped successfully"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error stopping training session {session_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/training/sessions/{session_id}/metrics", response_model=TrainingMetrics)
async def get_training_metrics(session_id: str):
    """Get training metrics"""
    try:
        metrics = await training_manager.get_metrics(session_id)
        if metrics is None:
            raise HTTPException(status_code=404, detail="Training session not found")
        return metrics
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting training metrics for session {session_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Model Management Endpoints
@app.get("/models", response_model=List[ModelInfo])
async def list_models():
    """List all trained models"""
    try:
        return await training_manager.list_models()
    except Exception as e:
        logger.error(f"Error listing models: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/models/{model_id}/evaluate", response_model=EvaluationResponse)
async def evaluate_model(model_id: str, request: EvaluationRequest):
    """Evaluate a trained model"""
    try:
        result = await training_manager.evaluate_model(model_id, request)
        if not result:
            raise HTTPException(status_code=404, detail="Model not found")
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error evaluating model {model_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/models/{model_id}/predict", response_model=PredictionResponse)
async def predict_action(model_id: str, request: PredictionRequest):
    """Get action prediction from a trained model"""
    try:
        result = await training_manager.predict_action(model_id, request)
        if not result:
            raise HTTPException(status_code=404, detail="Model not found")
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error predicting action with model {model_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Demo and Scenario Endpoints
@app.get("/scenarios", response_model=List[ScenarioInfo])
async def list_scenarios():
    """List available demo scenarios"""
    scenarios = [
        ScenarioInfo(
            id="bfsi_fraud_detection",
            name="BFSI Fraud Detection",
            description="Customer reports fraudulent transaction",
            industry="bfsi",
            difficulty="high",
            expected_strategies=["empathetic", "technical", "escalate"]
        ),
        ScenarioInfo(
            id="retail_product_return",
            name="Retail Product Return",
            description="Customer wants to return a defective product",
            industry="retail",
            difficulty="medium",
            expected_strategies=["apologetic", "quick_resolution", "product_recommend"]
        ),
        ScenarioInfo(
            id="tech_integration_help",
            name="Tech Integration Support",
            description="Customer needs help with API integration",
            industry="tech",
            difficulty="high",
            expected_strategies=["technical", "educational", "escalate"]
        ),
        ScenarioInfo(
            id="bfsi_investment_advice",
            name="BFSI Investment Advice",
            description="Customer seeking investment recommendations",
            industry="bfsi",
            difficulty="medium",
            expected_strategies=["educational", "product_recommend", "upsell"]
        ),
        ScenarioInfo(
            id="retail_discount_inquiry",
            name="Retail Discount Inquiry",
            description="Customer asking about available discounts",
            industry="retail",
            difficulty="low",
            expected_strategies=["product_recommend", "upsell", "empathetic"]
        )
    ]
    return scenarios


@app.post("/scenarios/{scenario_id}/run", response_model=ScenarioResult)
async def run_scenario(scenario_id: str, request: RunScenarioRequest):
    """Run a specific scenario"""
    try:
        # This would be implemented with predefined scenarios
        # For now, return a mock result
        return ScenarioResult(
            scenario_id=scenario_id,
            success=True,
            final_satisfaction=0.85,
            steps_taken=5,
            strategies_used=["empathetic", "technical", "quick_resolution"],
            feedback="Successfully resolved customer issue with high satisfaction"
        )
    except Exception as e:
        logger.error(f"Error running scenario {scenario_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Analytics Endpoints
@app.get("/analytics/overview", response_model=AnalyticsOverview)
async def get_analytics_overview():
    """Get overall analytics overview"""
    try:
        total_environments = len(env_manager.environments)
        total_training_sessions = len(training_manager.training_sessions)
        
        # Calculate aggregate metrics
        all_env_metrics = []
        for env_id in env_manager.environments:
            metrics = await env_manager.get_metrics(env_id)
            if metrics:
                all_env_metrics.append(metrics)
        
        avg_satisfaction = 0.0
        avg_resolution_time = 0.0
        total_episodes = 0
        
        if all_env_metrics:
            satisfactions = [m.average_satisfaction for m in all_env_metrics if m.average_satisfaction > 0]
            resolution_times = [m.average_resolution_time for m in all_env_metrics if m.average_resolution_time > 0]
            total_episodes = sum(m.total_episodes for m in all_env_metrics)
            
            if satisfactions:
                avg_satisfaction = sum(satisfactions) / len(satisfactions)
            if resolution_times:
                avg_resolution_time = sum(resolution_times) / len(resolution_times)
        
        return AnalyticsOverview(
            total_environments=total_environments,
            total_training_sessions=total_training_sessions,
            total_episodes=total_episodes,
            average_satisfaction=avg_satisfaction,
            average_resolution_time=avg_resolution_time,
            active_connections=len(websocket_manager.connections)
        )
    except Exception as e:
        logger.error(f"Error getting analytics overview: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# WebSocket Endpoint for Real-time Updates
@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    """WebSocket endpoint for real-time updates"""
    await websocket_manager.connect(websocket, client_id)
    try:
        while True:
            # Keep connection alive and handle incoming messages
            data = await websocket.receive_text()
            message = json.loads(data)
            
            # Handle different message types
            if message.get("type") == "subscribe":
                # Subscribe to updates for specific resources
                resource_type = message.get("resource_type")
                resource_id = message.get("resource_id")
                await websocket_manager.subscribe(client_id, resource_type, resource_id)
            
            elif message.get("type") == "unsubscribe":
                # Unsubscribe from updates
                resource_type = message.get("resource_type")
                resource_id = message.get("resource_id")
                await websocket_manager.unsubscribe(client_id, resource_type, resource_id)
            
            # Echo message back for testing
            await websocket_manager.send_to_client(client_id, {
                "type": "echo",
                "original_message": message,
                "timestamp": datetime.now().isoformat()
            })
            
    except WebSocketDisconnect:
        await websocket_manager.disconnect(client_id)
    except Exception as e:
        logger.error(f"WebSocket error for client {client_id}: {e}")
        await websocket_manager.disconnect(client_id)


# Utility Endpoints
@app.get("/config")
async def get_configuration():
    """Get API configuration"""
    return {
        "environment_types": ["standard", "vectorized", "advanced"],
        "industries": ["bfsi", "retail", "tech", "mixed"],
        "algorithms": ["ppo", "a2c", "dqn"],
        "response_strategies": [
            "empathetic", "technical", "escalate", "product_recommend",
            "apologetic", "educational", "quick_resolution", "upsell"
        ],
        "customer_tiers": ["basic", "premium", "vip"],
        "inquiry_types": {
            "bfsi": ["account_balance", "transaction_dispute", "loan_application", "fraud_report", "investment_advice"],
            "retail": ["order_status", "product_return", "product_recommendation", "shipping_issue", "discount_inquiry"],
            "tech": ["technical_support", "feature_request", "bug_report", "billing_issue", "integration_help"]
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
