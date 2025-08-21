"""
Training Manager for RL Models
"""

import asyncio
import uuid
import threading
import time
import json
from typing import Dict, Any, List, Optional
from datetime import datetime
from pathlib import Path
import logging
import numpy as np
from concurrent.futures import ThreadPoolExecutor

from backend.models import *

try:
    from training.rl_trainer import CustomerSupportTrainer
    TRAINING_AVAILABLE = True
except ImportError:
    TRAINING_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("Training modules not available")

logger = logging.getLogger(__name__)


class TrainingManager:
    """Manages RL training sessions and models"""
    
    def __init__(self):
        self.training_sessions: Dict[str, Dict[str, Any]] = {}
        self.models: Dict[str, Dict[str, Any]] = {}
        self.lock = asyncio.Lock()
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.models_dir = Path("models")
        self.models_dir.mkdir(exist_ok=True)
        
        # Initialize with some default models if available
        self._initialize_default_models()
    
    def _initialize_default_models(self):
        """Initialize with default pre-trained models if available"""
        default_models = [
            {
                "model_id": "default_bfsi_ppo",
                "name": "Default BFSI PPO Model",
                "industry": IndustryType.BFSI,
                "algorithm": AlgorithmType.PPO,
                "is_default": True,
                "final_reward": 2.5,
                "final_satisfaction": 0.78
            },
            {
                "model_id": "default_retail_ppo",
                "name": "Default Retail PPO Model",
                "industry": IndustryType.RETAIL,
                "algorithm": AlgorithmType.PPO,
                "is_default": True,
                "final_reward": 2.3,
                "final_satisfaction": 0.75
            },
            {
                "model_id": "default_tech_ppo",
                "name": "Default Tech PPO Model",
                "industry": IndustryType.TECH,
                "algorithm": AlgorithmType.PPO,
                "is_default": True,
                "final_reward": 2.7,
                "final_satisfaction": 0.82
            }
        ]
        
        for model_info in default_models:
            self.models[model_info["model_id"]] = {
                "info": {
                    **model_info,
                    "created_at": datetime.now(),
                    "training_session_id": "default",
                    "total_timesteps": 50000,
                    "model_size_mb": 2.5
                },
                "trainer": None,  # Will be loaded on demand
                "model_path": None
            }
    
    async def start_training(self, request: TrainingRequest, background_tasks) -> str:
        """Start a new training session"""
        if not TRAINING_AVAILABLE:
            raise RuntimeError("Training functionality not available")
        
        async with self.lock:
            session_id = str(uuid.uuid4())
            
            # Create training session info
            session_info = {
                "session_id": session_id,
                "industry": request.industry,
                "algorithm": request.algorithm,
                "status": TrainingStatus.PENDING,
                "created_at": datetime.now(),
                "started_at": None,
                "completed_at": None,
                "total_timesteps": request.total_timesteps,
                "current_timesteps": 0,
                "progress": 0.0,
                "num_envs": request.num_envs,
                "current_reward": None,
                "best_reward": None
            }
            
            # Store training session
            self.training_sessions[session_id] = {
                "info": session_info,
                "request": request,
                "trainer": None,
                "metrics": {
                    "timesteps_completed": 0,
                    "episodes_completed": 0,
                    "current_reward": 0.0,
                    "best_reward": -float('inf'),
                    "current_satisfaction": 0.0,
                    "current_length": 0.0,
                    "reward_history": [],
                    "satisfaction_history": [],
                    "length_history": [],
                    "training_time": 0.0,
                    "loss_history": []
                },
                "stop_event": threading.Event()
            }
            
            # Start training in background
            background_tasks.add_task(self._run_training, session_id)
            
            logger.info(f"Started training session {session_id} for {request.industry} industry")
            return session_id
    
    async def _run_training(self, session_id: str):
        """Run training in a separate thread"""
        try:
            session_data = self.training_sessions[session_id]
            request = session_data["request"]
            
            # Update status to running
            session_data["info"]["status"] = TrainingStatus.RUNNING
            session_data["info"]["started_at"] = datetime.now()
            
            # Create trainer
            trainer = CustomerSupportTrainer(
                industry=request.industry.value,
                algorithm=request.algorithm.value,
                num_envs=request.num_envs,
                device="auto"
            )
            
            session_data["trainer"] = trainer
            
            # Prepare model parameters
            model_params = request.model_params or {}
            if request.learning_rate:
                model_params["learning_rate"] = request.learning_rate
            
            # Create model
            trainer.create_model(**model_params)
            
            # Training parameters
            train_params = {
                "total_timesteps": request.total_timesteps,
                "eval_freq": request.eval_freq,
                "save_freq": request.save_freq
            }
            
            # Run training with progress tracking
            await self._train_with_progress(session_id, trainer, train_params)
            
            # Training completed successfully
            if not session_data["stop_event"].is_set():
                session_data["info"]["status"] = TrainingStatus.COMPLETED
                session_data["info"]["completed_at"] = datetime.now()
                
                # Save final model
                model_id = await self._save_trained_model(session_id, trainer)
                session_data["model_id"] = model_id
                
                logger.info(f"Training session {session_id} completed successfully")
            
        except Exception as e:
            logger.error(f"Training session {session_id} failed: {e}")
            if session_id in self.training_sessions:
                self.training_sessions[session_id]["info"]["status"] = TrainingStatus.FAILED
                self.training_sessions[session_id]["error"] = str(e)
    
    async def _train_with_progress(self, session_id: str, trainer, train_params):
        """Run training with progress tracking"""
        session_data = self.training_sessions[session_id]
        stop_event = session_data["stop_event"]
        
        def training_callback(locals_, globals_):
            """Callback to track training progress"""
            if stop_event.is_set():
                return False  # Stop training
            
            # Update progress
            timesteps = locals_.get("num_timesteps", 0)
            total_timesteps = train_params["total_timesteps"]
            progress = timesteps / total_timesteps
            
            session_data["info"]["current_timesteps"] = timesteps
            session_data["info"]["progress"] = progress
            session_data["metrics"]["timesteps_completed"] = timesteps
            
            # Extract metrics if available
            if hasattr(trainer, 'model') and hasattr(trainer.model, 'logger'):
                try:
                    # This would extract metrics from the training logs
                    # Implementation depends on stable-baselines3 version
                    pass
                except:
                    pass
            
            return True
        
        # Run training in executor to avoid blocking
        await asyncio.get_event_loop().run_in_executor(
            self.executor,
            lambda: trainer.train(**train_params)
        )
    
    async def _save_trained_model(self, session_id: str, trainer) -> str:
        """Save a trained model"""
        model_id = f"model_{session_id}_{int(time.time())}"
        session_data = self.training_sessions[session_id]
        request = session_data["request"]
        
        # Save model files
        model_path = self.models_dir / model_id
        model_path.mkdir(exist_ok=True)
        
        try:
            trainer.save_model(str(model_path / "model"))
            
            # Get final metrics
            eval_results = trainer.evaluate(n_episodes=50, render=False)
            
            # Create model info
            model_info = {
                "model_id": model_id,
                "name": f"{request.industry.value.upper()} {request.algorithm.value.upper()} Model",
                "industry": request.industry,
                "algorithm": request.algorithm,
                "created_at": datetime.now(),
                "training_session_id": session_id,
                "total_timesteps": request.total_timesteps,
                "final_reward": eval_results["mean_reward"],
                "final_satisfaction": eval_results["mean_satisfaction"],
                "model_size_mb": self._get_model_size(model_path),
                "is_default": False
            }
            
            # Store model
            self.models[model_id] = {
                "info": model_info,
                "trainer": trainer,
                "model_path": str(model_path / "model")
            }
            
            # Save model metadata
            with open(model_path / "metadata.json", 'w') as f:
                json.dump(model_info, f, indent=2, default=str)
            
            logger.info(f"Saved model {model_id} from training session {session_id}")
            return model_id
            
        except Exception as e:
            logger.error(f"Failed to save model from session {session_id}: {e}")
            raise
    
    def _get_model_size(self, model_path: Path) -> float:
        """Get model size in MB"""
        try:
            total_size = sum(f.stat().st_size for f in model_path.rglob('*') if f.is_file())
            return total_size / (1024 * 1024)  # Convert to MB
        except:
            return 0.0
    
    async def list_sessions(self) -> List[TrainingSessionInfo]:
        """List all training sessions"""
        async with self.lock:
            sessions = []
            for session_data in self.training_sessions.values():
                sessions.append(TrainingSessionInfo(**session_data["info"]))
            return sessions
    
    async def get_session_info(self, session_id: str) -> Optional[TrainingSessionInfo]:
        """Get information about a training session"""
        async with self.lock:
            if session_id not in self.training_sessions:
                return None
            
            info = self.training_sessions[session_id]["info"]
            return TrainingSessionInfo(**info)
    
    async def stop_training(self, session_id: str) -> bool:
        """Stop a training session"""
        async with self.lock:
            if session_id not in self.training_sessions:
                return False
            
            session_data = self.training_sessions[session_id]
            
            if session_data["info"]["status"] == TrainingStatus.RUNNING:
                session_data["stop_event"].set()
                session_data["info"]["status"] = TrainingStatus.STOPPED
                session_data["info"]["completed_at"] = datetime.now()
                logger.info(f"Stopped training session {session_id}")
            
            return True
    
    async def get_metrics(self, session_id: str) -> Optional[TrainingMetrics]:
        """Get training metrics"""
        async with self.lock:
            if session_id not in self.training_sessions:
                return None
            
            session_data = self.training_sessions[session_id]
            metrics_data = session_data["metrics"]
            
            return TrainingMetrics(
                session_id=session_id,
                **metrics_data
            )
    
    async def list_models(self) -> List[ModelInfo]:
        """List all trained models"""
        async with self.lock:
            models = []
            for model_data in self.models.values():
                models.append(ModelInfo(**model_data["info"]))
            return models
    
    async def evaluate_model(self, model_id: str, request: EvaluationRequest) -> Optional[EvaluationResponse]:
        """Evaluate a trained model"""
        async with self.lock:
            if model_id not in self.models:
                return None
            
            model_data = self.models[model_id]
            trainer = model_data.get("trainer")
            
            if not trainer:
                # Try to load model if not in memory
                trainer = await self._load_model(model_id)
                if not trainer:
                    return None
            
            try:
                # Run evaluation in executor
                eval_results = await asyncio.get_event_loop().run_in_executor(
                    self.executor,
                    lambda: trainer.evaluate(
                        n_episodes=request.n_episodes,
                        render=request.render
                    )
                )
                
                return EvaluationResponse(
                    model_id=model_id,
                    n_episodes=request.n_episodes,
                    mean_reward=eval_results["mean_reward"],
                    std_reward=eval_results["std_reward"],
                    mean_satisfaction=eval_results["mean_satisfaction"],
                    std_satisfaction=eval_results["std_satisfaction"],
                    mean_length=eval_results["mean_length"],
                    success_rate=eval_results["success_rate"],
                    escalation_rate=eval_results["escalation_rate"],
                    strategy_distribution=eval_results.get("strategy_distribution", {}),
                    tier_performance=eval_results.get("tier_performance", {}),
                    evaluation_time=0.0  # Would be measured in real implementation
                )
                
            except Exception as e:
                logger.error(f"Error evaluating model {model_id}: {e}")
                return None
    
    async def predict_action(self, model_id: str, request: PredictionRequest) -> Optional[PredictionResponse]:
        """Get action prediction from a model"""
        async with self.lock:
            if model_id not in self.models:
                return None
            
            model_data = self.models[model_id]
            trainer = model_data.get("trainer")
            
            if not trainer:
                trainer = await self._load_model(model_id)
                if not trainer:
                    return None
            
            try:
                observation = np.array(request.observation, dtype=np.float32)
                
                # Get prediction
                action, action_probs = trainer.model.predict(
                    observation,
                    deterministic=request.deterministic
                )
                
                # Map action to strategy name
                strategy_names = [
                    "empathetic", "technical", "escalate", "product_recommend",
                    "apologetic", "educational", "quick_resolution", "upsell"
                ]
                strategy_name = strategy_names[action] if action < len(strategy_names) else "unknown"
                
                return PredictionResponse(
                    action=int(action),
                    action_probabilities=action_probs.tolist() if action_probs is not None else None,
                    value=None,  # Would extract from value function if available
                    strategy_name=strategy_name
                )
                
            except Exception as e:
                logger.error(f"Error predicting with model {model_id}: {e}")
                return None
    
    async def _load_model(self, model_id: str):
        """Load a model from disk"""
        if not TRAINING_AVAILABLE:
            return None
        
        try:
            model_data = self.models[model_id]
            model_path = model_data.get("model_path")
            
            if not model_path or not Path(model_path).exists():
                logger.warning(f"Model file not found for {model_id}")
                return None
            
            # Create trainer and load model
            info = model_data["info"]
            trainer = CustomerSupportTrainer(
                industry=info["industry"].value,
                algorithm=info["algorithm"].value,
                num_envs=1
            )
            trainer.load_model(model_path)
            
            # Cache trainer
            model_data["trainer"] = trainer
            
            return trainer
            
        except Exception as e:
            logger.error(f"Error loading model {model_id}: {e}")
            return None
    
    async def cleanup_all(self):
        """Cleanup all training sessions and models"""
        async with self.lock:
            # Stop all running training sessions
            for session_id, session_data in self.training_sessions.items():
                if session_data["info"]["status"] == TrainingStatus.RUNNING:
                    session_data["stop_event"].set()
            
            # Clear all sessions and models
            self.training_sessions.clear()
            self.models.clear()
            
            logger.info("Cleaned up all training sessions and models")
    
    async def get_model_by_industry(self, industry: IndustryType) -> Optional[str]:
        """Get the best model for a specific industry"""
        async with self.lock:
            best_model_id = None
            best_satisfaction = 0.0
            
            for model_id, model_data in self.models.items():
                info = model_data["info"]
                if info["industry"] == industry and info["final_satisfaction"] > best_satisfaction:
                    best_satisfaction = info["final_satisfaction"]
                    best_model_id = model_id
            
            return best_model_id
