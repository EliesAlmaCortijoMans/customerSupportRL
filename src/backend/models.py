"""
Pydantic models for the Customer Support Environment API
"""

from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Union
from datetime import datetime
from enum import Enum


class IndustryType(str, Enum):
    BFSI = "bfsi"
    RETAIL = "retail"
    TECH = "tech"
    MIXED = "mixed"


class EnvironmentType(str, Enum):
    STANDARD = "standard"
    VECTORIZED = "vectorized"
    ADVANCED = "advanced"


class AlgorithmType(str, Enum):
    PPO = "ppo"
    A2C = "a2c"
    DQN = "dqn"


class TrainingStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    STOPPED = "stopped"


# Environment Models
class CreateEnvironmentRequest(BaseModel):
    industry: IndustryType = Field(..., description="Industry focus for the environment")
    max_conversation_length: int = Field(10, description="Maximum conversation length", ge=1, le=20)
    environment_type: EnvironmentType = Field(EnvironmentType.STANDARD, description="Type of environment")
    num_envs: Optional[int] = Field(1, description="Number of parallel environments for vectorized type", ge=1, le=16)
    add_curriculum: bool = Field(False, description="Enable curriculum learning")
    add_noise: bool = Field(False, description="Add observation noise for robustness")


class CreateEnvironmentResponse(BaseModel):
    environment_id: str = Field(..., description="Unique environment identifier")
    status: str = Field(..., description="Creation status")
    message: str = Field(..., description="Status message")


class EnvironmentInfo(BaseModel):
    environment_id: str = Field(..., description="Unique environment identifier")
    industry: IndustryType = Field(..., description="Industry focus")
    environment_type: EnvironmentType = Field(..., description="Environment type")
    max_conversation_length: int = Field(..., description="Maximum conversation length")
    created_at: datetime = Field(..., description="Creation timestamp")
    num_envs: int = Field(1, description="Number of parallel environments")
    is_active: bool = Field(..., description="Whether environment is active")
    total_episodes: int = Field(0, description="Total episodes completed")
    current_episode: Optional[int] = Field(None, description="Current episode number")


class CustomerProfile(BaseModel):
    tier: str = Field(..., description="Customer tier (basic, premium, vip)")
    satisfaction_history: float = Field(..., description="Historical satisfaction score", ge=0, le=1)
    interaction_count: int = Field(..., description="Number of previous interactions", ge=0)
    lifetime_value: float = Field(..., description="Customer lifetime value", ge=0)
    industry_segment: str = Field(..., description="Industry segment")


class InquiryInfo(BaseModel):
    inquiry_type: str = Field(..., description="Type of customer inquiry")
    sentiment: str = Field(..., description="Customer sentiment")
    urgency: int = Field(..., description="Urgency level", ge=0, le=4)


class ResetResponse(BaseModel):
    observation: List[float] = Field(..., description="Initial observation")
    customer_profile: CustomerProfile = Field(..., description="Customer profile information")
    inquiry_info: InquiryInfo = Field(..., description="Inquiry information")
    episode_number: int = Field(..., description="Episode number")


class StepRequest(BaseModel):
    action: int = Field(..., description="Action to take", ge=0, le=7)


class StepResponse(BaseModel):
    observation: List[float] = Field(..., description="New observation")
    reward: float = Field(..., description="Reward received")
    done: bool = Field(..., description="Whether episode is done")
    truncated: bool = Field(..., description="Whether episode was truncated")
    info: Dict[str, Any] = Field(..., description="Additional information")


class EnvironmentMetrics(BaseModel):
    total_episodes: int = Field(..., description="Total episodes completed")
    average_satisfaction: float = Field(..., description="Average customer satisfaction", ge=0, le=1)
    average_resolution_time: float = Field(..., description="Average resolution time")
    satisfaction_distribution: List[int] = Field(..., description="Distribution of satisfaction scores")
    strategy_usage: Dict[str, int] = Field(..., description="Usage count for each strategy")
    success_rate: float = Field(..., description="Rate of successful resolutions", ge=0, le=1)
    escalation_rate: float = Field(..., description="Rate of escalations", ge=0, le=1)


# Training Models
class TrainingRequest(BaseModel):
    industry: IndustryType = Field(..., description="Industry focus for training")
    algorithm: AlgorithmType = Field(..., description="RL algorithm to use")
    total_timesteps: int = Field(50000, description="Total training timesteps", ge=1000, le=1000000)
    num_envs: int = Field(4, description="Number of parallel environments", ge=1, le=16)
    learning_rate: float = Field(3e-4, description="Learning rate", gt=0, le=1)
    eval_freq: int = Field(5000, description="Evaluation frequency", ge=100, le=50000)
    save_freq: int = Field(10000, description="Model save frequency", ge=1000, le=100000)
    model_params: Optional[Dict[str, Any]] = Field(None, description="Additional model parameters")


class TrainingResponse(BaseModel):
    session_id: str = Field(..., description="Unique training session identifier")
    status: str = Field(..., description="Training status")
    message: str = Field(..., description="Status message")


class TrainingSessionInfo(BaseModel):
    session_id: str = Field(..., description="Unique session identifier")
    industry: IndustryType = Field(..., description="Industry focus")
    algorithm: AlgorithmType = Field(..., description="RL algorithm")
    status: TrainingStatus = Field(..., description="Current training status")
    created_at: datetime = Field(..., description="Creation timestamp")
    started_at: Optional[datetime] = Field(None, description="Training start timestamp")
    completed_at: Optional[datetime] = Field(None, description="Training completion timestamp")
    total_timesteps: int = Field(..., description="Total training timesteps")
    current_timesteps: int = Field(0, description="Current timesteps completed")
    progress: float = Field(0, description="Training progress", ge=0, le=1)
    num_envs: int = Field(..., description="Number of parallel environments")
    current_reward: Optional[float] = Field(None, description="Current average reward")
    best_reward: Optional[float] = Field(None, description="Best average reward achieved")


class TrainingMetrics(BaseModel):
    session_id: str = Field(..., description="Training session identifier")
    timesteps_completed: int = Field(..., description="Timesteps completed")
    episodes_completed: int = Field(..., description="Episodes completed")
    current_reward: float = Field(..., description="Current average reward")
    best_reward: float = Field(..., description="Best average reward")
    current_satisfaction: float = Field(..., description="Current average satisfaction")
    current_length: float = Field(..., description="Current average episode length")
    reward_history: List[float] = Field(..., description="Reward progression")
    satisfaction_history: List[float] = Field(..., description="Satisfaction progression")
    length_history: List[float] = Field(..., description="Length progression")
    training_time: float = Field(..., description="Total training time in seconds")
    loss_history: Optional[List[float]] = Field(None, description="Loss progression if available")


# Model Management Models
class ModelInfo(BaseModel):
    model_id: str = Field(..., description="Unique model identifier")
    name: str = Field(..., description="Model name")
    industry: IndustryType = Field(..., description="Industry focus")
    algorithm: AlgorithmType = Field(..., description="RL algorithm")
    created_at: datetime = Field(..., description="Creation timestamp")
    training_session_id: str = Field(..., description="Source training session")
    total_timesteps: int = Field(..., description="Training timesteps")
    final_reward: float = Field(..., description="Final training reward")
    final_satisfaction: float = Field(..., description="Final satisfaction score")
    model_size_mb: float = Field(..., description="Model file size in MB")
    is_default: bool = Field(False, description="Whether this is a default model")


class EvaluationRequest(BaseModel):
    n_episodes: int = Field(100, description="Number of evaluation episodes", ge=1, le=1000)
    industry: Optional[IndustryType] = Field(None, description="Industry for evaluation")
    render: bool = Field(False, description="Whether to render episodes")
    deterministic: bool = Field(True, description="Whether to use deterministic actions")


class EvaluationResponse(BaseModel):
    model_id: str = Field(..., description="Model identifier")
    n_episodes: int = Field(..., description="Number of episodes evaluated")
    mean_reward: float = Field(..., description="Mean episode reward")
    std_reward: float = Field(..., description="Standard deviation of rewards")
    mean_satisfaction: float = Field(..., description="Mean customer satisfaction")
    std_satisfaction: float = Field(..., description="Standard deviation of satisfaction")
    mean_length: float = Field(..., description="Mean episode length")
    success_rate: float = Field(..., description="Success rate")
    escalation_rate: float = Field(..., description="Escalation rate")
    strategy_distribution: Dict[str, int] = Field(..., description="Strategy usage distribution")
    tier_performance: Dict[str, float] = Field(..., description="Performance by customer tier")
    evaluation_time: float = Field(..., description="Evaluation time in seconds")


class PredictionRequest(BaseModel):
    observation: List[float] = Field(..., description="Current observation")
    deterministic: bool = Field(True, description="Whether to use deterministic prediction")


class PredictionResponse(BaseModel):
    action: int = Field(..., description="Predicted action")
    action_probabilities: Optional[List[float]] = Field(None, description="Action probabilities if available")
    value: Optional[float] = Field(None, description="State value if available")
    strategy_name: str = Field(..., description="Human-readable strategy name")


# Scenario Models
class ScenarioInfo(BaseModel):
    id: str = Field(..., description="Unique scenario identifier")
    name: str = Field(..., description="Scenario name")
    description: str = Field(..., description="Scenario description")
    industry: IndustryType = Field(..., description="Industry focus")
    difficulty: str = Field(..., description="Difficulty level")
    expected_strategies: List[str] = Field(..., description="Expected optimal strategies")


class RunScenarioRequest(BaseModel):
    model_id: Optional[str] = Field(None, description="Model to use for scenario")
    manual_actions: Optional[List[int]] = Field(None, description="Manual actions to test")


class ScenarioResult(BaseModel):
    scenario_id: str = Field(..., description="Scenario identifier")
    success: bool = Field(..., description="Whether scenario was successful")
    final_satisfaction: float = Field(..., description="Final customer satisfaction")
    steps_taken: int = Field(..., description="Number of steps taken")
    strategies_used: List[str] = Field(..., description="Strategies used during scenario")
    feedback: str = Field(..., description="Detailed feedback on performance")


# Analytics Models
class AnalyticsOverview(BaseModel):
    total_environments: int = Field(..., description="Total number of environments")
    total_training_sessions: int = Field(..., description="Total training sessions")
    total_episodes: int = Field(..., description="Total episodes across all environments")
    average_satisfaction: float = Field(..., description="Overall average satisfaction")
    average_resolution_time: float = Field(..., description="Overall average resolution time")
    active_connections: int = Field(..., description="Number of active WebSocket connections")


class IndustryAnalytics(BaseModel):
    industry: IndustryType = Field(..., description="Industry")
    total_episodes: int = Field(..., description="Total episodes for this industry")
    average_satisfaction: float = Field(..., description="Average satisfaction")
    success_rate: float = Field(..., description="Success rate")
    most_common_inquiries: List[str] = Field(..., description="Most common inquiry types")
    optimal_strategies: Dict[str, str] = Field(..., description="Optimal strategies by inquiry type")


# WebSocket Models
class WebSocketMessage(BaseModel):
    type: str = Field(..., description="Message type")
    data: Dict[str, Any] = Field(..., description="Message data")
    timestamp: datetime = Field(default_factory=datetime.now, description="Message timestamp")


class SubscriptionRequest(BaseModel):
    resource_type: str = Field(..., description="Type of resource to subscribe to")
    resource_id: str = Field(..., description="Specific resource ID")


# Error Models
class ErrorResponse(BaseModel):
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Detailed error information")
    timestamp: datetime = Field(default_factory=datetime.now, description="Error timestamp")
