"""
Environment Manager for Customer Support Environments
"""

import asyncio
import uuid
from typing import Dict, Any, List, Optional
from datetime import datetime
import logging
import numpy as np

from backend.models import *
from environment.customer_support_env import make_customer_support_env
from environment.vectorized_env import make_vectorized_env, make_advanced_env, make_multi_industry_env

logger = logging.getLogger(__name__)


class EnvironmentManager:
    """Manages customer support environments and their lifecycle"""
    
    def __init__(self):
        self.environments: Dict[str, Dict[str, Any]] = {}
        self.lock = asyncio.Lock()
    
    async def create_environment(self, request: CreateEnvironmentRequest) -> str:
        """Create a new environment"""
        async with self.lock:
            env_id = str(uuid.uuid4())
            
            try:
                # Create environment based on type
                if request.environment_type == EnvironmentType.STANDARD:
                    env = make_customer_support_env(
                        industry=request.industry,
                        max_conversation_length=request.max_conversation_length
                    )
                elif request.environment_type == EnvironmentType.VECTORIZED:
                    env = make_vectorized_env(
                        num_envs=request.num_envs or 4,
                        industry=request.industry,
                        max_conversation_length=request.max_conversation_length
                    )
                elif request.environment_type == EnvironmentType.ADVANCED:
                    env = make_advanced_env(
                        industry=request.industry,
                        max_conversation_length=request.max_conversation_length,
                        add_curriculum=request.add_curriculum,
                        add_noise=request.add_noise
                    )
                else:
                    raise ValueError(f"Unknown environment type: {request.environment_type}")
                
                # Initialize environment
                if hasattr(env, 'reset'):
                    initial_obs, initial_info = env.reset()
                else:
                    initial_obs, initial_info = None, None
                
                # Store environment info
                self.environments[env_id] = {
                    "env": env,
                    "info": {
                        "environment_id": env_id,
                        "industry": request.industry,
                        "environment_type": request.environment_type,
                        "max_conversation_length": request.max_conversation_length,
                        "created_at": datetime.now(),
                        "num_envs": request.num_envs or 1,
                        "is_active": True,
                        "total_episodes": 0,
                        "current_episode": 0 if initial_obs is not None else None
                    },
                    "state": {
                        "current_observation": initial_obs.tolist() if initial_obs is not None else None,
                        "current_info": initial_info,
                        "done": False,
                        "episode_reward": 0.0,
                        "episode_length": 0,
                        "episode_history": []
                    },
                    "metrics": {
                        "total_episodes": 0,
                        "episode_rewards": [],
                        "episode_satisfactions": [],
                        "episode_lengths": [],
                        "strategy_counts": {},
                        "customer_tier_counts": {},
                        "inquiry_type_counts": {},
                        "resolution_type_counts": {}
                    }
                }
                
                logger.info(f"Created {request.environment_type} environment {env_id} for {request.industry} industry")
                return env_id
                
            except Exception as e:
                logger.error(f"Failed to create environment: {e}")
                # Clean up if partially created
                if env_id in self.environments:
                    del self.environments[env_id]
                raise
    
    async def list_environments(self) -> List[EnvironmentInfo]:
        """List all environments"""
        async with self.lock:
            env_list = []
            for env_id, env_data in self.environments.items():
                info = env_data["info"]
                env_list.append(EnvironmentInfo(**info))
            return env_list
    
    async def get_environment_info(self, env_id: str) -> Optional[EnvironmentInfo]:
        """Get information about a specific environment"""
        async with self.lock:
            if env_id not in self.environments:
                return None
            
            info = self.environments[env_id]["info"]
            return EnvironmentInfo(**info)
    
    async def delete_environment(self, env_id: str) -> bool:
        """Delete an environment"""
        async with self.lock:
            if env_id not in self.environments:
                return False
            
            try:
                env_data = self.environments[env_id]
                env = env_data["env"]
                
                # Close environment if it has close method
                if hasattr(env, 'close'):
                    env.close()
                
                del self.environments[env_id]
                logger.info(f"Deleted environment {env_id}")
                return True
                
            except Exception as e:
                logger.error(f"Error deleting environment {env_id}: {e}")
                return False
    
    async def reset_environment(self, env_id: str) -> Optional[ResetResponse]:
        """Reset an environment"""
        async with self.lock:
            if env_id not in self.environments:
                return None
            
            try:
                env_data = self.environments[env_id]
                env = env_data["env"]
                
                # Reset environment
                observation, info = env.reset()
                
                # Update state
                env_data["state"].update({
                    "current_observation": observation.tolist() if hasattr(observation, 'tolist') else observation,
                    "current_info": info,
                    "done": False,
                    "episode_reward": 0.0,
                    "episode_length": 0,
                    "episode_history": []
                })
                
                # Update episode count
                env_data["info"]["total_episodes"] += 1
                env_data["info"]["current_episode"] = env_data["info"]["total_episodes"]
                env_data["metrics"]["total_episodes"] += 1
                
                # Extract customer and inquiry info
                env_customer_profile = info.get("customer_profile")
                customer_profile = CustomerProfile(
                    tier=env_customer_profile.tier.name if env_customer_profile and hasattr(env_customer_profile, 'tier') else "basic",
                    satisfaction_history=env_customer_profile.satisfaction_history if env_customer_profile and hasattr(env_customer_profile, 'satisfaction_history') else 0.0,
                    interaction_count=env_customer_profile.interaction_count if env_customer_profile and hasattr(env_customer_profile, 'interaction_count') else 0,
                    lifetime_value=env_customer_profile.lifetime_value if env_customer_profile and hasattr(env_customer_profile, 'lifetime_value') else 0.0,
                    industry_segment=env_customer_profile.industry_segment if env_customer_profile and hasattr(env_customer_profile, 'industry_segment') else "unknown"
                )
                
                env_inquiry = info.get("inquiry")
                inquiry_info = InquiryInfo(
                    inquiry_type=env_inquiry.name if env_inquiry and hasattr(env_inquiry, 'name') else "unknown",
                    sentiment="neutral",  # Default if not available
                    urgency=2  # Default urgency
                )
                
                return ResetResponse(
                    observation=observation.tolist() if hasattr(observation, 'tolist') else observation,
                    customer_profile=customer_profile,
                    inquiry_info=inquiry_info,
                    episode_number=env_data["info"]["total_episodes"]
                )
                
            except Exception as e:
                logger.error(f"Error resetting environment {env_id}: {e}")
                return None
    
    async def step_environment(self, env_id: str, action: int) -> Optional[StepResponse]:
        """Take a step in an environment"""
        async with self.lock:
            if env_id not in self.environments:
                return None
            
            try:
                env_data = self.environments[env_id]
                env = env_data["env"]
                state = env_data["state"]
                
                if state["done"]:
                    # Episode is already done, need to reset first
                    return None
                
                # Take step
                observation, reward, done, truncated, info = env.step(action)
                
                # Update state
                state["current_observation"] = observation.tolist() if hasattr(observation, 'tolist') else observation
                state["current_info"] = info
                state["done"] = done or truncated
                state["episode_reward"] += reward
                state["episode_length"] += 1
                state["episode_history"].append({
                    "action": action,
                    "reward": reward,
                    "info": info
                })
                
                # Update metrics if episode is done
                if done or truncated:
                    metrics = env_data["metrics"]
                    metrics["episode_rewards"].append(state["episode_reward"])
                    metrics["episode_lengths"].append(state["episode_length"])
                    
                    if "satisfaction" in info:
                        metrics["episode_satisfactions"].append(info["satisfaction"])
                    
                    if "strategy_used" in info:
                        strategy = info["strategy_used"]
                        metrics["strategy_counts"][strategy] = metrics["strategy_counts"].get(strategy, 0) + 1
                    
                    if "customer_tier" in info:
                        tier = info["customer_tier"]
                        metrics["customer_tier_counts"][tier] = metrics["customer_tier_counts"].get(tier, 0) + 1
                    
                    if "inquiry_type" in info:
                        inquiry = info["inquiry_type"]
                        metrics["inquiry_type_counts"][inquiry] = metrics["inquiry_type_counts"].get(inquiry, 0) + 1
                    
                    if "done_reason" in info:
                        reason = info["done_reason"]
                        metrics["resolution_type_counts"][reason] = metrics["resolution_type_counts"].get(reason, 0) + 1
                
                return StepResponse(
                    observation=observation.tolist() if hasattr(observation, 'tolist') else observation,
                    reward=reward,
                    done=done,
                    truncated=truncated,
                    info=info
                )
                
            except Exception as e:
                logger.error(f"Error stepping environment {env_id}: {e}")
                return None
    
    async def get_metrics(self, env_id: str) -> Optional[EnvironmentMetrics]:
        """Get metrics for an environment"""
        async with self.lock:
            if env_id not in self.environments:
                return None
            
            try:
                env_data = self.environments[env_id]
                metrics = env_data["metrics"]
                
                # Calculate aggregate metrics
                total_episodes = metrics["total_episodes"]
                
                if total_episodes == 0:
                    return EnvironmentMetrics(
                        total_episodes=0,
                        average_satisfaction=0.0,
                        average_resolution_time=0.0,
                        satisfaction_distribution=[],
                        strategy_usage={},
                        success_rate=0.0,
                        escalation_rate=0.0
                    )
                
                # Average satisfaction
                satisfactions = metrics["episode_satisfactions"]
                avg_satisfaction = np.mean(satisfactions) if satisfactions else 0.0
                
                # Average resolution time (episode length)
                lengths = metrics["episode_lengths"]
                avg_length = np.mean(lengths) if lengths else 0.0
                
                # Satisfaction distribution
                if satisfactions:
                    hist, _ = np.histogram(satisfactions, bins=5, range=(0, 1))
                    satisfaction_dist = hist.tolist()
                else:
                    satisfaction_dist = [0] * 5
                
                # Success rate (satisfaction > 0.7)
                success_count = sum(1 for s in satisfactions if s > 0.7)
                success_rate = success_count / total_episodes if total_episodes > 0 else 0.0
                
                # Escalation rate
                escalation_count = metrics["resolution_type_counts"].get("escalated", 0)
                escalation_rate = escalation_count / total_episodes if total_episodes > 0 else 0.0
                
                return EnvironmentMetrics(
                    total_episodes=total_episodes,
                    average_satisfaction=avg_satisfaction,
                    average_resolution_time=avg_length,
                    satisfaction_distribution=satisfaction_dist,
                    strategy_usage=metrics["strategy_counts"],
                    success_rate=success_rate,
                    escalation_rate=escalation_rate
                )
                
            except Exception as e:
                logger.error(f"Error getting metrics for environment {env_id}: {e}")
                return None
    
    async def get_environment_state(self, env_id: str) -> Optional[Dict[str, Any]]:
        """Get current state of an environment"""
        async with self.lock:
            if env_id not in self.environments:
                return None
            
            return self.environments[env_id]["state"].copy()
    
    async def cleanup_all(self):
        """Cleanup all environments"""
        async with self.lock:
            for env_id in list(self.environments.keys()):
                await self.delete_environment(env_id)
            logger.info("Cleaned up all environments")
    
    async def get_environment_by_industry(self, industry: IndustryType) -> Optional[str]:
        """Get the first environment for a specific industry"""
        async with self.lock:
            for env_id, env_data in self.environments.items():
                if env_data["info"]["industry"] == industry:
                    return env_id
            return None
    
    async def create_demo_environment(self, industry: IndustryType) -> str:
        """Create a demo environment with predefined settings"""
        request = CreateEnvironmentRequest(
            industry=industry,
            max_conversation_length=10,
            environment_type=EnvironmentType.STANDARD
        )
        return await self.create_environment(request)
    
    async def batch_reset_environments(self, env_ids: List[str]) -> Dict[str, Optional[ResetResponse]]:
        """Reset multiple environments in batch"""
        results = {}
        for env_id in env_ids:
            results[env_id] = await self.reset_environment(env_id)
        return results
    
    async def get_aggregated_metrics(self) -> Dict[str, Any]:
        """Get aggregated metrics across all environments"""
        async with self.lock:
            total_episodes = 0
            all_satisfactions = []
            all_lengths = []
            all_strategy_counts = {}
            all_tier_counts = {}
            
            for env_data in self.environments.values():
                metrics = env_data["metrics"]
                total_episodes += metrics["total_episodes"]
                all_satisfactions.extend(metrics["episode_satisfactions"])
                all_lengths.extend(metrics["episode_lengths"])
                
                # Aggregate strategy counts
                for strategy, count in metrics["strategy_counts"].items():
                    all_strategy_counts[strategy] = all_strategy_counts.get(strategy, 0) + count
                
                # Aggregate tier counts
                for tier, count in metrics["customer_tier_counts"].items():
                    all_tier_counts[tier] = all_tier_counts.get(tier, 0) + count
            
            return {
                "total_episodes": total_episodes,
                "total_environments": len(self.environments),
                "average_satisfaction": np.mean(all_satisfactions) if all_satisfactions else 0.0,
                "average_length": np.mean(all_lengths) if all_lengths else 0.0,
                "strategy_distribution": all_strategy_counts,
                "tier_distribution": all_tier_counts
            }
