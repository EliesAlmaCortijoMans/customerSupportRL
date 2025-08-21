"""
Vectorized Customer Support Environment for parallel training
"""

import gymnasium as gym
from gymnasium.vector import AsyncVectorEnv, SyncVectorEnv
from typing import List, Dict, Any, Optional, Callable
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import multiprocessing as mp

from environment.customer_support_env import CustomerSupportEnv, make_customer_support_env


class CustomerSupportVectorEnv:
    """
    Vectorized wrapper for CustomerSupportEnv that enables parallel environment execution
    """
    
    def __init__(
        self,
        num_envs: int = 4,
        industry: str = "mixed",
        asynchronous: bool = True,
        max_conversation_length: int = 10,
        **env_kwargs
    ):
        """
        Initialize vectorized environment
        
        Args:
            num_envs: Number of parallel environments
            industry: Industry focus ("bfsi", "retail", "tech", "mixed")
            asynchronous: Whether to use async or sync vectorization
            max_conversation_length: Maximum conversation length per episode
            **env_kwargs: Additional environment arguments
        """
        self.num_envs = num_envs
        self.industry = industry
        self.asynchronous = asynchronous
        
        # Create environment factory
        def env_fn():
            return make_customer_support_env(
                industry=industry,
                max_conversation_length=max_conversation_length,
                **env_kwargs
            )
        
        # Create vectorized environment
        if asynchronous:
            self.env = AsyncVectorEnv([env_fn for _ in range(num_envs)])
        else:
            self.env = SyncVectorEnv([env_fn for _ in range(num_envs)])
            
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        
        # Metrics aggregation
        self.episode_metrics = []
        self.total_episodes = 0
        
    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        """Reset all environments"""
        observations, infos = self.env.reset(seed=seed, options=options)
        return observations, infos
    
    def step(self, actions: np.ndarray):
        """Step all environments"""
        observations, rewards, dones, truncated, infos = self.env.step(actions)
        
        # Collect metrics from completed episodes
        for i, (done, info) in enumerate(zip(dones, infos)):
            if done and "episode_satisfaction" in info:
                self.episode_metrics.append({
                    "env_id": i,
                    "satisfaction": info["episode_satisfaction"],
                    "length": info["episode_length"],
                    "customer_tier": info.get("customer_tier", "unknown"),
                    "inquiry_type": info.get("inquiry_type", "unknown"),
                    "strategy_used": info.get("strategy_used", "unknown"),
                    "done_reason": info.get("done_reason", "unknown")
                })
                self.total_episodes += 1
        
        return observations, rewards, dones, truncated, infos
    
    def close(self):
        """Close all environments"""
        self.env.close()
    
    def get_aggregated_metrics(self) -> Dict[str, Any]:
        """Get aggregated metrics across all environments"""
        if not self.episode_metrics:
            return {"total_episodes": 0}
        
        metrics = {
            "total_episodes": len(self.episode_metrics),
            "average_satisfaction": np.mean([m["satisfaction"] for m in self.episode_metrics]),
            "average_length": np.mean([m["length"] for m in self.episode_metrics]),
            "satisfaction_std": np.std([m["satisfaction"] for m in self.episode_metrics]),
            "length_std": np.std([m["length"] for m in self.episode_metrics]),
        }
        
        # Strategy distribution
        strategies = [m["strategy_used"] for m in self.episode_metrics]
        unique_strategies, counts = np.unique(strategies, return_counts=True)
        metrics["strategy_distribution"] = dict(zip(unique_strategies, counts.tolist()))
        
        # Customer tier distribution
        tiers = [m["customer_tier"] for m in self.episode_metrics]
        unique_tiers, counts = np.unique(tiers, return_counts=True)
        metrics["tier_distribution"] = dict(zip(unique_tiers, counts.tolist()))
        
        # Done reason distribution
        reasons = [m["done_reason"] for m in self.episode_metrics]
        unique_reasons, counts = np.unique(reasons, return_counts=True)
        metrics["resolution_distribution"] = dict(zip(unique_reasons, counts.tolist()))
        
        return metrics
    
    def render(self, env_id: int = 0):
        """Render specific environment"""
        # Note: This is a simplified version as AsyncVectorEnv doesn't support render directly
        print(f"Vectorized Environment Status:")
        print(f"Number of environments: {self.num_envs}")
        print(f"Total episodes completed: {self.total_episodes}")
        if self.episode_metrics:
            recent_satisfaction = np.mean([m["satisfaction"] for m in self.episode_metrics[-10:]])
            print(f"Recent average satisfaction: {recent_satisfaction:.3f}")


class MultiIndustryVectorEnv:
    """
    Specialized vectorized environment that trains on multiple industries simultaneously
    """
    
    def __init__(self, num_envs_per_industry: int = 2, **env_kwargs):
        """
        Initialize multi-industry environment
        
        Args:
            num_envs_per_industry: Number of environments per industry
            **env_kwargs: Additional environment arguments
        """
        self.industries = ["bfsi", "retail", "tech"]
        self.num_envs_per_industry = num_envs_per_industry
        self.total_envs = len(self.industries) * num_envs_per_industry
        
        # Create environments for each industry
        def env_fn(industry):
            return lambda: make_customer_support_env(industry=industry, **env_kwargs)
        
        env_fns = []
        self.env_industry_map = {}
        
        for i, industry in enumerate(self.industries):
            for j in range(num_envs_per_industry):
                env_idx = i * num_envs_per_industry + j
                env_fns.append(env_fn(industry))
                self.env_industry_map[env_idx] = industry
        
        self.env = AsyncVectorEnv(env_fns)
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        
        # Industry-specific metrics
        self.industry_metrics = {industry: [] for industry in self.industries}
        
    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        """Reset all environments"""
        return self.env.reset(seed=seed, options=options)
    
    def step(self, actions: np.ndarray):
        """Step all environments and collect industry-specific metrics"""
        observations, rewards, dones, truncated, infos = self.env.step(actions)
        
        # Collect industry-specific metrics
        for env_idx, (done, info) in enumerate(zip(dones, infos)):
            if done and "episode_satisfaction" in info:
                industry = self.env_industry_map[env_idx]
                self.industry_metrics[industry].append({
                    "satisfaction": info["episode_satisfaction"],
                    "length": info["episode_length"],
                    "strategy_used": info.get("strategy_used", "unknown"),
                    "inquiry_type": info.get("inquiry_type", "unknown")
                })
        
        return observations, rewards, dones, truncated, infos
    
    def get_industry_metrics(self) -> Dict[str, Dict[str, Any]]:
        """Get metrics broken down by industry"""
        results = {}
        
        for industry, metrics in self.industry_metrics.items():
            if not metrics:
                results[industry] = {"episodes": 0}
                continue
                
            results[industry] = {
                "episodes": len(metrics),
                "avg_satisfaction": np.mean([m["satisfaction"] for m in metrics]),
                "avg_length": np.mean([m["length"] for m in metrics]),
                "satisfaction_std": np.std([m["satisfaction"] for m in metrics]),
                "most_common_strategy": max(
                    set(m["strategy_used"] for m in metrics),
                    key=lambda x: sum(1 for m in metrics if m["strategy_used"] == x)
                ) if metrics else "none"
            }
        
        return results
    
    def close(self):
        """Close all environments"""
        self.env.close()


class AdvancedEnvironmentWrapper(gym.Wrapper):
    """
    Advanced wrapper that adds additional features to the base environment
    """
    
    def __init__(self, env, add_curriculum: bool = True, add_noise: bool = False):
        """
        Initialize advanced wrapper
        
        Args:
            env: Base environment to wrap
            add_curriculum: Whether to add curriculum learning
            add_noise: Whether to add observation noise for robustness
        """
        super().__init__(env)
        self.add_curriculum = add_curriculum
        self.add_noise = add_noise
        
        # Curriculum parameters
        self.curriculum_level = 0
        self.episodes_at_level = 0
        self.episodes_per_level = 100
        self.success_threshold = 0.7
        
        # Performance tracking for curriculum
        self.recent_performance = []
        self.performance_window = 20
        
    def reset(self, **kwargs):
        """Reset with curriculum adjustments"""
        observation, info = self.env.reset(**kwargs)
        
        # Curriculum learning: adjust difficulty
        if self.add_curriculum:
            self._adjust_curriculum()
        
        return self._add_noise(observation), info
    
    def step(self, action):
        """Step with additional features"""
        observation, reward, done, truncated, info = self.env.step(action)
        
        # Track performance for curriculum
        if done and "episode_satisfaction" in info:
            self.recent_performance.append(info["episode_satisfaction"])
            if len(self.recent_performance) > self.performance_window:
                self.recent_performance.pop(0)
            
            self.episodes_at_level += 1
        
        return self._add_noise(observation), reward, done, truncated, info
    
    def _adjust_curriculum(self):
        """Adjust curriculum based on performance"""
        if (len(self.recent_performance) >= self.performance_window and
            self.episodes_at_level >= self.episodes_per_level):
            
            avg_performance = np.mean(self.recent_performance)
            
            if avg_performance >= self.success_threshold:
                self.curriculum_level = min(3, self.curriculum_level + 1)
                self.episodes_at_level = 0
                self.recent_performance = []
                print(f"Curriculum advanced to level {self.curriculum_level}")
            elif avg_performance < 0.4 and self.curriculum_level > 0:
                self.curriculum_level = max(0, self.curriculum_level - 1)
                self.episodes_at_level = 0
                self.recent_performance = []
                print(f"Curriculum reduced to level {self.curriculum_level}")
    
    def _add_noise(self, observation):
        """Add observation noise for robustness training"""
        if self.add_noise:
            noise_scale = 0.01 * (1 + self.curriculum_level * 0.5)
            noise = np.random.normal(0, noise_scale, observation.shape)
            observation = observation + noise.astype(observation.dtype)
        
        return observation
    
    def get_curriculum_info(self):
        """Get curriculum learning information"""
        return {
            "curriculum_level": self.curriculum_level,
            "episodes_at_level": self.episodes_at_level,
            "recent_avg_performance": np.mean(self.recent_performance) if self.recent_performance else 0,
            "performance_samples": len(self.recent_performance)
        }


# Factory functions for easy environment creation
def make_vectorized_env(num_envs: int = 4, industry: str = "mixed", **kwargs):
    """Create vectorized customer support environment"""
    return CustomerSupportVectorEnv(num_envs=num_envs, industry=industry, **kwargs)


def make_multi_industry_env(num_envs_per_industry: int = 2, **kwargs):
    """Create multi-industry vectorized environment"""
    return MultiIndustryVectorEnv(num_envs_per_industry=num_envs_per_industry, **kwargs)


def make_advanced_env(industry: str = "mixed", add_curriculum: bool = True, add_noise: bool = False, **kwargs):
    """Create advanced environment with additional features"""
    base_env = make_customer_support_env(industry=industry, **kwargs)
    return AdvancedEnvironmentWrapper(base_env, add_curriculum=add_curriculum, add_noise=add_noise)
