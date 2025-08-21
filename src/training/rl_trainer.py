"""
Reinforcement Learning Training Module for Customer Support Environment
"""

from typing import Dict, Any, List, Optional, Callable
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import time
from datetime import datetime

try:
    from stable_baselines3 import PPO, A2C, DQN
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
    from stable_baselines3.common.logger import configure
    STABLE_BASELINES_AVAILABLE = True
except ImportError:
    STABLE_BASELINES_AVAILABLE = False
    print("Warning: stable-baselines3 not available. Limited functionality.")

from environment.customer_support_env import CustomerSupportEnv, make_customer_support_env
from environment.vectorized_env import make_vectorized_env, make_advanced_env


class TrainingMetricsCallback(BaseCallback):
    """Custom callback to track training metrics"""
    
    def __init__(self, eval_env, eval_freq: int = 1000, verbose: int = 1):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_satisfactions = []
        self.best_mean_reward = -np.inf
        
    def _on_step(self) -> bool:
        """Called after each step"""
        if self.n_calls % self.eval_freq == 0:
            self._evaluate_agent()
        return True
    
    def _evaluate_agent(self):
        """Evaluate agent performance"""
        n_eval_episodes = 10
        episode_rewards = []
        episode_satisfactions = []
        episode_lengths = []
        
        for _ in range(n_eval_episodes):
            obs, _ = self.eval_env.reset()
            done = False
            episode_reward = 0
            episode_length = 0
            
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, done, truncated, info = self.eval_env.step(action)
                episode_reward += reward
                episode_length += 1
                
                if done and "satisfaction" in info:
                    episode_satisfactions.append(info["satisfaction"])
            
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
        
        mean_reward = np.mean(episode_rewards)
        mean_satisfaction = np.mean(episode_satisfactions) if episode_satisfactions else 0
        mean_length = np.mean(episode_lengths)
        
        self.episode_rewards.append(mean_reward)
        self.episode_satisfactions.append(mean_satisfaction)
        self.episode_lengths.append(mean_length)
        
        if mean_reward > self.best_mean_reward:
            self.best_mean_reward = mean_reward
            if self.verbose > 0:
                print(f"New best mean reward: {mean_reward:.2f}")
        
        # Log metrics
        self.logger.record("eval/mean_reward", mean_reward)
        self.logger.record("eval/mean_satisfaction", mean_satisfaction)
        self.logger.record("eval/mean_length", mean_length)
        
        if self.verbose > 0:
            print(f"Eval - Reward: {mean_reward:.2f}, Satisfaction: {mean_satisfaction:.2f}, Length: {mean_length:.1f}")


class CustomerSupportTrainer:
    """Main trainer class for customer support RL agents"""
    
    def __init__(
        self,
        industry: str = "mixed",
        algorithm: str = "ppo",
        num_envs: int = 4,
        device: str = "auto"
    ):
        """
        Initialize trainer
        
        Args:
            industry: Industry focus ("bfsi", "retail", "tech", "mixed")
            algorithm: RL algorithm ("ppo", "a2c", "dqn")
            num_envs: Number of parallel environments
            device: Device for training ("auto", "cpu", "cuda")
        """
        if not STABLE_BASELINES_AVAILABLE:
            raise ImportError("stable-baselines3 is required for training")
        
        self.industry = industry
        self.algorithm = algorithm.lower()
        self.num_envs = num_envs
        self.device = device
        
        # Create environments
        self.train_env = self._create_train_env()
        self.eval_env = self._create_eval_env()
        
        # Initialize model
        self.model = None
        self.training_metrics = {
            "rewards": [],
            "satisfactions": [],
            "lengths": [],
            "training_time": [],
            "episodes": []
        }
        
        # Setup logging
        self.log_dir = Path("logs") / f"{self.industry}_{self.algorithm}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
    def _create_train_env(self):
        """Create training environment"""
        def env_fn():
            env = make_customer_support_env(industry=self.industry)
            return Monitor(env)
        
        if self.num_envs > 1:
            return SubprocVecEnv([env_fn for _ in range(self.num_envs)])
        else:
            return DummyVecEnv([env_fn])
    
    def _create_eval_env(self):
        """Create evaluation environment"""
        env = make_customer_support_env(industry=self.industry)
        return Monitor(env)
    
    def create_model(self, **model_kwargs):
        """Create RL model"""
        # Default hyperparameters
        default_params = {
            "ppo": {
                "learning_rate": 3e-4,
                "n_steps": 2048,
                "batch_size": 64,
                "n_epochs": 10,
                "gamma": 0.99,
                "gae_lambda": 0.95,
                "clip_range": 0.2,
                "ent_coef": 0.01,
                "vf_coef": 0.5,
                "max_grad_norm": 0.5,
                "policy_kwargs": {"net_arch": [64, 64]}
            },
            "a2c": {
                "learning_rate": 7e-4,
                "n_steps": 5,
                "gamma": 0.99,
                "gae_lambda": 1.0,
                "ent_coef": 0.01,
                "vf_coef": 0.25,
                "max_grad_norm": 0.5,
                "policy_kwargs": {"net_arch": [64, 64]}
            },
            "dqn": {
                "learning_rate": 1e-4,
                "buffer_size": 50000,
                "learning_starts": 1000,
                "batch_size": 32,
                "tau": 1.0,
                "gamma": 0.99,
                "train_freq": 4,
                "gradient_steps": 1,
                "target_update_interval": 1000,
                "policy_kwargs": {"net_arch": [64, 64]}
            }
        }
        
        # Merge default params with user params
        params = {**default_params[self.algorithm], **model_kwargs}
        
        # Create model
        if self.algorithm == "ppo":
            self.model = PPO("MlpPolicy", self.train_env, device=self.device, **params)
        elif self.algorithm == "a2c":
            self.model = A2C("MlpPolicy", self.train_env, device=self.device, **params)
        elif self.algorithm == "dqn":
            # DQN doesn't support vectorized environments directly
            if self.num_envs > 1:
                print("Warning: DQN doesn't support vectorized environments. Using single environment.")
                self.train_env = DummyVecEnv([lambda: make_customer_support_env(industry=self.industry)])
            self.model = DQN("MlpPolicy", self.train_env, device=self.device, **params)
        else:
            raise ValueError(f"Unsupported algorithm: {self.algorithm}")
        
        # Setup logger
        self.model.set_logger(configure(str(self.log_dir), ["stdout", "csv", "tensorboard"]))
        
        return self.model
    
    def train(
        self,
        total_timesteps: int = 100000,
        eval_freq: int = 5000,
        save_freq: int = 10000,
        **train_kwargs
    ):
        """Train the RL agent"""
        if self.model is None:
            self.create_model()
        
        # Setup callbacks
        callback = TrainingMetricsCallback(
            self.eval_env,
            eval_freq=eval_freq,
            verbose=1
        )
        
        # Train model
        print(f"Starting training with {self.algorithm.upper()} on {self.industry} industry")
        print(f"Total timesteps: {total_timesteps}")
        print(f"Number of environments: {self.num_envs}")
        
        start_time = time.time()
        
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=callback,
            **train_kwargs
        )
        
        training_time = time.time() - start_time
        
        # Save final model
        model_path = self.log_dir / "final_model"
        self.model.save(model_path)
        
        # Store training metrics
        self.training_metrics.update({
            "final_rewards": callback.episode_rewards,
            "final_satisfactions": callback.episode_satisfactions,
            "final_lengths": callback.episode_lengths,
            "total_training_time": training_time,
            "total_timesteps": total_timesteps
        })
        
        print(f"Training completed in {training_time:.2f} seconds")
        print(f"Model saved to {model_path}")
        
        return self.model
    
    def evaluate(self, n_episodes: int = 100, render: bool = False) -> Dict[str, Any]:
        """Evaluate trained model"""
        if self.model is None:
            raise ValueError("Model not trained yet. Call train() first.")
        
        episode_metrics = []
        
        for episode in range(n_episodes):
            obs, info = self.eval_env.reset()
            done = False
            episode_reward = 0
            episode_length = 0
            actions_taken = []
            
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, done, truncated, step_info = self.eval_env.step(action)
                
                episode_reward += reward
                episode_length += 1
                actions_taken.append(action)
                
                if render and episode < 5:  # Render first few episodes
                    self.eval_env.render()
            
            # Collect episode metrics
            episode_data = {
                "episode": episode,
                "reward": episode_reward,
                "length": episode_length,
                "satisfaction": step_info.get("satisfaction", 0),
                "customer_tier": step_info.get("customer_tier", "unknown"),
                "inquiry_type": step_info.get("inquiry_type", "unknown"),
                "done_reason": step_info.get("done_reason", "unknown"),
                "actions": actions_taken
            }
            episode_metrics.append(episode_data)
        
        # Calculate aggregate metrics
        rewards = [ep["reward"] for ep in episode_metrics]
        satisfactions = [ep["satisfaction"] for ep in episode_metrics]
        lengths = [ep["length"] for ep in episode_metrics]
        
        results = {
            "n_episodes": n_episodes,
            "mean_reward": np.mean(rewards),
            "std_reward": np.std(rewards),
            "mean_satisfaction": np.mean(satisfactions),
            "std_satisfaction": np.std(satisfactions),
            "mean_length": np.mean(lengths),
            "std_length": np.std(lengths),
            "success_rate": sum(1 for ep in episode_metrics if ep["satisfaction"] > 0.7) / n_episodes,
            "escalation_rate": sum(1 for ep in episode_metrics if ep["done_reason"] == "escalated") / n_episodes,
            "episode_details": episode_metrics
        }
        
        # Save evaluation results
        eval_path = self.log_dir / "evaluation_results.json"
        with open(eval_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        return results
    
    def create_visualizations(self, eval_results: Optional[Dict[str, Any]] = None):
        """Create training and evaluation visualizations"""
        if eval_results is None and self.model is not None:
            eval_results = self.evaluate(n_episodes=50, render=False)
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle(f'Customer Support Agent Training Results - {self.industry.upper()} ({self.algorithm.upper()})')
        
        # Training metrics (if available)
        if hasattr(self, 'training_metrics') and self.training_metrics.get("final_rewards"):
            # Reward progression
            axes[0, 0].plot(self.training_metrics["final_rewards"])
            axes[0, 0].set_title('Training Reward Progression')
            axes[0, 0].set_xlabel('Evaluation Step')
            axes[0, 0].set_ylabel('Mean Reward')
            
            # Satisfaction progression
            axes[0, 1].plot(self.training_metrics["final_satisfactions"])
            axes[0, 1].set_title('Customer Satisfaction Progression')
            axes[0, 1].set_xlabel('Evaluation Step')
            axes[0, 1].set_ylabel('Mean Satisfaction')
            
            # Episode length progression
            axes[0, 2].plot(self.training_metrics["final_lengths"])
            axes[0, 2].set_title('Episode Length Progression')
            axes[0, 2].set_xlabel('Evaluation Step')
            axes[0, 2].set_ylabel('Mean Length')
        
        if eval_results:
            # Satisfaction distribution
            satisfactions = [ep["satisfaction"] for ep in eval_results["episode_details"]]
            axes[1, 0].hist(satisfactions, bins=20, alpha=0.7, edgecolor='black')
            axes[1, 0].set_title('Customer Satisfaction Distribution')
            axes[1, 0].set_xlabel('Satisfaction Score')
            axes[1, 0].set_ylabel('Frequency')
            
            # Action distribution
            all_actions = []
            for ep in eval_results["episode_details"]:
                all_actions.extend(ep["actions"])
            
            if all_actions:
                action_counts = np.bincount(all_actions)
                action_names = [f"Action {i}" for i in range(len(action_counts))]
                axes[1, 1].bar(action_names, action_counts)
                axes[1, 1].set_title('Action Usage Distribution')
                axes[1, 1].set_xlabel('Action')
                axes[1, 1].set_ylabel('Count')
                axes[1, 1].tick_params(axis='x', rotation=45)
            
            # Performance by customer tier
            tier_performance = {}
            for ep in eval_results["episode_details"]:
                tier = ep["customer_tier"]
                if tier not in tier_performance:
                    tier_performance[tier] = []
                tier_performance[tier].append(ep["satisfaction"])
            
            if tier_performance:
                tiers = list(tier_performance.keys())
                avg_satisfactions = [np.mean(tier_performance[tier]) for tier in tiers]
                axes[1, 2].bar(tiers, avg_satisfactions)
                axes[1, 2].set_title('Performance by Customer Tier')
                axes[1, 2].set_xlabel('Customer Tier')
                axes[1, 2].set_ylabel('Average Satisfaction')
        
        plt.tight_layout()
        
        # Save visualization
        viz_path = self.log_dir / "training_visualization.png"
        plt.savefig(viz_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        return fig
    
    def save_model(self, path: Optional[str] = None):
        """Save trained model"""
        if self.model is None:
            raise ValueError("No model to save")
        
        if path is None:
            path = self.log_dir / "saved_model"
        
        self.model.save(path)
        print(f"Model saved to {path}")
    
    def load_model(self, path: str):
        """Load trained model"""
        if self.algorithm == "ppo":
            self.model = PPO.load(path, env=self.train_env)
        elif self.algorithm == "a2c":
            self.model = A2C.load(path, env=self.train_env)
        elif self.algorithm == "dqn":
            self.model = DQN.load(path, env=self.train_env)
        
        print(f"Model loaded from {path}")
        return self.model


def quick_train(
    industry: str = "mixed",
    algorithm: str = "ppo",
    timesteps: int = 50000,
    num_envs: int = 4
) -> CustomerSupportTrainer:
    """Quick training function for prototyping"""
    trainer = CustomerSupportTrainer(
        industry=industry,
        algorithm=algorithm,
        num_envs=num_envs
    )
    
    trainer.create_model()
    trainer.train(total_timesteps=timesteps)
    
    # Evaluate and visualize
    results = trainer.evaluate(n_episodes=50)
    trainer.create_visualizations(results)
    
    print("\n=== Training Summary ===")
    print(f"Algorithm: {algorithm.upper()}")
    print(f"Industry: {industry}")
    print(f"Mean Satisfaction: {results['mean_satisfaction']:.3f} ± {results['std_satisfaction']:.3f}")
    print(f"Success Rate: {results['success_rate']:.1%}")
    print(f"Mean Episode Length: {results['mean_length']:.1f}")
    print(f"Escalation Rate: {results['escalation_rate']:.1%}")
    
    return trainer
