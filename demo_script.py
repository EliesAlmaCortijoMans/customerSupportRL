#!/usr/bin/env python3
"""
Demo Script for Customer Support RL Environment
Showcases key features and capabilities of the system
"""

import sys
import time
import numpy as np
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

try:
    from environment.customer_support_env import make_customer_support_env
    from environment.vectorized_env import make_vectorized_env, make_advanced_env
    from training.rl_trainer import quick_train
    DEPENDENCIES_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Some dependencies not available: {e}")
    DEPENDENCIES_AVAILABLE = False

def print_banner(text, char="="):
    """Print a formatted banner"""
    print(f"\n{char * 60}")
    print(f" {text.center(58)} ")
    print(f"{char * 60}\n")

def demo_basic_environment():
    """Demonstrate basic environment functionality"""
    print_banner("Basic Environment Demo")
    
    # Create environment for each industry
    industries = ["bfsi", "retail", "tech"]
    
    for industry in industries:
        print(f"🏢 Testing {industry.upper()} Environment:")
        
        env = make_customer_support_env(industry=industry)
        obs, info = env.reset()
        
        print(f"   📊 Observation shape: {len(obs)}")
        print(f"   🎯 Action space: {env.action_space}")
        print(f"   📋 Customer: {info.get('customer_profile', {}).get('tier', {}).get('name', 'Unknown')} tier")
        print(f"   ❓ Inquiry: {info.get('inquiry', {}).get('name', 'Unknown')}")
        
        # Take a few random actions
        total_reward = 0
        for step in range(3):
            action = np.random.randint(0, 8)
            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward
            
            strategy_names = [
                "Empathetic", "Technical", "Escalate", "Product Recommend",
                "Apologetic", "Educational", "Quick Resolution", "Upsell"
            ]
            
            print(f"   Step {step + 1}: {strategy_names[action]} → Reward: {reward:.2f}")
            
            if done:
                satisfaction = info.get("satisfaction", 0)
                print(f"   ✅ Episode completed! Final satisfaction: {satisfaction:.2f}")
                break
        
        print(f"   📈 Total reward: {total_reward:.2f}\n")
        env.close()

def demo_vectorized_environment():
    """Demonstrate vectorized environment"""
    print_banner("Vectorized Environment Demo")
    
    print("🚀 Creating vectorized environment with 4 parallel environments...")
    
    vec_env = make_vectorized_env(num_envs=4, industry="mixed")
    
    print(f"   📊 Observation space: {vec_env.observation_space}")
    print(f"   🎯 Action space: {vec_env.action_space}")
    print(f"   🔗 Number of environments: 4")
    
    # Reset all environments
    observations, infos = vec_env.reset()
    print(f"\n   📋 Reset complete - got {len(observations)} observations")
    
    # Take synchronized actions
    actions = np.random.randint(0, 8, size=4)
    observations, rewards, dones, truncated, infos = vec_env.step(actions)
    
    print(f"   🎮 Took actions: {actions}")
    print(f"   💰 Rewards: {[f'{r:.2f}' for r in rewards]}")
    print(f"   ✅ Done flags: {dones}")
    
    # Get aggregated metrics
    metrics = vec_env.get_aggregated_metrics()
    print(f"\n   📊 Aggregated Metrics:")
    print(f"      Total episodes: {metrics.get('total_episodes', 0)}")
    print(f"      Avg satisfaction: {metrics.get('average_satisfaction', 0):.3f}")
    
    vec_env.close()

def demo_advanced_features():
    """Demonstrate advanced environment features"""
    print_banner("Advanced Features Demo")
    
    print("🧠 Creating advanced environment with curriculum learning...")
    
    env = make_advanced_env(
        industry="mixed",
        add_curriculum=True,
        add_noise=True
    )
    
    obs, info = env.reset()
    
    print(f"   📊 Observation shape: {len(obs)}")
    print(f"   🎓 Curriculum level: {env.get_curriculum_info()['curriculum_level']}")
    
    # Simulate some episodes to show curriculum progression
    for episode in range(3):
        obs, info = env.reset()
        episode_reward = 0
        
        for step in range(5):
            action = np.random.randint(0, 8)
            obs, reward, done, truncated, info = env.step(action)
            episode_reward += reward
            
            if done:
                satisfaction = info.get("satisfaction", 0)
                print(f"   Episode {episode + 1}: Reward {episode_reward:.2f}, Satisfaction {satisfaction:.2f}")
                break
    
    curriculum_info = env.get_curriculum_info()
    print(f"\n   🎓 Final curriculum info:")
    print(f"      Level: {curriculum_info['curriculum_level']}")
    print(f"      Episodes at level: {curriculum_info['episodes_at_level']}")
    print(f"      Recent performance: {curriculum_info['recent_avg_performance']:.3f}")
    
    env.close()

def demo_training_integration():
    """Demonstrate training integration"""
    print_banner("Training Integration Demo")
    
    if not DEPENDENCIES_AVAILABLE:
        print("❌ Training dependencies not available. Install stable-baselines3 to run training demo.")
        return
    
    print("🏋️ Starting quick training demo (small scale for demo purposes)...")
    
    try:
        # Quick training with minimal timesteps for demo
        trainer = quick_train(
            industry="mixed",
            algorithm="ppo",
            timesteps=1000,  # Very small for demo
            num_envs=2
        )
        
        print("   ✅ Training completed successfully!")
        print(f"   📁 Results saved in: {trainer.log_dir}")
        
        # Show some metrics
        if hasattr(trainer, 'training_metrics'):
            metrics = trainer.training_metrics
            print(f"   📊 Training metrics available")
    
    except Exception as e:
        print(f"   ⚠️ Training demo skipped due to: {e}")
        print("   💡 Run 'python run_server.py train' for full training")

def demo_business_scenarios():
    """Demonstrate business-relevant scenarios"""
    print_banner("Business Scenarios Demo", "=")
    
    scenarios = [
        {
            "name": "BFSI Fraud Alert",
            "industry": "bfsi",
            "description": "Customer reports suspicious transaction",
            "optimal_strategies": ["Empathetic", "Technical", "Escalate"]
        },
        {
            "name": "Retail Product Return",
            "industry": "retail", 
            "description": "Defective product needs to be returned",
            "optimal_strategies": ["Apologetic", "Quick Resolution", "Product Recommend"]
        },
        {
            "name": "Tech API Integration",
            "industry": "tech",
            "description": "Developer needs API authentication help",
            "optimal_strategies": ["Technical", "Educational", "Escalate"]
        }
    ]
    
    strategy_names = [
        "Empathetic", "Technical", "Escalate", "Product Recommend",
        "Apologetic", "Educational", "Quick Resolution", "Upsell"
    ]
    
    for scenario in scenarios:
        print(f"🎭 Scenario: {scenario['name']}")
        print(f"   📝 Description: {scenario['description']}")
        print(f"   🎯 Optimal strategies: {', '.join(scenario['optimal_strategies'])}")
        
        env = make_customer_support_env(industry=scenario["industry"])
        obs, info = env.reset()
        
        # Simulate using optimal strategies
        optimal_actions = []
        for strategy_name in scenario["optimal_strategies"][:3]:  # Use first 3
            if strategy_name in strategy_names:
                optimal_actions.append(strategy_names.index(strategy_name))
        
        total_reward = 0
        for i, action in enumerate(optimal_actions):
            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward
            
            print(f"   Step {i + 1}: {strategy_names[action]} → Reward: {reward:.2f}")
            
            if done:
                satisfaction = info.get("satisfaction", 0)
                print(f"   ✅ Resolution: {satisfaction:.2f} satisfaction")
                break
        
        print(f"   📈 Total reward: {total_reward:.2f}")
        print()
        env.close()

def demo_system_overview():
    """Show system capabilities overview"""
    print_banner("System Overview", "=")
    
    print("🔧 Customer Support RL Environment Capabilities:")
    print()
    print("📊 Environment Features:")
    print("   • Multi-industry support (BFSI, Retail, Tech)")
    print("   • 17-dimensional state space")
    print("   • 8 response strategies")
    print("   • Dynamic customer profiles")
    print("   • Realistic inquiry types")
    print()
    print("🚀 Advanced Features:")
    print("   • Vectorized parallel training")
    print("   • Curriculum learning")
    print("   • Real-time monitoring")
    print("   • WebSocket integration")
    print()
    print("🤖 Training Support:")
    print("   • PPO, A2C, DQN algorithms")
    print("   • Automatic evaluation")
    print("   • Model persistence")
    print("   • Progress tracking")
    print()
    print("🌐 Full-Stack Implementation:")
    print("   • FastAPI backend")
    print("   • React frontend")
    print("   • RESTful API")
    print("   • Real-time dashboard")
    print()
    print("💼 Business Applications:")
    print("   • Agent training and optimization")
    print("   • Customer satisfaction improvement")
    print("   • Support process automation")
    print("   • Performance analytics")

def main():
    """Run the complete demo"""
    print_banner("Customer Support RL Environment Demo", "*")
    print("🎯 Demonstrating Gymnasium-based customer support agent training system")
    print("⚡ This demo showcases the key features and capabilities")
    
    try:
        demo_system_overview()
        demo_basic_environment()
        demo_vectorized_environment()
        demo_advanced_features()
        demo_business_scenarios()
        demo_training_integration()
        
        print_banner("Demo Complete!", "*")
        print("✅ All demo scenarios completed successfully!")
        print()
        print("🚀 Next Steps:")
        print("   1. Start the full system: python run_server.py serve")
        print("   2. Run training: python run_server.py train")
        print("   3. Access web interface: http://localhost:3000")
        print("   4. View API docs: http://localhost:8000/docs")
        print()
        print("📖 For more information, see README.md and WRITEUP.md")
        
    except KeyboardInterrupt:
        print("\n\n⚠️ Demo interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Demo failed with error: {e}")
        print("💡 Check dependencies and try again")

if __name__ == "__main__":
    main()
