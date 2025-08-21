#!/usr/bin/env python3
"""
Server startup script for Customer Support RL Environment
"""

import uvicorn
import typer
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

app = typer.Typer(help="Customer Support RL Environment Server")


@app.command()
def serve(
    host: str = typer.Option("0.0.0.0", help="Host to bind to"),
    port: int = typer.Option(8000, help="Port to bind to"),
    reload: bool = typer.Option(True, help="Enable auto-reload"),
    workers: int = typer.Option(1, help="Number of worker processes"),
    log_level: str = typer.Option("info", help="Log level")
):
    """Start the FastAPI server"""
    typer.echo(f"🚀 Starting Customer Support RL Environment API on {host}:{port}")
    
    if reload and workers > 1:
        typer.echo("⚠️  Warning: Auto-reload doesn't work with multiple workers. Setting workers=1")
        workers = 1
    
    uvicorn.run(
        "backend.main:app",
        host=host,
        port=port,
        reload=reload,
        workers=workers,
        log_level=log_level,
        app_dir="src"
    )


@app.command()
def train(
    industry: str = typer.Option("mixed", help="Industry to train for (bfsi/retail/tech/mixed)"),
    algorithm: str = typer.Option("ppo", help="RL algorithm (ppo/a2c/dqn)"),
    timesteps: int = typer.Option(50000, help="Total training timesteps"),
    num_envs: int = typer.Option(4, help="Number of parallel environments")
):
    """Quick training via CLI"""
    try:
        from training.rl_trainer import quick_train
        
        typer.echo(f"🎯 Starting training: {algorithm.upper()} on {industry} industry")
        typer.echo(f"📊 Timesteps: {timesteps}, Environments: {num_envs}")
        
        trainer = quick_train(
            industry=industry,
            algorithm=algorithm,
            timesteps=timesteps,
            num_envs=num_envs
        )
        
        typer.echo("✅ Training completed successfully!")
        typer.echo(f"📁 Model saved in: {trainer.log_dir}")
        
    except ImportError:
        typer.echo("❌ Training dependencies not available. Please install stable-baselines3")
        raise typer.Exit(1)
    except Exception as e:
        typer.echo(f"❌ Training failed: {e}")
        raise typer.Exit(1)


@app.command()
def demo(
    industry: str = typer.Option("mixed", help="Industry for demo (bfsi/retail/tech/mixed)"),
    episodes: int = typer.Option(5, help="Number of demo episodes")
):
    """Run interactive demo"""
    try:
        from environment.customer_support_env import make_customer_support_env
        import numpy as np
        
        typer.echo(f"🎮 Starting demo for {industry} industry")
        
        env = make_customer_support_env(industry=industry)
        
        for episode in range(episodes):
            typer.echo(f"\n📋 Episode {episode + 1}/{episodes}")
            obs, info = env.reset()
            env.render()
            
            done = False
            step = 0
            total_reward = 0
            
            while not done and step < 10:
                # Random action for demo
                action = np.random.randint(0, 8)
                obs, reward, done, truncated, info = env.step(action)
                total_reward += reward
                step += 1
                
                strategy_names = [
                    "Empathetic", "Technical", "Escalate", "Product Recommend",
                    "Apologetic", "Educational", "Quick Resolution", "Upsell"
                ]
                
                typer.echo(f"  Step {step}: {strategy_names[action]} -> Reward: {reward:.2f}")
                
                if done:
                    satisfaction = info.get("satisfaction", 0)
                    typer.echo(f"  ✅ Episode completed! Satisfaction: {satisfaction:.2f}")
                    break
            
            typer.echo(f"  📊 Total Reward: {total_reward:.2f}")
        
        env.close()
        typer.echo("\n🎉 Demo completed!")
        
    except Exception as e:
        typer.echo(f"❌ Demo failed: {e}")
        raise typer.Exit(1)


@app.command()
def test():
    """Run basic tests"""
    try:
        from environment.customer_support_env import make_customer_support_env
        
        typer.echo("🧪 Running basic tests...")
        
        # Test environment creation
        env = make_customer_support_env(industry="mixed")
        obs, info = env.reset()
        
        typer.echo(f"✅ Environment created - Observation shape: {len(obs)}")
        
        # Test step
        action = 0
        obs, reward, done, truncated, info = env.step(action)
        
        typer.echo(f"✅ Environment step - Reward: {reward}, Done: {done}")
        
        env.close()
        typer.echo("✅ All tests passed!")
        
    except Exception as e:
        typer.echo(f"❌ Tests failed: {e}")
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
