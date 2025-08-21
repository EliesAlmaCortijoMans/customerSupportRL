# Customer Support RL Environment

A comprehensive reinforcement learning environment for training customer support agents using Gymnasium. This project evaluates Gymnasium's potential for modeling interaction-based workflows relevant to UltraLab's products.

## 🎯 Project Overview

This prototype demonstrates a sophisticated customer support training environment that simulates realistic customer service interactions across multiple industries (BFSI, Retail, Tech). The system enables training of RL agents to optimize customer satisfaction, resolution efficiency, and business outcomes.

### Key Features

- **Multi-Industry Support**: Specialized scenarios for Banking/Financial Services, Retail, and Technology sectors
- **Sophisticated State Space**: Customer sentiment, tier, inquiry type, conversation context
- **Business-Relevant Actions**: 8 response strategies from empathetic to technical approaches
- **Comprehensive Reward System**: Multi-component rewards based on satisfaction, efficiency, and business impact
- **Vectorized Training**: Parallel environment execution for faster training
- **Real-Time Monitoring**: WebSocket-based live updates and progress tracking
- **Modern Web Interface**: React-based dashboard for environment management and analytics

## 🏗️ Architecture

### Environment Components

- **Core Environment** (`src/environment/customer_support_env.py`): Main Gymnasium environment
- **Vectorized Wrapper** (`src/environment/vectorized_env.py`): Parallel environment execution
- **Training Integration** (`src/training/rl_trainer.py`): RL algorithms and training utilities

### Backend Services

- **FastAPI Server** (`src/backend/main.py`): RESTful API and WebSocket endpoints
- **Environment Manager** (`src/backend/environment_manager.py`): Environment lifecycle management
- **Training Manager** (`src/backend/training_manager.py`): Training session orchestration
- **WebSocket Manager** (`src/backend/websocket_manager.py`): Real-time communication

### Frontend Application

- **React Dashboard** (`frontend/src/`): Modern web interface with real-time updates
- **Environment Management**: Create, monitor, and interact with environments
- **Training Interface**: Start, monitor, and analyze training sessions
- **Analytics Dashboard**: Performance metrics and trend analysis
- **Scenario Testing**: Pre-built test scenarios for model evaluation

## 🚀 Quick Start

### Prerequisites

```bash
# Python 3.8+ with pip
# Node.js 16+ with npm
```

### Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd GymnasiumFinal
```

2. **Install Python dependencies**
```bash
pip install -r requirements.txt
```

3. **Install frontend dependencies**
```bash
cd frontend
npm install
cd ..
```

### Running the Application

1. **Start the backend server**
```bash


```

2. **Start the frontend (in a new terminal)**
```bash
cd frontend
npm start
```

3. **Access the application**
- Frontend: http://localhost:3000
- Backend API: http://localhost:8000
- API Documentation: http://localhost:8000/docs

## 🎮 Usage Examples

### Quick Demo

Run a simple demo to see the environment in action:

```bash
python run_server.py demo --industry=bfsi --episodes=5
```

### Training a Model

Start training via CLI:

```bash
python run_server.py train --industry=mixed --algorithm=ppo --timesteps=50000
```

Or use the web interface to start training with custom parameters.

### Testing the Environment

```bash
python run_server.py test
```

## 📊 Environment Details

### State Space (17-dimensional)

| Component | Description | Range |
|-----------|-------------|--------|
| Inquiry Type | Customer inquiry category | 0-14 (15 types) |
| Sentiment | Customer emotional state | 0-4 (5 levels) |
| Urgency | Issue urgency level | 0-4 |
| Conversation Length | Current conversation turns | 0-10 |
| Customer Tier | Value tier (Basic/Premium/VIP) | 0-2 |
| Previous Satisfaction | Historical satisfaction | 0-1 |
| Time in Conversation | Normalized conversation time | 0-1 |
| Context Vector | Encoded conversation state | 10-dimensional |

### Action Space

| Action | Strategy | Description |
|--------|----------|-------------|
| 0 | Empathetic | Focus on emotional connection |
| 1 | Technical | Provide detailed technical solution |
| 2 | Escalate | Transfer to human agent |
| 3 | Product Recommend | Suggest products/services |
| 4 | Apologetic | Focus on apology and service recovery |
| 5 | Educational | Teach about features/processes |
| 6 | Quick Resolution | Fast, efficient problem solving |
| 7 | Upsell | Attempt to sell additional services |

### Reward Function

Multi-component reward system:

- **Satisfaction Reward**: 0 to +2.0 based on customer satisfaction improvement
- **Efficiency Reward**: +0.1 to +0.5 for faster resolution
- **Business Impact**: +0.0 to +1.0 based on customer tier and outcome
- **Escalation Penalty**: -0.3 for escalations (sometimes necessary)

## 🏭 Industry Scenarios

### BFSI (Banking & Financial Services)

- Account balance inquiries
- Transaction disputes
- Fraud reporting
- Loan applications
- Investment advice

### Retail & E-commerce

- Order status tracking
- Product returns
- Shipping issues
- Discount inquiries
- Product recommendations

### Technology & SaaS

- Technical support
- API integration help
- Billing disputes
- Bug reports
- Feature requests

## 🤖 Supported Algorithms

| Algorithm | Description | Best For |
|-----------|-------------|----------|
| **PPO** | Proximal Policy Optimization | Default choice, stable training |
| **A2C** | Advantage Actor-Critic | Faster training, simpler scenarios |
| **DQN** | Deep Q-Network | Value-based approach |

## 📈 Performance Metrics

### Training Metrics

- Episode rewards and satisfaction scores
- Training loss and convergence
- Strategy usage distribution
- Customer tier performance

### Business Metrics

- Average customer satisfaction
- Resolution efficiency (steps per episode)
- Escalation rates
- Success rates by industry

## 🔗 API Reference

### Core Endpoints

```
Environment Management:
GET    /environments              - List environments
POST   /environments              - Create environment
DELETE /environments/{id}         - Delete environment
POST   /environments/{id}/reset   - Reset environment
POST   /environments/{id}/step    - Take action

Training Management:
GET    /training/sessions         - List training sessions
POST   /training/start            - Start training
POST   /training/sessions/{id}/stop - Stop training
GET    /training/sessions/{id}/metrics - Get metrics

Model Operations:
GET    /models                    - List models
POST   /models/{id}/evaluate      - Evaluate model
POST   /models/{id}/predict       - Get predictions

Analytics:
GET    /analytics/overview        - System overview
GET    /config                    - Configuration
GET    /health                    - Health check
```

### WebSocket Events

```
ws://localhost:8000/ws/{client_id}

Events:
- environment_update: Real-time environment changes
- training_update: Training progress and completion
- model_update: Model creation and evaluation
- system_alert: System-wide notifications
```

## 📋 Key Findings

### Gymnasium Strengths

✅ **Excellent Framework Structure**: Well-designed API that's easy to extend  
✅ **Strong Ecosystem**: Seamless integration with stable-baselines3  
✅ **Vectorization Support**: Built-in parallel environment execution  
✅ **Flexible Spaces**: Supports complex business logic and state representations  

### Potential Limitations

⚠️ **Learning Curve**: Requires RL expertise for optimal environment design  
⚠️ **Training Time**: Complex environments need significant computational resources  
⚠️ **Debugging Complexity**: Reward shaping and tuning can be challenging  

### Recommendations

Gymnasium is **well-suited** for UltraLab's prototyping needs, particularly for:

- Agent-user interaction modeling
- Behavior-based evaluation systems  
- Adaptive testing and optimization workflows

## 🛠️ Development

### Project Structure

```
GymnasiumFinal/
├── src/
│   ├── environment/          # Gymnasium environments
│   ├── training/            # RL training utilities
│   └── backend/             # FastAPI server
├── frontend/                # React application
├── models/                  # Saved models
├── logs/                    # Training logs
├── requirements.txt         # Python dependencies
├── run_server.py           # CLI interface
└── README.md               # This file
```

### Adding New Industries

1. Extend `InquiryType` enum in `customer_support_env.py`
2. Update industry-specific inquiry distributions
3. Add strategy effectiveness mappings
4. Update frontend industry options

### Custom Training Algorithms

1. Implement algorithm in `training/rl_trainer.py`
2. Add to supported algorithms list
3. Update API models and frontend options

## 📝 License

This project is part of UltraLab's evaluation of Gymnasium for interaction-based workflow modeling.

## 🤝 Contributing

This is a prototype for evaluation purposes. For questions or suggestions, please reach out to the UltraLab team.

---

**Built with**: Gymnasium, FastAPI, React, stable-baselines3, Material-UI  
**Project Duration**: 3 story points  
**Status**: ✅ Prototype Complete
