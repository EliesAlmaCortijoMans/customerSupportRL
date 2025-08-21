"""
Customer Support Agent Training Environment for Gymnasium

This environment simulates customer support interactions where an AI agent
learns optimal response strategies based on customer inquiries, sentiment,
and business context. Designed for BFSI, Retail, and Tech industries.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Dict, Any, Tuple, Optional, List
from enum import Enum
import random
from dataclasses import dataclass
import json


class InquiryType(Enum):
    """Types of customer inquiries across different industries"""
    # BFSI
    ACCOUNT_BALANCE = "account_balance"
    TRANSACTION_DISPUTE = "transaction_dispute"
    LOAN_APPLICATION = "loan_application"
    FRAUD_REPORT = "fraud_report"
    INVESTMENT_ADVICE = "investment_advice"
    
    # Retail
    ORDER_STATUS = "order_status"
    PRODUCT_RETURN = "product_return"
    PRODUCT_RECOMMENDATION = "product_recommendation"
    SHIPPING_ISSUE = "shipping_issue"
    DISCOUNT_INQUIRY = "discount_inquiry"
    
    # Tech
    TECHNICAL_SUPPORT = "technical_support"
    FEATURE_REQUEST = "feature_request"
    BUG_REPORT = "bug_report"
    BILLING_ISSUE = "billing_issue"
    INTEGRATION_HELP = "integration_help"


class CustomerSentiment(Enum):
    """Customer emotional states"""
    ANGRY = 0
    FRUSTRATED = 1
    NEUTRAL = 2
    SATISFIED = 3
    DELIGHTED = 4


class ResponseStrategy(Enum):
    """Available response strategies for the agent"""
    EMPATHETIC = 0          # Focus on emotional connection
    TECHNICAL = 1           # Provide detailed technical solution
    ESCALATE = 2           # Transfer to human agent
    PRODUCT_RECOMMEND = 3   # Suggest products/services
    APOLOGETIC = 4         # Focus on apology and service recovery
    EDUCATIONAL = 5        # Teach customer about features/processes
    QUICK_RESOLUTION = 6   # Fast, efficient problem solving
    UPSELL = 7            # Attempt to sell additional services


class CustomerTier(Enum):
    """Customer value tiers"""
    BASIC = 0
    PREMIUM = 1
    VIP = 2


@dataclass
class CustomerProfile:
    """Customer profile information"""
    tier: CustomerTier
    satisfaction_history: float  # Average satisfaction (0-1)
    interaction_count: int
    lifetime_value: float
    industry_segment: str  # "bfsi", "retail", "tech"


class CustomerSupportEnv(gym.Env):
    """
    Gymnasium environment for training customer support agents.
    
    State Space:
    - inquiry_type: Current inquiry type (categorical)
    - sentiment: Customer sentiment (0-4)
    - urgency: Urgency level (0-4)
    - conversation_length: Number of exchanges in current conversation (0-10)
    - customer_tier: Customer value tier (0-2)
    - previous_satisfaction: Last interaction satisfaction (0-1)
    - time_in_conversation: Time spent in current conversation (normalized)
    - context_vector: Encoded conversation context (10-dimensional)
    
    Action Space:
    - response_strategy: Choice of response strategy (0-7)
    
    Reward Function:
    - Customer satisfaction improvement: +0.5 to +2.0
    - Resolution efficiency: +0.1 to +0.5
    - Business impact: +0.0 to +1.0 (based on customer tier and outcome)
    - Negative rewards for poor outcomes: -0.5 to -2.0
    """
    
    metadata = {"render_modes": ["human", "rgb_array"]}
    
    def __init__(self, industry: str = "mixed", max_conversation_length: int = 10):
        super().__init__()
        
        self.industry = industry
        self.max_conversation_length = max_conversation_length
        
        # Define observation space
        # [inquiry_type, sentiment, urgency, conv_length, customer_tier, 
        #  prev_satisfaction, time_in_conv, context_vector(10)]
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, 0, 0, 0, 0] + [0.0] * 10, dtype=np.float32),
            high=np.array([len(InquiryType)-1, 4, 4, self.max_conversation_length, 2, 1.0, 1.0] + [1.0] * 10, dtype=np.float32),
            dtype=np.float32
        )
        
        # Define action space (response strategies)
        self.action_space = spaces.Discrete(len(ResponseStrategy))
        
        # Initialize environment state
        self.current_customer = None
        self.current_inquiry = None
        self.conversation_history = []
        self.conversation_length = 0
        self.start_time = 0
        self.episode_metrics = {}
        
        # Industry-specific inquiry distributions
        self.industry_inquiries = {
            "bfsi": [InquiryType.ACCOUNT_BALANCE, InquiryType.TRANSACTION_DISPUTE, 
                    InquiryType.LOAN_APPLICATION, InquiryType.FRAUD_REPORT, InquiryType.INVESTMENT_ADVICE],
            "retail": [InquiryType.ORDER_STATUS, InquiryType.PRODUCT_RETURN, 
                      InquiryType.PRODUCT_RECOMMENDATION, InquiryType.SHIPPING_ISSUE, InquiryType.DISCOUNT_INQUIRY],
            "tech": [InquiryType.TECHNICAL_SUPPORT, InquiryType.FEATURE_REQUEST, 
                    InquiryType.BUG_REPORT, InquiryType.BILLING_ISSUE, InquiryType.INTEGRATION_HELP]
        }
        
        # Strategy effectiveness matrix (inquiry_type -> strategy -> base_effectiveness)
        self.strategy_effectiveness = self._initialize_strategy_effectiveness()
        
        # Performance tracking
        self.episode_count = 0
        self.total_satisfaction = 0.0
        self.resolution_times = []
        
    def _initialize_strategy_effectiveness(self) -> Dict[InquiryType, Dict[ResponseStrategy, float]]:
        """Initialize strategy effectiveness matrix"""
        effectiveness = {}
        
        for inquiry_type in InquiryType:
            effectiveness[inquiry_type] = {}
            for strategy in ResponseStrategy:
                # Base effectiveness (will be modified by context)
                if inquiry_type in [InquiryType.FRAUD_REPORT, InquiryType.BUG_REPORT]:
                    # Urgent issues
                    effectiveness[inquiry_type][strategy] = {
                        ResponseStrategy.EMPATHETIC: 0.7,
                        ResponseStrategy.TECHNICAL: 0.9,
                        ResponseStrategy.ESCALATE: 0.8,
                        ResponseStrategy.QUICK_RESOLUTION: 0.9,
                        ResponseStrategy.APOLOGETIC: 0.6,
                        ResponseStrategy.EDUCATIONAL: 0.4,
                        ResponseStrategy.PRODUCT_RECOMMEND: 0.2,
                        ResponseStrategy.UPSELL: 0.1
                    }[strategy]
                elif inquiry_type in [InquiryType.PRODUCT_RECOMMENDATION, InquiryType.INVESTMENT_ADVICE]:
                    # Sales opportunities
                    effectiveness[inquiry_type][strategy] = {
                        ResponseStrategy.EMPATHETIC: 0.6,
                        ResponseStrategy.TECHNICAL: 0.7,
                        ResponseStrategy.ESCALATE: 0.3,
                        ResponseStrategy.QUICK_RESOLUTION: 0.5,
                        ResponseStrategy.APOLOGETIC: 0.3,
                        ResponseStrategy.EDUCATIONAL: 0.8,
                        ResponseStrategy.PRODUCT_RECOMMEND: 0.9,
                        ResponseStrategy.UPSELL: 0.7
                    }[strategy]
                else:
                    # Standard inquiries
                    effectiveness[inquiry_type][strategy] = {
                        ResponseStrategy.EMPATHETIC: 0.6,
                        ResponseStrategy.TECHNICAL: 0.7,
                        ResponseStrategy.ESCALATE: 0.5,
                        ResponseStrategy.QUICK_RESOLUTION: 0.7,
                        ResponseStrategy.APOLOGETIC: 0.5,
                        ResponseStrategy.EDUCATIONAL: 0.6,
                        ResponseStrategy.PRODUCT_RECOMMEND: 0.4,
                        ResponseStrategy.UPSELL: 0.3
                    }[strategy]
                    
        return effectiveness
    
    def _generate_customer(self) -> CustomerProfile:
        """Generate a random customer profile"""
        tier_weights = [0.6, 0.3, 0.1]  # Basic, Premium, VIP
        tier = CustomerTier(np.random.choice(len(CustomerTier), p=tier_weights))
        
        # Generate profile based on tier
        if tier == CustomerTier.VIP:
            satisfaction_history = np.random.normal(0.8, 0.1)
            lifetime_value = np.random.normal(50000, 15000)
        elif tier == CustomerTier.PREMIUM:
            satisfaction_history = np.random.normal(0.7, 0.15)
            lifetime_value = np.random.normal(15000, 5000)
        else:
            satisfaction_history = np.random.normal(0.6, 0.2)
            lifetime_value = np.random.normal(3000, 1000)
        
        return CustomerProfile(
            tier=tier,
            satisfaction_history=np.clip(satisfaction_history, 0, 1),
            interaction_count=np.random.randint(1, 50),
            lifetime_value=max(0, lifetime_value),
            industry_segment=self.industry if self.industry != "mixed" else np.random.choice(["bfsi", "retail", "tech"])
        )
    
    def _generate_inquiry(self) -> Tuple[InquiryType, CustomerSentiment, int]:
        """Generate a customer inquiry with sentiment and urgency"""
        if self.industry == "mixed":
            all_inquiries = list(InquiryType)
            inquiry_type = np.random.choice(all_inquiries)
        else:
            inquiry_type = np.random.choice(self.industry_inquiries[self.current_customer.industry_segment])
        
        # Sentiment influenced by customer history and inquiry type
        base_sentiment = 2  # Neutral
        if self.current_customer.satisfaction_history < 0.5:
            base_sentiment -= 1
        elif self.current_customer.satisfaction_history > 0.8:
            base_sentiment += 1
            
        # Certain inquiry types tend to be more negative
        if inquiry_type in [InquiryType.FRAUD_REPORT, InquiryType.TRANSACTION_DISPUTE, 
                           InquiryType.BUG_REPORT, InquiryType.BILLING_ISSUE]:
            base_sentiment -= 1
            
        sentiment = CustomerSentiment(np.clip(base_sentiment + np.random.randint(-1, 2), 0, 4))
        
        # Urgency based on inquiry type and sentiment
        urgency_map = {
            InquiryType.FRAUD_REPORT: 4,
            InquiryType.BUG_REPORT: 3,
            InquiryType.TRANSACTION_DISPUTE: 3,
            InquiryType.BILLING_ISSUE: 3,
            InquiryType.TECHNICAL_SUPPORT: 2,
            InquiryType.ORDER_STATUS: 2,
            InquiryType.SHIPPING_ISSUE: 2,
        }
        base_urgency = urgency_map.get(inquiry_type, 1)
        if sentiment.value <= 1:  # Angry or frustrated
            base_urgency += 1
            
        urgency = np.clip(base_urgency, 0, 4)
        
        return inquiry_type, sentiment, urgency
    
    def _encode_context(self) -> np.ndarray:
        """Encode conversation context into a 10-dimensional vector"""
        context = np.zeros(10, dtype=np.float32)
        
        if len(self.conversation_history) > 0:
            # Last few interactions impact
            recent_actions = self.conversation_history[-3:]
            for i, (action, satisfaction) in enumerate(recent_actions):
                if i < 3:
                    context[i] = action / len(ResponseStrategy)
                    context[i + 3] = satisfaction
        
        # Customer tier impact
        context[6] = self.current_customer.tier.value / 2
        
        # Inquiry urgency
        context[7] = getattr(self, 'current_urgency', 0) / 4
        
        # Industry context
        industry_encoding = {"bfsi": 0.2, "retail": 0.5, "tech": 0.8}
        context[8] = industry_encoding.get(self.current_customer.industry_segment, 0.5)
        
        # Conversation progress
        context[9] = self.conversation_length / self.max_conversation_length
        
        return context
    
    def _get_observation(self) -> np.ndarray:
        """Get current environment observation"""
        context_vector = self._encode_context()
        
        obs = np.array([
            list(InquiryType).index(self.current_inquiry),
            self.current_sentiment.value,
            self.current_urgency,
            self.conversation_length,
            self.current_customer.tier.value,
            self.current_customer.satisfaction_history,
            min(1.0, self.conversation_length / self.max_conversation_length),
            *context_vector
        ], dtype=np.float32)
        
        return obs
    
    def _calculate_reward(self, action: int) -> Tuple[float, bool, Dict[str, Any]]:
        """Calculate reward based on action and context"""
        strategy = ResponseStrategy(action)
        
        # Base effectiveness for this inquiry type and strategy
        base_effectiveness = self.strategy_effectiveness[self.current_inquiry][strategy]
        
        # Modify effectiveness based on context
        effectiveness = base_effectiveness
        
        # Customer tier influences response to certain strategies
        if strategy in [ResponseStrategy.PRODUCT_RECOMMEND, ResponseStrategy.UPSELL]:
            if self.current_customer.tier == CustomerTier.VIP:
                effectiveness *= 1.2  # VIP customers are more receptive
            elif self.current_customer.tier == CustomerTier.BASIC:
                effectiveness *= 0.7  # Basic customers less receptive
        
        # Sentiment influences effectiveness
        sentiment_multipliers = {
            CustomerSentiment.ANGRY: 0.5,
            CustomerSentiment.FRUSTRATED: 0.7,
            CustomerSentiment.NEUTRAL: 1.0,
            CustomerSentiment.SATISFIED: 1.2,
            CustomerSentiment.DELIGHTED: 1.3
        }
        effectiveness *= sentiment_multipliers[self.current_sentiment]
        
        # Conversation length penalty for inefficiency
        if self.conversation_length > 5:
            effectiveness *= (1.0 - (self.conversation_length - 5) * 0.1)
        
        # Generate customer satisfaction for this interaction
        satisfaction = np.clip(
            effectiveness + np.random.normal(0, 0.1),
            0, 1
        )
        
        # Calculate reward components
        satisfaction_reward = satisfaction * 2.0  # 0 to 2.0
        
        # Efficiency reward (higher for faster resolution)
        efficiency_reward = max(0, (1.0 - self.conversation_length / self.max_conversation_length) * 0.5)
        
        # Business impact (based on customer tier and outcome)
        tier_multipliers = {CustomerTier.BASIC: 0.5, CustomerTier.PREMIUM: 1.0, CustomerTier.VIP: 2.0}
        business_impact = satisfaction * tier_multipliers[self.current_customer.tier] * 0.5
        
        # Penalty for escalation (sometimes necessary but should be minimized)
        escalation_penalty = -0.3 if strategy == ResponseStrategy.ESCALATE else 0
        
        total_reward = satisfaction_reward + efficiency_reward + business_impact + escalation_penalty
        
        # Determine if conversation is done
        done = (
            satisfaction > 0.8 or  # High satisfaction - resolved
            self.conversation_length >= self.max_conversation_length or  # Max length reached
            strategy == ResponseStrategy.ESCALATE  # Escalated to human
        )
        
        # Update conversation history
        self.conversation_history.append((action, satisfaction))
        
        # Update sentiment based on interaction
        if satisfaction > 0.7:
            self.current_sentiment = CustomerSentiment(min(4, self.current_sentiment.value + 1))
        elif satisfaction < 0.3:
            self.current_sentiment = CustomerSentiment(max(0, self.current_sentiment.value - 1))
        
        # Metrics for analysis
        info = {
            "satisfaction": satisfaction,
            "effectiveness": effectiveness,
            "customer_tier": self.current_customer.tier.name,
            "inquiry_type": self.current_inquiry.name,
            "strategy_used": strategy.name,
            "conversation_length": self.conversation_length,
            "done_reason": "resolved" if satisfaction > 0.8 else "escalated" if strategy == ResponseStrategy.ESCALATE else "max_length" if self.conversation_length >= self.max_conversation_length else "ongoing"
        }
        
        return total_reward, done, info
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the environment for a new episode"""
        super().reset(seed=seed)
        
        # Generate new customer and inquiry
        self.current_customer = self._generate_customer()
        self.current_inquiry, self.current_sentiment, self.current_urgency = self._generate_inquiry()
        
        # Reset conversation state
        self.conversation_history = []
        self.conversation_length = 0
        self.start_time = 0
        
        # Reset episode metrics
        self.episode_metrics = {
            "customer_tier": self.current_customer.tier.name,
            "inquiry_type": self.current_inquiry.name,
            "initial_sentiment": self.current_sentiment.name,
            "industry": self.current_customer.industry_segment
        }
        
        observation = self._get_observation()
        info = {"customer_profile": self.current_customer, "inquiry": self.current_inquiry}
        
        return observation, info
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute one step in the environment"""
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action {action}")
        
        self.conversation_length += 1
        
        # Calculate reward and check if done
        reward, done, info = self._calculate_reward(action)
        
        # Get new observation
        observation = self._get_observation()
        
        # Update episode metrics
        if done:
            self.episode_count += 1
            self.total_satisfaction += info["satisfaction"]
            self.resolution_times.append(self.conversation_length)
            
            # Add episode summary to info
            info.update({
                "episode_satisfaction": info["satisfaction"],
                "episode_length": self.conversation_length,
                "episode_number": self.episode_count,
                "total_episodes": self.episode_count,
                "average_satisfaction": self.total_satisfaction / self.episode_count,
                "average_resolution_time": np.mean(self.resolution_times) if self.resolution_times else 0
            })
        
        return observation, reward, done, False, info
    
    def render(self, mode: str = "human") -> Optional[np.ndarray]:
        """Render the environment"""
        if mode == "human":
            print(f"\n=== Customer Support Environment ===")
            print(f"Customer: {self.current_customer.tier.name} tier")
            print(f"Industry: {self.current_customer.industry_segment.upper()}")
            print(f"Inquiry: {self.current_inquiry.name}")
            print(f"Sentiment: {self.current_sentiment.name}")
            print(f"Urgency: {self.current_urgency}/4")
            print(f"Conversation Length: {self.conversation_length}/{self.max_conversation_length}")
            
            if self.conversation_history:
                print("\nRecent Actions:")
                for i, (action, satisfaction) in enumerate(self.conversation_history[-3:], 1):
                    strategy = ResponseStrategy(action)
                    print(f"  {i}. {strategy.name}: {satisfaction:.2f} satisfaction")
            
            print("=" * 40)
        
        return None
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get environment performance metrics"""
        return {
            "total_episodes": self.episode_count,
            "average_satisfaction": self.total_satisfaction / max(1, self.episode_count),
            "average_resolution_time": np.mean(self.resolution_times) if self.resolution_times else 0,
            "satisfaction_distribution": np.histogram([h[1] for h in self.conversation_history], bins=5)[0].tolist() if self.conversation_history else [],
            "strategy_usage": {strategy.name: sum(1 for h in self.conversation_history if h[0] == strategy.value) 
                             for strategy in ResponseStrategy} if self.conversation_history else {}
        }


def make_customer_support_env(industry: str = "mixed", **kwargs) -> CustomerSupportEnv:
    """Factory function to create customer support environment"""
    return CustomerSupportEnv(industry=industry, **kwargs)
