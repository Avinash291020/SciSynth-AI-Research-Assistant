"""Reinforcement Learning system for paper selection and recommendation."""
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
import json
from pathlib import Path
from app.model_cache import ModelCache

class PaperEnvironment:
    """Environment for RL paper selection."""
    
    def __init__(self, papers_data: List[Dict[str, Any]]):
        self.papers = papers_data
        self.current_state = 0
        self.visited_papers = set()
        self.embedding_model = ModelCache.get_sentence_transformer()
        
        # Create paper embeddings
        self.paper_embeddings = self._create_embeddings()
        
        # State space: paper features + user preferences
        self.state_size = len(self.paper_embeddings[0]) + 10  # +10 for user preferences
        self.action_size = len(papers_data)
        
    def _create_embeddings(self) -> np.ndarray:
        """Create embeddings for all papers."""
        paper_texts = []
        for paper in self.papers:
            metadata = paper.get('metadata', {})
            text = f"{metadata.get('title', '')} {paper.get('insights', '')}"
            paper_texts.append(text)
        
        embeddings = self.embedding_model.encode(paper_texts)
        return embeddings
    
    def reset(self) -> np.ndarray:
        """Reset environment to initial state."""
        self.current_state = 0
        self.visited_papers = set()
        return self._get_state()
    
    def _get_state(self) -> np.ndarray:
        """Get current state representation."""
        # Paper embedding
        paper_embedding = self.paper_embeddings[self.current_state]
        
        # User preferences (simulated)
        user_prefs = np.random.rand(10)  # In real app, this would be learned
        
        return np.concatenate([paper_embedding, user_prefs])
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """Take action and return next state, reward, done, info."""
        if action >= len(self.papers):
            action = len(self.papers) - 1
        
        # Calculate reward based on paper relevance
        reward = self._calculate_reward(action)
        
        # Update state
        self.current_state = action
        self.visited_papers.add(action)
        
        # Check if done (visited all papers or max steps)
        done = len(self.visited_papers) >= min(10, len(self.papers))
        
        next_state = self._get_state()
        info = {
            'paper_name': self.papers[action].get('paper_name', f'paper_{action}'),
            'visited_count': len(self.visited_papers)
        }
        
        return next_state, reward, done, info
    
    def _calculate_reward(self, action: int) -> float:
        """Calculate reward for selecting a paper."""
        paper = self.papers[action]
        
        # Base reward for relevance
        relevance_score = 0.5
        
        # Bonus for new papers
        if action not in self.visited_papers:
            relevance_score += 0.3
        
        # Bonus for papers with insights
        if 'insights' in paper and paper['insights']:
            relevance_score += 0.2
        
        # Bonus for papers with hypotheses
        if 'hypotheses' in paper and paper['hypotheses']:
            relevance_score += 0.1
        
        return relevance_score

class DQNNetwork(nn.Module):
    """Deep Q-Network for paper selection."""
    
    def __init__(self, state_size: int, action_size: int):
        super(DQNNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, action_size)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)

class DQNAgent:
    """Deep Q-Learning agent for paper recommendation."""
    
    def __init__(self, state_size: int, action_size: int):
        self.state_size = state_size
        self.action_size = action_size
        
        # Neural networks
        self.q_network = DQNNetwork(state_size, action_size)
        self.target_network = DQNNetwork(state_size, action_size)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=0.001)
        
        # Hyperparameters
        self.memory = deque(maxlen=1000)
        self.batch_size = 32
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        
        # Update target network
        self.update_target_network()
    
    def update_target_network(self):
        """Update target network with current Q-network weights."""
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def remember(self, state: np.ndarray, action: int, reward: float, 
                next_state: np.ndarray, done: bool):
        """Store experience in memory."""
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state: np.ndarray) -> int:
        """Choose action using epsilon-greedy policy."""
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        q_values = self.q_network(state_tensor)
        return int(np.argmax(q_values.detach().numpy()))
    
    def replay(self):
        """Train the agent on a batch of experiences."""
        if len(self.memory) < self.batch_size:
            return
        
        batch = random.sample(self.memory, self.batch_size)
        states = torch.FloatTensor([e[0] for e in batch])
        actions = torch.LongTensor([e[1] for e in batch])
        rewards = torch.FloatTensor([e[2] for e in batch])
        next_states = torch.FloatTensor([e[3] for e in batch])
        dones = torch.BoolTensor([e[4] for e in batch])
        
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        next_q_values = self.target_network(next_states).max(1)[0].detach()
        target_q_values = rewards + (self.gamma * next_q_values * ~dones)
        
        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def load(self, name: str):
        """Load trained model."""
        self.q_network.load_state_dict(torch.load(name))
    
    def save(self, name: str):
        """Save trained model."""
        torch.save(self.q_network.state_dict(), name)

class RLPaperRecommender:
    """Main RL-based paper recommendation system."""
    
    def __init__(self, papers_data: List[Dict[str, Any]]):
        self.papers = papers_data
        self.environment = PaperEnvironment(papers_data)
        self.agent = DQNAgent(self.environment.state_size, self.environment.action_size)
        self.training_history = []
    
    def train(self, episodes: int = 100) -> List[float]:
        """Train the RL agent."""
        scores = []
        
        for episode in range(episodes):
            state = self.environment.reset()
            total_reward = 0
            
            for step in range(50):  # Max 50 steps per episode
                action = self.agent.act(state)
                next_state, reward, done, info = self.environment.step(action)
                
                self.agent.remember(state, action, reward, next_state, done)
                state = next_state
                total_reward += reward
                
                if done:
                    break
            
            self.agent.replay()
            scores.append(total_reward)
            
            if episode % 10 == 0:
                print(f"Episode {episode}/{episodes}, Score: {total_reward:.2f}, Epsilon: {self.agent.epsilon:.2f}")
        
        self.training_history = scores
        return scores
    
    def recommend_papers(self, num_recommendations: int = 5) -> List[Dict[str, Any]]:
        """Recommend papers using trained agent."""
        state = self.environment.reset()
        recommendations = []
        visited = set()
        
        q_values = self.agent.q_network(torch.FloatTensor(state).unsqueeze(0)).detach().numpy().flatten()

        # Get top N actions based on Q-values
        top_actions = np.argsort(q_values)[::-1]

        for action in top_actions:
            if len(recommendations) >= num_recommendations:
                break
            
            if action not in visited:
                paper = self.papers[action]
                metadata = paper.get('metadata', {})
                recommendations.append({
                    'paper_name': paper.get('paper_name', f'paper_{action}'),
                    'title': metadata.get('title', 'Title not available'),
                    'confidence': self._get_confidence(action),
                    'action_id': action
                })
                visited.add(action)
        
        return recommendations
    
    def _get_confidence(self, action: int) -> float:
        """Get confidence score for an action."""
        state = self.environment._get_state()
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        q_values = self.agent.q_network(state_tensor)
        return float(q_values[0][action].item())
    
    def save_model(self, path: str = "models/rl_paper_recommender.pth"):
        """Save the trained model."""
        Path(path).parent.mkdir(exist_ok=True)
        self.agent.save(path)
        
        # Save training history
        history_path = path.replace('.pth', '_history.json')
        with open(history_path, 'w') as f:
            json.dump(self.training_history, f)
    
    def load_model(self, path: str = "models/rl_paper_recommender.pth"):
        """Load a trained model."""
        self.agent.load(path)
        
        # Load training history
        history_path = path.replace('.pth', '_history.json')
        if Path(history_path).exists():
            with open(history_path, 'r') as f:
                self.training_history = json.load(f)

# Example usage
if __name__ == "__main__":
    # Load papers
    with open("results/all_papers_results.json", 'r', encoding='utf-8') as f:
        papers = json.load(f)
    
    # Initialize RL recommender
    recommender = RLPaperRecommender(papers)
    
    # Train the agent
    print("Training RL agent...")
    scores = recommender.train(episodes=50)
    
    # Get recommendations
    recommendations = recommender.recommend_papers(num_recommendations=3)
    
    print("\nRecommended papers:")
    for i, rec in enumerate(recommendations, 1):
        print(f"{i}. {rec['paper_name']} (confidence: {rec['confidence']:.3f})")
    
    # Save model
    recommender.save_model() 