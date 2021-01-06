import numpy as np
import random
import copy

class OUNoise:
    """
    Ornstein-Uhlenbeck process.
    
    For continuous action spaces, exploration is done via 
    adding noise to the action itself.
    
    """

    def __init__(self, action_size, seed, mu=0., theta=0.15, sigma=0.2):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(action_size) 
        self.theta = theta
        self.seed = random.seed(seed)
        self.sigma = sigma
        self.reset()
            
    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        
        random_noise_weights = [random.random() for i in range(len(x))] # random noise
        # dx = theta * (mu - x) + sigma * derivative of weights
        dx = self.theta * (self.mu - x) + self.sigma * np.array(random_noise_weights)
        
        self.state = x + dx
        return self.state