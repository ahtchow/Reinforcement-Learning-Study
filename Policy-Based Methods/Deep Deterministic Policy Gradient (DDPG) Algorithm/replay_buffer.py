from collections import namedtuple, deque
import numpy as np
import random
import torch

field_names = ["state", "action", "reward", "next_state", "done"]

class ReplayBuffer:
    """ Fixed-size buffer to store experience tuples"""
    
    def __init__(self, action_size, buffer_size, batch_size, seed, device):
        """Initialize a ReplayBuffer object. """
        self.action_size = action_size
        self.buffer_size = buffer_size # size of replay buffer
        self.batch_size = batch_size # how many mem tuples to sample at a time
        self.seed = random.seed(seed)
        self.device = device
        
        # Define Named Tuple - field_names=["state", "action", "reward", "next_state", "done"]
        self.experience = namedtuple("Experience", field_names=field_names)
        
        # Data structure to hold the memories
        self.memory = deque(maxlen=buffer_size)
      
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

        
    def sample(self):
        """ Randomly sample a batch of experiences """

        # Sample an experience with length k from list of memories
        experiences = random.sample(self.memory, k=self.batch_size)

        # For each item in the tuple, stack vertically and convert to GPU torch tensor
        states = np.vstack([e.state for e in experiences if e is not None])
        states = torch.from_numpy(states).float().to(self.device) # (float)

        actions = np.vstack([e.action for e in experiences if e is not None])
        actions = torch.from_numpy(actions).long().to(self.device) # (long)

        rewards = np.vstack([e.reward for e in experiences if e is not None])
        rewards = torch.from_numpy(rewards).float().to(self.device) # (float)

        next_states = np.vstack([e.next_state for e in experiences if e is not None])
        next_states = torch.from_numpy(next_states).float().to(self.device) # float

        dones = np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8) # Make bool an int
        dones = torch.from_numpy(dones).float().to(self.device)
        
        return (states, actions, rewards, next_states, dones)
    

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)