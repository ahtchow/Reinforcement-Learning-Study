import numpy as np
import random

#Import Required Data Structures
from collections import namedtuple, deque

# Import  Model from model.py
from model import QNetwork

# Import Deep Learning Frameworks
import torch
import torch.nn.functional as F
import torch.optim as optim

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 64         # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR = 5e-4               # learning rate 
UPDATE_EVERY = 4        # how often to update the network

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent:
    """Interacts with and learns from the environment."""

    
    def __init__(self, state_size, action_size, seed):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)

        # Q-Network - Local, Neural Net for Target
        self.qnetwork_local = QNetwork(state_size, action_size, seed).to(device)
        self.qnetwork_target = QNetwork(state_size, action_size, seed).to(device)
        # Use same optimizer for both neural network
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)

        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0
        
        
    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)
        
        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
        # If enough samples are available in memory, get random subset
            if len(self.memory) > BATCH_SIZE:
                experiences = self.memory.sample()
                # learn step
                self.learn(experiences, GAMMA)
                
                
    def act(self, state, eps=0.):
        """Returns actions for given state as per current policy.
        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        state = torch.from_numpy(state).float() # CONVERT Tensor
        ''' Returns a new tensor with a dimension of size one inserted at the 
        specified position. '''
        state = state.unsqueeze(0).to(device)
        
        '''
        eval() -notify all your layers that you are in eval mode, that way, 
        batchnorm or dropout layers will work in eval mode instead of training 
        mode.
        
        no_grad() - impacts the autograd engine and deactivate it. It will reduce 
        memory usage and speed up computations but you won’t be able to backprop.
        '''
        self.qnetwork_local.eval() # Eval Mode
        with torch.no_grad(): # No Gradient Descent
            # Returns vector of action values
            action_values = self.qnetwork_local.foward(state) 

        self.qnetwork_local.train() # Back to train mode

        # Epsilon-greedy action selection
        if random.random() > eps:
            greedy_action_to_cpu = action_values.cpu().data.numpy()
            action = np.argmax(greedy_action_to_cpu)
            return action
        else:
            # EXPLORATION, choose random the 
            return random.choice(np.arange(self.action_size))

        
    def learn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Variable]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor        
        """
        #Unpack the tuple
        states, actions, rewards, next_states, dones = experiences
        
        # Get max predicted Q values (for next states) from target model
        Q_targets_next = self.qnetwork_target.foward(next_states)
        Q_targets_next = Q_targets_next.detach().max(1)[0].unsqueeze(1)
            
        TD_target = rewards + (gamma * Q_targets_next * (1 - dones))
        
        # Get expected Q values from local model
        Q_expected = self.qnetwork_local.foward(states).gather(1, actions)

        ## compute loss
        loss = F.mse_loss(Q_expected, TD_target)
        
        # Minimize the loss
        self.optimizer.zero_grad() # reset grad
        loss.backward() # Calculate the gradient
        self.optimizer.step() # Update weights
        
        # ------------------- update target network ------------------- #
        
        """
        Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        """
        for target_param, local_param in zip(self.qnetwork_target.parameters(), self.qnetwork_local.parameters()):
            target_param.data.copy_(TAU*local_param.data + (1.0-TAU)*target_param.data)
        

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  
        self.batch_size = batch_size
        # < S, A, R, S> - tuple < State, Action, Reward, State Next> + Done
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        
        # Sample an experience with length k from list of memories
        experiences = random.sample(self.memory, k=self.batch_size)
        
        # For each item in the tuple, stack vertically and conver to torch tensor
        states = np.vstack([e.state for e in experiences if e is not None])
        states = torch.from_numpy(states).float().to(device)
          
        actions = np.vstack([e.action for e in experiences if e is not None])
        actions = torch.from_numpy(actions).long().to(device) 
        
        rewards = np.vstack([e.reward for e in experiences if e is not None])
        rewards = torch.from_numpy(rewards).float().to(device)
        
        next_states = np.vstack([e.next_state for e in experiences if e is not None])
        next_states = torch.from_numpy(next_states).float().to(device)   
        
        dones = np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8) # Make bool an int
        dones = torch.from_numpy(dones).float().to(device)
        
        return (states, actions, rewards, next_states, dones)
    
    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
        