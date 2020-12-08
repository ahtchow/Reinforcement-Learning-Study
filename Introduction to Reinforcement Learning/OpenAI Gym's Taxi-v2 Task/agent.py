import numpy as np
from collections import defaultdict

class Agent:

    def __init__(self, nA=6):
        """ Initialize agent.

        Params
        ======
        - nA: number of actions available to the agent
        """
        self.algorithm = 'sarsa(0)'
        np.seterr('raise')
        self.nA = nA
        self.Q = defaultdict(lambda: np.zeros(self.nA))
        self.epsilon = 0.1
        self.gamma = 0.8
        self.alpha = 0.07
        self.prob_of_choosing_action = None
        print(f'Algorithm: {self.algorithm}')
        
    def probabilities_of_action(self, state):
        """ Given the state, return probability distribution for actions.

        Params
        ======
        - state: the current state of the environment

        Returns
        =======
        - action: an integer, compatible with the task's action space
        """        #initialize policy as all ones
        epsilon_greedy_policy = np.ones(self.nA)
        # multiply all actions of probability to (epsilon / num_action)
        epsilon_greedy_policy = epsilon_greedy_policy * (self.epsilon / self.nA)
         # Use Q-table, find the greedy action for state-action
        greedy_action_index = np.argmax(self.Q[state])
        # Change greedy action's probability
        epsilon_greedy_policy[greedy_action_index] = 1 - self.epsilon + (self.epsilon / self.nA)
        return epsilon_greedy_policy

    def select_action(self, state):
        """ Given the state, select an action.

        Params
        ======
        - state: the current state of the environment

        Returns
        =======
        - action: an integer, compatible with the task's action space
        """
        epsilon_greedy_policy = self.probabilities_of_action(state)
        return np.random.choice(np.arange(self.nA), p=epsilon_greedy_policy)
   

    def step(self, state, action, reward, next_state, done):
        """ Update the agent's knowledge, using the most recently sampled tuple.

        Params
        ======
        - state: the previous state of the environment
        - action: the agent's previous choice of action
        - reward: last reward received
        - next_state: the current state of the environment
        - done: whether the episode is complete (True or False)
        """
        next_return = self.Q[next_state][self.select_action(next_state)]
        self.Q[state][action] += (self.alpha * (reward + (self.gamma * next_return) - self.Q[state][action]))
        
        #Update epsilon
        if done:
            self.epsilon /= 2.0