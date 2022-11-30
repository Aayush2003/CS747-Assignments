"""
NOTE: You are only allowed to edit this file between the lines that say:
    # START EDITING HERE
    # END EDITING HERE

This file contains the AlgorithmManyArms class. Here are the method details:
    - __init__(self, num_arms, horizon): This method is called when the class
        is instantiated. Here, you can add any other member variables that you
        need in your algorithm.
    
    - give_pull(self): This method is called when the algorithm needs to
        select an arm to pull. The method should return the index of the arm
        that it wants to pull (0-indexed).
    
    - get_reward(self, arm_index, reward): This method is called just after the 
        give_pull method. The method should update the algorithm's internal
        state based on the arm that was pulled and the reward that was received.
        (The value of arm_index is the same as the one returned by give_pull.)
"""

import numpy as np

# START EDITING HERE
# You can use this space to define any helper functions that you need
# END EDITING HERE

class AlgorithmManyArms:
    def __init__(self, num_arms, horizon):
        self.num_arms = num_arms
        self.curr_time = 0
        self.counts = np.zeros(num_arms)
        self.values = np.zeros(num_arms)
        self.tuning = 2
        self.k = horizon
        self.d = 1/horizon
        self.c = 25/(horizon**3)
        self.const = self.c*self.k/(self.d)**2
        # Horizon is same as number of arms
    
    def give_pull(self):
        # START EDITING HERE
        eps = min(1, self.const/(self.curr_time + 1))
        # Implementing Modified Eps-Greedy
        if(np.random.random() < eps):
            return np.random.randint(self.num_arms)
        else:
            return np.argmax(self.values)
        # END EDITING HERE
    
    def get_reward(self, arm_index, reward):
        # START EDITING HERE
        self.counts[arm_index] += 1
        n = self.counts[arm_index]
        value = self.values[arm_index]
        new_value = ((n - 1) / n) * value + (1 / n) * reward
        self.values[arm_index] = new_value
        self.curr_time += 1
        # END EDITING HERE
