from importlib.resources import path
from gym_driving.assets.car import *
from gym_driving.envs.environment import *
from gym_driving.envs.driving_env import *
from gym_driving.assets.terrain import *

import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"

import time
import pygame, sys
from pygame.locals import *
import random
import math
import argparse

# Do NOT change these values
TIMESTEPS = 1000
FPS = 30
NUM_EPISODES = 10

class Task1():

    def __init__(self):
        """
        Can modify to include variables as required
        """
        self.dim = 2
        self.win = [1,1]
        self.nsteer = 3
        self.nacc = 5
        self.steer = [-3, 0 ,3]
        self.acceleration = [-5, -3.95, 0, 3.95, 5]
        self.weights = np.array([1, -0.5])
        self.eps = 0
        
        super().__init__()
        
    def discrete(self, features):
        new_features = np.zeros(2, dtype = np.float64)
        for i in range(2):
            new_features[i] = (features[i]//self.win[i])*self.win[i] + (self.win[i]//2)
        return new_features
        
    def newFeatures(self, state, actions):
        v = max(state[2] + self.acceleration[actions[1]]/5, 0)
        angle = state[3] + self.steer[actions[0]]
        x = state[0] + v*math.cos(angle*math.pi/180)
        y = state[1] + v*math.sin(angle*math.pi/180)
        #new = self.discrete([x, y])
        new = [x, y]
        
        dot_prod = np.dot([new[0]-state[0],new[1]-state[1]],[350-state[0],-state[1]])
        cross_prod = np.linalg.norm(np.cross([new[0]-state[0],new[1]-state[1]],[350-state[0],-state[1]]))
        return np.array([dot_prod, cross_prod])
        
    
    def actionValue(self, features, weights):
        qval = 0
        for i in range(len(weights)):
            qval += features[i]*weights[i]
        return qval
    
    def next_action(self, state, old_action):
        """
        Input: The current state
        Output: Action to be taken
        TO BE FILLED
        """

        # Replace with your implementation to determine actions to be taken
        
        action_steer = old_action[0]
        action_acc = old_action[1]
        if(np.random.random() < self.eps):
            action_steer = np.random.randint(self.nsteer)
            action_acc = np.random.randint(self.nacc)
        else:
            maxqval = -1e18
            for i in range(self.nsteer):
                for j in range(self.nacc):
                    qval = self.actionValue(self.newFeatures(state, [i,j]), self.weights)
                    
                    if(abs(qval - maxqval) < 1e-4):
                        action_steer = np.random.choice([i, action_steer])
                        action_acc = np.random.choice([j, action_acc])
                    elif(qval - maxqval > 0):
                        maxqval = qval
                        action_steer = i
                        action_acc = j

        action = np.array([action_steer, action_acc])  

        return action

    def controller_task1(self, config_filepath=None, render_mode=False):
        """
        This is the main controller function. You can modify it as required except for the parts specifically not to be modified.
        Additionally, you can define helper functions within the class if needed for your logic.
        """
    
        ######### Do NOT modify these lines ##########
        pygame.init()
        fpsClock = pygame.time.Clock()

        if config_filepath is None:
            config_filepath = '../configs/config.json'

        simulator = DrivingEnv('T1', render_mode=render_mode, config_filepath=config_filepath)

        time.sleep(3)
        ##############################################

        # e is the number of the current episode, running it for 10 episodes
        for e in range(NUM_EPISODES):
        
            ######### Do NOT modify these lines ##########
            
            # To keep track of the number of timesteps per epoch
            cur_time = 0

            # To reset the simulator at the beginning of each episode
            state = simulator._reset()
            
            # Variable representing if you have reached the road
            road_status = False
            ##############################################

            # The following code is a basic example of the usage of the simulator
            for t in range(TIMESTEPS):
        
                # Checks for quit
                if render_mode:
                    for event in pygame.event.get():
                        if event.type == QUIT:
                            pygame.quit()
                            sys.exit()
                            
                if(t == 0):
                    action = self.next_action(state, [1,3])
                    x = self.newFeatures(list(state), list(action))
                
                state, reward, terminate, reached_road, info_dict = simulator._step(action)   #sprime
                action = self.next_action(state, action)   #aprime
                fpsClock.tick(FPS)
                
                if terminate:
                    road_status = reached_road
                    break                
                
                cur_time += 1
                

            # Writing the output at each episode to STDOUT
            print(str(road_status) + ' ' + str(cur_time))

class Task2():

    def __init__(self):
        """
        Can modify to include variables as required
        """
        self.dim = 2
        self.win = [1,1]
        self.nsteer = 3
        self.nacc = 5
        self.steer = [-3, 0 ,3]
        self.acceleration = [-5, -3.95, 0, 3.95, 5]
        self.pits = []
        self.weights = np.array([1, -0.5])
        self.eps = 0
        
        super().__init__()
        
    def discrete(self, features):
        new_features = np.zeros(2, dtype = np.float64)
        for i in range(2):
            new_features[i] = (features[i]//self.win[i])*self.win[i] + (self.win[i]//2)
        return new_features
        
    def newFeatures(self, state, actions):
        v = max(state[2] + self.acceleration[actions[1]]/5, 0)
        angle = state[3] + self.steer[actions[0]]
        x = state[0] + v*math.cos(angle*math.pi/180)
        y = state[1] + v*math.sin(angle*math.pi/180)
        #new = self.discrete([x, y])
        new = np.array([x, y])
        heading = np.array([new[0]-state[0],new[1]-state[1]])
        target = np.array([350-state[0],-state[1]])
        target = target/(pow(np.linalg.norm(target), 2))
        
        for i in range(len(self.pits)):
            obst = np.array([state[0] - self.pits[i][0], state[1] - self.pits[i][1]])
            obst = obst/(pow(np.linalg.norm(obst)-73, 3))
            target += obst
            
        target += np.array([0, state[1] - 350])/(pow(np.linalg.norm([0, state[1] - 350]), 3))
        target += np.array([0, state[1] + 350])/(pow(np.linalg.norm([0, state[1] + 350]), 3))
        target += np.array([state[0] - 350, 0])/(pow(np.linalg.norm([state[0] - 350, 0]), 3))
        target += np.array([state[0] + 350, 0])/(pow(np.linalg.norm([state[0] + 350, 0]), 3))
        
        dot_prod = np.dot(heading, target)
        cross_prod = np.linalg.norm(np.cross(heading, target))

        return np.array([dot_prod, cross_prod])

    
    def actionValue(self, features, weights):
        qval = 0
        for i in range(len(weights)):
            qval += features[i]*weights[i]
        return qval
    
    def next_action(self, state, old_action):
        """
        Input: The current state
        Output: Action to be taken
        TO BE FILLED
        """

        # Replace with your implementation to determine actions to be taken
        
        action_steer = old_action[0]
        action_acc = old_action[1]
        if(np.random.random() < self.eps):
            action_steer = np.random.randint(self.nsteer)
            action_acc = np.random.randint(self.nacc)
        else:
            maxqval = -1e18
            for i in range(self.nsteer):
                for j in range(self.nacc):
                    qval = self.actionValue(self.newFeatures(state, [i,j]), self.weights)
                    
                    if(abs(qval - maxqval) < 1e-4):
                        action_steer = np.random.choice([i, action_steer])
                        action_acc = np.random.choice([j, action_acc])
                    elif(qval - maxqval > 0):
                        maxqval = qval
                        action_steer = i
                        action_acc = j

        action = np.array([action_steer, action_acc])  

        return action

    def controller_task2(self, config_filepath=None, render_mode=False):
        """
        This is the main controller function. You can modify it as required except for the parts specifically not to be modified.
        Additionally, you can define helper functions within the class if needed for your logic.
        """
        
        ################ Do NOT modify these lines ################
        pygame.init()
        fpsClock = pygame.time.Clock()

        if config_filepath is None:
            config_filepath = '../configs/config.json'

        time.sleep(3)
        ###########################################################

        # e is the number of the current episode, running it for 10 episodes
        for e in range(NUM_EPISODES):

            ################ Setting up the environment, do NOT modify these lines ################
            # To randomly initialize centers of the traps within a determined range
            ran_cen_1x = random.randint(120, 230)
            ran_cen_1y = random.randint(120, 230)
            ran_cen_1 = [ran_cen_1x, ran_cen_1y]

            ran_cen_2x = random.randint(120, 230)
            ran_cen_2y = random.randint(-230, -120)
            ran_cen_2 = [ran_cen_2x, ran_cen_2y]

            ran_cen_3x = random.randint(-230, -120)
            ran_cen_3y = random.randint(120, 230)
            ran_cen_3 = [ran_cen_3x, ran_cen_3y]

            ran_cen_4x = random.randint(-230, -120)
            ran_cen_4y = random.randint(-230, -120)
            ran_cen_4 = [ran_cen_4x, ran_cen_4y]

            ran_cen_list = [ran_cen_1, ran_cen_2, ran_cen_3, ran_cen_4]  
            eligible_list = []

            # To randomly initialize the car within a determined range
            for x in range(-300, 300):
                for y in range(-300, 300):

                    if x >= (ran_cen_1x - 110) and x <= (ran_cen_1x + 110) and y >= (ran_cen_1y - 110) and y <= (ran_cen_1y + 110):
                        continue

                    if x >= (ran_cen_2x - 110) and x <= (ran_cen_2x + 110) and y >= (ran_cen_2y - 110) and y <= (ran_cen_2y + 110):
                        continue

                    if x >= (ran_cen_3x - 110) and x <= (ran_cen_3x + 110) and y >= (ran_cen_3y - 110) and y <= (ran_cen_3y + 110):
                        continue

                    if x >= (ran_cen_4x - 110) and x <= (ran_cen_4x + 110) and y >= (ran_cen_4y - 110) and y <= (ran_cen_4y + 110):
                        continue

                    eligible_list.append((x,y))

            simulator = DrivingEnv('T2', eligible_list, render_mode=render_mode, config_filepath=config_filepath, ran_cen_list=ran_cen_list)
        
            # To keep track of the number of timesteps per episode
            cur_time = 0

            # To reset the simulator at the beginning of each episode
            state = simulator._reset(eligible_list=eligible_list)
            ###########################################################

            # The following code is a basic example of the usage of the simulator
            self.pits = np.array(ran_cen_list)
            road_status = False

            for t in range(TIMESTEPS):
        
                # Checks for quit
                if render_mode:
                    for event in pygame.event.get():
                        if event.type == QUIT:
                            pygame.quit()
                            sys.exit()
                
                if(t == 0):
                    action = self.next_action(state, [1,3])
                    x = self.newFeatures(list(state), list(action))
                    
                state, reward, terminate, reached_road, info_dict = simulator._step(action)
                action = self.next_action(state, action)
                fpsClock.tick(FPS)

                cur_time += 1

                if terminate:
                    road_status = reached_road
                    break

            print(str(road_status) + ' ' + str(cur_time))

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", help="config filepath", default=None)
    parser.add_argument("-t", "--task", help="task number", choices=['T1', 'T2'])
    parser.add_argument("-r", "--random_seed", help="random seed", type=int, default=0)
    parser.add_argument("-m", "--render_mode", action='store_true')
    parser.add_argument("-f", "--frames_per_sec", help="fps", type=int, default=30) # Keep this as the default while running your simulation to visualize results
    args = parser.parse_args()

    config_filepath = args.config
    task = args.task
    random_seed = args.random_seed
    render_mode = args.render_mode
    fps = args.frames_per_sec

    FPS = fps

    random.seed(random_seed)
    np.random.seed(random_seed)

    if task == 'T1':
        
        agent = Task1()
        agent.controller_task1(config_filepath=config_filepath, render_mode=render_mode)

    else:

        agent = Task2()
        agent.controller_task2(config_filepath=config_filepath, render_mode=render_mode)
