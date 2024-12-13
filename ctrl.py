import gym
from gym import spaces
import pickle
import numpy as np
import pandas as pd

class CTRL(gym.Env):
    def __init__(self, beta=0.2, gamma=0.55132, cpp=0.000067):
        super(CTRL, self).__init__()

        self.beta  = beta
        self.gamma = gamma
        self.cpp   = cpp
        # self.cpp   = 0


        self.action_space = spaces.Box(0.0, 1.0, (1,), dtype=np.float32) # reduction ratio
        self.observation_space = spaces.Box(low=-1, high=1, shape=(7,), dtype=np.float32) # GHG, RTLMP, sin(hour), cos(hour), peak, is_working_day, normalized current reducible load

    def step(self, action):
        
        # if self.GHG[self.t] > 30:
        #     action = 1
        # elif self.peak[self.t] and self.GHG[self.t] > 29:
        #     action = 1
        # else:
        #     action = 0

        # action = 0
        self.acts[self.t] = action
        tmp = self.db.iloc[self.t,:]
        reward = tmp['reducible [kWh]'] * action * self.peak[self.t] * self.gamma\
                - self.P[self.t] * tmp['reducible [kWh]'] * (1-action) \
                - self.beta * (tmp['reducible [kWh]'] * action) ** 2 \
                - self.cpp * self.GHG[self.t] * (1-action) * tmp['reducible [kWh]']
        # print()
        self.cost_components[self.t, :] = np.array([tmp['reducible [kWh]'] * action * self.peak[self.t] * self.gamma, # 0: peak load reduction reward
                self.P[self.t] * tmp['reducible [kWh]'] * (1-action), # 1: electricity cost
                self.beta * ((tmp['reducible [kWh]'] * action) ** 2), # 2: dissatisfaction
                self.cpp * self.GHG[self.t] * (1-action) * tmp['reducible [kWh]']]).reshape(-1) # 3: GHG emission

        self.load[self.t] = tmp['reducible [kWh]'] * (1-action)
        self.cost += self.P[self.t] * tmp['reducible [kWh]'] * (1-action) - tmp['reducible [kWh]'] * action * self.peak[self.t] * self.gamma
        self.t += 1
        if self.t < self.T :
            tmp = self.db.iloc[self.t,:]
            next_state = np.array([self.GHG[self.t]/self.GHG_max, self.P[self.t]/2.7, tmp['sin'], tmp['cos'], self.peak[self.t], tmp['working_day'], tmp['reducible [kWh]']/self.light_max])
            done = False
        else:
            next_state = np.array([0, 0, 0, 0, 0, 0, 0])
            done = True
        
        return next_state, reward, done, {}

    def reset(self, week_num=1):
        self.T = 7 * 12 * 24
        
        with open(r'datasets/df.pkl', 'rb') as f:
            db = pickle.load(f)
        self.db = db.iloc[(week_num-1)*self.T:week_num*self.T,:].copy()
        self.light_max = np.max(self.db['reducible [kWh]'])
        self.P = np.array(self.db['P [$/kWh]'])
        self._max_episode_steps = self.T
        self.load = np.zeros(self.T)  
        self.acts = np.zeros(self.T)  
        self.cost_components = np.zeros((self.T,4))
        self.t = 0
        self.cost = 0
        tmp = self.db.iloc[self.t,:]
        self.peak = np.array(self.db['peak'])
        self.GHG = np.array(self.db['Total Emission [gCO₂eq/kWh]'])
        self.GHG_max = np.max(self.db['Total Emission [gCO₂eq/kWh]'])

        # db['P [$/kWh]'].max() - db['P [$/kWh]'].min() : 2.7
        return np.array([self.GHG[self.t]/self.GHG_max, self.P[self.t]/2.7, tmp['sin'], tmp['cos'], self.peak[self.t], tmp['working_day'], tmp['reducible [kWh]']/self.light_max])
    
    def render(self):
        pass

    def close(self):
        pass