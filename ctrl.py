import gym
from gym import spaces
import pickle
import numpy as np
import pandas as pd

class CTRL(gym.Env):
    def __init__(self, num_days=7, alpha=0.05, beta=0.5, P = 0.1):
        super(CTRL, self).__init__()
        self.num_days = num_days
        with open(r'datasets/df.pkl', 'rb') as f:
            db = pickle.load(f)
            
        db = db[((db['Date/Time (LST)'].dt.hour >= 16) & (db['Date/Time (LST)'].dt.hour < 20)) | ((db['Date/Time (LST)'].dt.hour >= 6) & (db['Date/Time (LST)'].dt.hour < 9))]

        self.df = db
        self.load_max = self.df['Reducible Load (kW)'].max()
        self.alpha = alpha
        self.beta = beta
        # self.high_consumption_hours = [6, 7, 8, 16, 17, 18, 19]
        self.P = P
        self._max_episode_steps = self.num_days * 7 * 12
        self.load = np.zeros(self.num_days * 7 * 12)  

        self.action_space = spaces.Box(0.0, 1.0, (1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-1, high=1, shape=(4,), dtype=np.float32) # sin(hour), cos(hour), is_working_day, normalized current reducible load

    def step(self, action):

        tmp = self.df.iloc[self.t,:]
        reward = tmp['Reducible Load (kW)'] * action * self.beta\
                - self.P * tmp['Reducible Load (kW)'] * (1-action) \
                - self.alpha * (tmp['Reducible Load (kW)'] * action) ** 2

        self.load[self.t] = tmp['Reducible Load (kW)'] * action
        self.t += 1
        if self.t < self.num_days * 7 * 12 :
            tmp = self.df.iloc[self.t,:]
            next_state = np.array([tmp['sin'], tmp['cos'],  tmp['working_day'], tmp['Reducible Load (kW)']/self.load_max])
            done = False
        else:
            next_state = np.array([0, 0, 0, 0])
            done = True
        
        return next_state, reward, done, {}

    def reset(self):
        self.t = 0
        tmp = self.df.iloc[self.t,:]
        return np.array([tmp['sin'], tmp['cos'],  tmp['working_day'], tmp['Reducible Load (kW)']/self.load_max])
    
    def render(self):
        pass

    def close(self):
        pass