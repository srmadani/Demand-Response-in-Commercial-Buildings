import gym
from gym import spaces
import numpy as np
import pickle

class HVAC(gym.Env):
    def __init__(self, num_days=7, C=75, R=8, h=10, T_des=22, alpha=0.5, gamma=0.5, P = 0.1):
        super(HVAC, self).__init__()
        
        self.num_days = num_days
        self.C = C
        self.R = R
        self.h = h
        self.T_des = T_des
        self.alpha = alpha
        self.gamma = gamma
        self.high_consumption_hours = [6, 7, 8, 16, 17, 18, 19]
        self.P = P
        self.r = np.zeros((3,self.num_days * 24 * 12))

        with open(r'datasets/df.pkl', 'rb') as f:
            self.db = pickle.load(f)
        
        self.E_B = self.db['E_B (kW)'].to_numpy()/50
        T_out = np.array(self.db['Temp (°C)']) # the outdoor temperature
        self.T_out = np.append(T_out, T_out[-1])

        high_consumption_hours = [6, 7, 8, 16, 17, 18, 19] # Define the time slots with higher consumption (6-9 am and 4-8 pm)

        # Initialize the binary vector a with zeros
        self.a = np.zeros(self.num_days * 24 * 12, dtype=int)

        # Fill in the binary vector a based on high_consumption_hours
        for step in range(self.num_days * 24 * 12):
            current_hour = (step // 12) % 24
            if current_hour in high_consumption_hours:
                self.a[step] = 1

        self.action_space = spaces.Discrete(2) # 0 for HVAC off and 1 for HVAC on
        self.observation_space = spaces.Box(low=-1, high=1, shape=(5,), dtype=np.float32) # sin(hour), cos(hour), T_in[t-1], T_out, E_B

    def step(self, action):

        self.T_in[self.t+1] = self.T_in[self.t] + ((self.T_out[self.t+1] - self.T_in[self.t]) / self.R + action * self.h) * (5 / self.C)

        peak_load_reduction_reward = self.gamma * self.a[self.t] * (self.E_B[self.t] - action * self.h)
        reward = peak_load_reduction_reward \
                - self.P * action * self.h \
                - self.alpha * np.abs(self.T_in[self.t+1] - self.T_des)

        
        self.r[:, self.t] = peak_load_reduction_reward, self.P * action * self.h, self.alpha * np.abs(self.T_in[self.t+1] - self.T_des)
        self.t += 1
        if self.t < self.num_days * 24 * 12 :
            tmp = self.db.iloc[self.t,:]
            next_state = np.array([tmp['sin'], tmp['cos'], (self.T_in[self.t] + 25.8)/40.7, (tmp['Temp (°C)'] + 25.8)/40.7, self.E_B[self.t]/self.h])
            done = False
        else:
            next_state = np.array([0, 0, 0, 0, 0])
            done = True
        
        return next_state, reward, done, {}

    def reset(self):
        self.t = 0
        self.T_in = np.zeros(self.num_days * 24 * 12 +1)
        self.T_in[0] = self.T_des
        tmp = self.db.iloc[self.t,:]
        # db['Temp (°C)'].max(),min() = 14.9, -25.8
        return np.array([tmp['sin'], tmp['cos'], (self.T_des + 25.8)/40.7, (tmp['Temp (°C)'] + 25.8)/40.7, self.E_B[self.t]/self.h])
    
    def render(self):
        pass

    def close(self):
        pass

# Create the environment
# env = HVAC()
# state = env.reset()
# state