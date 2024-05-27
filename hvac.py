import gym
from gym import spaces
import numpy as np
import pickle

class HVAC(gym.Env):
    def __init__(self, C=100, R=1, h=50, alpha=0.2, gamma=0.55132, t_lim=2):
        super(HVAC, self).__init__()
        
        self.T = 7 * 12 * 24
        self.C = C
        self.R = R
        self.h = h
        self.alpha = alpha
        self.gamma = gamma
        self.t_lim = t_lim-.5
        self.action_space = spaces.Discrete(2) # 0 for HVAC off and 1 for HVAC on
        self.observation_space = spaces.Box(low=-1, high=1, shape=(10,), dtype=np.float32) # RTLMP, sin(hour), cos(hour), T_in[t-1], T_out, peak, working_day, last three hvac status
        with open(r'datasets/ref_res.pkl', 'rb') as f:
            ref_res = pickle.load(f)
        self.ref_hvac = []
        for day in range((8-1)*7+1,8*7+1):
            self.ref_hvac.extend(ref_res[(day, C, R, h)][0])

    def step(self, action):
        ## turn hvac off
        # action = 0
        ## main constraints
        if self.T_in[self.t] - self.T_des[self.t] > self.t_lim:
            action = 0
        if self.T_in[self.t] - self.T_des[self.t] < -self.t_lim:
            action = 1
        # for status quo
        # if self.T_in[self.t] - self.T_des[self.t] < -0.5:
        #     action = 1
        # else:
        #     action = 0
        # to read from OR model results
        # action = self.ref_hvac[self.t]
        #heuristic alg -2
        # if self.T_in[self.t] < self.T_des[self.t] - 2:
        #     action = 1
        # else:
        #     action = 0
        self.acts[self.t] = action
        self.T_in[self.t+1] = self.T_in[self.t] + ((self.T_out[self.t+1] - self.T_in[self.t]) / self.R + action * self.h) * (5 / self.C)
        reward =  self.gamma * self.peak[self.t] * (self.ref_hvac[self.t] - action) * self.h \
                - self.P[self.t] * action * self.h \
                - self.alpha * np.power(self.T_in[self.t+1] - self.T_des[self.t], 2)
        
        self.cost_components[self.t, :] = [self.gamma * self.peak[self.t] * (self.ref_hvac[self.t] - action) * self.h, # 0: peak load reduction reward
                self.P[self.t] * action * self.h, # 1: electricity cost
                self.alpha * np.power(self.T_in[self.t+1] - self.T_des[self.t], 2)] # 2: dissatisfaction

        self.load[self.t] = action * self.h
        self.cost += self.P[self.t] * action * self.h - self.gamma * self.peak[self.t] * (self.ref_hvac[self.t] - action) * self.h
        self.t += 1
        if self.t < self.T :
            tmp = self.db.iloc[self.t,:]
            self.last_acts = self.last_acts[-2:] + [action]
            next_state = np.array([self.P[self.t]/2.7, tmp['sin'], tmp['cos'], (self.T_in[self.t] - self.T_des[self.t])/self.T_range, (tmp['Temp [°C]'] - self.T_min)/self.T_range, tmp['peak'], tmp['working_day']] + self.last_acts)
            done = False
        else:
            next_state = np.zeros(10)
            done = True
        
        return next_state, reward, done, {}

    def reset(self, week_num=1):
        self.t = 0
        self.T_in = np.zeros(self.T +1)
        self.load = np.zeros(self.T)
        self.acts = np.zeros(self.T)
        self.cost = 0
        self.last_acts = [1, 1, 1]
        self.cost_components = np.zeros((self.T,3))

        self.week_num = week_num
        # df stores general data like time and temp, ref_res includes hvac consumption for each profile in reference model
        with open(r'datasets/df.pkl', 'rb') as f:
            db = pickle.load(f)
        self.db = db.iloc[(self.week_num-1)*self.T:self.week_num*self.T,:].copy()
        T_out = np.array(self.db['Temp [°C]']) # the outdoor temperature
        self.T_out = np.append(T_out, T_out[-1])
        self.peak = np.array(self.db['peak'])
        self.P = np.array(self.db['P [$/kWh]'])

        with open(r'datasets/ref_res.pkl', 'rb') as f:
            ref_hvac = pickle.load(f)
        
        self.T_des = self.db['Des Temp [°C]'].to_numpy()
        self.T_in[0] = self.T_des[0]
        self.ref_hvac = []
        for day in range((self.week_num-1)*7+1,self.week_num*7+1):
            self.ref_hvac.extend(ref_hvac[(day, self.C, self.R, self.h)][0])

        self.T_min, self.T_range = T_out.min(), T_out.max() - T_out.min()
        tmp = self.db.iloc[self.t,:]
        # db['Temp (°C)'].max(),min() = 14.9, -25.8
        return np.array([self.P[0]/2.7, tmp['sin'], tmp['cos'], 0, (tmp['Temp [°C]'] - self.T_min)/self.T_range, tmp['peak'], tmp['working_day']] + self.last_acts)
    
    def render(self):
        pass

    def close(self):
        pass

# Create the environment
# env = HVAC()
# state = env.reset()
# state