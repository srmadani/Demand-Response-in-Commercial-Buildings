import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Beta,Normal
import math
import os, shutil
from datetime import datetime
from ctrl import CTRL
import argparse
import pickle
import time
#import mlflow

#mlflow.set_tracking_uri("sqlite:///mlflow.db") #The name of the database to use
#mlflow.set_experiment("ctrl-td3") #If already exists mlflow will append to existing data. Else it will make a new experiment.

def act_clipper(a):
	if a>1:
		return 1
	elif a<0:
		return 0
	else:
		return a

class Actor(nn.Module):
	def __init__(self, state_dim, action_dim, net_width, maxaction):
		super(Actor, self).__init__()

		self.l1 = nn.Linear(state_dim, net_width)
		self.l2 = nn.Linear(net_width, net_width)
		self.l3 = nn.Linear(net_width, action_dim)

		self.maxaction = maxaction

	def forward(self, state):
		a = torch.tanh(self.l1(state))
		a = torch.tanh(self.l2(a))
		a = torch.tanh(self.l3(a)) * self.maxaction
		return a


class Double_Q_Critic(nn.Module):
	def __init__(self, state_dim, action_dim, net_width):
		super(Double_Q_Critic, self).__init__()

		# Q1 architecture
		self.l1 = nn.Linear(state_dim + action_dim, net_width)
		self.l2 = nn.Linear(net_width, net_width)
		self.l3 = nn.Linear(net_width, 1)

		# Q2 architecture
		self.l4 = nn.Linear(state_dim + action_dim, net_width)
		self.l5 = nn.Linear(net_width, net_width)
		self.l6 = nn.Linear(net_width, 1)


	def forward(self, state, action):
		sa = torch.cat([state, action], 1)

		q1 = F.relu(self.l1(sa))
		q1 = F.relu(self.l2(q1))
		q1 = self.l3(q1)

		q2 = F.relu(self.l4(sa))
		q2 = F.relu(self.l5(q2))
		q2 = self.l6(q2)
		return q1, q2


	def Q1(self, state, action):
		sa = torch.cat([state, action], 1)

		q1 = F.relu(self.l1(sa))
		q1 = F.relu(self.l2(q1))
		q1 = self.l3(q1)
		return q1

def evaluate_policy(env, model, mode='validation'):
	if mode == 'validation':
		week_num = 7
	else:
		week_num = 8
	s, done, scores= env.reset(week_num = week_num), False, 0
	while not done:
		a = model.select_action(s, deterministic = True)
		s_next, r, done, _ = env.step(act_clipper(a))
		scores += r
		s = s_next
	return scores, env.load, env.cost, env.cost_components


#Just ignore this function~
def str2bool(v):
	'''transfer str to bool for argparse'''
	if isinstance(v, bool):
		return v
	if v.lower() in ('yes', 'True','true','TRUE', 't', 'y', '1'):
		return True
	elif v.lower() in ('no', 'False','false','FALSE', 'f', 'n', '0'):
		return False
	else:
		raise argparse.ArgumentTypeError('Boolean value expected.')

#reward engineering for better training
def Reward_adapter(r, EnvIdex):
	# For Pendulum-v0
	if EnvIdex == 0:
		r = (r + 8) / 8

	# For LunarLander
	elif EnvIdex == 1:
		if r <= -100: r = -10

	# For BipedalWalker
	elif EnvIdex == 4 or EnvIdex == 5:
		if r <= -100: r = -1
	return r

class TD3_agent():
	def __init__(self, **kwargs):
		# Init hyperparameters for agent, just like "self.gamma = opt.gamma, self.lambd = opt.lambd, ..."
		self.__dict__.update(kwargs)
		self.policy_noise = 0.2*self.max_action
		self.noise_clip = 0.5*self.max_action
		self.tau = 0.005
		self.delay_counter = 0

		self.actor = Actor(self.state_dim, self.action_dim, self.net_width, self.max_action).to(self.dvc)
		self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.a_lr)
		self.actor_target = copy.deepcopy(self.actor)

		self.q_critic = Double_Q_Critic(self.state_dim, self.action_dim, self.net_width).to(self.dvc)
		self.q_critic_optimizer = torch.optim.Adam(self.q_critic.parameters(), lr=self.c_lr)
		self.q_critic_target = copy.deepcopy(self.q_critic)

		self.replay_buffer = ReplayBuffer(self.state_dim, self.action_dim, max_size=int(1e6), dvc=self.dvc)
		
	def select_action(self, state, deterministic):
		with torch.no_grad():
			state = torch.FloatTensor(state[np.newaxis, :]).to(self.dvc)  # from [x,x,...,x] to [[x,x,...,x]]
			a = self.actor(state).cpu().numpy()[0] # from [[x,x,...,x]] to [x,x,...,x]
			if deterministic:
				return a
			else:
				noise = np.random.normal(0, self.max_action * self.explore_noise, size=self.action_dim)
				return (a + noise).clip(-self.max_action, self.max_action)

	def train(self):
		self.delay_counter += 1
		with torch.no_grad():
			s, a, r, s_next = self.replay_buffer.sample(self.batch_size)

			# Compute the target Q
			target_a_noise = (torch.randn_like(a) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
			'''↓↓↓ Target Policy Smoothing Regularization ↓↓↓'''
			smoothed_target_a = (self.actor_target(s_next) + target_a_noise).clamp(-self.max_action, self.max_action)
			target_Q1, target_Q2 = self.q_critic_target(s_next, smoothed_target_a)
			'''↓↓↓ Clipped Double Q-learning ↓↓↓'''
			target_Q = torch.min(target_Q1, target_Q2)
			target_Q = r + self.gamma * target_Q  

		# Get current Q estimates
		current_Q1, current_Q2 = self.q_critic(s, a)

		# Compute critic loss, and Optimize the q_critic
		q_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
		self.q_critic_optimizer.zero_grad()
		q_loss.backward()
		self.q_critic_optimizer.step()

		'''↓↓↓ Clipped Double Q-learning ↓↓↓'''
		if self.delay_counter > self.delay_freq:
			# Update the Actor
			a_loss = -self.q_critic.Q1(s,self.actor(s)).mean()
			self.actor_optimizer.zero_grad()
			a_loss.backward()
			self.actor_optimizer.step()

			# Update the frozen target models
			with torch.no_grad():
				for param, target_param in zip(self.q_critic.parameters(), self.q_critic_target.parameters()):
					target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

				for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
					target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

			self.delay_counter = 0

	def save(self,EnvName):
		torch.save(self.actor.state_dict(), "./model/{}_actor.pth".format(EnvName))
		torch.save(self.q_critic.state_dict(), "./model/{}_critic.pth".format(EnvName))

	def load(self,EnvName):
		self.actor.load_state_dict(torch.load("./model/{}_actor.pth".format(EnvName)))
		self.q_critic.load_state_dict(torch.load("./model/{}_critic.pth".format(EnvName)))


class ReplayBuffer():
	def __init__(self, state_dim, action_dim, max_size, dvc):
		self.max_size = max_size
		self.dvc = dvc
		self.ptr = 0
		self.size = 0

		self.s = torch.zeros((max_size, state_dim) ,dtype=torch.float,device=self.dvc)
		self.a = torch.zeros((max_size, action_dim) ,dtype=torch.float,device=self.dvc)
		self.r = torch.zeros((max_size, 1) ,dtype=torch.float,device=self.dvc)
		self.s_next = torch.zeros((max_size, state_dim) ,dtype=torch.float,device=self.dvc)

	def add(self, s, a, r, s_next):
		self.s[self.ptr] = torch.from_numpy(s).to(self.dvc)
		self.a[self.ptr] = torch.from_numpy(a).to(self.dvc)
		r_np = np.array([r])
		self.r[self.ptr] = torch.from_numpy(r_np).to(self.dvc)
		self.s_next[self.ptr] = torch.from_numpy(s_next).to(self.dvc)

		self.ptr = (self.ptr + 1) % self.max_size
		self.size = min(self.size + 1, self.max_size)

	def sample(self, batch_size):
		ind = torch.randint(0, self.size, device=self.dvc, size=(batch_size,))
		return self.s[ind], self.a[ind], self.r[ind], self.s_next[ind]
	
'''Hyperparameter Setting'''
parser = argparse.ArgumentParser()
parser.add_argument('--dvc', type=str, default='cuda', help='running device: cuda or cpu')
parser.add_argument('--EnvIdex', type=int, default=0, help='CTRL_TD3')
parser.add_argument('--render', type=str2bool, default=False, help='Render or Not')
parser.add_argument('--Loadmodel', type=str2bool, default=False, help='Load pretrained model or Not')
parser.add_argument('--ModelIdex', type=int, default=30, help='which model to load')

parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--update_every', type=int, default=50, help='training frequency')
parser.add_argument('--Max_train_steps', type=int, default=int(4e5), help='Max training steps')
parser.add_argument('--Max_train_time', type=int, default=1e5, help='Max training time')
parser.add_argument('--save_interval', type=int, default=int(1e5), help='Model saving interval, in steps.')
parser.add_argument('--eval_interval', type=int, default=int(2e2), help='Model evaluating interval, in steps.')

parser.add_argument('--delay_freq', type=int, default=1, help='Delayed frequency for Actor and Target Net')
parser.add_argument('--gamma', type=float, default=0.97, help='Discounted Factor')
parser.add_argument('--net_width', type=int, default=256, help='Hidden net width, s_dim-400-300-a_dim')
parser.add_argument('--a_lr', type=float, default=5e-4, help='Learning rate of actor')
parser.add_argument('--c_lr', type=float, default=5e-4, help='Learning rate of critic')
parser.add_argument('--batch_size', type=int, default=256, help='batch_size of training')
parser.add_argument('--explore_noise', type=float, default=0.35, help='exploring noise when interacting')
parser.add_argument('--explore_noise_decay', type=float, default=0.998, help='Decay rate of explore noise')
opt = parser.parse_args()
opt.dvc = torch.device(opt.dvc) # from str to torch.device
# print(opt)

# def main(beta=0.1, render=False, compare=False):
# 	start_time = time.time()
# 	EnvName = [f'CTRL_TD3_beta_{beta}']
# 	BrifEnvName = [f'CTRL_TD3_{beta}']
def main(beta=0.2, gamma=0.55132, render=False, compare=False, cpp=0.000067):	
	#mlflow.set_tag("alg", 'td3')
	#mlflow.set_tag("beta", beta)
	#mlflow.set_tag("gamma", gamma)
	#mlflow.set_tag("cpp", cpp)
	start_time = time.time()
	EnvName = [f'CTRL_TD3_beta_{beta}_gamma_{gamma}']
	BrifEnvName = [f'CTRL_TD3_{beta}_{gamma}']

	# Build Env
	env = CTRL(beta=beta, gamma=gamma)
	eval_env = CTRL(beta=beta, gamma=gamma)
	opt.state_dim = env.observation_space.shape[0]
	opt.action_dim = env.action_space.shape[0]
	opt.max_action = 1
	opt.max_e_steps = 7*7*12
	# print(f'Env:{EnvName[opt.EnvIdex]}  state_dim:{opt.state_dim}  action_dim:{opt.action_dim}  '
	# 	  f'max_a:{opt.max_action}  min_a:{env.action_space.low[0]}  max_e_steps:{opt.max_e_steps}')

	# Seed Everything
	env_seed = opt.seed
	np.random.seed(opt.seed)
	torch.manual_seed(opt.seed)
	torch.cuda.manual_seed(opt.seed)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False
	# print("Random Seed: {}".format(opt.seed))
	#mlflow.log_params(opt.__dict__)

	# Build DRL model
	if not os.path.exists('model'): os.mkdir('model')
	agent = TD3_agent(**vars(opt)) # var: transfer argparse to dictionary
	if opt.Loadmodel: agent.load(BrifEnvName[opt.EnvIdex])

	if render:
		agent.load(BrifEnvName[opt.EnvIdex])
		res = {}
		res['score'], res['load'], res['cost'], res['cost_components'] = evaluate_policy(eval_env, agent, mode='test')
		print(f"Env: {BrifEnvName[opt.EnvIdex]}, score: {res['score']}, cost: {res['cost']}")
		with open(f'res/{EnvName[opt.EnvIdex]}.pkl', 'wb') as file:
			pickle.dump(res, file)
		return res['score'], res['load'], res['cost'], res['cost_components']
	elif compare:
		agent.load(BrifEnvName[opt.EnvIdex])
		score, _, _, _ = evaluate_policy(eval_env, agent, mode='validation')
		return score
	else:
		elapsed_times = []
		rewards = []
		total_steps, elapsed_time, act, rew, best_score = 0, 0, [], [], -np.inf
		# while total_steps < opt.Max_train_steps:
		while elapsed_time < opt.Max_train_time:
			s = env.reset(week_num=np.random.randint(1,7))
			done = False

			'''Interact & trian'''
			while not done:
				if total_steps < (10*opt.max_e_steps): a = env.action_space.sample() # warm up
				else: a = agent.select_action(s, deterministic=False)
				s_next, r, done, _ = env.step(act_clipper(a))

				agent.replay_buffer.add(s, a, r, s_next)
				s = s_next
				total_steps += 1

				'''train if its time'''
				# train 50 times every 50 steps rather than 1 training per step. Better!
				if (total_steps >= 2*opt.max_e_steps) and (total_steps % opt.update_every == 0):
					for j in range(opt.update_every):
						agent.train()

				'''record & log'''
				if total_steps % opt.eval_interval == 0:
					agent.explore_noise *= opt.explore_noise_decay
					ep_r, _, _, _ = evaluate_policy(eval_env, agent, mode='validation')
					elapsed_time = time.time() - start_time
					print(f'EnvName:{BrifEnvName[opt.EnvIdex]} , Steps: {int(total_steps/1000)}k, Episode Reward:{ep_r}, Elapsed Time:{elapsed_time}')
					# rew.append(np.array(ep_r))
					if ep_r > best_score:
						agent.save(BrifEnvName[opt.EnvIdex])
						best_score = ep_r

					elapsed_times.append(elapsed_time)
					rewards.append(best_score)
					#mlflow.log_metric("score",ep_r[0])
					#mlflow.log_metric("time",elapsed_time)

		# with open(f'res/track/track_{BrifEnvName[opt.EnvIdex]}.pkl', 'wb') as f:
		# 	pickle.dump((elapsed_times, rewards), f)
		env.close()
		eval_env.close()
	with open(f'res/act_{EnvName[opt.EnvIdex]}.pkl', 'wb') as file:
		pickle.dump(act, file)
	with open(f'res/rew_{EnvName[opt.EnvIdex]}.pkl', 'wb') as file:
		pickle.dump(rew, file)


if __name__ == '__main__':
	main(render=False)