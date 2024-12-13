import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Beta,Normal
import os, shutil
from datetime import datetime
from ctrl import CTRL
import argparse
import pickle
import plotly.graph_objects as go
import time
#import mlflow

#mlflow.set_tracking_uri("sqlite:///mlflow.db") #The name of the database to use
#mlflow.set_experiment("ctrl-ddpg") #If already exists mlflow will append to existing data. Else it will make a new experiment.

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
		self.l2 = nn.Linear(net_width, 300)
		self.l3 = nn.Linear(300, action_dim)

		self.maxaction = maxaction

	def forward(self, state):
		a = torch.relu(self.l1(state))
		a = torch.relu(self.l2(a))
		a = torch.tanh(self.l3(a)) * self.maxaction
		return a


class Q_Critic(nn.Module):
	def __init__(self, state_dim, action_dim, net_width):
		super(Q_Critic, self).__init__()

		self.l1 = nn.Linear(state_dim + action_dim, net_width)
		self.l2 = nn.Linear(net_width, 300)
		self.l3 = nn.Linear(300, 1)

	def forward(self, state, action):
		sa = torch.cat([state, action], 1)
		q = F.relu(self.l1(sa))
		q = F.relu(self.l2(q))
		q = self.l3(q)
		return q

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
	return scores, env.load, env.cost


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
	

class DDPG_agent():
	def __init__(self, **kwargs):
		# Init hyperparameters for agent, just like "self.gamma = opt.gamma, self.lambd = opt.lambd, ..."
		self.__dict__.update(kwargs)
		self.tau = 0.005

		self.actor = Actor(self.state_dim, self.action_dim, self.net_width, self.max_action).to(self.dvc)
		self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.a_lr)
		self.actor_target = copy.deepcopy(self.actor)

		self.q_critic = Q_Critic(self.state_dim, self.action_dim, self.net_width).to(self.dvc)
		self.q_critic_optimizer = torch.optim.Adam(self.q_critic.parameters(), lr=self.c_lr)
		self.q_critic_target = copy.deepcopy(self.q_critic)

		self.replay_buffer = ReplayBuffer(self.state_dim, self.action_dim, max_size=int(5e5), dvc=self.dvc)
		
	def select_action(self, state, deterministic):
		with torch.no_grad():
			state = torch.FloatTensor(state[np.newaxis, :]).to(self.dvc)  # from [x,x,...,x] to [[x,x,...,x]]
			a = self.actor(state).cpu().numpy()[0] # from [[x,x,...,x]] to [x,x,...,x]
			if deterministic:
				return a
			else:
				noise = np.random.normal(0, self.max_action * self.noise, size=self.action_dim)
				return (a + noise).clip(-self.max_action, self.max_action)

	def train(self):
		# Compute the target Q
		with torch.no_grad():
			s, a, r, s_next = self.replay_buffer.sample(self.batch_size)
			target_a_next = self.actor_target(s_next)
			target_Q= self.q_critic_target(s_next, target_a_next)
			target_Q = r + self.gamma * target_Q  

		# Get current Q estimates
		current_Q = self.q_critic(s, a)

		# Compute critic loss
		q_loss = F.mse_loss(current_Q, target_Q)

		# Optimize the q_critic
		self.q_critic_optimizer.zero_grad()
		q_loss.backward()
		self.q_critic_optimizer.step()

		# Update the Actor
		a_loss = -self.q_critic(s,self.actor(s)).mean()
		self.actor_optimizer.zero_grad()
		a_loss.backward()
		self.actor_optimizer.step()

		# Update the frozen target models
		with torch.no_grad():
			for param, target_param in zip(self.q_critic.parameters(), self.q_critic_target.parameters()):
				target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

			for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
				target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

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
parser.add_argument('--EnvIdex', type=int, default=0, help='CTRL_DDPG')
parser.add_argument('--render', type=str2bool, default=False, help='Render or Not')
parser.add_argument('--Loadmodel', type=str2bool, default=False, help='Load pretrained model or Not')
parser.add_argument('--ModelIdex', type=int, default=100, help='which model to load')

parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--Max_train_steps', type=int, default=4e5, help='Max training steps')
parser.add_argument('--Max_train_time', type=int, default=1e5, help='Max training time')
parser.add_argument('--save_interval', type=int, default=4e4, help='Model saving interval, in steps.')
parser.add_argument('--eval_interval', type=int, default=2e2, help='Model evaluating interval, in steps.')

parser.add_argument('--gamma', type=float, default=0.97, help='Discounted Factor')
parser.add_argument('--net_width', type=int, default=400, help='Hidden net width, s_dim-400-300-a_dim')
parser.add_argument('--a_lr', type=float, default=2e-4, help='Learning rate of actor')
parser.add_argument('--c_lr', type=float, default=2e-4, help='Learning rate of critic')
parser.add_argument('--batch_size', type=int, default=128, help='batch_size of training')
parser.add_argument('--random_steps', type=int, default=5e3, help='random steps before trianing')
parser.add_argument('--noise', type=float, default=0.05, help='exploring noise')
opt = parser.parse_args()
opt.dvc = torch.device(opt.dvc) # from str to torch.device
# print(opt)


def main(beta=0.1, gamma=0.55132, render = False, compare=False, cpp=0.000067):
	# mlflow.set_tag("alg", 'ddpg')
	# mlflow.set_tag("beta", beta)
	# mlflow.set_tag("gamma", gamma)
	# mlflow.set_tag("cpp", cpp)
	start_time = time.time()
	EnvName = [f'CTRL_DDPG_beta_{beta}_gamma_{gamma}']
	BrifEnvName = [f'CTRL_DDPG_{beta}_{gamma}']

	# Build Env
	env = CTRL(beta=beta, gamma=gamma)
	eval_env = CTRL(beta=beta, gamma=gamma)
	opt.state_dim = env.observation_space.shape[0]
	opt.action_dim = env.action_space.shape[0]
	opt.max_action = 1
	# print(f'Env:{EnvName[opt.EnvIdex]}  state_dim:{opt.state_dim}  action_dim:{opt.action_dim} '
	# 	  f'max_a:{opt.max_action}  min_a:{env.action_space.low[0]} max_steps: {7*12*24}')

	# Seed Everything
	env_seed = opt.seed
	np.random.seed(opt.seed)
	torch.manual_seed(opt.seed)
	torch.cuda.manual_seed(opt.seed)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False
	# print("Random Seed: {}".format(opt.seed))

	# Build DRL model
	if not os.path.exists('model'): os.mkdir('model')
	agent = DDPG_agent(**vars(opt)) # var: transfer argparse to dictionary
	if opt.Loadmodel: agent.load(BrifEnvName[opt.EnvIdex])
	# mlflow.log_params(opt.__dict__)

	if render:
		agent.load(EnvName[opt.EnvIdex])
		res = {}
		res['score'], res['load'], res['cost'] = evaluate_policy(eval_env, agent, mode='test')
		print(f"Env: {EnvName[opt.EnvIdex]}, score: {res['score']}, cost: {res['cost']}")
		with open(f'res/{EnvName[opt.EnvIdex]}.pkl', 'wb') as file:
			pickle.dump(res, file)
		return res['score'], res['load'], res['cost']
	elif compare:
		agent.load(EnvName[opt.EnvIdex])
		score, _, _ = evaluate_policy(eval_env, agent, mode='validation')
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
				if total_steps < opt.random_steps: a = env.action_space.sample()
				else: a = agent.select_action(s, deterministic=False)
				s_next, r, done, _ = env.step(act_clipper(a))

				agent.replay_buffer.add(s, a, r, s_next)
				s = s_next
				total_steps += 1

				'''train'''
				if total_steps >= opt.random_steps:
					agent.train()

				'''record & log'''
				if total_steps % opt.eval_interval == 0:
					ep_r, _, _ = evaluate_policy(eval_env, agent, mode='validation')
					elapsed_time = time.time() - start_time
					# print(ep_r)
					# print(type(ep_r))
					print(f'EnvName:{BrifEnvName[opt.EnvIdex]}, Steps: {int(total_steps/1000)}k, Episode Reward:{ep_r}, Elapsed Time:{elapsed_time}')
					# rew.append(np.array(ep_r))
					if ep_r > best_score:
						agent.save(EnvName[opt.EnvIdex])
						best_score = ep_r

					
					elapsed_times.append(elapsed_time)
					rewards.append(best_score)
					elapsed_time = time.time() - start_time
					# mlflow.log_metric("score",ep_r[0])
					# mlflow.log_metric("time",elapsed_time)
		
		with open(f'res/track/track_{EnvName[opt.EnvIdex]}.pkl', 'wb') as f:
			pickle.dump((elapsed_times, rewards), f)
		env.close()
		eval_env.close()
	# with open(f'res/act_{EnvName[opt.EnvIdex]}.pkl', 'wb') as file:
	# 	pickle.dump(act, file)
	# with open(f'res/rew_{EnvName[opt.EnvIdex]}.pkl', 'wb') as file:
	# 	pickle.dump(rew, file)



if __name__ == '__main__':
	main()