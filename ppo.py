import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Beta,Normal
import numpy as np
import copy
import math
from datetime import datetime
import os, shutil
from ctrl import CTRL
import argparse
import pickle
import plotly.graph_objects as go
import time
# import mlflow

# mlflow.set_tracking_uri("sqlite:///mlflow.db") #The name of the database to use
# mlflow.set_experiment("ctrl-ppo") #If already exists mlflow will append to existing data. Else it will make a new experiment.

def act_clipper(a):
	if a>1:
		return 1
	elif a<0:
		return 0
	else:
		return a

class BetaActor(nn.Module):
	def __init__(self, state_dim, action_dim, net_width):
		super(BetaActor, self).__init__()

		self.l1 = nn.Linear(state_dim, net_width)
		self.l2 = nn.Linear(net_width, net_width)
		self.alpha_head = nn.Linear(net_width, action_dim)
		self.beta_head = nn.Linear(net_width, action_dim)

	def forward(self, state):
		a = torch.tanh(self.l1(state))
		a = torch.tanh(self.l2(a))

		alpha = F.softplus(self.alpha_head(a)) + 1.0
		beta = F.softplus(self.beta_head(a)) + 1.0

		return alpha,beta

	def get_dist(self,state):
		alpha,beta = self.forward(state)
		dist = Beta(alpha, beta)
		return dist

	def deterministic_act(self, state):
		alpha, beta = self.forward(state)
		mode = (alpha) / (alpha + beta)
		return mode

class GaussianActor_musigma(nn.Module):
	def __init__(self, state_dim, action_dim, net_width):
		super(GaussianActor_musigma, self).__init__()

		self.l1 = nn.Linear(state_dim, net_width)
		self.l2 = nn.Linear(net_width, net_width)
		self.mu_head = nn.Linear(net_width, action_dim)
		self.sigma_head = nn.Linear(net_width, action_dim)

	def forward(self, state):
		a = torch.tanh(self.l1(state))
		a = torch.tanh(self.l2(a))
		mu = torch.sigmoid(self.mu_head(a))
		sigma = F.softplus( self.sigma_head(a) )
		return mu,sigma

	def get_dist(self, state):
		mu,sigma = self.forward(state)
		dist = Normal(mu,sigma)
		return dist

	def deterministic_act(self, state):
		mu, sigma = self.forward(state)
		return mu


class GaussianActor_mu(nn.Module):
	def __init__(self, state_dim, action_dim, net_width, log_std=0):
		super(GaussianActor_mu, self).__init__()

		self.l1 = nn.Linear(state_dim, net_width)
		self.l2 = nn.Linear(net_width, net_width)
		self.mu_head = nn.Linear(net_width, action_dim)
		self.mu_head.weight.data.mul_(0.1)
		self.mu_head.bias.data.mul_(0.0)

		self.action_log_std = nn.Parameter(torch.ones(1, action_dim) * log_std)

	def forward(self, state):
		a = torch.relu(self.l1(state))
		a = torch.relu(self.l2(a))
		mu = torch.sigmoid(self.mu_head(a))
		return mu

	def get_dist(self,state):
		mu = self.forward(state)
		action_log_std = self.action_log_std.expand_as(mu)
		action_std = torch.exp(action_log_std)

		dist = Normal(mu, action_std)
		return dist

	def deterministic_act(self, state):
		return self.forward(state)


class Critic(nn.Module):
	def __init__(self, state_dim,net_width):
		super(Critic, self).__init__()

		self.C1 = nn.Linear(state_dim, net_width)
		self.C2 = nn.Linear(net_width, net_width)
		self.C3 = nn.Linear(net_width, 1)

	def forward(self, state):
		v = torch.tanh(self.C1(state))
		v = torch.tanh(self.C2(v))
		v = self.C3(v)
		return v

def str2bool(v):
	'''transfer str to bool for argparse'''
	if isinstance(v, bool):
		return v
	if v.lower() in ('yes', 'True','true','TRUE', 't', 'y', '1'):
		return True
	elif v.lower() in ('no', 'False','false','FALSE', 'f', 'n', '0'):
		return False
	else:
		print('Wrong Input.')
		raise


def evaluate_policy(env, model, mode='validation'):
	if mode == 'validation':
		week_num = 7
	else:
		week_num = 8
	s, done, scores= env.reset(week_num = week_num), False, 0
	while not done:
		a = model.select_action(s, deterministic = True)
		s_next, r, done, _ = env.step(act_clipper(a[0]))
		scores += r
		s = s_next
	return scores, env.load, env.cost
	
class PPO_agent(object):
	def __init__(self, **kwargs):
		# Init hyperparameters for PPO agent, just like "self.gamma = opt.gamma, self.lambd = opt.lambd, ..."
		self.__dict__.update(kwargs)

		# Choose distribution for the actor
		if self.Distribution == 'Beta':
			self.actor = BetaActor(self.state_dim, self.action_dim, self.net_width).to(self.dvc)
		elif self.Distribution == 'GS_ms':
			self.actor = GaussianActor_musigma(self.state_dim, self.action_dim, self.net_width).to(self.dvc)
		elif self.Distribution == 'GS_m':
			self.actor = GaussianActor_mu(self.state_dim, self.action_dim, self.net_width).to(self.dvc)
		else: print('Dist Error')
		self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.a_lr)

		# Build Critic
		self.critic = Critic(self.state_dim, self.net_width).to(self.dvc)
		self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.c_lr)

		# Build Trajectory holder
		self.s_hoder = np.zeros((self.T_horizon, self.state_dim),dtype=np.float32)
		self.a_hoder = np.zeros((self.T_horizon, self.action_dim),dtype=np.float32)
		self.r_hoder = np.zeros((self.T_horizon, 1),dtype=np.float32)
		self.s_next_hoder = np.zeros((self.T_horizon, self.state_dim),dtype=np.float32)
		self.logprob_a_hoder = np.zeros((self.T_horizon, self.action_dim),dtype=np.float32)
		self.done_hoder = np.zeros((self.T_horizon, 1),dtype=np.bool_)

	def select_action(self, state, deterministic):
		with torch.no_grad():
			state = torch.FloatTensor(state.reshape(1, -1)).to(self.dvc)
			if deterministic:
				# only used when evaluate the policy.Making the performance more stable
				a = self.actor.deterministic_act(state)
				return a.cpu().numpy()[0], None  # action is in shape (adim, 0)
			else:
				# only used when interact with the env
				dist = self.actor.get_dist(state)
				a = dist.sample()
				a = torch.clamp(a, 0, 1)
				logprob_a = dist.log_prob(a).cpu().numpy().flatten()
				return a.cpu().numpy()[0], logprob_a # both are in shape (adim, 0)


	def train(self):
		self.entropy_coef*=self.entropy_coef_decay

		'''Prepare PyTorch data from Numpy data'''
		s = torch.from_numpy(self.s_hoder).to(self.dvc)
		a = torch.from_numpy(self.a_hoder).to(self.dvc)
		r = torch.from_numpy(self.r_hoder).to(self.dvc)
		s_next = torch.from_numpy(self.s_next_hoder).to(self.dvc)
		logprob_a = torch.from_numpy(self.logprob_a_hoder).to(self.dvc)
		done = torch.from_numpy(self.done_hoder).to(self.dvc)

		''' Use TD+GAE+LongTrajectory to compute Advantage and TD target'''
		with torch.no_grad():
			vs = self.critic(s)
			vs_ = self.critic(s_next)

			deltas = r + self.gamma * vs_ - vs
			deltas = deltas.cpu().flatten().numpy()
			adv = [0]

			'''done for GAE'''
			for dlt, mask in zip(deltas[::-1], done.cpu().flatten().numpy()[::-1]):
				advantage = dlt + self.gamma * self.lambd * adv[-1] * (~mask)
				adv.append(advantage)
			adv.reverse()
			adv = copy.deepcopy(adv[0:-1])
			adv = torch.tensor(adv).unsqueeze(1).float().to(self.dvc)
			td_target = adv + vs
			adv = (adv - adv.mean()) / ((adv.std()+1e-4))  #sometimes helps


		"""Slice long trajectopy into short trajectory and perform mini-batch PPO update"""
		a_optim_iter_num = int(math.ceil(s.shape[0] / self.a_optim_batch_size))
		c_optim_iter_num = int(math.ceil(s.shape[0] / self.c_optim_batch_size))
		for i in range(self.K_epochs):

			#Shuffle the trajectory, Good for training
			perm = np.arange(s.shape[0])
			np.random.shuffle(perm)
			perm = torch.LongTensor(perm).to(self.dvc)
			s, a, td_target, adv, logprob_a = \
				s[perm].clone(), a[perm].clone(), td_target[perm].clone(), adv[perm].clone(), logprob_a[perm].clone()

			'''update the actor'''
			for i in range(a_optim_iter_num):
				index = slice(i * self.a_optim_batch_size, min((i + 1) * self.a_optim_batch_size, s.shape[0]))
				distribution = self.actor.get_dist(s[index])
				dist_entropy = distribution.entropy().sum(1, keepdim=True)
				logprob_a_now = distribution.log_prob(a[index])
				ratio = torch.exp(logprob_a_now.sum(1,keepdim=True) - logprob_a[index].sum(1,keepdim=True))  # a/b == exp(log(a)-log(b))

				surr1 = ratio * adv[index]
				surr2 = torch.clamp(ratio, 1 - self.clip_rate, 1 + self.clip_rate) * adv[index]
				a_loss = -torch.min(surr1, surr2) - self.entropy_coef * dist_entropy

				self.actor_optimizer.zero_grad()
				a_loss.mean().backward()
				torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 40)
				self.actor_optimizer.step()

			'''update the critic'''
			for i in range(c_optim_iter_num):
				index = slice(i * self.c_optim_batch_size, min((i + 1) * self.c_optim_batch_size, s.shape[0]))
				c_loss = (self.critic(s[index]) - td_target[index]).pow(2).mean()
				for name,param in self.critic.named_parameters():
					if 'weight' in name:
						c_loss += param.pow(2).sum() * self.l2_reg

				self.critic_optimizer.zero_grad()
				c_loss.backward()
				self.critic_optimizer.step()

	def put_data(self, s, a, r, s_next, logprob_a, done, idx):
		self.s_hoder[idx] = s
		self.a_hoder[idx] = a
		self.r_hoder[idx] = r
		self.s_next_hoder[idx] = s_next
		self.logprob_a_hoder[idx] = logprob_a
		self.done_hoder[idx] = done


	def save(self,name):
		torch.save(self.critic.state_dict(), "./model/{}_critic.pth".format(name))
		torch.save(self.actor.state_dict(), "./model/{}_actor.pth".format(name))


	def load(self,name):
		self.critic.load_state_dict(torch.load("./model/{}_critic.pth".format(name)))
		self.actor.load_state_dict(torch.load("./model/{}_actor.pth".format(name)))
		
'''Hyperparameter Setting'''
parser = argparse.ArgumentParser()
parser.add_argument('--dvc', type=str, default='cuda', help='running device: cuda or cpu')
parser.add_argument('--EnvIdex', type=int, default=0, help='CTRL_PPO')
parser.add_argument('--render', type=str2bool, default=False, help='Render or Not')
parser.add_argument('--Loadmodel', type=str2bool, default=False, help='Load pretrained model or Not')
parser.add_argument('--ModelIdex', type=int, default=400, help='which model to load')

parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--T_horizon', type=int, default=7*7*12, help='lenth of long trajectory')
parser.add_argument('--Distribution', type=str, default='Beta', help='Should be one of Beta ; GS_ms  ;  GS_m')
parser.add_argument('--Max_train_steps', type=int, default=int(4e5), help='Max training steps')
parser.add_argument('--Max_train_time', type=int, default=1e5, help='Max training time')
parser.add_argument('--save_interval', type=int, default=int(4e4), help='Model saving interval, in steps.')
parser.add_argument('--eval_interval', type=int, default=int(2e2), help='Model evaluating interval, in steps.')

parser.add_argument('--gamma', type=float, default=0.97, help='Discounted Factor')
parser.add_argument('--lambd', type=float, default=0.95, help='GAE Factor')
parser.add_argument('--clip_rate', type=float, default=0.2, help='PPO Clip rate')
parser.add_argument('--K_epochs', type=int, default=10, help='PPO update times')
parser.add_argument('--net_width', type=int, default=150, help='Hidden net width')
parser.add_argument('--a_lr', type=float, default=2e-5, help='Learning rate of actor')
parser.add_argument('--c_lr', type=float, default=2e-5, help='Learning rate of critic')
parser.add_argument('--l2_reg', type=float, default=1e-2, help='L2 regulization coefficient for Critic')
parser.add_argument('--a_optim_batch_size', type=int, default=64, help='lenth of sliced trajectory of actor')
parser.add_argument('--c_optim_batch_size', type=int, default=64, help='lenth of sliced trajectory of critic')
parser.add_argument('--entropy_coef', type=float, default=1e-3, help='Entropy coefficient of Actor')
parser.add_argument('--entropy_coef_decay', type=float, default=0.99, help='Decay rate of entropy_coef')
opt = parser.parse_args()
opt.dvc = torch.device(opt.dvc) # from str to torch.device
# print(opt)


def main(beta=0.1, gamma=0.55132, render=False, compare=False, cpp=0.000067):
	#mlflow.set_tag("alg", 'ppo')
	#mlflow.set_tag("beta", beta)
	#mlflow.set_tag("gamma", gamma)
	#mlflow.set_tag("cpp", cpp)
	start_time = time.time()
	EnvName = [f'CTRL_PPO_beta_{beta}_gamma_{gamma}']
	BrifEnvName = [f'CTRL_PPO_{beta}_{gamma}']

	# Build Env
	env = CTRL(beta=beta, gamma=gamma)
	eval_env = CTRL(beta=beta, gamma=gamma)
	opt.state_dim = env.observation_space.shape[0]
	opt.action_dim = env.action_space.shape[0]
	opt.max_action = 1
	opt.max_steps = 7*24*12
	# print('Env:',EnvName[opt.EnvIdex],'  state_dim:',opt.state_dim,'  action_dim:',opt.action_dim,
	# 	  '  max_a:',opt.max_action,'  min_a:',env.action_space.low[0], 'max_steps', opt.max_steps)

	# Seed Everything
	env_seed = opt.seed
	np.random.seed(opt.seed)
	torch.manual_seed(opt.seed)
	torch.cuda.manual_seed(opt.seed)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False
	# print("Random Seed: {}".format(opt.seed))

	# Beta dist maybe need larger learning rate, Sometimes helps
	# if Dist[distnum] == 'Beta' :
	#     kwargs["a_lr"] *= 2
	#     kwargs["c_lr"] *= 4

	if not os.path.exists('model'): os.mkdir('model')
	agent = PPO_agent(**vars(opt)) # transfer opt to dictionary, and use it to init PPO_agent
	if opt.Loadmodel: agent.load(BrifEnvName[opt.EnvIdex])
	#mlflow.log_params(opt.__dict__)

	if render:
		agent.load(BrifEnvName[opt.EnvIdex])
		res = {}
		res['score'], res['load'], res['cost'] = evaluate_policy(eval_env, agent, mode='test')
		print(f"EnvName:{BrifEnvName[opt.EnvIdex]}, score: {res['score']}, cost: {res['cost']}")
		with open(f'res/{EnvName[opt.EnvIdex]}.pkl', 'wb') as file:
			pickle.dump(res, file)
	elif compare:
		agent.load(BrifEnvName[opt.EnvIdex])
		score, _, _ = evaluate_policy(eval_env, agent, mode='validation')
		return score
	else:
		elapsed_times = []
		rewards = []
		total_steps, elapsed_time, traj_lenth, act, rew, best_score = 0, 0, 0, [], [], -np.inf
		# while total_steps < opt.Max_train_steps:
		while elapsed_time < opt.Max_train_time:
			s = env.reset(week_num=np.random.randint(1,7))
			done = False

			'''Interact & trian'''
			while not done:
				'''Interact with Env'''
				a, logprob_a = agent.select_action(s, deterministic=False) # use stochastic when training
				s_next, r, done, _ = env.step(act_clipper(a)) 

				'''Store the current transition'''
				agent.put_data(s, a, r, s_next, logprob_a, done, idx = traj_lenth)
				s = s_next

				traj_lenth += 1
				total_steps += 1

				'''Update if its time'''
				if traj_lenth % opt.T_horizon == 0:
					agent.train()
					traj_lenth = 0

				'''Record & log'''
				if total_steps % opt.eval_interval == 0:
					score, _, _ = evaluate_policy(eval_env, agent, mode='validation') # evaluate the policy for 3 times, and get averaged result
					rew.append(np.array(score))
					if score > best_score:
						agent.save(BrifEnvName[opt.EnvIdex])
						best_score = score

					elapsed_time = time.time() - start_time
					elapsed_times.append(elapsed_time)
					rewards.append(best_score)
					elapsed_time = time.time() - start_time
					print(f'EnvName:{BrifEnvName[opt.EnvIdex]} , Steps: {int(total_steps/1000)}k, Episode Reward:{score}, Elapsed Time:{elapsed_time}')
					#mlflow.log_metric("score",score[0])
					#mlflow.log_metric("time",elapsed_time)


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