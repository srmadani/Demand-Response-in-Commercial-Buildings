import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.distributions import Categorical
import copy
import math
import os#, shutil
# from datetime import datetime
from hvac import HVAC
import argparse
import pickle
import plotly.graph_objects as go
import time
import os

class Actor(nn.Module):
	def __init__(self, state_dim, action_dim, net_width):
		super(Actor, self).__init__()

		self.l1 = nn.Linear(state_dim, net_width)
		self.l2 = nn.Linear(net_width, net_width)
		self.l3 = nn.Linear(net_width, action_dim)

	def forward(self, state):
		n = torch.tanh(self.l1(state))
		n = torch.tanh(self.l2(n))
		return n

	def pi(self, state, softmax_dim = 0):
		n = self.forward(state)
		prob = F.softmax(self.l3(n), dim=softmax_dim)
		return prob

class Critic(nn.Module):
	def __init__(self, state_dim,net_width):
		super(Critic, self).__init__()

		self.C1 = nn.Linear(state_dim, net_width)
		self.C2 = nn.Linear(net_width, net_width)
		self.C3 = nn.Linear(net_width, 1)

	def forward(self, state):
		v = torch.relu(self.C1(state))
		v = torch.relu(self.C2(v))
		v = self.C3(v)
		return v

def evaluate_policy(env, model, mode='validation'):
	if mode == 'validation':
		week_num = 7
		s = env.reset(week_num=week_num)
		done, score = False, 0
		while not done:
			a = model.select_action(s, deterministic=True)
			s_next, r, done, _ = env.step(a[0])
			score += r
			s = s_next
		return score, env.T_in[1:]
	else:
		week_num = 8
		s = env.reset(week_num=week_num)
		done, score = False, 0
		while not done:
			a = model.select_action(s, deterministic=True)
			s_next, r, done, _ = env.step(a[0])
			score += r
			s = s_next
		return score, env.T_in[1:], env.load, env.cost, env.cost_components


#You can just ignore this funciton. Is not related to the RL.
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

class PPO_discrete():
	def __init__(self, **kwargs):
		# Init hyperparameters for PPO agent, just like "self.gamma = opt.gamma, self.lambd = opt.lambd, ..."
		self.__dict__.update(kwargs)

		'''Build Actor and Critic'''
		self.actor = Actor(self.state_dim, self.action_dim, self.net_width).to(self.dvc)
		self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.lr)
		self.critic = Critic(self.state_dim, self.net_width).to(self.dvc)
		self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.lr)

		'''Build Trajectory holder'''
		self.s_hoder = np.zeros((self.T_horizon, self.state_dim), dtype=np.float32)
		self.a_hoder = np.zeros((self.T_horizon, 1), dtype=np.int64)
		self.r_hoder = np.zeros((self.T_horizon, 1), dtype=np.float32)
		self.s_next_hoder = np.zeros((self.T_horizon, self.state_dim), dtype=np.float32)
		self.logprob_a_hoder = np.zeros((self.T_horizon, 1), dtype=np.float32)
		self.done_hoder = np.zeros((self.T_horizon, 1), dtype=np.bool_)

	def select_action(self, s, deterministic):
		s = torch.from_numpy(s).float().to(self.dvc)
		with torch.no_grad():
			pi = self.actor.pi(s, softmax_dim=0)
			if deterministic:
				a = torch.argmax(pi).item()
				return a, None
			else:
				m = Categorical(pi)
				a = m.sample().item()
				pi_a = pi[a].item()
				return a, pi_a

	def train(self):
		self.entropy_coef *= self.entropy_coef_decay #exploring decay
		'''Prepare PyTorch data from Numpy data'''
		s = torch.from_numpy(self.s_hoder).to(self.dvc)
		a = torch.from_numpy(self.a_hoder).to(self.dvc)
		r = torch.from_numpy(self.r_hoder).to(self.dvc)
		s_next = torch.from_numpy(self.s_next_hoder).to(self.dvc)
		old_prob_a = torch.from_numpy(self.logprob_a_hoder).to(self.dvc)
		done = torch.from_numpy(self.done_hoder).to(self.dvc)

		''' Use TD+GAE+LongTrajectory to compute Advantage and TD target'''
		with torch.no_grad():
			vs = self.critic(s)
			vs_ = self.critic(s_next)

			'''dw(dead and win) for TD_target and Adv'''
			deltas = r + self.gamma * vs_ * (~done) - vs
			deltas = deltas.cpu().flatten().numpy()
			adv = [0]

			'''done for GAE'''
			for dlt, done in zip(deltas[::-1], done.cpu().flatten().numpy()[::-1]):
				advantage = dlt + self.gamma * self.lambd * adv[-1] * (~done)
				adv.append(advantage)
			adv.reverse()
			adv = copy.deepcopy(adv[0:-1])
			adv = torch.tensor(adv).unsqueeze(1).float().to(self.dvc)
			td_target = adv + vs
			if self.adv_normalization:
				adv = (adv - adv.mean()) / ((adv.std() + 1e-4))  #sometimes helps

		"""PPO update"""
		#Slice long trajectopy into short trajectory and perform mini-batch PPO update
		optim_iter_num = int(math.ceil(s.shape[0] / self.batch_size))

		for _ in range(self.K_epochs):
			#Shuffle the trajectory, Good for training
			perm = np.arange(s.shape[0])
			np.random.shuffle(perm)
			perm = torch.LongTensor(perm).to(self.dvc)
			s, a, td_target, adv, old_prob_a = \
				s[perm].clone(), a[perm].clone(), td_target[perm].clone(), adv[perm].clone(), old_prob_a[perm].clone()

			'''mini-batch PPO update'''
			for i in range(optim_iter_num):
				index = slice(i * self.batch_size, min((i + 1) * self.batch_size, s.shape[0]))

				'''actor update'''
				prob = self.actor.pi(s[index], softmax_dim=1)
				entropy = Categorical(prob).entropy().sum(0, keepdim=True)
				prob_a = prob.gather(1, a[index])
				ratio = torch.exp(torch.log(prob_a) - torch.log(old_prob_a[index]))  # a/b == exp(log(a)-log(b))

				surr1 = ratio * adv[index]
				surr2 = torch.clamp(ratio, 1 - self.clip_rate, 1 + self.clip_rate) * adv[index]
				a_loss = -torch.min(surr1, surr2) - self.entropy_coef * entropy

				self.actor_optimizer.zero_grad()
				a_loss.mean().backward()
				torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 40)
				self.actor_optimizer.step()

				'''critic update'''
				c_loss = (self.critic(s[index]) - td_target[index]).pow(2).mean()
				for name, param in self.critic.named_parameters():
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

	def save(self, name):
		torch.save(self.critic.state_dict(), "./model/{}_critic.pth".format(name))
		torch.save(self.actor.state_dict(), "./model/{}_actor.pth".format(name))

	def load(self, name):
		self.critic.load_state_dict(torch.load("./model/{}_critic.pth".format(name)))
		self.actor.load_state_dict(torch.load("./model/{}_actor.pth".format(name)))

'''Hyperparameter Setting'''
parser = argparse.ArgumentParser()
parser.add_argument('--dvc', type=str, default='cuda', help='running device: cuda or cpu')
parser.add_argument('--EnvIdex', type=int, default=0, help='HVAC_PPOD')
parser.add_argument('--render', type=str2bool, default=False, help='Render or Not')
parser.add_argument('--Loadmodel', type=str2bool, default=False, help='Load pretrained model or Not')

parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--T_horizon', type=int, default=7*24*12, help='lenth of long trajectory')
parser.add_argument('--Max_train_steps', type=int, default=1e5, help='Max training steps')
parser.add_argument('--Max_train_time', type=int, default=2e5, help='Max training time')
parser.add_argument('--save_interval', type=int, default=1e5, help='Model saving interval, in steps.')
parser.add_argument('--eval_interval', type=int, default=2e3, help='Model evaluating interval, in steps.')

parser.add_argument('--gamma', type=float, default=0.98, help='Discounted Factor')
parser.add_argument('--lambd', type=float, default=0.92, help='GAE Factor')
parser.add_argument('--clip_rate', type=float, default=0.15, help='PPO Clip rate')
parser.add_argument('--K_epochs', type=int, default=40, help='PPO update times')
parser.add_argument('--net_width', type=int, default=64, help='Hidden net width')
parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
parser.add_argument('--l2_reg', type=float, default=1e-3, help='L2 regulization coefficient for Critic')
parser.add_argument('--batch_size', type=int, default=2048, help='lenth of sliced trajectory')
parser.add_argument('--entropy_coef', type=float, default=1e-3, help='Entropy coefficient of Actor')
parser.add_argument('--entropy_coef_decay', type=float, default=0.97, help='Decay rate of entropy_coef')
parser.add_argument('--adv_normalization', type=str2bool, default=True, help='Advantage normalization')
opt = parser.parse_args()
opt.dvc = torch.device(opt.dvc) # from str to torch.device
# print(opt)

def main(C=100, R=2, h=80, alpha=0.3, render=False, compare=False, gamma=0.55132, cpp = 0.000067, temperature=0):
	start_time = time.time()
	EnvName = [f'HVAC_PPOD_C_{C}_R_{R}_h_{h}_alpha_{alpha}_gamma_{gamma}']
	BriefEnvName = [f'HVAC_PPOD_{C}_{R}_{h}_{alpha}_{gamma}']
	env = HVAC(C=C, R=R, h=h, alpha=alpha, gamma=gamma, cpp=cpp, temperature=temperature)
	eval_env = HVAC(C=C, R=R, h=h, alpha=alpha, gamma=gamma, cpp=cpp, temperature=temperature)
	opt.state_dim = env.observation_space.shape[0]
	opt.action_dim = env.action_space.n
	opt.max_e_steps = env.T

	# Seed Everything
	env_seed = opt.seed
	np.random.seed(opt.seed)
	torch.manual_seed(opt.seed)
	torch.cuda.manual_seed(opt.seed)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False
	# print("Random Seed: {}".format(opt.seed))

	# print('Env:',BriefEnvName[opt.EnvIdex],'  state_dim:',opt.state_dim,'  action_dim:',opt.action_dim,'   Random Seed:',opt.seed, '  max_e_steps:',opt.max_e_steps)
	# print('\n')

	if not os.path.exists('model'): os.mkdir('model')
	agent = PPO_discrete(**vars(opt))
	if opt.Loadmodel: agent.load(BriefEnvName[opt.EnvIdex])

	if render:
		agent.load(BriefEnvName[opt.EnvIdex])
		res = {}		
		res['score'], res['T_in'], res['load'], res['cost'], res['cost_components'] = evaluate_policy(eval_env, agent, mode='test')
		print(f"Env: {EnvName[opt.EnvIdex]}, score: {res['score']}, cost: {res['cost']}")
		with open(f'res/{EnvName[opt.EnvIdex]}.pkl', 'wb') as file:
			pickle.dump(res, file)
		return res['score'], res['T_in'], res['load'], res['cost'], res['cost_components']
	elif compare:
		agent.load(BriefEnvName[opt.EnvIdex])
		score, _ = evaluate_policy(eval_env, agent, mode='validation')
		return score
	else:
		log_dir = os.path.join("res", "log")
		log_file_name = f"training_log_{BriefEnvName[opt.EnvIdex]}.txt"
		log_file_path = os.path.join(log_dir, log_file_name)
		elapsed_times = []
		rewards = []
		traj_lenth, total_steps, elapsed_time, best_score = 0, 0, 0, -np.inf
		tin, rew = [], []
		# while total_steps < opt.Max_train_steps:
		while elapsed_time < opt.Max_train_time:
			s = env.reset(week_num=np.random.randint(1,7))
			done = False

			'''Interact & trian'''
			while not done:
				'''Interact with Env'''
				a, logprob_a = agent.select_action(s, deterministic=False) # use stochastic when training
				s_next, r, done, _ = env.step(a)

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
					score, Tin = evaluate_policy(eval_env, agent, mode='validation')
					rew.append(score)
					if (total_steps) % (10*opt.eval_interval) == 0:
						tin.append(Tin)
					if score > best_score:
						agent.save(BriefEnvName[opt.EnvIdex])
						best_score = score

					elapsed_time = time.time() - start_time
					elapsed_times.append(elapsed_time)
					rewards.append(best_score)
					log_message = f"EnvName: {EnvName[opt.EnvIdex]}, seed: {opt.seed}, steps: {int(total_steps / 1000)}k, score: {int(score)}, elapsed time: {int(elapsed_time)}\n"
					with open(log_file_path, "a") as log_file:
						log_file.write(log_message)
					print(log_message)

		with open(f'res/track/track_{EnvName[opt.EnvIdex]}.pkl', 'wb') as f:
			pickle.dump((elapsed_times, rewards), f)
		env.close()
		eval_env.close()
	# with open(f'res/tin_{EnvName[opt.EnvIdex]}.pkl', 'wb') as file:
	# 	pickle.dump(tin, file)
	# with open(f'res/rew_{EnvName[opt.EnvIdex]}.pkl', 'wb') as file:
	# 	pickle.dump(rew, file)

if __name__ == '__main__':
	main()