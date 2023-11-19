import torch
import numpy as np
import copy
import torch.nn as nn
# import torch.nn.functional as F
# from torch.distributions import Normal
import os, shutil
from datetime import datetime
import argparse
from hvac import HVAC
import pickle

def evaluate_policy(env, model, turns = 3):
	scores = 0
	for _ in range(turns):
		s = env.reset()
		done = False
		while not done:
			# Take deterministic actions at test time
			a = model.select_action(s, deterministic=True)
			s_next, r, done, _ = env.step(a) # dw: terminated; tr: truncated
			scores += r
			s = s_next
	return scores/turns, env.T_in[1:]


class LinearSchedule(object):
	def __init__(self, schedule_timesteps, initial_p, final_p):
		"""Linear interpolation between initial_p and final_p over
		schedule_timesteps. After this many timesteps pass final_p is
		returned.
		Parameters
		----------
		schedule_timesteps: int
			Number of timesteps for which to linearly anneal initial_p
			to final_p
		initial_p: float
			initial output value
		final_p: float
			final output value
		"""
		self.schedule_timesteps = schedule_timesteps
		self.initial_p = initial_p
		self.final_p = final_p

	def value(self, t):
		fraction = min(float(t) / self.schedule_timesteps, 1.0)
		return self.initial_p + fraction * (self.final_p - self.initial_p)


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
		raise argparse.ArgumentTypeError('Boolean value expected.')
	
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class LightPriorReplayBuffer():
	'''
	Obviate the need for explicately saving s_next, more menmory friendly, especially for image state.

	When iterating, use the following way to add new transitions:
		a = model.select(s)
		s_next, r, dw, tr, info = env.step(a)
		buffer.add(s, a, r, dw, tr)  
		# dw: whether the 's_next' is the terminal state
		# tr: whether the episode has been truncated.

	When sampling,
	ind = [ptr - 1] and ind = [size - 1] should be avoided to ensure the consistence of state[ind] and state[ind+1]
	Then,
	s = self.state[ind]
	s_next = self.state[ind+1]

	Importantly, because we do not explicitly save 's_next', when dw or tr is True, the s[ind] and s[ind+1] is not from one episode. 
	when encounter dw=True,
	self.state[ind+1] is not the true next state of self.state[ind], but a new resetted state.
	It doesn't matter, since Q_target[s[ind],a[ind]] = r[ind] + gamma*(1-dw[ind])* max_Q(s[ind+1],·),
	when dw=true, we won't use s[ind+1] at all.
	however, when encounter tr=True,
	self.state[ind+1] is not the true next state of self.state[ind], but a new resetted state, 
	so we have to discard this transition through (1-tr) in the loss function

	Thus, when training,
	Q_target = r + self.gamma * (1-dw) * max_q_next
	current_Q = self.q_net(s).gather(1,a)
	q_loss = torch.square((1-tr) * (current_Q - Q_target)).mean()

	'''

	def __init__(self, opt):
		self.device = device
		
		self.ptr = 0
		self.size = 0

		self.state = torch.zeros((opt.buffer_size, opt.state_dim), device=device)  #如果是图像，可以用unit8节省空间
		self.action = torch.zeros((opt.buffer_size, 1), dtype=torch.int64, device=device)
		self.reward = torch.zeros((opt.buffer_size, 1), device=device)
		self.done = torch.zeros((opt.buffer_size, 1), dtype=torch.bool, device=device) #only 0/1
		self.priorities = torch.zeros(opt.buffer_size, dtype=torch.float32, device=device) # (|TD-error|+0.01)^alpha
		self.buffer_size = opt.buffer_size

		self.alpha = opt.alpha
		self.beta = opt.beta_init
		self.replacement = opt.replacement


	def add(self, state, action, reward, done, priority):
		self.state[self.ptr] = torch.from_numpy(state).to(device)
		self.action[self.ptr] = action
		self.reward[self.ptr] = reward
		self.done[self.ptr] = done
		self.priorities[self.ptr] = priority

		self.ptr = (self.ptr + 1) % self.buffer_size
		self.size = min(self.size + 1, self.buffer_size)


	def sample(self, batch_size):
		# 因为state[ptr-1]和state[ptr]，state[size-1]和state[size]不来自同一个episode
		Prob_torch_gpu = self.priorities[0: self.size - 1].clone() # 所以从[0, size-1)中sample; 这里必须clone
		if self.ptr < self.size: Prob_torch_gpu[self.ptr-1] = 0 # 并且不能包含ptr-1
		ind = torch.multinomial(Prob_torch_gpu, num_samples=batch_size, replacement=self.replacement) # replacement=True数据可能重复，但是快很多; (batchsize,)
		# 注意，这里的ind对于self.priorities和Prob_torch_gpu是通用的，并没有错位

		IS_weight = (self.size * Prob_torch_gpu[ind])**(-self.beta)
		Normed_IS_weight = (IS_weight / IS_weight.max()).unsqueeze(-1) #(batchsize,1)

		return self.state[ind], self.action[ind], self.reward[ind], self.state[ind + 1], self.done[ind], ind, Normed_IS_weight
	
def build_net(layer_shape, activation, output_activation):
	'''build net with for loop'''
	layers = []
	for j in range(len(layer_shape)-1):
		act = activation if j < len(layer_shape)-2 else output_activation
		layers += [nn.Linear(layer_shape[j], layer_shape[j+1]), act()]
	return nn.Sequential(*layers)


class Q_Net(nn.Module):
	def __init__(self, state_dim, action_dim, hid_shape):
		super(Q_Net, self).__init__()
		layers = [state_dim] + list(hid_shape) + [action_dim]
		self.Q = build_net(layers, nn.ReLU, nn.Identity)

	def forward(self, s):
		q = self.Q(s)
		return q



class DQN_Agent(object):
	def __init__(self,opt,):
		self.q_net = Q_Net(opt.state_dim, opt.action_dim, (opt.net_width,opt.net_width)).to(device)
		self.q_net_optimizer = torch.optim.Adam(self.q_net.parameters(), lr=opt.lr_init)
		self.q_target = copy.deepcopy(self.q_net)
		# Freeze target networks with respect to optimizers (only update via polyak averaging)
		for p in self.q_target.parameters(): p.requires_grad = False
		self.gamma = opt.gamma
		self.tau = 0.005
		self.batch_size = opt.batch_size
		self.exp_noise = opt.exp_noise_init
		self.action_dim = opt.action_dim
		self.DDQN = opt.DDQN

	def select_action(self, state, deterministic):#only used when interact with the env
		with torch.no_grad():
			state = torch.FloatTensor(state.reshape(1, -1)).to(device)
			if deterministic:
				a = self.q_net(state).argmax().item()
				return a
			else:
				Q = self.q_net(state)
				if np.random.rand() < self.exp_noise:
					a = np.random.randint(0,self.action_dim)
					q_a = Q[0,a] # on device
				else:
					a = Q.argmax().item()
					q_a = Q[0,a] # on device
				return a, q_a


	def train(self,replay_buffer):
		s, a, r, s_next, done, ind, Normed_IS_weight = replay_buffer.sample(self.batch_size)
		# s, a, r, s_next, done, Normed_IS_weight : (batchsize, dim)
		# ind, : (batchsize,)

		'''Compute the target Q value'''
		with torch.no_grad():
			if self.DDQN:
				argmax_a = self.q_net(s_next).argmax(dim=1).unsqueeze(-1)
				max_q_prime = self.q_target(s_next).gather(1,argmax_a)
			else:
				max_q_prime = self.q_target(s_next).max(1)[0].unsqueeze(1)

			'''Avoid impacts caused by reaching max episode steps'''
			Q_target = r + (~done) * self.gamma * max_q_prime 

		# Get current Q estimates
		current_Q = self.q_net(s).gather(1,a)

		# BP
		q_loss = torch.square((~done) * Normed_IS_weight * (Q_target - current_Q)).mean()
		self.q_net_optimizer.zero_grad()
		q_loss.backward()
		torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), 10.0)
		self.q_net_optimizer.step()

		# update priorites of the current batch
		with torch.no_grad():
			batch_priorities = ((torch.abs(Q_target - current_Q) + 0.01)**replay_buffer.alpha).squeeze(-1) #(batchsize,) on devive
			replay_buffer.priorities[ind] = batch_priorities

		# Update the frozen target models
		for param, target_param in zip(self.q_net.parameters(), self.q_target.parameters()):
			target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)


	def save(self,EnvName):
		torch.save(self.q_net.state_dict(), "./model/{}.pth".format(EnvName))

	def load(self,EnvName):
		self.q_net.load_state_dict(torch.load("./model/{}.pth".format(EnvName)))
		self.q_target.load_state_dict(torch.load("./model/{}.pth".format(EnvName)))
		
'''Hyperparameter Setting'''
parser = argparse.ArgumentParser()
parser.add_argument('--EnvIdex', type=int, default=0, help='HVAC_D3QN')
parser.add_argument('--render', type=str2bool, default=False, help='Render or Not')
parser.add_argument('--Loadmodel', type=str2bool, default=False, help='Load pretrained model or Not')
parser.add_argument('--ModelIdex', type=int, default=250, help='which model to load')

parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--Max_train_steps', type=int, default=int(4e5), help='Max training steps')
parser.add_argument('--buffer_size', type=int, default=int(2e5), help='size of the replay buffer')
parser.add_argument('--save_interval', type=int, default=int(5e4), help='Model saving interval, in steps.')
parser.add_argument('--eval_interval', type=int, default=int(2e3), help='Model evaluating interval, in steps.')
parser.add_argument('--warmup', type=int, default=int(3e3), help='steps for random policy to explore')
parser.add_argument('--update_every', type=int, default=50, help='training frequency')

parser.add_argument('--gamma', type=float, default=0.97, help='Discounted Factor')
parser.add_argument('--net_width', type=int, default=256, help='Hidden net width')
parser.add_argument('--lr_init', type=float, default=5e-3, help='Initial Learning rate')
parser.add_argument('--lr_end', type=float, default=6e-5, help='Final Learning rate')
parser.add_argument('--lr_decay_steps', type=float, default=int(3e5), help='Learning rate decay steps')
parser.add_argument('--batch_size', type=int, default=256, help='lenth of sliced trajectory')
parser.add_argument('--exp_noise_init', type=float, default=0.2, help='init explore noise')
parser.add_argument('--exp_noise_end', type=float, default=0.03, help='final explore noise')
parser.add_argument('--noise_decay_steps', type=int, default=int(1e5), help='decay steps of explore noise')
parser.add_argument('--DDQN', type=str2bool, default=True, help='True:DDQN; False:DQN')

parser.add_argument('--alpha', type=float, default=0.6, help='alpha for PER')
parser.add_argument('--beta_init', type=float, default=0.4, help='beta for PER')
parser.add_argument('--beta_gain_steps', type=int, default=int(3e5), help='steps of beta from beta_init to 1.0')
parser.add_argument('--replacement', type=str2bool, default=True, help='sample method')
opt = parser.parse_args()
print(opt)


def main():
	EnvName = ['HVAC_D3QN']
	BriefEnvName = ['HVAC_D3QN']
	env = HVAC()
	eval_env = HVAC()
	opt.state_dim = env.observation_space.shape[0]
	opt.action_dim = env.action_space.n
	opt.max_e_steps = 7*12*24

	#Use DDQN or DQN
	if opt.DDQN: algo_name = 'DDQN'
	else: algo_name = 'DQN'

	#Seed everything
	torch.manual_seed(opt.seed)
	np.random.seed(opt.seed)

	print('Algorithm:',algo_name,'  Env:',BriefEnvName[opt.EnvIdex],'  state_dim:',opt.state_dim,
		  '  action_dim:',opt.action_dim,'  Random Seed:',opt.seed, '  max_e_steps:',opt.max_e_steps, '\n')

	#Build model and replay buffer
	if not os.path.exists('model'): os.mkdir('model')
	model = DQN_Agent(opt)
	if opt.Loadmodel: model.load(algo_name,BriefEnvName[opt.EnvIdex],opt.ModelIdex)
	buffer = LightPriorReplayBuffer(opt)

	exp_noise_scheduler = LinearSchedule(opt.noise_decay_steps, opt.exp_noise_init, opt.exp_noise_end)
	beta_scheduler = LinearSchedule(opt.beta_gain_steps, opt.beta_init, 1.0)
	lr_scheduler = LinearSchedule(opt.lr_decay_steps, opt.lr_init, opt.lr_end)

	if opt.render:
		score, _ = evaluate_policy(eval_env, model, 20)
		print('EnvName:', BriefEnvName[opt.EnvIdex], 'seed:', opt.seed, 'score:', score)
	else:
		total_steps, tin, rew, best_score = 0, [], [], -np.inf
		while total_steps < opt.Max_train_steps:
			s = env.reset()
			a, q_a = model.select_action(s, deterministic=False)
			# ↑ cover s, a, q_a from last episode ↑

			while True:
				s_next, r, done, _ = env.step(a)
				if r <= -100: r = -10  # good for LunarLander
				a_next, q_a_next = model.select_action(s_next, deterministic=False)

				# [s; a, q_a; r, dw, tr, s_next; a_next, q_a_next] have been all collected.
				priority = (torch.abs(r + (~done)*opt.gamma*q_a_next - q_a) + 0.01)**opt.alpha #scalar
				buffer.add(s, a, r, done, priority)

				s, a, q_a = s_next, a_next, q_a_next

				'''update if its time'''
				# train 50 times every 50 steps rather than 1 training per step. Better!
				if total_steps >= opt.warmup and total_steps % opt.update_every == 0:
					for j in range(opt.update_every):
						model.train(buffer)

					# parameter annealing
					model.exp_noise = exp_noise_scheduler.value(total_steps)
					buffer.beta = beta_scheduler.value(total_steps)
					for p in model.q_net_optimizer.param_groups: p['lr'] = lr_scheduler.value(total_steps)

				'''record & log'''
				if (total_steps) % opt.eval_interval == 0:
					score, Tin = evaluate_policy(eval_env, model)
					rew.append(score)
					if (total_steps) % (10*opt.eval_interval) == 0:
						tin.append(Tin)
					if score > best_score:
						model.save(BriefEnvName[opt.EnvIdex])
						best_score = score

					print('EnvName:',BriefEnvName[opt.EnvIdex],'seed:',opt.seed,'steps: {}k'.format(int(total_steps/1000)),'score:', int(score))

				total_steps += 1

				'''save model'''
				if (total_steps) % opt.save_interval == 0:
					model.save(BriefEnvName[opt.EnvIdex])

				if done: break

	env.close()
	eval_env.close()
	with open(f'res/tin_{EnvName[opt.EnvIdex]}.pkl', 'wb') as file:
		pickle.dump(tin, file)
	with open(f'res/rew_{EnvName[opt.EnvIdex]}.pkl', 'wb') as file:
		pickle.dump(rew, file)

if __name__ == '__main__':
	main()