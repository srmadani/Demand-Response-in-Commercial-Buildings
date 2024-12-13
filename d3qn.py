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
import time
import os

def evaluate_policy(env, model, mode='validation'):
	if mode == 'validation':
		week_num = 7
		s = env.reset(week_num=week_num)
		done, score = False, 0
		while not done:
			a = model.select_action(s, deterministic=True)
			s_next, r, done, _ = env.step(a)
			score += r
			s = s_next
		return score, env.T_in[1:]
	else:
		week_num = 8
		s = env.reset(week_num=week_num)
		done, score = False, 0
		while not done:
			a = model.select_action(s, deterministic=True)
			s_next, r, done, _ = env.step(a)
			score += r
			s = s_next
		return score, env.T_in[1:], env.load, env.cost, env.cost_components


class LinearSchedule(object):
	def __init__(self, schedule_timesteps, initial_p, final_p):
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

	def __init__(self, opt):
		self.device = device
		
		self.ptr = 0
		self.size = 0

		self.state = torch.zeros((opt.buffer_size, opt.state_dim), device=device)  
		self.action = torch.zeros((opt.buffer_size, 1), dtype=torch.int64, device=device)
		self.reward = torch.zeros((opt.buffer_size, 1), device=device)
		self.done = torch.zeros((opt.buffer_size, 1), dtype=torch.bool, device=device)
		self.priorities = torch.zeros(opt.buffer_size, dtype=torch.float32, device=device) 
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
		Prob_torch_gpu = self.priorities[0: self.size - 1].clone() 
		if self.ptr < self.size: Prob_torch_gpu[self.ptr-1] = 0 
		ind = torch.multinomial(Prob_torch_gpu, num_samples=batch_size, replacement=self.replacement)

		IS_weight = (self.size * Prob_torch_gpu[ind])**(-self.beta)
		Normed_IS_weight = (IS_weight / IS_weight.max()).unsqueeze(-1) 

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
			batch_priorities = ((torch.abs(Q_target - current_Q) + 0.01)**replay_buffer.alpha).squeeze(-1)
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

parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--Max_train_steps', type=int, default=int(1e5), help='Max training steps')
parser.add_argument('--Max_train_time', type=int, default=int(2e5), help='Max training time')
parser.add_argument('--buffer_size', type=int, default=int(2e5), help='size of the replay buffer')
parser.add_argument('--save_interval', type=int, default=int(5e4), help='Model saving interval, in steps.')
parser.add_argument('--eval_interval', type=int, default=int(2e3), help='Model evaluating interval, in steps.')
parser.add_argument('--warmup', type=int, default=int(3e3), help='steps for random policy to explore')
parser.add_argument('--update_every', type=int, default=50, help='training frequency')

parser.add_argument('--gamma', type=float, default=0.98, help='Discounted Factor')
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
# print(opt)


def main(C=100, R=1, h=50, alpha=0.2, render=False, compare=False, gamma=0.55132, cpp = 0.000067, temperature=0):
	start_time = time.time()
	EnvName = [f'HVAC_D3QN_C_{C}_R_{R}_h_{h}_alpha_{alpha}_gamma_{gamma}']
	BriefEnvName = [f'HVAC_D3QN_{C}_{R}_{h}_{alpha}_{gamma}']
	env = HVAC(C=C, R=R, h=h, alpha=alpha, gamma=gamma)
	eval_env = HVAC(C=C, R=R, h=h, alpha=alpha, gamma=gamma)
	opt.state_dim = env.observation_space.shape[0]
	opt.action_dim = env.action_space.n
	opt.max_e_steps = env.T


	#Use DDQN or DQN
	if opt.DDQN: algo_name = 'DDQN'
	else: algo_name = 'DQN'

	#Seed everything
	torch.manual_seed(opt.seed)
	np.random.seed(opt.seed)

	# print('Algorithm:',algo_name,'  Env:',BriefEnvName[opt.EnvIdex],'  state_dim:',opt.state_dim,
	# 	  '  action_dim:',opt.action_dim,'  Random Seed:',opt.seed, '  max_e_steps:',opt.max_e_steps, '\n')

	#Build model and replay buffer
	if not os.path.exists('model'): os.mkdir('model')
	model = DQN_Agent(opt)
	if opt.Loadmodel: model.load(BriefEnvName[opt.EnvIdex])
	buffer = LightPriorReplayBuffer(opt)

	exp_noise_scheduler = LinearSchedule(opt.noise_decay_steps, opt.exp_noise_init, opt.exp_noise_end)
	beta_scheduler = LinearSchedule(opt.beta_gain_steps, opt.beta_init, 1.0)
	lr_scheduler = LinearSchedule(opt.lr_decay_steps, opt.lr_init, opt.lr_end)

	if render:
		model.load(BriefEnvName[opt.EnvIdex])
		res = {}
		res['score'], res['T_in'], res['load'], res['cost'], res['cost_components'] = evaluate_policy(eval_env, model, mode='test')
		print(f"Env: {BriefEnvName[0]}, score: {res['score']}, cost: {res['cost']}")
		with open(f'res/{EnvName[opt.EnvIdex]}.pkl', 'wb') as file:
			pickle.dump(res, file)
		return res['score'], res['T_in'], res['load'], res['cost'], res['cost_components']
	elif compare:
		model.load(BriefEnvName[opt.EnvIdex])
		score, _ = evaluate_policy(eval_env, model, mode='validation')
		return score
	else:
		log_dir = os.path.join("res", "log")
		log_file_name = f"training_log_{BriefEnvName[opt.EnvIdex]}.txt"
		log_file_path = os.path.join(log_dir, log_file_name)
		elapsed_times = []
		rewards = []
		total_steps, elapsed_time, tin, rew, best_score = 0, 0, [], [], -np.inf
		# while total_steps < opt.Max_train_steps:
		while elapsed_time < opt.Max_train_time:
			s = env.reset(week_num=np.random.randint(1,7))
			a, q_a = model.select_action(s, deterministic=False)

			while True:
				s_next, r, done, _ = env.step(a)
				# if r <= -100: r = -10  # good for LunarLander
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
					score, Tin = evaluate_policy(eval_env, model, mode='validation')
					rew.append(score)
					if (total_steps) % (10*opt.eval_interval) == 0:
						tin.append(Tin)
					if score > best_score:
						model.save(BriefEnvName[opt.EnvIdex])
						best_score = score

					elapsed_time = time.time() - start_time
					elapsed_times.append(elapsed_time)
					rewards.append(best_score)
					log_message = f"EnvName: {EnvName[opt.EnvIdex]}, seed: {opt.seed}, steps: {int(total_steps / 1000)}k, score: {int(score)}, elapsed time: {int(elapsed_time)}\n"
					with open(log_file_path, "a") as log_file:
						log_file.write(log_message)
					print(log_message)

				total_steps += 1

				'''save model'''
				if (total_steps) % opt.save_interval == 0:
					model.save(BriefEnvName[opt.EnvIdex])

				if done: break

		with open(f'res/track/track_{EnvName[opt.EnvIdex]}.pkl', 'wb') as f:
			pickle.dump((elapsed_times, rewards), f)
		env.close()
		eval_env.close()
		# with open(f'res/tin_{EnvName[opt.EnvIdex]}.pkl', 'wb') as file:
		# 	pickle.dump(tin, file)
		# with open(f'res/rew_{EnvName[opt.EnvIdex]}.pkl', 'wb') as file:
		# 	pickle.dump(rew, file)

if __name__ == '__main__':
	main(render=True)