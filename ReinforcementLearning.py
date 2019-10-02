from parameters import *
from collections import deque
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from neural_net import *
from collections import deque
if WITHGUI:
	import wx


# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

critic_DQN = q_network().to(device)
actor_DQN = q_network().to(device)
actor_DQN.load_state_dict(critic_DQN.state_dict())
actor_DQN.eval()

loss_function = nn.MSELoss()
optimizer = optim.Adam(critic_DQN.parameters(), lr=INIT_LEARNING_RATE)

def GetActionsEGreedy(q_values):
	actions=np.empty(len(q_values), dtype=np.int64)
	for i in range(len(actions)):
		if np.random.rand() < EPSILON:
			actions[i]=np.random.randint(N_ACTIONS) # random action
		else:
			actions[i]=np.argmax(q_values[i]) # optimal action
	return actions

def GetActionsBoltzmann(q_values):
	Boltzmann_T=0.1 # temperature parameter which monotonically decreases with rising number of training episodes
	actionlist=np.arange(N_ACTIONS)
	actions=np.empty(len(q_values), dtype=np.int64)
	for i in range(N_ACTIONS):
		softmax_distribution=F.softmax(q_values[i]/Boltzmann_T, dim=0).numpy()
		actions[i]=np.random.choice(actionlist, p=softmax_distribution)
	return actions

if EXPLORATION_STRATEGY=='e-greedy':
	GetActions=GetActionsEGreedy
else:
	GetActions=GetActionsBoltzmann

def AdjustLearningRate(optimizer, training_step):
	halving_steps=int(training_step / LEARNING_RATE_LIFETIME)
	lr = INIT_LEARNING_RATE/pow(2,halving_steps)
	for param_group in optimizer.param_groups:
		param_group['lr'] = lr
		
class ReinforcementLearning():
	def __init__(self, model):
		self.model=model
		self.reward_queue = deque([], maxlen=MEMORY_SIZE) #queue list of rewards for performance measurement
		self.replay_memory = deque([], maxlen=MEMORY_SIZE) #IMPLEMENTING THE REPLAY MEMORY

	def GetMemorySample(self, batch_size):
		indices = np.random.permutation(len(self.replay_memory))[:batch_size]
		cols = [[], [], [], []] # state, action, reward, next_state
		for idx in indices:
			for col, value in zip(cols, self.replay_memory[idx]):
				col.append(value)
		cols = [np.array(col) for col in cols]
		return (cols[0], cols[1], cols[2], cols[3])

	def SetCanvas(self, modelcanvas):
		self.modelcanvas=modelcanvas

	def StartLearning(self):
		training_step=0
		for iteration in np.arange(TRAINING_START+TRAINING_STEPS*TRAINING_INTERVAL):
			if WITHGUI:
				wx.Yield()
			# Actor evaluates action
			states=self.model.GetVIs()
			with torch.no_grad():
				Q_values = actor_DQN(torch.from_numpy(states).type('torch.FloatTensor'))
				actions = GetActions(Q_values)

			# Actor plays
			self.model.Update(actions)
			rewards = self.model.GetRewards()
			next_states=self.model.GetVIs()

			# Append to reward list for performance measurement
			self.reward_queue.extend(rewards)

			# Add experience to replay_memory
			self.replay_memory.extend(zip(states, actions, rewards, next_states))

			#print('iteration:', iteration)
			if WITHGUI and DRAWING:
				self.modelcanvas.OnDraw()

			# Check the condition for learning step
			if iteration < TRAINING_START or (iteration-TRAINING_START) % TRAINING_INTERVAL != 0:
				continue

			f = open('data.txt','a') 
			f.write(str(training_step)+' '+str(np.mean(self.reward_queue))+'\n')
			f.close()
			print(training_step, np.mean(self.reward_queue))

			# Critic learning step
			old_VIs, actions, rewards, next_VIs = self.GetMemorySample(BATCH_SIZE)
			next_Q_values = actor_DQN(torch.from_numpy(next_VIs).type('torch.FloatTensor')).detach()
			max_next_Q_values = torch.max(next_Q_values, dim=1, keepdim=True)[0] #we need only the first element of the resulting tuple
			target_Q_values = torch.from_numpy(rewards).view(-1, 1).type(torch.FloatTensor) +  DISCOUNT_RATE * max_next_Q_values  #expected_state_action_values

			Q_values_for_actions=critic_DQN(torch.from_numpy(old_VIs).type('torch.FloatTensor')).gather(1, torch.unsqueeze(torch.from_numpy(np.array(actions)), 1))

			loss = loss_function(Q_values_for_actions, target_Q_values)

			AdjustLearningRate(optimizer, training_step)
			optimizer.zero_grad()
			loss.backward()
			for param in critic_DQN.parameters():
				param.grad.data.clamp(-1, 1)
			optimizer.step()
			self.reward_queue
			training_step+=1

			# Regularly copy critic to actor
			if training_step % COPY_STEPS == 0:
				actor_DQN.load_state_dict(critic_DQN.state_dict())



