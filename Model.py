import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import math
import time
from IPython import embed
from collections import OrderedDict
import numpy as np
import copy

MultiVariateNormal = torch.distributions.Normal
temp = MultiVariateNormal.log_prob
MultiVariateNormal.log_prob = lambda self, val: temp(self,val).sum(-1, keepdim = True)

temp2 = MultiVariateNormal.entropy
MultiVariateNormal.entropy = lambda self: temp2(self).sum(-1)
MultiVariateNormal.mode = lambda self: self.mean

use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
Tensor = FloatTensor

def weights_init(m):
	classname =	m.__class__.__name__
	if classname.find('Linear') != -1:
		torch.nn.init.xavier_uniform_(m.weight)
		m.bias.data.zero_()

class ActorCriticNN(nn.Module):
	def __init__(self, num_states, num_actions, log_std = 0.0):
		super(ActorCriticNN, self).__init__()
		self.num_policyInput = num_states

		self.hidden_size = 128
		self.num_layers = 1

		num_h1 = 512
		num_h2 = 256
		num_h3 = 256

		self.policy = nn.Sequential(
			nn.Linear(self.num_policyInput, num_h1),
			nn.LeakyReLU(0.2, inplace=True),
			nn.Linear(num_h1, num_h2),
			nn.LeakyReLU(0.2, inplace=True),
			nn.Linear(num_h2, num_h3),
			nn.LeakyReLU(0.2, inplace=True),
			nn.Linear(num_h3, num_actions),

		)

		self.value = nn.Sequential(
			nn.Linear(self.num_policyInput, num_h1),
			nn.LeakyReLU(0.2, inplace=True),
			nn.Linear(num_h1, num_h2),
			nn.LeakyReLU(0.2, inplace=True),
			nn.Linear(num_h2, num_h3),
			nn.LeakyReLU(0.2, inplace=True),
			nn.Linear(num_h3, 1),
			# nn.LeakyReLU(0.2, inplace=True),
			# nn.Linear(num_h3, 1)
		)
		# embed()
		# exit(0)

		self.log_std = nn.Parameter(log_std * torch.ones(num_actions))

		self.policy.apply(weights_init)
		self.value.apply(weights_init)



	def forward(self,x):
		x = x.cuda()
		# embed()
		# exit(0)

		action = self.policy(x)
		return MultiVariateNormal(action.unsqueeze(0),self.log_std.exp()), self.value(x)


	def load(self,path):
		print('load nn {}'.format(path))
		self.load_state_dict(torch.load(path))
		print('load sucess')

	def save(self,path):
		print('save nn {}'.format(path))
		torch.save(self.state_dict(),path)
