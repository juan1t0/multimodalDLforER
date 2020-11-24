import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils.componets import WeightedSum, EmbraceNet

import numpy as np

class ModelOne(nn.Module):
	'''
		Eembrace net and ocasionaly linear layer
	'''
	def __init__(self, num_classes, input_sizes, embrace_size, docker_architecture, finalouts, 
								use_ll, ll_config, trainable_probs):
		super(ModelOne, self).__init__()
		self.NClasses =  num_classes
		self.InputSize = len(input_sizes)
		self.EmbNet = EmbraceNet(input_sizes, embrace_size, docker_architecture)
		self.FinalOut = finalouts
		self.UseLL = use_ll
		self.TrainableProbs = trainable_probs
		self.initProbabilities()
		if use_ll or num_classes != embrace_size:
			self.UseLL = True
			self.LL = self.gen_ll(ll_config, embrace_size)

	def gen_ll(self, config ,embrace_size):
		layers = []
		inC = embrace_size
		for x in config:
			if x == 'D':
				layers += [nn.Dropout()]
			elif x == 'R':
				layers += [nn.ReLU()]
			else:
				layers += [nn.Linear(inC, x)]
				inC = x

		return nn.Sequential(*layers)
	
	def initProbabilities(self):
		p = torch.ones(1, self.InputSize, dtype=torch.float)
		self.p = torch.div(p, torch.sum(p, dim=-1, keepdim=True))

		self.P = nn.Parameter(self.p, requires_grad=self.TrainableProbs)

	def forward(self, outputs1, outputs2, available):
		# batch_size = outputs[0][0].shape[0]
		availabilities = torch.ones(1 , self.InputSize, dtype=torch.float) # len(outputs)
		for i, av in enumerate(available):
			if av == 0.0:
				availabilities[0,i] == 0.0 ## review if it works
		
		# probabilities = torch.stack([self.p]*batch_size, dim=-1)
		probabilities = self.p
		if self.FinalOut:
			out = self.EmbNet.forward(outputs2, availabilities, probabilities)
		else:
			out = self.EmbNet.forward(outputs1, availabilities, probabilities)
		if self.UseLL:
			outl = self.LL(out)
			return outl, out
		return out, None

class ModelTwo(nn.Module):
	'''
		Two embraces, one after another
	'''
	def __init__(self, num_classes, input_sizes, embrace1_param, embrace2_param,
								trainable_probs, use_ll1, use_ll2, ll_config1={}, ll_config2={}):
		super(ModelTwo, self).__init__()
		self.NClasses =  num_classes
		self.InputSize = input_sizes
		self.EmbNet1 = EmbraceNet(**embrace1_param)
		self.EmbNet2 = EmbraceNet(**embrace2_param)
		self.UseLL1 = use_ll1
		self.UseLL2 = use_ll2
		self.TrainableProbs = trainable_probs
		self.initProbabilities()
		if use_ll1:
			self.LL1 = self.gen_ll(**ll_config1)
		if use_ll2:
			self.LL2 = self.gen_ll(**ll_config2)

	def gen_ll(self, config ,embrace_size):
		layers = []
		inC = embrace_size
		for x in config:
			if x == 'D':
				layers += [nn.Dropout()]
			elif x == 'R':
				layers += [nn.ReLU()]
			else:
				layers += [nn.Linear(inC, x)]
				inC = x

		return nn.Sequential(*layers)
	
	def initProbabilities(self):
		p1 = torch.ones(1, self.InputSize, dtype=torch.float)
		p2 = torch.ones(1, self.InputSize+1, dtype=torch.float)
		self.p1 = torch.div(p1, torch.sum(p1, dim=-1, keepdim=True))
		self.p2 = torch.div(p2, torch.sum(p2, dim=-1, keepdim=True))

		self.P1 = nn.Parameter(self.p1, requires_grad=self.TrainableProbs)
		self.P2 = nn.Parameter(self.p2, requires_grad=self.TrainableProbs)

	def forward(self, outputs1, outputs2, available):
		# batch_size = outputs[0][0].shape[0]
		availabilities = torch.ones(1 , self.InputSize+1, dtype=torch.float) # len(outputs)
		for i, av in enumerate(available):
			if av == 0.0:
				availabilities[0,i] == 0.0 ## review if it works
		
		# probabilities = torch.stack([self.p]*batch_size, dim=-1)
		probabilities1 = self.p1
		out1 = self.EmbNet1.forward(outputs1, availabilities[:,:-1], probabilities1)#availabilities without last column
		if self.UseLL1:
			out1 = self.LL1(out1)
		
		# probabilities = torch.stack([self.p]*batch_size, dim=-1)
		probabilities2 = self.p2
		out = self.EmbNet2.forward(outputs2+[out1], availabilities, probabilities2)
		if self.UseLL2:
			out = self.LL2(out)
		return out, out1

class ModelThree(nn.Module):
	'''
		Two embraces plus weigthed sum
	'''
	def __init__(self, num_classes, input_sizes, embrace1_param, embrace2_param, wsum_confg,
								trainable_probs, use_ll1, use_ll2, ll_config1={}, ll_config2={}):
		super(ModelThree, self).__init__()
		self.NClasses =  num_classes
		self.InputSize = input_sizes
		self.EmbNet1 = EmbraceNet(**embrace1_param)
		self.EmbNet2 = EmbraceNet(**embrace2_param)
		self.WeightedSum = WeightedSum(**wsum_confg)
		self.UseLL1 = use_ll1
		self.UseLL2 = use_ll2
		self.TrainableProbs = trainable_probs
		self.initProbabilities()
		if use_ll1:
			self.LL1 = self.gen_ll(**ll_config1)
		if use_ll2:
			self.LL2 = self.gen_ll(**ll_config2)

	def gen_ll(self, config ,embrace_size):
		layers = []
		inC = embrace_size
		for x in config:
			if x == 'D':
				layers += [nn.Dropout()]
			elif x == 'R':
				layers += [nn.ReLU()]
			else:
				layers += [nn.Linear(inC, x)]
				inC = x

		return nn.Sequential(*layers)
	
	def initProbabilities(self):
		p1 = torch.ones(1, self.InputSize, dtype=torch.float)
		p2 = torch.ones(1, self.InputSize+2, dtype=torch.float)
		self.p1 = torch.div(p1, torch.sum(p1, dim=-1, keepdim=True))
		self.p2 = torch.div(p2, torch.sum(p2, dim=-1, keepdim=True))

		self.P1 = nn.Parameter(self.p1, requires_grad=self.TrainableProbs)
		self.P2 = nn.Parameter(self.p2, requires_grad=self.TrainableProbs)

	def forward(self, outputs1, outputs2, available):
		# batch_size = outputs[0][0].shape[0]
		availabilities = torch.ones(1 , self.InputSize+2, dtype=torch.float) # len(outputs)
		for i, av in enumerate(available):
			if av == 0.0:
				availabilities[0,i] == 0.0 ## review if it works
		
		# probabilities = torch.stack([self.p]*batch_size, dim=-1)
		probabilities1 = self.p1
		out1 = self.EmbNet1.forward(outputs1, availabilities[:,:-2], probabilities1)#availabilities without last column
		if self.UseLL1:
			out1 = self.LL1(out1)
		
		wsout = self.WeightedSum.forward(outputs2,availabilities[:,:-2])
		
		# probabilities = torch.stack([self.p]*batch_size, dim=-1)
		probabilities2 = self.p2
		out = self.EmbNet2.forward(outputs2+[out1,wsout], availabilities, probabilities2)
		if self.UseLL2:
			out = self.LL2(out)
		return out, (out1, wsout)

class ModelFour(nn.Module):
	'''
		Three embraces, two in branches and one for merge all
	'''
	def __init__(self, num_classes, input_sizes, embrace1_param, embrace2_param, embrace3_param, wsum_confg,
								trainable_probs, use_ll, ll_configs):
		super(ModelFour, self).__init__()
		self.NClasses =  num_classes
		self.InputSize = input_sizes
		self.EmbNet1 = EmbraceNet(**embrace1_param)
		self.EmbNet2 = EmbraceNet(**embrace2_param)
		self.EmbNet3 = EmbraceNet(**embrace3_param)
		self.WeightedSum = WeightedSum(**wsum_confg)
		self.UseLL1 = use_ll[0]
		self.UseLL2 = use_ll[1]
		self.UseLL3 = use_ll[2]
		self.TrainableProbs = trainable_probs
		self.initProbabilities()
		if self.UseLL1:
			self.LL1 = self.gen_ll(**ll_configs[0])
		if self.UseLL2:
			self.LL2 = self.gen_ll(**ll_configs[1])
		if self.UseLL3:
			self.LL3 = self.gen_ll(**ll_configs[2])

	def gen_ll(self, config ,embrace_size):
		layers = []
		inC = embrace_size
		for x in config:
			if x == 'D':
				layers += [nn.Dropout()]
			elif x == 'R':
				layers += [nn.ReLU()]
			else:
				layers += [nn.Linear(inC, x)]
				inC = x

		return nn.Sequential(*layers)
	
	def initProbabilities(self):
		p1 = torch.ones(1, self.InputSize, dtype=torch.float)
		p2 = torch.ones(1, self.InputSize+2, dtype=torch.float)
		p3 = torch.ones(1, self.InputSize+2, dtype=torch.float)
		self.p1 = torch.div(p1, torch.sum(p1, dim=-1, keepdim=True))
		self.p2 = torch.div(p2, torch.sum(p2, dim=-1, keepdim=True))
		self.p3 = torch.div(p3, torch.sum(p3, dim=-1, keepdim=True))

		self.P1 = nn.Parameter(p1,requires_grad=self.TrainableProbs)
		self.P2 = nn.Parameter(p2,requires_grad=self.TrainableProbs)
		self.P3 = nn.Parameter(p3, requires_grad=self.TrainableProbs)

	def forward(self, outputs1, outputs2, available):
		# batch_size = outputs[0][0].shape[0]
		availabilities = torch.ones(1 , self.InputSize+3, dtype=torch.float) # len(outputs)
		for i, av in enumerate(available):
			if av == 0.0:
				availabilities[0,i] == 0.0 ## review if it works
		
		# probabilities = torch.stack([self.p]*batch_size, dim=-1)
		probabilities1 = self.p1
		out1 = self.EmbNet1.forward(outputs1, availabilities[:,:-3], probabilities1)#availabilities without last column
		if self.UseLL1:
			out1 = self.LL1(out1)
		# probabilities = torch.stack([self.p]*batch_size, dim=-1)
		probabilities2 = self.p2
		out2 = self.EmbNet2.forward(outputs1, availabilities[:,:-3], probabilities2)#availabilities without last column
		if self.UseLL2:
			out2 = self.LL2(out2)

		wsout = self.WeightedSum.forward(outputs2,availabilities[:,:-3])
		
		# probabilities = torch.stack([self.p]*batch_size, dim=-1)
		probabilities3 = self.p3
		out = self.EmbNet3.forward(outputs2+[out1,out2,wsout], availabilities, probabilities3)
		if self.UseLL3:
			out = self.LL3(out)
		return out, (out1, out2, wsout)

class MergeClass():
	'''
		This is a wrapper class of the real merge trainable class
	'''
	def __init__(self, models={}, config={}, # mode='train', # review if dictionay is better than list <memory, vel,etc>
								labels={}, self_embeding=False,debug_mode=False):
		'''
			- models				: dictionary with the d-models already loaded
			- config				: dictionary whit the parameters for define the merge module
			- mode					: using mode, can be 'train', 'test' or 'predic'
			- dataset				: Dataset already loaded with all data
			- labels				: dictionary with the labels used
			- self_embeding	: boolean, if true the embeding heuristic is used
			- collate				: Collate that helps to deal with missing data
			- debug_mode		: bolean, if true show several final and middle information
		'''
		self.Modalities = models
		self.MergeConfig = config
		self.MergeModel = self.get_model(self.MergeConfig)
		# self.Mode = mode
		self.Classes = labels
		self.SelfEmbeddin = self_embeding

	def get_model(self, config):
		'''
			config has 'type' and 'parameters' as keys
			- type				: specify the model type
			- parameters	: dictionary with the parameters for instantiate the model
		'''
		type = config['type']
		# self.build_transform(type)
		if type == 1:
			return ModelOne(**config['parameters'])
		elif type == 2:
			return ModelTwo(**config['parameters'])
		if type == 3:
			return ModelThree(**config['parameters'])
		elif type == 4:
			return ModelFour(**config['parameters'])
		else:
			raise NameError('type {} is not supported yet'.format(type))
	
	def parameters(self):
		'''
			return the parameters of merge model
		'''
		return self.MergeModel.parameters()

	def forward(self, data):
		'''
		'''
		assert data['context'].shape[0] == 1 # just with batch size of 1
		availables = [1.0] *4
		fb,_,mb = self.Modalities['body'].forward(data['body'])
		fc,_,mc = self.Modalities['context'].forward(data['context'])
		middle_out = [mb[3], mc[3]]
		final_out = [fb, fc]
		if data['face'].sum().item() != 0.0:
			ff, mf = self.Modalities['face'].forward(data['face'])
			middle_out += [mf]
			final_out += [ff]
		else:
			availables[2] = 0.0
			middle_out += [mc[3]]
			final_out += [fc]
		if data['joint'].sum().item() != 0.0:
			fs, ms = self.Modalities['pose'].forward((data['joint'],data['bone']),0)
			ms = torch.cat((ms[0], ms[1]), dim=-1)
			middle_out += [ms]
			final_out += [fs]
		else:
			availables[3] = 0.0
			middle_out += [mc[3]]
			final_out += [fc]

		out, middle = self.MergeModel.forward(middle_out, final_out, availables)

		return out, middle

	def predict(self, data): #wait
		'''
		'''
