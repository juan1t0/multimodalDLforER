# https://github.com/kenziyuliu/Unofficial-DGNN-PyTorch

import os
import time
import numpy as np
import yaml
import pickle
from collections import OrderedDict, defaultdict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
# from tqdm import tqdm
from tensorboardX import SummaryWriter
import shutil
from torch.optim.lr_scheduler import MultiStepLR
import random
import inspect
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

from utils.dataset import Emotic_MDB
from models.scofmer import Model as SCOF_model

'''
	eliminar lo innecesario
	revisar opt, train, start
'''

def init_seed(_):
	torch.cuda.manual_seed_all(1)
	torch.manual_seed(1)
	np.random.seed(1)
	random.seed(1)
	# torch.backends.cudnn.enabled = False
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False

class Processor():
	"""Processor for Skeleton-based Action Recgnition"""
	def __init__(self, config):
		self.Configuration = config
		self.save_config()

		self.global_step = 0
		self.load_model()
		self.load_param_groups()    # Group parameters to apply different learning rules
		self.load_optimizer()
		self.load_data()
		self.lr = config['base_lr']
		self.best_acc = 0
		self.best_acc_epoch = 0

	def load_data(self):
		dict = self.Configuration['dataset']
		self.data_loader = dict()

		if dict['mode'] == 'train':
			dataset = Emotic_MDB(root_dir=dict['root_dir'],
													 annotation_dir=dict['annotation_dir'],
													 mode=dict['mode'],
													 modals_names=dict['modals_names'],
													 categories=dict['categories'],
													 transform=dict['transform'])
			self.data_loader['train'] = DataLoader(dataset=dataset,
																						 batch_size=dict['batch_size'],
																						 shuffle=True,
																						 num_workers=dict['num_worker'],
																						 drop_last=True,
																						 worker_init_fn=init_seed)
			
			dataset = Emotic_MDB(root_dir=dict['root_dir'],
													 annotation_dir=dict['annotation_dir'],
													 mode= 'val',
													 modals_names=dict['modals_names'],
													 categories=dict['categories'],
													 transform=dict['transform'])
			self.data_loader['val'] = DataLoader(dataset=dataset,
																						 batch_size=dict['batch_size'],
																						 shuffle=True,
																						 num_workers=dict['num_worker'],
																						 drop_last=True,
																						 worker_init_fn=init_seed)
			
		dataset = Emotic_MDB(root_dir=dict['root_dir'],
													 annotation_dir=dict['annotation_dir'],
													 mode='test',
													 modals_names=dict['modals_names'],
													 categories=dict['categories'],
													 transform=dict['transform'])
		self.data_loader['test'] = DataLoader(dataset=dataset,
																					batch_size=dict['batch_size'],
																					shuffle=False,
																					num_workers=dict['num_worker'],
																					drop_last=False,
																					worker_init_fn=init_seed)

	def load_model(self):
		dict = self.Configuration['model']
		self.output_device = dict['device']
	
		Model = SCOF_model(n_classes=dict['n_classes'],
											 inner_models_config=dict['inner_models_config'],
											 beta=dict['beta'],
											 weights=dict['weights'])

		self.Model = Model.cuda(self.output_device)
		self.loss = nn.CrossEntropyLoss().cuda(self.output_device) ###### review

		# Load weights
		if dict['checkpoint']:
			self.global_step = int(dict['global_step'])  #int(arg.weights[:-3].split('-')[-1])
			
			weights = torch.load(dict['weights_dir'])
			# weights = OrderedDict([[k.split('module.')[-1],
			# 											v.cuda(output_device)] for k, v in weights.items()])
			#
			# for w in dict['ignore_weights']:
			# 	if weights.pop(w, None) is not None:
			# 		self.print_log('Sucessfully Remove Weights: {}.'.format(w))
			# 	else:
			# 		self.print_log('Can Not Remove Weights: {}.'.format(w))
			try:
				self.model.load_state_dict(weights)
			except:
				state = self.model.state_dict()
				diff = list(set(state.keys()).difference(set(weights.keys())))
				print('Can not find these weights:')
				for d in diff:
					print('  ' + d)
				state.update(weights)
				self.model.load_state_dict(state)

		# Parallelise data if mulitple GPUs
		# if type(self.output_device) is list and len(self.output_device) > 1:
		# 	self.model = nn.DataParallel(self.Model,
		# 															 device_ids=self.output_device,
		# 															 output_device=output_device)

	### no yet
	def load_optimizer(self):
			p_groups = list(self.optim_param_groups.values())
			if self.arg.optimizer == 'SGD':
					self.optimizer = optim.SGD(
							p_groups,
							lr=self.arg.base_lr,
							momentum=0.9,
							nesterov=self.arg.nesterov,
							weight_decay=self.arg.weight_decay)
			elif self.arg.optimizer == 'Adam':
					self.optimizer = optim.Adam(
							p_groups,
							lr=self.arg.base_lr,
							weight_decay=self.arg.weight_decay)
			else:
					raise ValueError('Unsupported optimizer: {}'.format(self.arg.optimizer))

			self.lr_scheduler = MultiStepLR(self.optimizer, milestones=self.arg.step, gamma=0.1)

	def save_config(self):
			# save arg
			dict = vars(self.Configuration)
			if not os.path.exists(dict['work_dir']):
					os.makedirs(dict['work_dir'])
			with open('{}/config.yaml'.format(dict['work_dir']), 'w') as f:
					yaml.dump(dict, f)

	# def print_time(self):
	# 		localtime = time.asctime(time.localtime(time.time()))
	# 		self.print_log("Local current time :  " + localtime)

	def record_time(self):
			self.cur_time = time.time()
			return self.cur_time

	def split_time(self):
			split_time = time.time() - self.cur_time
			self.record_time()
			return split_time

	def train(self, epoch, save_model=False):
		print('Training epoch: {}'.format(epoch + 1))
		self.Model.train()
		loader = self.data_loader['train']
		loss_values, acc_values = [], []

		for batch_idx, batch_sample in enumerate(loader):
			self.global_step += 1
			# get data
			with torch.no_grad():
				joint_data = joint_data.float().cuda(self.output_device)
				bone_data = bone_data.float().cuda(self.output_device)
				label = label.long().cuda(self.output_device)
			# timer['dataloader'] += self.split_time()

			self.optimizer.zero_grad()
			output = self.model(batch_sample)
			loss = self.loss(output, batch_sample['label'])

			loss.backward()
			self.optimizer.step()

			loss_values.append(loss.item())

			value, predict_label = torch.max(output, 1)
			acc_values.append(torch.mean((predict_label == label).float()))

			self.optimizer.step()

			# statistics
			self.lr = self.optimizer.param_groups[0]['lr']
			# self.train_writer.add_scalar('lr', self.lr, self.global_step)
			# timer['statistics'] += self.split_time()

		print ('\tMean training loss: {:.4f} ; epoch {}'.format(np.mean(loss_values),epoch))
		print ('\tMean training acc: {:.4f} ; epoch {}'.format(np.mean(acc_values), epoch))

		self.lr_scheduler.step(epoch)

		if save_model:
			weights = self.model.state_dict()
			# weights = OrderedDict([[k.split('module.')[-1], v.cpu()] for k, v in weights.items()])
			self.model_saved_name = self.Configuration['model_saved_name'] +'_'+ str(epoch) +'_'+ str(int(self.global_step)) + '.pt'
			torch.save(weights, self.model_saved_name)

	def eval(self, epoch, save_score=False, loader_name=['test']):
		self.Model.eval()

		for ln in loader_name:
			loss_values, score_batches = [], []
			step = 0
			# process = tqdm(self.data_loader[ln])
			for batch_idx, (joint_data, bone_data, label, index) in enumerate(data_loader[ln]):
				step += 1
				with torch.no_grad():
					joint_data = joint_data.float().cuda(self.output_device)
					bone_data = bone_data.float().cuda(self.output_device)
					label = label.long().cuda(self.output_device)
					output = self.model(joint_data, bone_data)

					loss = self.loss(output, label)
					score_batches.append(output.cpu().numpy())
					loss_values.append(loss.item())
					# Argmax over logits = labels
					_, predict_label = torch.max(output, dim=1)

			# Concatenate along the batch dimension, and 1st dim ~= `len(dataset)`
			score = np.concatenate(score_batches)
			loss = np.mean(loss_values)
			accuracy = self.data_loader[ln].dataset.top_k(score, 1)
			if accuracy > self.best_acc:
				self.best_acc = accuracy
				self.best_acc_epoch = epoch

			print('Accuracy: ', accuracy, ' Model: ', self.arg.model_saved_name)

			if save_score:
				score_dict = dict(zip(self.data_loader[ln].dataset.sample_name, score))
				with open('{}/epoch{}_{}_score.pkl'.format(self.arg.work_dir, epoch + 1, ln), 'wb') as f:
					pickle.dump(score_dict, f)

	def start(self):
		
		if self.Configuration['mode'] == 'train':
			dict = self.Configuration['train']
			self.global_step = dict['start_epoch'] * len(self.data_loader['train']) / dict['batch_size']

			for epoch in range(dict['start_epoch'], dict['num_epoch']):
				if self.lr < 1e-3:
					break
				save_model = ((epoch + 1) % dict['save_interval'] == 0) or (epoch + 1 == dict['num_epoch'])

				self.train(epoch, save_model=save_model)
				self.eval(epoch, save_score=dict['save_score'], loader_name=['val'])

			print('Best accuracy: {}, epoch: {}, model_name: {}'\
				.format(self.best_acc, self.best_acc_epoch, self.model_saved_name))

		elif self.Configuration['mode'] == 'test':
			dict = self.Configuration['test']
			if dict['weights_dir'] is None:
				raise ValueError('Please provide weights')

			self.eval(epoch=0, save_score=dict['save_score'], loader_name=['test'])
			

def str2bool(v):
	if v.lower() in ('yes', 'true', 't', 'y', '1'):
		return True
	elif v.lower() in ('no', 'false', 'f', 'n', '0'):
		return False
	else:
		raise Exception('Boolean value expected.')

# def import_class(name):
# 	components = name.split('.')
# 	mod = __import__(components[0])  # import return model
# 	for comp in components[1:]:
# 			mod = getattr(mod, comp)
# 	return mod

# if p.config is not None:
#   with open(p.config, 'r') as f:
#     default_arg = yaml.load(f)
#   key = vars(p).keys()
#   for k in default_arg.keys():
#     if k not in key:
#       print('WRONG ARG: {}'.format(k))
#       assert (k in key)
#   parser.set_defaults(**default_arg)

# arg = parser.parse_args()
# init_seed(0)
# processor = Processor(arg)
# processor.start()