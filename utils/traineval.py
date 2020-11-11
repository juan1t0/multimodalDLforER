import os
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from sklearn.metrics import average_precision_score, precision_recall_curve

class DiscreteLoss(nn.Module):
	def __init__(self, weight_type='mean', device=torch.device('cpu')):
		super(DiscreteLoss, self).__init__()
		self.weight_type = weight_type
		self.device = device
		if self.weight_type == 'mean':
			self.weights = torch.ones((1,26))/26.0
			self.weights = self.weights.to(self.device)
		elif self.weight_type == 'static':
			self.weights = torch.FloatTensor([0.1435, 0.1870, 0.1692, 0.1165, 0.1949, 0.1204, 0.1728, 0.1372, 0.1620,
				0.1540, 0.1987, 0.1057, 0.1482, 0.1192, 0.1590, 0.1929, 0.1158, 0.1907,
				0.1345, 0.1307, 0.1665, 0.1698, 0.1797, 0.1657, 0.1520, 0.1537]).unsqueeze(0)
			self.weights = self.weights.to(self.device)
		
	def forward(self, pred, target):
		if self.weight_type == 'dynamic':
			self.weights = self.prepare_dynamic_weights(target)
			self.weights = self.weights.to(self.device)
		loss = (((pred - target)**2) * self.weights)
		return loss.sum() 

	def prepare_dynamic_weights(self, target):
		target_stats = torch.sum(target, dim=0).float().unsqueeze(dim=0).cpu()
		weights = torch.zeros((1,26))
		weights[target_stats != 0 ] = 1.0/torch.log(target_stats[target_stats != 0].data + 1.2)
		weights[target_stats == 0] = 0.0001
		return weights

def test_AP(cat_preds, cat_labels, n_classes=26):
	ap = np.zeros(n_classes, dtype=np.float32)
	for i in range(n_classes):
		ap[i] = average_precision_score(cat_labels[i, :], cat_preds[i, :])
	print ('AveragePrecision: {} |{}| mAP: {}'.format(ap, ap.shape[0], ap.mean()))
	return ap.mean()

# modal could be: all, pose , context, body, face
def train(Model, dataset, Loss, optimizer,
					epoch=0, model_saved_dir='checkpoints', save_model=False,
					modal='all', device=torch.device('cpu'), debug_mode=False):
	print('Training epoch: {}'.format(epoch + 1))
	Model.train()
	loader = DataLoader(dataset, batch_size=32, num_workers=2)
	loss_values= []
	
	for batch_idx, batch_sample in enumerate(loader):
		with torch.no_grad():
			if modal == 'all':
				sample = dict()
				sample['context'] = batch_sample['context'].float().permute(0,3,1,2).to(device)
				sample['body'] = batch_sample['body'].float().permute(0,3,1,2).to(device)
				sample['face'] = batch_sample['face'].float().permute(0,3,1,2).to(device)
				sample['joint'] = batch_sample['joint'].float().to(device)
				sample['bone'] = batch_sample['bone'].float().to(device)
			elif modal == 'pose':
				sample = (batch_sample['joint'].float().to(device),
									batch_sample['bone'].float().to(device))
			else:
				sample = batch_sample[modal].float().permute(0,3,1,2).to(device)
			
			label = batch_sample['label'].float().to(device)

		optimizer.zero_grad()
		if modal =='pose':
			output, _ = Model.forward(sample)
			loss = Loss(output, label)
		elif modal == 'body' or modal == 'context':
			att_outs, per_outs, _ = Model.forward(sample)
			loss = (Loss(att_outs, label)) + (Loss(per_outs, label))
		elif modal == 'all':
			output, _ = Model.forward(sample)
			loss = Loss(output, label)
		
		loss.backward()
		optimizer.step()

		loss_values.append(loss.item())
	
	gloss = np.mean(loss_values)
	
	if debug_mode:
		print ('\tMean training loss: {:.4f} ; epoch {}'.format(gloss, epoch+1))
		# print ('\tMean training acc: {:.4f} ; epoch {}'.format(np.mean(acc_values), epoch))
	if save_model:
		weights = Model.state_dict()
		# weights = OrderedDict([[k.split('module.')[-1], v.cpu()] for k, v in weights.items()])
		model_saved_name = os.path.join(model_saved_dir, Model.name +'_cp_'+ str(epoch+1) + '.pth')
		torch.save(weights, model_saved_name)
		print ('Model {} saved'.format(model_saved_name))
	
	return gloss

def eval(Model, dataset, epoch=0, model_saved_dir='checkpoints', save_model=False,
				 modal='all', device=torch.device('cpu'), debug_mode=False):
	print('Eval epoch: {}'.format(epoch + 1))
	Model.eval()
	loader = DataLoader(dataset, batch_size=32, num_workers=2)
	predictions, labels = np.zeros((len(dataset),26)), np.zeros((len(dataset),26))
	for batch_idx, batch_sample in enumerate(loader):
		sample = dict()
		with torch.no_grad():
			if modal == 'all':
				sample = dict()
				sample['context'] = batch_sample['context'].float().permute(0,3,1,2).to(device)
				sample['body'] = batch_sample['body'].float().permute(0,3,1,2).to(device)
				sample['face'] = batch_sample['face'].float().permute(0,3,1,2).to(device)
				sample['joint'] = batch_sample['joint'].float().to(device)
				sample['bone'] = batch_sample['bone'].float().to(device)
				output, _ = Model.forward(sample)
			elif modal == 'pose':
				sample = (batch_sample['joint'].float().to(device),
									batch_sample['bone'].float().to(device))
				output, _ = Model.forward(sample)
			else:
				sample = batch_sample[modal].float().permute(0,3,1,2).to(device)
				_, output, _ = Model.forward(sample)
			
			label = batch_sample['label'].float() # .to(device)
		
		predictions[(batch_idx*32):(batch_idx*32 +32),:] = output.to('cpu').data.numpy()
		labels[(batch_idx*32):(batch_idx*32 +32),:] = label.data.numpy()
		# value, predict_label = torch.max(output, 1)
		# acc_values.append(torch.mean((predict_label == label).float()))

	predictions = np.asarray(predictions).T
	labels = np.asarray(labels).T
	mAP = test_AP(predictions, labels)
	
	return mAP