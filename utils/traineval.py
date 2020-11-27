import os
import numpy as np
from time import sleep

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from sklearn.metrics import average_precision_score, precision_recall_curve

class DiscreteLoss(nn.Module):
	def __init__(self, weight_type='mean', device=torch.device('cpu'), classes=26):
		super(DiscreteLoss, self).__init__()
		self.weight_type = weight_type
		self.device = device
		self.Classes = classes
		if self.weight_type == 'mean':
			self.weights = torch.ones((1,classes))/float(classes)
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
		weights = torch.zeros((1,self.Classes))
		weights[target_stats != 0 ] = 1.0/torch.log(target_stats[target_stats != 0].data + 1.2)
		weights[target_stats == 0] = 0.0001
		return weights

def savemodel(epoch, model_dict, opt_dict, losstrain, acctrain, lossval, accval,
							save_dir, modelname, save_name):
	model_saved_name = os.path.join(save_dir,modelname + save_name +'.pth')
	torch.save({'epoch':epoch,
							'train_loss':losstrain,
							'train_acc':acctrain,
							'val_loss':lossval,
							'val_acc':accval,
							'model_state_dict':model_dict,
							'optimizer_state_dict':opt_dict},
						 model_saved_name)
	print('Model {} saved'.format(model_saved_name))

def test_AP(cat_preds, cat_labels, n_classes=8):
	n_classes=cat_labels.shape[0]
	ap = np.zeros(n_classes, dtype=np.float32)
	for i in range(n_classes):
		ap[i] = average_precision_score(cat_labels[i, :], cat_preds[i, :])
	ap[np.isnan(ap)] = 0.0
	print ('AveragePrecision: {} |{}| mAP: {}'.format(ap, ap.shape[0], ap.mean()))
	return ap.mean()

# modal could be: all, pose , context, body, face
def train(Model, train_dataset, Loss, optimizer, val_dataset, bsz=32,
					collate=None, train_sampler=None, val_sampler=None, epoch=0,
					modal='all', device=torch.device('cpu'), debug_mode=False, tqdm=None):
	Model.train()
	if collate is not None:
		loader = tqdm(DataLoader(train_dataset, batch_size=bsz, num_workers=0, sampler=train_sampler, collate_fn=collate),
									unit='batch')
	else:
		loader = tqdm(DataLoader(train_dataset, batch_size=bsz, num_workers=0, sampler=train_sampler,),
									unit='batch')
	loader.set_description("{} Epoch {}".format(train_dataset.Mode, epoch + 1))
	loss_values = []
	predictions, labeles = [], []
	for batch_idx, batch_sample in enumerate(loader):
		with torch.no_grad():
			if modal == 'all':
				sample = dict()
				sample['context'] = batch_sample['context'].to(device)#.float().permute(0,3,1,2).to(device)
				sample['body'] = batch_sample['body'].to(device)#.float().permute(0,3,1,2).to(device)
				sample['face'] = batch_sample['face'].to(device)#.float().permute(0,3,1,2).to(device)
				sample['joint'] = batch_sample['joint'].to(device)#.float().to(device)
				sample['bone'] = batch_sample['bone'].to(device)#.float().to(device)
			elif modal == 'pose':
				sample = (batch_sample['joint'].to(device),#.float().to(device),
									batch_sample['bone'].to(device))#.float().to(device))
			else:
				sample = batch_sample[modal].to(device)#.float().permute(0,3,1,2).to(device)
			
			label = batch_sample['label'].to(device)#.float().to(device)

		optimizer.zero_grad()
		if modal =='pose':
			output, _ = Model.forward(sample, 0)
			predictions += [output[i].to('cpu').data.numpy() for i in range(output.shape[0])]
			loss = Loss(output, label)
		elif modal == 'face':
			output, _ = Model.forward(sample)
			predictions += [output[i].to('cpu').data.numpy() for i in range(output.shape[0])]
			loss = Loss(output, label)
		elif modal == 'body' or modal == 'context':
			per_outs, att_outs, _ = Model.forward(sample)
			predictions += [per_outs[i].to('cpu').data.numpy() for i in range(per_outs.shape[0])]
			loss = (Loss(att_outs, label)) + (Loss(per_outs, label))
		elif modal == 'all':
			output, _ = Model.forward(sample)
			predictions += [output[i].to('cpu').data.numpy() for i in range(output.shape[0])]
			loss = Loss(output, label)

		loss.backward()
		optimizer.step()

		labeles += [label[i].to('cpu').data.numpy() for i in range(label.shape[0])]
		
		loss_values.append(loss.item())
		loader.set_postfix(loss=loss.item())
		sleep(0.1)
	
	train_gloss = np.mean(loss_values)
	train_mAP = test_AP(np.asarray(predictions).T, np.asarray(labeles).T)#, n_classes=nclasses)

	if collate is not None:
		loader = tqdm(DataLoader(val_dataset, batch_size=bsz, num_workers=0, sampler=val_sampler, collate_fn=collate),
									unit='batch')
	else:
		loader = tqdm(DataLoader(val_dataset, batch_size=bsz, num_workers=0, sampler=val_sampler,),
									unit='batch')
	loader.set_description("{} Epoch {}".format(val_dataset.Mode, epoch + 1))
	loss_values = []
	predictions, labeles = [], []
	
	Model.eval()
	with torch.no_grad():
		for batch_idx, batch_sample in enumerate(loader):
			if modal == 'all':
				sample = dict()
				sample['context'] = batch_sample['context'].to(device)#.float().permute(0,3,1,2).to(device)
				sample['body'] = batch_sample['body'].to(device)#.float().permute(0,3,1,2).to(device)
				sample['face'] = batch_sample['face'].to(device)#.float().permute(0,3,1,2).to(device)
				sample['joint'] = batch_sample['joint'].to(device)#.float().to(device)
				sample['bone'] = batch_sample['bone'].to(device)#.float().to(device)
			elif modal == 'pose':
				sample = (batch_sample['joint'].to(device),#.float().to(device),
									batch_sample['bone'].to(device))#.float().to(device))
			else:
				sample = batch_sample[modal].to(device)#.float().permute(0,3,1,2).to(device)

			label = batch_sample['label'].to(device)#.float().to(device)

			if modal =='pose':
				output, _ = Model.forward(sample, 0)
				predictions += [output[i].to('cpu').data.numpy() for i in range(output.shape[0])]
				loss = Loss(output, label)
			elif modal == 'face':
				output, _ = Model.forward(sample)
				predictions += [output[i].to('cpu').data.numpy() for i in range(output.shape[0])]
				loss = Loss(output, label)
			elif modal == 'body' or modal == 'context':
				per_outs, att_outs, _ = Model.forward(sample)
				predictions += [per_outs[i].to('cpu').data.numpy() for i in range(per_outs.shape[0])]
				loss = (Loss(att_outs, label)) + (Loss(per_outs, label))
			elif modal == 'all':
				output, _ = Model.forward(sample)
				predictions += [output[i].to('cpu').data.numpy() for i in range(output.shape[0])]
				loss = Loss(output, label)
			labeles += [label[i].to('cpu').data.numpy() for i in range(label.shape[0])]
			
			loss_values.append(loss.item())
			loader.set_postfix(loss=loss.item())
			sleep(0.1)
	val_gloss = np.mean(loss_values)
	val_mAP = test_AP(np.asarray(predictions).T, np.asarray(labeles).T)#, n_classes=nclasses)

	if debug_mode:
		print ('- Mean training loss: {:.4f} ; epoch {}'.format(train_gloss, epoch+1))
		print ('- Mean validation loss: {:.4f} ; epoch {}'.format(val_gloss, epoch+1))
		print ('- Mean training mAP: {:.4f} ; epoch {}'.format(train_mAP, epoch+1))
		print ('- Mean validation mAP: {:.4f} ; epoch {}'.format(val_mAP, epoch+1))
	return train_gloss, train_mAP, val_gloss, val_mAP

def eval(Model, dataset, bsz=32, test_sampler=None, collate=None, epoch=0, modal='all',
				 device=torch.device('cpu'), debug_mode=False, tqdm=None):
	Model.eval()
	if collate is not None:
		loader = tqdm(DataLoader(dataset, batch_size=bsz, num_workers=0, sampler=test_sampler, collate_fn=collate),
									unit='batch')
	else:
		loader = tqdm(DataLoader(dataset, batch_size=bsz, num_workers=0, sampler=test_sampler),
									unit='batch')
	loader.set_description("{} Epoch {}".format(dataset.Mode, epoch + 1))
	predictions, labeles = [], []
	for batch_idx, batch_sample in enumerate(loader):
		sample = dict()
		with torch.no_grad():
			if modal == 'all':
				sample = dict()
				sample['context'] = batch_sample['context'].to(device)#.float().permute(0,3,1,2).to(device)
				sample['body'] = batch_sample['body'].to(device)#.float().permute(0,3,1,2).to(device)
				sample['face'] = batch_sample['face'].to(device)#.float().permute(0,3,1,2).to(device)
				sample['joint'] = batch_sample['joint'].to(device)#.float().to(device)
				sample['bone'] = batch_sample['bone'].to(device)#.float().to(device)
				output, _ = Model.forward(sample)
				predictions += [output[i].to('cpu').data.numpy() for i in range(output.shape[0])]
			elif modal == 'pose':
				sample = (batch_sample['joint'].to(device),#.float().to(device),
									batch_sample['bone'].to(device))#.float().to(device))
				output, _ = Model.forward(sample,0)
				predictions += [output[i].to('cpu').data.numpy() for i in range(output.shape[0])]
			else:
				sample = batch_sample[modal].to(device)#.float().permute(0,3,1,2).to(device)
				if modal == 'face':
					output, _ = Model.forward(sample)
				else: # context or body
					output, _, _ = Model.forward(sample)
				predictions += [output[i].to('cpu').data.numpy() for i in range(output.shape[0])]
			
			label = batch_sample['label']# .float() # .to(device)
		
		labeles += [label[i].data.numpy() for i in range(label.shape[0])]
		# predictions[(batch_idx*32):(batch_idx*32 +32),:] = output.to('cpu').data.numpy()
		# labels[(batch_idx*32):(batch_idx*32 +32),:] = label.data.numpy()
		# value, predict_label = torch.max(output, 1)
		# acc_values.append(torch.mean((predict_label == label).float()))

	# predictions = np.asarray(predictions).T
	# labels = np.asarray(labels).T
	mAP = test_AP(np.asarray(predictions).T, np.asarray(labeles).T)#, n_classes=nclasses)
	return mAP

def train_step(Model, dataset_t, dataset_v, bsz, Loss, optimizer, collate, epoch,
								tsampler, vsampler,
								last_epoch, modal, device, debug_mode, tqdm, train_loss, train_map,
								val_loss, val_map, maxacc, step2val, step2save, checkpointdir, model_name):
	
	tl, ta, vl, va = train(Model=Model, train_dataset=dataset_t, Loss=Loss, optimizer=optimizer,
													val_dataset=dataset_v , bsz=bsz, collate=collate, train_sampler=tsampler,
													val_sampler=vsampler,epoch=epoch, modal=modal,
													device=device, debug_mode=debug_mode, tqdm=tqdm)
	train_loss[epoch] = tl
	train_map[epoch] = ta
	val_loss[epoch] = vl
	val_map[epoch] = va
	
	if ta > maxacc:
		maxacc = ta
		savemodel(epoch=epoch,
							model_dict=Model.state_dict(),
							opt_dict=optimizer.state_dict(),
							losstrain=tl,	acctrain=ta,
							lossval=tl,	accval=ta,
							save_dir=checkpointdir, modelname=model_name, save_name='_best')
	# if (epoch+1) % step2val == 0:
	# 	l, a = train(Model=Model, dataset=dataset_v, Loss=Loss, optimizer=optimizer, bsz=bsz, collate=collate,
	# 								epoch=epoch, modal=modal, device=device, debug_mode=debug_mode, tqdm=tqdm)
	# 	val_loss[epoch:(epoch+step2val)] = [l]*step2val
	# 	val_map[epoch:(epoch+step2val)] = [a]*step2val
	if (epoch+1) % step2save == 0 or (epoch+1) == last_epoch:
		savemodel(epoch=epoch,
							model_dict=Model.state_dict(),
							opt_dict=optimizer.state_dict(),
							losstrain=tl,	acctrain=ta,
							lossval=tl,	accval=ta,
							save_dir=checkpointdir, modelname=model_name, save_name='_last')
	return maxacc