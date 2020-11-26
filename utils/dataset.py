import os
import pandas as pd
import numpy as np
import cv2

import torch
from torch.utils.data import Dataset#, DataLoader
from torch.utils.data.dataloader import default_collate
from torchvision import transforms, utils

'''
	revisar todo / comentar
'''

def draw_points(image, points, color_palette='tab20', palette_samples=16):
	import matplotlib.pyplot as plt
	try:
		colors = np.round(np.array(plt.get_cmap(color_palette).colors) * 255
											).astype(np.uint8)[:, ::-1].tolist()
	except AttributeError:  # if palette has not pre-defined colors
		colors = np.round(np.array(plt.get_cmap(color_palette)(np.linspace(0, 1, palette_samples))) * 255
											).astype(np.uint8)[:, -2::-1].tolist()

	circle_size = max(1, min(image.shape[:2]) // 160)  # ToDo Shape it taking into account the size of the detection
	# circle_size = max(2, int(np.sqrt(np.max(np.max(points, axis=0) - np.min(points, axis=0)) // 16)))
	for i, pt in enumerate(points):
		if pt[2] > 0.01:
			image = cv2.circle(image, (int(pt[1]), int(pt[0])), circle_size, tuple(colors[i % len(colors)]), -1)
	return image

def my_collate(batch):
  batch = filter(lambda img: img is not None, batch)
  return default_collate(list(batch))

newlabeles = {'len':8,
              'cat':{'joy': ['Excitement', 'Happiness', 'Peace',
                             # 'Affection',
                             'Pleasure',],
                     'trust': ['Confidence', 'Esteem',
                               'Affection',],
                     'fear': ['Disquietment','Embarrassment','Fear',],
                     'surprice': ['Doubt/Confusion','Surprise',],
                     'sadness': ['Pain', 'Sadness', 'Sensitivity', 'Suffering',],
                     'disgust': ['Aversion','Disconnection', 'Fatigue','Yearning'],
                     'anger': ['Anger', 'Annoyance', 'Disapproval',],
                     'anticipation': ['Anticipation', 'Engagement', 'Sympathy',]
										 }
              }

class Emotic_MultiDB(Dataset):
	def __init__ (self, root_dir='Emotic_MDB', annotation_dir='annotations', mode='train',
								modality='all',
								#missin='',
								takeone=False, modals_dirs=[],
							 	categories=[], continuous=[], transform=None):

		super(Emotic_MultiDB, self).__init__()
		self.RootDir = root_dir
		self.Mode = mode
		self.AnnotationDir = os.path.join(root_dir, annotation_dir)
		self.Modality = modality
		# self.Missing = missin
		self.TakeOne = takeone
		self.Categories = categories
		self.Continuous = continuous
		self.Transform = transform
		self.Relabel = False
		self.Resize_Face = None
		self.loadData(modals_dirs)

	def loadData(self, modals_dirs):
		self.Annotations = pd.read_csv(os.path.join(self.AnnotationDir,self.Mode + '.csv'))
		md = []
		for nm in modals_dirs:
			if '-' in nm:
				nm = nm.split('-')
				md.append(os.path.join(self.RootDir, self.Mode, nm[0], nm[1]))
				continue
			md.append(os.path.join(self.RootDir, self.Mode, nm))
		self.ModalsDirs = md

	def __len__(self):
		return len(self.Annotations)

	def relabeled(self, newlabels):
		self.Relabel = True
		self.NewLabel = newlabels
		self.Categories = list(newlabels['cat'].keys())

	# def ReviewMissing(self, sample):
	# 	# ctx_sum = np.sum(sample['context'])
	# 	# bdy_sum = np.sum(sample['body'])
	# 	fce_sum = np.sum(sample['face'],dtype=np.float_)
	# 	pos_sum = np.sum(sample['joint'],dtype=np.float_)
	# 	skip = True
	# 	if self.Missing == 'none' and (fce_sum != 0.0 and pos_sum != 0.0): # none should be missing (1111)
	# 		skip = False
	# 	elif self.Missing == 'face' and (fce_sum == 0.0 and pos_sum != 0.0): # face should be missing (1101)
	# 		skip = False
	# 	elif self.Missing == 'pose' and (fce_sum != 0.0 and pos_sum == 0.0): # pose should be missing (1110)
	# 		skip = False
	# 	elif self.Missing == 'both' and (fce_sum == 0.0 and pos_sum == 0.0): # both should be missing (1100)
	# 		skip = False
	# 	elif self.Missing == '':
	# 		skip = False
	# 	return skip

	def __getitem__(self, idx):
		if torch.is_tensor(idx):
			 idx = idx.tolist()
		
		if self.Modality == 'label':
			if len(self.Continuous)==0:
				nplbl = self.getlabel(self.Annotations.iloc[idx,0])
			else:
				nplbl = self.getlabel(self.Annotations.iloc[idx,1])
			return {'label': nplbl}

		ctx_dir = os.path.join(self.RootDir, self.Annotations.iloc[idx, 2],self.Annotations.iloc[idx, 3])
		npctx = np.load(ctx_dir)
	
		bod_dir = os.path.join(self.RootDir, self.Annotations.iloc[idx, 4],self.Annotations.iloc[idx, 5])
		npbod = np.load(bod_dir)
		
		if isinstance(self.Annotations.iloc[idx, 7], str): # there are face
			fac_dir = os.path.join(self.RootDir, self.Annotations.iloc[idx, 6],self.Annotations.iloc[idx, 7])
			npfac = np.load(fac_dir)
		else:
			if self.Modality == 'face': # there are not face, but it is required
				return None
			npfac = np.zeros((64,64,3))
		
		if isinstance(self.Annotations.iloc[idx, 9], str): # there are posture
			joi_dir = os.path.join(self.RootDir, self.Annotations.iloc[idx, 8],self.Annotations.iloc[idx, 9])
			npjoi = np.load(joi_dir)
			bon_dir = os.path.join(self.RootDir, self.Annotations.iloc[idx, 10],self.Annotations.iloc[idx, 11])
			npbon = np.load(bon_dir)
		else:
			if self.Modality == 'pose': # there are not pose, but it is required
				return None
			npjoi = np.zeros((3, 1, 15, 1))
			npbon = np.zeros((3, 1, 15, 1))
		
		if len(self.Continuous)==0:
			nplbl = self.getlabel(self.Annotations.iloc[idx,0])
		else:
			nplbl = self.getlabel(self.Annotations.iloc[idx,1])

		sample = {'label': nplbl,
			'context': npctx,
			'body': npbod,
			'face': npfac,
			'joint': npjoi,
			'bone': npbon}
		
		if self.Transform:
			sample = self.Transform(sample)
		# if self.ReviewMissing(sample):
		# 	return None
		return sample
	
	def getlabel(self, categories):
		curr_categories = [ct[1:-1].replace('\'','') for ct in (categories[1:-1]).split(',')]
		if self.Continuous:
			lbl = np.zeros(len(self.Continuous))
			# not yet
			return lbl
		lbl = np.zeros(len(self.Categories))
		for i, ct in enumerate(self.Categories):
			if self.Relabel:
				for cct in self.NewLabel['cat'][ct]:
					if cct in curr_categories:
						lbl[i] = 1.0
						if self.TakeOne:
							return np.argmax(lbl)[0]
				continue
			if ct in curr_categories:
				lbl[i] = 1.0
				if self.TakeOne:
					return np.argmax(lbl)[0]
		
		return lbl

class Rescale(object):
	def __init__(self, context_output_size, body_output_size, face_output_size):
		assert isinstance(context_output_size, (int, tuple))
		assert isinstance(body_output_size, (int, tuple))
		assert isinstance(face_output_size, (int, tuple))

		if isinstance(context_output_size, int):
			self.ContextOutputSize = (context_output_size, context_output_size)
		else:
			assert len(context_output_size) == 2
			self.ContextOutputSize = context_output_size
		if isinstance(body_output_size, int):
			self.BodyOutputSize = (body_output_size, body_output_size)
		else:
			assert len(body_output_size) == 2
			self.BodyOutputSize = body_output_size
		if isinstance(face_output_size, int):
			self.FaceOutputSize = (face_output_size, face_output_size)
		else:
			assert len(face_output_size) == 2
			self.FaceOutputSize = face_output_size

	def __call__(self, sample):
		lbl = sample['label']
		ctx = sample['context']
		bod = sample['body']
		fac = sample['face']
		joi = sample['joint']
		bon = sample['bone']

		new_ch, new_cw = self.ContextOutputSize
		new_bh, new_bw = self.BodyOutputSize
		new_fh, new_fw = self.FaceOutputSize

		ctx = cv2.resize(ctx, (new_ch, new_cw))
		bod = cv2.resize(bod, (new_bh, new_bw))
		fac = cv2.resize(fac, (new_fh, new_fw))

		return {'label': lbl, 'context': ctx, 'body': bod, 'face': fac, 'joint': joi, 'bone': bon}

class RandomCrop(object):
	def __init__(self, context_output_size, body_output_size, face_output_size):
		assert isinstance(context_output_size, (int, tuple))
		assert isinstance(body_output_size, (int, tuple))
		assert isinstance(face_output_size, (int, tuple))

		if isinstance(context_output_size, int):
			self.ContextOutputSize = (context_output_size, context_output_size)
		else:
			assert len(context_output_size) == 2
			self.ContextOutputSize = context_output_size
		if isinstance(body_output_size, int):
			self.BodyOutputSize = (body_output_size, body_output_size)
		else:
			assert len(body_output_size) == 2
			self.BodyOutputSize = body_output_size
		if isinstance(face_output_size, int):
			self.FaceOutputSize = (face_output_size, face_output_size)
		else:
			assert len(face_output_size) == 2
			self.FaceOutputSize = face_output_size

	def __call__(self, sample):
		lbl = sample['label']
		ctx = sample['context']
		bod = sample['body']
		fac = sample['face']
		joi = sample['joint']
		bon = sample['bone']

		hc, wc = ctx.shape[:2]
		new_hc, new_wc = self.ContextOutputSize
		hb, wb = bod.shape[:2]
		new_hb, new_wb = self.BodyOutputSize
		hf, wf = fac.shape[:2]
		new_hf, new_wf = self.FaceOutputSize

		top_c = np.random.randint(0, hc - new_hc)
		left_c = np.random.randint(0, wc - new_wc)
		top_b = np.random.randint(0, hb - new_hb)
		left_b = np.random.randint(0, wb - new_wb)
		top_f = np.random.randint(0, hf - new_hf)
		left_f = np.random.randint(0, wf - new_wf)

		ctx = ctx[top_c: top_c + new_hc,
							left_c: left_c + new_wc]
		bod = bod[top_b: top_b + new_hb,
							left_b: left_b + new_wb]
		fac = fac[top_f: top_f + new_hf,
							left_f: left_f + new_wf]

		return {'label': lbl, 'context': ctx, 'body': bod, 'face': fac, 'joint': joi, 'bone': bon}

class ToTensor(object):
	def __call__(self, sample):
		lbl = sample['label']
		ctx = sample['context']
		bod = sample['body']
		fac = sample['face']
		joi = sample['joint']
		bon = sample['bone']

		ctx = ctx.transpose((2, 0, 1))
		bod = bod.transpose((2, 0, 1))
		fac = fac.transpose((2, 0, 1))

		return {'label': torch.from_numpy(lbl).float(),
						 'context': torch.from_numpy(ctx).float(),
						 'body': torch.from_numpy(bod).float(),
						 'face': torch.from_numpy(fac).float(),
						 'joint': torch.from_numpy(joi).float(),
						 'bone': torch.from_numpy(bon).float()}