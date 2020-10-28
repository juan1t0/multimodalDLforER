import os
import pandas as pd
import numpy as np
import cv2

import torch
from torch.utils.data import Dataset#, DataLoader

'''
	revisar todo / comentar
'''

class Emotic_MDB(Dataset):
	# This Dataset provide the process adecuated input for MER
	def __init__ (self,
								root_dir='/Emotic_MDB',
								annotation_dir='/annotations',
								mode='train',
								modals_names=[],
								categories=[],
								transform=None):

		super(Emotic_MDB, self).__init__()
		self.RootDir = root_dir
		self.Mode = mode
		self.AnnotationsDir = root_dir + annotation_dir + '/' + mode
		self.ModalsNames = modals_names
		self.Categories = categories
		self.Transform = transform
		self.loadData()

	def loadData(self):
		annlist = os.listdir(self.AnnotationsDir)
		la = {}
		for nm in annlist:
			k = nm[:-4]
			la[k]= pd.read_csv(self.AnnotationsDir + '/'+nm)
		self.Annotations = la
		md = []
		for nm in self.ModalsNames:
			md.append(self.RootDir+'/'+self.Mode + '/' + nm + '/')
		self.ModalsDirs = md

	def __len__(self):
		return len(self.Annotations[0])

	def __getitem__(self, idx):
		if torch.is_tensor(idx):
			 idx = idx.tolist()
    
		flm_dir = self.Annotations['face_landmarks'].iloc[idx, 1] + self.Annotations['face_landmarks'].iloc[idx, 0]
		sklj_dir = self.Annotations['skeleton'].iloc[idx, 1] + self.Annotations['skeleton'].iloc[idx, 0]
		sklb_dir = self.Annotations['skeleton'].iloc[idx, 3] + self.Annotations['skeleton'].iloc[idx, 2]
		ctx_dir = self.Annotations['context_blur'].iloc[idx, 1] + self.Annotations['context_blur'].iloc[idx, 0]

		flm = np.load(flm_dir)
		skl_j = np.load(sklj_dir)
		skl_b = np.load(sklb_dir)
		ctx = np.load(ctx_dir)
		
		lbl = self.getlabel(self.Annotations['context_blur'].iloc[idx,2])

		sample = {'label': lbl,
			'face_landmarks': flm[:,None],
			'skeleton_joints': skl_j,
			'skeleton_bones': skl_b,
			'context': ctx}
		
		if self.Transform:
			sample = self.Transform(sample)

		return sample
	
	def getlabel(self, categories):
		curr_categories = [ct[1:-1].replace('\'','') for ct in (categories[1:-1]).split(',')]
		
		lbl = np.zeros(len(self.Categories))
		for i, ct in enumerate(self.Categories):
			if ct in curr_categories:
				lbl[i] = 1.0
		return lbl


class Emotic_PP(Dataset):
	# This Dataset provide the original images and
	def __init__(self,
							 root_dir='/emotic',
							 annotations_dir='/annotaions',
							 mode='train',
							 transform = None):
		super(Emotic_PP, self).__init__()
		self.RootDir = root_dir
		self.Annotations = pd.read_csv(annotations_dir +'/'+ mode +'.csv')
		self.Mode = mode
		self.Transform = transform
	
	def __len__(self):
		return len(self.Annotations)
	
	def __getitem__(self, idx):
		if torch.is_tensor(idx):
			 idx = idx.tolist()
		
		img_folder = self.RootDir +'/'+ self.Annotations.iloc[idx,1] +'/'
		img_file = self.Annotations.iloc[idx,2]
		img = cv2.imread(img_folder + img_file)

		person_folder = (self.RootDir +'/'+ self.Annotations.iloc[idx,1]).replace('images','persons') +'/'
		person_file = self.Mode + '_person_'+str(idx)+'.jpg'
		person = cv2.imread(person_folder + person_file)
		
		label = self.Annotations.iloc[idx,5]
		bbox = self.Annotations.iloc[idx,4]

		sample = {'label': label,
							'imagen': img,
							'person': person,
							'bounding_box': bbox,
							'image_filename': img_file, 
							'person_filename': person_file,
							'folder': self.Annotations.iloc[idx,1]}

		if self.Transform:
			sample = self.Transform(sample)

		return sample