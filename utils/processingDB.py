import numpy as np
import cv2
import csv
import json
import os
from os.path import exists

"""
	revisar el extractor de pose, debe retornar joints y bones
"""

directed_edges =[(0,1),(0,15),(0,16),(1,2),(1,5),
    (1,8),(2,3),(3,4),(5,6),(6,7),
    (8,9),(8,12),(9,10),(10,11),(12,13),
    (13,14),(15,17),(16,18),(14,21),(14,19),
    (19,20),(11,24),(11,22),(22,23),(1,1)]

def get_skeleton_pose(img, img_folder, img_filename, temp_folder, colab=True):
	'''
	img = image
	img_folder : Folder where images are placed
	img_filename : name of the image
	img_idx : images' index #### del
	temp_folder : Folder to save the result
	colab : true if running on colaboratory

	return the numpy array with the skeleton
	'''
	if colab: 
		if not exists(temp_folder):
			os.mkdir(temp_folder)

		imgpath = '--video '+ '/content/emotic/' + img_folder +'/'+ img_filename
		imgsave = ' --write_json '+ temp_folder

		state = os.system('cd openpose && ./build/examples/openpose/openpose.bin '+ imgpath + imgsave + ' --display 0 --render_pose 0')
		if state != 0:
				C, T, V, N = 3, 1, 25, 1 #chanels, frame, joints, persons
				return np.zeros((C, T, V, N))

		sample_name = img_filename[:-4] + '_000000000000_keypoints.json'
		sample_path = os.path.join('./temp', sample_name)

		with open(sample_path, 'r') as f:
			skln = json.load(f)
		data = skln['people']

		C, T, V, N = 3, 1, 25, 1 #chanels, frame, joints, persons

		data_np_joint = np.zeros((C, T, V, N))
		pose = data[0]['pose_keypoints_2d']
		data_np_joint[0, 0, :, 0] = pose[0::3]
		data_np_joint[1, 0, :, 0] = pose[1::3]
		data_np_joint[2, 0, :, 0] = pose[2::3]

		data_np_bone = np.zeros((C, T, V, N))
		for v1,v2 in directed_edges:
			data_np_bone[:,:,v1,:] = data_np_joint[:,:,v1,:] - data_np_joint[:,:,v2,:]

		rmfn = temp_folder + '/' + sample_name
		os.system('rm ' + rmfn)
		return data_np_joint, data_np_bone
	else:
		return 0

def get_face_landmarks(img, model=None, npoints=68):
	'''
	img : image as numpy array
	flmodel : the model for obtain the landmarks
	npoints : the number of landmarks to detect, 68 by default
	return a 1D numpy array with the landmarks
	'''
	face = model.get_landmarks(img)
	if face == None:
		return np.ones(npoints*2)

	face = face[0]
	flm = np.zeros(npoints*2)
	for i, lm in enumerate(face):
		flm[i*2] = lm[0]
		flm[(i*2)+1] = lm[1]
	
	return flm

def get_context(img, bbox, mode='blank', finalW=224, finalH=224):
	'''
	img : numpy array
	bbox : can be a str with the bbox, format '[y1,x1,y2,x2]' or the actually bbox
	finalW & finalH : final dimension
	return img with zeros intead of bbox
	'''
	if type(bbox) == str:
		bbox = [int(x) for x in (bbox[1:-1]).split(',')]

	x1, x2, y1, y2 = bbox[1], bbox[3], bbox[0], bbox[2]
	if x2 > img.shape[0]:
		x2 = img.shape[0]
	if y2 > img.shape[1]:
		y2 = img.shape[1]

	cm = np.copy(img)

	if mode == 'blank':
		cm[x1:x2,y1:y2,:] = np.zeros((x2-x1, y2-y1, 3))
	elif mode == 'blur':
		blr = cv2.GaussianBlur(cm, (21, 21), 0)
		condition = np.ones(cm.shape)
		condition[x1:x2,y1:y2,:] = np.zeros((x2-x1, y2-y1, 3))
		cm = np.where(condition,cm,blr)
	else:
		raise ValueError('No mode')

	cm = cv2.resize(cm, (finalW, finalH))
	return cm

def gen_emotic_mdb_skl(EmoticDataset, root_dir, annotation_dir, colab=True):
	mode = EmoticDataset.Mode
	save_dir = root_dir +'/'+ mode +'/posture/'
	annotation_dir = root_dir + '/'+ annotation_dir +'/'+ mode +'/'

	save_Jdir = save_dir +'joints/'
	save_Bdir = save_dir +'bones/'

	with open(annotation_dir+'posture.csv','w') as file:
		csv_file = csv.writer(file, delimiter=',')
		row = ['File_Joint_Name', 'Folder_Joint_Path',
					 'File_Bone_Name', 'Folder_Bone_Path',
					 'Label', 'Original_image', 'Process_image']
		csv_file.writerow(row)
	
		if colab:
			for i, elem in enumerate(EmoticDataset):
				imgFolder = elem['folder'].replace('images', 'persons')
				imgFile = elem['person_filename']

				skl_join, skl_bone = get_skeleton_pose(img_folder=imgFolder, img_filename=imgFile, temp_folder='/content/temp')
				
				imgJFile = 'skl_join_'+ str(i) +'.npy'
				np.save(save_Jdir + imgJFile, skl_join)

				imgBFile = 'skl_bone_'+ str(i) +'.npy'
				np.save(save_Bdir + imgBFile, skl_bone)
				
				row = [imgJFile, save_Jdir,
							 imgBFile , save_Bdir,
							 elem['label'], elem['image_filename'], elem['person_filename']]
				csv_file.writerow(row)
				
				if (i+1) % 2000 == 0:
					print(i, 'images processed')
		else:
			raise ValueError('Not yet')

def gen_emotic_mdb_flm(EmoticDataset, root_dir, annotation_dir, model=None):
	mode = EmoticDataset.Mode
	save_dir = root_dir + '/' + mode +'/face_landmarks/'
	annotation_dir = root_dir +'/'+ annotation_dir +'/'+ mode +'/'

	with open(annotation_dir+'face_landmarks.csv', 'w') as file:
		csv_file = csv.writer(file, delimiter=',')
		row = ['File_Name', 'Folder_Path', 'Label',
					 'Original_image', 'Process_image']
		csv_file.writerow(row)
		for i, elem in enumerate(EmoticDataset):
			flm = get_face_landmarks(elem['person'], model=model)

			imgFile = 'flm_'+ str(i) +'.npy'
			np.save(save_dir + imgFile, flm)
			
			row = [imgFile, save_dir, elem['label'],
						 elem['image_filename'], elem['person_filename']]
			csv_file.writerow(row)

			if (i+1) % 2000 == 0:
				print(i, 'images processed')

def gen_emotic_mdb_ctx(EmoticDataset, root_dir, annotation_dir, mode='blank'):
	set_mode = EmoticDataset.Mode
	save_dir = root_dir + '/' + set_mode +'/context_'+ mode + '/'
	annotation_dir = root_dir +'/'+ annotation_dir +'/'+ set_mode +'/'
	
	with open(annotation_dir+'context_'+ mode + '.csv', 'w') as file:
		csv_file = csv.writer(file, delimiter=',')
		row = ['File_Name', 'Folder_Path', 'Label',
					 'Original_image']
		csv_file.writerow(row)
		for i, elem in enumerate(EmoticDataset):
			ctx = get_context(img=elem['imagen'], bbox=elem['bounding_box'], mode=mode)
			imgFile = 'ctx_'+ str(i) +'.npy'
			np.save(save_dir + imgFile, ctx)

			row = [imgFile, save_dir, elem['label'], 
						 elem['image_filename']]
			csv_file.writerow(row)

			if (i+1) % 2000 == 0:
				print(i, 'images processed')

def gen_emotic_mdb_all(EmoticDataset, annotationDestDir, mode='blank', colab=true):
    # create csv
    if colab:
        for i, elem in enumerate(EmoticDataset):
            img = elem['person_dir']
            skl = get_skeleton_pose(img, colab)
            img = elem['person']
            flm = get_face_landmarks(img)
            img = elem['imagen']
            ctx = get_context(img,mode)
            # save skl
            # write row
    else:
        for i, elem in enumerate(EmoticDataset):
            img = elem['person']
            skl = get_skeleton_pose(img, colab)
            flm = get_face_landmarks(img)
            img = elem['imagen']
            ctx = get_context(img,mode)
            # save skl
            # write row