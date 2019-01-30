import argparse
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import sys
import os
import torch
from glob import glob
import scipy.io as sio
import random
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import torch.nn as nn
from PIL import Image
import math 
from time import time as time 
from scipy import stats
from tqdm import tqdm 

#import chamfer distance 
sys.path.insert(0, "chamfer_distance/")
from modules.nnd import NNDModule
dist =  NNDModule()
np.random.seed(2)
random.seed(2)

from torchvision.transforms import Normalize as norm 
norms = norm(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
from torchvision import transforms
preprocess = transforms.Compose([
   transforms.CenterCrop(224),
   transforms.ToTensor(),
   norms
])


# calculates the point to point differntiable surface sampling
def point_to_point(pred_vert, adj_info, gt_points, num = 1000 ):
	# grab the faces still in use 
	active_faces_indecies = adj_info['face_list'][:,0,2].astype(int)
	all_faces = adj_info['faces']
	pred_face = all_faces[active_faces_indecies]
	pred_faces = torch.LongTensor(pred_face).cuda()

	# sample from faces and calculate pairs
	pred_points = sample(pred_vert, pred_faces, num =num)
	pred_points_pass = pred_points.view([1,pred_points.shape[0], pred_points.shape[1]])
	gt_points_pass = gt_points.view([1,gt_points.shape[0], gt_points.shape[1]])
	id_1, id_2 = dist( gt_points_pass, pred_points_pass)
	
	# select pairs and calculate chamfer distance 
	pred_counters = torch.index_select(pred_points, 0, id_1[0].long())
	gt_counters =  torch.index_select(gt_points, 0, id_2[0].long())
	dist_1 = torch.mean(torch.sum((pred_counters - gt_points)**2, dim  = 1 ))
	dist_2 = torch.mean(torch.sum((gt_counters - pred_points)**2, dim = 1))
	loss = (dist_1 + dist_2) * 3000 	
	return loss, pred_points

# loads the initial mesh and stores vertex, face, and adjacency matrix information
def load_initial( obj='386.obj'):
	# load obj file
	obj = ObjLoader(obj)
	labels = np.array(obj.vertices) 
	features = torch.FloatTensor(labels).cuda()
	faces = np.array(obj.faces) -1

	# get adjacency matrix infomation
	adj_info = adj_init(faces, len(labels))

	# get face information 
	adj_info['face_list'] = calc_face_list(faces)
	return adj_info, features  

# loads object file
# involves identifying face and vertex infomation in .obj file 
# needs to be triangulated to work 
class ObjLoader(object):
	def __init__(self, fileName):
		self.vertices = []
		self.faces = []
		##
		try:
			f = open(fileName)
			for line in f:
				line = line.replace('//','/')
				if line[:2] == "v ":
					index1 = line.find(" ") + 1
					index2 = line.find(" ", index1 + 1)
					index3 = line.find(" ", index2 + 1)
					vertex = (float(line[index1:index2]), float(line[index2:index3]), float(line[index3:-1]))
					self.vertices.append(vertex)

				elif line[0] == "f":
					string = line.split(' ')
					string = string[1:]
					string.reverse()
					face = [int(s.split('/')[0]) for s in string]

					self.faces.append(face)
			f.close()
		except IOError:
			print(".obj file not found.")
			exit()


def adj_init(faces,num_verts):
	adj = np.zeros((num_verts,num_verts))
	faces = np.array(faces) 

	# calculate binary adjacency matrix
	for e,f in enumerate(faces):
		x,y,z= f.astype(int)
		adj[x,y] = 1
		adj[y,x] = 1

		adj[x,z] = 1
		adj[z,x] = 1

		adj[y,z] = 1
		adj[z,y] = 1 
	
	# make symmetric matrix, and save binary version 
	adj_orig = (adj + np.eye(adj.shape[0]))
	adj = torch.FloatTensor(normalize_adj(adj_orig)).cuda()
	adj_info = {}

	adj_info['adj'] = adj 
	adj_info['adj_orig'] = adj_orig
	adj_info['faces'] = faces
	return adj_info 


# normalizes symetric, binary adj matrix such that sum of each row is 1 
def normalize_adj(mx):
	rowsum = np.array(mx.sum(1))
	r_inv = np.power(rowsum, -1).flatten()
	r_inv[np.isinf(r_inv)] = 0.
	r_mat_inv = sp.diags(r_inv)

	mx = r_mat_inv.dot(mx)
	return mx

# calcualtes the a list of information for each face 
# the mesh is triangulated so each face is connected to 3 faces or neighbors
# we label the 0,1,2, and everyface poses a list of information about each neighbor 
# this information is: 
	#[ the index of the nighbor in the full list of faces (adj_info['faces']), 
	# what label this neighbor has given the face , 
	# its own face number in the list of faces ]
# face i stores info on the neighbor labelled j in face_list[i,j]
# only active faces are stored in the face_list
def calc_face_list(faces): 
	face_list = [[[],[],[]] for i in range(len(faces))]
	for e,f1 in enumerate(faces):
		for ee, f2 in enumerate(faces):
			f1_position = -1
			f2_position = -1
			if ee == e: continue
			if f1[0] in f2 and f1[1] in f2: 
				f1_position = 0
			elif f1[1] in f2 and f1[2] in f2: 
				f1_position = 1
			elif f1[0] in f2 and f1[2] in f2: 
				f1_position = 2
			if f1_position >= 0 : 
				if f2[0] in f1 and f2[1] in f1: 
					face_list[e][f1_position] = [ee,0, e]
				if f2[1] in f1 and f2[2] in f1: 
					face_list[e][f1_position] = [ee,1, e]
				if f2[0] in f1 and f2[2] in f1: 
					face_list[e][f1_position] = [ee,2, e]
						
	return np.array(face_list)

# loader for GEOMetrics
class Mesh_loader(object):
	def __init__(self, Img_location, Mesh_loction, Sample_location):

		# initialization of data locations 
		self.Mesh_loction = Mesh_loction
		self.Img_location = Img_location
		self.Sample_location = Sample_location
		names = [f.split('/')[-1] for f in glob(Img_location + '*')]
		
		self.names = []
		for n in tqdm(names):
			# only examples which have both images and ground-truth sampled points are loaded
			if os.path.exists(self.Sample_location + n  +'.mat'):
					self.names.append(n)	
		print 'The number of objects found : '  + str(len(self.names))

	def load_batch(self, size, ordered = False): 
		batch_names = []
		latent = []
		
		for i in range(size): 
			# select example 
			if ordered: 
				obj = self.names[i]
			else: 
				obj = random.choice(self.names)
			obj_choice = obj.split('/')[-1]
			batch_names.append(obj)
			if i == 0 : batch_verts, batch_samples, batch_adjs = [], [], []
			
			# if reduced mesh object exists, load all stuff for latent loss
			if os.path.isfile(self.Mesh_loction +obj +  '.mat'):
				mesh_info  =  sio.loadmat(self.Mesh_loction +obj +  '.mat'  )
				verts = mesh_info['verts']
				verts= torch.FloatTensor(np.array(verts)).cuda()
				adj = mesh_info['orig_adj']
				adj = normalize_adj(adj)
				adj = torch.FloatTensor(adj).cuda()
				adj_info={'adj': adj}
				batch_verts.append(torch.FloatTensor(np.array(verts)).cuda())
				batch_adjs.append(adj_info)
				latent.append(True)
			else: 
				latent.append(False)
				batch_verts.append(False)
				batch_adjs.append(False)

			# load sampled ground truth points
			samples = sio.loadmat(self.Sample_location +obj +  '.mat'  )['points'] 
			np.random.shuffle(samples)
			batch_samples.append(torch.FloatTensor(samples).cuda() )
			
			#load images
			num = random.randint(0,23) if ordered is False else 0   
			str_num = str(num)
			if i == 0 : batch_images = []
			img = (Image.open(self.Img_location + obj + '/rendering/' + str_num + '.png'))
			img = preprocess(img)
			batch_images.append(np.asarray(img))
			if i == 0 : batch_matricies = []
			batch_matricies.append(np.load(self.Img_location + obj + '/rendering/' + str(num) + '.npy'))

		batch = {'names':np.array(batch_names)}	
		batch['latent'] = latent
		batch['verts'] = batch_verts
		batch['adjs'] = batch_adjs
		batch['samples'] = batch_samples
		batch['imgs'] = torch.FloatTensor(np.array(batch_images)).cuda()
		batch['mats'] = torch.FloatTensor(np.array(batch_matricies)).cuda()
		return batch




# loader to Auto-Encoder
class Voxel_loader(object):
	def __init__(self, Img_location, Mesh_loction,Voxel_location):

		self.Voxel_location = Voxel_location
		self.Mesh_loction = Mesh_loction
		self.Img_location = Img_location
		names = [f.split('/')[-1] for f in glob(Img_location + '*')]
		self.names = []

		for n in tqdm(names): 
			if os.path.exists(self.Mesh_loction + n  +'.mat'):
				self.names.append(n)
		print 'The number of objects found : '  + str(len(self.names))

	def load_batch(self, size, images = False,  ordered = False): 
		batch_names = []
		faces = []
		for i in range(size): 
			if ordered: 
				obj = self.names[i]
			else: 
				obj = random.choice(self.names)
			obj_choice = obj.split('/')[-1]
			batch_names.append(obj)
	
			
			if i == 0 : batch_voxels = []
			voxs = sio.loadmat(self.Voxel_location +obj_choice+ '.mat')['model']
			batch_voxels.append(np.asarray(voxs))

			
			if i == 0 : batch_verts, batch_norms, batch_adjs, batch_faces = [], [], [], []
			mesh_info  =  sio.loadmat(self.Mesh_loction +obj +  '.mat'  )
			verts = mesh_info['verts']
			verts= torch.FloatTensor(np.array(verts)).cuda()
			faces = mesh_info['faces']
			adj = mesh_info['orig_adj']
			adj[np.where(adj > 0)] = 1. 
			adj = normalize_adj(adj)

			adj = torch.FloatTensor(adj).cuda()
			adj_info={'adj': adj}
			batch_verts.append(verts)
			batch_faces.append(torch.LongTensor(np.array(faces)).cuda())
			batch_adjs.append(adj_info)

			if images: 
				num = random.randint(0,23) if ordered is False else 0  
				str_num = str(num)
				if i == 0 : batch_images = []
				img = (Image.open(self.Img_location + obj + '/rendering/' + str_num + '.png'))
				img = preprocess(img)
				batch_images.append(np.asarray(img))
				if i == 0 : batch_matricies = []
				batch_matricies.append(np.load(self.Img_location + obj + '/rendering/' + str(num) + '.npy'))


		batch = {'names':np.array(batch_names)}
		batch['voxels'] = torch.FloatTensor(np.array(batch_voxels)).cuda()
		batch['verts'] =  batch_verts 
		batch['faces'] =  batch_faces 
		batch['adjs'] =  batch_adjs
		if images:
			batch['imgs'] = torch.FloatTensor(np.array(batch_images)).cuda()
			batch['mats'] = torch.FloatTensor(np.array(batch_matricies)).cuda()

		return batch




# graphs the training and validation information 
def graph( location, train, valid): 
	train =  savitzky_golay(np.array(train), 25, 3)
	plt.plot(train, color='blue') 
	plt.grid()
	plt.savefig( location + 'train.png' )
	plt.clf()

	plt.plot(valid,color='green')
	plt.grid()
	plt.savefig( location + 'valid.png' )
	plt.clf()

 
	diff = len(train) // len(valid)
	x_t = [(i+1) for i in range(len(train))]
	x_v = [diff* (i+1) for i in range(len(valid))]
	
	
	plt.plot(x_t,train, color='blue')
	plt.plot(x_v,valid,color='red')
	plt.grid()
	plt.savefig( location + 'together.png' )
	plt.clf()

# smooths array on numbers within window, used for smoothing graphs 
def savitzky_golay(y, window_size, order, deriv=0, rate=1):
	import numpy as np
	from math import factorial
	try:
		window_size = np.abs(np.int(window_size))
		order = np.abs(np.int(order))
	except ValueError, msg:
		raise ValueError("window_size and order have to be of type int")
	if window_size % 2 != 1 or window_size < 1:
		raise TypeError("window_size size must be a positive odd number")
	if window_size < order + 2:
		raise TypeError("window_size is too small for the polynomials order")
	order_range = range(order+1)
	half_window = (window_size -1) // 2
	b = np.mat([[k**i for i in order_range] for k in range(-half_window, half_window+1)])
	m = np.linalg.pinv(b).A[deriv] * rate**deriv * factorial(deriv)
	firstvals = y[0] - np.abs( y[1:half_window+1][::-1] - y[0] )
	lastvals = y[-1] + np.abs(y[-half_window-1:-1][::-1] - y[-1])
	y = np.concatenate((firstvals, y, lastvals))
	return np.convolve( m[::-1], y, mode='valid')

# this is how each vertex extracts information from image features 
# blocks is a set of feature blocks, block = [b0, .. bn], where each bi is a n*n*m, where n is the resolution, and m is the colour channel
# verts_position is a k*3 array of vertex positions 
# matrix is the projection matrix associated wiht the image, this is matrix which converts the vertex position into image coordinates, was used to produce the image from the objects origionally 
def pooling(blocks, verts_pos, matrix):

	# convert vertex positions to x,y coordinates in the image, scaled to fractions of image dimension 
	ext_verts_pos = torch.cat((verts_pos, torch.FloatTensor(np.ones((verts_pos.shape[0],1))).cuda()), dim = 1)
	ext_verts_pos = torch.mm(ext_verts_pos, matrix.permute(1,0))
	ys = (ext_verts_pos[:,0]/ext_verts_pos[:,3] + 1. ) /2.  
	xs = 1  - (ext_verts_pos[:,1]/ext_verts_pos[:,3] + 1.)  / 2. 

	full_features = None 
	for block in blocks: 
	   
	   	# scale coordinated to block dimensions/resolution 
		dim = block.shape[1]
		cur_xs = torch.clamp(xs * dim, 0, block.shape[1]-1)
		cur_ys = torch.clamp(ys * dim, 0, block.shape[1]-1) 
	
		# this is basically bilinear interpolation of the 4 closest feature vectors to where the vertex lands in the block 
		# https://en.wikipedia.org/wiki/Bilinear_interpolation
		x1s, y1s, x2s, y2s = torch.floor(cur_xs), torch.floor(cur_ys), torch.ceil(cur_xs), torch.ceil(cur_ys)
		A =x2s -cur_xs
		B = cur_xs-x1s
		G = y2s-cur_ys
		H = cur_ys-y1s
 
		# some nice stuff happens here to make it faster, not too complicated 
		x1s = x1s.type(torch.cuda.LongTensor)
		y1s = [(y+(dim*i)).view(1) for i, y in zip(range(verts_pos.shape[0]), y1s)]
		y1s = torch.cat(y1s).type(torch.cuda.LongTensor)
		x2s = x2s.type(torch.cuda.LongTensor)
		y2s = [(y+(dim*i)).view(1) for i, y in zip(range(verts_pos.shape[0]), y2s)]
		y2s = torch.cat(y2s).type(torch.cuda.LongTensor)


		C =torch.index_select(block, 1, x1s).view(block.shape[0], -1 )
		C = torch.index_select(C, 1, y1s)
		D =torch.index_select(block, 1, x1s).view(block.shape[0], -1 )
		D = torch.index_select(D, 1, y2s)
		E =torch.index_select(block, 1, x2s).view(block.shape[0], -1 )
		E = torch.index_select(E, 1, y1s)
		F =torch.index_select(block, 1, x2s).view(block.shape[0], -1 )
		F = torch.index_select(F, 1, y2s)


		features = (A*C*G + H*D*A + G*E*B + B*F*H).permute(1,0)
	
		if full_features is None: full_features = features
		else: full_features = torch.cat((full_features, features), dim = 1)
 
	return full_features


# here we calculate the curvature of each face and return the faces above a provided curvature angle
def calc_curve( verts, info, angle = 70): 
	eps = .00001

	# extract vertex coordinated for each vertex in face 
	faces = torch.LongTensor(info['faces']).cuda()
	face_list = torch.LongTensor(info['face_list']).cuda()
	p1 = torch.index_select(verts, 0,faces[:,1])
	p2 = torch.index_select(verts, 0,faces[:,0])
	p3 = torch.index_select(verts, 0,faces[:,2])
 
 	# cauculate normals of each face 
	e1 = p2-p1
	e2 = p3-p1
	face_normals = torch.cross(e1, e2)
	qn = torch.norm(face_normals, p=2, dim=1).detach().view(-1,1)
	face_normals = face_normals.div(qn.expand_as(face_normals))
	main_face_normals = torch.index_select(face_normals, 0, face_list[:,0,2])

	# cauculate the curvature with the 3 nighbor faces 
	#1
	face_1_normals = torch.index_select(face_normals, 0, face_list[:,0,0])
	curvature_proxi_rad = torch.sum(main_face_normals*face_1_normals, dim = 1).clamp(-1.0 + eps, 1.0 - eps).acos()
	curvature_proxi_1 = (curvature_proxi_rad).view(-1,1)
	#2
	face_2_normals = torch.index_select(face_normals, 0, face_list[:,1,0])
	curvature_proxi_rad = torch.sum(main_face_normals*face_2_normals, dim = 1).clamp(-1.0 + eps, 1.0 - eps).acos()
	curvature_proxi_2 = (curvature_proxi_rad).view(-1,1)
	#3
	face_3_normals = torch.index_select(face_normals, 0, face_list[:,2,0])
	curvature_proxi_rad = torch.sum(main_face_normals*face_3_normals, dim = 1).clamp(-1.0 + eps, 1.0 - eps).acos()
	curvature_proxi_3 = (curvature_proxi_rad).view(-1,1)
	
	# get average over neighbors 
	curvature_proxi_full = torch.cat( (curvature_proxi_1, curvature_proxi_2, curvature_proxi_3), dim = 1)
	curvature_proxi = torch.mean(curvature_proxi_full, dim = 1)

	#select faces with high curvature and return their index
	splitting_faces  = np.where(curvature_proxi*180/np.pi  > angle )[0]
	if splitting_faces.shape[0] <3:
		splitting_faces  = curvature_proxi.topk(3, sorted = False)[1] 
	else:
		splitting_faces  = torch.LongTensor(splitting_faces).cuda()
	return splitting_faces


# this function take the faces feature vectors defined on the graph and splits indicated faces by:
	# adding a new vertex to the center of the face, with features equal to the averge of the 3 verts around it 
	# deleteing the previous face, ie removing from face list 
	# adding 3 new faces by connecting old verts to the new vert 
# second grossest function in this file 
def split_info( info, verts, features,  splitting_face_list_indecies ,number ):

	adj = info['adj_orig'] # this is binary 
	faces_verts = np.array(info['faces'] ) # vertex info of all faces made
	face_list = torch.LongTensor(np.array(info['face_list'])).cuda()


	splitting_face_list_values = torch.index_select(face_list, 0,  splitting_face_list_indecies ) # neighbor info of faces to be split 
	splitting_face_list_len = splitting_face_list_values.shape[0]

	counter = np.zeros((face_list.shape[0]))
	counter[ splitting_face_list_indecies ] = 1
	unsplitting_faces_list_indecies  = np.where(counter == 0 )[0]
	unsplitting_face_list_values = torch.index_select(face_list, 0, torch.LongTensor(unsplitting_faces_list_indecies).cuda())  # neighbor info of faces not split 
	

	splitting_faces_indecies = splitting_face_list_values[:,0,2] # indecies of faces being split in faces_verts
	unsplitting_faces_indecies = unsplitting_face_list_values[:,0,2] # indecies of faces not being split from face_verts 

	 
 	# indecies of new faces being made in, in the unpdated faces_verts array 
	new_faces_indecies_1 = np.arange(splitting_face_list_len).reshape(-1,1) + faces_verts.shape[0]
	new_faces_indecies_2 = new_faces_indecies_1 + splitting_face_list_len
	new_faces_indecies_3 = new_faces_indecies_2 + splitting_face_list_len
	splitting_new_faces_indecies = np.concatenate((new_faces_indecies_1, new_faces_indecies_2, new_faces_indecies_3), axis = 1 )
	unsplitting_new_faces_indecies = np.concatenate( (unsplitting_faces_indecies.reshape(-1,1), unsplitting_faces_indecies.reshape(-1,1), unsplitting_faces_indecies.reshape(-1,1)), axis = 1)

	# saving where each face will be held in the updated face_verts array, saved in this manner for quick selection
	new_positions = np.zeros((faces_verts.shape[0], 3))
	new_positions[splitting_faces_indecies] = splitting_new_faces_indecies
	new_positions[unsplitting_faces_indecies] = unsplitting_new_faces_indecies


	# adding unsplitting triangles to new face_list 
	#get location of 3 neighbors 
	unsplitting_connecting_face_1  = new_positions[unsplitting_face_list_values[:,0,0],unsplitting_face_list_values[:,0,1] ].reshape(-1,1,1)
	unsplitting_connecting_face_2  = new_positions[unsplitting_face_list_values[:,1,0],unsplitting_face_list_values[:,1,1] ].reshape(-1,1,1)
	unsplitting_connecting_face_3  = new_positions[unsplitting_face_list_values[:,2,0],unsplitting_face_list_values[:,2,1] ].reshape(-1,1,1)	
	# get the niegbors index in updated face_verts array 
	unsplitting_connecting_side_1 = unsplitting_face_list_values[:,0,1].reshape(-1,1,1)
	unsplitting_connecting_side_2 = unsplitting_face_list_values[:,1,1].reshape(-1,1,1)
	unsplitting_connecting_side_3 = unsplitting_face_list_values[:,2,1].reshape(-1,1,1)
	# make new face_list 
	unsplitting_face_number = unsplitting_faces_indecies.reshape(-1,1,1)
	new_unsplitting_face_list_1 = np.concatenate((unsplitting_connecting_face_1,unsplitting_connecting_side_1, unsplitting_face_number ), axis = 2 )
	new_unsplitting_face_list_2 = np.concatenate((unsplitting_connecting_face_2,unsplitting_connecting_side_2, unsplitting_face_number ), axis = 2 )
	new_unsplitting_face_list_3 = np.concatenate((unsplitting_connecting_face_3,unsplitting_connecting_side_3, unsplitting_face_number ), axis = 2 )
	new_unsplitting_face_list = np.concatenate((new_unsplitting_face_list_1, new_unsplitting_face_list_2, new_unsplitting_face_list_3), axis = 1)

	# adding splitting triangles to new face_list 
	# new triangle 1
	#get location of 3 neighbors
	splitting_connecting_face_1_1  = new_positions[splitting_face_list_values[:,0,0],splitting_face_list_values[:,0,1] ].reshape(-1,1,1) # one old face is its neigboors
	splitting_connecting_face_1_2  = new_faces_indecies_2.reshape(-1,1,1) # 2 new faces are its neighbor 
	splitting_connecting_face_1_3  = new_faces_indecies_3.reshape(-1,1,1)
	# get the nigbors index in updated face_verts array 
	splitting_connecting_side_1_1 = splitting_face_list_values[:,0,1].reshape(-1,1,1) # get old face's index in face_verts 
	splitting_connecting_side_1_2 = np.zeros(splitting_face_list_len).reshape(-1,1,1) # use new faces' known indices 
	splitting_connecting_side_1_3 = np.zeros(splitting_face_list_len).reshape(-1,1,1)
	# make new face_list 
	splitting_face_number_1 = new_faces_indecies_1.reshape(-1,1,1)
	new_splitting_face_list_1_1 = np.concatenate((splitting_connecting_face_1_1,splitting_connecting_side_1_1, splitting_face_number_1 ), axis = 2 ) 
	new_splitting_face_list_1_2 = np.concatenate((splitting_connecting_face_1_2,splitting_connecting_side_1_2, splitting_face_number_1 ), axis = 2 )
	new_splitting_face_list_1_3 = np.concatenate((splitting_connecting_face_1_3,splitting_connecting_side_1_3, splitting_face_number_1 ), axis = 2 )
	new_splitting_face_list_1 = np.concatenate((new_splitting_face_list_1_1, new_splitting_face_list_1_2, new_splitting_face_list_1_3), axis = 1)

	# new triangle 2
	splitting_connecting_face_2_1  = new_faces_indecies_1.reshape(-1,1,1)
	splitting_connecting_face_2_2  = new_positions[splitting_face_list_values[:,1,0],splitting_face_list_values[:,1,1] ].reshape(-1,1,1)
	splitting_connecting_face_2_3  = new_faces_indecies_3.reshape(-1,1,1)

	splitting_connecting_side_2_1 = np.ones(splitting_face_list_len).reshape(-1,1,1)
	splitting_connecting_side_2_2 = splitting_face_list_values[:,1,1].reshape(-1,1,1)
	splitting_connecting_side_2_3 = np.ones(splitting_face_list_len).reshape(-1,1,1)

	splitting_face_number_2 = new_faces_indecies_2.reshape(-1,1,1)
	new_splitting_face_list_2_1 = np.concatenate((splitting_connecting_face_2_1,splitting_connecting_side_2_1, splitting_face_number_2 ), axis = 2 )
	new_splitting_face_list_2_2 = np.concatenate((splitting_connecting_face_2_2,splitting_connecting_side_2_2, splitting_face_number_2 ), axis = 2 )
	new_splitting_face_list_2_3 = np.concatenate((splitting_connecting_face_2_3,splitting_connecting_side_2_3, splitting_face_number_2 ), axis = 2 )
	new_splitting_face_list_2 = np.concatenate((new_splitting_face_list_2_1, new_splitting_face_list_2_2, new_splitting_face_list_2_3), axis = 1)


	# new triangle 3
	splitting_connecting_face_3_1  = new_faces_indecies_1.reshape(-1,1,1)
	splitting_connecting_face_3_2  = new_faces_indecies_2.reshape(-1,1,1)
	splitting_connecting_face_3_3  = new_positions[splitting_face_list_values[:,2,0],splitting_face_list_values[:,2,1] ].reshape(-1,1,1)

	splitting_connecting_side_3_1 = np.ones(splitting_face_list_len).reshape(-1,1,1)*2
	splitting_connecting_side_3_2 = np.ones(splitting_face_list_len).reshape(-1,1,1)*2
	splitting_connecting_side_3_3 = splitting_face_list_values[:,2,1].reshape(-1,1,1)

	splitting_face_number_3 = new_faces_indecies_3.reshape(-1,1,1)
	new_splitting_face_list_3_1 = np.concatenate((splitting_connecting_face_3_1,splitting_connecting_side_3_1, splitting_face_number_3 ), axis = 2 )
	new_splitting_face_list_3_2 = np.concatenate((splitting_connecting_face_3_2,splitting_connecting_side_3_2, splitting_face_number_3 ), axis = 2 )
	new_splitting_face_list_3_3 = np.concatenate((splitting_connecting_face_3_3,splitting_connecting_side_3_3, splitting_face_number_3 ), axis = 2 )
	new_splitting_face_list_3 = np.concatenate((new_splitting_face_list_3_1, new_splitting_face_list_3_2, new_splitting_face_list_3_3), axis = 1)

	# conplete new face_list is made 
	new_splitting_face_list = np.concatenate((new_unsplitting_face_list, new_splitting_face_list_1, new_splitting_face_list_2, new_splitting_face_list_3))
	split_faces = faces_verts[splitting_faces_indecies] 

	# now to make the new vertex 
	vertex_count = adj.shape[0]
	new_len = number + vertex_count

	# select the vertices of the faces to be split 
	x_f = split_faces[:,0]
	y_f = split_faces[:,1]
	z_f = split_faces[:,2]
	verts = torch.cat((verts, features), dim = 1) 
	x_v = verts[x_f] 
	y_v = verts[y_f]
	z_v = verts[z_f]
	#average the featurs and position 
	v1 = x_v/3  + y_v/3 + z_v/3 
	verts = torch.cat((verts, v1))
	v1_inds = (vertex_count + np.arange(number) ).reshape(-1,1)
	x_f = x_f.reshape(-1,1)
	y_f = y_f.reshape(-1,1)
	z_f = z_f.reshape(-1,1)
	# define verts of new faces 
	new_face_1 = np.concatenate((x_f ,y_f ,v1_inds) , axis =1)
	new_face_2 = np.concatenate((v1_inds , y_f, z_f ) , axis =1)
	new_face_3 = np.concatenate((x_f ,v1_inds,z_f) , axis =1)
	# name new face_verts array by appending then to old in order previously defined 
	faces_verts = np.concatenate( (faces_verts, new_face_1, new_face_2, new_face_3))

	# now to update the adacency matrix 
	
	# new verts are only connected to old vert, not to any other new verts
	# old verts maintain old connects but add connections to new verts 
	# so use old adj for top left of adj, make top right of adj with new vert connection to old
	# then make syemetric and we are done 

	new_adj = np.zeros((new_len, new_len)) # new adj to be filled 
	v1_adj = np.zeros((v1.shape[0], new_len )) # this is top right
	v1_repeat = np.arange(0,v1.shape[0])
	v1_places_1 = np.concatenate((v1_repeat, v1_repeat,v1_repeat,v1_repeat)).reshape(-1) # every new nert is connected to 3 old verts and itsself
	v1_places_2 = np.concatenate((v1_inds, x_f, y_f, z_f)).reshape(-1) # indecies for new connection s
	v1_adj[v1_places_1, v1_places_2] = 1 # add ones for new conections in adj 

	new_adj[v1_inds.reshape((-1))] =  v1_adj # top right 
	new_adj[:vertex_count, :vertex_count] =   adj # top right 
	new_adj = np.maximum(new_adj, new_adj.T) # make symetric 
	adj_orig = np.array(new_adj) 
	
	# update all of out list holders 
	adj = normalize_adj(new_adj)
	adj = torch.FloatTensor(adj).cuda()

	adj_info = {'adj' : adj, 'faces': faces_verts, 'adj_orig': adj_orig, 'face_list' : new_splitting_face_list }
	features = verts[:,3:]
	verts = verts[:,:3]
	return adj_info, verts, features



# how I sample from the faces 
# verts = list of vertex positions 
# faces = list of faces, as vertex indecies in verts 
# num = number of points to sample 
def sample(verts, faces, num=10000): 

	dist_uni = torch.distributions.Uniform(torch.tensor([0.0]).cuda(), torch.tensor([1.0]).cuda())

	# calculate area of each face 
	x1,x2,x3 = torch.split(torch.index_select(verts, 0,faces[:,0]) - torch.index_select(verts, 0,faces[:,1]), 1, dim = 1)
	y1,y2,y3 = torch.split(torch.index_select(verts, 0,faces[:,1]) - torch.index_select(verts, 0,faces[:,2]), 1, dim = 1)
	a = (x2*y3-x3*y2)**2
	b = (x3*y1 - x1*y3)**2
	c = (x1*y2 - x2*y1)**2
	Areas = torch.sqrt(a+b+c)/2
	Areas = Areas /  torch.sum(Areas) # percentage of each face w.r.t. full surface area 

	# define descrete distribution w.r.t. face area ratios caluclated 
	choices = np.arange(Areas.shape[0])
	dist = stats.rv_discrete(name='custm', values=(choices, Areas.data.cpu().numpy()))
	choices = dist.rvs(size=num) # list of faces to be sampled from 

	# from each face sample a point 
	select_faces = faces[choices] 
	xs = torch.index_select(verts, 0,select_faces[:,0])
	ys = torch.index_select(verts, 0,select_faces[:,1])
	zs = torch.index_select(verts, 0,select_faces[:,2])
	u = torch.sqrt(dist_uni.sample_n(num))
	v = dist_uni.sample_n(num)
	points = (1- u)*xs + (u*(1-v))*ys + u*v*zs

	return points 

# calculate length of each edge 
def calc_edge( verts, info): 
	# get only actively used faces 
	active_faces_indecies = info['face_list'][:,0,2].astype(int)
	all_faces = info['faces']
	faces = all_faces[active_faces_indecies]
	faces = torch.LongTensor(faces).cuda()
	# get vertex loccations of faces 
	p1 = torch.index_select(verts, 0,faces[:,0])
	p2 = torch.index_select(verts, 0,faces[:,1])
	p3 = torch.index_select(verts, 0,faces[:,2])
	# get edge lentgh 
	e1 = p2-p1
	e2 = p3-p1
	e3 = p2-p3

	edge_length = (torch.sum(e1**2, 1).mean() + torch.sum(e2**2, 1).mean() + torch.sum(e3**2, 1).mean())/3.

	return edge_length

# calculate laplacian of each vertex, see paper for detials 
def get_lap_info(positions, adj_info):  
	orig = torch.FloatTensor(adj_info['adj_orig']).cuda()
	nieghbor_sum = torch.mm(orig,positions ) - positions
	degrees = torch.sum(orig, dim = 1 ) - 1
	degree_scaler = (1./degrees).view(-1,1)

	nieghbor_sum =nieghbor_sum*degree_scaler
	lap = positions - nieghbor_sum
	return lap  


# fucntion to cauculate the point to surface loss 
def point_to_surface(pred_vert, adj_info, gt_points, num = 1000): 
	
	# get active faces
	active_faces_indecies = adj_info['face_list'][:,0,2].astype(int)
	all_faces = adj_info['faces']
	pred_face = all_faces[active_faces_indecies]
	pred_face = torch.LongTensor(pred_face).cuda()
	# sample point from predicted faces 
	pred_points = sample(pred_vert,pred_face, num = num)
	
	# cualculate point to surface loss from predicted mesh to ground truth points
	pred_dist = point_to_line(pred_vert, pred_face, gt_points[:num])

	# cualculate point to point loss from groun truth points to predicted mesh
	pred_points = sample(pred_vert, pred_face, num =num)
	pred_points_pass = pred_points.view([1,pred_points.shape[0], pred_points.shape[1]])
	gt_points_pass = gt_points.view([1,gt_points.shape[0], gt_points.shape[1]])
	id_1, id_2 = dist( gt_points_pass, pred_points_pass)
	pred_counters = torch.index_select(pred_points, 0, id_1[0].long())
	gt_counters =  torch.index_select(gt_points, 0, id_2[0].long())
	gt_dist = torch.mean(torch.sum((gt_counters - pred_points)**2, dim = 1))
	loss = (pred_dist + gt_dist ) * 3000 

	return loss, 0 


# see https://www.mathworks.com/matlabcentral/fileexchange/22857-distance-between-a-point-and-a-triangle-in-3d 
# for to understand whats going on, or well, know where I am comming from anyway 
# this is basically a vectorized version of the 3D point to triangle distance function 
# grossest function 
def point_to_line(verts, faces, points): 

	# 3 verts from each face 
	p1 = torch.index_select(verts, 0,faces[:,0])
	p2 = torch.index_select(verts, 0,faces[:,1])
	p3 = torch.index_select(verts, 0,faces[:,2])

	# make more of them, one set for each sampled point 
	p1 = p1.view(1, -1, 3).expand(points.shape[0], p1.shape[0], 3).contiguous().view(-1,3)
	p2 = p2.view(1, -1, 3).expand(points.shape[0], p2.shape[0], 3).contiguous().view(-1,3)
	p3 = p3.view(1, -1, 3).expand(points.shape[0], p3.shape[0], 3).contiguous().view(-1,3)
	p0 = points.view(-1,1,3).expand(points.shape[0], faces.shape[0], 3).contiguous().view(-1,3)
	
	# here we define the traingle, see supplemental of paper to discription 
	length = p1.shape[0]
	B = p1
	E0 = p2-B;
	E1 = p3-B;
	D = B - p0;
	a = torch.bmm(E0.view(length, 1, 3), E0.view(length, 3, 1)).view(-1)
	b = torch.bmm(E0.view(length, 1, 3), E1.view(length, 3, 1)).view(-1)
	c = torch.bmm(E1.view(length, 1, 3), E1.view(length, 3, 1)).view(-1)
	d = torch.bmm(E0.view(length, 1, 3), D.view(length, 3, 1)).view(-1)
	e = torch.bmm(E1.view(length, 1, 3), D.view(length, 3, 1)).view(-1)
	f = torch.bmm(D.view(length, 1, 3), D.view(length, 3, 1)).view(-1)

	# distances holds the distances from every point to every face 
	distances = f

	det = (a*c - b*b)
	s   =( b*e - c*d)
	t   = (b*d - a*e)
	tmp0 = b+d
	tmp1 = c+e
	numer = tmp1 - tmp0
	denom = a - 2*b + c 


	# quicker to index in numpy then torch 
	det_index = det.detach().cpu().numpy()
	s_index = s.detach().cpu().numpy()
	t_index = t.detach().cpu().numpy()
	a_index = a.detach().cpu().numpy()
	c_index = c.detach().cpu().numpy()
	d_index = d.detach().cpu().numpy()
	e_index = e.detach().cpu().numpy()
	tmp0_index = tmp0.detach().cpu().numpy()
	tmp1_index = tmp1.detach().cpu().numpy()
	numer_index = numer.detach().cpu().numpy()
	denom_index = denom.detach().cpu().numpy()

	
	# these are various conditions 
	aa = np.less(s_index+ t_index, det_index )
	not_aa = np.logical_not(aa)
	bb = np.less(s_index, 0 )
	not_bb = np.logical_not(bb)
	cc = np.less(t_index,0 )
	not_cc = np.logical_not(cc)
	dd = np.less(d_index, 0 )
	not_dd = np.logical_not(dd)
	ee = np.greater_equal(-d_index,a_index )
	not_ee = np.logical_not(ee)
	ff = np.greater_equal(e_index, 0 )
	not_ff = np.logical_not(ff)
	gg = np.greater_equal(-e_index, c_index)
	not_gg = np.logical_not(gg)
	hh = np.greater_equal(d_index, 0 )
	not_hh = np.logical_not(hh)
	ii = np.greater( tmp1_index, tmp0_index)
	not_ii = np.logical_not(ii)
	jj = np.greater_equal( numer_index, denom_index)
	not_jj = np.logical_not(jj)
	kk = np.less_equal(tmp1_index, 0 )
	not_kk = np.logical_not(kk)

	# then we combine them, this is equavalent to nesting the if functions in the matlab code
	aa_bb = np.logical_and(aa, bb)
	aa_bb_cc = np.logical_and(aa_bb,cc)
	aa_bb_cc_dd = np.logical_and(aa_bb_cc, dd)
	aa_bb_cc_dd_ee= np.logical_and(aa_bb_cc_dd, ee)	
	aa_bb_cc_dd_NOT_ee= np.logical_and(aa_bb_cc_dd, not_ee)
	aa_bb_cc_NOT_dd = np.logical_and(aa_bb_cc, not_dd)
	# aa_bb_cc_NOT_dd_ff = np.logical_and(aa_bb_cc_NOT_dd, ff)
	aa_bb_cc_NOT_dd_NOT_ff = np.logical_and(aa_bb_cc_NOT_dd, not_ff)
	aa_bb_cc_NOT_dd_NOT_ff_gg = np.logical_and(aa_bb_cc_NOT_dd_NOT_ff, gg )
	aa_bb_cc_NOT_dd_NOT_ff_NOT_gg =  np.logical_and(aa_bb_cc_NOT_dd_NOT_ff, not_gg )
	aa_bb_NOT_cc = np.logical_and(aa_bb, not_cc)
	# aa_bb_NOT_cc_ff = np.logical_and(aa_bb_NOT_cc, ff)
	aa_bb_NOT_cc_NOT_ff = np.logical_and(aa_bb_NOT_cc, not_ff)
	aa_bb_NOT_cc_NOT_ff_gg = np.logical_and(aa_bb_NOT_cc_NOT_ff, gg)
	aa_bb_NOT_cc_NOT_ff_NOT_gg = np.logical_and(aa_bb_NOT_cc_NOT_ff, not_gg)
	aa_NOT_bb = np.logical_and(aa, not_bb)
	aa_NOT_bb_cc = np.logical_and(aa_NOT_bb, cc)
	# aa_NOT_bb_cc_hh = np.logical_and(aa_NOT_bb_cc, hh)
	aa_NOT_bb_cc_NOT_hh = np.logical_and(aa_NOT_bb_cc, not_hh)
	aa_NOT_bb_cc_NOT_hh_ee = np.logical_and(aa_NOT_bb_cc_NOT_hh, ee)
	aa_NOT_bb_cc_NOT_hh_NOT_ee = np.logical_and(aa_NOT_bb_cc_NOT_hh, not_ee)
	aa_NOT_bb_NOT_cc = np.logical_and( aa_NOT_bb, not_cc)
	NOT_aa_bb = np.logical_and(not_aa, bb)
	NOT_aa_bb_ii = np.logical_and(NOT_aa_bb, ii)
	NOT_aa_bb_ii_jj = np.logical_and(NOT_aa_bb_ii, jj)
	NOT_aa_bb_ii_NOT_jj = np.logical_and(NOT_aa_bb_ii, not_jj)
	NOT_aa_bb_NOT_ii = np.logical_and(NOT_aa_bb, not_ii)
	NOT_aa_bb_NOT_ii_kk = np.logical_and(NOT_aa_bb_NOT_ii, kk)
	NOT_aa_bb_NOT_ii_NOT_kk = np.logical_and(NOT_aa_bb_NOT_ii, not_kk)
	# NOT_aa_bb_NOT_ii_NOT_kk_ff = np.logical_and(NOT_aa_bb_NOT_ii_NOT_kk, ff)
	NOT_aa_bb_NOT_ii_NOT_kk_NOT_ff = np.logical_and(NOT_aa_bb_NOT_ii_NOT_kk,not_ff)

	tmp0 = b+e
	tmp1 = a+d
	numer = tmp1 - tmp0
	tmp0_index = tmp0.detach().cpu().numpy()
	tmp1_index = tmp1.detach().cpu().numpy()
	numer_index = numer.detach().cpu().numpy()

	ll = np.greater(tmp1_index, tmp0_index )
	not_ll = np.logical_not(ll)
	mm = np.greater_equal(numer_index, denom_index )
	not_mm = np.logical_not(mm)
	nn = np.less_equal(tmp1_index, 0)
	not_nn = np.logical_not(nn)



	NOT_aa_NOT_bb = np.logical_and(not_aa, not_bb)
	NOT_aa_NOT_bb_cc = np.logical_and(NOT_aa_NOT_bb, cc)
	NOT_aa_NOT_bb_cc_ll = np.logical_and(NOT_aa_NOT_bb_cc, ll)
	NOT_aa_NOT_bb_cc_ll_mm = np.logical_and(NOT_aa_NOT_bb_cc_ll, mm)
	NOT_aa_NOT_bb_cc_ll_NOT_mm = np.logical_and(NOT_aa_NOT_bb_cc_ll, not_mm)
	NOT_aa_NOT_bb_cc_NOT_ll = np.logical_and(NOT_aa_NOT_bb_cc, not_ll)
	NOT_aa_NOT_bb_cc_NOT_ll_nn = np.logical_and(NOT_aa_NOT_bb_cc_NOT_ll, nn)
	NOT_aa_NOT_bb_cc_NOT_ll_NOT_nn = np.logical_and(NOT_aa_NOT_bb_cc_NOT_ll, not_nn)
	# NOT_aa_NOT_bb_cc_NOT_ll_NOT_nn_hh = np.logical_and(NOT_aa_NOT_bb_cc_NOT_ll_NOT_nn, hh)
	NOT_aa_NOT_bb_cc_NOT_ll_NOT_nn_NOT_hh = np.logical_and(NOT_aa_NOT_bb_cc_NOT_ll_NOT_nn, not_hh)

	numer = c + e - b - d
	numer_index = numer.detach().cpu().numpy()

	oo = np.less_equal(numer_index, 0)
	not_oo = np.logical_not(oo)
	pp = np.greater_equal(numer_index, denom_index )
	not_pp = np.logical_not(pp)


	NOT_aa_NOT_bb_NOT_cc = np.logical_and(NOT_aa_NOT_bb, not_cc)
	NOT_aa_NOT_bb_NOT_cc_oo = np.logical_and(NOT_aa_NOT_bb_NOT_cc, oo)
	NOT_aa_NOT_bb_NOT_cc_NOT_oo = np.logical_and(NOT_aa_NOT_bb_NOT_cc, not_oo)
	NOT_aa_NOT_bb_NOT_cc_NOT_oo_pp = np.logical_and(NOT_aa_NOT_bb_NOT_cc_NOT_oo, pp)
	NOT_aa_NOT_bb_NOT_cc_NOT_oo_NOT_pp = np.logical_and(NOT_aa_NOT_bb_NOT_cc_NOT_oo, not_pp)


	# here we get the indecies of the nested conditions, and change the distance to this value where true
	# should be the order as the matlab code, some are commented out where already the right value
	
	#1
	index = np.where(aa_bb_cc_dd_ee)
	distances[index] += a[index] + 2*d[index] 
	#1

	#2
	index = np.where(aa_bb_cc_dd_NOT_ee)
	distances[index] += -(d[index]**2)/a[index] 
	#2

	#3 
	# index = np.where(aa_bb_cc_NOT_dd_ff)
	#3

	#4  
	index = np.where(aa_bb_cc_NOT_dd_NOT_ff_gg)
	distances[index] +=  c[index] + 2* e[index] 
	#4
	
	#5  
	index = np.where(aa_bb_cc_NOT_dd_NOT_ff_NOT_gg)
	distances[index] +=  -(e[index]**2)/c[index] 
	#5

	# #6  
	# index = np.where(aa_bb_NOT_cc_ff)
	# #6

	#7
	index = np.where(aa_bb_NOT_cc_NOT_ff_gg)
	distances[index] += c[index] + 2* e[index] 
	#7

	#8
	index = np.where(aa_bb_NOT_cc_NOT_ff_NOT_gg)
	distances[index] +=  -(e[index]**2)/c[index] 
	#8

	# #9
	# index = np.where(aa_NOT_bb_cc_hh)
	# #9

	#10
	index = np.where(aa_NOT_bb_cc_NOT_hh_ee)
	distances[index] += a[index] + 2*d[index]
	#10

	#11
	index = np.where(aa_NOT_bb_cc_NOT_hh_NOT_ee)
	distances[index] += -(d[index]**2)/a[index] 
	#11

	#12
	index = np.where(aa_NOT_bb_NOT_cc)
	invDet = 1/det[index]
	s_temp = s[index]*invDet
	t_temp = t[index]*invDet
	distances[index] += s_temp*(a[index]*s_temp + b[index]*t_temp+ 2*d[index] ) + \
					   t_temp*(b[index]*s_temp + c[index]*t_temp+ 2*e[index] ) 
	#12

	#13
	index = np.where(NOT_aa_bb_ii_jj)
	distances[index] += a[index] + 2*d[index] 
	#13

	#14
	index = np.where(NOT_aa_bb_ii_NOT_jj)
	s_temp = numer[index]/denom[index]
	t_temp = 1-s_temp
	distances[index] += s_temp*(a[index]*s_temp + b[index]*t_temp+ 2*d[index] ) + \
					   t_temp*(b[index]*s_temp + c[index]*t_temp+ 2*e[index] ) 
	#14

	#15
	index = np.where(NOT_aa_bb_NOT_ii_kk)
	distances[index] += c[index] + 2*e[index] 
	#15

	# #16
	# index = np.where(NOT_aa_bb_NOT_ii_NOT_kk_ff)
	# #16

	#17
	index = np.where(NOT_aa_bb_NOT_ii_NOT_kk_NOT_ff)
	distances[index] +=   -(e[index]**2)/c[index] 
	#17

	#18
	index = np.where(NOT_aa_NOT_bb_cc_ll_mm)
	distances[index] +=  c[index] + 2*e[index]
	#18

	#19
	index = np.where(NOT_aa_NOT_bb_cc_ll_NOT_mm)
	s_temp = numer[index]/denom[index]
	t_temp = 1-s_temp
	distances[index] += s_temp*(a[index]*s_temp + b[index]*t_temp+ 2*d[index] ) + \
					   t_temp*(b[index]*s_temp + c[index]*t_temp+ 2*e[index] ) 

	#20
	index = np.where(NOT_aa_NOT_bb_cc_NOT_ll_nn)
	distances[index] += a[index] + 2*d[index] 
	#20

	# #21
	# index = np.where(NOT_aa_NOT_bb_cc_NOT_ll_NOT_nn_hh)
	# #21

	#22
	index = np.where(NOT_aa_NOT_bb_cc_NOT_ll_NOT_nn_NOT_hh)
	distances[index] += -(d[index]**2)/a[index] 
	#22

	#23
	index = np.where(NOT_aa_NOT_bb_NOT_cc_oo)
	distances[index] +=  c[index] + 2*e[index]
	#23

	#24
	index = np.where(NOT_aa_NOT_bb_NOT_cc_NOT_oo_pp)
	distances[index] += a[index] + 2*d[index]
	#24

	#25
	index = np.where(NOT_aa_NOT_bb_NOT_cc_NOT_oo_NOT_pp)
	s_temp = numer[index]/denom[index]
	t_temp = 1-s_temp
	distances[index] += s_temp*(a[index]*s_temp + b[index]*t_temp+ 2*d[index] ) + \
					   t_temp*(b[index]*s_temp + c[index]*t_temp+ 2*e[index] ) 
	#25

	# now to just order then correctly and take the min for each point
	distances = distances.view(points.shape[0], faces.shape[0])
	min_distaces = torch.min(distances, dim = 1 )[0]
	# BEAUtiful

	return torch.mean(min_distaces)