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
from tqdm import tqdm 
from voxel import *
from torch.utils.data import DataLoader
from PIL import Image



from torchvision.transforms import Normalize as norm 
from torchvision import transforms
preprocess = transforms.Compose([
   transforms.Scale((224, 224)), 
   transforms.ToTensor()
])

from chamfer_distance import ChamferDistance
chamfer_dist = ChamferDistance()

from tri_distance import TriDistance
tri_dist = TriDistance()




# loads the initial mesh and stores vertex, face, and adjacency matrix information
def load_initial( obj='386.obj'):
	# load obj file
	obj = ObjLoader(obj)
	labels = np.array(obj.vertices) 
	features = torch.FloatTensor(labels).cuda()
	faces = torch.LongTensor(np.array(obj.faces) -1).cuda()

	points = torch.rand([1000, 3]).cuda() - .5
	verts = features.clone()
	tri1 =  torch.index_select(verts, 0,faces[:,0]).unsqueeze(0)
	tri2 =  torch.index_select(verts, 0,faces[:,1]).unsqueeze(0)
	tri3 =  torch.index_select(verts, 0,faces[:,2]).unsqueeze(0)
	
	# get adjacency matrix infomation
	adj_info = adj_init(faces)

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




# normalizes symetric, binary adj matrix such that sum of each row is 1 
def normalize_adj(mx):
	rowsum = mx.sum(1)
	r_inv = (1./rowsum).view(-1)
	r_inv[r_inv != r_inv] = 0.
	mx = torch.mm(torch.eye(r_inv.shape[0]).to(mx.device)*r_inv, mx)
	return mx


def adj_init(faces):
	adj = calc_adj(faces)
	adj_orig = adj.clone()
	adj = normalize_adj(adj)
	adj_info = {}

	adj_info['adj'] = adj 
	adj_info['adj_orig'] = adj_orig
	adj_info['faces'] = faces
	return adj_info 
			
def calc_adj(faces): 
	v1 = faces[:, 0]
	v2 = faces[:, 1]
	v3 = faces[:, 2]
	num_verts = int(faces.max())
	adj = torch.eye(num_verts+1).to(faces.device)
	
	adj[(v1, v2)] = 1 
	adj[(v1, v3)] = 1 

	adj[(v2, v1)] = 1
	adj[(v2, v3)] = 1 

	adj[(v3, v1)] = 1
	adj[(v3, v2)] = 1 

	return adj
 


# loader for GEOMetrics
class Mesh_loader(object):
	def __init__(self, Img_location, Mesh_loction, Sample_location, set_type='train', num= 23, sample_num= 3000):

		# initialization of data locations 
		self.Mesh_loction = Mesh_loction
		self.Img_location = Img_location
		if '*' in self.Img_location: 
			self.Img_location = self.Img_location[:-1]
		else: 
			self.Img_location = self.Img_location[:-len(Img_location.split('/')[-1])]
		self.Sample_location = Sample_location
		self.num = num
		self.sample_num = sample_num
		self.set_type = set_type
		names = [[f.split('/')[-2],f.split('/')[-4]]  for f in glob((Img_location + '/{}/*/').format(set_type)) ]
		self.names = []

		for n in tqdm(names):
			# only examples which have both images and ground-truth sampled points are loaded
			if os.path.exists(self.Sample_location + n[1] +  '/' + n[0] ):
				self.names.append(n)	
		print (f'The number of {set_type} set objects found : {len(self.names)}')

	def __len__(self): 
		return len(self.names)
	
	def __getitem__(self, index): 
		
		obj, obj_class = self.names[index]
		data = {}

		mesh_obj = 'data/latent/' + obj + '_latent.npy'

		data['names'] = obj
		data['class'] = obj_class
		try: 
			data['latent'] = torch.FloatTensor(np.load(mesh_obj))
			data['encode'] = torch.FloatTensor([1])
		except: 
			data['latent'] = torch.FloatTensor(np.zeros(50))
			data['encode'] = torch.FloatTensor([0])


		# load sampled ground truth points
		samples = sio.loadmat(self.Sample_location + obj_class + '/' + obj )['points'] 
		np.random.shuffle(samples)
		data['samples'] = torch.FloatTensor(samples[:self.sample_num])
		
		#load images
		if self.num > 0: 
			num = random.randint(1,self.num)
		else: 
			num = 0   
		str_num  = str(num).zfill(2)
		
		img = (Image.open(self.Img_location + obj_class + '/' + self.set_type + '/' + obj +  '/rendering/' + str_num + '.png'))
		img = preprocess(img)
		img_info = np.genfromtxt(self.Img_location + obj_class + '/' + self.set_type + '/' + obj + '/rendering/rendering_metadata.txt', delimiter=' ')[num]
		img_info = [img_info[0] , img_info[1], img_info[3]]
		
		data['imgs'] = torch.FloatTensor(img)
		data['img_info'] =  torch.FloatTensor(np.array(img_info))
		
		return data

	def collate(self, batch):
	
		data = {}
		data['names'] = [ item['names'] for item in batch]
		data['class'] = [ item['class'] for item in batch]
		data['encode'] = torch.cat([item['encode'] for item in batch])
		data['latent'] = torch.cat([item['latent'].unsqueeze(0) for item in batch])
		data['img_info'] = torch.cat([item['img_info'].unsqueeze(0) for item in batch])
		data['samples'] = torch.cat([item['samples'].unsqueeze(0) for item in batch])
		data['imgs'] = torch.cat([item['imgs'].unsqueeze(0) for item in batch])
		
		

		return data



# loader to Auto-Encoder
class Voxel_loader(object):
	def __init__(self, Img_location, Mesh_loction, Voxel_location,  set_type='train'):

		self.Voxel_location = Voxel_location
		self.Mesh_loction = Mesh_loction
		self.Img_location = Img_location
		names = [[f.split('/')[-2],f.split('/')[-4]]  for f in glob((Img_location + '/{}/*/').format(set_type)) ]


		self.names = []

		for n in tqdm(names): 
	
			if os.path.exists(self.Mesh_loction + n[1] +  '/' + n[0]  +'.mat'):
				self.names.append(n)

		print (f'The number of {set_type} set objects found : {len(self.names)}')

	def __len__(self): 
		return len(self.names)

	def __getitem__(self, index): 

		data = {}
		# select object
		obj, obj_class = self.names[index]
		data['names'] = obj

		#load voxels 
		voxels = sio.loadmat(self.Voxel_location + obj_class + '/' + obj )['model']
		data['voxels'] = torch.FloatTensor(voxels)

		#load mesh 
		mesh_info  =  sio.loadmat(self.Mesh_loction + obj_class + '/' + obj  +  '.mat')
		verts = torch.FloatTensor(mesh_info['verts'])
		faces = torch.LongTensor(mesh_info['faces'])
		adj = calc_adj(faces)
		adj = normalize_adj( torch.FloatTensor(adj))
		data['verts'] = verts
		data['faces'] = faces 
		data['adj'] = adj

		return data


	def collate(self, batch):
	
		data = {}
		data['names'] = [ item['names'] for item in batch]
		data['voxels'] = torch.cat([item['voxels'].unsqueeze(0) for item in batch])
		data['verts'] = [item['verts']for item in batch]
		data['faces'] = [item['faces']for item in batch]
		data['adjs'] = [item['adj'] for item in batch]

		return data


def unit(v):
	norm = np.linalg.norm(v)
	if norm == 0:
		return v
	return v / norm



def render_mesh( verts, faces):
	mesh2obj( verts, faces +1) 

def batch_camera_info(param):
	
	theta = (np.pi*param[:,0]/180.) % 360.
	phi = (np.pi*param[:,1]/180.) % 360.

	camY = param[:,2]*torch.sin(phi)
	temp = param[:,2]*torch.cos(phi)
	
	camX = temp * torch.cos(theta)    
	camZ = temp * torch.sin(theta)    

	cam_pos = torch.cat((camX.unsqueeze(1), camY.unsqueeze(1), camZ.unsqueeze(1)), dim = 1 )    
	
	axisZ = cam_pos.clone()
	axisY = torch.FloatTensor([0,1,0]).cuda().unsqueeze(0).expand(axisZ.shape[0], 3)

	axisX = torch.cross(axisY, axisZ)
	axisY = torch.cross(axisZ, axisX)



	axisX = axisX / torch.sqrt(torch.sum(axisX**2, dim = 1)).unsqueeze(-1)
	axisY = axisY / torch.sqrt(torch.sum(axisY**2, dim = 1)).unsqueeze(-1) 
	axisZ = axisZ / torch.sqrt(torch.sum(axisZ**2, dim = 1)).unsqueeze(-1) 

	cam_mat = torch.cat((axisX.unsqueeze(1), axisY.unsqueeze(1), axisZ.unsqueeze(1)), dim = 1 )
	
	return cam_mat, cam_pos


def batched_pooling(blocks, verts_pos, img_info):
	# convert vertex positions to x,y coordinates in the image, scaled to fractions of image dimension 

	cam_mat, cam_pos = batch_camera_info( img_info)
	
	A = ((verts_pos*.57)-cam_pos.unsqueeze(1))
	B = (cam_mat.permute(0,2,1))

	pt_trans = torch.matmul(A,B)
	X = pt_trans[:,:,0]
	Y = pt_trans[:,:,1]
	Z = pt_trans[:,:,2]
	F = 248

	h = (-Y)/(-Z)*F + 224/2.0
	w = X/(-Z)*F + 224/2.0
	xs = h / 223. 
	ys = w /223.

	full_features = None 
	batch_size = verts_pos.shape[0]
	for block in blocks: 
		# scale coordinated to block dimensions/resolution 
		dim = block.shape[-1]

		cur_xs = torch.clamp(xs * dim, 0, dim-1)
		cur_ys = torch.clamp(ys * dim, 0, dim-1) 
	
		# this is basically bilinear interpolation of the 4 closest feature vectors to where the vertex lands in the block 
		# https://en.wikipedia.org/wiki/Bilinear_interpolation
		x1s, y1s, x2s, y2s = torch.floor(cur_xs), torch.floor(cur_ys), torch.ceil(cur_xs), torch.ceil(cur_ys)
		A = x2s -cur_xs
		B = cur_xs-x1s
		G = y2s-cur_ys
		H = cur_ys-y1s
			
		x1s = x1s.type(torch.cuda.LongTensor)
		y1s = y1s.type(torch.cuda.LongTensor)
		x2s = x2s.type(torch.cuda.LongTensor)
		y2s = y2s.type(torch.cuda.LongTensor) 

		flat_block = block.permute(1,0,2,3).contiguous().view(block.shape[1], -1)
		block_upper = torch.arange(0, verts_pos.shape[0]).cuda().unsqueeze(-1).expand(batch_size, verts_pos.shape[1] )
		
		selection = ((block_upper * dim * dim) + (x1s * dim) + y1s).view(-1)
		C = torch.index_select(flat_block, 1, selection)
		C = C.view(-1,batch_size,verts_pos.shape[1]).permute(1,0,2)
		selection = ((block_upper * dim * dim) + (x1s * dim) + y2s).view(-1)
		D = torch.index_select(flat_block, 1, selection)
		D = D .view(-1,batch_size,verts_pos.shape[1]).permute(1,0,2)
		selection = ((block_upper * dim * dim) + (x2s * dim) + y1s).view(-1)
		E = torch.index_select(flat_block, 1, selection)
		E = E.view(-1,batch_size,verts_pos.shape[1]).permute(1,0,2)
		selection = ((block_upper * dim * dim) + (x2s * dim) + y2s).view(-1) 
		F = torch.index_select(flat_block, 1, selection)
		F = F.view(-1,batch_size,verts_pos.shape[1]).permute(1,0,2)
	
		
		section1 = A.unsqueeze(1)*C*G.unsqueeze(1)
		section2 = H.unsqueeze(1)*D*A.unsqueeze(1)
		section3 = G.unsqueeze(1)*E*B.unsqueeze(1)
		section4 = B.unsqueeze(1)*F*H.unsqueeze(1)
		

		features = (section1 + section2 + section3 + section4)
		features = features.permute(0,2,1)
	

	
		if full_features is None: full_features = features
		else: full_features = torch.cat((full_features, features), dim = 2)
			
	
	return full_features


 
def batch_point_to_point(pred_vert, adj_info, gt_points, num = 1000, f1 = False ):
	# grab the faces still in use 
	batch_size = pred_vert.shape[0]

	# sample from faces and calculate pairs

	pred_points = batch_sample(pred_vert, adj_info['faces'], num =num)

	
	id_p, id_g = chamfer_dist( gt_points, pred_points)

	# select pairs and calculate chamfer distance 

	pred_points = pred_points.view(-1,3)
	gt_points = gt_points.contiguous().view(-1,3)
	
	points_range = num*torch.arange(0, batch_size).cuda().unsqueeze(-1).expand(batch_size,num)
	id_p = (id_p.long() + points_range).view(-1)
	id_g = (id_g.long() + points_range).view(-1)

	pred_counters = torch.index_select(pred_points, 0, id_p)
	gt_counters =  torch.index_select(gt_points, 0, id_g)

	dist_1 = torch.mean(torch.sum((gt_counters - pred_points)**2, dim  = 1))
	dist_2 = torch.mean(torch.sum((pred_counters - gt_points)**2, dim  = 1))
	

	loss = (dist_1 + dist_2) * 3000 

	

	if f1: 
		dist_to_pred = torch.sqrt(torch.sum((.57*pred_counters - .57*gt_points)**2, dim  = 1 )).view(batch_size, -1)
		dist_to_gt = torch.sqrt(torch.sum((.57*gt_counters - .57*pred_points)**2, dim = 1)).view(batch_size, -1)

		f_score = 0
		for i in range(dist_to_pred.shape[0]): 
			fn = float(torch.where(dist_to_pred[i] > 1e-2)[0].shape[0])
			fp =  float(torch.where(dist_to_gt[i] > 1e-2)[0].shape[0])
			tp =  float(torch.where(dist_to_gt[i] <= 1e-2)[0].shape[0])
			precision = tp/(tp+fp)
			recall = tp/(tp + fn )
			if precision + recall == 0 : continue 
			f_score +=  2*(precision * recall)/(precision + recall)
		f_score = f_score / (batch_size)
		return loss, f_score
	else: 
		return loss
		

def batch_point_to_surface(pred_vert, adj_info, gt_points, num = 1000, f1 = False ):
	# grab the faces still in use 
	batch_size = pred_vert.shape[0]

	# sample from faces and calculate pairs

	pred_points = batch_sample(pred_vert, adj_info['faces'], num =num)

	#####
	# ptp
	##### 
	id_p, id_g = chamfer_dist( gt_points, pred_points)

	# select pairs and calculate chamfer distance 
	points_range = pred_points.shape[1]*torch.arange(0, batch_size).cuda().unsqueeze(-1).expand(batch_size,gt_points.shape[1])
	id_p = (id_p.long() + points_range).view(-1)
	pred_counters = torch.index_select(pred_points.view(-1,3), 0, id_p)

	points_range = gt_points.shape[1]*torch.arange(0, batch_size).cuda().unsqueeze(-1).expand(batch_size,pred_points.shape[1])
	id_g = (id_g.long() + points_range).view(-1)
	gt_counters =  torch.index_select(gt_points.contiguous().view(-1,3), 0, id_g)

	dist_1 = torch.mean(torch.sum((gt_counters - pred_points.view(-1,3))**2, dim  = 1))
	
	#####
	# ptp
	#####
	tri1 =  torch.index_select(pred_vert, 1,adj_info['faces'][:,0])
	tri2 =  torch.index_select(pred_vert, 1,adj_info['faces'][:,1])
	tri3 =  torch.index_select(pred_vert, 1,adj_info['faces'][:,2])
	_, point_options, index = tri_dist(gt_points, tri1, tri2, tri3)

	tri_sets = [tri1, tri2, tri3]
	point_options = point_options.view(-1)
	points_range = tri1.shape[1]*torch.arange(0, batch_size).cuda().unsqueeze(-1).expand(batch_size,gt_points.shape[1])
	index = (index.long() + points_range).view(-1)
	
	for i,t in enumerate(tri_sets):
		t = t.view(-1,3)
		tri_sets[i] = torch.index_select(t, 0, index)

	dist_2 = calc_point_to_line(gt_points.contiguous().view(-1,3), tri_sets, point_options)
	ratio = dist_1.data.cpu().numpy()/ (dist_2.data.cpu().numpy())

	loss = (dist_1 + dist_2) * 3000

	####
	# f1
	####
	if f1: 
		dist_to_pred = torch.sqrt(torch.sum((.57*pred_counters - .57*gt_points.contiguous().view(-1,3))**2, dim  = 1 )).view(batch_size, -1)
		dist_to_gt = torch.sqrt(torch.sum((.57*gt_counters - .57*pred_points.view(-1,3))**2, dim = 1)).view(batch_size, -1)

		f_score = 0
		for i in range(dist_to_pred.shape[0]): 
			fn = float(torch.where(dist_to_pred[i] > 1e-2)[0].shape[0])
			fp =  float(torch.where(dist_to_gt[i] > 1e-2)[0].shape[0])
			tp =  float(torch.where(dist_to_gt[i] <= 1e-2)[0].shape[0])
			precision = tp/(tp+fp)
			recall = tp/(tp + fn )
			if precision + recall == 0 : continue 
			f_score +=  2*(precision * recall)/(precision + recall)
		f_score = f_score / (batch_size)
		return loss, f_score
	else: 
		return loss

def calc_point_to_line(p, triangles, point_options): 
	a, b, c = triangles
	counter_p = torch.zeros(p.shape).cuda()
	
	EdgeAb = edge( a, b )
	EdgeBc = edge( b, c )
	EdgeCa = edge( c, a )

	uab = EdgeAb.Project( p )
	uca = EdgeCa.Project( p )
	ubc = EdgeBc.Project( p )

	TriNorm = torch.cross(a - b, a - c)
	TriPlane = Plane(EdgeAb.A, TriNorm);

	# type 1 
	cond = (point_options == 1)
	counter_p[cond] = EdgeAb.A[cond]

	# type 2 
	cond = (point_options == 2)
	counter_p[cond] = EdgeBc.A[cond]

	# type 3 
	cond = (point_options == 3)
	counter_p[cond] = EdgeCa.A[cond]

	# type 4 
	cond = (point_options == 4)
	counter_p[cond] = EdgeAb.PointAt( uab )[cond]

	# type 5
	cond = (point_options == 5)
	counter_p[cond] = EdgeBc.PointAt( ubc )[cond]

	# type 6
	cond = (point_options == 6)
	counter_p[cond] = EdgeCa.PointAt( uca )[cond]

	# type 0
	cond = (point_options == 0)
	counter_p[cond] = TriPlane.Project( p )[cond]
	
	distances = torch.mean(torch.sum((counter_p - p)**2, dim = -1))
	return distances


class edge(): 
	def __init__(self, a, b):
		self.A = a.clone() 
		self.B = b.clone() 
		self.Delta = b - a 
	
	def PointAt(self, t): 
		return self.A + t.unsqueeze(-1)* self.Delta
	
	def LengthSquared(self): 
		return torch.sum(self.Delta**2, dim = -1)
	
	def Project(self, p): 
		vec = p - self.A 
		vx, vy, vz = vec[:,0], vec[:,1], vec[:,2]
		nx, ny, nz = self.Delta[:,0], self.Delta[:,1], self.Delta[:,2]
		vec = (vx*nx + vy*ny + vz*nz)
		return vec / self.LengthSquared() 

class Plane(): 
	def __init__(self, point, direction): 
		self.Point = point.clone()
		self.Direction = direction / torch.sqrt(torch.sum(direction**2, dim=-1)).unsqueeze(-1) 
	
	def IsAbove(self, q): 
		return torch.bmm(q.unsqueeze(1), self.Point.unsqueeze(-1)).view(-1) <=0 
	
	def Project(self, point): 
		orig = self.Point 
		v = point -orig 
		vx, vy, vz = v[:,0], v[:,1], v[:,2]
		nx, ny, nz = self.Direction[:,0], self.Direction[:,1], self.Direction[:,2]
		dist = (vx*nx + vy*ny + vz*nz).unsqueeze(-1)
		projected_point = point - dist*self.Direction
		return projected_point
 
	
def batch_sample(verts, faces, num=10000): 

	dist_uni = torch.distributions.Uniform(torch.tensor([0.0]).cuda(), torch.tensor([1.0]).cuda())
	batch_size = verts.shape[0]
	# calculate area of each face 

	x1,x2,x3 = torch.split(torch.index_select(verts, 1,faces[:,0]) - torch.index_select(verts, 1,faces[:,1]), 1, dim = -1)
	y1,y2,y3 = torch.split(torch.index_select(verts, 1,faces[:,1]) - torch.index_select(verts, 1,faces[:,2]), 1, dim = -1)
	a = (x2*y3-x3*y2)**2
	b = (x3*y1 - x1*y3)**2
	c = (x1*y2 - x2*y1)**2
	Areas = torch.sqrt(a+b+c)/2
	Areas = Areas.squeeze(-1) /  torch.sum(Areas, dim = 1) # percentage of each face w.r.t. full surface area 


	# define descrete distribution w.r.t. face area ratios caluclated 
	choices = None 
	for A in Areas: 
		
		if choices is None:
			choices = torch.multinomial(A, num, True)# list of faces to be sampled from 
		else: 
			choices = torch.cat((choices, torch.multinomial(A, num, True)))
	
	# from each face sample a point 
	select_faces = faces[choices].view(verts.shape[0],3,num)


	face_arange = verts.shape[1] * torch.arange(0, batch_size).cuda().unsqueeze(-1).expand(batch_size,num)
	select_faces = select_faces + face_arange.unsqueeze(1)

	select_faces = select_faces.view(-1,3)
	flat_verts = verts.view(-1, 3)
	
	xs = torch.index_select(flat_verts, 0, select_faces[:,0])
	ys = torch.index_select(flat_verts, 0, select_faces[:,1])
	zs = torch.index_select(flat_verts, 0, select_faces[:,2])
	u = torch.sqrt(dist_uni.sample_n(batch_size*num))
	v = dist_uni.sample_n(batch_size*num)

	points = (1- u)*xs + (u*(1-v))*ys + u*v*zs
	points = points.view(batch_size, num, 3)
	
	return points


def batch_calc_edge( verts, info): 

	faces = info['faces']
	# get vertex loccations of faces 
	p1 = torch.index_select(verts, 1,faces[:,0])
	p2 = torch.index_select(verts, 1,faces[:,1])
	p3 = torch.index_select(verts, 1,faces[:,2])

	# get edge lentgh 
	e1 = p2-p1
	e2 = p3-p1
	e3 = p2-p3

	edge_length = (torch.sum(e1**2, -1).mean() + torch.sum(e2**2, -1).mean() + torch.sum(e3**2, -1).mean())/3.

	return edge_length


def batch_get_lap_info(positions, adj_info):  
	orig = adj_info['adj_orig']
	nieghbor_sum = torch.matmul(orig,positions ) - positions
	degrees = torch.sum(orig, dim = 1 ) - 1
	degree_scaler = (1./degrees).view(-1,1)

	nieghbor_sum =nieghbor_sum*degree_scaler
	lap = positions - nieghbor_sum
	return lap  


