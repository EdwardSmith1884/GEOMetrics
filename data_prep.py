import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(BASE_DIR + '/scripts/')
import urllib
from multiprocessing import Pool
import binvox_rw
import scipy.io as sio
import numpy as np
from tqdm import tqdm
from glob import glob
import random
import shutil
from PIL import Image
import argparse
from scipy import ndimage
from subprocess import call
import scipy.sparse as sp
import torch
import utils
from voxel import voxel2obj

parser = argparse.ArgumentParser(description='Dataset prep for image to 3D object super resolution')
parser.add_argument('-o','--object', default=['chair'], help='List of object classes to be used downloaded and converted.', nargs='+' )
parser.add_argument('-no','--num_objects', default=10000, help='number of objects to be converted', type = int)
args = parser.parse_args()


#labels for the union of the core shapenet classes and the ikea dataset classes
labels = {'04379243':'table','03211117':'monitor','04401088':'cellphone','04530566': 'watercraft',  '03001627' : 'chair','03636649' : 'lamp',  '03691459': 'speaker' ,  '02828884':'bench',
'02691156': 'plane', '02808440': 'bathtub',  '02871439': 'bookcase',
'02773838': 'bag', '02801938': 'basket', '02828884' : 'bench','02880940': 'bowl' ,
'02924116': 'bus', '02933112': 'cabinet', '02942699': 'camera', '02958343': 'car', '03207941': 'dishwasher',
'03337140': 'file', '03624134': 'knife', '03642806': 'laptop', '03710193': 'mailbox',
'03761084': 'microwave', '03928116': 'piano', '03938244':'pillow', '03948459': 'pistol', '04004475': 'printer',
'04099429': 'rocket', '04256520': 'sofa', '04554684': 'washer', '04090263': 'rifle'}




wanted_classes=[]
for l in labels:
	if labels[l] in args.object:
		wanted_classes.append(l)
print wanted_classes

debug_mode = False # change to make all of the called scripts print their errors and warnings
if debug_mode:
	io_redirect = ''
else:
	io_redirect = ' > /dev/null 2>&1'


# make data directories
if not os.path.exists('data/objects/'):
	os.makedirs('data/objects/')



# download .obj obect files
def download():
	with open('scripts/binvox_file_locations.txt','rb') as f: # location of all the binvoxes for shapenet's core classes
		content = f.readlines()

	# make data sub-directories for each class
	for s in wanted_classes:
		obj = 'data/objects/' + labels[s]+'/'
		if not os.path.exists(obj):
			os.makedirs(obj)


	# search object for correct object classes
	binvox_urls = []
	obj_urls = []
	for file in content:
		current_class = file.split('/')
		if current_class[1] in wanted_classes:
			if '_' in current_class[3]: continue
			if 'presolid' in current_class[3]: continue
			obj_urls.append(['http://shapenet.cs.stanford.edu/shapenet/obj-zip/ShapeNetCore.v1/'+file.split('/')[1]+'/'+file.split('/')[2]+'/model.obj', 'data/objects/'+labels[current_class[1]]+ '/'+ current_class[2]+'.obj'])

	# get randomized sample from each object class of correct size
	random.shuffle(obj_urls)
	final_urls = []
	dictionary = {}
	for o in obj_urls:
		obj_class = o[1].split('/')[-2]
		if obj_class in dictionary:
			dictionary[obj_class] += 1
			if dictionary[obj_class]> args.num_objects:
				continue
		else:
			dictionary[obj_class] = 1
		final_urls.append(o)

	# parallel downloading of object .obj files
	pool = Pool()
	pool.map(down, final_urls)


def process_mtl():
	import requests
	from bs4 import BeautifulSoup
	location = 'http://shapenet.cs.stanford.edu/shapenet/obj-zip/ShapeNetCore.v1/'
	for s in wanted_classes:
		files = glob('data/objects/' + labels[s]+'/*.obj')
		commands = []
		for f in tqdm(files):
			file = f.split('/')[-1][:-4]
			if not os.path.exists('data/objects/' + labels[s]+'/' + file + '/images/'):
				os.makedirs('data/objects/' + labels[s]+'/' + file + '/images/')
			if not os.path.exists('data/objects/' + labels[s]+'/' + file +  '/' + file + '/'):
				os.makedirs('data/objects/' + labels[s]+'/' + file +  '/' + file + '/')



			shutil.move(f,'data/objects/' + labels[s]+'/' + file + '/' + f.split('/')[-1])
			commands.append([location+s+'/'+file+'/model.mtl', 'data/objects/' + labels[s]+'/' + file + '/model.mtl'])

			soup = BeautifulSoup(requests.get(location+s+'/'+file+'/images/').text,  "html5lib")
			for a in soup.find_all('a', href=True):
				if 'textu' in a['href']:
					commands.append([location+s+'/'+file+'/images/'+a['href'], 'data/objects/' + labels[s]+'/' + file + '/images/'+ a['href'] ])

			soup = BeautifulSoup(requests.get(location+s+'/'+file+ '/' + file + '/').text,  "html5lib")
			for a in soup.find_all('a', href=True):
				if 'jpg' in a['href']  or 'png' in a['href']:
					commands.append([location+s+'/'+file+  '/' + file + '/'+a['href'], 'data/objects/' + labels[s]+'/' + file + '/'+ file +'/'+ a['href'] ])

			if len(commands) == 100:
				pool = Pool()
				pool.map(down, commands)
				commands = []

		pool = Pool()
		pool.map(down, commands)



# this take object files and makes then a more managable size
# this is only done for training the latent loss
# it makes it far quicker to load the object during training
# I also belive that by haveing a 'uniform' size it will be easier to learn
# never confirmed this though
def manage_objects():
	for s in wanted_classes:
			# final all downloaded objects from the class
			objs = glob('data/objects/' + labels[s]+'/*/*.obj')
			location_meshinfo = 'data/mesh_info/' + labels[s]+'/'
			location_obj = 'data/managable_objects/' + labels[s]+'/'
			if not os.path.exists(location_meshinfo):
				os.makedirs(location_meshinfo)
			if not os.path.exists(location_obj):
				os.makedirs(location_obj)
			l = 0
			commands = []
			# operate on each object
			for o in tqdm(objs):
				name = o.split('/')[-1][:-4]
				file_name_mesh =  location_meshinfo+ name
				file_name_new_obj = location_obj + name + '.obj'
				cmd = 'blender scripts/manage.blend -b -P scripts/blender_convert.py -- %s %s %s' %( o, file_name_mesh,file_name_new_obj )
				commands.append(cmd)
				# run in parallel for speed
				if l%10 == 9:
					pool = Pool()
					pool.map(call, commands)
					pool.close()
					pool.join()
					commands = []
				l+=1
			pool = Pool()
			pool.map(call, commands)
			pool.close()
			pool.join()
			commands = []


# converts obj files to binvox, is an intermediary for voxel computation
def binvox():
	for s in wanted_classes:
		dirs = glob('data/managable_objects/' + labels[s]+'/*.obj')
		commands =[]
		count = 0
		for d in tqdm(dirs):
			command = 'scripts/binvox ' + d  + ' -d ' + str(32)+ ' -pb -cb -c -e'   # this executable can be found at http://www.patrickmin.com/binvox/ ,
			# -d x idicates resoltuion will be x by x by x , -pb is to stop the visualization, the rest of the commnads are to help make the object water tight
			commands.append(command)
			if count %20 == 0  and count != 0:
				pool = Pool()
				pool.map(call, commands)
				pool.close()
				pool.join()
				commands = []
			count +=1
		pool = Pool()
		pool.map(call, commands)
		pool.close()
		pool.join()


# converts binvox files to voxel files
def convert_bin():
	for s in wanted_classes:
		directory = 'data/voxels/'+labels[s] +'/'
		# find all binvoxes
		models  = glob('data/managable_objects/'+labels[s]+'/*.binvox')
		if not os.path.exists(directory):
			os.makedirs(directory)
		for m in tqdm(models):
			with open(m, 'rb') as f:
				try:
					model = binvox_rw.read_as_3d_array(f).data
				except ValueError:
					continue

			# remove internals from models
			# I think this makes it easier to learn
			positions = np.where(model != 0 )
			new_mod = np.zeros(model.shape)
			for i, j, k in zip(*positions):
				# identifies if current voxel has an exposed face
				if np.sum(model[i-1:i+2, j-1:j+2, k-1:k+2]) < 27:
					new_mod[i,j,k] = 1
			# save as np array
			sio.savemat(directory +m.split('/')[-1][:-7], {'model': new_mod.astype(np.uint8)})

# we render each object into 24 images
# the imagesa are also split into training, validaion and test sets
def render():
	for s in wanted_classes:

		train_dir = 'data/images/'+labels[s]+ '/train/'
		valid_dir = 'data/images/'+labels[s]+ '/valid/'
		test_dir = 'data/images/'+labels[s]+ '/test/'
		if not os.path.exists(train_dir):
			os.makedirs(train_dir)
		if not os.path.exists(valid_dir):
			os.makedirs(valid_dir)
		if not os.path.exists(test_dir):
			os.makedirs(test_dir)
		Model_dir = 'data/objects/'+labels[s]+ '/'
		models = glob(Model_dir+'/*/*.obj')
		l=0
		commands = []

		valid_spot = int(len(models) * .7)
		test_spot = int(len(models) * .8)
		img_dir = train_dir
		for model in tqdm(models):
			if l == valid_spot:
				img_dir = valid_dir
			elif l == test_spot:
				img_dir = test_dir

			model_name = model.split('/')[-1].split('.')[0]

			target = img_dir  + model_name + '/rendering/'

			if not os.path.exists(target):
				os.makedirs(target)


			python_cmd = 'blender scripts/blank.blend -b -P scripts/blender_render.py -- %s %s %s' %(24, model, target)
			commands.append(python_cmd)

			if l%50 == 49:

				pool = Pool()
				pool.map(call, commands)
				pool.close()
				pool.join()
				commands = []

			l+=1
		pool = Pool()
		pool.map(call, commands)
		pool.close()
		pool.join()
		commands = []








# these are two simple functions for parallel processing
# down() downloads , and call() calls functions
def down(url):
	urllib.urlretrieve(url[0], url[1])
def call(command):
	os.system('%s %s' % (command, io_redirect))




# calculates the surface of an object
def surface_computation(data):
	dim = data.shape[0]

	a,b,c = np.where(data == 1)
	large = int(dim *1.5)
	big_list = [[[[-1,large]for j in range(dim)] for i in range(dim)] for k in range(3)]
	# over the whole object extract for each face the first and last occurance of a voxel at each pixel
	# we take highest for convinience
	for i,j,k in zip(a,b,c):
		big_list[0][i][j][0] = (max(k,big_list[0][i][j][0]))
		big_list[0][i][j][1] = (min(k,big_list[0][i][j][1]))
		big_list[1][i][k][0] = (max(j,big_list[1][i][k][0]))
		big_list[1][i][k][1] = (min(j,big_list[1][i][k][1]))
		big_list[2][j][k][0] = (max(i,big_list[2][j][k][0]))
		big_list[2][j][k][1] = (min(i,big_list[2][j][k][1]))
	faces = np.zeros((6,dim,dim)) # will hold odms
	for i in range(dim):
		for j in range(dim):
			faces[0,i,j] =   dim -1 - big_list[0][i][j][0]         if    big_list[0][i][j][0]   > -1 else dim
			# we subtract from the (dimension -1) as we computed the last occurance, instead of the first for half of the faces
			faces[1,i,j] =   big_list[0][i][j][1]        		   if    big_list[0][i][j][1]   < large else dim
			faces[2,i,j] =   dim -1 - big_list[1][i][j][0]         if    big_list[1][i][j][0]   > -1 else dim
			faces[3,i,j] =   big_list[1][i][j][1]        		   if    big_list[1][i][j][1]   < large else dim
			faces[4,i,j] =   dim -1 - big_list[2][i][j][0]         if    big_list[2][i][j][0]   > -1 else dim
			faces[5,i,j] =   big_list[2][i][j][1]         		   if    big_list[2][i][j][1]   < large else dim
	return faces






def calc_surface():
	dims = 128
	# first render as voxel array at high resolution
	for s in wanted_classes:
		dirs = glob('data/objects/' + labels[s]+'/*/*.obj')

		commands =[]
		count = 0
		for d in tqdm(dirs):
			command = 'scripts/binvox ' + d  + ' -d ' + str(dims)+ ' -pb -cb -c -e'   # this executable can be found at http://www.patrickmin.com/binvox/ ,
			# -d x idicates resoltuion will be x by x by x , -pb is to stop the visualization, the rest of the commnads are to help make the object water tight
			commands.append(command)
			if count %20 == 0  and count != 0:
				pool = Pool()
				pool.map(call, commands)
				pool.close()
				pool.join()
				commands = []
			count +=1
		pool = Pool()
		pool.map(call, commands)
		pool.close()
		pool.join()


	for s in wanted_classes:

		models  = glob('data/objects/'+labels[s]+'/*/*.binvox')
		location = 'data/surfaces/'+labels[s] +'/'
		if not os.path.exists(location):
			os.makedirs(location)

		for m in tqdm(models):
			if '_' in m: continue
			with open(m, 'rb') as f:
				try:
					model = binvox_rw.read_as_3d_array(f).data
				except ValueError:
					continue
			faces = surface_computation(model)
			# calculate low resolution version, to make watertight
			high, low = dims, 32
			down = high // low
			a,b,c = np.where(model==1)
			low_model = np.zeros((low,low,low))
			for x,y,z in zip(a,b,c):
					low_model[ x//down, y//down, z//down] =1
			# fill internals
			low_model[ndimage.binary_fill_holes(low_model)] = 1

			# obtain surface projections
			corrected = np.zeros((high,high,high))
			for i in range(low):
				for j in range(low):
					for k in range(low):
						corrected[i*down: (i+1)*down, j*down:(j+1)*down, k*down:(k+1)*down] = low_model[i,j,k]
			# carve away from low res model
			for i in range(high):
				for j in range(high):
					if faces[0,i,j] <high:
						corrected[i,j,int((high - faces[0,i,j])):high]=0
					else:
						corrected[i,j,:] =0

					if faces[1,i,j] <high:
						corrected[i,j,0:int(faces[1,i,j])]=0
					else:
						corrected[i,j,:] =0

					if faces[2,i,j] <high:
						corrected[i,int((high - faces[2,i,j])):high, j] =0
					else:
						corrected[i,:,j] =0

					if faces[3,i,j] <high:
						corrected[i,0:int(faces[3,i,j]), j] =0
					else:
						corrected[i,:,j] =0

					if faces[4,i,j] <high:
						corrected[int((high - faces[4,i,j])):high,i,j] =0
					else:
						corrected[:,i,j] =0

					if faces[5,i,j] <high:
						corrected[0:int(faces[5,i,j]),i,j] =0
					else:
						corrected[:,i,j] =0

			corrected[ndimage.binary_fill_holes(corrected)] = 1
			positions = np.where(corrected != 0 )
			new_mod = np.zeros(corrected.shape)
			points = []
			# get only surface objects
			for i, j, k in zip(*positions):
				# identifies if current voxel has an exposed face
				if np.sum(corrected[i-1:i+2, j-1:j+2, k-1:k+2]) < 27:
					points.append([i,j,k])
			voxel_points = np.array(points).astype(float)

			# sample from surface
			dist_uni = torch.distributions.Uniform(torch.tensor([0.0]).cuda(), torch.tensor([1.0]).cuda())
			from scipy import stats
			from pycpd import rigid_registration

			obj = m[:-7] + '.obj'
			try:
				obj = utils.ObjLoader(obj)
			except ValueError:
				continue
			verts = np.array(obj.vertices)
			verts = torch.FloatTensor(verts).cuda()
			faces = np.array(obj.faces) -1
			faces =  torch.LongTensor(faces).cuda()

			x1,x2,x3 = torch.split(torch.index_select(verts, 0,faces[:,0]) - torch.index_select(verts, 0,faces[:,1]), 1, dim = 1)
			y1,y2,y3 = torch.split(torch.index_select(verts, 0,faces[:,1]) - torch.index_select(verts, 0,faces[:,2]), 1, dim = 1)

			xs = torch.index_select(verts, 0,faces[:,0])
			ys = torch.index_select(verts, 0,faces[:,1])
			zs = torch.index_select(verts, 0,faces[:,2])
			num = faces.shape[0]

			u = torch.sqrt(dist_uni.sample_n(num))
			v = dist_uni.sample_n(num)
			mesh_points = (1- u)*xs + (u*(1-v))*ys + u*v*zs

			voxel_points = np.array(voxel_points)
			mesh_points = mesh_points.data.cpu().numpy()


			# make computed surface be same size as origional obejct
			xx = np.amax(mesh_points[:,0]) - np.amin(mesh_points[:,0])
			xx_v = np.amax(voxel_points[:,0]) - np.amin(voxel_points[:,0])
			x_diff = xx/xx_v

			yy = np.amax(mesh_points[:,1]) - np.amin(mesh_points[:,1])
			yy_v = np.amax(voxel_points[:,1]) - np.amin(voxel_points[:,1])
			y_diff = yy/yy_v

			zz = np.amax(mesh_points[:,2]) - np.amin(mesh_points[:,2])
			zz_v = np.amax(voxel_points[:,2]) - np.amin(voxel_points[:,2])
			z_diff = zz/zz_v
			voxel_points*=[x_diff, y_diff, z_diff]

			xx = np.amax(mesh_points[:,0])
			xx_v = np.amax(voxel_points[:,0])
			x_diff = xx-xx_v

			yy = np.amax(mesh_points[:,1])
			yy_v = np.amax(voxel_points[:,1])
			y_diff = yy-yy_v

			zz = np.amax(mesh_points[:,2])
			zz_v = np.amax(voxel_points[:,2])
			z_diff = zz-zz_v

			voxel_points+=[x_diff, y_diff, z_diff]

			sio.savemat(location + m.split('/')[-1][:-7] , {'points': voxel_points})




print '------------'
print'downloading'
download()
process_mtl()
print '------------'
print'changing meshes'
manage_objects()
print '------------'
print'making binvoxes'
binvox()
print '------------'
print'making voxels'
convert_bin()
print '------------'
print'rendering'
render()
print '------------'
print 'calculate surface points'
calc_surface()
print '---------------'

print'finished eratin'
