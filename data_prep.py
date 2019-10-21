import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(BASE_DIR + '/scripts/')
import urllib.request 
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
import torch
import utils
from time import time

parser = argparse.ArgumentParser(description='Dataset prep for image to 3D object super resolution')
parser.add_argument('-no','--num_objects', default=500, help='number of objects to be converted', type = int)
args = parser.parse_args()


#labels for the union of the core shapenet classes and the ikea dataset classes
labels = {'04379243':'table','03211117':'monitor','04401088':'cellphone','04530566': 'watercraft',  '03001627' : 'chair','03636649' : 'lamp',  '03691459': 'speaker' ,  '02828884':'bench',
'02691156': 'plane', '02808440': 'bathtub',  '02871439': 'bookcase',
'02773838': 'bag', '02801938': 'basket', '02828884' : 'bench','02880940': 'bowl' ,
'02924116': 'bus', '02933112': 'cabinet', '02942699': 'camera', '02958343': 'car', '03207941': 'dishwasher',
'03337140': 'file', '03624134': 'knife', '03642806': 'laptop', '03710193': 'mailbox',
'03761084': 'microwave', '03928116': 'piano', '03938244':'pillow', '03948459': 'pistol', '04004475': 'printer',
'04099429': 'rocket', '04256520': 'sofa', '04554684': 'washer', '04090263': 'rifle'}

objects = ['bench','cabinet','car','cellphone','chair','lamp','monitor','plane','rifle','sofa','speaker','table','watercraft']
wanted_classes=[]
for l in labels:
	if labels[l] in objects:
		wanted_classes.append(l)

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
		file = str(file)
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
	pool = Pool(processes=16)
	pbar = tqdm(pool.imap_unordered(down, final_urls), total=len(final_urls))
	pbar.set_description(f"Downloading Meshes")
	for _ in pbar:
		pass
	

# this take object files and makes then a more managable size
# this is only done for training the latent loss
# it makes it far quicker to load the object during training
def manage_objects():
	commands = []
	for s in wanted_classes:
		# final all downloaded objects from the class
		objs = glob('data/objects/' + labels[s]+'/*.obj')
		location_meshinfo = 'data/mesh_info/' + labels[s]+'/'
		location_obj = 'data/managable_objects/' + labels[s]+'/'
		if not os.path.exists(location_meshinfo):
			os.makedirs(location_meshinfo)
		if not os.path.exists(location_obj):
			os.makedirs(location_obj)
		l = 0

		for o in objs:
			name = o.split('/')[-1][:-4]
			file_name_mesh =  location_meshinfo+ name
			file_name_new_obj = location_obj + name + '.obj'
			cmd = 'blender scripts/manage.blend -b -P scripts/blender_convert.py -- %s %s %s' %( o, file_name_mesh,file_name_new_obj )
			commands.append(cmd)


	random.shuffle(commands)
	pool = Pool(processes=16)
	pbar = tqdm(pool.imap_unordered(call, commands), total=len(commands))
	pbar.set_description(f"Downscaling meshes")
	for _ in pbar:
		pass

	message = 'The blender commands failed. Please check why using the following command: '
	assert len(glob('data/managable_objects/' + labels[s]+'/*.obj')) > 0, message + cmd


# converts obj files to binvox, is an intermediary for voxel computation
def binvox():
	commands =[]
	for s in wanted_classes:
		dirs = glob('data/managable_objects/' + labels[s]+'/*.obj')
		
		count = 0
		for d in (dirs):
			command = 'scripts/binvox ' + d  + ' -d ' + str(32)+ ' -pb -cb -c -e'   # this executable can be found at http://www.patrickmin.com/binvox/ ,
			# -d x idicates resoltuion will be x by x by x , -pb is to stop the visualization, the rest of the commnads are to help make the object water tight
			commands.append(command)
		
	
	random.shuffle(commands)
	pool = Pool(processes=16)
	pbar = tqdm(pool.imap_unordered(call, commands), total=len(commands))
	pbar.set_description(f"converting meshes to small binvoxes")
	for _ in pbar:
		pass

	message = 'The binvox executable failed. Please check its permissions, and that it can run properly from the commandline using the following command: '
	assert len(glob('data/managable_objects/' + labels[s]+'/*.binvox')) > 0, message + command
		


# converts binvox files to voxel files
def convert_bin():
	models = []
	for s in wanted_classes:
		directory = 'data/voxels/'+labels[s] +'/'
		# find all binvoxes
		models  += glob('data/managable_objects/'+labels[s]+'/*.binvox')
		if not os.path.exists(directory):
			os.makedirs(directory)

	cur_class = ''
	random.shuffle(models)
	pbar = tqdm(models)
	pbar.set_description(f"Converting binvoxes to voxels")
	for m in pbar:
		with open(m, 'rb') as f:
			try:
				model = binvox_rw.read_as_3d_array(f).data
			except ValueError:
				continue
		directory = 'data/voxels/' + m.split('/')[-2] + '/'
			
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





# these are two simple functions for parallel processing
# down() downloads , and call() calls functions
def down(url):
	urllib.request.urlretrieve(url[0], url[1])
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
	models = []
	for s in wanted_classes:
		models += glob('data/objects/' + labels[s]+'/*.obj')

	commands =[]
	random.shuffle(models)
	for m in models:
		command = 'scripts/binvox ' + m  + ' -d ' + str(dims)+ ' -pb -cb -c -e'   # this executable can be found at http://www.patrickmin.com/binvox/ ,
		# -d x idicates resoltuion will be x by x by x , -pb is to stop the visualization, the rest of the commnads are to help make the object water tight
		commands.append(command)

	pool = Pool(processes=16)
	pbar = tqdm(pool.imap_unordered(call, commands), total=len(commands))
	pbar.set_description(f"Making large binvoxes from meshes")
	for _ in pbar:
		pass

	message = 'The binvox executable failed. Please check its permissions, and that it can run properly from the commandline using the following command: '
	assert len(glob('data/objects/' + labels[s]+'/*.binvox')) > 0, message + command
		

	

	models = []
	for s in wanted_classes:

		models  += glob('data/objects/'+labels[s]+'/*.binvox')
		location = 'data/surfaces/'+labels[s] +'/'
		if not os.path.exists(location):
			os.makedirs(location)
	random.shuffle(models)
	pbar = tqdm(models)
	pbar.set_description(f"Extracting point cloud from converted objects")
	for m in pbar:
		location = 'data/surfaces/'+m.split('/')[-2] +'/'
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

		obj = m[:-7] + '.obj'
		try:
			obj = utils.ObjLoader(obj)
		except ValueError:
			continue

		voxel_points = np.array(voxel_points)
		mesh_points = np.array(obj.vertices)

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
		
		while voxel_points.shape[0] < 10000: 
			voxel_points = np.concatenate((voxel_points, voxel_points))
		sio.savemat(location + m.split('/')[-1][:-7] , {'points': voxel_points})

def download_images():
	print('downloading shapenet images')
	command = 'wget http://cvgl.stanford.edu/data2/ShapeNetRendering.tgz -O data/images.tgz'
	os.system(command)
	
	print('extracting shapenet images')
	command = 'tar -xzf data/images.tgz --directory data/'
	os.system(command)
	
	os.rename('data/ShapeNetRendering','data/images' )

	print('Splitting dataset in training, valudation, and tests.')
	folders = glob('data/images/*/')
	for folder in tqdm(folders): 

		im_fol = glob( folder + '*')

		if not os.path.exists(folder + 'train'):
			os.mkdir(folder + 'train')
			os.mkdir(folder + 'valid')
			os.mkdir(folder + 'test')

		first = (7*len(im_fol))//10
		second  = (8*len(im_fol))//10
		
		train = im_fol[:first]
		valid = im_fol[first:second]
		test = im_fol[second:]

		for ex in train: 
			source = ex 
			dest = folder + 'train/' + ex.split('/')[-1]
			
			shutil.move(source, dest)

		for  ex in valid: 
			source = ex 
			dest = folder + 'valid/' + ex.split('/')[-1]
			
			shutil.move(source, dest)

		for  ex in test: 
			source = ex 
			dest = folder + 'test/' + ex.split('/')[-1]
			
			shutil.move(source, dest)
	for folder in folders:
		class_num = folder.split('/')[-2]
		class_obj = labels[class_num]
		os.rename(folder, folder[:-len(class_num)-1] + class_obj)







download()
manage_objects()
binvox()
convert_bin()
calc_surface()
download_images()
print ('finished eratin')
