import os
import bpy 
import bmesh
import scipy.io as sio
import sys
import numpy as np


# this whole file is to make managable obj files from overly large or samll ones 
# this is done for the latent loss calcualtions only


def triangulate_edit_object(obj):
	me = obj.data
	bm = bmesh.from_edit_mesh(me)
	bmesh.ops.triangulate(bm, faces=bm.faces[:], quad_method=0, ngon_method=0)
	bmesh.update_edit_mesh(me, True)



# import arguements
model = sys.argv[-3]
location_info = sys.argv[-2]
location_obj = sys.argv[-1]

#import object
bpy.ops.import_scene.obj(filepath=model ) 
scene = bpy.context.scene


# join components of mesh
obs = []
for ob in scene.objects:
	if ob.type == 'MESH':
		obs.append(ob)
ctx = bpy.context.copy()
ctx['active_object'] = obs[0]
ctx['selected_objects'] = obs
ctx['selected_editable_bases'] = [scene.object_bases[ob.name] for ob in obs]
bpy.ops.object.join(ctx)
o = bpy.context.selected_objects[0]


# removes split normal, helps with decimation
bpy.context.scene.objects.active = o 
bpy.ops.object.editmode_toggle()
bpy.ops.mesh.select_all(action='SELECT')
bpy.ops.mesh.customdata_custom_splitnormals_clear()
bpy.ops.mesh.remove_doubles()
bpy.ops.object.editmode_toggle()



# shrinking the mesh to be a uniform size 
# no idea if this actually helps with training
# it does make objects smaller which makes loading them much quicker to load during traineing 
# ideally all objects will be between 500 and 600 verts, but I allow between 400 and 700 verts 
# the object is used during training regardless, but not for the latent loss 
not_possible = False
num = float(len(o.data.vertices))
new_num  = num 
orig = num
full_ratio = .01
if num<30: 
	not_possible = True # if its too small then subsampling doesnt work well 

# if large then decimate to make the right size
# or at least try to 
elif num> 550:
	for i in range(5): 	
		mod = o.modifiers.new(name='decimate', type='DECIMATE')
		mod.ratio = max(550./num, full_ratio)
		full_ratio /= (mod.ratio) 
		bpy.ops.object.modifier_apply(modifier = mod.name)
		o.modifiers.clear()
		if float(len(o.data.vertices)) < 550:
			new_num = float(len(o.data.vertices))
			break
		else: 
			num = float(len(o.data.vertices)) 
		if i == 4 and float(len(o.data.vertices)) > 700: # if it can't be made small enough then don't convert it 
			not_possible = True
		new_num = float(len(o.data.vertices))
# if small then try to make it larger 
elif num < 400: 
	mod = o.modifiers.new(name="Remesh", type='REMESH')
	mod.octree_depth = 6
	mod.use_remove_disconnected = False
	bpy.ops.object.modifier_apply(modifier = mod.name)
	o.modifiers.clear()
	num = float(len(o.data.vertices))
	# and then shrink it again 
	if num> 500:
		for i in range(5): 	
			mod = o.modifiers.new(name='decimate', type='DECIMATE')
			mod.ratio = 550./num 
			bpy.ops.object.modifier_apply(modifier = mod.name)
			o.modifiers.clear()
			if float(len(o.data.vertices)) < 600:
				break
			else: 
				num = float(len(o.data.vertices)) 
	else:
		not_possible = True 
print ('-------------------------------------------------------------------------')
print (  float(len(o.data.vertices)) )
print ('-------------------------------------------------------------------------')
if not_possible: 
	exit()

# triangluate the object 
bpy.ops.object.editmode_toggle()
bpy.ops.mesh.dissolve_limited()
triangulate_edit_object(o)
bpy.ops.object.editmode_toggle()



# now we record the object info

# get initial face info
me = o.data
adj_new = np.zeros((600,600))
max_len = 0
faces =  []
for poly in me.polygons:
	vs = []
	for loop_index in range(poly.loop_start, poly.loop_start + poly.loop_total):
		vs.append(me.loops[loop_index].vertex_index)
	faces.append(vs)
		
	

# get initial vertex info, and normal info if you want it (I dont)
bm = bmesh.new()
bm.from_mesh(me)
verts, normals  = [0 for i in range(len(bm.verts))],[0 for i in range(len(bm.verts))]
for e,v in enumerate(bm.verts):
	verts[v.index] = v.co 
	normals[v.index] = v.normal


# calculate adjacency matrix and final face, vertex and nromal infor 
verts_map = {}
count = 0
for face in faces: 
	v1, v2, v3 = face 
	for v in face: 
		if v not in verts_map: 
			verts_map[v] = [count, verts[v], normals[v] ]
			count += 1 
adj = np.zeros((len(verts_map), len(verts_map)))
true_verts = np.zeros((len(verts_map), 3))
true_normals = np.zeros((len(verts_map), 3))
for e,face in enumerate(faces): 
	v1, v2, v3 = face 
	adj[verts_map[v1][0], verts_map[v1][0]] = 1 
	adj[verts_map[v2][0], verts_map[v2][0]] = 1 
	adj[verts_map[v3][0], verts_map[v3][0]] = 1 
	adj[verts_map[v1][0], verts_map[v2][0]] = 1 
	adj[verts_map[v2][0], verts_map[v1][0]] = 1 
	adj[verts_map[v1][0], verts_map[v3][0]] = 1 
	adj[verts_map[v3][0], verts_map[v1][0]] = 1 
	adj[verts_map[v2][0], verts_map[v3][0]] = 1 
	adj[verts_map[v3][0], verts_map[v2][0]] = 1 
	faces[e] = [verts_map[v1][0], verts_map[v2][0], verts_map[v3][0]]

for _ , info in verts_map.items(): 
	spot, position, normal = info 
	true_verts[spot] = position 
	true_normals[spot] = normal

for obj in bpy.data.objects:
    obj.select = False
o.select = True




# save updated object, and object info 
bpy.ops.export_scene.obj(filepath=location_obj)
sio.savemat(location_info, {'verts':np.array(true_verts), 
				'normals': np.array(true_normals), 
				'faces': np.array(faces),  
				'orig_adj': adj
				}
				)

