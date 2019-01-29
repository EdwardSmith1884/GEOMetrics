BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(BASE_DIR + '../scripts/')
from __future__ import division
from collections import namedtuple
from utils import * 
from torch.nn.parameter import Parameter


init_verts = [[0,0,0], [0,1,0], [1,0,0], [1,1,0]] 
init_faces = [[0,1,2], [1,3,2]]

tar_verts = [[.25,.19,0],[.75,.22,0], [.5,.7,0] ]
tar_faces = [[0,1,2]]

init_verts = torch.FloatTensor(init_verts).cuda()
init_faces = torch.LongTensor(init_faces).cuda()
tar_verts = torch.FloatTensor(tar_verts).cuda()
tar_faces = torch.LongTensor(tar_faces).cuda()


class changer(nn.Module): 
	def __init__(self,): 
		super(changer, self).__init__()
		self.bias = Parameter(torch.Tensor(4, 3))
		self.reset_parameters()
	def reset_parameters(self):
		
		self.bias.data.uniform_(-0, 0)

	def forward(self, points):
		points = points + self.bias 
		return points






def vert_to_point(p_vert, p_face, t_vert, t_face, num = 3):
	
	pred_points =  p_vert
	pred_points_pass = pred_points.view([1,pred_points.shape[0], pred_points.shape[1]])
	
	gt_points = sample(t_vert, t_face, num =num)
	gt_points_pass = gt_points.view([1,gt_points.shape[0], gt_points.shape[1]])


	id_1, id_2 = dist( gt_points_pass, pred_points_pass)
	
	# select pairs and calculate chamfer distance 
	pred_counters = torch.index_select(pred_points, 0, id_1[0].long())
	gt_counters =  torch.index_select(gt_points, 0, id_2[0].long())
	dist_1 = torch.mean(torch.sum((pred_counters - gt_points)**2, dim  = 1 ))
	dist_2 = torch.mean(torch.sum((gt_counters - pred_points)**2, dim = 1))
	loss = dist_1+ dist_2
	return loss, gt_points 


def point_to_point(p_vert, p_face, t_vert, t_face, num = 5000):
	
	pred_points = sample(p_vert, p_face, num =num)
	pred_points_pass = pred_points.view([1,pred_points.shape[0], pred_points.shape[1]])
	
	gt_points = sample(t_vert, t_face, num =num)
	gt_points_pass = gt_points.view([1,gt_points.shape[0], gt_points.shape[1]])


	id_1, id_2 = dist( gt_points_pass, pred_points_pass)
	
	# select pairs and calculate chamfer distance 
	pred_counters = torch.index_select(pred_points, 0, id_1[0].long())
	gt_counters =  torch.index_select(gt_points, 0, id_2[0].long())
	dist_1 = torch.mean(torch.sum((pred_counters - gt_points)**2, dim  = 1 ))
	dist_2 = torch.mean(torch.sum((gt_counters - pred_points)**2, dim = 1))
	loss = dist_1+ dist_2
	return loss, gt_points 

def surface_to_point(p_vert, p_face, t_vert, t_face, num = 500):
	
	pred_points = sample(p_vert, p_face, num =num)
	pred_points_pass = pred_points.view([1,pred_points.shape[0], pred_points.shape[1]])
	
	gt_points = sample(t_vert, t_face, num =num)
	gt_points_pass = gt_points.view([1,gt_points.shape[0], gt_points.shape[1]])


	dist_1 = point_to_line(p_vert, p_face, gt_points)
	dist_2 = point_to_line(t_vert, t_face, pred_points)
	
	loss = dist_1+ dist_2
	return loss, gt_points 


def line_intersection(line1, line2):
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
       return False, 0,0

    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    return True, x, y

def find_intersection(L1, L2): 
	result = line_intersection(L1, L2)
	if not result[0]: 
		return False, 0, 0 
	
	x, y, = result[1:]
	minx = min(L1[0][0], L1[1][0])
	maxx = max(L1[0][0], L1[1][0])
	if x> maxx or x< minx: 
		return False, 0, 0 
	minx = min(L2[0][0], L2[1][0])
	maxx = max(L2[0][0], L2[1][0])
	if x> maxx or x< minx: 
		return False, 0, 0 
	miny = min(L1[0][1], L1[1][1])
	maxy = max(L1[0][1], L1[1][1])
	if y> maxy or y< miny: 
		return False, 0, 0 
	miny = min(L2[0][1], L2[1][1])
	maxy = max(L2[0][1], L2[1][1])
	if y> maxy or y< miny: 
		return False, 0, 0 
	return True, x, y 




def sign(p1, p2, p3):
  return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])


def PointInAABB(pt, c1, c2):
  return c2[0] <= pt[0] <= c1[0] and \
		 c2[1] <= pt[1] <= c1[1]

def PointInTriangle(pt, v1, v2, v3):
  b1 = sign(pt, v1, v2) <= 0
  b2 = sign(pt, v2, v3) <= 0
  b3 = sign(pt, v3, v1) <= 0

  return ((b1 == b2) and (b2 == b3)) and \
		 PointInAABB(pt, map(max, v1, v2, v3), map(min, v1, v2, v3))






def intersection(t1, t2): 
	# find intersectio points
	p1, p2, p3 = t1
	p4, p5, p6 = t2 

	
	points = []
	for i in range(3): 
		for j in range(3): 
		
			P1, P2, P3, P4 = (t1[i-1][0] , t1[i-1][1]), (t1[i][0], t1[i][1]),(t2[j-1][0], t2[j-1][1]), (t2[j][0], t2[j][1])
			L1 = (P1, P2)
			L2 = (P3, P4)
			results = find_intersection(L1, L2)
			
			if results[0] :

				points.append([results[1], results[2]]) 
	
	# find points inside other triangles 
	for i in range(3): 
		if PointInTriangle(t1[i],p4,p5,p6): 
			points.append(t1[i])
	for i in range(3): 
		if PointInTriangle(t2[i],p1,p2,p3): 
			points.append(t2[i])
	points = np.array(points)
	if len(points)<3 or len(points)>6: 
		print points
		print t1
		print t2
		exit()
	# if area(points)> area(t1): 
	# 	print area(points), area(t1)
	# 	print 1
	# 	print points, t1, exit()

	# if area(points)> area(t2): 
	# 	print 2
	# 	print points, t2, exit()

	return area(points)







def area_tri(verts): 
	x1, y1 = verts[0]
	x2, y2 = verts[1]
	x3, y3 = verts[2]
	return abs(.5*( x1*(y2-y3)+x2*(y3-y1)+x3*(y1-y2)))
from itertools import permutations
def area_poly(verts):

	origin = np.mean(verts, axis = 0 )
	refvec = [0, 1]
	def clockwiseangle_and_distance(point):
	    # Vector between point and the origin: v = p - o
	    vector = [point[0]-origin[0], point[1]-origin[1]]
	    # Length of vector: ||v||
	    lenvector = math.hypot(vector[0], vector[1])
	    # If length is zero there is no angle
	    if lenvector == 0:
	        return -math.pi, 0
	    # Normalize vector: v/||v||
	    normalized = [vector[0]/lenvector, vector[1]/lenvector]
	    dotprod  = normalized[0]*refvec[0] + normalized[1]*refvec[1]     # x1*x2 + y1*y2
	    diffprod = refvec[1]*normalized[0] - refvec[0]*normalized[1]     # x1*y2 - y1*x2
	    angle = math.atan2(diffprod, dotprod)
	    # Negative angles represent counter-clockwise angles so we need to subtract them 
	    # from 2*pi (360 degrees)
	    if angle < 0:
	        return 2*math.pi+angle, lenvector
	    # I return first the angle because that's the primary sorting criterium
	    # but if two vectors have the same angle then the shorter distance should come first.
	    return angle

	new_verts = np.array(sorted(verts, key = clockwiseangle_and_distance))


	x = new_verts[:,0]
	y = new_verts[:,1]
	return 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))
	

def area(verts): 
	if verts.shape[0] == 3: 
		return  area_tri(verts)
	else: 
		return area_poly(verts)



def IoU( square, triangle): 
	# print square.shape, exit()
	square = (square.data.cpu().numpy()[:,:2])

	sq_tri1 = square[:3]
	sq_tri2 = np.concatenate((square[1:2], square[3:4], square[2:3]), axis = 0 )
	triangle = (triangle.data.cpu().numpy()[:,:2])
	# print square, exit()

	inter1 = intersection(sq_tri1, triangle)
	inter2 = intersection(sq_tri2, triangle)


	full_inter = inter1 + inter2
	# print inter1, inter2
	# print inter1, inter2, area(sq_tri1), area(sq_tri2), area(triangle)

	union = area(sq_tri1) + area(sq_tri2)+area(triangle) - full_inter

	return full_inter / union




setter = []
losses = [[],[],[]]
funcs = [vert_to_point, point_to_point, surface_to_point]
for j in range(3):
	loss_function = funcs[j]
	for x in tqdm(range(1, 20)):
		number = x
		setter.append(x)
		change = changer()

		change.cuda()

		params = list(change.parameters()) 
		optimizer = optim.Adam(params,lr=0.01)

		change.train()
		for i in (range(600)): 
			optimizer.zero_grad() 
			pred_verts = change(init_verts)
			loss, points = loss_function( pred_verts, init_faces, tar_verts, tar_faces, num = number )
			
			loss.backward()
			optimizer.step()



		change.eval()
		pred_verts = pred_verts = change(init_verts)
		loss = IoU(pred_verts, tar_verts)

	
		losses[j].append(loss)

losses = np.array(losses)

setter = setter[:len(setter)//3]




from scipy.ndimage.filters import gaussian_filter1d
import matplotlib.pyplot as plt
y1_smooth = gaussian_filter1d(losses[0], sigma = 2 )
plt.plot(setter, y1_smooth, '.-', color ='green', label='Vertex to Point') 
plt.plot(setter,losses[1], '.-', color = 'red', label = 'Point to Point')
plt.plot(setter, losses[2], '.-', color = 'blue', label = 'Point to Surface')
plt.xlabel('Sampled Points')
plt.ylabel('IoU')
plt.title('Comparison of Loss Functions ')
plt.grid()
plt.legend()

ax = plt.gca()
plt.legend(bbox_to_anchor=(.95, .78), bbox_transform=ax.transAxes)






plt.show()
plt.savefig('Loss_graph.png')




