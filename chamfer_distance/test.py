import torch
import torch.nn as nn
from torch.autograd import Variable

from modules.nnd import NNDModule

dist =  NNDModule()

p1 = torch.rand(1,3,3)*20
p2 = torch.rand(1,3,3)*20
# p1 = p1.int()
# p1.random_(0,2)
# p1 = p1.float()
# p2 = p2.int()
# p2.random_(0,2)
p2 = p2.float()
print(p1)
print(p2)

print('cpu')
points1 = Variable(p1,requires_grad = True)
points2 = Variable(p2)
idx1, idx2 = dist(points1, points2)
print(idx1, idx1)


print('gpu')
points1_cuda = Variable(p1.cuda(), requires_grad = True)
points2_cuda = Variable(p2.cuda())
idx1, idx2 = dist(points1_cuda, points2_cuda)
print(idx1, idx2)


