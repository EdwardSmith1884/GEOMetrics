
import torch

from torch.utils.cpp_extension import load
tri = load(name="tri",
          sources=["tri_distance/tri_distance.cpp",
                   "tri_distance/tri_distance.cu"])

class TriDistanceFunction(torch.autograd.Function):
    @staticmethod
    def forward(self, xyz1, tri1, tri2, tri3):
    
        batchsize, n, _ = xyz1.size()
        _, m, _ = tri1.size()
        xyz1 = xyz1.contiguous()
        tri1 = tri1.contiguous()
        tri2 = tri2.contiguous()
        tri3 = tri3.contiguous()
        


        dist = torch.zeros(batchsize, n)
        point = torch.zeros(batchsize, n).int()
        index = torch.zeros(batchsize, n).int()
       

   
        dist = dist.cuda()
        point = point.cuda()
        index = index.cuda()
      

        tri.forward_cuda( xyz1, tri1, tri2, tri3, dist, point, index)

      
        return dist, point, index

   


class TriDistance(torch.nn.Module):
    def forward(self, xyz1, tri1, tri2, tri3):
        return TriDistanceFunction.apply(xyz1, tri1, tri2, tri3)
