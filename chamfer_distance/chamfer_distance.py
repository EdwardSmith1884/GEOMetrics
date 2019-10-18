
import torch

from torch.utils.cpp_extension import load
cd = load(name="cd",
          sources=["chamfer_distance/chamfer_distance.cpp",
                   "chamfer_distance/chamfer_distance.cu"])

class ChamferDistanceFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, xyz1, xyz2):
        batchsize, n, _ = xyz1.size()
        _, m, _ = xyz2.size()
        xyz1 = xyz1.contiguous()
        xyz2 = xyz2.contiguous()
        dist1 = torch.zeros(batchsize, n)
        dist2 = torch.zeros(batchsize, m)

        idx1 = torch.zeros(batchsize, n, dtype=torch.int)
        idx2 = torch.zeros(batchsize, m, dtype=torch.int)

   
        dist1 = dist1.cuda()
        dist2 = dist2.cuda()
        idx1 = idx1.cuda()
        idx2 = idx2.cuda()
        cd.forward_cuda(xyz1, xyz2, dist1, dist2, idx1, idx2)

       

        return idx1, idx2

   


class ChamferDistance(torch.nn.Module):
    def forward(self, xyz1, xyz2):
        return ChamferDistanceFunction.apply(xyz1, xyz2)
