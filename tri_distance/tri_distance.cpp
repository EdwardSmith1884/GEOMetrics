#include <torch/torch.h>

// CUDA forward declarations
int TriDistanceKernelLauncher(
    const int b, const int n,
    const float* xyz,
    const int m,
    const float* tri1,
    const float* tri2,
    const float* tri3,
    float* dist,
    int* point,
    int* index);


void tri_distance_forward_cuda(
    const at::Tensor xyz1, 
    const at::Tensor tri1, 
    const at::Tensor tri2, 
    const at::Tensor tri3, 
    const at::Tensor dist, 
    const at::Tensor point,
    const at::Tensor index) 
{
    TriDistanceKernelLauncher(xyz1.size(0), xyz1.size(1), xyz1.data<float>(),
                                            tri1.size(1), tri1.data<float>(),
                                            tri2.data<float>(),tri3.data<float>(),
                                            dist.data<float>(), point.data<int>(), 
                                            index.data<int>());
}



PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward_cuda", &tri_distance_forward_cuda, "TriDistance forward (CUDA)");
}
