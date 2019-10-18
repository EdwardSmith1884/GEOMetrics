#include <torch/torch.h>

// CUDA forward declarations
int ChamferDistanceKernelLauncher(
    const int b, const int n,
    const float* xyz,
    const int m,
    const float* xyz2,
    float* result,
    int* result_i,
    float* result2,
    int* result2_i);


void chamfer_distance_forward_cuda(
    const at::Tensor xyz1, 
    const at::Tensor xyz2, 
    const at::Tensor dist1, 
    const at::Tensor dist2, 
    const at::Tensor idx1, 
    const at::Tensor idx2) 
{
    ChamferDistanceKernelLauncher(xyz1.size(0), xyz1.size(1), xyz1.data<float>(),
                                            xyz2.size(1), xyz2.data<float>(),
                                            dist1.data<float>(), idx1.data<int>(),
                                            dist2.data<float>(), idx2.data<int>());
}








PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward_cuda", &chamfer_distance_forward_cuda, "ChamferDistance forward (CUDA)");
}
