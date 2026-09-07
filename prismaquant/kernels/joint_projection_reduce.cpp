#include <torch/extension.h>

at::Tensor joint_projection_mul_sum_cuda(const at::Tensor& left,
                                        const at::Tensor& right);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("mul_sum", &joint_projection_mul_sum_cuda,
             "FP32 product with the installed PyTorch sum reduction tree");
}
