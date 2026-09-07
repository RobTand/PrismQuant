// Research candidate: the installed PyTorch sum reduction tree with an inline
// separately rounded product. No large product tensor and no fused multiply-add.
#include <ATen/ATen.h>
#include <ATen/TensorIterator.h>
#include <ATen/native/cuda/Reduce.cuh>
#include <c10/cuda/CUDAGuard.h>
#include <cstdint>

namespace {
struct ProductSumOps {
  const float* right;

  __device__ float reduce(float accumulated, float left, int64_t index) const {
    return __fadd_rn(accumulated, __fmul_rn(left, right[index]));
  }
  __device__ float combine(float left, float right_value) const {
    return __fadd_rn(left, right_value);
  }
  __device__ float project(float value) const { return value; }
  __device__ float translate_idx(float value, int64_t) const { return value; }
  __device__ float warp_shfl_down(float value, int offset) const {
    return WARP_SHFL_DOWN(value, offset);
  }
};
}  // namespace

at::Tensor joint_projection_mul_sum_cuda(const at::Tensor& left,
                                        const at::Tensor& right) {
  TORCH_CHECK(left.is_cuda() && right.is_cuda(), "CUDA operands required");
  TORCH_CHECK(left.device() == right.device(), "operand devices differ");
  TORCH_CHECK(left.scalar_type() == at::kFloat && right.scalar_type() == at::kFloat,
              "FP32 operands required");
  TORCH_CHECK(left.sizes() == right.sizes(), "operand shapes differ");
  TORCH_CHECK(left.is_contiguous() && right.is_contiguous(), "contiguous operands required");
  TORCH_CHECK(left.numel() > 0, "nonempty operands required");
  // The reference materialized product is aligned. Requiring the same input
  // alignment prevents ReduceOp's misaligned-header branch changing its tree.
  TORCH_CHECK(reinterpret_cast<uintptr_t>(left.const_data_ptr<float>()) % 16 == 0 &&
              reinterpret_cast<uintptr_t>(right.const_data_ptr<float>()) % 16 == 0,
              "16-byte-aligned operands required");
  const c10::cuda::CUDAGuard guard(left.device());
  auto output = at::empty({}, left.options());
  auto iter = at::TensorIterator::reduce_op(output, left);
  // Recursive sub-iterator offsets need a separately derived index contract.
  TORCH_CHECK(iter.can_use_32bit_indexing(), "32-bit reduction indexing required");
  at::native::gpu_reduce_kernel<float, float, 4, 4>(
      iter, ProductSumOps{right.const_data_ptr<float>()}, 0.0f);
  return output;
}
