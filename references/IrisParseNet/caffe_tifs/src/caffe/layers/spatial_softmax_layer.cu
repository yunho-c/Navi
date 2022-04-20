#include <algorithm>
#include <cfloat>
#include <vector>

#include "thrust/device_vector.h"

#include "caffe/layers/spatial_softmax_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
__global__ void kernel_spatial_max(const int num, const int channels,
    const int spatial_dim, const Dtype* data, Dtype* out) {
  CUDA_KERNEL_LOOP(index, num * channels) {
    int n = index / channels;
    int c = index % channels;
    int dim = channels * spatial_dim;
    int offset = n * dim + c * spatial_dim;
    Dtype maxval = -FLT_MAX;
    for(int k = 0; k < spatial_dim; k++) {
      maxval = max(data[offset + k], maxval);
    }
    out[index] = maxval;
  }
}

template <typename Dtype>
__global__ void kernel_spatial_subtract(const int count,
    const int num, const int channels,
    const int spatial_dim, const Dtype* spatial_max, Dtype* data) {
  CUDA_KERNEL_LOOP(index, count) {
    int n = (index / spatial_dim) / channels ;
    int c = (index / spatial_dim) % channels;
    data[index] -= spatial_max[n * channels + c];
  }
}

template <typename Dtype>
__global__ void kernel_exp(const int count, const Dtype* data, Dtype* out) {
  CUDA_KERNEL_LOOP(index, count) {
    out[index] = exp(data[index]);
  }
}

template <typename Dtype>
__global__ void kernel_spatial_sum(const int num, const int channels,
    const int spatial_dim, const Dtype* data, Dtype* spatial_sum) {
  CUDA_KERNEL_LOOP(index, num * channels) {
    int n = index / channels;
    int c = index % channels;
    int dim = channels * spatial_dim;
    int offset = n * dim + c * spatial_dim;
    Dtype sum = 0;
    for(int k = 0; k < spatial_dim; k++) {
      sum += data[offset + k];
    }
    spatial_sum[index] = sum;
  }
}

template <typename Dtype>
__global__ void kernel_spatial_div(const int count,
    const int num, const int channels,
    const int spatial_dim, const Dtype* spatial_sum, Dtype* data) {
  CUDA_KERNEL_LOOP(index, count) {
    int n = (index / spatial_dim) / channels ;
    int c = (index / spatial_dim) % channels;
    data[index] /= spatial_sum[n * channels + c];
  }
}

template <typename Dtype>
void SpatialSoftmaxLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  // const int num = bottom[0]->num();
  const int channels = bottom[0]->channels();
  // const int height = bottom[0]->height();
  // const int width = bottom[0]->width();
  int dim = top[0]->count() / outer_num_;
  int count = bottom[0]->count();
  // data
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  Dtype* scale_data = scale_.mutable_gpu_data();
  // copy
  caffe_copy(count, bottom_data, top_data);
  // We need to subtract the max to avoid numerical issues, compute the exp,
  // and then normalize.
  // *** compute max ***
  // NOLINT_NEXT_LINE(whitespace/operators)
  kernel_spatial_max<Dtype><<<CAFFE_GET_BLOCKS(outer_num_ * channels),
      CAFFE_CUDA_NUM_THREADS>>>(outer_num_, channels, inner_num_, top_data,
      scale_data);
  // *** subtract ***
  // NOLINT_NEXT_LINE(whitespace/operators)
  kernel_spatial_subtract<Dtype><<<CAFFE_GET_BLOCKS(count),
      CAFFE_CUDA_NUM_THREADS>>>(count, outer_num_, channels, inner_num_,
      scale_data, top_data);
  // *** exponentiate ***
  // NOLINT_NEXT_LINE(whitespace/operators)
  kernel_exp<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, top_data, top_data);
  // *** sum after exp ***
  // NOLINT_NEXT_LINE(whitespace/operators)
  kernel_spatial_sum<Dtype><<<CAFFE_GET_BLOCKS(outer_num_ * channels),
      CAFFE_CUDA_NUM_THREADS>>>(outer_num_, channels, inner_num_, top_data,
      scale_data);
  // *** divide ***
  // NOLINT_NEXT_LINE(whitespace/operators)
  kernel_spatial_div<Dtype><<<CAFFE_GET_BLOCKS(count),
      CAFFE_CUDA_NUM_THREADS>>>(count, outer_num_, channels, inner_num_,
      scale_data, top_data);
}

template <typename Dtype>
__global__ void kernel_spatial_dot(const int num, const int channels,
    const int spatial_dim, const Dtype* diff, const Dtype* data,
    Dtype* spatial_dot) {
  CUDA_KERNEL_LOOP(index, num * channels) {
    int n = index / channels;
    int c = index % channels;
    int dim = channels * spatial_dim;
    int offset = n * dim + c * spatial_dim;
    Dtype dot = 0;
    for (int k = 0; k < spatial_dim; k++) {
      dot += data[offset + k]* diff[offset + k];
    }
    spatial_dot[index] = dot;
  }
}

template <typename Dtype>
void SpatialSoftmaxLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  // const int num = bottom[0]->num();
  const int channels = bottom[0]->channels();
  // const int height = bottom[0]->height();
  // const int width = bottom[0]->width();
  int dim = top[0]->count() / outer_num_;
  int count = top[0]->count();
  // data && diff
  const Dtype* top_diff = top[0]->gpu_diff();
  const Dtype* top_data = top[0]->gpu_data();
  Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
  Dtype* scale_data = scale_.mutable_gpu_data();
  // copy
  caffe_copy(count, top_diff, bottom_diff);
  // Deri(Xi) = Deri(Pi)*Pi - sum(Deri(Pj)*Pj*Pi)
  //          = Pi * [Deri(Pi) - sum(Deri(Pj) * Pj)]
  //   for all j = 0,1,..., inner_num_-1
  // *** cumpute *** -- sum(Deri(Pj) * Pj)
  // Compute inner1d(top_diff, top_data) and subtract them from the bottom diff.
  // NOLINT_NEXT_LINE(whitespace/operators)
  kernel_spatial_dot<Dtype><<<CAFFE_GET_BLOCKS(outer_num_ * channels),
      CAFFE_CUDA_NUM_THREADS>>>(outer_num_, channels, inner_num_,
      top_diff, top_data, scale_data);
  // *** subtract *** -- Deri(Pi) - sum(Deri(Pj) * Pj)
  // NOLINT_NEXT_LINE(whitespace/operators)
  kernel_spatial_subtract<Dtype><<<CAFFE_GET_BLOCKS(count),
      CAFFE_CUDA_NUM_THREADS>>>(count, outer_num_, channels, inner_num_,
      scale_data, bottom_diff);
  // *** elementwise multiplication *** -- Pi * [Deri(Pi) - sum(Deri(Pj) * Pj)]
  caffe_gpu_mul<Dtype>(top[0]->count(), bottom_diff, top_data, bottom_diff);
}

INSTANTIATE_LAYER_GPU_FUNCS(SpatialSoftmaxLayer);

}  // namespace caffe