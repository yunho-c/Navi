#include <algorithm>
#include <vector>

#include "caffe/layers/spatial_softmax_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void SpatialSoftmaxLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const int num = bottom[0]->num();
  const int channels = bottom[0]->channels();
  const int height = bottom[0]->height();
  const int width = bottom[0]->width();
  inner_num_ = height * width;
  outer_num_ = num;
  // sum_multiplier_
  vector<int> mult_dims(1, inner_num_);
  sum_multiplier_.Reshape(mult_dims);
  Dtype* multiplier_data = sum_multiplier_.mutable_cpu_data();
  caffe_set(sum_multiplier_.count(), Dtype(1), multiplier_data);
  // scale_
  vector<int> scale_dims;
  scale_dims.push_back(outer_num_);
  scale_dims.push_back(channels);
  scale_.Reshape(scale_dims);
  // top
  top[0]->ReshapeLike(*bottom[0]);
}

template <typename Dtype>
void SpatialSoftmaxLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const int num = bottom[0]->num();
  const int channels = bottom[0]->channels();
  // const int height = bottom[0]->height();
  // const int width = bottom[0]->width();
  int dim = bottom[0]->count() / outer_num_;
  // data
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  Dtype* scale_data = scale_.mutable_cpu_data();
  // copy
  caffe_copy(bottom[0]->count(), bottom_data, top_data);
  // We need to subtract the max to avoid numerical issues, compute the exp,
  // and then normalize.
  for(int n = 0; n < num; n++) {
    // find the max
    for(int c = 0; c < channels; c++) {
      const int offset = n * dim + c * inner_num_;
      scale_data[c] = bottom_data[offset];
      for(int k = 0; k < inner_num_; k++) {
        scale_data[c] = std::max(scale_data[c], bottom_data[offset + k]);
      }
    }
    // subtract the max
    // M: channels, N: inner_num_, K: 1
    // A (M*K): scale_data - channels * 1
    // B (K*N): sum_multiplier_ - 1 * inner_num_
    // C (M*N): top_data - channels * inner_num_
    // C = alpha*A*B + beta*C
    const Dtype alpha_gemm = Dtype(-1);
    const Dtype beta_gemm = Dtype(1);
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, channels, inner_num_,
        1, alpha_gemm, scale_data, sum_multiplier_.cpu_data(), beta_gemm, top_data);
    // exp
    caffe_exp<Dtype>(dim, top_data, top_data);
    // sum after exp
    // M: channels, N: inner_num_
    // A (M*N): top_data - channels * inner_num_
    // x (N): sum_multiplier_ - inner_num_
    // y (M): scale_data - channels
    // y = alpha*A*x + beta*y
    const Dtype alpha_gemv = Dtype(1);
    const Dtype beta_gemv = Dtype(0);
    caffe_cpu_gemv<Dtype>(CblasNoTrans, channels, inner_num_, alpha_gemv,
        top_data, sum_multiplier_.cpu_data(), beta_gemv, scale_data);
    // div
    for(int c = 0; c < channels; c++) {
      const Dtype factor = Dtype(1) / scale_data[c];
      caffe_scal(inner_num_, factor, top_data);
      // increase offset for top data
      top_data += inner_num_;
    }
  }
}

template <typename Dtype>
void SpatialSoftmaxLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  const int num = bottom[0]->num();
  const int channels = bottom[0]->channels();
  // const int height = bottom[0]->height();
  // const int width = bottom[0]->width();
  int dim = top[0]->count() / outer_num_;
  // data && diff
  const Dtype* top_diff = top[0]->cpu_diff();
  const Dtype* top_data = top[0]->cpu_data();
  Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
  Dtype* scale_data = scale_.mutable_cpu_data();
  // copy
  caffe_copy(top[0]->count(), top_diff, bottom_diff);
  // compute the gradients for each feature map
  // Deri(Xi) = Deri(Pi)*Pi - sum(Deri(Pj)*Pj*Pi)
  //          = Pi * [Deri(Pi) - sum(Deri(Pj) * Pj)]
  //   for all j = 0,1,..., inner_num_-1
  for(int n = 0; n < num; n++) {
    // compute dot(top_diff, top_data) and subtract them from the bottom diff
    // cumpute -- sum(Deri(Pj) * Pj)
    for(int c = 0; c < channels; c++) {
      const int offset = n * dim + c * inner_num_;
      scale_data[c] = caffe_cpu_dot<Dtype>(inner_num_, bottom_diff + offset, top_data + offset);
    }
    // subtract -- Deri(Pi) - sum(Deri(Pj) * Pj)
    // M: channels, N: inner_num_, K: 1
    // A (M*K): scale_data - channels * 1
    // B (K*N): sum_multiplier_ - 1 * inner_num_
    // C (M*N): bottom_diff - channels * inner_num_
    // C = alpha*A*B + beta*C
    const Dtype alpha_gemm = Dtype(-1);
    const Dtype beta_gemm = Dtype(1);
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, channels, inner_num_, 1, 
        alpha_gemm, scale_data, sum_multiplier_.cpu_data(), beta_gemm, bottom_diff + n * dim);
  }
  // elementwise multiplication -- Pi * [Deri(Pi) - sum(Deri(Pj) * Pj)]
  caffe_mul(top[0]->count(), bottom_diff, top_data, bottom_diff);
}

#ifdef CPU_ONLY
STUB_GPU(SpatialSoftmaxLayer);
#endif

INSTANTIATE_CLASS(SpatialSoftmaxLayer);
REGISTER_LAYER_CLASS(SpatialSoftmax);

}  // namespace caffe
