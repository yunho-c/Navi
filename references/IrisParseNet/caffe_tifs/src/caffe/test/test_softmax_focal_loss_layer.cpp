#include <cmath>
#include <vector>

#include "boost/scoped_ptr.hpp"
#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/softmax_focal_loss_layer.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

using boost::scoped_ptr;

namespace caffe {

template <typename TypeParam>
class SoftmaxWithFocalLossLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  SoftmaxWithFocalLossLayerTest()
      : blob_bottom_data_(new Blob<Dtype>(10, 2, 2, 3)),  //必须channel为2
        blob_bottom_targets_(new Blob<Dtype>(10, 1, 2, 3)), 
        blob_top_loss_(new Blob<Dtype>()) {
    // Fill the data vector
    FillerParameter data_filler_param;
    data_filler_param.set_std(10);  //标准差为10
    GaussianFiller<Dtype> data_filler(data_filler_param);//高斯初始    data_filler.Fill(blob_bottom_data_);
    blob_bottom_vec_.push_back(blob_bottom_data_);  //填充bottom_blob[0]
    // Fill the targets vector
    for (int i = 0; i < blob_bottom_targets_->count(); ++i) {
      blob_bottom_targets_->mutable_cpu_data()[i] = caffe_rng_rand() % 2;
    }
    blob_bottom_vec_.push_back(blob_bottom_targets_); //填充bottom_blob[1]
    blob_top_vec_.push_back(blob_top_loss_);   //填充top_blob[0]
  }
  virtual ~SoftmaxWithFocalLossLayerTest() {
    delete blob_bottom_data_;
    delete blob_bottom_targets_;
    delete blob_top_loss_;
  }

  Blob<Dtype>* const blob_bottom_data_;
  Blob<Dtype>* const blob_bottom_targets_;
  Blob<Dtype>* const blob_top_loss_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(SoftmaxWithFocalLossLayerTest, TestDtypesAndDevices);

TYPED_TEST(SoftmaxWithFocalLossLayerTest, TestGradient) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  layer_param.add_loss_weight(3);
  const Dtype alpha=0.75;
  const Dtype  gamma=2;
  FocalLossParameter* focal_loss_param = layer_param.mutable_focal_loss_param();
  focal_loss_param->set_alpha(alpha);
  focal_loss_param->set_gamma(gamma);
  SoftmaxWithFocalLossLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  GradientChecker<Dtype> checker(1e-2, 1e-2, 1701);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_, 0);
}

//这段可以去掉。
/* TYPED_TEST(SoftmaxWithFocalLossLayerTest, TestForwardIgnoreLabel) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  layer_param.mutable_loss_param()->set_normalize(false);
  const Dtype alpha=0.75;
  const Dtype  gamma=2;
  FocalLossParameter* focal_loss_param = layer_param.mutable_focal_loss_param();
  focal_loss_param->set_alpha(alpha);
  focal_loss_param->set_gamma(gamma);
  // First, compute the loss with all labels
  scoped_ptr<SoftmaxWithFocalLossLayer<Dtype> > layer(
      new SoftmaxWithFocalLossLayer<Dtype>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  Dtype full_loss = this->blob_top_loss_->cpu_data()[0];
  // Now, accumulate the loss, ignoring each label in {0, ..., 4} in turn.
  Dtype accum_loss = 0;
  for (int label = 0; label < 2; ++label) {
    layer_param.mutable_loss_param()->set_ignore_label(label);
    layer.reset(new SoftmaxWithFocalLossLayer<Dtype>(layer_param));
    layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
    layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
    accum_loss += this->blob_top_loss_->cpu_data()[0];
  }
  // Check that each label was included all but once.
  EXPECT_NEAR(full_loss, accum_loss, 1e-4);
} */

TYPED_TEST(SoftmaxWithFocalLossLayerTest, TestGradientIgnoreLabel) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  // labels are in {0, ..., 4}, so we'll ignore about a fifth of them
  layer_param.mutable_loss_param()->set_ignore_label(0);
  const Dtype alpha=0.75;
  const Dtype  gamma=2;
  FocalLossParameter* focal_loss_param = layer_param.mutable_focal_loss_param();
  focal_loss_param->set_alpha(alpha);
  focal_loss_param->set_gamma(gamma);
  SoftmaxWithFocalLossLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-2, 1e-2, 1701);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_, 0);
}

TYPED_TEST(SoftmaxWithFocalLossLayerTest, TestGradientUnnormalized) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  layer_param.mutable_loss_param()->set_normalize(false);
  const Dtype alpha=0.75;
  const Dtype  gamma=2;
  FocalLossParameter* focal_loss_param = layer_param.mutable_focal_loss_param();
  focal_loss_param->set_alpha(alpha);
  focal_loss_param->set_gamma(gamma);
  SoftmaxWithFocalLossLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-2, 1e-2, 1701);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_, 0);
}

}  // namespace caffe
