#include <cmath>
#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/focal_loss_layer.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

template <typename TypeParam>
class FocalLossLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  FocalLossLayerTest()
      : blob_bottom_data_(new Blob<Dtype>(1, 1, 3, 3)),
        blob_bottom_targets_(new Blob<Dtype>(1, 1, 3, 3)),
        blob_top_loss_(new Blob<Dtype>()) {
    // Fill the data vector
    FillerParameter data_filler_param;
    data_filler_param.set_std(1);  //标准差为1
    GaussianFiller<Dtype> data_filler(data_filler_param);//高斯初始化    data_filler.Fill(blob_bottom_data_);
    blob_bottom_vec_.push_back(blob_bottom_data_);  //填充bottom_blob[0]
    // Fill the targets vector
    FillerParameter targets_filler_param;
    targets_filler_param.set_min(0);
    targets_filler_param.set_max(1);
    UniformFiller<Dtype> targets_filler(targets_filler_param); //均匀分布
    targets_filler.Fill(blob_bottom_targets_);
    for (int i=0; i<blob_bottom_targets_->count(); ++i) {    //对label标号，前景为1，背景为0
      if (blob_bottom_targets_->cpu_data()[i] >= 0.5)
        blob_bottom_targets_->mutable_cpu_data()[i] = 1;
      else
        blob_bottom_targets_->mutable_cpu_data()[i] = 0;
    }  
    blob_bottom_vec_.push_back(blob_bottom_targets_); //填充bottom_blob[1]
    blob_top_vec_.push_back(blob_top_loss_);   //填充top_blob[0]
  }
  virtual ~FocalLossLayerTest() {
    delete blob_bottom_data_;
    delete blob_bottom_targets_;
    delete blob_top_loss_;
  }

  Dtype FocalLossReference(const int count, const int num,
                                         const Dtype* input,
                                         const Dtype* target, const Dtype alpha,const Dtype gamma) {
    Dtype loss = 0;
	Dtype count_pos=0,count_neg=0;
	Dtype alpha2=-1;
	if(alpha<0)
	{
		for (int i = 0; i < count; ++i) {
		  const int target_value = static_cast<int>(target[i]);
		  if (target_value == 1) {
			  count_pos++;
		  }
		  else if (target_value == 0)
		  {
			  count_neg++;
		  }
	  }
	  if (static_cast<int>(count_pos)== 0 || static_cast<int>(count_neg) == 0)
	  {
		  alpha2 = 0.5;
	  }
	  else
	  {
		  alpha2 = count_neg / (count_neg + count_pos);
	  }
	}
	else
	{
		alpha2=alpha;
	}
    for (int i = 0; i < count; ++i) {
      const Dtype prediction = 1 / (1 + exp(-input[i]));
      EXPECT_LE(prediction, 1);
      EXPECT_GE(prediction, 0);
      EXPECT_LE(target[i], 1);
      EXPECT_GE(target[i], 0);
      loss -= target[i] * log(prediction + (target[i] == Dtype(0)))*powf(1-prediction+ (target[i] == Dtype(0)),gamma)*(alpha2+ (target[i] == Dtype(0)));
      loss -= (1 - target[i]) * log(1 - prediction + (target[i] == Dtype(1)))*powf(prediction+ (target[i] == Dtype(1)),gamma)*(1-alpha2+ (target[i] == Dtype(1)));
    }
    return loss / num;
  }

  void TestForward() {
    LayerParameter layer_param;
    const Dtype kLossWeight = 3.7;
    layer_param.add_loss_weight(kLossWeight);
    const Dtype alpha=-1;
	const Dtype  gamma=2;
	FocalLossParameter* focal_loss_param = layer_param.mutable_focal_loss_param();
	focal_loss_param->set_alpha(alpha);
	focal_loss_param->set_gamma(gamma);
    FillerParameter data_filler_param;
    data_filler_param.set_std(1);
    GaussianFiller<Dtype> data_filler(data_filler_param);
    FillerParameter targets_filler_param;
    targets_filler_param.set_min(0.0);
    targets_filler_param.set_max(1.0);
    UniformFiller<Dtype> targets_filler(targets_filler_param);
    Dtype eps = 2e-2;
    for (int i = 0; i < 100; ++i) {
      // Fill the data vector
      data_filler.Fill(this->blob_bottom_data_);
      // Fill the targets vector
      targets_filler.Fill(this->blob_bottom_targets_);
	   for (int k=0; k<blob_bottom_targets_->count(); ++k){
        if (blob_bottom_targets_->cpu_data()[k] <= 0.5)
          blob_bottom_targets_->mutable_cpu_data()[k] = 0;
        else
          blob_bottom_targets_->mutable_cpu_data()[k] = 1;
      }  //以上准备数据与label
	 
      FocalLossLayer<Dtype> layer(layer_param);
      layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
      Dtype layer_loss =
          layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
      const int count = this->blob_bottom_data_->count();
      const int num = this->blob_bottom_data_->num();
      const Dtype* blob_bottom_data = this->blob_bottom_data_->cpu_data();
      const Dtype* blob_bottom_targets =
          this->blob_bottom_targets_->cpu_data(); 
      Dtype reference_loss = kLossWeight * FocalLossReference(
          count, num, blob_bottom_data, blob_bottom_targets,alpha,gamma);
      EXPECT_NEAR(reference_loss, layer_loss, eps) << "debug: trial #" << i;
    }
  }

  Blob<Dtype>* const blob_bottom_data_;
  Blob<Dtype>* const blob_bottom_targets_;
  Blob<Dtype>* const blob_top_loss_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(FocalLossLayerTest, TestDtypesAndDevices);

TYPED_TEST(FocalLossLayerTest, TestFocalLoss){
  this->TestForward();
}

TYPED_TEST(FocalLossLayerTest, TestGradient) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  const Dtype kLossWeight = 3.7;
  layer_param.add_loss_weight(kLossWeight);
  const Dtype alpha=-1;
  const Dtype  gamma=2;
  FocalLossParameter* focal_loss_param = layer_param.mutable_focal_loss_param();
  focal_loss_param->set_alpha(alpha);
  focal_loss_param->set_gamma(gamma);
  FocalLossLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  GradientChecker<Dtype> checker(1e-2, 1e-2, 1701);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_, 0);
}

TYPED_TEST(FocalLossLayerTest, TestIgnoreGradient) {
  typedef typename TypeParam::Dtype Dtype;
  FillerParameter data_filler_param;
  data_filler_param.set_std(1);
  GaussianFiller<Dtype> data_filler(data_filler_param);
  data_filler.Fill(this->blob_bottom_data_);
  LayerParameter layer_param;
  LossParameter* loss_param = layer_param.mutable_loss_param();
  const Dtype alpha=-1;
  const Dtype  gamma=2;
  FocalLossParameter* focal_loss_param = layer_param.mutable_focal_loss_param();
  focal_loss_param->set_alpha(alpha);
  focal_loss_param->set_gamma(gamma);
  loss_param->set_ignore_label(-1);
  Dtype* target = this->blob_bottom_targets_->mutable_cpu_data();
  const int count = this->blob_bottom_targets_->count();
  // Ignore half of targets, then check that diff of this half is zero,
  // while the other half is nonzero.
  caffe_set(count / 2, Dtype(-1), target);
  FocalLossLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  vector<bool> propagate_down(2);
  propagate_down[0] = true;
  propagate_down[1] = false;
  layer.Backward(this->blob_top_vec_, propagate_down, this->blob_bottom_vec_);
  const Dtype* diff = this->blob_bottom_data_->cpu_diff();
  for (int i = 0; i < count / 2; ++i) {
    EXPECT_FLOAT_EQ(diff[i], 0.);
    EXPECT_NE(diff[i + count / 2], 0.);
  }
}


}  // namespace caffe
