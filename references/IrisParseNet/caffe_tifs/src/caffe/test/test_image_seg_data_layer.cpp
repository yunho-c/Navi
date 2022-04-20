#ifdef USE_OPENCV
#include <map>
#include <string>
#include <vector>
#include <algorithm>  

#include "opencv2/core/core.hpp"  
#include "opencv2/highgui/highgui.hpp"

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/image_seg_data_layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/io.hpp"

#include "caffe/test/test_caffe_main.hpp"

namespace caffe {

template <typename TypeParam>
class ImageSegDataLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  ImageSegDataLayerTest()
      : seed_(1701),
        blob_top_data_(new Blob<Dtype>()),
        blob_top_label_(new Blob<Dtype>()),
        blob_top_edge_(new Blob<Dtype>()){}
  virtual void SetUp() {
    blob_top_vec_.push_back(blob_top_data_);
    blob_top_vec_.push_back(blob_top_label_);
	blob_top_vec_.push_back(blob_top_edge_);
    Caffe::set_random_seed(seed_);
    filename_="test_mask_edge_single.txt";    //内含两幅图像
	filename_reshape_="test_mask_edge_single.txt";
    filename_crop_="test_mask_edge_single.txt";
	root_file_="/home/caiyong.wang/program/myhed/test/";
	minLabel=0;
	maxLabel=0;
  }
  
  virtual ~ImageSegDataLayerTest() {
    delete blob_top_data_;
    delete blob_top_label_;
	delete blob_top_edge_;
  }
  void  countScaleImageMinMaxLabel(  Blob<Dtype>* const  Label)
  { //必须假定label为8位的图像，且个数为1,channel为1
       int height= Label->height();
	   int width = Label->width();
     EXPECT_EQ(1, Label->num());
     EXPECT_EQ(1, Label->channels()); 
	 cv::Mat LableMat=cv::Mat::zeros(height,width,CV_8UC1);
 
	   for(int h=0;h<height;h++)
	   	{
	   		for(int w=0;w<width;w++)
	   			{
	   			   LableMat.at<uchar>(h,w)= static_cast<uchar>(Label->data_at(0,0,h,w));
	   			}
	   	}
	  double minVal=0.0,maxVal=0.0; 
      minMaxIdx(LableMat, &minVal, &maxVal);
  	  minLabel=minVal;  
	  maxLabel=maxVal;
  	}
  void  countMinMaxLabel(  Blob<Dtype>* const  Label)
  { //必须假定label为8位的图像，且个数为1,channel为1
      const Dtype* target=Label->cpu_data();
	  int num=Label->count();
      std::vector<Dtype> temp(target,target+num);
	  std::sort(temp.begin(),temp.end());
	  typename std::vector<Dtype>::iterator it = std::unique(temp.begin(), temp.end());
	  temp.erase(it, temp.end());
	  EXPECT_EQ(2, temp.size()); 
	  minLabel=temp[0];
	  maxLabel=temp[1];
  	} 
  int seed_;
  string filename_;
  string filename_reshape_;
  string filename_crop_;
  string root_file_; 
  int  minLabel;
  int  maxLabel;
  Blob<Dtype>* const blob_top_data_;
  Blob<Dtype>* const blob_top_label_;
  Blob<Dtype>* const blob_top_edge_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(ImageSegDataLayerTest, TestDtypesAndDevices);

TYPED_TEST(ImageSegDataLayerTest, TestRead) {
	  typedef typename TypeParam::Dtype Dtype;
	  LayerParameter param;
	  ImageDataParameter* image_data_param = param.mutable_image_data_param();
	  image_data_param->set_batch_size(1);
	  image_data_param->set_source(this->filename_.c_str());
	  image_data_param->set_root_folder(this->root_file_.c_str());
	  image_data_param->set_shuffle(true);
	   image_data_param->set_label_type(ImageDataParameter_LabelType_PIXEL); 
    image_data_param->set_ignore_label(255);
	   TransformationParameter* transform_param = param.mutable_transform_param();
	  transform_param->set_mirror(false);
	  transform_param->add_mean_value(104.008);
	  transform_param->add_mean_value(116.669);
	  transform_param->add_mean_value(122.675); 
	  ImageSegDataLayer<Dtype> layer(param);
	  
	  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
	  EXPECT_EQ(this->blob_top_data_->num(), 1);
	  EXPECT_EQ(this->blob_top_data_->channels(), 3);
	  EXPECT_EQ(this->blob_top_data_->height(), 300);
	  EXPECT_EQ(this->blob_top_data_->width(), 400);
	  EXPECT_EQ(this->blob_top_label_->num(), 1);
	  EXPECT_EQ(this->blob_top_label_->channels(), 1);
	  EXPECT_EQ(this->blob_top_label_->height(), 300);
	  EXPECT_EQ(this->blob_top_label_->width(), 400);
	  EXPECT_EQ(this->blob_top_edge_->num(), 1);
	  EXPECT_EQ(this->blob_top_edge_->channels(), 1);
	  EXPECT_EQ(this->blob_top_edge_->height(), 300);
	  EXPECT_EQ(this->blob_top_edge_->width(), 400);
	  // Go through the data twice
	  for (int iter = 0; iter < 2; ++iter) {
		layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
		this->countMinMaxLabel(this->blob_top_label_);
		EXPECT_EQ(0, this->minLabel);
		EXPECT_EQ(1, this->maxLabel);
		this->countMinMaxLabel(this->blob_top_edge_);
		EXPECT_EQ(0, this->minLabel);
		EXPECT_EQ(1, this->maxLabel);
	  } 
}

 TYPED_TEST(ImageSegDataLayerTest, TestResize) {
	 typedef typename TypeParam::Dtype Dtype;
	 LayerParameter param;
	 ImageDataParameter* image_data_param = param.mutable_image_data_param();
	 image_data_param->set_batch_size(1);
	 image_data_param->set_source(this->filename_.c_str());
	 image_data_param->set_root_folder(this->root_file_.c_str());
	 image_data_param->set_new_height(256);
	 image_data_param->set_new_width(256);
	 image_data_param->set_shuffle(false);
	 image_data_param->set_label_type(ImageDataParameter_LabelType_PIXEL); 
	 image_data_param->set_ignore_label(255);
	 TransformationParameter* transform_param = param.mutable_transform_param();
	 transform_param->set_mirror(false);
	  transform_param->add_mean_value(104.008);
	  transform_param->add_mean_value(116.669);
	  transform_param->add_mean_value(122.675); 
	 ImageSegDataLayer<Dtype> layer(param);
	 layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
	 EXPECT_EQ(this->blob_top_data_->num(), 1);
	 EXPECT_EQ(this->blob_top_data_->channels(), 3);
	 EXPECT_EQ(this->blob_top_data_->height(), 256);
	 EXPECT_EQ(this->blob_top_data_->width(), 256);
	 EXPECT_EQ(this->blob_top_label_->num(), 1);
	 EXPECT_EQ(this->blob_top_label_->channels(), 1);
	 EXPECT_EQ(this->blob_top_label_->height(), 256);
	 EXPECT_EQ(this->blob_top_label_->width(), 256);
	  EXPECT_EQ(this->blob_top_edge_->num(), 1);
	  EXPECT_EQ(this->blob_top_edge_->channels(), 1);
	  EXPECT_EQ(this->blob_top_edge_->height(), 256);
	  EXPECT_EQ(this->blob_top_edge_->width(), 256);
	 // Go through the data twice
	 for (int iter = 0; iter < 2; ++iter) {
		layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
		 this->countScaleImageMinMaxLabel(this->blob_top_label_);
		EXPECT_EQ(0, this->minLabel);
		EXPECT_GE(1, this->maxLabel);
		this->countScaleImageMinMaxLabel(this->blob_top_edge_);
		EXPECT_EQ(0, this->minLabel);
		EXPECT_GE(1, this->maxLabel);
	 }
}

TYPED_TEST(ImageSegDataLayerTest, TestReshape) {
	 typedef typename TypeParam::Dtype Dtype;
	 LayerParameter param;
	 ImageDataParameter* image_data_param = param.mutable_image_data_param();
	 image_data_param->set_batch_size(2);
	 image_data_param->set_source(this->filename_reshape_.c_str());
	 image_data_param->set_root_folder(this->root_file_.c_str());
	 image_data_param->set_shuffle(true);
	  image_data_param->set_label_type(ImageDataParameter_LabelType_PIXEL);
	 image_data_param->set_ignore_label(255);
	 TransformationParameter* transform_param = param.mutable_transform_param();
	 transform_param->set_mirror(false);
	 transform_param->add_mean_value(104.008);
	  transform_param->add_mean_value(116.669);
	  transform_param->add_mean_value(122.675); 
	 ImageSegDataLayer<Dtype> layer(param);
	 layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
	 EXPECT_EQ(this->blob_top_label_->num(),2);
	 EXPECT_EQ(this->blob_top_label_->channels(), 1);
	 EXPECT_EQ(this->blob_top_label_->height(), 300);
	 EXPECT_EQ(this->blob_top_label_->width(), 400);
	 EXPECT_EQ(this->blob_top_edge_->num(), 2);
	 EXPECT_EQ(this->blob_top_edge_->channels(), 1);
	 EXPECT_EQ(this->blob_top_edge_->height(), 300);
	 EXPECT_EQ(this->blob_top_edge_->width(), 400);
	 // 0_1_1004.jpg
	 layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
	 EXPECT_EQ(this->blob_top_data_->num(), 2);
	 EXPECT_EQ(this->blob_top_data_->channels(), 3);
	 EXPECT_EQ(this->blob_top_data_->height(), 300);
	 EXPECT_EQ(this->blob_top_data_->width(), 400);
}


TYPED_TEST(ImageSegDataLayerTest, TestCrop) {
	 typedef typename TypeParam::Dtype Dtype;
	 LayerParameter param;
	 ImageDataParameter* image_data_param = param.mutable_image_data_param();
	 image_data_param->set_batch_size(1);
	 image_data_param->set_source(this->filename_crop_.c_str());
	 image_data_param->set_root_folder(this->root_file_.c_str());
	 image_data_param->set_shuffle(false);
	 image_data_param->set_label_type(ImageDataParameter_LabelType_PIXEL);
	  image_data_param->set_ignore_label(255);
	  TransformationParameter* transform_param = param.mutable_transform_param();
	  transform_param->set_mirror(false);
    //transform_param->set_crop_size(513);
    param.set_phase(TRAIN);
    transform_param->set_crop_width(513);
  	transform_param->set_crop_height(300);
    transform_param->add_mean_value(104.008);
	  transform_param->add_mean_value(116.669);
	  transform_param->add_mean_value(122.675); 
	 ImageSegDataLayer<Dtype> layer(param);
	 layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
	 EXPECT_EQ(this->blob_top_label_->num(), 1);
	 EXPECT_EQ(this->blob_top_label_->channels(), 1);
	 EXPECT_EQ(this->blob_top_label_->height(), 300);
	 EXPECT_EQ(this->blob_top_label_->width(), 513);
	 EXPECT_EQ(this->blob_top_edge_->num(), 1);
	 EXPECT_EQ(this->blob_top_edge_->channels(), 1);
	 EXPECT_EQ(this->blob_top_edge_->height(), 300);
	 EXPECT_EQ(this->blob_top_edge_->width(), 513);
	  // 0_1_1004.jpg
	 layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
	 EXPECT_EQ(this->blob_top_data_->num(), 1);
	 EXPECT_EQ(this->blob_top_data_->channels(), 3);
	 EXPECT_EQ(this->blob_top_data_->height(), 300);
	 EXPECT_EQ(this->blob_top_data_->width(), 513);
	 this->countScaleImageMinMaxLabel(this->blob_top_label_);
	 EXPECT_EQ(0, this->minLabel);
	 EXPECT_EQ(255, this->maxLabel);
	  this->countScaleImageMinMaxLabel(this->blob_top_edge_);
	 EXPECT_EQ(0, this->minLabel);
	 EXPECT_EQ(255, this->maxLabel);
	 // 0_1_1039.jpg
	 layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
	 EXPECT_EQ(this->blob_top_data_->num(), 1);
	 EXPECT_EQ(this->blob_top_data_->channels(), 3);
	 EXPECT_EQ(this->blob_top_data_->height(), 300);
	 EXPECT_EQ(this->blob_top_data_->width(), 513);
	 this->countScaleImageMinMaxLabel(this->blob_top_label_);
	 EXPECT_EQ(0, this->minLabel);
	 EXPECT_EQ(255, this->maxLabel);
	  this->countScaleImageMinMaxLabel(this->blob_top_edge_);
	 EXPECT_EQ(0, this->minLabel);
	 EXPECT_EQ(255, this->maxLabel);
}

}  // namespace caffe
#endif  // USE_OPENCV
