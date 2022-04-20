#ifndef CAFFE_IMAGE_SEG_DATA_LAYER_HPP_
#define CAFFE_IMAGE_SEG_DATA_LAYER_HPP_

#include <string>
#include <utility>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/data_transformer.hpp"
#include "caffe/internal_thread.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/base_data_layer.hpp"
#include "caffe/proto/caffe.pb.h"


namespace caffe {

struct Data_Label_Edge  //定义image,mask,edge的地址
{
	Data_Label_Edge(const std::string&  data, const std::string& label,const std::string& edge ) {
		this->data= data;
		this->label = label;
		this->edge = edge;
	};
	Data_Label_Edge() {
		this->data = "";
		this->label = "";
		this->edge = "";
	};
	std::string data;
	std::string label;
	std::string edge;
};

template <typename Dtype>
class ImageSegDataLayer : public BasePrefetchingLabelmapDataLayer<Dtype> {
 public:
  explicit ImageSegDataLayer(const LayerParameter& param)
    : BasePrefetchingLabelmapDataLayer<Dtype>(param) {}
  virtual ~ImageSegDataLayer();
  virtual void DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "ImageSegData"; }
  virtual inline int ExactNumBottomBlobs() const { return 0; }
  virtual inline int ExactNumTopBlobs() const { return 3; }


 protected:
  virtual void ShuffleImages();
  virtual void load_batch(LabelmapBatch<Dtype>* batch);

  
  shared_ptr<Caffe::RNG> prefetch_rng_;
  vector<Data_Label_Edge> lines_;
  int lines_id_;
};

}  // namespace caffe

#endif  // CAFFE_IMAGE_SEG_DATA_LAYER_HPP_
