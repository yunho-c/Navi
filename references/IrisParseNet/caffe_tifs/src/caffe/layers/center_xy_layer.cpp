#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include <string> 

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/imgproc/imgproc.hpp>

#include "caffe/layers/center_xy_layer.hpp"

namespace caffe {

template <typename Dtype>
void CenterXYLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
 
  const int num_images = bottom[0]->num();
  const int label_height = bottom[0]->height();
  const int label_width = bottom[0]->width();
	
  top[0]->Reshape(num_images, 1, label_height,label_width);
  top[1]->Reshape(num_images, 1, label_height,label_width);

}


template <typename Dtype>
void CenterXYLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
		  
    const int num_images = bottom[0]->num();
    const int label_height = bottom[0]->height();
    const int label_width = bottom[0]->width();
    const int label_img_size = label_height * label_width*1;
	
	 
	Dtype* x_ptr = top[0]->mutable_cpu_data();
	Dtype* y_ptr = top[1]->mutable_cpu_data();
     
	   for (int idx_img = 0; idx_img < num_images; idx_img++)
       {
		 for (int i = 0; i < label_height; i++)
			{
				for (int j = 0; j < label_width; j++)
				{
					int image_idx = idx_img * label_img_size + i * label_width + j;
					x_ptr[image_idx]= (j*1.0)/label_width;
					y_ptr[image_idx]= (i*1.0)/label_height;						
				}           
			}
	
	  }  
	   
}

INSTANTIATE_CLASS(CenterXYLayer);
REGISTER_LAYER_CLASS(CenterXY);

}  // namespace caffe
