#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include <string> 

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/imgproc/imgproc.hpp>

#include "caffe/layers/new_center_layer.hpp"

namespace caffe {

template <typename Dtype>
void NewCenterLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  CHECK_NE(top[0], bottom[0]) << this->type() << " Layer does not "
      "allow in-place computation.";
  const int num_images = bottom[0]->num();
  const int num_channels = bottom[0]->channels();
  CHECK_EQ(num_channels, 1) << "Bottom channel must be 1.";
	
  vector<int> top_shape(2);
  top_shape[0]=num_images;
  top_shape[1]=1;
  top[0]->Reshape(top_shape);
  top_shape[1]=2;
  top[1]->Reshape(top_shape);
  top[2]->ReshapeLike(*bottom[0]);
  top[3]->ReshapeLike(*bottom[0]); 
  CHECK_EQ(bottom[0]->count(),  top[2]->count());
  CHECK_EQ(bottom[0]->count(),  top[3]->count());


}

cv::Point find_point2(const cv::Mat& src,uchar data=1)
{
	int  width = src.cols;
	int height = src.rows;
	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			uchar value = src.at<uchar>(i, j);  //等价
			if (data == value)
				return cv::Point(j,i);
		}
	}
	return cv::Point(-1, -1);
}


template <typename Dtype>
void NewCenterLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
		  
    const Dtype* gt_label = bottom[0]->cpu_data(); 
    const int num_images = bottom[0]->num();
    const int label_height = bottom[0]->height();
    const int label_width = bottom[0]->width();
    const int num_channels = bottom[0]->channels();
	const int label_channel_size = label_height * label_width;
    const int label_img_size = label_channel_size * num_channels;
	
	 
	Dtype* label_ptr = top[0]->mutable_cpu_data();
	Dtype* target_ptr = top[1]->mutable_cpu_data();
	Dtype* x_ptr = top[2]->mutable_cpu_data();
	Dtype* y_ptr = top[3]->mutable_cpu_data();
     
	   for (int idx_img = 0; idx_img < num_images; idx_img++)
     {
        for (int idx_ch = 0; idx_ch < num_channels; idx_ch++)
        {
			 cv::Mat label_img(label_height,label_width, CV_8UC1);
            for (int i = 0; i < label_height; i++)
            {
                for (int j = 0; j < label_width; j++)
                {
                    int image_idx = idx_img * label_img_size + idx_ch * label_channel_size + i * label_width + j;
					const int target_value = static_cast<int>(gt_label[image_idx]);
				
				    label_img.at<uchar>(i, j)=target_value;
                }           
            }
		     cv::Point my_center = find_point2(label_img,1);
		    
			  const int target_size = 2 * num_channels;
			  int target_index = idx_img * target_size + idx_ch * 2 ;
			  if(my_center == cv::Point(-1, -1))
			  {
					label_ptr[idx_img]=0;
			  }
			  else
			  {
					label_ptr[idx_img]=1;
			  }
			  target_ptr[target_index]=my_center.x*1.0/label_width;
			  target_ptr[target_index+1]=my_center.y*1.0/label_height;
		
           
			 for (int i = 0; i < label_height; i++)
				{
					for (int j = 0; j < label_width; j++)
					{
						int image_idx = idx_img * label_img_size + idx_ch * label_channel_size + i * label_width + j;
					    x_ptr[image_idx]= (j*1.0)/label_width;
						y_ptr[image_idx]= (i*1.0)/label_height;						
					}           
				}
           }
	  }  
	   
}

#ifdef CPU_ONLY
STUB_GPU(NewCenterLayer);
#endif


INSTANTIATE_CLASS(NewCenterLayer);
REGISTER_LAYER_CLASS(NewCenter);

}  // namespace caffe
