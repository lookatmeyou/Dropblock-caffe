#include <cmath>
#include <cstring>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/dropblock_layer.hpp"
#include "google/protobuf/text_format.h"
#include "gtest/gtest.h"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

template <typename TypeParam>
class DropblockLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;
 protected:
    DropblockLayerTest()
      : blob_bottom_(new Blob<Dtype>(2, 3, 6, 6)),
        blob_top_(new Blob<Dtype>()) {
    // fill the values
    FillerParameter filler_param;
    // GaussianFiller<Dtype> filler(filler_param);
    filler_param.set_value(1);
    ConstantFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_);
    blob_bottom_vec_.push_back(blob_bottom_);
    blob_top_vec_.push_back(blob_top_);
  }
  virtual ~DropblockLayerTest() { delete blob_bottom_; delete blob_top_; }
  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(DropblockLayerTest, TestDtypesAndDevices);

TYPED_TEST(DropblockLayerTest, TestForward) {
    typedef typename TypeParam::Dtype Dtype;
    Caffe::set_current_iter(1);
    Caffe::set_max_iter(1);
    LayerParameter layer_param;
    DropblockParameter* dropblock_param = layer_param.mutable_dropblock_param();
    dropblock_param->set_keep_prob(0.8);
    dropblock_param->set_block_size(3);
    DropblockLayer<Dtype> layer(layer_param);
    layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
    layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
    // Test norm
    int num = this->blob_bottom_->num();
    int channels = this->blob_bottom_->channels();
    int height = this->blob_bottom_->height();
    int width = this->blob_bottom_->width();
    
    printf("DropblockLayerTest\n");
    printf("DropblockLayerTest bottom\n");
    for (int i = 0; i < num; ++i) {
        // Dtype norm = 0;
        for (int j = 0; j < channels; ++j) {
            for (int k = 0; k < height; ++k) {
                for (int l = 0; l < width; ++l) {
                    Dtype data = this->blob_bottom_->data_at(i, j, k, l);
                    printf("%.5f,", data);
                }
                printf("\n");
            }
            printf("\n\n");
        }
        printf("\n\n\n");
    }

    printf("DropblockLayerTest top\n");
    for (int i = 0; i < num; ++i) {
        // Dtype norm = 0;
        for (int j = 0; j < channels; ++j) {
            for (int k = 0; k < height; ++k) {
                for (int l = 0; l < width; ++l) {
                    Dtype data = this->blob_top_->data_at(i, j, k, l);
                    printf("%.5f,", data);
                }
                printf("\n");
            }
            printf("\n\n");
        }
        printf("\n\n\n");
    }
}
}
