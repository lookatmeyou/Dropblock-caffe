// TODO (sergeyk): effect should not be dependent on phase. wasted memcpy.

#include <vector>

#include "caffe/layers/dropblock_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void DropblockLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  NeuronLayer<Dtype>::LayerSetUp(bottom, top);
  keep_prob_ = this->layer_param_.dropblock_param().keep_prob();
  block_size_ = this->layer_param_.dropblock_param().block_size();
  DCHECK(keep_prob_ > 0.);
  DCHECK(keep_prob_ < 1.);
}

template <typename Dtype>
void DropblockLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  NeuronLayer<Dtype>::Reshape(bottom, top);
  // Set up the cache for random number generation
  // ReshapeLike does not work because rand_vec_ is of Dtype uint
  rand_vec_.Reshape(bottom[0]->shape());
  feat_size_ = bottom[0]->shape(2);
  DCHECK(feat_size_ >= block_size_);
  vector<int> tmp_shape = bottom[0]->shape();
  tmp_shape[2] = feat_size_ - block_size_ + 1;
  tmp_shape[3] = feat_size_ - block_size_ + 1;
  tmp_rand_vec_.Reshape(tmp_shape);
}

void DropblockFillZeroCPU(float* mask, int x, int y, int feat_size, int block_size) {
  int p = y * feat_size + x;
  for (int i = 0; i < block_size; ++i, p += feat_size) {
    caffe_set<float>(block_size, 0, mask + p);
  }
}

void DropblockExpandMaskCPU(float* tmp_mask, float* mask, float gamma,
  int block_size, int batch_size, int channels, int feat_size) {
  caffe_set<float>(batch_size * channels * feat_size * feat_size, 1, mask);
  int tborder_size = feat_size - block_size + 1;
  int tp = 0, p = 0;
  for (int b = 0; b < batch_size; ++b) {
    for (int c = 0; c < channels; ++c, tp += tborder_size * tborder_size, p += feat_size * feat_size) {
      for (int i = 0; i < tborder_size; ++i) {
        for (int j = 0; j < tborder_size; ++j) {
          if (tmp_mask[tp + i * tborder_size + j] <= gamma) {
            DropblockFillZeroCPU(mask + p, j, i, feat_size, block_size);
          }
        }
      }
    }
  }
}

template <typename Dtype>
void DropblockLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  const int count = bottom[0]->count();
  if (this->phase_ == TRAIN) {
    float* mask = rand_vec_.mutable_cpu_data();
    float* tmp_mask = tmp_rand_vec_.mutable_cpu_data();
    Dtype t_keep_prob = 1.0 - (1.0 - keep_prob_) * Caffe::get_current_iter() / Caffe::get_max_iter();
    Dtype gamma_ = (1.0 - t_keep_prob) * feat_size_ * feat_size_ /
        (block_size_ * block_size_ * (feat_size_ - block_size_ + 1) * (feat_size_ - block_size_ + 1));
    // Create random numbers
    caffe_rng_uniform<float>(tmp_rand_vec_.count(), 0, 1, tmp_mask);
    DropblockExpandMaskCPU(tmp_mask, mask, gamma_, block_size_, bottom[0]->shape(0), bottom[0]->shape(1), feat_size_);
    Dtype scale_ = 1.0 * count / rand_vec_.asum_data();
    for (int i = 0; i < count; ++i) {
      top_data[i] = bottom_data[i] * mask[i] * scale_;
    }
  } else {
    caffe_copy(count, bottom_data, top_data);
  }
}

template <typename Dtype>
void DropblockLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    const Dtype* top_diff = top[0]->cpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    if (this->phase_ == TRAIN) {
      const float* mask = rand_vec_.cpu_data();
      const int count = bottom[0]->count();
      for (int i = 0; i < count; ++i) {
        bottom_diff[i] = top_diff[i] * mask[i] * scale_;
      }
    } else {
      caffe_copy(top[0]->count(), top_diff, bottom_diff);
    }
  }
}


#ifdef CPU_ONLY
STUB_GPU(DropblockLayer);
#endif

INSTANTIATE_CLASS(DropblockLayer);
REGISTER_LAYER_CLASS(Dropblock);

}  // namespace caffe
