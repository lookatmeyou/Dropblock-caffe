#include <vector>

#include "caffe/layers/dropblock_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
__global__ void DropblockForward(const int n, const Dtype* in,
    const float* mask, const float scale,
    Dtype* out) {
  CUDA_KERNEL_LOOP(index, n) {
    out[index] = in[index] * mask[index] * scale;
  }
}

__global__ void DropblockExpandMaskGPU(const int n, float* tmp_mask, float* mask,
  float gamma, int block_size, int batch_size, int channels, int feat_size) {
  const int tborder_size = feat_size - block_size + 1;
  // const int tcount = block_size * block_size;
  const int ds1 = channels * tborder_size * tborder_size;
  const int ds2 = tborder_size * tborder_size;
  CUDA_KERNEL_LOOP(index, n) {
    if (tmp_mask[index] <= gamma) {
      const int b = index / ds1;
      const int c = (index % ds1) / ds2;
      const int i = ((index % ds1) % ds2) / tborder_size;
      const int j = ((index % ds1) % ds2) % tborder_size;
      const int p = (b * channels + c) * feat_size * feat_size;
      int tp = p + i * feat_size + j;
      for (int y = 0; y < block_size; ++y, tp += feat_size) {
        memset(mask + tp, 0, block_size * sizeof(float));
      }
    }
  }
}

template <typename Dtype>
void DropblockLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  const int count = bottom[0]->count();
  if (this->phase_ == TRAIN) {
    float* mask =
        static_cast<float*>(rand_vec_.mutable_gpu_data());
    float* tmp_mask =
        static_cast<float*>(tmp_rand_vec_.mutable_gpu_data());
    Dtype t_keep_prob = 1.0 - (1.0 - keep_prob_) * Caffe::get_current_iter() / Caffe::get_max_iter();
    Dtype gamma_ = (1.0 - t_keep_prob) * feat_size_ * feat_size_ /
        (block_size_ * block_size_ * (feat_size_ - block_size_ + 1) * (feat_size_ - block_size_ + 1));
    const int tcount = tmp_rand_vec_.count();
    caffe_gpu_rng_uniform(tcount, 0.0f, 1.0f, tmp_mask);

    caffe_gpu_set<float>(count, 1, mask);
    DropblockExpandMaskGPU<<<CAFFE_GET_BLOCKS(tcount), CAFFE_CUDA_NUM_THREADS>>>(
      tcount, tmp_mask, mask, gamma_, block_size_,
      bottom[0]->shape(0), bottom[0]->shape(1), feat_size_
    );

    scale_ = 1.0 * count / rand_vec_.asum_data();
    DropblockForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, bottom_data, mask, scale_, top_data);
    CUDA_POST_KERNEL_CHECK;
  } else {
    caffe_copy(count, bottom_data, top_data);
  }
}

template <typename Dtype>
__global__ void DropblockBackward(const int n, const Dtype* in_diff,
    const float* mask, const float scale, Dtype* out_diff) {
  CUDA_KERNEL_LOOP(index, n) {
    out_diff[index] = in_diff[index] * scale * mask[index];
  }
}

template <typename Dtype>
void DropblockLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    const Dtype* top_diff = top[0]->gpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    if (this->phase_ == TRAIN) {
      const float* mask =
          static_cast<const float*>(rand_vec_.gpu_data());
      const int count = bottom[0]->count();
      // NOLINT_NEXT_LINE(whitespace/operators)
      DropblockBackward<Dtype><<<CAFFE_GET_BLOCKS(count),
        CAFFE_CUDA_NUM_THREADS>>>(
          count, top_diff, mask, scale_, bottom_diff);
      CUDA_POST_KERNEL_CHECK;
    } else {
      caffe_copy(top[0]->count(), top_diff, bottom_diff);
    }
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(DropblockLayer);

}  // namespace caffe
