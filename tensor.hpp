#ifndef TENSOR_HPP
#define TENSOR_HPP

#include <iostream>
#include <memory>
#include <vector>
#include <algorithm>

class Tensor {
public:
    Tensor() : Tensor(0, 0, 0, 0) {}

    Tensor(size_t n) : Tensor(n, 1, 1, 1) {}

    Tensor(size_t n, size_t c) : Tensor(n, c, 1, 1) {}

    Tensor(size_t n, size_t c, size_t h) : Tensor(n, c, h, 1) {}

    Tensor(size_t n, size_t c, size_t h, size_t w)
        : N(n), C(c), H(h), W(w),
          offset_(0),
          data_(std::make_shared<std::vector<float>>(n * c * h * w)) {}

    Tensor(size_t n, size_t c, size_t h, size_t w, size_t offset,
           std::shared_ptr<std::vector<float>> data)
        : N(n), C(c), H(h), W(w),
          offset_(offset),
          data_(data) {}

    bool empty() const {
        return !data_ || data_->empty();
    }

    float* data() {
        return data_ ? data_->data() : nullptr;
    }

    const float* data() const {
        return data_ ? data_->data() : nullptr;
    }

    void fill(float value) {
        if (empty()) return;
        std::fill(data_->begin() + offset_,
                  data_->begin() + offset_ + N * C * H * W,
                  value);
    }


    size_t N_() const { return N; }
    size_t C_() const { return C; }
    size_t H_() const { return H; }
    size_t W_() const { return W; }


    size_t N() const { return this->N; }
    size_t C() const { return this->C; }
    size_t H() const { return this->H; }
    size_t W() const { return this->W; }



    float& operator()(size_t n, size_t c = 0, size_t h = 0, size_t w = 0) {
        size_t idx = ((n * C + c) * H + h) * W + w;
        return (*data_)[offset_ + idx];
    }

    const float& operator()(size_t n, size_t c = 0, size_t h = 0, size_t w = 0) const {
        size_t idx = ((n * C + c) * H + h) * W + w;
        return (*data_)[offset_ + idx];
    }


    Tensor slice(size_t idx, size_t num) const {
        size_t off = offset_ + idx * C * H * W;
        return Tensor(num, C, H, W, off, data_);
    }

    std::ostream& write(std::ostream& os) const {
        return os << N << "x" << C << "x" << H << "x" << W;
    }


    size_t N, C, H, W;

private:
    size_t offset_;
    std::shared_ptr<std::vector<float>> data_;
};

inline std::ostream& operator<<(std::ostream& os, const Tensor& t) {
    return t.write(os);
}

#endif

