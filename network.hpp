#ifndef NETWORK_HPP
#define NETWORK_HPP

#include "tensor.hpp"

#include <cstdint>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <cmath>
#include <algorithm>

enum class LayerType : uint8_t {
    Conv2d = 0,
    Linear,
    MaxPool2d,
    ReLu,
    SoftMax,
    Flatten
};

std::ostream& operator<<(std::ostream& os, LayerType layer_type) {
    switch (layer_type) {
        case LayerType::Conv2d: return os << "Conv2d";
        case LayerType::Linear: return os << "Linear";
        case LayerType::MaxPool2d: return os << "MaxPool2d";
        case LayerType::ReLu: return os << "ReLu";
        case LayerType::SoftMax: return os << "SoftMax";
        case LayerType::Flatten: return os << "Flatten";
    }
    return os;
}

class Layer {
public:
    Layer(LayerType t) : layer_type_(t), input_(), weights_(), bias_(), output_() {}
    virtual ~Layer() = default;

    virtual void fwd() = 0;
    virtual void read_weights_bias(std::ifstream& is) = 0;

    void set_input(const Tensor& x) { input_ = x; }
    const Tensor& output() const { return output_; }

protected:
    const LayerType layer_type_;
    Tensor input_;
    Tensor weights_;
    Tensor bias_;
    Tensor output_;
};

class Conv2d : public Layer {
public:
    Conv2d(size_t in_c, size_t out_c, size_t k, size_t s=1, size_t p=0)
        : Layer(LayerType::Conv2d),
          in_channels_(in_c),
          out_channels_(out_c),
          kernel_size_(k),
          stride_(s),
          pad_(p)
    {
        weights_ = Tensor(out_channels_, in_channels_, kernel_size_, kernel_size_);
        bias_ = Tensor(1, out_channels_, 1, 1);
    }

    void fwd() override {
        const Tensor& x = input_;
        size_t N = x.N;
        size_t H = x.H;
        size_t W = x.W;

        size_t out_h = (H + 2 * pad_ - kernel_size_) / stride_ + 1;
        size_t out_w = (W + 2 * pad_ - kernel_size_) / stride_ + 1;

        Tensor y(N, out_channels_, out_h, out_w);

        for (size_t n = 0; n < N; ++n)
            for (size_t oc = 0; oc < out_channels_; ++oc)
                for (size_t oh = 0; oh < out_h; ++oh)
                    for (size_t ow = 0; ow < out_w; ++ow) {
                        float sum = bias_(0, oc, 0, 0);
                        for (size_t ic = 0; ic < in_channels_; ++ic)
                            for (size_t kh = 0; kh < kernel_size_; ++kh)
                                for (size_t kw = 0; kw < kernel_size_; ++kw) {
                                    int ih = (int)(oh * stride_ + kh) - (int)pad_;
                                    int iw = (int)(ow * stride_ + kw) - (int)pad_;
                                    if (ih >= 0 && ih < (int)H && iw >= 0 && iw < (int)W)
                                        sum += x(n, ic, ih, iw) * weights_(oc, ic, kh, kw);
                                }
                        y(n, oc, oh, ow) = sum;
                    }

        output_ = y;
    }

    void read_weights_bias(std::ifstream& is) override {
        is.read(reinterpret_cast<char*>(weights_.data()), sizeof(float)*weights_.N*weights_.C*weights_.H*weights_.W);
        is.read(reinterpret_cast<char*>(bias_.data()), sizeof(float)*bias_.N*bias_.C*bias_.H*bias_.W);
    }

private:
    size_t in_channels_;
    size_t out_channels_;
    size_t kernel_size_;
    size_t stride_;
    size_t pad_;
};

class Linear : public Layer {
public:
    Linear(size_t in_f, size_t out_f)
        : Layer(LayerType::Linear), in_features_(in_f), out_features_(out_f)
    {
        weights_ = Tensor(out_features_, in_features_, 1, 1);
        bias_ = Tensor(1, out_features_, 1, 1);
    }

    void fwd() override {
        const Tensor& x = input_;
        size_t N = x.N;

        Tensor y(N, out_features_, 1, 1);

        for (size_t n = 0; n < N; ++n)
            for (size_t o = 0; o < out_features_; ++o) {
                float sum = bias_(0, o, 0, 0);
                for (size_t i = 0; i < in_features_; ++i)
                    sum += x(n, i, 0, 0) * weights_(o, i, 0, 0);
                y(n, o, 0, 0) = sum;
            }

        output_ = y;
    }

    void read_weights_bias(std::ifstream& is) override {
        is.read(reinterpret_cast<char*>(weights_.data()), sizeof(float)*weights_.N*weights_.C*weights_.H*weights_.W);
        is.read(reinterpret_cast<char*>(bias_.data()), sizeof(float)*bias_.N*bias_.C*bias_.H*bias_.W);
    }

private:
    size_t in_features_;
    size_t out_features_;
};

class MaxPool2d : public Layer {
public:
    MaxPool2d(size_t k, size_t s=1, size_t p=0)
        : Layer(LayerType::MaxPool2d), kernel_size_(k), stride_(s), pad_(p) {}

    void fwd() override {
        const Tensor& x = input_;
        size_t N = x.N, C = x.C, H = x.H, W = x.W;

        size_t out_h = (H - kernel_size_) / stride_ + 1;
        size_t out_w = (W - kernel_size_) / stride_ + 1;

        Tensor y(N, C, out_h, out_w);

        for (size_t n = 0; n < N; ++n)
            for (size_t c = 0; c < C; ++c)
                for (size_t oh = 0; oh < out_h; ++oh)
                    for (size_t ow = 0; ow < out_w; ++ow) {
                        float m = -1e9f;
                        for (size_t kh = 0; kh < kernel_size_; ++kh)
                            for (size_t kw = 0; kw < kernel_size_; ++kw) {
                                size_t ih = oh * stride_ + kh;
                                size_t iw = ow * stride_ + kw;
                                m = std::max(m, x(n,c,ih,iw));
                            }
                        y(n,c,oh,ow) = m;
                    }

        output_ = y;
    }

    void read_weights_bias(std::ifstream&) override {}

private:
    size_t kernel_size_;
    size_t stride_;
    size_t pad_;
};

class ReLu : public Layer {
public:
    ReLu() : Layer(LayerType::ReLu) {}

    void fwd() override {
        const Tensor& x = input_;
        Tensor y(x.N, x.C, x.H, x.W);

        for (size_t n = 0; n < x.N; ++n)
            for (size_t c = 0; c < x.C; ++c)
                for (size_t h = 0; h < x.H; ++h)
                    for (size_t w = 0; w < x.W; ++w)
                        y(n,c,h,w) = std::max(0.0f, x(n,c,h,w));

        output_ = y;
    }

    void read_weights_bias(std::ifstream&) override {}
};

class SoftMax : public Layer {
public:
    SoftMax() : Layer(LayerType::SoftMax) {}

    void fwd() override {
        const Tensor& x = input_;
        Tensor y(x.N, x.C, x.H, x.W);

        for (size_t n = 0; n < x.N; ++n)
            for (size_t h = 0; h < x.H; ++h)
                for (size_t w = 0; w < x.W; ++w) {
                    float m = -1e9f;
                    for (size_t c = 0; c < x.C; ++c)
                        m = std::max(m, x(n,c,h,w));
                    float s = 0.0f;
                    for (size_t c = 0; c < x.C; ++c) {
                        float v = std::exp(x(n,c,h,w)-m);
                        y(n,c,h,w) = v;
                        s += v;
                    }
                    for (size_t c = 0; c < x.C; ++c)
                        y(n,c,h,w) /= s;
                }

        output_ = y;
    }

    void read_weights_bias(std::ifstream&) override {}
};

class Flatten : public Layer {
public:
    Flatten() : Layer(LayerType::Flatten) {}

    void fwd() override {
        const Tensor& x = input_;
        Tensor y(x.N, x.C*x.H*x.W, 1, 1);

        for (size_t n = 0; n < x.N; ++n) {
            size_t idx = 0;
            for (size_t c = 0; c < x.C; ++c)
                for (size_t h = 0; h < x.H; ++h)
                    for (size_t w = 0; w < x.W; ++w)
                        y(n, idx++, 0, 0) = x(n,c,h,w);
        }

        output_ = y;
    }

    void read_weights_bias(std::ifstream&) override {}
};

class NeuralNetwork {
public:
    NeuralNetwork(bool dbg=false) : debug_(dbg) {}

    void add(Layer* l) { layers_.push_back(l); }

    void load(std::string file) {
        std::ifstream is(file, std::ios::binary);
        for (auto* l : layers_) l->read_weights_bias(is);
    }

    Tensor predict(Tensor x) {
        for (auto* l : layers_) {
            l->set_input(x);
            l->fwd();
            x = l->output();
        }
        return x;
    }

private:
    bool debug_;
    std::vector<Layer*> layers_;
};

#endif

