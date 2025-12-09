#ifndef NETWORK_HPP
#define NETWORK_HPP

#include "tensor.hpp"

#include <cstdint>
#include <fstream>
#include <iostream>
#include <string>

enum class LayerType : uint8_t {
    Conv2d = 0,
    Linear,
    MaxPool2d,
    ReLu,
    SoftMax,
    Flatten
};

std::ostream& operator<< (std::ostream& os, LayerType layer_type) {
    switch (layer_type) {
        case LayerType::Conv2d:     return os << "Conv2d";
        case LayerType::Linear:     return os << "Linear";
        case LayerType::MaxPool2d:  return os << "MaxPool2d";
        case LayerType::ReLu:       return os << "ReLu";
        case LayerType::SoftMax:    return os << "SoftMax";
        case LayerType::Flatten:    return os << "Flatten";
    };
    return os << static_cast<std::uint8_t>(layer_type);
}

class Layer {
    public:
        Layer(LayerType layer_type) : layer_type_(layer_type), input_(), weights_(), bias_(), output_() {}

        virtual void fwd() = 0;
        virtual void read_weights_bias(std::ifstream& is) = 0;

        void print() {
            std::cout << layer_type_ << std::endl;
            if (!input_.empty())   std::cout << "  input: "   << input_   << std::endl;
            if (!weights_.empty()) std::cout << "  weights: " << weights_ << std::endl;
            if (!bias_.empty())    std::cout << "  bias: "    << bias_    << std::endl;
            if (!output_.empty())  std::cout << "  output: "  << output_  << std::endl;
        }
        // TODO: additional required methods

    protected:
        const LayerType layer_type_;
        Tensor input_;
        Tensor weights_;
        Tensor bias_;
        Tensor output_;
};


class Conv2d : public Layer {
    public:
        Conv2d(size_t in_channels, size_t out_channels, size_t kernel_size, size_t stride=1, size_t pad=0) : Layer(LayerType::Conv2d) {}
    // TODO
};


class Linear : public Layer {
    public:
        Linear(size_t in_features, size_t out_features) : Layer(LayerType::Linear) {}
    // TODO
};


class MaxPool2d : public Layer {
    public:
        MaxPool2d(size_t kernel_size, size_t stride=1, size_t pad=0) 
		: Layer(LayerType::MaxPool2d) {}
		kernel_size_(kernel_size),
	        stride_(stride),
          	pad_(pad) {}

	Tensor fwd(const Tensor& x) override;
    private:
	size_t kernel_size_;
	size_t stride_;
	size_t pad_;
};


class ReLu : public Layer {
	public:
    		ReLu() : Layer(LayerType::ReLu) {}

    		Tensor fwd(const Tensor& x) override;
};


class SoftMax : public Layer {
    public:
        SoftMax() : Layer(LayerType::SoftMax) {}
    	Tensor fwd(const Tensor& x) override;
};


class Flatten : public Layer {
    public:
        Flatten() : Layer(LayerType::Flatten) {}
    	Tensor fwd(const tensor& x) override;
};


class NeuralNetwork {
    public:
        NeuralNetwork(bool debug=false) : debug_(debug) {}

        void add(Layer* layer) {
            // TODO
        }

        void load(std::string file) {
            // TODO
        }

        Tensor predict(Tensor input) {
            // TODO
        }

    private:
        bool debug_;
        // TODO: storage for layers
};

Tensor ReLu::fwd(const Tensor& x) {
    Tensor y(x.N(), x.C(), x.H(), x.W());
    for (size_t n = 0; n < x.N(); ++n)
        for (size_t c = 0; c < x.C(); ++c)
            for (size_t h = 0; h < x.H(); ++h)
                for (size_t w = 0; w < x.W(); ++w)
                    y(n, c, h, w) = std::max(0.0f, x(n,c,h,w));
    return y;
}


Tensor MaxPool2d::fwd(const Tensor& x) {
    size_t N = x.N();
    size_t C = x.C();
    size_t H = x.H();
    size_t W = x.W();

    size_t out_h = (H - kernel_size_) / stride_ + 1;
    size_t out_w = (W - kernel_size_) / stride_ + 1;

    Tensor y(N, C, out_h, out_w);

    for (size_t n = 0; n < N; ++n) {
        for (size_t c = 0; c < C; ++c) {
            for (size_t oh = 0; oh < out_h; ++oh) {
                for (size_t ow = 0; ow < out_w; ++ow) {

                    float max_val = -1e9;

                    for (size_t kh = 0; kh < kernel_size_; ++kh) {
                        for (size_t kw = 0; kw < kernel_size_; ++kw) {

                            size_t ih = oh * stride_ + kh;
                            size_t iw = ow * stride_ + kw;

                            float v = x(n, c, ih, iw);
                            if (v > max_val) max_val = v;
                        }
                    }

                    y(n, c, oh, ow) = max_val;
                }
            }
        }
    }

    return y;
}

Tensor Flatten::fwd(const Tensor& x) {

    size_t N = x.N();
    size_t C = x.C();
    size_t H = x.H();
    size_t W = x.W();


    size_t F = C * H * W;


    Tensor y(N, F, 1, 1);

    for (size_t n = 0; n < N; ++n) {
        size_t idx = 0;

        for (size_t c = 0; c < C; ++c) {
            for (size_t h = 0; h < H; ++h) {
                for (size_t w = 0; w < W; ++w) {
                    y(n, idx, 0, 0) = x(n, c, h, w);
                    ++idx;
                }
            }
        }
    }

    return y;
}

Tensor SoftMax::fwd(const Tensor& x) {
    size_t N = x.N();
    size_t C = x.C();
    size_t H = x.H();
    size_t W = x.W();

    Tensor y(N, C, H, W);

    // Softmax over the channel dimension C
    for (size_t n = 0; n < N; ++n) {
        for (size_t h = 0; h < H; ++h) {
            for (size_t w = 0; w < W; ++w) {
                // 1) find max for numerical stability
                float max_val = -1e9f;
                for (size_t c = 0; c < C; ++c) {
                    float v = x(n, c, h, w);
                    if (v > max_val) max_val = v;
                }

                // 2) compute exp(x - max) and sum
                float sum_exp = 0.0f;
                for (size_t c = 0; c < C; ++c) {
                    float v = std::exp(x(n, c, h, w) - max_val);
                    y(n, c, h, w) = v;
                    sum_exp += v;
                }

                // 3) normalize
                if (sum_exp > 0.0f) {
                    for (size_t c = 0; c < C; ++c) {
                        y(n, c, h, w) /= sum_exp;
                    }
                }
            }
        }
    }

    return y;
}


#endif // NETWORK_HPP
