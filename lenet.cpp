#include "tensor.hpp"
#include "network.hpp"
#include "mnist.hpp"
#include "student.hpp"
#include "minicnn_task.hpp"

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

static bool is_little_endian() {
    int n = 1;
    return *(char*)&n == 1;
}

static uint32_t bswap_u32(uint32_t v) {
    return ((v & 0x000000FFu) << 24) |
           ((v & 0x0000FF00u) << 8)  |
           ((v & 0x00FF0000u) >> 8)  |
           ((v & 0xFF000000u) >> 24);
}

static std::vector<uint8_t> load_mnist_labels(const std::string& path) {
    std::ifstream is(path.c_str(), std::ios::in | std::ios::binary);
    if (!is) throw std::runtime_error("Could not open labels file: " + path);

    uint32_t magic = 0, num = 0;
    is.read(reinterpret_cast<char*>(&magic), 4);
    is.read(reinterpret_cast<char*>(&num), 4);

    if (is_little_endian()) {
        magic = bswap_u32(magic);
        num = bswap_u32(num);
    }

    if (magic != 0x00000801u) throw std::runtime_error("Label file has wrong magic: " + path);

    std::vector<uint8_t> labels(num);
    is.read(reinterpret_cast<char*>(labels.data()), static_cast<std::streamsize>(num));
    if (!is) throw std::runtime_error("Failed reading labels: " + path);

    return labels;
}

static int argmax_10(const Tensor& out) {
    int best = 0;
    float bestv = out(0, 0, 0, 0);
    for (int c = 1; c < 10; ++c) {
        float v = out(0, c, 0, 0);
        if (v > bestv) {
            bestv = v;
            best = c;
        }
    }
    return best;
}

static const char* fashion_name(int c) {
    switch (c) {
        case 0: return "T-shirt/top";
        case 1: return "Trouser";
        case 2: return "Pullover";
        case 3: return "Dress";
        case 4: return "Coat";
        case 5: return "Sandal";
        case 6: return "Shirt";
        case 7: return "Sneaker";
        case 8: return "Bag";
        case 9: return "Ankle boot";
        default: return "?";
    }
}

static char pix(float v) {
    if (v > 0.85f) return '#';
    if (v > 0.60f) return 'O';
    if (v > 0.40f) return 'o';
    if (v > 0.20f) return '.';
    return ' ';
}

static void print_image_28x28(const Tensor& img) {
    size_t start = (img.H >= 32 && img.W >= 32) ? 2 : 0;
    for (size_t h = start; h < start + 28; ++h) {
        for (size_t w = start; w < start + 28; ++w) {
            std::cout << pix(img(0, 0, h, w));
        }
        std::cout << '\n';
    }
}


int main(int argc, char** argv) {
    std::cout << "MiniCNN LeNet inference test" << std::endl;
    std::cout << "Student: " << student_name << " (" << student_id << ")" << std::endl;

    bool use_fashion = true;
    bool show_some = false;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--mnist") use_fashion = false;
        if (a == "--fashion") use_fashion = true;
        if (a == "--show") show_some = true;
    }

    NeuralNetwork net(false);

	net.add(new Conv2d(1, 6, 5));
	net.add(new ReLu());
	net.add(new MaxPool2d(2, 2));

	net.add(new Conv2d(6, 16, 5));
	net.add(new ReLu());
	net.add(new MaxPool2d(2, 2));

	net.add(new Flatten());

	net.add(new Linear(400, 120));
	net.add(new ReLu());

	net.add(new Linear(120, 84));
	net.add(new ReLu());

	net.add(new Linear(84, 10));
	net.add(new SoftMax());

    std::string wfile, ifile, lfile;
    if (use_fashion) {
        wfile = "data-fashion-mnist-lenet.raw";
        ifile = "data-fashion-mnist-t10k-images-idx3-ubyte";
        lfile = "data-fashion-mnist-t10k-labels-idx1-ubyte";
    } else {
        wfile = "lenet.raw";
        ifile = "t10k-images-idx3-ubyte";
        lfile = "t10k-labels-idx1-ubyte";
    }

    net.load(wfile);

    MNIST images(ifile);
    std::vector<uint8_t> labels = load_mnist_labels(lfile);

    size_t total = std::min(static_cast<size_t>(labels.size()), images.slice(0, 1).N == 0 ? static_cast<size_t>(0) : labels.size());
    if (total == 0) total = labels.size();

    size_t correct = 0;
    size_t n_eval = std::min(static_cast<size_t>(10000), labels.size());

    for (size_t i = 0; i < n_eval; ++i) {
        Tensor img = images.at(i);
        Tensor out = net.predict(img);
        int pred = argmax_10(out);
        int gt = static_cast<int>(labels[i]);
        if (pred == gt) ++correct;

        if (show_some && i < 10) {
            if (use_fashion) {
                std::cout << "Image " << i << " pred=" << pred << " (" << fashion_name(pred) << ")"
                          << " gt=" << gt << " (" << fashion_name(gt) << ")" << std::endl;
            } else {
                std::cout << "Image " << i << " pred=" << pred << " gt=" << gt << std::endl;
            }
            std::cout << "Probabilities:" << std::endl;
            for (int c = 0; c < 10; ++c) {
                std::cout << " " << c << ": " << std::fixed << std::setprecision(6) << out(0, c, 0, 0);
                if (c == pred) std::cout << " <-- max";
                std::cout << std::endl;
            }

		std::cout << "Image preview:\n";
        	print_image_28x28(img);
            std::cout << std::endl;
        }
    }

    float acc = (n_eval > 0) ? (static_cast<float>(correct) / static_cast<float>(n_eval)) : 0.0f;
    std::cout << "Accuracy: " << std::fixed << std::setprecision(4) << acc
              << " (" << correct << "/" << n_eval << ")" << std::endl;

    return 0;
}

