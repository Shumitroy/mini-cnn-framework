#include "tensor.hpp"
#include "network.hpp"
#include "mnist.hpp"
#include "student.hpp"
#include "minicnn_task.hpp"

#include <iostream>

int argmax_10(const Tensor& out) {
    int best_idx = 0;
    float best_val = out(0, 0, 0, 0);
    for (int c = 1; c < 10; ++c) {
        float v = out(0, c, 0, 0);
        if (v > best_val) {
            best_val = v;
            best_idx = c;
        }
    }
    return best_idx;
}


int main() {
        std::cout << "MiniCNN LeNet inference test" << std::endl;
    std::cout << "Student: " << student_name << " (" << student_id << ")" << std::endl;

    NeuralNetwork net(false);

    net.add(new Conv2d(1, 6, 5));
    net.add(new ReLu());
    net.add(new MaxPool2d(2, 2));

    net.add(new Conv2d(6, 16, 5));
    net.add(new ReLu());
    net.add(new MaxPool2d(2, 2));

    net.add(new Flatten());
    net.add(new Linear(256, 120));
    net.add(new ReLu());
    net.add(new Linear(120, 10));
    net.add(new SoftMax());

    bool use_fashion = true;   // change to false for digit MNIST

MNIST* mnist = nullptr;

if (use_fashion) {
    net.load("data-fashion-mnist-lenet.raw");
    mnist = new MNIST("data-fashion-mnist-t10k-images-idx3-ubyte");
	} else {
    	net.load("lenet.raw");
    	mnist = new MNIST("t10k-images-idx3-ubyte");
	}

	Tensor img = mnist->at(0);
	Tensor out = net.predict(img);


    int pred = argmax_10(out);

    std::cout << "Image " << i << " -> predicted digit: " << pred << std::endl;
    std::cout << "Probabilities:" << std::endl;

    for (int d = 0; d < 10; ++d) {
        std::cout << "  " << d << ": " << out(0, d, 0, 0);
        if (d == pred) std::cout << "  <-- max";
        std::cout << std::endl;
    }

    std::cout << std::endl;
}





    return 0;
}
