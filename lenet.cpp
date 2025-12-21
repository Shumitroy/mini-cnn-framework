#include "tensor.hpp"
#include "network.hpp"
#include "mnist.hpp"
#include "student.hpp"
#include "minicnn_task.hpp"

#include <iostream>

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

    net.load("lenet.raw");

    MNIST mnist("t10k-images-idx3-ubyte");
    Tensor img = mnist.at(0);

    Tensor out = net.predict(img);

    std::cout << "Output probabilities:" << std::endl;
    std::cout << out << std::endl;


    return 0;
}
