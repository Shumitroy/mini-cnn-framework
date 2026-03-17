# Mini CNN Framework (C++)

Minimal Convolutional Neural Network (CNN) inference framework implemented in C++ from scratch.
Supports Conv2D, ReLU, MaxPool, Linear, Flatten, and Softmax layers, with LeNet inference on MNIST and Fashion-MNIST.

---

## Highlights

* C++17 implementation
* Custom Tensor data structure
* Modular neural network layers
* Baseline direct convolution
* Optimized im2col convolution (performance branch)
* MNIST / LeNet inference
* Clean Git history showing development and optimization steps

---

## Branches

* `master` – baseline implementation
* `performance` – optimized implementation (im2col)

---

## Build

```bash
make
```

---

## Run

```bash
./lenet --mnist
./lenet --fashion
```

Optional:

```bash
./lenet --mnist --show
./lenet --fashion --show
```

---

## Results

| Dataset       | Accuracy |
| ------------- | -------- |
| MNIST         | 0.9839   |
| Fashion-MNIST | 0.8450   |

---

## Example Predictions

### MNIST

![MNIST Example 1](assets/mnist_example1.png)
![MNIST Example 2](assets/mnist_example2.png)
![MNIST Example 3](assets/mnist_example3.png)

---

### Fashion-MNIST

![Fashion Example 1](assets/fashion_mnist_example1.png)
![Fashion Example 2](assets/fashion_mnist_example2.png)
![Fashion Example 3](assets/fashion_mnist_example3.png)

---

## Observations

* High accuracy on MNIST (~98%) confirms correct implementation of LeNet
* Lower accuracy on Fashion-MNIST (~84%) due to higher dataset complexity
* Model shows strong confidence for correct predictions
* Performance differences highlight generalization challenges

---

## Acknowledgement

This project was developed as part of the course
**"Systems Engineering and Architecting for Edge Computing"**
at Technische Hochschule Ingolstadt.

The initial project skeleton was provided by the course instructor.

---

