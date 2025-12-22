CXX = g++
CXXFLAGS = -std=c++17 -O2 -Wall -DMNIST_PRE_PAD

TARGET = lenet
SRCS = lenet.cpp
HDRS = network.hpp tensor.hpp mnist.hpp minicnn_task.hpp student.hpp

all: $(TARGET)

$(TARGET): $(SRCS) $(HDRS)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(SRCS)

clean:
	rm -f $(TARGET)

.PHONY: all clean

