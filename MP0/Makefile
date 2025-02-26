WB = $(WB_DIR)
NVCC = nvcc
CXX = g++
CC = gcc

template.o: template.cu
	$(NVCC) --std=c++11 -ccbin=$(CXX) -rdc=true -I $(WB) -c template.cu -o template.o

template: template.o
	$(NVCC) --std=c++11 -ccbin=$(CXX) -o template template.o $(WB)/lib/libwb.so
