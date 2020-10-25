CXX = dpcpp
CXXFLAGS = -O2 -g -std=c++17 -Wno-ignored-attributes

CUB_EXE_NAME = prefixScan_t_cub
CUB_SOURCES = prefixScanCub/prefixScan_t.cpp

EXE_NAME = prefixScan_t
SOURCES = prefixScan_t.cpp
LINK_FLAG = /opt/intel/oneapi/compiler/latest/linux/lib/libsycl-glibc.o

build:
	$(CXX) $(CXXFLAGS) -o $(EXE_NAME) $(SOURCES)

run:
	./$(EXE_NAME)

run_cpu:
	SYCL_DEVICE_TYPE=CPU ./$(EXE_NAME)

run_ocl:
	SYCL_BE=PI_OPENCL ./$(EXE_NAME)

build_cub:
	$(CXX) $(CXXFLAG) -o $(CUB_EXE_NAME) $(CUB_SOURCES)

run_cub:
	./$(CUB_EXE_NAME)

build_l:
	$(CXX) $(CXXFLAGS) $(SOURCES) $(LINK_FLAG) -o $(EXE_NAME)

clean:
	rm -rf $(CUB_EXE_NAME)
	rm -rf $(EXE_NAME)

prefixScan: prefixScan_t.o
	nvcc -o prefixScan prefixScan_t.o -std=c++11

prefixScan_t.o: prefixScan_t.cu prefixScan.h cudaCheck.h requireDevices.h
	nvcc -c prefixScan_t.cu --expt-relaxed-constexpr -std=c++11

source:
	source /opt/intel/oneapi/setvars.sh