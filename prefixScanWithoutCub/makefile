prefixScan: prefixScan_t.o
	nvcc -o prefixScan prefixScan_t.o -std=c++11

prefixScan_t.o: prefixScan_t.cu prefixScan.h cudaCheck.h requireDevices.h
	nvcc -c prefixScan_t.cu --expt-relaxed-constexpr -std=c++11
