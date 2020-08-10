build:
	dpcpp -o prefixScan prefixScan_t.dp.o /opt/intel/oneapi/compiler/latest/linux/lib/libsycl-glibc.o -std=c++11
	dpcpp -c prefixScan_t.dp.cpp -std=c++11

run:
	./prefixScan

source:
	source /opt/intel/oneapi/setvars.sh
