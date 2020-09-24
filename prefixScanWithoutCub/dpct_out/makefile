build:
	dpcpp -c prefixScan_t.dp.cpp -std=c++17
#	dpcpp -o prefixScan prefixScan_t.dp.o /opt/intel/oneapi/compiler/latest/linux/lib/libsycl-glibc.o -std=c++14
	dpcpp -o prefixScan prefixScan_t.dp.o -std=c++17

run_cpu:
	SYCL_DEVICE_TYPE=CPU ./prefixScan

run:
	./prefixScan

source:
	source /opt/intel/oneapi/setvars.sh
