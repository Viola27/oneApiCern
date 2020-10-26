#include <ios>
#include <iostream>
#include <CL/sycl.hpp>

//da aggiustare questo
//#include "SYCLCore/prefixScan.h"
#include "prefixScan.h"

template <typename T>
void testPrefixScan(uint32_t size, sycl::nd_item<3> item, sycl::stream sycl_stream, T *ws, T *c, T *co, int subgroupSize) {
  auto first = item.get_local_id(2);
  for (auto i = first; i < size; i += item.get_local_range().get(2))
    c[i] = 1;
  item.barrier();

  blockPrefixScan(c, co, size, ws, item, sycl_stream, subgroupSize);
  blockPrefixScan(c, size, ws, item, sycl_stream, subgroupSize);

  //assert(1 == c[0]);
  if(1 != c[0]){
    sycl_stream << "failed (testPrefixScan): 1 != c[0] " << sycl::endl;
    return;
  }
  //assert(1 == co[0]);
  if(1 != co[0]){
    sycl_stream << "failed (testPrefixScan): 1 != co[0] " << sycl::endl;
    return;
  }

  for (auto i = first + 1; i < size; i += item.get_local_range().get(2)) {
    if (c[i] != c[i - 1] + 1)
      sycl_stream << "failed " << size << " " << i << " " << item.get_local_range(2) << " " << c[i] << " " << c[i - 1] << sycl::endl;

    //assert(c[i] == c[i - 1] + 1);
    if(c[i] != c[i - 1] + 1){
      sycl_stream << "failed (testPrefixScan): c[i] != c[i - 1] + 1 " << sycl::endl;
      return;
    }
    //assert(c[i] == i + 1);
    if(c[i] != i + 1){
      sycl_stream << "failed (testPrefixScan): c[i] != i + 1 " << sycl::endl;
      return;
    }
    //assert(c[i] == co[i]);
    if(c[i] != co[i]){
      sycl_stream << "failed (testPrefixScan): c[i] != co[i] " << sycl::endl;
      return;
    }
  }
}

template <typename T>
void testWarpPrefixScan(uint32_t size, sycl::nd_item<3> item, sycl::stream sycl_stream, T *c, T *co, int subgroupSize) {
  //assert(size <= 32);
  if(size > 32){
    sycl_stream << "failed (testWarpPrefixScan): size > 32 " << sycl::endl;
    return;
  }

  auto i = item.get_local_id(2);
  c[i] = 1;
  item.barrier();

  warpPrefixScan(c, co, i, item, subgroupSize);
  warpPrefixScan(c, i, item, subgroupSize);
  item.barrier();

  //assert(1 == c[0]);
  if(1 != c[0]){
    sycl_stream << "failed (testWarpPrefixScan): 1 != c[0] " << sycl::endl;
    return;
  }
  //assert(1 == co[0]);
  if(1 != co[0]){
    sycl_stream << "failed (testWarpPrefixScan): 1 != co[0] " << sycl::endl;
    return;
  }

  if (i != 0) {
    if (c[i] != c[i - 1] + 1)
      sycl_stream << "failed " << size << " " << i << " " << item.get_local_range(2) << " " << c[i] << " " << c[i - 1] << sycl::endl;
    
    //assert(c[i] == c[i - 1] + 1);
    if(c[i] != c[i - 1] + 1){
      sycl_stream << "failed (testWarpPrefixScan): c[i] != c[i - 1] + 1 " << sycl::endl;
      return;
    }
    //assert(c[i] == i + 1);
    if(c[i] != i + 1){
      sycl_stream << "failed (testWarpPrefixScan): c[i] != i + 1 " << sycl::endl;
      return;
    }
    //assert(c[i] == co[i]);
    if(c[i] != co[i]){
      sycl_stream << "failed (testWarpPrefixScan): c[i] != co[i] " << sycl::endl;
      return;
    }
  }
}

void init(uint32_t *v, uint32_t val, uint32_t n, sycl::nd_item<3> item, sycl::stream sycl_stream) {
  auto i = item.get_global_id(2);
  if (i < n)
    v[i] = val;
  if (i == 0)
    sycl_stream << "init\n" << sycl::endl;
}

void verify(uint32_t const *v, uint32_t n, sycl::nd_item<3> item, sycl::stream sycl_stream) {
  auto i = item.get_global_id(2);
  if (i < n){
    //assert(v[i] == i + 1);
    if (v[i] != i + 1) {
      sycl_stream << "failed (verify): v[i] != i + 1 " << sycl::endl;
      return;
    }
  }    
  if (i == 0)
    sycl_stream << "verify\n" << sycl::endl;
}

int main() try {

  sycl::default_selector device_selector;
  sycl::queue stream(device_selector); 

  // FIXME the OpenCL CPU device reports a maximum workgroup size of 8192,
  // but workgroups bigger than 4096 result in a CL_OUT_OF_RESOURCES error
  const unsigned int workgroupSizeLimit = 4096;
  auto maxWorkItemSizes = stream.get_device().get_info<sycl::info::device::max_work_item_sizes>();
  auto maxWorkItemSize = std::min((int) maxWorkItemSizes[2], 1024);

  auto subgroupSizes = stream.get_device().get_info<sycl::info::device::sub_group_sizes>();
  auto subgroupSize = std::min(16, (int) *std::end(subgroupSizes));

  std::cout << "warp level" << std::endl;
  // std::cout << "warp 32" << std::endl;
  stream.submit([&](sycl::handler &cgh) {
    sycl::stream sycl_stream(64 * 1024, 80, cgh);
    sycl::accessor<int, 1, sycl::access::mode::read_write, sycl::access::target::local> c_acc(sycl::range(1024),cgh);
    sycl::accessor<int, 1, sycl::access::mode::read_write, sycl::access::target::local> co_acc(sycl::range(1024),cgh);

    cgh.parallel_for(sycl::nd_range(sycl::range(1, 1, subgroupSize), sycl::range(1, 1, subgroupSize)), [=](sycl::nd_item<3> item) [[cl::intel_reqd_sub_group_size(16)]] {
      testWarpPrefixScan<int>(32, item, sycl_stream, c_acc.get_pointer(), co_acc.get_pointer(), subgroupSize);
    });
  });
  stream.wait();

  // std::cout << "warp 16" << std::endl;
  stream.submit([&](sycl::handler &cgh) {
    sycl::stream sycl_stream(64 * 1024, 80, cgh);
    sycl::accessor<int, 1, sycl::access::mode::read_write, sycl::access::target::local> c_acc(sycl::range(1024),cgh);
    sycl::accessor<int, 1, sycl::access::mode::read_write, sycl::access::target::local> co_acc(sycl::range(1024),cgh);

    cgh.parallel_for(sycl::nd_range(sycl::range(1, 1, subgroupSize), sycl::range(1, 1, subgroupSize)), [=](sycl::nd_item<3> item) [[cl::intel_reqd_sub_group_size(16)]] {
      testWarpPrefixScan<int>(16, item, sycl_stream, c_acc.get_pointer(), co_acc.get_pointer(), subgroupSize);
    });
  });
  stream.wait();

  // std::cout << "warp 5" << std::endl;
  stream.submit([&](sycl::handler &cgh) {
    sycl::stream sycl_stream(64 * 1024, 80, cgh);
    sycl::accessor<int, 1, sycl::access::mode::read_write, sycl::access::target::local> c_acc(sycl::range(1024),cgh);
    sycl::accessor<int, 1, sycl::access::mode::read_write, sycl::access::target::local> co_acc(sycl::range(1024),cgh);

    cgh.parallel_for(sycl::nd_range(sycl::range(1, 1, subgroupSize), sycl::range(1, 1, subgroupSize)), [=](sycl::nd_item<3> item) [[cl::intel_reqd_sub_group_size(16)]] {
      testWarpPrefixScan<int>(5, item, sycl_stream, c_acc.get_pointer(), co_acc.get_pointer(), subgroupSize);
    });
  });
  stream.wait();

  std::cout << "block level" << std::endl;
  for (int bs = 32; bs <= maxWorkItemSize; bs += 32) {
    for (int j = 1; j <= maxWorkItemSize; ++j) {
      stream.submit([&](sycl::handler &cgh) {
        sycl::stream sycl_stream(64 * 1024, 80, cgh);

        sycl::accessor<uint16_t, 1, sycl::access::mode::read_write, sycl::access::target::local> ws_acc(sycl::range(32), cgh);
        sycl::accessor<uint16_t, 1, sycl::access::mode::read_write, sycl::access::target::local> c_acc(sycl::range(1024), cgh);
        sycl::accessor<uint16_t, 1, sycl::access::mode::read_write, sycl::access::target::local> co_acc(sycl::range(1024), cgh);

        cgh.parallel_for(sycl::nd_range(sycl::range(1, 1, bs), sycl::range(1, 1, bs)), [=](sycl::nd_item<3> item) [[cl::intel_reqd_sub_group_size(16)]] {
          testPrefixScan<uint16_t>(
              j, item, sycl_stream, ws_acc.get_pointer(), c_acc.get_pointer(), co_acc.get_pointer(), subgroupSize);
        });
      });
      stream.wait();

      stream.submit([&](sycl::handler &cgh) {
        sycl::stream sycl_stream(64 * 1024, 80, cgh);

        sycl::accessor<float, 1, sycl::access::mode::read_write, sycl::access::target::local> ws_acc(sycl::range(32), cgh);
        sycl::accessor<float, 1, sycl::access::mode::read_write, sycl::access::target::local> c_acc(sycl::range(1024), cgh);
        sycl::accessor<float, 1, sycl::access::mode::read_write, sycl::access::target::local> co_acc(sycl::range(1024), cgh);

        cgh.parallel_for(sycl::nd_range(sycl::range(1, 1, bs), sycl::range(1, 1, bs)), [=](sycl::nd_item<3> item) [[cl::intel_reqd_sub_group_size(16)]] {
          testPrefixScan<float>(
              j, item, sycl_stream, ws_acc.get_pointer(), c_acc.get_pointer(), co_acc.get_pointer(), subgroupSize);
        });
      });
      stream.wait();
    }
  }
  stream.wait();

  int num_items = 200;
  for (int ksize = 1; ksize < 4; ++ksize) {
    // test multiblock
    std::cout << "multiblok" << std::endl;
    // Declare, allocate, and initialize device-accessible pointers for input and output
    num_items *= 10;
    uint32_t *d_in;
    uint32_t *d_out1;
    uint32_t *d_out2;

    d_in = (uint32_t *) sycl::malloc_device(num_items * sizeof(uint32_t), stream.get_device(), stream.get_context());
    d_out1 = (uint32_t *) sycl::malloc_device(num_items * sizeof(uint32_t), stream.get_device(), stream.get_context());
    d_out2 = (uint32_t *) sycl::malloc_device(num_items * sizeof(uint32_t), stream.get_device(), stream.get_context());

    auto nthreads = std::min((int) maxWorkItemSizes[2], 256);
    auto nblocks = (num_items + nthreads - 1) / nthreads;
    
    stream.submit([&](sycl::handler &cgh) {
      sycl::stream sycl_stream(64 * 1024, 80, cgh);

      cgh.parallel_for(
          sycl::nd_range(sycl::range(1, 1, nblocks) * sycl::range(1, 1, nthreads), sycl::range(1, 1, nthreads)), [=](sycl::nd_item<3> item) [[cl::intel_reqd_sub_group_size(16)]] {
            init(d_in, 1, num_items, item, sycl_stream);
          });
    });

    // the block counter
    int32_t *d_pc;
    d_pc = sycl::malloc_device<int32_t>(1, stream);

    nthreads = maxWorkItemSize;
    nblocks = (num_items + nthreads - 1) / nthreads;

    stream.submit([&](sycl::handler &cgh) {
      sycl::stream sycl_stream(64 * 1024, 80, cgh);

      sycl::accessor<unsigned int, 1, sycl::access::mode::read_write, sycl::access::target::local> ws_acc(sycl::range(32), cgh);
      sycl::accessor<bool, 0, sycl::access::mode::read_write, sycl::access::target::local> isLastBlockDone_acc(cgh);
      sycl::accessor<unsigned int, 1, sycl::access::mode::read_write, sycl::access::target::local> psum_acc(sycl::range(1024), cgh);

      cgh.parallel_for(
          sycl::nd_range(sycl::range(1, 1, nblocks) * sycl::range(1, 1, nthreads), sycl::range(1, 1, nthreads)), [=](sycl::nd_item<3> item) [[cl::intel_reqd_sub_group_size(16)]] {
            multiBlockPrefixScan<uint32_t>(d_in, d_out1, num_items, d_pc, item, (unsigned int *)ws_acc.get_pointer(), isLastBlockDone_acc.get_pointer(), (unsigned int *)psum_acc.get_pointer(), sycl_stream, subgroupSize);
          });
    });

    stream.submit([&](sycl::handler &cgh) {
      sycl::stream sycl_stream(64 * 1024, 80, cgh);

      cgh.parallel_for(
          sycl::nd_range(sycl::range(1, 1, nblocks) * sycl::range(1, 1, nthreads), sycl::range(1, 1, nthreads)), [=](sycl::nd_item<3> item) [[cl::intel_reqd_sub_group_size(16)]] {
            verify(d_out1, num_items, item, sycl_stream);
          });
    });
    stream.wait();

    stream.submit([&](sycl::handler &cgh) {
      sycl::stream sycl_stream(64 * 1024, 80, cgh);

      cgh.parallel_for(
          sycl::nd_range(sycl::range(1, 1, nblocks) * sycl::range(1, 1, nthreads), sycl::range(1, 1, nthreads)), [=](sycl::nd_item<3> item) [[cl::intel_reqd_sub_group_size(16)]] {
            verify(d_out2, num_items, item, sycl_stream);
          });
    });
    stream.wait();
  }  // ksize
  
  return 0;
} catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__ << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}
