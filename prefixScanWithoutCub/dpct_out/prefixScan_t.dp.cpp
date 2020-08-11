#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include <iostream>
#include <stdlib.h>

//#include "cudaCheck.h"
#include "prefixScan.h"
#include "requireDevices.h"

using namespace cms::cuda;

template <typename T> struct format_traits {
public:
  static const constexpr char *failed_msg = "failed %d %d %d: %d %d\n";
};

template <> struct format_traits<float> {
public:
  static const constexpr char *failed_msg = "failed %d %d %d: %f %f\n";
};

template <typename T>
int SYCL_EXTERNAL testPrefixScan(uint32_t size, sycl::nd_item<3> item_ct1,
                                  sycl::stream stream_ct1, T *ws, T *c, T *co) {

  auto first = item_ct1.get_local_id(2);
  for (auto i = first; i < size; i += item_ct1.get_local_range().get(2))
    c[i] = 1;
  item_ct1.barrier();

  blockPrefixScan(c, co, size, ws, item_ct1);
  blockPrefixScan(c, size, ws, item_ct1);

  if (!(1 == c[0])) {
    stream_ct1 << "Assertion failed during testPrefixScan (file "
                  "'prefixScan_t.dp.cpp)\nAborting...\n";
    return -1;
  }
  if (!(1 == co[0])) {
    stream_ct1 << "Assertion failed during testPrefixScan (file "
                  "'prefixScan_t.dp.cpp)\nAborting...\n";
    return -1;
  }
  for (auto i = first + 1; i < size; i += item_ct1.get_local_range().get(2)) {
    if (c[i] != c[i - 1] + 1) {
      stream_ct1 << format_traits<unsigned short>::failed_msg;
      stream_ct1 << format_traits<float>::failed_msg;
    }
    if (!(c[i] == c[i - 1] + 1)) {
      stream_ct1 << "Assertion failed during testPrefixScan (file "
                   "'prefixScan_t.dp.cpp)\nAborting...\n";
      return -1;
    }
    if (!(c[i] == i + 1)) {
      stream_ct1 << "Assertion failed during testPrefixScan (file "
                   "'prefixScan_t.dp.cpp)\nAborting...\n";
      return -1;
    }
    if (!(c[i] = co[i])) {
      stream_ct1 << "Assertion failed during testPrefixScan (file "
                   "'prefixScan_t.dp.cpp)\nAborting...\n";
      return -1;
    }
  }
  return 0;
}

template <typename T>
int SYCL_EXTERNAL testWarpPrefixScan(uint32_t size, sycl::nd_item<3> item_ct1,
                                      sycl::stream stream_ct1, T *c, T *co) {
  if (!(size <= 32)) {
    stream_ct1 << "Assertion failed during testWarpPrefixScan (file "
                 "'prefixScan_t.dp.cpp)\nAborting...\n";
    return -1;
  }

  auto i = item_ct1.get_local_id(2);
  c[i] = 1;
  item_ct1.barrier();

  warpPrefixScan(c, co, i, item_ct1);
  warpPrefixScan(c, i, item_ct1);
  item_ct1.barrier();

  if (!(1 == c[0])) {
    stream_ct1 << "Assertion failed during testWarpPrefixScan (file "
                 "'prefixScan_t.dp.cpp)\nAborting...\n";
    return -1;
  }
  if (!(1 == co[0])) {
    stream_ct1 << "Assertion failed during testWarpPrefixScan (file "
                 "'prefixScan_t.dp.cpp)\nAborting...\n";
    return -1;
  }
  if (i != 0) {
    if (c[i] != c[i - 1] + 1)
      stream_ct1 << format_traits<int>::failed_msg;
    if (!(c[i] == c[i - 1] + 1)) {
      stream_ct1 << "Assertion failed during testWarpPrefixScan (file "
                   "'prefixScan_t.dp.cpp)\nAborting...\n";
      return -1;
    }
    if (!(c[i] == i + 1)) {
      stream_ct1 << "Assertion failed during testWarpPrefixScan (file "
                   "'prefixScan_t.dp.cpp)\nAborting...\n";
      return -1;
    }
    if (!(c[i] = co[i])) {
      stream_ct1 << "Assertion failed during testWarpPrefixScan (file "
                   "'prefixScan_t.dp.cpp)\nAborting...\n";
      return -1;
    }
  }
  return 0;
}

void init(uint32_t *v, uint32_t val, uint32_t n, sycl::nd_item<3> item_ct1,
          sycl::stream stream_ct1) {
  auto i = item_ct1.get_group(2) * item_ct1.get_local_range().get(2) +
           item_ct1.get_local_id(2);
  if (i < n)
    v[i] = val;
  if (i == 0)
    stream_ct1 << "init\n";
}

int verify(uint32_t const *v, uint32_t n, sycl::nd_item<3> item_ct1,
            sycl::stream stream_ct1) {
  auto i = item_ct1.get_group(2) * item_ct1.get_local_range().get(2) +
           item_ct1.get_local_id(2);
  if (i < n)
    if (!(v[i] == i + 1)) {
      stream_ct1 << "Assertion failed during 'verify' (file "
                   "'prefixScan_t.dp.cpp)\nAborting...\n";
      return -1;
    }
  if (i == 0)
    stream_ct1 << "verify\n";
  return 0;
}

int main() {
  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  sycl::queue &q_ct1 = dev_ct1.default_queue();
  // cms::oneapitest::requireDevices();

  std::cout << "\nmax item sizes: ";
  auto info = dev_ct1.get_info<sycl::info::device::max_work_item_sizes>();
  std::cout << info[0] << ' ' << info[1] << ' ' << info[2];
  auto N = info[2]; // numero max di thread per fila
  std::cout << "\nmax work item dimentions: ";
  std::cout << dev_ct1.get_info<sycl::info::device::max_work_item_dimensions>();
  std::cout << "\nwarp level" << std::endl;
  std::cout << "warp 32" << std::endl;
  q_ct1.submit([&](sycl::handler &cgh) {
    sycl::stream stream_ct1(64 * 1024, 80, cgh);

    // accessors to device memory
    sycl::accessor<int, 1, sycl::access::mode::read_write,
                   sycl::access::target::local>
        c_acc_ct1(sycl::range<1>(1024), cgh);
    sycl::accessor<int, 1, sycl::access::mode::read_write,
                   sycl::access::target::local>
        co_acc_ct1(sycl::range<1>(1024), cgh);

    cgh.parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, 32), sycl::range<3>(1, 1, 32)),
        [=](sycl::nd_item<3> item_ct1) {
          testWarpPrefixScan<int>(32, item_ct1, stream_ct1,
                                  c_acc_ct1.get_pointer(),
                                  co_acc_ct1.get_pointer());
        });
  });
  dev_ct1.queues_wait_and_throw();
  std::cout << "warp 16" << std::endl;
  q_ct1.submit([&](sycl::handler &cgh) {
    sycl::stream stream_ct1(64 * 1024, 80, cgh);

    // accessors to device memory
    sycl::accessor<int, 1, sycl::access::mode::read_write,
                   sycl::access::target::local>
        c_acc_ct1(sycl::range<1>(1024), cgh);
    sycl::accessor<int, 1, sycl::access::mode::read_write,
                   sycl::access::target::local>
        co_acc_ct1(sycl::range<1>(1024), cgh);

    cgh.parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, 32), sycl::range<3>(1, 1, 32)),
        [=](sycl::nd_item<3> item_ct1) {
          testWarpPrefixScan<int>(16, item_ct1, stream_ct1,
                                  c_acc_ct1.get_pointer(),
                                  co_acc_ct1.get_pointer());
        });
  });
  dev_ct1.queues_wait_and_throw();
  std::cout << "warp 5" << std::endl;
  q_ct1.submit([&](sycl::handler &cgh) {
    sycl::stream stream_ct1(64 * 1024, 80, cgh);

    // accessors to device memory
    sycl::accessor<int, 1, sycl::access::mode::read_write,
                   sycl::access::target::local>
        c_acc_ct1(sycl::range<1>(1024), cgh);
    sycl::accessor<int, 1, sycl::access::mode::read_write,
                   sycl::access::target::local>
        co_acc_ct1(sycl::range<1>(1024), cgh);

    cgh.parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, 32), sycl::range<3>(1, 1, 32)),
        [=](sycl::nd_item<3> item_ct1) {
          testWarpPrefixScan<int>(5, item_ct1, stream_ct1,
                                  c_acc_ct1.get_pointer(),
                                  co_acc_ct1.get_pointer());
        });
  });
  dev_ct1.queues_wait_and_throw();

  std::cout << "block level" << std::endl;
  for (int bs = 32; bs <= min(1024, N); bs += 32) {
    // std::cout << "bs " << bs << std::endl;
    for (int j = 1; j <= 1024; ++j) {
      // std::cout << j << std::endl;
      q_ct1.submit([&](sycl::handler &cgh) {
        sycl::stream stream_ct1(64 * 1024, 80, cgh);

        // accessors to device memory
        sycl::accessor<uint16_t, 1, sycl::access::mode::read_write,
                       sycl::access::target::local>
            ws_acc_ct1(sycl::range<1>(32), cgh);
        sycl::accessor<uint16_t, 1, sycl::access::mode::read_write,
                       sycl::access::target::local>
            c_acc_ct1(sycl::range<1>(1024), cgh);
        sycl::accessor<uint16_t, 1, sycl::access::mode::read_write,
                       sycl::access::target::local>
            co_acc_ct1(sycl::range<1>(1024), cgh);

        cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, bs),
                                           sycl::range<3>(1, 1, bs)),
                         [=](sycl::nd_item<3> item_ct1) {
                           testPrefixScan<uint16_t>(j, item_ct1, stream_ct1,
                                                    ws_acc_ct1.get_pointer(),
                                                    c_acc_ct1.get_pointer(),
                                                    co_acc_ct1.get_pointer());
                         });
      });
      dev_ct1.queues_wait_and_throw();
      q_ct1.submit([&](sycl::handler &cgh) {
        sycl::stream stream_ct1(64 * 1024, 80, cgh);

        // accessors to device memory
        sycl::accessor<float, 1, sycl::access::mode::read_write,
                       sycl::access::target::local>
            ws_acc_ct1(sycl::range<1>(32), cgh);
        sycl::accessor<float, 1, sycl::access::mode::read_write,
                       sycl::access::target::local>
            c_acc_ct1(sycl::range<1>(1024), cgh);
        sycl::accessor<float, 1, sycl::access::mode::read_write,
                       sycl::access::target::local>
            co_acc_ct1(sycl::range<1>(1024), cgh);

        cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, bs),
                                           sycl::range<3>(1, 1, bs)),
                         [=](sycl::nd_item<3> item_ct1) {
                           testPrefixScan<float>(j, item_ct1, stream_ct1,
                                                 ws_acc_ct1.get_pointer(),
                                                 c_acc_ct1.get_pointer(),
                                                 co_acc_ct1.get_pointer());
                         });
      });
      dev_ct1.queues_wait_and_throw();
    }
  }
  dev_ct1.queues_wait_and_throw();

  int num_items = 200;
  for (int ksize = 1; ksize < 4; ++ksize) {
    // test multiblock
    std::cout << "multiblok" << std::endl;
    // Declare, allocate, and initialize device-accessible pointers for input
    // and output
    num_items *= 10;
    uint32_t *d_in;
    uint32_t *d_out1;
    uint32_t *d_out2;

    d_in = (uint32_t *)sycl::malloc_device(num_items * sizeof(uint32_t),
                                           dev_ct1, q_ct1.get_context());
    d_out1 = (uint32_t *)sycl::malloc_device(num_items * sizeof(uint32_t),
                                             dev_ct1, q_ct1.get_context());
    d_out2 = (uint32_t *)sycl::malloc_device(num_items * sizeof(uint32_t),
                                             dev_ct1, q_ct1.get_context());

    auto nthreads = 256;
    auto nblocks = (num_items + nthreads - 1) / nthreads;

    q_ct1.submit([&](sycl::handler &cgh) {
      sycl::stream stream_ct1(64 * 1024, 80, cgh);

      cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, nblocks) *
                                             sycl::range<3>(1, 1, nthreads),
                                         sycl::range<3>(1, 1, nthreads)),
                       [=](sycl::nd_item<3> item_ct1) {
                         init(d_in, 1, num_items, item_ct1, stream_ct1);
                       });
    });

    // the block counter
    int32_t *d_pc;

    d_pc = (int32_t *)sycl::malloc_device(1, dev_ct1, q_ct1.get_context());

    q_ct1.memset(d_pc, 0, sizeof(int32_t)).wait();

    nthreads = min(1024, N);
    nblocks = (num_items + nthreads - 1) / nthreads;
    std::cout << "launch multiBlockPrefixScan " << num_items << ' ' << nblocks
              << std::endl;

    try {
      q_ct1.submit([&](sycl::handler &cgh) {
        // accessors to device memory
        sycl::accessor<uint8_t, 1, sycl::access::mode::read_write,
                       sycl::access::target::local>
            dpct_local_acc_ct1(sycl::range<1>(4 * nblocks), cgh);
        sycl::accessor<uint32_t, 1, sycl::access::mode::read_write,
                       sycl::access::target::local>
            ws_acc_ct1(sycl::range<1>(32), cgh);
        sycl::accessor<bool, 0, sycl::access::mode::read_write,
                       sycl::access::target::local>
            isLastBlockDone_acc_ct1(cgh);

        cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, nblocks) *
                                               sycl::range<3>(1, 1, nthreads),
                                           sycl::range<3>(1, 1, nthreads)),
                         [=](sycl::nd_item<3> item_ct1) {
                           multiBlockPrefixScan<uint32_t>(
                               d_in, d_out1, num_items, d_pc, item_ct1,
                               dpct_local_acc_ct1.get_pointer(),
                               ws_acc_ct1.get_pointer(),
                               isLastBlockDone_acc_ct1.get_pointer());
			   });
      });
    } catch (std::exception &e) {
      std::cerr << e.what();
    }

    try {
      q_ct1.submit([&](sycl::handler &cgh) {
        sycl::stream stream_ct1(64 * 1024, 80, cgh);

        cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, nblocks) *
                                               sycl::range<3>(1, 1, nthreads),
                                           sycl::range<3>(1, 1, nthreads)),
                         [=](sycl::nd_item<3> item_ct1) {
                           verify(d_out1, num_items, item_ct1, stream_ct1);
                         });
      });
    } catch (std::exception &e) {
      std::cerr << e.what();
    }

    dev_ct1.queues_wait_and_throw();

  } // ksize
  return 0;
}
