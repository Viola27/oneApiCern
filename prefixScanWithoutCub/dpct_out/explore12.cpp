/*
Copyright (C) 2019 James Reinders

Permission is hereby granted, free of charge, to any person obtaining
a copy of this software and associated documentation files (the
"Software"), to deal in the Software without restriction, including
without limitation the rights to use, copy, modify, merge, publish,
distribute, sublicense, and/or sell copies of the Software, and to
permit persons to whom the Software is furnished to do so, subject to
the following conditions:

The above copyright notice and this permission notice shall be
included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

SPDX-License-Identifier: MIT
*/

#include <CL/sycl.hpp>
#include <cstdio>
#include <cstdlib>

using namespace cl::sycl;

#define KernelToPrintsNdItem6_1_0                                              \
  [=](nd_item<3> item) [[cl::intel_reqd_sub_group_size(32)]] {                 \
    if ((item.get_global_id()[0] == 6) && (item.get_global_id()[1] == 1) &&    \
        (item.get_global_id()[2] == 0)) {                                      \
      intel::sub_group sg = item.get_sub_group();                              \
      kernelout << "global range       " << item.get_global_range()            \
                << cl::sycl::endl;                                             \
      kernelout << "global id          " << item.get_global_id()               \
                << cl::sycl::endl;                                             \
      kernelout << "global linear id   " << item.get_global_linear_id()        \
                << cl::sycl::endl;                                             \
      kernelout << "group range        " << item.get_group_range()             \
                << cl::sycl::endl;                                             \
      kernelout << "group              " << item.get_group().get_id()          \
                << cl::sycl::endl;                                             \
      kernelout << "group linear id    " << item.get_group_linear_id()         \
                << cl::sycl::endl;                                             \
      kernelout << "local range        " << item.get_local_range()             \
                << cl::sycl::endl;                                             \
      kernelout << "local id           " << item.get_local_id()                \
                << cl::sycl::endl;                                             \
      kernelout << "local linear id    " << item.get_local_linear_id()         \
                << cl::sycl::endl                                              \
                << cl::sycl::endl;                                             \
      kernelout << "subgroup group range         " << sg.get_group_range()     \
                << cl::sycl::endl;                                             \
      kernelout << "subgroup group id            " << sg.get_group_id()        \
                << cl::sycl::endl;                                             \
      kernelout << "subgroup local range         " << sg.get_local_range()     \
                << cl::sycl::endl;                                             \
      kernelout << "subgroup local id            " << sg.get_local_id()        \
                << cl::sycl::endl;                                             \
      kernelout << "subgroup uniform group range "                             \
                << sg.get_uniform_group_range() << cl::sycl::endl;             \
      kernelout << "subgroup max local range     " << sg.get_max_local_range() \
                << cl::sycl::endl;                                             \
    }                                                                          \
  }

int main(int argc, char *argv[]) {
  auto choice = (argc > 1) ? atoi(argv[1]) : 1;
  if (!choice)
    choice = 1;

  auto usehost = (argc > 2) ? (argv[2][0] == 'H') || (argv[2][0] == 'h') : 0;

  // Set up SYCL device and queue.
  // queue q = queue( usehost ? host_selector() : default_selector() );
  queue q = usehost ? queue(host_selector()) : queue(default_selector());

  if (usehost) {
    std::cout << "host_selector being used" << std::endl;
  }

  // ultimately we will be 12 x 8 x 12 in 3D
  const uint32_t D1 = 12;
  const uint32_t D2 = 8;
  const uint32_t D3 = 16;
  const uint32_t D3a = 12;
  uint32_t G1, G2, G3, L1, L2, L3;

  std::vector<int> x(D1 * D2 * D3);
  for (auto i = 0; i < D1 * D2 * D3; ++i)
    x[i] = i + 100000;

  {
    buffer<int, 1> x_buf(x.data(), range<1>(D1 * D2 * D3));
    q.submit([&](handler &cgh) {
      auto xx = x_buf.get_access<access::mode::read_write>(cgh);
      cl::sycl::stream kernelout(10 * 1024, 20 * 512, cgh);

      //
      // I got VERY tired of making up unique class names for each lambda
      // so I'm using the -fsycl-unnamed-lambda compile option now
      // without it - the code will not compile
      //
      switch (choice) {
      default:
      case 1:
        G1 = 12;
        G2 = 8;
        G3 = 16;
        L1 = 6;
        L2 = 2;
        L3 = 4;
        break;
      case 2:
        G1 = 12;
        G2 = 8;
        G3 = 16;
        L1 = 3;
        L2 = 2;
        L3 = 4;
        break;
      case 3:
        G1 = 12;
        G2 = 8;
        G3 = 16;
        L1 = 4;
        L2 = 4;
        L3 = 4;
        break;
      case 4:
        G1 = 12;
        G2 = 8;
        G3 = 16;
        L1 = 3;
        L2 = 4;
        L3 = 8;
        break;
      }
      const range<3> glb(G1, G2, G3);
      const range<3> lcl(L1, L2, L3);
      std::cout << "nd_range<3>("
                << "{" << glb[0] << "," << glb[1] << "," << glb[2] << "}{"
                << lcl[0] << "," << lcl[1] << "," << lcl[2] << "})"
                << std::endl;
      cgh.parallel_for<>(nd_range<3>(glb, lcl), KernelToPrintsNdItem6_1_0);
    });
  }

  return 0;
}

#if 0
// The output...

#endif
