//==============================================================
// Copyright (c) 2020 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0
// =============================================================

// oneDPL headers should be included before standard headers!
#include <oneapi/dpl/algorithm>
#include <oneapi/dpl/execution>
#include <oneapi/dpl/iterator>
#include <oneapi/dpl/random>

#include <chrono>
#include <iostream>
#include <vector>
#include <sys/time.h>
#include <complex>

#include <CL/sycl.hpp>

//#define N 500

int main() {

    const int N = 4096*32768;// std::numeric_limits<int>::max();

    struct timeval time_begin, time_end;

    gettimeofday(&time_begin, NULL);

    using oneapi::dpl::counting_iterator;

    {
        sycl::queue q; //(sycl::gpu_selector{});

        cl::sycl::usm_allocator<std::complex<float>, cl::sycl::usm::alloc::shared> alloc(q);

        std::vector<std::complex<float>, decltype(alloc)> x(N, alloc);
        std::vector<std::complex<float>, decltype(alloc)> y(N, alloc);

        std::complex<float> a{0.1f, 0.2f};

        auto policy = oneapi::dpl::execution::make_device_policy(q);
        //auto policy = oneapi::dpl::execution::dpcpp_default;

        fill(policy, x.begin(), x.end(), std::complex<float>(1.0, 1.0));
        fill(policy, y.begin(), y.end(), std::complex<float>(2.0, 2.0));

        auto sum = transform_reduce( policy, counting_iterator(0), counting_iterator(N), std::complex<float>(0.0f, 0.0f), std::plus<std::complex<float>>{}
                                    , [=, x_ = x.data(), y_ = y.data()](const auto i) {
                                        y_[i] = a*x_[i]+y_[i];
                                        return (conj(y_[i])*x_[i]);
                                    });
        printf("CHECK (%le, %le)\n", sum.real(), sum.imag());
    }
    gettimeofday(&time_end, NULL);
    double elapsed_time = (time_end.tv_sec - time_begin.tv_sec)+(time_end.tv_usec - time_begin.tv_usec)*1.0e-6;
    // Printing Results
    std::cout << "Elapsed time  "  << elapsed_time << std::endl;

    return 0;
}
