#pragma once

#include "reduce_helper.h"

namespace impl {

  namespace reducer {

    template <typename T_> struct init_arg {
      using T = T_;
      T *count;
      std::array<int, 3> threads;
      init_arg(T *count) :
        count(count),
        threads({max_n_reduce(), 1, 1}) {}
    };

    template <typename Arg> struct init_count {
      Arg &arg;
      constexpr init_count(Arg &arg) : arg(arg) {}
      void operator()(int i) { new (arg.count + i) typename Arg::T {0}; }
    };

  }
}
