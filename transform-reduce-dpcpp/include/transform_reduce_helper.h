#pragma once

#include <reduction_kernel.h>
#include <limits>

namespace impl {

  template <typename reduce_t, int n, typename reducer_, typename transformer>
  struct TransformReduceArg : public ReduceArg<reduce_t> {
    using reducer = reducer_;
    static constexpr int n_batch_max = 8;//?

    int n_items;
    int n_batch;
    
    reduce_t init_value;
    reducer r;
    transformer h;    
    
    std::array<int, 3> threads;

    TransformReduceArg(int n_items, reduce_t init_value, reducer r, transformer h) :
      ReduceArg<reduce_t>(),
      n_items(n_items),
      n_batch(n),
      init_value(init_value),
      r(r),
      h(h),      
      threads({n_items, n_batch, 1})
    {
      if (n_items > std::numeric_limits<count_t>::max()) 
        std::cerr << "Requested size greater than max supported : " << (uint64_t)n_items << " > " << (uint64_t)std::numeric_limits<count_t>::max() << std::endl;
    }

    reduce_t init() const { return init_value; }
  };

  template <typename Arg> struct transform_reducer {//this functor initialized in reduction_kernel.h
    using count_t  = decltype(Arg::n_items);
    using reduce_t = decltype(Arg::init_value);

    Arg arg;//keeps TransfromReduceArg object
    //

    constexpr transform_reducer(Arg arg) : arg(arg) {}//no refs
    
    inline reduce_t operator()(reduce_t a, reduce_t b) const { return arg.r(a, b); }

    inline reduce_t operator() (reduce_t &value, count_t i, count_t j, count_t k) const//j is a batch index 
    {
      auto t = arg.h(i, j);//i is an array index, j is a batch index
      return arg.r(t, value);//?? batch version ??
    }
  };

}
