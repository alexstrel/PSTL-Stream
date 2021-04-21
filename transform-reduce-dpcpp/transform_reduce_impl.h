#pragma once

#include "device.h"
#include "transform_reduce_helper.h"
#include "tunable_reduction.h"

namespace impl
{

  template <typename policy_t, typename reduce_t, int n_batch, typename count_t, typename reducer, typename transformer>
  class TransformReduce : TunableMultiReduction<1>
  {
    using Arg = TransformReduceArg<reduce_t, n_batch, count_t, reducer, transformer>;//define in transform_reduce_helper.h
    
    policy_t &policy;//copy policy?
    
    std::array<reduce_t, n_batch> &result;
    
    count_t n_items;
    reduce_t init;
    reducer r;
    transformer h;

  public:
    TransformReduce(policy_t &policy, std::array<reduce_t, n_batch> &result, count_t n_items, reduce_t init, reducer r, transformer h) :
      TunableMultiReduction(),
      policy(policy),
      result(result),
      n_items(n_items),
      init(init),
      r(r),
      h(h)      
    {
      //apply(policy); //for autotuning?
    }

    void apply( ) // former argument was const qudaStream_t &stream
    {
      Arg arg(n_items, init, r, h);//defined in transform_reduce_helper.h (TransformReduceArg) and reduction_helper.h (ReduceArg)
      //Remark :: transform_reducer functor defined in transform_reduce.cuh, launch defined in tunable_reduction.h see TunableMultiReduction
      launch<policy_t, transform_reducer, reduce_t, n_batch>(policy, result, arg);//explicit pass to transform reducer
      //
      return;
    }
  };


  template <typename Policy, typename reduce_t, typename count_t, typename reducer, typename transformer, bool is_async = true>
  reduce_t transform_reduce(Policy &policy, count_t n_items, reduce_t init, reducer r, transformer h)
  {
    std::array<reduce_t, 1> result{init};
    
    TransformReduce<Policy, reduce_t, 1, count_t, reducer, transformer> transformReducer(policy, result, n_items, init, r, h);
    transformReducer.apply();
    
    if constexpr (!is_async) policy.get_queue().wait();
    
    return result[0];
  }
  
  template <typename Policy, typename reduce_t, int n_batch, typename count_t, typename reducer, typename transformer, bool is_async = true>
  void transform_reduce(Policy &policy, count_t n_items, std::array<reduce_t, n_batch> result, reduce_t init, reducer r, transformer h)
  {
    constexpr int n_batch_ = result.size();
    TransformReduce<Policy, reduce_t, n_batch_, count_t, reducer, transformer> transformReducer(policy, result, n_items, init, r, h);
    transformReducer.apply();
    
    if constexpr (!is_async) policy.get_queue().wait();
    
    return;
  }

} // namespace impl



