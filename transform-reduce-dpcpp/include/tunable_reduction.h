#pragma once

namespace impl {

  template <int block_size_y = 1>
  class TunableMultiReduction 
  {
    static_assert(block_size_y <= 4, "only block_size_y <= 4 supported");

  protected:

    template <typename policy_t, template <typename> class Transformer, typename Arg>
    void launch_device(const policy_t &policy, Arg &arg)
    {
      MultiReduction<impl::device::max_multi_reduce_block_size(), block_size_y, policy_t, Transformer, Arg>(policy, arg);
      return;
    }
    
    template <typename policy_t, template <typename> class Transformer, typename T, int n, typename Arg>
    void launch(const policy_t &policy, std::array<T, n> &result, Arg &arg)//arg is transform_reducer from transform_reduce_helper.h
    {
      launch_device<policy_t, Transformer>(policy, arg);
      //get the result:
      arg.template complete<policy_t, T, n>(policy, result);
      
      return;
    }


  public:

    TunableMultiReduction() { }

  };

}



