#pragma once

#include <reduce_helper.h>

namespace impl {
  //MultiReduction is called from tunable_reduction.h
  template <int block_size_x, int block_size_y, typename policy_t, template <typename> class Transformer, typename Arg, bool grid_stride = true>
  void MultiReduction(const policy_t &policy, Arg arg)//arg contains TransformReduceArg
  {
    using reduce_t = typename Transformer<Arg>::reduce_t;
    Transformer<Arg> t(arg);//that initializes transform_reducer functor from transform_reduce_helper.h

    auto comm_queue = policy.get_queue(); 
    
    auto gRng = sycl::range<3>(arg.threads[0], arg.threads[1], arg.threads[2]);
    auto lRng = sycl::range<3>(block_size_x , block_size_y , arg.threads[2] );    

    comm_queue.parallel_for(nd_range<3>(gRng, lRng),
       [=, arg_ref = arg, t_ref = t](nd_item<3> it) {
       
           auto i = it.get_global_id(0);
           auto j = it.get_global_id(1);
           auto k = it.get_local_id(2) ;
           
           auto gRngX = it.get_global_range(0);
           auto gRngY = it.get_global_range(1);

           if (j >= gRngY) return;

           reduce_t value = arg_ref.init();//comes from TransferArgs, just get value

           while (i < gRngX) {
             value = t_ref(value, i, j, k);//transformer+reduction , ok 
             if (grid_stride) i += gRngX; else break;//warning!
          }
          // perform final inter-block reduction and write out result
          reduce<block_size_x, block_size_y, Arg, decltype(t_ref), reduce_t>(it, arg_ref, t_ref, value, j);//see reduce helper
       });
    
  }
  

}
