#pragma once

#include <CL/sycl.hpp>
#include <device.h>

using namespace sycl;

using device_reduce_t = float;
using count_t         = unsigned int;

constexpr int MAX_MULTI_REDUCE = 16;

namespace impl
{

  namespace reducer
  {
    /** returns the reduce buffer size allocated */
    size_t buffer_count();

    void*       get_device_buffer();
    void*       get_mapped_buffer();
    void*       get_host_buffer();
    count_t*    get_count();
  } // namespace reducer

  constexpr int max_n_reduce() { return MAX_MULTI_REDUCE; }

  template <typename T> constexpr T init_value()      { return -std::numeric_limits<T>::infinity(); }

  template <typename T> constexpr T terminate_value() { return std::numeric_limits<T>::infinity(); }


  template <typename T> struct ReduceArg {

    template <int blkx, int blky, typename Arg, typename Reducer, typename I>
    friend void reduce(sycl::nd_item<3> it, const Arg, const Reducer , const I &, const int);//?

  private:
    const int n_reduce; // number of reductions of length n_item
    bool reset = false; // reset the counter post completion (required for multiple calls with the same arg instance
    
    T *partial;
    T *result_d;//keep the device reducion output
    T *result_h;

    count_t *count;

  public:
    /**
       @brief Constructor for ReduceArg
       @param[in] n_reduce The number of reductions of length n_item
       @param[in] reset Whether to reset the atomics after the
       reduction has completed; required if the same ReduceArg
       instance will be used for multiple reductions.
    */
    ReduceArg(int n_reduce = 1, bool reset = false) :
      n_reduce(n_reduce),
      reset(reset),
      partial (static_cast<decltype(partial )>(reducer::get_device_buffer())),
      result_d(static_cast<decltype(result_d)>(reducer::get_mapped_buffer())),
      result_h(static_cast<decltype(result_h)>(reducer::get_host_buffer())  ),
      count {reducer::get_count()}
    {
      // check reduction buffers are large enough if requested
      auto max_reduce_blocks = 2 * impl::device::processor_count();//e.g. : deviceProp.multiProcessorCount
      auto reduce_count      = max_reduce_blocks * n_reduce;
      
      if (reduce_count > reducer::buffer_count()) std::cerr << "Requested reduction requires a larger buffer than allocated : " << reduce_count*sizeof(*partial) << " > " << reducer::buffer_count()*sizeof(*partial) << std::endl;

      //if (commAsyncReduction()) result_d = partial;
    }
    
    template <typename policy_t, typename host_t, int n, typename device_t = host_t>
    void complete(const policy_t policy, std::array<host_t, n> &result)
    {
      auto q = policy.get_queue();
      q.wait(); 

      // copy back result element by element and convert if necessary to host reduce type
      // unit size here may differ from system_atomic_t size, e.g., if doing double-double
      const int n_element = n_reduce * sizeof(host_t) / sizeof(device_t);
      if (result.size() != (unsigned)n_element)
        std::cerr << "Result vector length does not match n_reduce : " << result.size() << " :: "<< n_element << std::endl;
      for (int i = 0; i < n_element; i++) result[i] = reinterpret_cast<device_t *>(result_h)[i];
    }      
  };

    /**
       @brief Generic reduction function that reduces block-distributed
       data "in" per thread to a single value.  This is the legacy
       variant which require explicit host-device synchronization to
       signal the completion of the reduction to the host.

       @param in The input per-thread data to be reduced
       @param idx In the case of multiple reductions, idx identifies
       which reduction this thread block corresponds to.  Typically idx
       will be constant along constant blockIdx.y and blockIdx.z.
    */
    template <int block_size_x, int block_size_y = 1, typename Arg, typename Reducer, typename T>
    inline void reduce(sycl::nd_item<3> it, const Arg arg, const Reducer tr_fn, const T &in, const int idx = 0)//, r is a reference to transform_reducer idx is a batch index
    {
    
      using memory_order = ONEAPI::memory_order;
      using memory_scope = ONEAPI::memory_scope;
    
      auto this_grp = it.get_group();
      
      auto r{tr_fn.arg.r};//extract reducer from transform_reducer 

      T aggregate = ONEAPI::reduce(this_grp, in, r);


      auto ltid0 = it.get_local_id(0);
      auto ltid1 = it.get_local_id(1);

      auto grpRng0 = it.get_group_range(0);//!
      auto grpIdx0 = it.get_group(0);//!
      
      bool isLastBlockDone = false;//!
    
      if (ltid0 == 0 && ltid1 == 0) {
        arg.partial[idx*grpRng0 + grpIdx0] = aggregate;
        //!!new (arg.partial + idx*grpRng0 + grpIdx0) ONEAPI::atomic_ref<unsigned int, memory_order::acq_rel, memory_scope::device, access::address_space::global_space>{aggregate};

        atomic_fence(memory_order::relaxed, memory_scope::device);
        
        ONEAPI::atomic_ref<unsigned int, memory_order::acq_rel, memory_scope::device, access::address_space::global_space> acount(arg.count[idx]);        
        
        // increment global block counter
        auto value = acount.fetch_add(1);
        // determine if last block
        isLastBlockDone = (value == (grpRng0 - 1));
      }

      isLastBlockDone = ONEAPI::broadcast(this_grp, isLastBlockDone);

      // finish the reduction if last block
      if (isLastBlockDone) {
        //auto i = threadIdx.y * block_size_x + threadIdx.x;
        auto i = ltid1 * block_size_x + ltid0;
        // 
        T sum = arg.init();//set zero
        
        while (i < grpRng0) {
          sum = r(sum, const_cast<T &>(static_cast<volatile T *>(arg.partial)[idx * grpRng0 + i]));
          //!!sum = r(sum, arg.partial[idx * grpRng0 + i].load());
          i += block_size_x * block_size_y;
        }

        sum = ONEAPI::reduce(this_grp, sum, r); //(Reducer::do_sum ? BlockReduce(cub_tmp).Sum(sum) : BlockReduce(cub_tmp).Reduce(sum, r));

        // write out the final reduced value
        if (ltid0 == 0 && ltid1 == 0) {
          arg.result_d[idx] = sum;
          arg.count[idx]    = 0; // set to zero for next time
        }
      }
      
      return;
    }


} // namespace impl
