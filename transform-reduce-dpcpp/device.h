#pragma once
#include <CL/sycl.hpp>


namespace impl
{

  namespace device
  {
  
    void init( const sycl::queue &custom_queue );
    void init(  ); 
    
    const sycl::queue& get_default_queue(); 
    const sycl::queue& get_current_queue();
  
    class PolicyImpl {
    
      const sycl::queue &q_ref;
       
    public:
       
      PolicyImpl() : q_ref(get_default_queue()) {
        init();
        //
        return; 
      }
       
      PolicyImpl(const sycl::queue &custom_queue) : q_ref(custom_queue) {
        //now initialize reduction buffers:
        init(custom_queue);
        //
        return; 
      }

      ~PolicyImpl() { clean_resources(); }
       
      const sycl::queue& get_queue() const { return q_ref; }
      //
      void clean_resources();
      //
    };
  
    PolicyImpl& make_device_policy(const sycl::queue &q);
    
    PolicyImpl& make_device_policy( );
  
    int  processor_count(); 
    
    constexpr unsigned int  max_multi_reduce_block_size() {return 128;}//returns the maximum number of threads  in a block in the x dimension for reduction kernels
  }
  
}// impl ns









