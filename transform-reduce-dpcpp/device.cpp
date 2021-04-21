#include "device.h"
#include <memory>


using namespace std;
using namespace sycl;

namespace impl
{

  static sycl::queue default_queue;
  
  namespace reducer
  {
    /** allocate internal buffers: */
    void init( sycl::queue cq );
    
    /** deallocate internal buffers: */
    void destroy( sycl::queue cq );    
    
  } // namespace reducer  
  
  namespace device
  {

    static bool initialized  = false;
    
    static std::shared_ptr<sycl::device> dev_ptr            = nullptr;
    static std::shared_ptr<PolicyImpl>   default_policy_ptr = nullptr;    

    const sycl::queue& get_default_queue() {return default_queue;}
    const sycl::queue& get_current_queue() {return default_policy_ptr->get_queue();}
    
    void init_device( const sycl::queue &q) {
      if (initialized) return;
      initialized = true;
     
      if( !dev_ptr ) dev_ptr = std::make_shared<sycl::device>(q.get_device());//check!
      
      std::cout << "Device: " << dev_ptr->get_info<info::device::name>() << std::endl;
      
      return;
    }// init_device

    void init( const sycl::queue &custom_queue ) { init_device(custom_queue); }
    void init(  ) { init_device(default_queue); } 
          
    PolicyImpl& make_device_policy(const sycl::queue &q){
  
      default_policy_ptr = std::make_shared<PolicyImpl>(q);

      reducer::init(q);
      
      return *default_policy_ptr;
    }   

    PolicyImpl& make_device_policy(){
  
      default_policy_ptr = std::make_shared<PolicyImpl>();
      
      reducer::init(default_policy_ptr->get_queue());
      
      return *default_policy_ptr;
    }   

    void PolicyImpl::clean_resources() { reducer::destroy(q_ref); };
  
    int processor_count() { return dev_ptr->get_info<info::device::max_compute_units>(); }  
    
  }// device ns
  
}// impl ns









