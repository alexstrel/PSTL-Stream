#include "reduce_helper.h"
#include "reduce_init.h"

// These are used for reduction kernels
static device_reduce_t *d_reduce  = nullptr;
static device_reduce_t *h_reduce  = nullptr;
static device_reduce_t *hd_reduce = nullptr;

static count_t *reduce_count      = nullptr;


namespace impl
{

  namespace reducer
  {

    // FIXME need to dynamically resize these
    void *get_device_buffer() { return d_reduce;     }
    void *get_mapped_buffer() { return hd_reduce;    }
    void *get_host_buffer()   { return h_reduce;     }
    count_t *get_count()      { return reduce_count; }

    size_t buffer_count()
    {
      /* we have these different reductions to cater for:

         - regular reductions (reduce_quda.cu) where are reducing to a
           single vector type (max length 4 presently), and a
           grid-stride loop with max number of blocks = 2 x SM count

         - multi-reductions where we are reducing to a matrix of size
           of size QUDA_MAX_MULTI_REDUCE of vectors (max length 4),
           and a grid-stride loop with maximum number of blocks = 2 x
           SM count
      */

      const int reduce_size       = 4;
      const int max_reduce        = reduce_size;
      const int max_multi_reduce  = max_n_reduce() * reduce_size;
      const int max_reduce_blocks = 2 * device::processor_count();

      // reduction buffer number of elements
      return max_reduce_blocks * std::max(max_reduce, max_multi_reduce);
    }

    template <typename T>
    struct init_reduce {
      T *reduce_count;
      long long bytes()         const { return max_n_reduce() * sizeof(T); }
      unsigned int minThreads() const { return max_n_reduce(); }

      init_reduce(sycl::queue cq, T *reduce_count) : reduce_count(reduce_count) { 
        apply(cq);
        
        return; 
      }

      void apply(sycl::queue comm_queue)
      {
        init_arg<T> arg(reduce_count);

        auto grange = range<3>(arg.threads[0], arg.threads[1], arg.threads[2]);//96
        auto lrange = range<3>(8, 1, 1);//Intel Xe graphics        
        //
        std::cout << "Allocate resources for the reduction operations : " << std::endl; 
        comm_queue.parallel_for(nd_range<3>(grange, lrange),
           [=, arg_ref = arg](nd_item<3> it) {
             auto gi = it.get_global_linear_id();

             init_count f(arg_ref);

             f(gi);
         }).wait();     
         
         std::cout << "... done. " << std::endl; 
       }
    };

    void init( sycl::queue cq )
    {
      auto n = buffer_count();
      
      if (!d_reduce) d_reduce = malloc_device<device_reduce_t>(n, cq);

      // these arrays are actually oversized currently (only needs to be device_reduce_t x 3)

      // if the device supports host-mapped memory then use a host-mapped array for the reduction
      if (!h_reduce) {
        h_reduce  = malloc_host<device_reduce_t>(n, cq); 
        cq.wait();
        hd_reduce = h_reduce; // set the matching device pointer

        cq.memset(h_reduce, 0, n*sizeof(device_reduce_t)).wait(); // added to ensure that valgrind doesn't report h_reduce is unitialised
      }

      if (!reduce_count) {
        reduce_count = malloc_device<count_t>(max_n_reduce(), cq);
        init_reduce<count_t> init(cq, reduce_count);
      }
    }

    void destroy( sycl::queue cq)
    {
      cq.wait();
      
      std::cout << "Cleaning resources.. " << std::endl;

      if (reduce_count) {
        free(reduce_count, cq);
        reduce_count = nullptr;
      }
      if (d_reduce) {
        free(d_reduce, cq);
        d_reduce = nullptr;
      }
      if (h_reduce) {
        free(h_reduce, cq);
        h_reduce = nullptr;
      }
      hd_reduce = nullptr;
    }

  } // namespace reducer
} // namespace impl
