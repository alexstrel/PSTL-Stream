#pragma once

#ifdef DPCPP_BACKEND

#include <oneapi/dpl/algorithm>
#include <oneapi/dpl/execution>//<= why does it need tbb?
#include <oneapi/dpl/iterator>
#include <oneapi/dpl/random>

#else

#include <algorithm>
#include <execution>
#include <numeric>
#include <limits>

#endif

#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
#include <memory>

//BACKEND selector
#if defined (DPCPP_BACKEND)

#include <CL/sycl.hpp>
using oneapi::dpl::counting_iterator;

#else

#if defined(__INTEL_COMPILER)
#include <malloc.h>
#else
#include <mm_malloc.h>
#endif

constexpr int alloc_align  = (2*1024*1024);

#endif //BACKEND selector

   template<typename Tp>
   struct AlignedAllocator {
     public:

       typedef Tp value_type;

       AlignedAllocator () {};

       AlignedAllocator(const AlignedAllocator&) { }

       template<typename Tp1> constexpr AlignedAllocator(const AlignedAllocator<Tp1>&) { }

       ~AlignedAllocator() { }

       Tp* address(Tp& x) const { return &x; }

       std::size_t  max_size() const throw() { return size_t(-1) / sizeof(Tp); }

       [[nodiscard]] Tp* allocate(std::size_t n){

         Tp* ptr = nullptr;
#ifdef __NVCOMPILER_CUDA__
         auto err = cudaMallocManaged((void **)&ptr,n*sizeof(Tp));

         if( err != cudaSuccess ) {
           ptr = (Tp *) NULL;
           std::cerr << " cudaMallocManaged failed for " << n*sizeof(Tp) << " bytes " <<cudaGetErrorString(err)<< std::endl;
           assert(0);
         }
#elif !defined(DPCPP_BACKEND)
         //ptr = (Tp*)aligned_malloc(alloc_align, n*sizeof(Tp));
#if defined(__INTEL_COMPILER)
         ptr = (Tp*)malloc(bytes);
#else
         ptr = (Tp*)_mm_malloc(n*sizeof(Tp),alloc_align);
#endif
         if(!ptr) throw std::bad_alloc();
#endif

         return ptr;
       }

      void deallocate( Tp* p, std::size_t n) noexcept {
#ifdef __NVCOMPILER_CUDA__
         cudaFree((void *)p);
#elif !defined(DPCPP_BACKEND)

#if defined(__INTEL_COMPILER)
         free((void*)p);
#else
         _mm_free((void *)p);
#endif

#endif
       }
     };
     
 namespace impl {

   template <typename IntType>
   class counting_iterator {
       static_assert(std::numeric_limits<IntType>::is_integer, "Cannot instantiate counting_iterator with a non-integer type");
     public:
       using value_type = IntType;
       using difference_type = typename std::make_signed<IntType>::type;
       using pointer = IntType*;
       using reference = IntType&;
       using iterator_category = std::random_access_iterator_tag;

       counting_iterator() : value(0) { }
       explicit counting_iterator(IntType v) : value(v) { }

       value_type operator*() const { return value; }
       value_type operator[](difference_type n) const { return value + n; }

       counting_iterator& operator++() { ++value; return *this; }
       counting_iterator operator++(int) {
         counting_iterator result{value};
         ++value;
         return result;
       }  
       counting_iterator& operator--() { --value; return *this; }
       counting_iterator operator--(int) {
         counting_iterator result{value};
         --value;
         return result;
       }
       counting_iterator& operator+=(difference_type n) { value += n; return *this; }
       counting_iterator& operator-=(difference_type n) { value -= n; return *this; }

       friend counting_iterator operator+(counting_iterator const& i, difference_type n)          { return counting_iterator(i.value + n);  }
       friend counting_iterator operator+(difference_type n, counting_iterator const& i)          { return counting_iterator(i.value + n);  }
       friend difference_type   operator-(counting_iterator const& x, counting_iterator const& y) { return x.value - y.value;  }
       friend counting_iterator operator-(counting_iterator const& i, difference_type n)          { return counting_iterator(i.value - n);  }

       friend bool operator==(counting_iterator const& x, counting_iterator const& y) { return x.value == y.value;  }
       friend bool operator!=(counting_iterator const& x, counting_iterator const& y) { return x.value != y.value;  }
       friend bool operator<(counting_iterator const& x, counting_iterator const& y)  { return x.value < y.value; }
       friend bool operator<=(counting_iterator const& x, counting_iterator const& y) { return x.value <= y.value; }
       friend bool operator>(counting_iterator const& x, counting_iterator const& y)  { return x.value > y.value; }
       friend bool operator>=(counting_iterator const& x, counting_iterator const& y) { return x.value >= y.value; }

     private:
       IntType value;
   };

} //impl



#define IMPLEMENTATION_STRING "PSTL"

#ifdef DPCPP_BACKEND
namespace pstl_impl = oneapi::dpl;
#else
namespace pstl_impl = std;
using namespace impl;
#endif


template <typename T> class Stream {
  public:

    virtual ~Stream(){}

    // Kernels
    // These must be blocking calls
    virtual void copy() = 0;
    virtual void mul(const T &s) = 0;
    virtual void add() = 0;
    virtual void triad(const T &s) = 0;
    virtual T dot() = 0;

    // Copy memory between host and device
    virtual void init_arrays(T &&initA, T &&initB, T &&initC) = 0;
};


template <typename T, typename Policy, class Allocator> class PSTLStream : public Stream<T> {
  protected:
    // Device side refs
    Policy &p;
    std::vector<T, Allocator> a;
    std::vector<T, Allocator> b;
    std::vector<T, Allocator> c;

  public:
#if !defined(DPCPP_BACKEND)  //var args!
    PSTLStream(Policy &p_, const int N) : p(p_), a(N), b(N), c(N)
#else    
    PSTLStream(Policy &p_, const int N, cl::sycl::usm_allocator<T, cl::sycl::usm::alloc::shared> alloc) : p(p_), a(N, alloc), b(N, alloc), c(N, alloc)
#endif    
    {}
    ~PSTLStream()
    {}

    virtual void copy()             override;
    virtual void add()              override;
    virtual void mul(const T &s)   override;
    virtual void triad(const T &s) override;
    virtual T dot()                 override;

    virtual void init_arrays(T &&initA, T &&initB, T &&initC) override;
};

template <typename T, typename Policy, class Allocator>
void PSTLStream<T, Policy, Allocator>::init_arrays(T &&initA, T &&initB, T &&initC)
{
  pstl_impl::fill(p, a.begin(), a.end(), initA);
  pstl_impl::fill(p, b.begin(), b.end(), initB);
  pstl_impl::fill(p, c.begin(), c.end(), initC);
}

template <typename T, typename Policy, class Allocator>
void PSTLStream<T, Policy, Allocator>::copy()
{
  pstl_impl::copy(p, a.begin(), a.end(), c.begin());
}

template <typename T, typename Policy, class Allocator>
void PSTLStream<T, Policy, Allocator>::mul(const T &s_)
{
  const T s = s_;
  const int N = b.size();
  pstl_impl::for_each(p, counting_iterator(0), counting_iterator(N), [=, b_= b.data(), c_ = c.data() ](const auto i) { b_[i] = s*c_[i];});
}

template <typename T, typename Policy, class Allocator>
void PSTLStream<T, Policy, Allocator>::add()
{
  const int N    = c.size();
  pstl_impl::for_each(p, counting_iterator(0), counting_iterator(N), [a_ = a.data(), b_= b.data(), c_ = c.data()](const auto i) { c_[i] = a_[i] + b_[i];});
}

template <typename T, typename Policy, class Allocator>
void PSTLStream<T, Policy, Allocator>::triad(const T &s_)
{
  const T s   = s_;
  const int N = a.size();
  pstl_impl::for_each(p, counting_iterator(0), counting_iterator(N), [=, a_ = a.data(), b_= b.data(), c_ = c.data()](const auto i) { a_[i] = b_[i] + s*c_[i];});
}

template <typename T, typename Policy, class Allocator>
T PSTLStream<T, Policy, Allocator>::dot()
{
  T sum = pstl_impl::transform_reduce(p,
                                a.begin(),
                                a.end(),
                                b.begin(),
                                static_cast<T>(0.0),
                                pstl_impl::plus<T>(),
                                [=](const auto &ai, const auto &bi) { return ai*bi;} );
  return sum;
}
