#pragma once

#include <iostream>
#include <stdexcept>
#include <string>

#include <algorithm>
#include <vector>
#include <memory>
#include <numeric>
#include <execution>


#ifndef __NVCOMPILER_CUDA__

#if defined(__INTEL_COMPILER)
#include <malloc.h>
#else
#include <mm_malloc.h>
#endif

#include <tbb/tbb.h>
using namespace tbb;

constexpr int alloc_align  = (2*1024*1024);

#else

#include <thrust/iterator/counting_iterator.h>
using namespace thrust;

#endif //__NVCOMPILER_CUDA__

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
#else
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
#else

#if defined(__INTEL_COMPILER)
         free((void*)p);
#else
         _mm_free((void *)p);
#endif

#endif
       }
     };


#define IMPLEMENTATION_STRING "PSTL"

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


template <typename T, typename Policy, template<typename Tp> class Allocator> class PSTLStream : public Stream<T> {
  protected:
    // Device side refs
    Policy &p;
    std::vector<T, Allocator<T>> a;
    std::vector<T, Allocator<T>> b;
    std::vector<T, Allocator<T>> c;

  public:
    PSTLStream(Policy &p_, const int N) : p(p_), a(N), b(N), c(N)
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

template <typename T, typename Policy, template<typename Tp> class Allocator>
void PSTLStream<T, Policy, Allocator>::init_arrays(T &&initA, T &&initB, T &&initC)
{
  std::fill(p, a.begin(), a.end(), initA);
  std::fill(p, b.begin(), b.end(), initB);
  std::fill(p, c.begin(), c.end(), initC);
}

template <typename T, typename Policy, template<typename Tp> class Allocator>
void PSTLStream<T, Policy, Allocator>::copy()
{
  std::copy(p, a.begin(), a.end(), c.begin());
}

template <typename T, typename Policy, template<typename Tp> class Allocator>
void PSTLStream<T, Policy, Allocator>::mul(const T &s_)
{
  const T s = s_;
  const int N = b.size();
  std::for_each(p, counting_iterator(0), counting_iterator(N), [=](auto i) { b[i] = s*c[i];});
}

template <typename T, typename Policy, template<typename Tp> class Allocator>
void PSTLStream<T, Policy, Allocator>::add()
{
  const int N    = c.size();
  std::for_each(p, counting_iterator(0), counting_iterator(N), [&](auto i) { c[i] = a[i] + b[i];});
}

template <typename T, typename Policy, template<typename Tp> class Allocator>
void PSTLStream<T, Policy, Allocator>::triad(const T &s_)
{
  const T s   = s_;
  const int N = a.size();
  std::for_each(p, counting_iterator(0), counting_iterator(N), [=](auto i) { a[i] = b[i] + s*c[i];});
}

template <typename T, typename Policy, template<typename Tp> class Allocator>
T PSTLStream<T, Policy, Allocator>::dot()
{
  T sum = std::transform_reduce(p,
                                a.begin(),
                                a.end(),
                                b.begin(),
                                static_cast<T>(0.0),
                                std::plus<T>(),
                                [=](auto &ai, auto &bi) { return ai*bi;} );
  return sum;
}
