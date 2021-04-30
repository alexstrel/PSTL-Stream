#include <vector>
#include <cstdio>

#include <transform_reduce_impl.h>

#define USE_LAMBDA

template <typename T, typename count_t> struct compute_axpyDot {
  const T *x;
  T *y;
  const T a;
  count_t n_items;

  compute_axpyDot(const T a_, const T *x_, T *y_,  count_t n) : a(a_), x(x_), y(y_), n_items(n) {}

  T operator() (count_t idx, count_t j = 0) const {
    y[idx] = a*x[idx] + y[idx];
    return (y[idx]*x[idx]);
  }
};

int main() {

  using data_t   = float;
  using reduce_t = float;
  using alloc_t  = sycl::usm_allocator<data_t, sycl::usm::alloc::shared>;
//internal queue: 
  auto p = impl::device::make_device_policy();
  auto q = p.get_queue();
 
  const int n = 262144;

  std::vector<data_t, alloc_t> x(n, alloc_t(q));
  std::vector<data_t, alloc_t> y(n, alloc_t(q));

  for(int i=0; i<n; i++) {  x[i] = i + 1; }
  for(int i=0; i<n; i++) {  y[i] = i + 2; }

  data_t a = 2.2f;

  std::array<reduce_t, 1> r;
#ifndef USE_LAMBDA 
  std::unique_ptr< compute_axpyDot<data_t, int> > fn_ptr(new compute_axpyDot(a, x.data(), y.data(), n));
  auto &fn_ref = *fn_ptr;
  constexpr int nbatch = r.size();

  impl::transform_reduce<decltype(p), data_t, nbatch>(p, 0, n, r, 0.0f, ONEAPI::plus<data_t>(), fn_ref);
#else
  auto compute_axpyDot_ = [=, x_ = x.data(), y_ = y.data()] (const int i, const int j = 0) {y_[i] = a*x_[i]+y_[i]; return (y_[i]*x_[i]);};
  constexpr int nbatch = r.size();

  impl::transform_reduce<decltype(p), data_t, nbatch>(p, 0, n, r, 0.0f, ONEAPI::plus<data_t>(), compute_axpyDot_);
#endif

  q.wait();
  printf("r: %g\n", r[0]);
  
  for(int i=0; i<n; i++) {  x[i] = i + 1; }
  for(int i=0; i<n; i++) {  y[i] = i + 2; }

  double rc = 0.0;
  for(int i=0; i<n; i++) {
    auto ynew = a*x[i] + y[i];
    rc += ynew*x[i];
  }
  if(fabs( (rc-r[0]) / rc ) > 1e-7) {
    std::cout << r[0] << " != " << rc << " diff = " << fabs( (rc-r[0]) / rc) << std::endl;
  }

  
  return 0;

}

