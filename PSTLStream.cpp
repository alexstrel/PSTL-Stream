#include <iostream>
#include <vector>
#include <numeric>
#include <cmath>
#include <limits>
#include <chrono>
#include <algorithm>
#include <iomanip>
#include <cstring>

#define VERSION_STRING "0.1"

#include "PSTLStream.h"

// Default size of 2^25
unsigned int N = 33554432;
unsigned int num_times = 100;
bool use_float     = false;
bool output_as_csv = false;
bool mibibytes     = false;
std::string csv_separator = ",";

template <typename T, template<typename Tp> class Allocator>
void check_solution(const unsigned int ntimes, std::vector<T, Allocator<T>>& a, std::vector<T, Allocator<T>>& b, std::vector<T, Allocator<T>>& c, T& sum);

template <typename T>
void run();


void parseArguments(int argc, char *argv[]);

int main(int argc, char *argv[])
{

  parseArguments(argc, argv);

  if (!output_as_csv)
  {
    std::cout
      << "Modified Stream benchmark" << std::endl
      << "Version: " << VERSION_STRING << std::endl
      << "Implementation: " << IMPLEMENTATION_STRING << std::endl;
  }


  if (use_float) run<float>();
  else run<double>();
 
  return 0;
}

template <typename T>
void run()
{
  std::streamsize ss = std::cout.precision();

  if (!output_as_csv)
  {
    std::cout << "Running kernels " << num_times << " times" << std::endl;

    if (sizeof(T) == sizeof(float))
      std::cout << "Precision: float" << std::endl;
    else
      std::cout << "Precision: double" << std::endl;


    if (mibibytes)
    {
      // MiB = 2^20
      std::cout << std::setprecision(1) << std::fixed
                << "Array size: " << N*sizeof(T)*pow(2.0, -20.0) << " MiB"
                << " (=" << N*sizeof(T)*pow(2.0, -30.0) << " GiB)" << std::endl;
      std::cout << "Total size: " << 3.0*N*sizeof(T)*pow(2.0, -20.0) << " MiB"
                << " (=" << 3.0*N*sizeof(T)*pow(2.0, -30.0) << " GiB)" << std::endl;
    }
    else
    {
      // MB = 10^6
      std::cout << std::setprecision(1) << std::fixed
                << "Array size: " << N*sizeof(T)*1.0E-6 << " MB"
                << " (=" << N*sizeof(T)*1.0E-9 << " GB)" << std::endl;
      std::cout << "Total size: " << 3.0*N*sizeof(T)*1.0E-6 << " MB"
                << " (=" << 3.0*N*sizeof(T)*1.0E-9 << " GB)" << std::endl;
    }
    std::cout.precision(ss);

  }

  // Create objects
#ifndef DPCPP_BACKEND  
  using alloc = AlignedAllocator<T>;
  
  auto policy = std::execution::par_unseq;  
#else
  sycl::queue q; //(sycl::gpu_selector{});
  
  cl::sycl::usm_allocator<T, cl::sycl::usm::alloc::shared> alloc_(q);
  
  using alloc   = decltype(alloc_);
  
  auto policy = oneapi::dpl::execution::make_device_policy(q);
  //auto policy = oneapi::dpl::execution::dpcpp_default;  
#endif  

  std::vector<T, alloc> a(N);
  std::vector<T, alloc> b(N);
  std::vector<T, alloc> c(N);


  std::unique_ptr<Stream<T>> stream_ptr(new PSTLStream<T, decltype(policy), alloc>(policy, N));

  auto &stream = *stream_ptr;
  stream.init_arrays(0.1, 0.2, 0.0);

  // List of times
  std::vector<std::vector<double>> timings(5);

  // Declare timers
  std::chrono::high_resolution_clock::time_point t1, t2;

  // Main loop
  T sum = 0.0;

  const T scalar = 0.4;

  for (unsigned int k = 0; k < num_times; k++)
  {
    // Execute Copy
    t1 = std::chrono::high_resolution_clock::now();
    stream.copy();
    t2 = std::chrono::high_resolution_clock::now();
    timings[0].push_back(std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count());

    // Execute Mul
    t1 = std::chrono::high_resolution_clock::now();
    stream.mul(scalar);
    t2 = std::chrono::high_resolution_clock::now();
    timings[1].push_back(std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count());

    // Execute Add
    t1 = std::chrono::high_resolution_clock::now();
    stream.add();
    t2 = std::chrono::high_resolution_clock::now();
    timings[2].push_back(std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count());

    // Execute Triad
    t1 = std::chrono::high_resolution_clock::now();
    stream.triad(scalar);
    t2 = std::chrono::high_resolution_clock::now();
    timings[3].push_back(std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count());

    // Execute Dot
    t1 = std::chrono::high_resolution_clock::now();
    sum = stream.dot();
    t2 = std::chrono::high_resolution_clock::now();
    timings[4].push_back(std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count());

  }

  // Check solutions
  //check_solution<T, decltype(alloc)>(num_times, a, b, c, sum);

  // Display timing results
  if (output_as_csv)
  {
    std::cout
      << "function" << csv_separator
      << "num_times" << csv_separator
      << "n_elements" << csv_separator
      << "sizeof" << csv_separator
      << ((mibibytes) ? "max_mibytes_per_sec" : "max_mbytes_per_sec") << csv_separator
      << "min_runtime" << csv_separator
      << "max_runtime" << csv_separator
      << "avg_runtime" << std::endl;
  }
  else
  {
    std::cout
      << std::left << std::setw(12) << "Function"
      << std::left << std::setw(12) << ((mibibytes) ? "MiBytes/sec" : "MBytes/sec")
      << std::left << std::setw(12) << "Min (sec)"
      << std::left << std::setw(12) << "Max"
      << std::left << std::setw(12) << "Average"
      << std::endl
      << std::fixed;
  }



  std::string labels[5] = {"Copy", "Mul", "Add", "Triad", "Dot"};
  size_t sizes[5] = {
    2 * sizeof(T) * N,
    2 * sizeof(T) * N,
    3 * sizeof(T) * N,
    3 * sizeof(T) * N,
    2 * sizeof(T) * N
  };

  for (int i = 0; i < 5; i++)
  {
    // Get min/max; ignore the first result
    auto minmax = std::minmax_element(timings[i].begin()+1, timings[i].end());

    // Calculate average; ignore the first result
    double average = std::accumulate(timings[i].begin()+1, timings[i].end(), 0.0) / (double)(num_times - 1);

    // Display results
    if (output_as_csv)
    {
      std::cout
        << labels[i] << csv_separator
        << num_times << csv_separator
        << N << csv_separator
        << sizeof(T) << csv_separator
        << ((mibibytes) ? pow(2.0, -20.0) : 1.0E-6) * sizes[i] / (*minmax.first) << csv_separator
        << *minmax.first << csv_separator
        << *minmax.second << csv_separator
        << average
        << std::endl;
    }
    else
    {
      std::cout
        << std::left << std::setw(12) << labels[i]
        << std::left << std::setw(12) << std::setprecision(3) <<
          ((mibibytes) ? pow(2.0, -20.0) : 1.0E-6) * sizes[i] / (*minmax.first)
        << std::left << std::setw(12) << std::setprecision(5) << *minmax.first
        << std::left << std::setw(12) << std::setprecision(5) << *minmax.second
        << std::left << std::setw(12) << std::setprecision(5) << average
        << std::endl;
    }
  }
}


template <typename T, template<typename Tp> class Allocator>
void check_solution(const unsigned int ntimes, std::vector<T, Allocator<T>>& a, std::vector<T, Allocator<T>>& b, std::vector<T, Allocator<T>>& c, T& sum)
{
  // Generate correct solution
  T goldA = 0.1;
  T goldB = 0.2;
  T goldC = 0.3;
  T goldSum = 0.0;

  const T scalar = 0.4;

  for (unsigned int i = 0; i < ntimes; i++)
  {
    goldC = goldA;
    goldB = scalar * goldC;
    goldC = goldA + goldB;
    goldA = goldB + scalar * goldC;
  }

  // Do the reduction
  goldSum = goldA * goldB * N;

  // Calculate the average error
  double errA = std::accumulate(a.begin(), a.end(), 0.0, [&](double sum, const T val){ return sum + fabs(val - goldA); });
  errA /= a.size();
  double errB = std::accumulate(b.begin(), b.end(), 0.0, [&](double sum, const T val){ return sum + fabs(val - goldB); });
  errB /= b.size();
  double errC = std::accumulate(c.begin(), c.end(), 0.0, [&](double sum, const T val){ return sum + fabs(val - goldC); });
  errC /= c.size();
  double errSum = fabs(sum - goldSum);

  double epsi = std::numeric_limits<T>::epsilon() * 100.0;

  if (errA > epsi)
    std::cerr
      << "Validation failed on a[]. Average error " << errA
      << std::endl;
  if (errB > epsi)
    std::cerr
      << "Validation failed on b[]. Average error " << errB
      << std::endl;
  if (errC > epsi)
    std::cerr
      << "Validation failed on c[]. Average error " << errC
      << std::endl;
  // Check sum to 8 decimal places
  if (errSum > 1.0E-8)
    std::cerr
      << "Validation failed on sum. Error " << errSum
      << std::endl << std::setprecision(15)
      << "Sum was " << sum << " but should be " << goldSum
      << std::endl;

}

int parseUInt(const char *str, unsigned int *output)
{
  char *next;
  *output = strtoul(str, &next, 10);
  return !strlen(next);
}

void parseArguments(int argc, char *argv[])
{
  for (int i = 1; i < argc; i++)
  {
    if (!std::string("--list").compare(argv[i]))
    {
      //listDevices();
      exit(EXIT_SUCCESS);
    }
    else if (!std::string("--arraysize").compare(argv[i]) ||
             !std::string("-s").compare(argv[i]))
    {
      if (++i >= argc || !parseUInt(argv[i], &N))
      {
        std::cerr << "Invalid array size." << std::endl;
        exit(EXIT_FAILURE);
      }
    }
    else if (!std::string("--numtimes").compare(argv[i]) ||
             !std::string("-n").compare(argv[i]))
    {
      if (++i >= argc || !parseUInt(argv[i], &num_times))
      {
        std::cerr << "Invalid number of times." << std::endl;
        exit(EXIT_FAILURE);
      }
      if (num_times < 2)
      {
        std::cerr << "Number of times must be 2 or more" << std::endl;
        exit(EXIT_FAILURE);
      }
    }
    else if (!std::string("--float").compare(argv[i]))
    {
      use_float = true;
    }
    else if (!std::string("--csv").compare(argv[i]))
    {
      output_as_csv = true;
    }
    else if (!std::string("--mibibytes").compare(argv[i]))
    {
      mibibytes = true;
    }
    else if (!std::string("--help").compare(argv[i]) ||
             !std::string("-h").compare(argv[i]))
    {
      std::cout << std::endl;
      std::cout << "Usage: " << argv[0] << " [OPTIONS]" << std::endl << std::endl;
      std::cout << "Options:" << std::endl;
      std::cout << "  -h  --help               Print the message" << std::endl;
      std::cout << "  -s  --arraysize  SIZE    Use SIZE elements in the array" << std::endl;
      std::cout << "  -n  --numtimes   NUM     Run the test NUM times (NUM >= 2)" << std::endl;
      std::cout << "      --float              Use floats (rather than doubles)" << std::endl;
      std::cout << "      --csv                Output as csv table" << std::endl;
      std::cout << "      --mibibytes          Use MiB=2^20 for bandwidth calculation (default MB=10^6)" << std::endl;
      std::cout << std::endl;
      exit(EXIT_SUCCESS);
    }
    else
    {
      std::cerr << "Unrecognized argument '" << argv[i] << "' (try '--help')"
                << std::endl;
      exit(EXIT_FAILURE);
    }
  }
}
