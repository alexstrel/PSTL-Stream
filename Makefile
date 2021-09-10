CXX = g++
NVCXX = nvc++
DPCXX = dpcpp

NVARCH = cc70

CXXCOPT_AVX512  = -O2 -mavx512f -std=c++17 -fopenmp
CXXCOPT_AVX2  = -O2 -mavx2 -mfma -std=c++17 -fopenmp
LDFLAG =  -lm  -fopenmp -L/opt/intel/tbb-gnu9.3/lib -ltbb

NVCXXOPT     = -O2 -std=c++17 -stdpar=gpu -gpu=$(NVARCH) -gpu=managed -gpu=fma -gpu=fastmath -gpu=autocollapse -gpu=loadcache:L1 -gpu=unroll
NVCXXOPT_X86 = -O2 -std=c++17 -stdpar=multicore	

DPCXXOPT = -std=c++17 -O2 -DDPCPP_BACKEND

INC = -I.

all: pstl_stream.avx512 pstl_stream.avx2 pstl_stream.$(NVARCH) pstl_stream.nvcpp pstl_stream.dpcpp

pstl_stream.avx512: pstl_stream_avx512.o
	$(CXX) -o  $@  $^  $(LDFLAG)

pstl_stream_avx512.o : PSTLStream.cpp
	$(CXX) $(CXXCOPT_AVX512) -c -o  $@  $?  $(INC)

pstl_stream.avx2: pstl_stream_avx2.o
	$(CXX) -o  $@  $^  $(LDFLAG)

pstl_stream_avx2.o : PSTLStream.cpp
	$(CXX) $(CXXCOPT_AVX2) -c -o  $@  $?  $(INC)

pstl_stream.nvcpp: pstl_stream_nvcpp.o
	$(NVCXX) -o  $@  $^

pstl_stream_nvcpp.o : PSTLStream.cpp
	$(NVCXX) $(NVCXXOPT_X86) -c -o  $@  $?  $(INC)

pstl_stream.dpcpp: pstl_stream_dpcpp.o
	$(DPCXX) -o  $@  $^

pstl_stream_dpcpp.o : PSTLStream.cpp
	$(DPCXX) $(DPCXXOPT) -c -o  $@  $?  $(INC)

pstl_stream.$(NVARCH): pstl_stream_$(NVARCH).o
	$(NVCXX) -o  $@  $^

pstl_stream_$(NVARCH).o : PSTLStream.cpp
	$(NVCXX) $(NVCXXOPT) -c -o  $@  $?  $(INC)

clean:
	rm *.o *.avx512 *.avx2 *.$(NVARCH) *.dpcpp *.nvcpp

.PHONY:	clean
