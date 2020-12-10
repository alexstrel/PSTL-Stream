CXX = g++
NVCXX = nvc++

NVARCH = cc70

CXXCOPT_AVX512  = -mavx512f -std=c++17 -fopenmp
CXXCOPT_AVX2  = -mavx2 -mfma -std=c++17 -fopenmp
LDFLAG =  -lm  -fopenmp -L/opt/intel/tbb-gnu9.3/lib -ltbb

NVCXXOPT = -O2 -std=c++17 -stdpar -gpu=$(NVARCH)

INC = -I.

all: pstl_stream.avx512 pstl_stream.avx2 pstl_stream.$(NVARCH)

pstl_stream.avx512: main_avx512.o
	$(CXX) -o  $@  $^  $(LDFLAG)

main_avx512.o : PSTLStream.cpp
	$(CXX) $(CXXCOPT_AVX512) -c -o  $@  $?  $(INC)

pstl_stream.avx2: main_avx2.o
	$(CXX) -o  $@  $^  $(LDFLAG)

main_avx2.o : PSTLStream.cpp
	$(CXX) $(CXXCOPT_AVX2) -c -o  $@  $?  $(INC)

pstl_stream.$(NVARCH): main_$(NVARCH).o
	$(NVCXX) -o  $@  $^

main_$(NVARCH).o : PSTLStream.cpp
	$(NVCXX) $(NVCXXOPT) -c -o  $@  $?  $(INC)

clean:
	rm *.o *.avx512 *.avx2 *.$(NVARCH)

.PHONY:	clean
