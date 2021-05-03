
clang++ -O3 -std=c++17 -I../include  -I/opt/intel/dpcpp-2021-04-DEV/include/sycl -c -o tr_test.o tr_test.cpp 
clang++ -O3 -std=c++17 -I../include  -I/opt/intel/dpcpp-2021-04-DEV/include/sycl -c -o tr_multi_test.o tr_multi_test.cpp
clang++ -O3 -std=c++17 -I../include  -I/opt/intel/dpcpp-2021-04-DEV/include/sycl -c -o device.o ../lib/device.cpp
clang++ -O3 -std=c++17 -I../include  -I/opt/intel/dpcpp-2021-04-DEV/include/sycl -c -o reduce_helper.o ../lib/reduce_helper.cpp
clang++ -o tr_test.exe tr_test.o device.o reduce_helper.o
clang++ -o tr_multi_test.exe tr_multi_test.o device.o reduce_helper.o
rm *.o

