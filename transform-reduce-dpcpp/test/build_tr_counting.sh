
dpcpp -O3 -std=c++17 -I../include -c -o tr_test_counting_iters.o tr_test_counting_iters.cpp 
dpcpp -O3 -std=c++17 -I../include -c -o device.o ../lib/device.cpp
dpcpp -O3 -std=c++17 -I../include -c -o reduce_helper.o ../lib/reduce_helper.cpp
dpcpp -o tr_test_counting_iters.exe tr_test_counting_iters.o device.o reduce_helper.o
rm *.o

