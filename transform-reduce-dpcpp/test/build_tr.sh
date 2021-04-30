
dpcpp -O3 -std=c++17 -I../include -c -o tr_test.o tr_test.cpp 
dpcpp -O3 -std=c++17 -I../include -c -o tr_multi_test.o tr_multi_test.cpp
dpcpp -O3 -std=c++17 -I../include -c -o device.o ../lib/device.cpp
dpcpp -O3 -std=c++17 -I../include -c -o reduce_helper.o ../lib/reduce_helper.cpp
dpcpp -o tr_test.exe tr_test.o device.o reduce_helper.o
dpcpp -o tr_multi_test.exe tr_multi_test.o device.o reduce_helper.o
rm *.o

