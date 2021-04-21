
dpcpp -O3 -std=c++17 -c -o tr_test.o tr_test.cpp 
dpcpp -O3 -std=c++17 -c -o device.o device.cpp
dpcpp -O3 -std=c++17 -c -o reduce_helper.o reduce_helper.cpp
dpcpp -o tr_test.exe tr_test.o device.o reduce_helper.o


