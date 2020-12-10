#dpcpp -std=c++17 -O2 blas_template.cpp -o blas_template.exe -tbb
dpcpp -std=c++17 -O2 blas_caxpyDot.cpp -ltbb -o blas_caxpyDot.exe
