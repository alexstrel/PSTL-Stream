export PSTL_USAGE_WARNINGS=1
export ONEDPL_USE_DPCPP_BACKEND=1

#dpcpp -std=c++17 -O2 blas_template.cpp -o blas_template.exe -tbb
dpcpp -std=c++17 -O2 blas_caxpyDot.cpp -o blas_caxpyDot.exe
