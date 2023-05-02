#include "blas.hpp"
#include <vector>
#include <iostream>

int main(int argc, char const *argv[])
{
    std::vector<double> m(100, 1.0), v(10, 1.0), r(10, 0.0);
    blas::gemv(CblasColMajor, CblasNoTrans, 10, 10, 1.0, m.data(), 10, v.data(), 1, 0., r.data(), 1);
    for(int i = 0; i < 10; ++i)
    {
        std::cout << r[i] << ',';
    }
    return 0;
}
