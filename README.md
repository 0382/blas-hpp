# blas-hpp

A c++ wrapper of CBLAS, for generic programming.

A simple example is like
```c++
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
```

Every function like `blas::gemv` supports `float, double, std::complex<float>, std::complex<double>`, so that your can use them in a `template` function.

In addition, `blas::dot` for `std::complex<float>, std::complex<double>` is alias of `cdotc, zdotc`, and `blas::hemv` for `float, double` is alias of `ssymv, dsymv`. `blas::her, blas::her2, blas::hemm, blas::herk, blas::her2k` also works similarly. Thus you can write `template` simpler.
