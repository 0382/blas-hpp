#pragma once
#ifndef BLAS_HPP
#define BLAS_HPP

#include <cblas.h>
#include <complex>

namespace blas
{

using f32 = float;
using f64 = double;
using z32 = std::complex<float>;
using z64 = std::complex<double>;

template <typename T>
inline void *cast_void(T *x)
{
    return static_cast<void *>(x);
}
template <typename T>
inline const void *cast_cvoid(const T *x)
{
    return static_cast<const void *>(x);
}

// ----- level 1 -----

inline void rotg(f32 *a, f32 *b, f32 *c, f32 *s) { cblas_srotg(a, b, c, s); }
inline void rotg(f64 *a, f64 *b, f64 *c, f64 *s) { cblas_drotg(a, b, c, s); }
inline void rotg(z32 *a, z32 *b, f32 *c, z32 *s) { cblas_crotg(cast_void(a), cast_void(b), c, cast_void(s)); }
inline void rotg(z64 *a, z64 *b, f64 *c, z64 *s) { cblas_zrotg(cast_void(a), cast_void(b), c, cast_void(s)); }

inline void rotmg(f32 *d1, f32 *d2, f32 *a, f32 b, f32 *p) { cblas_srotmg(d1, d2, a, b, p); }
inline void rotmg(f64 *d1, f64 *d2, f64 *a, f64 b, f64 *p) { cblas_drotmg(d1, d2, a, b, p); }

inline void rot(blasint n, f32 *x, blasint incx, f32 *y, blasint incy, f32 c, f32 s)
{
    cblas_srot(n, x, incx, y, incy, c, s);
}
inline void rot(blasint n, f64 *x, blasint incx, f64 *y, blasint incy, f64 c, f64 s)
{
    cblas_drot(n, x, incx, y, incy, c, s);
}
inline void rot(blasint n, z32 *x, blasint incx, z32 *y, blasint incy, f32 c, f32 s)
{
    cblas_csrot(n, cast_cvoid(x), incx, cast_void(y), incy, c, s);
}
inline void rot(blasint n, z64 *x, blasint incx, z64 *y, blasint incy, f64 c, f64 s)
{
    cblas_zdrot(n, cast_cvoid(x), incx, cast_void(y), incy, c, s);
}

inline void rotm(blasint n, f32 *x, blasint incx, f32 *y, blasint incy, const f32 *p)
{
    cblas_srotm(n, x, incx, y, incy, p);
}
inline void rotm(blasint n, f64 *x, blasint incx, f64 *y, blasint incy, const f64 *p)
{
    cblas_drotm(n, x, incx, y, incy, p);
}

inline void swap(blasint n, f32 *x, blasint incx, f32 *y, blasint incy) { cblas_sswap(n, x, incx, y, incy); }
inline void swap(blasint n, f64 *x, blasint incx, f64 *y, blasint incy) { cblas_dswap(n, x, incx, y, incy); }
inline void swap(blasint n, z32 *x, blasint incx, z32 *y, blasint incy)
{
    cblas_cswap(n, cast_void(x), incx, cast_void(y), incy);
}
inline void swap(blasint n, z64 *x, blasint incx, z64 *y, blasint incy)
{
    cblas_zswap(n, cast_void(x), incx, cast_void(y), incy);
}

inline void scal(blasint n, f32 alpha, f32 *x, blasint incx) { cblas_sscal(n, alpha, x, incx); }
inline void scal(blasint n, f64 alpha, f64 *x, blasint incx) { cblas_dscal(n, alpha, x, incx); }
inline void scal(blasint n, z32 alpha, z32 *x, blasint incx) { cblas_cscal(n, cast_cvoid(&alpha), cast_void(x), incx); }
inline void scal(blasint n, z64 alpha, z64 *x, blasint incx) { cblas_zscal(n, cast_cvoid(&alpha), cast_void(x), incx); }
inline void scal(blasint n, f32 alpha, z32 *x, blasint incx) { cblas_csscal(n, alpha, cast_void(x), incx); }
inline void scal(blasint n, f64 alpha, z64 *x, blasint incx) { cblas_zdscal(n, alpha, cast_void(x), incx); }

inline void copy(blasint n, const f32 *x, blasint incx, f32 *y, blasint incy) { cblas_scopy(n, x, incx, y, incy); }
inline void copy(blasint n, const f64 *x, blasint incx, f64 *y, blasint incy) { cblas_dcopy(n, x, incx, y, incy); }
inline void copy(blasint n, const z32 *x, blasint incx, z32 *y, blasint incy)
{
    cblas_ccopy(n, cast_cvoid(x), incx, cast_void(y), incy);
}
inline void copy(blasint n, const z64 *x, blasint incx, z64 *y, blasint incy)
{
    cblas_zcopy(n, cast_cvoid(x), incx, cast_void(y), incy);
}

inline void axpy(blasint n, f32 alpha, const f32 *x, blasint incx, f32 *y, blasint incy)
{
    cblas_saxpy(n, alpha, x, incx, y, incy);
}
inline void axpy(blasint n, f64 alpha, const f64 *x, blasint incx, f64 *y, blasint incy)
{
    cblas_daxpy(n, alpha, x, incx, y, incy);
}
inline void axpy(blasint n, z32 alpha, const z32 *x, blasint incx, z32 *y, blasint incy)
{
    cblas_caxpy(n, cast_cvoid(&alpha), cast_cvoid(x), incx, cast_void(y), incy);
}
inline void axpy(blasint n, z64 alpha, const z64 *x, blasint incx, z64 *y, blasint incy)
{
    cblas_zaxpy(n, cast_cvoid(&alpha), cast_cvoid(x), incx, cast_void(y), incy);
}

inline void axpby(blasint n, f32 alpha, const f32 *x, blasint incx, f32 beta, f32 *y, blasint incy)
{
    cblas_saxpby(n, alpha, x, incx, beta, y, incy);
}
inline void axpby(blasint n, f64 alpha, const f64 *x, blasint incx, f64 beta, f64 *y, blasint incy)
{
    cblas_daxpby(n, alpha, x, incx, beta, y, incy);
}
inline void axpby(blasint n, z32 alpha, const z32 *x, blasint incx, z32 beta, z32 *y, blasint incy)
{
    cblas_caxpby(n, cast_cvoid(&alpha), cast_cvoid(x), incx, cast_cvoid(&beta), cast_void(y), incy);
}
inline void axpby(blasint n, z64 alpha, const z64 *x, blasint incx, z64 beta, z64 *y, blasint incy)
{
    cblas_zaxpby(n, cast_cvoid(&alpha), cast_cvoid(x), incx, cast_cvoid(&beta), cast_void(y), incy);
}

inline f32 dot(blasint n, const f32 *x, blasint incx, const f32 *y, blasint incy)
{
    return cblas_sdot(n, x, incx, y, incy);
}
inline f64 dot(blasint n, const f64 *x, blasint incx, const f64 *y, blasint incy)
{
    return cblas_ddot(n, x, incx, y, incy);
}
inline z32 dotc(blasint n, const z32 *x, blasint incx, const z32 *y, blasint incy)
{
    z32 out;
    cblas_cdotc_sub(n, cast_cvoid(x), incx, cast_cvoid(y), incy, cast_void(&out));
    return out;
}
inline z64 dotc(blasint n, const z64 *x, blasint incx, const z64 *y, blasint incy)
{
    z64 out;
    cblas_zdotc_sub(n, cast_cvoid(x), incx, cast_cvoid(y), incy, cast_void(&out));
    return out;
}
inline z32 dotu(blasint n, const z32 *x, blasint incx, const z32 *y, blasint incy)
{
    z32 out;
    cblas_cdotu_sub(n, cast_cvoid(x), incx, cast_cvoid(y), incy, cast_void(&out));
    return out;
}
inline z64 dotu(blasint n, const z64 *x, blasint incx, const z64 *y, blasint incy)
{
    z64 out;
    cblas_zdotu_sub(n, cast_cvoid(x), incx, cast_cvoid(y), incy, cast_void(&out));
    return out;
}
// use dotc as default dot for complex
inline z32 dot(blasint n, const z32 *x, blasint incx, const z32 *y, blasint incy) { return dotc(n, x, incx, y, incy); }
inline z64 dot(blasint n, const z64 *x, blasint incx, const z64 *y, blasint incy) { return dotc(n, x, incx, y, incy); }

inline f64 dsdot(blasint n, const f32 *x, blasint incx, const f32 *y, blasint incy)
{
    return cblas_dsdot(n, x, incx, y, incy);
}
inline f32 sdsdot(blasint n, f32 alpha, const f32 *x, blasint incx, const f32 *y, blasint incy)
{
    return cblas_sdsdot(n, alpha, x, incx, y, incy);
}

inline f32 nrm2(blasint n, const f32 *x, blasint incx) { return cblas_snrm2(n, x, incx); }
inline f64 nrm2(blasint n, const f64 *x, blasint incx) { return cblas_dnrm2(n, x, incx); }
inline f32 nrm2(blasint n, const z32 *x, blasint incx) { return cblas_scnrm2(n, cast_cvoid(x), incx); }
inline f64 nrm2(blasint n, const z64 *x, blasint incx) { return cblas_dznrm2(n, cast_cvoid(x), incx); }

inline f32 asum(blasint n, const f32 *x, blasint incx) { return cblas_sasum(n, x, incx); }
inline f64 asum(blasint n, const f64 *x, blasint incx) { return cblas_dasum(n, x, incx); }
inline f32 asum(blasint n, const z32 *x, blasint incx) { return cblas_scasum(n, cast_cvoid(x), incx); }
inline f64 asum(blasint n, const z64 *x, blasint incx) { return cblas_dzasum(n, cast_cvoid(x), incx); }

inline size_t imax(blasint n, const f32 *x, blasint incx) { return cblas_ismax(n, x, incx); }
inline size_t imax(blasint n, const f64 *x, blasint incx) { return cblas_idmax(n, x, incx); }
inline size_t imax(blasint n, const z32 *x, blasint incx) { return cblas_icmax(n, cast_cvoid(x), incx); }
inline size_t imax(blasint n, const z64 *x, blasint incx) { return cblas_izmax(n, cast_cvoid(x), incx); }

inline size_t imin(blasint n, const f32 *x, blasint incx) { return cblas_ismin(n, x, incx); }
inline size_t imin(blasint n, const f64 *x, blasint incx) { return cblas_idmin(n, x, incx); }
inline size_t imin(blasint n, const z32 *x, blasint incx) { return cblas_icmin(n, cast_cvoid(x), incx); }
inline size_t imin(blasint n, const z64 *x, blasint incx) { return cblas_izmin(n, cast_cvoid(x), incx); }

inline size_t iamax(blasint n, const f32 *x, blasint incx) { return cblas_isamax(n, x, incx); }
inline size_t iamax(blasint n, const f64 *x, blasint incx) { return cblas_idamax(n, x, incx); }
inline size_t iamax(blasint n, const z32 *x, blasint incx) { return cblas_icamax(n, cast_cvoid(x), incx); }
inline size_t iamax(blasint n, const z64 *x, blasint incx) { return cblas_izamax(n, cast_cvoid(x), incx); }

// ----- level 2 -----

inline void gemv(const enum CBLAS_ORDER order, const enum CBLAS_TRANSPOSE trans, blasint m, blasint n, f32 alpha,
                 const f32 *a, blasint lda, const f32 *x, blasint incx, f32 beta, f32 *y, blasint incy)
{
    cblas_sgemv(order, trans, m, n, alpha, a, lda, x, incx, beta, y, incy);
}
inline void gemv(const enum CBLAS_ORDER order, const enum CBLAS_TRANSPOSE trans, blasint m, blasint n, f64 alpha,
                 const f64 *a, blasint lda, const f64 *x, blasint incx, f64 beta, f64 *y, blasint incy)
{
    cblas_dgemv(order, trans, m, n, alpha, a, lda, x, incx, beta, y, incy);
}
inline void gemv(const enum CBLAS_ORDER order, const enum CBLAS_TRANSPOSE trans, blasint m, blasint n, z32 alpha,
                 const z32 *a, blasint lda, const z32 *x, blasint incx, z32 beta, z32 *y, blasint incy)
{
    cblas_cgemv(order, trans, m, n, cast_cvoid(&alpha), cast_cvoid(a), lda, cast_cvoid(x), incx, cast_cvoid(&beta),
                cast_void(y), incy);
}
inline void gemv(const enum CBLAS_ORDER order, const enum CBLAS_TRANSPOSE trans, blasint m, blasint n, z64 alpha,
                 const z64 *a, blasint lda, const z64 *x, blasint incx, z64 beta, z64 *y, blasint incy)
{
    cblas_zgemv(order, trans, m, n, cast_cvoid(&alpha), cast_cvoid(a), lda, cast_cvoid(x), incx, cast_cvoid(&beta),
                cast_void(y), incy);
}

inline void gbmv(const enum CBLAS_ORDER order, const enum CBLAS_TRANSPOSE trans, blasint m, blasint n, blasint kl,
                 blasint ku, f32 alpha, const f32 *a, blasint lda, const f32 *x, blasint incx, f32 beta, f32 *y,
                 blasint incy)
{
    cblas_sgbmv(order, trans, m, n, kl, ku, alpha, a, lda, x, incx, beta, y, incy);
}
inline void gbmv(const enum CBLAS_ORDER order, const enum CBLAS_TRANSPOSE trans, blasint m, blasint n, blasint kl,
                 blasint ku, f64 alpha, const f64 *a, blasint lda, const f64 *x, blasint incx, f64 beta, f64 *y,
                 blasint incy)
{
    cblas_dgbmv(order, trans, m, n, kl, ku, alpha, a, lda, x, incx, beta, y, incy);
}
inline void gbmv(const enum CBLAS_ORDER order, const enum CBLAS_TRANSPOSE trans, blasint m, blasint n, blasint kl,
                 blasint ku, z32 alpha, const z32 *a, blasint lda, const z32 *x, blasint incx, z32 beta, z32 *y,
                 blasint incy)
{
    cblas_cgbmv(order, trans, m, n, kl, ku, cast_cvoid(&alpha), cast_cvoid(a), lda, cast_cvoid(x), incx,
                cast_cvoid(&beta), cast_void(y), incy);
}
inline void gbmv(const enum CBLAS_ORDER order, const enum CBLAS_TRANSPOSE trans, blasint m, blasint n, blasint kl,
                 blasint ku, z64 alpha, const z64 *a, blasint lda, const z64 *x, blasint incx, z64 beta, z64 *y,
                 blasint incy)
{
    cblas_zgbmv(order, trans, m, n, kl, ku, cast_cvoid(&alpha), cast_cvoid(a), lda, cast_cvoid(x), incx,
                cast_cvoid(&beta), cast_void(y), incy);
}

inline void symv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO uplo, blasint n, f32 alpha, const f32 *a,
                 blasint lda, const f32 *x, blasint incx, f32 beta, f32 *y, blasint incy)
{
    cblas_ssymv(order, uplo, n, alpha, a, lda, x, incx, beta, y, incy);
}
inline void symv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO uplo, blasint n, f64 alpha, const f64 *a,
                 blasint lda, const f64 *x, blasint incx, f64 beta, f64 *y, blasint incy)
{
    cblas_dsymv(order, uplo, n, alpha, a, lda, x, incx, beta, y, incy);
}

inline void sbmv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO uplo, blasint n, blasint k, f32 alpha,
                 const f32 *a, blasint lda, const f32 *x, blasint incx, f32 beta, f32 *y, blasint incy)
{
    cblas_ssbmv(order, uplo, n, k, alpha, a, lda, x, incx, beta, y, incy);
}
inline void sbmv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO uplo, blasint n, blasint k, f64 alpha,
                 const f64 *a, blasint lda, const f64 *x, blasint incx, f64 beta, f64 *y, blasint incy)
{
    cblas_dsbmv(order, uplo, n, k, alpha, a, lda, x, incx, beta, y, incy);
}

inline void spmv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO uplo, blasint n, f32 alpha, const f32 *ap,
                 const f32 *x, blasint incx, f32 beta, f32 *y, blasint incy)
{
    cblas_sspmv(order, uplo, n, alpha, ap, x, incx, beta, y, incy);
}
inline void spmv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO uplo, blasint n, f64 alpha, const f64 *ap,
                 const f64 *x, blasint incx, f64 beta, f64 *y, blasint incy)
{
    cblas_dspmv(order, uplo, n, alpha, ap, x, incx, beta, y, incy);
}

// for generic programing, also define hemv, hbmv, hpmv for `float` and `double`

inline void hemv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO uplo, blasint n, f32 alpha, const f32 *a,
                 blasint lda, const f32 *x, blasint incx, f32 beta, f32 *y, blasint incy)
{
    cblas_ssymv(order, uplo, n, alpha, a, lda, x, incx, beta, y, incy);
}
inline void hemv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO uplo, blasint n, f64 alpha, const f64 *a,
                 blasint lda, const f64 *x, blasint incx, f64 beta, f64 *y, blasint incy)
{
    cblas_dsymv(order, uplo, n, alpha, a, lda, x, incx, beta, y, incy);
}
inline void hemv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO uplo, blasint n, z32 alpha, const z32 *a,
                 blasint lda, const z32 *x, blasint incx, z32 beta, z32 *y, blasint incy)
{
    cblas_chemv(order, uplo, n, cast_cvoid(&alpha), cast_cvoid(a), lda, cast_cvoid(x), incx, cast_cvoid(&beta),
                cast_void(y), incy);
}
inline void hemv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO uplo, blasint n, z64 alpha, const z64 *a,
                 blasint lda, const z64 *x, blasint incx, z64 beta, z64 *y, blasint incy)
{
    cblas_zhemv(order, uplo, n, cast_cvoid(&alpha), cast_cvoid(a), lda, cast_cvoid(x), incx, cast_cvoid(&beta),
                cast_void(y), incy);
}

inline void hbmv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO uplo, blasint n, blasint k, f32 alpha,
                 const f32 *a, blasint lda, const f32 *x, blasint incx, f32 beta, f32 *y, blasint incy)
{
    cblas_ssbmv(order, uplo, n, k, alpha, a, lda, x, incx, beta, y, incy);
}
inline void hbmv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO uplo, blasint n, blasint k, f64 alpha,
                 const f64 *a, blasint lda, const f64 *x, blasint incx, f64 beta, f64 *y, blasint incy)
{
    cblas_dsbmv(order, uplo, n, k, alpha, a, lda, x, incx, beta, y, incy);
}
inline void hbmv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO uplo, blasint n, blasint k, z32 alpha,
                 const z32 *a, blasint lda, const z32 *x, blasint incx, z32 beta, z32 *y, blasint incy)
{
    cblas_chbmv(order, uplo, n, k, cast_cvoid(&alpha), cast_cvoid(a), lda, cast_cvoid(x), incx, cast_cvoid(&beta),
                cast_void(y), incy);
}
inline void hbmv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO uplo, blasint n, blasint k, z64 alpha,
                 const z64 *a, blasint lda, const z64 *x, blasint incx, z64 beta, z64 *y, blasint incy)
{
    cblas_zhbmv(order, uplo, n, k, cast_cvoid(&alpha), cast_cvoid(a), lda, cast_cvoid(x), incx, cast_cvoid(&beta),
                cast_void(y), incy);
}

inline void hpmv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO uplo, blasint n, f32 alpha, const f32 *ap,
                 const f32 *x, blasint incx, f32 beta, f32 *y, blasint incy)
{
    cblas_sspmv(order, uplo, n, alpha, ap, x, incx, beta, y, incy);
}
inline void hpmv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO uplo, blasint n, f64 alpha, const f64 *ap,
                 const f64 *x, blasint incx, f64 beta, f64 *y, blasint incy)
{
    cblas_dspmv(order, uplo, n, alpha, ap, x, incx, beta, y, incy);
}
inline void hpmv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO uplo, blasint n, z32 alpha, const z32 *ap,
                 const z32 *x, blasint incx, z32 beta, z32 *y, blasint incy)
{
    cblas_chpmv(order, uplo, n, cast_cvoid(&alpha), cast_cvoid(ap), cast_cvoid(x), incx, cast_cvoid(&beta),
                cast_void(y), incy);
}
inline void hpmv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO uplo, blasint n, z64 alpha, const z64 *ap,
                 const z64 *x, blasint incx, z64 beta, z64 *y, blasint incy)
{
    cblas_zhpmv(order, uplo, n, cast_cvoid(&alpha), cast_cvoid(ap), cast_cvoid(x), incx, cast_cvoid(&beta),
                cast_void(y), incy);
}

inline void trmv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO uplo, const enum CBLAS_TRANSPOSE trans,
                 const enum CBLAS_DIAG diag, blasint n, const f32 *a, blasint lda, f32 *x, blasint incx)
{
    cblas_strmv(order, uplo, trans, diag, n, a, lda, x, incx);
}
inline void trmv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO uplo, const enum CBLAS_TRANSPOSE trans,
                 const enum CBLAS_DIAG diag, blasint n, const f64 *a, blasint lda, f64 *x, blasint incx)
{
    cblas_dtrmv(order, uplo, trans, diag, n, a, lda, x, incx);
}
inline void trmv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO uplo, const enum CBLAS_TRANSPOSE trans,
                 const enum CBLAS_DIAG diag, blasint n, const z32 *a, blasint lda, z32 *x, blasint incx)
{
    cblas_ctrmv(order, uplo, trans, diag, n, cast_cvoid(a), lda, cast_void(x), incx);
}
inline void trmv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO uplo, const enum CBLAS_TRANSPOSE trans,
                 const enum CBLAS_DIAG diag, blasint n, const z64 *a, blasint lda, z64 *x, blasint incx)
{
    cblas_ztrmv(order, uplo, trans, diag, n, cast_cvoid(a), lda, cast_void(x), incx);
}

inline void tbmv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO uplo, const enum CBLAS_TRANSPOSE trans,
                 const enum CBLAS_DIAG diag, blasint n, blasint k, const f32 *a, blasint lda, f32 *x, blasint incx)
{
    cblas_stbmv(order, uplo, trans, diag, n, k, a, lda, x, incx);
}
inline void tbmv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO uplo, const enum CBLAS_TRANSPOSE trans,
                 const enum CBLAS_DIAG diag, blasint n, blasint k, const f64 *a, blasint lda, f64 *x, blasint incx)
{
    cblas_dtbmv(order, uplo, trans, diag, n, k, a, lda, x, incx);
}
inline void tbmv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO uplo, const enum CBLAS_TRANSPOSE trans,
                 const enum CBLAS_DIAG diag, blasint n, blasint k, const z32 *a, blasint lda, z32 *x, blasint incx)
{
    cblas_ctbmv(order, uplo, trans, diag, n, k, cast_cvoid(a), lda, cast_void(x), incx);
}
inline void tbmv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO uplo, const enum CBLAS_TRANSPOSE trans,
                 const enum CBLAS_DIAG diag, blasint n, blasint k, const z64 *a, blasint lda, z64 *x, blasint incx)
{
    cblas_ztbmv(order, uplo, trans, diag, n, k, cast_cvoid(a), lda, cast_void(x), incx);
}

inline void tpmv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO uplo, const enum CBLAS_TRANSPOSE trans,
                 const enum CBLAS_DIAG diag, blasint n, const f32 *ap, f32 *x, blasint incx)
{
    cblas_stpmv(order, uplo, trans, diag, n, ap, x, incx);
}
inline void tpmv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO uplo, const enum CBLAS_TRANSPOSE trans,
                 const enum CBLAS_DIAG diag, blasint n, const f64 *ap, f64 *x, blasint incx)
{
    cblas_dtpmv(order, uplo, trans, diag, n, ap, x, incx);
}
inline void tpmv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO uplo, const enum CBLAS_TRANSPOSE trans,
                 const enum CBLAS_DIAG diag, blasint n, const z32 *ap, z32 *x, blasint incx)
{
    cblas_ctpmv(order, uplo, trans, diag, n, cast_cvoid(ap), cast_void(x), incx);
}
inline void tpmv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO uplo, const enum CBLAS_TRANSPOSE trans,
                 const enum CBLAS_DIAG diag, blasint n, const z64 *ap, z64 *x, blasint incx)
{
    cblas_ztpmv(order, uplo, trans, diag, n, cast_cvoid(ap), cast_void(x), incx);
}

inline void trsv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO uplo, const enum CBLAS_TRANSPOSE trans,
                 const enum CBLAS_DIAG diag, blasint n, const f32 *a, blasint lda, f32 *x, blasint incx)
{
    cblas_strsv(order, uplo, trans, diag, n, a, lda, x, incx);
}
inline void trsv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO uplo, const enum CBLAS_TRANSPOSE trans,
                 const enum CBLAS_DIAG diag, blasint n, const f64 *a, blasint lda, f64 *x, blasint incx)
{
    cblas_dtrsv(order, uplo, trans, diag, n, a, lda, x, incx);
}
inline void trsv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO uplo, const enum CBLAS_TRANSPOSE trans,
                 const enum CBLAS_DIAG diag, blasint n, const z32 *a, blasint lda, z32 *x, blasint incx)
{
    cblas_ctrsv(order, uplo, trans, diag, n, cast_cvoid(a), lda, cast_void(x), incx);
}
inline void trsv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO uplo, const enum CBLAS_TRANSPOSE trans,
                 const enum CBLAS_DIAG diag, blasint n, const z64 *a, blasint lda, z64 *x, blasint incx)
{
    cblas_ztrsv(order, uplo, trans, diag, n, cast_cvoid(a), lda, cast_void(x), incx);
}

inline void tbsv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO uplo, const enum CBLAS_TRANSPOSE trans,
                 const enum CBLAS_DIAG diag, blasint n, blasint k, const f32 *a, blasint lda, f32 *x, blasint incx)
{
    cblas_stbsv(order, uplo, trans, diag, n, k, a, lda, x, incx);
}
inline void tbsv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO uplo, const enum CBLAS_TRANSPOSE trans,
                 const enum CBLAS_DIAG diag, blasint n, blasint k, const f64 *a, blasint lda, f64 *x, blasint incx)
{
    cblas_dtbsv(order, uplo, trans, diag, n, k, a, lda, x, incx);
}
inline void tbsv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO uplo, const enum CBLAS_TRANSPOSE trans,
                 const enum CBLAS_DIAG diag, blasint n, blasint k, const z32 *a, blasint lda, z32 *x, blasint incx)
{
    cblas_ctbsv(order, uplo, trans, diag, n, k, cast_cvoid(a), lda, cast_void(x), incx);
}
inline void tbsv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO uplo, const enum CBLAS_TRANSPOSE trans,
                 const enum CBLAS_DIAG diag, blasint n, blasint k, const z64 *a, blasint lda, z64 *x, blasint incx)
{
    cblas_ztbsv(order, uplo, trans, diag, n, k, cast_cvoid(a), lda, cast_void(x), incx);
}

inline void tpsv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO uplo, const enum CBLAS_TRANSPOSE trans,
                 const enum CBLAS_DIAG diag, blasint n, const f32 *ap, f32 *x, blasint incx)
{
    cblas_stpsv(order, uplo, trans, diag, n, ap, x, incx);
}
inline void tpsv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO uplo, const enum CBLAS_TRANSPOSE trans,
                 const enum CBLAS_DIAG diag, blasint n, const f64 *ap, f64 *x, blasint incx)
{
    cblas_dtpsv(order, uplo, trans, diag, n, ap, x, incx);
}
inline void tpsv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO uplo, const enum CBLAS_TRANSPOSE trans,
                 const enum CBLAS_DIAG diag, blasint n, const z32 *ap, z32 *x, blasint incx)
{
    cblas_ctpsv(order, uplo, trans, diag, n, cast_cvoid(ap), cast_void(x), incx);
}
inline void tpsv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO uplo, const enum CBLAS_TRANSPOSE trans,
                 const enum CBLAS_DIAG diag, blasint n, const z64 *ap, z64 *x, blasint incx)
{
    cblas_ztpsv(order, uplo, trans, diag, n, cast_cvoid(ap), cast_void(x), incx);
}

inline void ger(const enum CBLAS_ORDER order, blasint m, blasint n, f32 alpha, const f32 *x, blasint incx, const f32 *y,
                blasint incy, f32 *a, blasint lda)
{
    cblas_sger(order, m, n, alpha, x, incx, y, incy, a, lda);
}
inline void ger(const enum CBLAS_ORDER order, blasint m, blasint n, f64 alpha, const f64 *x, blasint incx, const f64 *y,
                blasint incy, f64 *a, blasint lda)
{
    cblas_dger(order, m, n, alpha, x, incx, y, incy, a, lda);
}

inline void geru(const enum CBLAS_ORDER order, blasint m, blasint n, z32 alpha, const z32 *x, blasint incx,
                 const z32 *y, blasint incy, z32 *a, blasint lda)
{
    cblas_cgeru(order, m, n, cast_cvoid(&alpha), cast_cvoid(x), incx, cast_cvoid(y), incy, cast_void(a), lda);
}

inline void geru(const enum CBLAS_ORDER order, blasint m, blasint n, z64 alpha, const z64 *x, blasint incx,
                 const z64 *y, blasint incy, z64 *a, blasint lda)
{
    cblas_zgeru(order, m, n, cast_cvoid(&alpha), cast_cvoid(x), incx, cast_cvoid(y), incy, cast_void(a), lda);
}
inline void gerc(const enum CBLAS_ORDER order, blasint m, blasint n, z32 alpha, const z32 *x, blasint incx,
                 const z32 *y, blasint incy, z32 *a, blasint lda)
{
    cblas_cgerc(order, m, n, cast_cvoid(&alpha), cast_cvoid(x), incx, cast_cvoid(y), incy, cast_void(a), lda);
}

inline void gerc(const enum CBLAS_ORDER order, blasint m, blasint n, z64 alpha, const z64 *x, blasint incx,
                 const z64 *y, blasint incy, z64 *a, blasint lda)
{
    cblas_zgerc(order, m, n, cast_cvoid(&alpha), cast_cvoid(x), incx, cast_cvoid(y), incy, cast_void(a), lda);
}
inline void gerc(const enum CBLAS_ORDER order, blasint m, blasint n, f32 alpha, const f32 *x, blasint incx,
                 const f32 *y, blasint incy, f32 *a, blasint lda)
{
    cblas_sger(order, m, n, alpha, x, incx, y, incy, a, lda);
}
inline void gerc(const enum CBLAS_ORDER order, blasint m, blasint n, f64 alpha, const f64 *x, blasint incx,
                 const f64 *y, blasint incy, f64 *a, blasint lda)
{
    cblas_dger(order, m, n, alpha, x, incx, y, incy, a, lda);
}

inline void syr(const enum CBLAS_ORDER order, const enum CBLAS_UPLO uplo, blasint n, f32 alpha, const f32 *x,
                blasint incx, f32 *a, blasint lda)
{
    cblas_ssyr(order, uplo, n, alpha, x, incx, a, lda);
}
inline void syr(const enum CBLAS_ORDER order, const enum CBLAS_UPLO uplo, blasint n, f64 alpha, const f64 *x,
                blasint incx, f64 *a, blasint lda)
{
    cblas_dsyr(order, uplo, n, alpha, x, incx, a, lda);
}
inline void her(const enum CBLAS_ORDER order, const enum CBLAS_UPLO uplo, blasint n, f32 alpha, const f32 *x,
                blasint incx, f32 *a, blasint lda)
{
    cblas_ssyr(order, uplo, n, alpha, x, incx, a, lda);
}
inline void her(const enum CBLAS_ORDER order, const enum CBLAS_UPLO uplo, blasint n, f64 alpha, const f64 *x,
                blasint incx, f64 *a, blasint lda)
{
    cblas_dsyr(order, uplo, n, alpha, x, incx, a, lda);
}
inline void her(const enum CBLAS_ORDER order, const enum CBLAS_UPLO uplo, blasint n, f32 alpha, const z32 *x,
                blasint incx, z32 *a, blasint lda)
{
    cblas_cher(order, uplo, n, alpha, cast_cvoid(x), incx, cast_void(a), lda);
}
inline void her(const enum CBLAS_ORDER order, const enum CBLAS_UPLO uplo, blasint n, f64 alpha, const z64 *x,
                blasint incx, z64 *a, blasint lda)
{
    cblas_zher(order, uplo, n, alpha, cast_cvoid(x), incx, cast_void(a), lda);
}

inline void spr(const enum CBLAS_ORDER order, const enum CBLAS_UPLO uplo, blasint n, f32 alpha, const f32 *x,
                blasint incx, f32 *ap)
{
    cblas_sspr(order, uplo, n, alpha, x, incx, ap);
}
inline void spr(const enum CBLAS_ORDER order, const enum CBLAS_UPLO uplo, blasint n, f64 alpha, const f64 *x,
                blasint incx, f64 *ap)
{
    cblas_dspr(order, uplo, n, alpha, x, incx, ap);
}
inline void hpr(const enum CBLAS_ORDER order, const enum CBLAS_UPLO uplo, blasint n, f32 alpha, const f32 *x,
                blasint incx, f32 *ap)
{
    cblas_sspr(order, uplo, n, alpha, x, incx, ap);
}
inline void hpr(const enum CBLAS_ORDER order, const enum CBLAS_UPLO uplo, blasint n, f64 alpha, const f64 *x,
                blasint incx, f64 *ap)
{
    cblas_dspr(order, uplo, n, alpha, x, incx, ap);
}
inline void hpr(const enum CBLAS_ORDER order, const enum CBLAS_UPLO uplo, blasint n, f32 alpha, const z32 *x,
                blasint incx, z32 *ap)
{
    cblas_chpr(order, uplo, n, alpha, cast_cvoid(x), incx, cast_void(ap));
}
inline void hpr(const enum CBLAS_ORDER order, const enum CBLAS_UPLO uplo, blasint n, f64 alpha, const z64 *x,
                blasint incx, z64 *ap)
{
    cblas_zhpr(order, uplo, n, alpha, cast_cvoid(x), incx, cast_void(ap));
}

inline void syr2(const enum CBLAS_ORDER order, const enum CBLAS_UPLO uplo, blasint n, f32 alpha, const f32 *x,
                 blasint incx, const f32 *y, blasint incy, f32 *a, blasint lda)
{
    cblas_ssyr2(order, uplo, n, alpha, x, incx, y, incy, a, lda);
}
inline void syr2(const enum CBLAS_ORDER order, const enum CBLAS_UPLO uplo, blasint n, f64 alpha, const f64 *x,
                 blasint incx, const f64 *y, blasint incy, f64 *a, blasint lda)
{
    cblas_dsyr2(order, uplo, n, alpha, x, incx, y, incy, a, lda);
}
inline void her2(const enum CBLAS_ORDER order, const enum CBLAS_UPLO uplo, blasint n, f32 alpha, const f32 *x,
                 blasint incx, const f32 *y, blasint incy, f32 *a, blasint lda)
{
    cblas_ssyr2(order, uplo, n, alpha, x, incx, y, incy, a, lda);
}
inline void her2(const enum CBLAS_ORDER order, const enum CBLAS_UPLO uplo, blasint n, f64 alpha, const f64 *x,
                 blasint incx, const f64 *y, blasint incy, f64 *a, blasint lda)
{
    cblas_dsyr2(order, uplo, n, alpha, x, incx, y, incy, a, lda);
}
inline void her2(const enum CBLAS_ORDER order, const enum CBLAS_UPLO uplo, blasint n, z32 alpha, const z32 *x,
                 blasint incx, const z32 *y, blasint incy, z32 *a, blasint lda)
{
    cblas_cher2(order, uplo, n, cast_cvoid(&alpha), cast_cvoid(x), incx, cast_cvoid(y), incy, cast_void(a), lda);
}
inline void her2(const enum CBLAS_ORDER order, const enum CBLAS_UPLO uplo, blasint n, z64 alpha, const z64 *x,
                 blasint incx, const z64 *y, blasint incy, z64 *a, blasint lda)
{
    cblas_zher2(order, uplo, n, cast_cvoid(&alpha), cast_cvoid(x), incx, cast_cvoid(y), incy, cast_void(a), lda);
}

inline void spr2(const enum CBLAS_ORDER order, const enum CBLAS_UPLO uplo, blasint n, f32 alpha, const f32 *x,
                 blasint incx, const f32 *y, blasint incy, f32 *ap)
{
    cblas_sspr2(order, uplo, n, alpha, x, incx, y, incy, ap);
}
inline void spr2(const enum CBLAS_ORDER order, const enum CBLAS_UPLO uplo, blasint n, f64 alpha, const f64 *x,
                 blasint incx, const f64 *y, blasint incy, f64 *ap)
{
    cblas_dspr2(order, uplo, n, alpha, x, incx, y, incy, ap);
}
inline void hpr2(const enum CBLAS_ORDER order, const enum CBLAS_UPLO uplo, blasint n, f32 alpha, const f32 *x,
                 blasint incx, const f32 *y, blasint incy, f32 *ap)
{
    cblas_sspr2(order, uplo, n, alpha, x, incx, y, incy, ap);
}
inline void hpr2(const enum CBLAS_ORDER order, const enum CBLAS_UPLO uplo, blasint n, f64 alpha, const f64 *x,
                 blasint incx, const f64 *y, blasint incy, f64 *ap)
{
    cblas_dspr2(order, uplo, n, alpha, x, incx, y, incy, ap);
}
inline void hpr2(const enum CBLAS_ORDER order, const enum CBLAS_UPLO uplo, blasint n, z32 alpha, const z32 *x,
                 blasint incx, const z32 *y, blasint incy, z32 *ap)
{
    cblas_chpr2(order, uplo, n, cast_cvoid(&alpha), cast_cvoid(x), incx, cast_cvoid(y), incy, cast_void(ap));
}
inline void hpr2(const enum CBLAS_ORDER order, const enum CBLAS_UPLO uplo, blasint n, z64 alpha, const z64 *x,
                 blasint incx, const z64 *y, blasint incy, z64 *ap)
{
    cblas_zhpr2(order, uplo, n, cast_cvoid(&alpha), cast_cvoid(x), incx, cast_cvoid(y), incy, cast_void(ap));
}

// ----- level 3 -----
inline void gemm(const enum CBLAS_ORDER order, const enum CBLAS_TRANSPOSE transA, const enum CBLAS_TRANSPOSE transB,
                 blasint m, blasint n, blasint k, f32 alpha, const f32 *a, blasint lda, const f32 *b, blasint ldb,
                 f32 beta, f32 *c, blasint ldc)
{
    cblas_sgemm(order, transA, transB, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}
inline void gemm(const enum CBLAS_ORDER order, const enum CBLAS_TRANSPOSE transA, const enum CBLAS_TRANSPOSE transB,
                 blasint m, blasint n, blasint k, f64 alpha, const f64 *a, blasint lda, const f64 *b, blasint ldb,
                 f64 beta, f64 *c, blasint ldc)
{
    cblas_dgemm(order, transA, transB, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}
inline void gemm(const enum CBLAS_ORDER order, const enum CBLAS_TRANSPOSE transA, const enum CBLAS_TRANSPOSE transB,
                 blasint m, blasint n, blasint k, z32 alpha, const z32 *a, blasint lda, const z32 *b, blasint ldb,
                 z32 beta, z32 *c, blasint ldc)
{
    cblas_cgemm(order, transA, transB, m, n, k, cast_cvoid(&alpha), cast_cvoid(a), lda, cast_cvoid(b), ldb,
                cast_cvoid(&beta), cast_void(c), ldc);
}
inline void gemm(const enum CBLAS_ORDER order, const enum CBLAS_TRANSPOSE transA, const enum CBLAS_TRANSPOSE transB,
                 blasint m, blasint n, blasint k, z64 alpha, const z64 *a, blasint lda, const z64 *b, blasint ldb,
                 z64 beta, z64 *c, blasint ldc)
{
    cblas_zgemm(order, transA, transB, m, n, k, cast_cvoid(&alpha), cast_cvoid(a), lda, cast_cvoid(b), ldb,
                cast_cvoid(&beta), cast_void(c), ldc);
}

inline void symm(const enum CBLAS_ORDER order, const enum CBLAS_SIDE side, const enum CBLAS_UPLO uplo, blasint m,
                 blasint n, f32 alpha, const f32 *a, blasint lda, const f32 *b, blasint ldb, f32 beta, f32 *c,
                 blasint ldc)
{
    cblas_ssymm(order, side, uplo, m, n, alpha, a, lda, b, ldb, beta, c, ldc);
}
inline void symm(const enum CBLAS_ORDER order, const enum CBLAS_SIDE side, const enum CBLAS_UPLO uplo, blasint m,
                 blasint n, f64 alpha, const f64 *a, blasint lda, const f64 *b, blasint ldb, f64 beta, f64 *c,
                 blasint ldc)
{
    cblas_dsymm(order, side, uplo, m, n, alpha, a, lda, b, ldb, beta, c, ldc);
}
inline void symm(const enum CBLAS_ORDER order, const enum CBLAS_SIDE side, const enum CBLAS_UPLO uplo, blasint m,
                 blasint n, z32 alpha, const z32 *a, blasint lda, const z32 *b, blasint ldb, z32 beta, z32 *c,
                 blasint ldc)
{
    cblas_csymm(order, side, uplo, m, n, cast_cvoid(&alpha), cast_cvoid(a), lda, cast_cvoid(b), ldb, cast_cvoid(&beta),
                cast_void(c), ldc);
}
inline void symm(const enum CBLAS_ORDER order, const enum CBLAS_SIDE side, const enum CBLAS_UPLO uplo, blasint m,
                 blasint n, z64 alpha, const z64 *a, blasint lda, const z64 *b, blasint ldb, z64 beta, z64 *c,
                 blasint ldc)
{
    cblas_zsymm(order, side, uplo, m, n, cast_cvoid(&alpha), cast_cvoid(a), lda, cast_cvoid(b), ldb, cast_cvoid(&beta),
                cast_void(c), ldc);
}

inline void hemm(const enum CBLAS_ORDER order, const enum CBLAS_SIDE side, const enum CBLAS_UPLO uplo, blasint m,
                 blasint n, f32 alpha, const f32 *a, blasint lda, const f32 *b, blasint ldb, f32 beta, f32 *c,
                 blasint ldc)
{
    cblas_ssymm(order, side, uplo, m, n, alpha, a, lda, b, ldb, beta, c, ldc);
}
inline void hemm(const enum CBLAS_ORDER order, const enum CBLAS_SIDE side, const enum CBLAS_UPLO uplo, blasint m,
                 blasint n, f64 alpha, const f64 *a, blasint lda, const f64 *b, blasint ldb, f64 beta, f64 *c,
                 blasint ldc)
{
    cblas_dsymm(order, side, uplo, m, n, alpha, a, lda, b, ldb, beta, c, ldc);
}
inline void hemm(const enum CBLAS_ORDER order, const enum CBLAS_SIDE side, const enum CBLAS_UPLO uplo, blasint m,
                 blasint n, z32 alpha, const z32 *a, blasint lda, const z32 *b, blasint ldb, z32 beta, z32 *c,
                 blasint ldc)
{
    cblas_chemm(order, side, uplo, m, n, cast_cvoid(&alpha), cast_cvoid(a), lda, cast_cvoid(b), ldb, cast_cvoid(&beta),
                cast_void(c), ldc);
}
inline void hemm(const enum CBLAS_ORDER order, const enum CBLAS_SIDE side, const enum CBLAS_UPLO uplo, blasint m,
                 blasint n, z64 alpha, const z64 *a, blasint lda, const z64 *b, blasint ldb, z64 beta, z64 *c,
                 blasint ldc)
{
    cblas_zhemm(order, side, uplo, m, n, cast_cvoid(&alpha), cast_cvoid(a), lda, cast_cvoid(b), ldb, cast_cvoid(&beta),
                cast_void(c), ldc);
}

inline void syrk(const enum CBLAS_ORDER order, const enum CBLAS_UPLO uplo, const enum CBLAS_TRANSPOSE trans, blasint n,
                 blasint k, f32 alpha, const f32 *a, blasint lda, f32 beta, f32 *c, blasint ldc)

{
    cblas_ssyrk(order, uplo, trans, n, k, alpha, a, lda, beta, c, ldc);
}
inline void syrk(const enum CBLAS_ORDER order, const enum CBLAS_UPLO uplo, const enum CBLAS_TRANSPOSE trans, blasint n,
                 blasint k, f64 alpha, const f64 *a, blasint lda, f64 beta, f64 *c, blasint ldc)

{
    cblas_dsyrk(order, uplo, trans, n, k, alpha, a, lda, beta, c, ldc);
}
inline void syrk(const enum CBLAS_ORDER order, const enum CBLAS_UPLO uplo, const enum CBLAS_TRANSPOSE trans, blasint n,
                 blasint k, z32 alpha, const z32 *a, blasint lda, z32 beta, z32 *c, blasint ldc)

{
    cblas_csyrk(order, uplo, trans, n, k, cast_cvoid(&alpha), cast_cvoid(a), lda, cast_cvoid(&beta), cast_void(c), ldc);
}
inline void syrk(const enum CBLAS_ORDER order, const enum CBLAS_UPLO uplo, const enum CBLAS_TRANSPOSE trans, blasint n,
                 blasint k, z64 alpha, const z64 *a, blasint lda, z64 beta, z64 *c, blasint ldc)

{
    cblas_zsyrk(order, uplo, trans, n, k, cast_cvoid(&alpha), cast_cvoid(a), lda, cast_cvoid(&beta), cast_void(c), ldc);
}

inline void herk(const enum CBLAS_ORDER order, const enum CBLAS_UPLO uplo, const enum CBLAS_TRANSPOSE trans, blasint n,
                 blasint k, f32 alpha, const f32 *a, blasint lda, f32 beta, f32 *c, blasint ldc)

{
    cblas_ssyrk(order, uplo, trans, n, k, alpha, a, lda, beta, c, ldc);
}
inline void herk(const enum CBLAS_ORDER order, const enum CBLAS_UPLO uplo, const enum CBLAS_TRANSPOSE trans, blasint n,
                 blasint k, f64 alpha, const f64 *a, blasint lda, f64 beta, f64 *c, blasint ldc)

{
    cblas_dsyrk(order, uplo, trans, n, k, alpha, a, lda, beta, c, ldc);
}
inline void herk(const enum CBLAS_ORDER order, const enum CBLAS_UPLO uplo, const enum CBLAS_TRANSPOSE trans, blasint n,
                 blasint k, f32 alpha, const z32 *a, blasint lda, f32 beta, z32 *c, blasint ldc)

{
    cblas_cherk(order, uplo, trans, n, k, alpha, cast_cvoid(a), lda, beta, cast_void(c), ldc);
}
inline void herk(const enum CBLAS_ORDER order, const enum CBLAS_UPLO uplo, const enum CBLAS_TRANSPOSE trans, blasint n,
                 blasint k, f64 alpha, const z64 *a, blasint lda, f64 beta, z64 *c, blasint ldc)

{
    cblas_zherk(order, uplo, trans, n, k, alpha, cast_cvoid(a), lda, beta, cast_void(c), ldc);
}

inline void syr2k(const enum CBLAS_ORDER order, const enum CBLAS_UPLO uplo, const enum CBLAS_TRANSPOSE trans, blasint n,
                  blasint k, f32 alpha, const f32 *a, blasint lda, const f32 *b, blasint ldb, f32 beta, f32 *c,
                  blasint ldc)

{
    cblas_ssyr2k(order, uplo, trans, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}
inline void syr2k(const enum CBLAS_ORDER order, const enum CBLAS_UPLO uplo, const enum CBLAS_TRANSPOSE trans, blasint n,
                  blasint k, f64 alpha, const f64 *a, blasint lda, const f64 *b, blasint ldb, f64 beta, f64 *c,
                  blasint ldc)

{
    cblas_dsyr2k(order, uplo, trans, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}
inline void syr2k(const enum CBLAS_ORDER order, const enum CBLAS_UPLO uplo, const enum CBLAS_TRANSPOSE trans, blasint n,
                  blasint k, z32 alpha, const z32 *a, blasint lda, const z32 *b, blasint ldb, z32 beta, z32 *c,
                  blasint ldc)

{
    cblas_csyr2k(order, uplo, trans, n, k, cast_cvoid(&alpha), cast_cvoid(a), lda, cast_cvoid(b), ldb,
                 cast_cvoid(&beta), cast_void(c), ldc);
}
inline void syr2k(const enum CBLAS_ORDER order, const enum CBLAS_UPLO uplo, const enum CBLAS_TRANSPOSE trans, blasint n,
                  blasint k, z64 alpha, const z64 *a, blasint lda, const z64 *b, blasint ldb, z64 beta, z64 *c,
                  blasint ldc)

{
    cblas_zsyr2k(order, uplo, trans, n, k, cast_cvoid(&alpha), cast_cvoid(a), lda, cast_cvoid(b), ldb,
                 cast_cvoid(&beta), cast_void(c), ldc);
}

inline void her2k(const enum CBLAS_ORDER order, const enum CBLAS_UPLO uplo, const enum CBLAS_TRANSPOSE trans, blasint n,
                  blasint k, f32 alpha, const f32 *a, blasint lda, const f32 *b, blasint ldb, f32 beta, f32 *c,
                  blasint ldc)

{
    cblas_ssyr2k(order, uplo, trans, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}
inline void her2k(const enum CBLAS_ORDER order, const enum CBLAS_UPLO uplo, const enum CBLAS_TRANSPOSE trans, blasint n,
                  blasint k, f64 alpha, const f64 *a, blasint lda, const f64 *b, blasint ldb, f64 beta, f64 *c,
                  blasint ldc)

{
    cblas_dsyr2k(order, uplo, trans, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}
inline void her2k(const enum CBLAS_ORDER order, const enum CBLAS_UPLO uplo, const enum CBLAS_TRANSPOSE trans, blasint n,
                  blasint k, z32 alpha, const z32 *a, blasint lda, const z32 *b, blasint ldb, f32 beta, z32 *c,
                  blasint ldc)

{
    cblas_cher2k(order, uplo, trans, n, k, cast_cvoid(&alpha), cast_cvoid(a), lda, cast_cvoid(b), ldb, beta,
                 cast_void(c), ldc);
}
inline void her2k(const enum CBLAS_ORDER order, const enum CBLAS_UPLO uplo, const enum CBLAS_TRANSPOSE trans, blasint n,
                  blasint k, z64 alpha, const z64 *a, blasint lda, const z64 *b, blasint ldb, f64 beta, z64 *c,
                  blasint ldc)

{
    cblas_zher2k(order, uplo, trans, n, k, cast_cvoid(&alpha), cast_cvoid(a), lda, cast_cvoid(b), ldb, beta,
                 cast_void(c), ldc);
}

inline void trmm(const enum CBLAS_ORDER order, const enum CBLAS_SIDE side, const enum CBLAS_UPLO uplo,
                 const enum CBLAS_TRANSPOSE transA, const enum CBLAS_DIAG diag, blasint m, blasint n, f32 alpha,
                 const f32 *a, blasint lda, f32 *b, blasint ldb)
{
    cblas_strmm(order, side, uplo, transA, diag, m, n, alpha, a, lda, b, ldb);
}

inline void trmm(const enum CBLAS_ORDER order, const enum CBLAS_SIDE side, const enum CBLAS_UPLO uplo,
                 const enum CBLAS_TRANSPOSE transA, const enum CBLAS_DIAG diag, blasint m, blasint n, f64 alpha,
                 const f64 *a, blasint lda, f64 *b, blasint ldb)
{
    cblas_dtrmm(order, side, uplo, transA, diag, m, n, alpha, a, lda, b, ldb);
}
inline void trmm(const enum CBLAS_ORDER order, const enum CBLAS_SIDE side, const enum CBLAS_UPLO uplo,
                 const enum CBLAS_TRANSPOSE transA, const enum CBLAS_DIAG diag, blasint m, blasint n, z32 alpha,
                 const z32 *a, blasint lda, z32 *b, blasint ldb)
{
    cblas_ctrmm(order, side, uplo, transA, diag, m, n, cast_cvoid(&alpha), cast_cvoid(a), lda, cast_void(b), ldb);
}

inline void trmm(const enum CBLAS_ORDER order, const enum CBLAS_SIDE side, const enum CBLAS_UPLO uplo,
                 const enum CBLAS_TRANSPOSE transA, const enum CBLAS_DIAG diag, blasint m, blasint n, z64 alpha,
                 const z64 *a, blasint lda, z64 *b, blasint ldb)
{
    cblas_ztrmm(order, side, uplo, transA, diag, m, n, cast_cvoid(&alpha), cast_cvoid(a), lda, cast_void(b), ldb);
}

inline void trsm(const enum CBLAS_ORDER order, const enum CBLAS_SIDE side, const enum CBLAS_UPLO uplo,
                 const enum CBLAS_TRANSPOSE transA, const enum CBLAS_DIAG diag, blasint m, blasint n, f32 alpha,
                 const f32 *a, blasint lda, f32 *b, blasint ldb)
{
    cblas_strsm(order, side, uplo, transA, diag, m, n, alpha, a, lda, b, ldb);
}

inline void trsm(const enum CBLAS_ORDER order, const enum CBLAS_SIDE side, const enum CBLAS_UPLO uplo,
                 const enum CBLAS_TRANSPOSE transA, const enum CBLAS_DIAG diag, blasint m, blasint n, f64 alpha,
                 const f64 *a, blasint lda, f64 *b, blasint ldb)
{
    cblas_dtrsm(order, side, uplo, transA, diag, m, n, alpha, a, lda, b, ldb);
}
inline void trsm(const enum CBLAS_ORDER order, const enum CBLAS_SIDE side, const enum CBLAS_UPLO uplo,
                 const enum CBLAS_TRANSPOSE transA, const enum CBLAS_DIAG diag, blasint m, blasint n, z32 alpha,
                 const z32 *a, blasint lda, z32 *b, blasint ldb)
{
    cblas_ctrsm(order, side, uplo, transA, diag, m, n, cast_cvoid(&alpha), cast_cvoid(a), lda, cast_void(b), ldb);
}

inline void trsm(const enum CBLAS_ORDER order, const enum CBLAS_SIDE side, const enum CBLAS_UPLO uplo,
                 const enum CBLAS_TRANSPOSE transA, const enum CBLAS_DIAG diag, blasint m, blasint n, z64 alpha,
                 const z64 *a, blasint lda, z64 *b, blasint ldb)
{
    cblas_ztrsm(order, side, uplo, transA, diag, m, n, cast_cvoid(&alpha), cast_cvoid(a), lda, cast_void(b), ldb);
}

} // namespace blas

#endif // BLAS_HPP