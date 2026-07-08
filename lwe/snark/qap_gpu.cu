// GPU implementation of the QAP witness-map field arithmetic + NTT primitives.
//
// The lattice-zkSNARK prover's "Compute H polynomial" block (r1cs_to_qap.tcc)
// is a sequence of FFT/iFFT/cosetFFT/icosetFFT over the SNARK prime field
// (libsnark::Field<T, p>, an ordinary residue representation — NOT Montgomery).
// Because every operation is exact modular arithmetic, ANY correct radix-2
// NTT using the same root of unity omega produces results bit-identical to
// libfqfft's serial/parallel FFT. We exploit that here: the kernels below
// compute the same DFT, so the GPU witness map matches the CPU one exactly.
//
// These are generic primitives (NTT, scale, coset-multiply, pointwise combine)
// operating on raw residue arrays of type T (= the field's storage type).
// The host orchestration that sequences them into the full witness map lives in
// qap_gpu.hpp; this file only provides the device kernels + thin launchers,
// explicitly instantiated for the two field widths actually used:
//   - uint64_t           (<=32-bit primes, the native B*C* fields)
//   - unsigned __int128  (<=60-bit primes, the big-int-ring B60/stage fields)
//
// The CPU path is untouched; this is an additive, opt-in acceleration.

#include <cuda_runtime.h>
#include <stdint.h>

namespace libsnark {
namespace qap_gpu_detail {

// ---- device modular arithmetic (mirrors libsnark::Field exactly) ----
// add: a,b in [0,m) -> a+b < 2m, single conditional subtract.
template <typename T> __device__ __forceinline__ T addmod(T a, T b, T m) {
    T s = a + b;
    return (s >= m) ? (s - m) : s;
}
// sub: matches Field::operator-= (a>=b ? a-b : m-b+a).
template <typename T> __device__ __forceinline__ T submod(T a, T b, T m) {
    return (a >= b) ? (a - b) : (m - b + a);
}
// mul: (a*b) mod m. For uint64_t the largest prime is < 2^32 so the product
// fits in 64 bits without overflow; for unsigned __int128 the 60-bit operands
// give a < 2^120 product, well within 128 bits.
template <typename T> __device__ __forceinline__ T mulmod(T a, T b, T m) {
    return (a * b) % m;
}

constexpr int TPB = 256;

// out-of-place bit-reversal permutation: out[k] = in[bitreverse(k, logn)].
template <typename T>
__global__ void k_bitrev(const T *__restrict__ in, T *__restrict__ out,
                         uint32_t n, uint32_t logn) {
    uint32_t k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= n) return;
    uint32_t r = 0, x = k;
#pragma unroll
    for (uint32_t i = 0; i < logn; ++i) {
        r = (r << 1) | (x & 1u);
        x >>= 1;
    }
    out[k] = in[r];
}

// one Cooley-Tukey radix-2 stage (decimation-in-time), in place.
// stage s in [1, logn]: half = 2^{s-1}; twiddle for column j is tw[j << (logn-s)]
// where tw[i] = omega^i (i in [0, n/2)).
template <typename T>
__global__ void k_ntt_stage(T *__restrict__ a, const T *__restrict__ tw,
                            uint32_t n, uint32_t logn, uint32_t s, T m) {
    uint32_t t = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t half = 1u << (s - 1);
    if (t >= (n >> 1)) return;
    uint32_t j = t & (half - 1);
    uint32_t group = t >> (s - 1);
    uint32_t i1 = (group << s) + j;
    uint32_t i2 = i1 + half;
    T w = tw[(uint64_t)j << (logn - s)];
    T u = a[i1];
    T v = mulmod<T>(w, a[i2], m);
    a[i1] = addmod<T>(u, v, m);
    a[i2] = submod<T>(u, v, m);
}

template <typename T>
__global__ void k_scale(T *__restrict__ a, uint32_t n, T s, T m) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) a[i] = mulmod<T>(a[i], s, m);
}

// a[i] *= tab[i]  (used for multiply-by-coset, tab[i] = g^i)
template <typename T>
__global__ void k_mul_vec(T *__restrict__ a, const T *__restrict__ tab,
                          uint32_t n, T m) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) a[i] = mulmod<T>(a[i], tab[i], m);
}

// out[i] = ca*A[i] + cb*B[i]
template <typename T>
__global__ void k_lincomb2(T *__restrict__ out, const T *__restrict__ A,
                           const T *__restrict__ B, T ca, T cb, uint32_t n, T m) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        out[i] = addmod<T>(mulmod<T>(ca, A[i], m), mulmod<T>(cb, B[i], m), m);
}

// A[i] = A[i] * B[i]
template <typename T>
__global__ void k_pointwise_mul(T *__restrict__ A, const T *__restrict__ B,
                                uint32_t n, T m) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) A[i] = mulmod<T>(A[i], B[i], m);
}

// A[i] = A[i] - B[i]
template <typename T>
__global__ void k_pointwise_sub(T *__restrict__ A, const T *__restrict__ B,
                                uint32_t n, T m) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) A[i] = submod<T>(A[i], B[i], m);
}

// A[i] = A[i] + B[i]
template <typename T>
__global__ void k_pointwise_add(T *__restrict__ A, const T *__restrict__ B,
                                uint32_t n, T m) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) A[i] = addmod<T>(A[i], B[i], m);
}

// a[idx] += addval   (addval already reduced into [0,m); subtract = pass m-x)
template <typename T>
__global__ void k_add_elt(T *__restrict__ a, uint32_t idx, T addval, T m) {
    if (threadIdx.x == 0 && blockIdx.x == 0)
        a[idx] = addmod<T>(a[idx], addval, m);
}

// d[i] = i<sm ? a[i]+a[i+bm] : a[i]   (step_radix2 FFT pre-pass, the "c" half)
template <typename T>
__global__ void k_step_c(const T *__restrict__ a, T *__restrict__ c, uint32_t bm,
                         uint32_t sm, T m) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= bm) return;
    c[i] = (i < sm) ? addmod<T>(a[i], a[i + bm], m) : a[i];
}

// d[i] = tw[i] * (i<sm ? a[i]-a[i+bm] : a[i])   (the "d" half; tw[i]=omega^i)
template <typename T>
__global__ void k_step_d(const T *__restrict__ a, const T *__restrict__ tw,
                         T *__restrict__ d, uint32_t bm, uint32_t sm, T m) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= bm) return;
    T base = (i < sm) ? submod<T>(a[i], a[i + bm], m) : a[i];
    d[i] = mulmod<T>(tw[i], base, m);
}

// e[i] = sum_{j=0}^{compr-1} d[i + j*sm]   (fold "d" down to small_m)
template <typename T>
__global__ void k_step_fold(const T *__restrict__ d, T *__restrict__ e,
                            uint32_t sm, uint32_t compr, T m) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= sm) return;
    T acc = d[i];
    for (uint32_t j = 1; j < compr; ++j) acc = addmod<T>(acc, d[i + j * sm], m);
    e[i] = acc;
}

// ======================= extended_radix2_large helpers ======================
// col[j] = a[j*nroots + i]   (gather a strided column)
template <typename T>
__global__ void k_gather(const T *__restrict__ a, T *__restrict__ col,
                         uint32_t ncosets, uint32_t nroots, uint32_t i) {
    uint32_t j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j < ncosets) col[j] = a[(uint64_t)j * nroots + i];
}
// col[j] = a[j*nroots + i] * two[j*nroots + i]   (gather + 2D scale)
template <typename T>
__global__ void k_gather_scaled(const T *__restrict__ a, const T *__restrict__ two,
                                T *__restrict__ col, uint32_t ncosets,
                                uint32_t nroots, uint32_t i, T m) {
    uint32_t j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j < ncosets) {
        uint64_t idx = (uint64_t)j * nroots + i;
        col[j] = mulmod<T>(a[idx], two[idx], m);
    }
}
// a[j*nroots + i] = col[j]
template <typename T>
__global__ void k_scatter(T *__restrict__ a, const T *__restrict__ col,
                          uint32_t ncosets, uint32_t nroots, uint32_t i) {
    uint32_t j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j < ncosets) a[(uint64_t)j * nroots + i] = col[j];
}
// a[j*nroots + i] = col[j] * two[j*nroots + i]   (scatter + 2D scale)
template <typename T>
__global__ void k_scatter_scaled(T *__restrict__ a, const T *__restrict__ col,
                                 const T *__restrict__ two, uint32_t ncosets,
                                 uint32_t nroots, uint32_t i, T m) {
    uint32_t j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j < ncosets) {
        uint64_t idx = (uint64_t)j * nroots + i;
        a[idx] = mulmod<T>(col[j], two[idx], m);
    }
}

// Batched radix-2 NTT: one CUDA block per coset, nroots elements in shared mem.
// blockDim.x == nroots/2. Pure NTT (no 1/n scaling). tw[k] = root^k (nroots/2).
template <typename T>
__global__ void k_batched_ntt(T *__restrict__ data, const T *__restrict__ tw,
                              uint32_t nroots, uint32_t logn, T m) {
    extern __shared__ char smem[];
    T *s = reinterpret_cast<T *>(smem);
    const uint32_t blk = blockIdx.x;
    const uint32_t tid = threadIdx.x;       // 0 .. nroots/2-1
    const uint64_t base = (uint64_t)blk * nroots;
    const uint32_t half_n = nroots >> 1;
    // bit-reversed load (two elements per thread)
    for (uint32_t p = tid; p < nroots; p += half_n) {
        uint32_t r = 0, x = p;
        for (uint32_t b = 0; b < logn; ++b) { r = (r << 1) | (x & 1u); x >>= 1; }
        s[p] = data[base + r];
    }
    __syncthreads();
    for (uint32_t stage = 1; stage <= logn; ++stage) {
        uint32_t half = 1u << (stage - 1);
        uint32_t j = tid & (half - 1);
        uint32_t grp = tid >> (stage - 1);
        uint32_t i1 = (grp << stage) + j;
        uint32_t i2 = i1 + half;
        T w = tw[(uint64_t)j << (logn - stage)];
        T u = s[i1];
        T v = mulmod<T>(w, s[i2], m);
        s[i1] = addmod<T>(u, v, m);
        s[i2] = submod<T>(u, v, m);
        __syncthreads();
    }
    for (uint32_t p = tid; p < nroots; p += half_n) data[base + p] = s[p];
}

// P[j*nroots + i] *= vci[j]   (divide_by_Z_on_coset, per-coset scalar)
template <typename T>
__global__ void k_divz_percoset(T *__restrict__ P, const T *__restrict__ vci,
                                uint64_t total, uint32_t nroots, T m) {
    uint64_t idx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total) P[idx] = mulmod<T>(P[idx], vci[idx / nroots], m);
}

// H[j*nroots] += van[j] * coeff   for j in [0, ncosets+1)
template <typename T>
__global__ void k_addpolyZ(T *__restrict__ H, const T *__restrict__ van, T coeff,
                           uint32_t ncosets1, uint32_t nroots, T m) {
    uint32_t j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j < ncosets1)
        H[(uint64_t)j * nroots] =
            addmod<T>(H[(uint64_t)j * nroots], mulmod<T>(van[j], coeff, m), m);
}

// warp shuffle-down for T (uint64_t or unsigned __int128): __shfl_down_sync only
// handles ≤64-bit, so a 128-bit value is shuffled as its two 64-bit halves.
template <typename T>
__device__ __forceinline__ T warp_shfl_down(T v, int off) {
    if constexpr (sizeof(T) == 8) {
        unsigned long long x = (unsigned long long)v;
        return (T)__shfl_down_sync(0xffffffffu, x, off);
    } else {
        unsigned long long lo = (unsigned long long)v;
        unsigned long long hi = (unsigned long long)(v >> 64);
        lo = __shfl_down_sync(0xffffffffu, lo, off);
        hi = __shfl_down_sync(0xffffffffu, hi, off);
        return ((T)hi << 64) | (T)lo;
    }
}

// QAP instance-map sparse gather (the generator's r1cs_to_qap sparse A/B/C eval).
// For each output variable v: out[v] = init[v] + Σ_{k ∈ [col_ptr[v],col_ptr[v+1])}
// u[row_idx[k]] * coeff[k]. This is the transpose (CSC) form of the CPU scatter
// `At[term.index] += u[i]*term.coeff`; field addition is exact/commutative so the
// per-column sum is bit-identical to the CPU's constraint-order accumulation.
// `init` (nullptr => 0) carries At's input-consistency seed u[num_constraints+v].
//
// ONE WARP PER COLUMN: the 32 lanes stride the column's entries and warp-reduce.
// A hot column (e.g. the constant-1 variable, ~num_constraints entries) would
// serialize one thread under one-thread-per-column (the 128-bit modmul is costly),
// so spread it across a warp — turns the tail column from ~num_constraints ops on
// one lane into num_constraints/32.
template <typename T>
__global__ void k_spmv_col(const uint64_t *__restrict__ col_ptr,
                           const uint32_t *__restrict__ row_idx,
                           const T *__restrict__ coeff,
                           const T *__restrict__ u,
                           const T *__restrict__ init, T *__restrict__ out,
                           uint32_t num_out, T m) {
    const uint32_t gtid = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t v = gtid >> 5;        // one warp per column
    const uint32_t lane = gtid & 31u;
    if (v >= num_out) return;
    const uint64_t beg = col_ptr[v], end = col_ptr[v + 1];
    T acc = (T)0;
    for (uint64_t k = beg + lane; k < end; k += 32)
        acc = addmod<T>(acc, mulmod<T>(u[row_idx[k]], coeff[k], m), m);
#pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        acc = addmod<T>(acc, warp_shfl_down<T>(acc, off), m);
    if (lane == 0) {
        if (init) acc = addmod<T>(acc, init[v], m);
        out[v] = acc;
    }
}

// ---- device modpow + Lagrange-eval helper kernels (generator QAP) ----
// b^e mod m by square-and-multiply (e fits in 64 bits: exponents are either a
// domain index < 2^32 or the Fermat exponent p-2 for the native ≤60-bit fields).
template <typename T>
__device__ __forceinline__ T powmod(T b, unsigned long long e, T m) {
    T r = (m == (T)1) ? (T)0 : (T)1;
    b %= m;
    while (e) {
        if (e & 1ull) r = mulmod<T>(r, b, m);
        b = mulmod<T>(b, b, m);
        e >>= 1;
    }
    return r;
}
// out[i] = base^i mod m  (geometric sequence; used for ω^i, t^i, coset tables).
template <typename T>
__global__ void k_geom(T *__restrict__ out, T base, uint32_t n, T m) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = powmod<T>(base, (unsigned long long)i, m);
}
// a[i] = a[i]^exp mod m in place (exp = p-2 => the unique field inverse, so the
// result is bit-identical to any correct CPU batch inverse regardless of method).
template <typename T>
__global__ void k_field_pow(T *__restrict__ a, unsigned long long exp, uint32_t n,
                            T m) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) a[i] = powmod<T>(a[i], exp, m);
}
// a[i] = submod(a[i], s)  (a - s).
template <typename T>
__global__ void k_sub_scalar(T *__restrict__ a, T s, uint32_t n, T m) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) a[i] = submod<T>(a[i], s, m);
}
// a[i] = submod(s, a[i])  (s - a).
template <typename T>
__global__ void k_rsub_scalar(T *__restrict__ a, T s, uint32_t n, T m) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) a[i] = submod<T>(s, a[i], m);
}

// ===========================================================================
// Launchers (host-callable). Defined per width via explicit instantiation.
// ===========================================================================
static inline uint32_t grid(uint32_t n) { return (n + TPB - 1) / TPB; }

// IMPORTANT (ABI): all 128-bit field scalars (mod and field constants) are taken
// by `const T&`, NOT by value. unsigned __int128 passed by value has a SysV ABI
// divergence between gcc and clang in mixed argument lists, and these launchers
// are compiled by nvcc's host compiler (gcc) while callers may be clang
// (LatticeZKSink) — a by-value __int128 silently corrupted the scalar/`n` args.
// `const T&` lowers to a pointer, which both compilers pass identically. The
// device kernels still take T by value (single TU, marshalled by nvcc).
template <typename T>
void qap_ntt_gpu(T *d_data, const T *d_tw, T *d_scratch, uint32_t logn,
                 const T &mod) {
    const uint32_t n = 1u << logn;
    k_bitrev<T><<<grid(n), TPB>>>(d_data, d_scratch, n, logn);
    cudaMemcpy(d_data, d_scratch, (size_t)n * sizeof(T),
               cudaMemcpyDeviceToDevice);
    const uint32_t halfn = n >> 1;
    for (uint32_t s = 1; s <= logn; ++s)
        k_ntt_stage<T><<<grid(halfn), TPB>>>(d_data, d_tw, n, logn, s, mod);
}

template <typename T>
void qap_scale_gpu(T *d, uint32_t n, const T &s, const T &mod) {
    k_scale<T><<<grid(n), TPB>>>(d, n, s, mod);
}
template <typename T>
void qap_mul_vec_gpu(T *d, const T *tab, uint32_t n, const T &mod) {
    k_mul_vec<T><<<grid(n), TPB>>>(d, tab, n, mod);
}
template <typename T>
void qap_lincomb2_gpu(T *out, const T *A, const T *B, const T &ca, const T &cb,
                      uint32_t n, const T &mod) {
    k_lincomb2<T><<<grid(n), TPB>>>(out, A, B, ca, cb, n, mod);
}
template <typename T>
void qap_pointwise_mul_gpu(T *A, const T *B, uint32_t n, const T &mod) {
    k_pointwise_mul<T><<<grid(n), TPB>>>(A, B, n, mod);
}
template <typename T>
void qap_pointwise_sub_gpu(T *A, const T *B, uint32_t n, const T &mod) {
    k_pointwise_sub<T><<<grid(n), TPB>>>(A, B, n, mod);
}
template <typename T>
void qap_pointwise_add_gpu(T *A, const T *B, uint32_t n, const T &mod) {
    k_pointwise_add<T><<<grid(n), TPB>>>(A, B, n, mod);
}
template <typename T>
void qap_add_elt_gpu(T *a, uint32_t idx, const T &addval, const T &mod) {
    k_add_elt<T><<<1, 1>>>(a, idx, addval, mod);
}
template <typename T>
void qap_step_c_gpu(const T *a, T *c, uint32_t bm, uint32_t sm, const T &mod) {
    k_step_c<T><<<grid(bm), TPB>>>(a, c, bm, sm, mod);
}
template <typename T>
void qap_step_d_gpu(const T *a, const T *tw, T *d, uint32_t bm, uint32_t sm,
                    const T &mod) {
    k_step_d<T><<<grid(bm), TPB>>>(a, tw, d, bm, sm, mod);
}
template <typename T>
void qap_step_fold_gpu(const T *d, T *e, uint32_t sm, uint32_t compr,
                       const T &mod) {
    k_step_fold<T><<<grid(sm), TPB>>>(d, e, sm, compr, mod);
}

// extended_radix2_large launchers
template <typename T>
void qap_gather_gpu(const T *a, T *col, uint32_t ncosets, uint32_t nroots,
                    uint32_t i) {
    k_gather<T><<<grid(ncosets), TPB>>>(a, col, ncosets, nroots, i);
}
template <typename T>
void qap_gather_scaled_gpu(const T *a, const T *two, T *col, uint32_t ncosets,
                           uint32_t nroots, uint32_t i, const T &mod) {
    k_gather_scaled<T><<<grid(ncosets), TPB>>>(a, two, col, ncosets, nroots, i, mod);
}
template <typename T>
void qap_scatter_gpu(T *a, const T *col, uint32_t ncosets, uint32_t nroots,
                     uint32_t i) {
    k_scatter<T><<<grid(ncosets), TPB>>>(a, col, ncosets, nroots, i);
}
template <typename T>
void qap_scatter_scaled_gpu(T *a, const T *col, const T *two, uint32_t ncosets,
                            uint32_t nroots, uint32_t i, const T &mod) {
    k_scatter_scaled<T><<<grid(ncosets), TPB>>>(a, col, two, ncosets, nroots, i, mod);
}
template <typename T>
void qap_batched_ntt_gpu(T *data, const T *tw, uint32_t ncosets, uint32_t nroots,
                         uint32_t logn, const T &mod) {
    if (nroots == 1) return; // logn==0: NTT of size 1 is identity
    k_batched_ntt<T><<<ncosets, nroots / 2, (size_t)nroots * sizeof(T)>>>(
        data, tw, nroots, logn, mod);
}
template <typename T>
void qap_divz_percoset_gpu(T *P, const T *vci, uint64_t total, uint32_t nroots,
                           const T &mod) {
    uint32_t blocks = (uint32_t)((total + TPB - 1) / TPB);
    k_divz_percoset<T><<<blocks, TPB>>>(P, vci, total, nroots, mod);
}
template <typename T>
void qap_addpolyZ_gpu(T *H, const T *van, const T &coeff, uint32_t ncosets1,
                      uint32_t nroots, const T &mod) {
    k_addpolyZ<T><<<grid(ncosets1), TPB>>>(H, van, coeff, ncosets1, nroots, mod);
}
template <typename T>
void qap_spmv_col_gpu(const uint64_t *col_ptr, const uint32_t *row_idx,
                      const T *coeff, const T *u, const T *init, T *out,
                      uint32_t num_out, const T &mod) {
    // one warp (32 lanes) per output column; TPB/32 warps per block.
    const uint32_t blocks = (num_out + (TPB / 32) - 1) / (TPB / 32);
    k_spmv_col<T><<<blocks, TPB>>>(col_ptr, row_idx, coeff, u, init, out,
                                   num_out, mod);
}
template <typename T>
void qap_geom_gpu(T *out, const T &base, uint32_t n, const T &mod) {
    k_geom<T><<<grid(n), TPB>>>(out, base, n, mod);
}
template <typename T>
void qap_field_pow_gpu(T *a, unsigned long long exp, uint32_t n, const T &mod) {
    k_field_pow<T><<<grid(n), TPB>>>(a, exp, n, mod);
}
template <typename T>
void qap_sub_scalar_gpu(T *a, const T &s, uint32_t n, const T &mod) {
    k_sub_scalar<T><<<grid(n), TPB>>>(a, s, n, mod);
}
template <typename T>
void qap_rsub_scalar_gpu(T *a, const T &s, uint32_t n, const T &mod) {
    k_rsub_scalar<T><<<grid(n), TPB>>>(a, s, n, mod);
}

// ---- explicit instantiations for the two field widths ----
#define QAP_INST(T)                                                            \
    template void qap_ntt_gpu<T>(T *, const T *, T *, uint32_t, const T &);     \
    template void qap_scale_gpu<T>(T *, uint32_t, const T &, const T &);        \
    template void qap_mul_vec_gpu<T>(T *, const T *, uint32_t, const T &);      \
    template void qap_lincomb2_gpu<T>(T *, const T *, const T *, const T &,     \
                                      const T &, uint32_t, const T &);          \
    template void qap_pointwise_mul_gpu<T>(T *, const T *, uint32_t, const T &);\
    template void qap_pointwise_sub_gpu<T>(T *, const T *, uint32_t, const T &);\
    template void qap_pointwise_add_gpu<T>(T *, const T *, uint32_t, const T &);\
    template void qap_add_elt_gpu<T>(T *, uint32_t, const T &, const T &);      \
    template void qap_step_c_gpu<T>(const T *, T *, uint32_t, uint32_t,         \
                                    const T &);                                \
    template void qap_step_d_gpu<T>(const T *, const T *, T *, uint32_t,        \
                                    uint32_t, const T &);                       \
    template void qap_step_fold_gpu<T>(const T *, T *, uint32_t, uint32_t,      \
                                       const T &);                             \
    template void qap_gather_gpu<T>(const T *, T *, uint32_t, uint32_t,         \
                                    uint32_t);                                 \
    template void qap_gather_scaled_gpu<T>(const T *, const T *, T *, uint32_t, \
                                           uint32_t, uint32_t, const T &);      \
    template void qap_scatter_gpu<T>(T *, const T *, uint32_t, uint32_t,        \
                                     uint32_t);                                \
    template void qap_scatter_scaled_gpu<T>(T *, const T *, const T *,          \
                                            uint32_t, uint32_t, uint32_t,       \
                                            const T &);                        \
    template void qap_batched_ntt_gpu<T>(T *, const T *, uint32_t, uint32_t,    \
                                         uint32_t, const T &);                  \
    template void qap_divz_percoset_gpu<T>(T *, const T *, uint64_t, uint32_t,  \
                                           const T &);                          \
    template void qap_addpolyZ_gpu<T>(T *, const T *, const T &, uint32_t,      \
                                      uint32_t, const T &);                     \
    template void qap_spmv_col_gpu<T>(const uint64_t *, const uint32_t *,       \
                                      const T *, const T *, const T *, T *,     \
                                      uint32_t, const T &);                     \
    template void qap_geom_gpu<T>(T *, const T &, uint32_t, const T &);         \
    template void qap_field_pow_gpu<T>(T *, unsigned long long, uint32_t,       \
                                       const T &);                             \
    template void qap_sub_scalar_gpu<T>(T *, const T &, uint32_t, const T &);   \
    template void qap_rsub_scalar_gpu<T>(T *, const T &, uint32_t, const T &);

QAP_INST(uint64_t)
QAP_INST(unsigned __int128)
#undef QAP_INST

} // namespace qap_gpu_detail
} // namespace libsnark
