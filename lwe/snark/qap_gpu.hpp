#ifndef __QAP_GPU_HPP__
#define __QAP_GPU_HPP__

// GPU QAP witness map (host orchestration).
//
// Mirrors libsnark::r1cs_to_qap_witness_map (r1cs_to_qap.tcc) but runs the
// FFT-domain pipeline (iFFT / cosetFFT / pointwise / divide_by_Z / icosetFFT)
// on the GPU using the primitives in qap_gpu.cu, and assembles the prover's
// `pi` vector directly in device memory so the subsequent "CRS key * pi"
// response accumulation runs on it with no extra host<->device round-trip.
//
// Determinism: the SNARK field is an ordinary residue representation and every
// op is exact modular arithmetic, so a radix-2 NTT with the same root of unity
// computes a DFT bit-identical to libfqfft's FFT. The GPU `pi` therefore equals
// the CPU `pi` exactly (validated by test_qap_gpu).
//
// The sparse "evaluate A,B,C on set S" assembly (the R1CS matrix * witness
// products) is kept on the host — it reuses the exact CPU code path, which both
// guarantees bit-identical inputs to the FFTs and keeps the irregular sparse
// work off the critical FFT path. Only domains we replicate exactly
// (basic_radix2, step_radix2) take the GPU path; any other domain returns false
// so the caller falls back to the CPU witness map.

#include <cuda_runtime.h>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include <libff/algebra/fields/field_utils.hpp>
#include <libfqfft/evaluation_domain/get_evaluation_domain.hpp>
#include <libfqfft/evaluation_domain/domains/basic_radix2_domain.hpp>
#include <libfqfft/evaluation_domain/domains/step_radix2_domain.hpp>
#include <libfqfft/evaluation_domain/domains/extended_radix2_large_domain.hpp>
#include <libsnark/relations/constraint_satisfaction_problems/r1cs/r1cs.hpp>
#include <libsnark/reductions/r1cs_to_qap/r1cs_to_qap.hpp>

namespace libsnark {
namespace qap_gpu_detail {

// Launchers defined+instantiated in qap_gpu.cu (for T = uint64_t and
// unsigned __int128). Declared as templates here; the linker binds to the
// explicit instantiations. All 128-bit field scalars are passed by `const T&`
// (NOT by value): unsigned __int128 by-value has a gcc/clang SysV ABI
// divergence, and these launchers (nvcc/gcc host) are called from clang
// (LatticeZKSink). A reference lowers to a pointer, identical across compilers.
template <typename T>
void qap_ntt_gpu(T *d_data, const T *d_tw, T *d_scratch, uint32_t logn,
                 const T &mod);
template <typename T> void qap_scale_gpu(T *d, uint32_t n, const T &s, const T &mod);
template <typename T> void qap_mul_vec_gpu(T *d, const T *tab, uint32_t n, const T &mod);
template <typename T>
void qap_lincomb2_gpu(T *out, const T *A, const T *B, const T &ca, const T &cb,
                      uint32_t n, const T &mod);
template <typename T> void qap_pointwise_mul_gpu(T *A, const T *B, uint32_t n, const T &mod);
template <typename T> void qap_pointwise_sub_gpu(T *A, const T *B, uint32_t n, const T &mod);
template <typename T> void qap_pointwise_add_gpu(T *A, const T *B, uint32_t n, const T &mod);
template <typename T> void qap_add_elt_gpu(T *a, uint32_t idx, const T &addval, const T &mod);
template <typename T>
void qap_step_c_gpu(const T *a, T *c, uint32_t bm, uint32_t sm, const T &mod);
template <typename T>
void qap_step_d_gpu(const T *a, const T *tw, T *d, uint32_t bm, uint32_t sm, const T &mod);
template <typename T>
void qap_step_fold_gpu(const T *d, T *e, uint32_t sm, uint32_t compr, const T &mod);
template <typename T>
void qap_gather_gpu(const T *a, T *col, uint32_t ncosets, uint32_t nroots,
                    uint32_t i);
template <typename T>
void qap_gather_scaled_gpu(const T *a, const T *two, T *col, uint32_t ncosets,
                           uint32_t nroots, uint32_t i, const T &mod);
template <typename T>
void qap_scatter_gpu(T *a, const T *col, uint32_t ncosets, uint32_t nroots,
                     uint32_t i);
template <typename T>
void qap_scatter_scaled_gpu(T *a, const T *col, const T *two, uint32_t ncosets,
                            uint32_t nroots, uint32_t i, const T &mod);
template <typename T>
void qap_batched_ntt_gpu(T *data, const T *tw, uint32_t ncosets, uint32_t nroots,
                         uint32_t logn, const T &mod);
template <typename T>
void qap_divz_percoset_gpu(T *P, const T *vci, uint64_t total, uint32_t nroots,
                           const T &mod);
template <typename T>
void qap_addpolyZ_gpu(T *H, const T *van, const T &coeff, uint32_t ncosets1,
                      uint32_t nroots, const T &mod);
// Sparse gather for the generator's QAP instance map (see k_spmv_col in
// qap_gpu.cu). out[v] = init[v] + Σ u[row_idx[k]]*coeff[k] over column v.
template <typename T>
void qap_spmv_col_gpu(const uint64_t *col_ptr, const uint32_t *row_idx,
                      const T *coeff, const T *u, const T *init, T *out,
                      uint32_t num_out, const T &mod);
// Lagrange-eval helpers (generator QAP): geometric powers, per-element Fermat
// inverse (a^(p-2)), and vector±scalar. See qap_gpu.cu.
template <typename T>
void qap_geom_gpu(T *out, const T &base, uint32_t n, const T &mod);
template <typename T>
void qap_field_pow_gpu(T *a, unsigned long long exp, uint32_t n, const T &mod);
template <typename T>
void qap_sub_scalar_gpu(T *a, const T &s, uint32_t n, const T &mod);
template <typename T>
void qap_rsub_scalar_gpu(T *a, const T &s, uint32_t n, const T &mod);

#define QAP_CUDA_CHECK(call)                                                    \
    do {                                                                       \
        cudaError_t _e = (call);                                              \
        if (_e != cudaSuccess) {                                              \
            std::fprintf(stderr, "[qap_gpu] CUDA error %s:%d: %s\n", __FILE__, \
                         __LINE__, cudaGetErrorString(_e));                   \
            throw std::runtime_error("qap_gpu CUDA failure");                 \
        }                                                                     \
    } while (0)

// ---- tiny host helpers over FieldT (exact, used only for scalars/tables) ----
template <typename FieldT> static inline uint32_t ilog2(uint32_t n) {
    uint32_t r = 0;
    while ((1u << r) < n) ++r;
    return r;
}

template <typename FieldT> static inline FieldT field_pow(FieldT b, uint64_t e) {
    FieldT r = FieldT::one();
    while (e) {
        if (e & 1ull) r = r * b;
        b = b * b;
        e >>= 1;
    }
    return r;
}

// Montgomery batch inverse of a host vector (1 field inversion + 3n mults).
template <typename FieldT>
static inline void batch_inverse(std::vector<FieldT> &v) {
    const size_t n = v.size();
    if (n == 0) return;
    std::vector<FieldT> prefix(n);
    FieldT acc = FieldT::one();
    for (size_t i = 0; i < n; ++i) {
        prefix[i] = acc;
        acc = acc * v[i];
    }
    FieldT inv = acc.inverse();
    for (size_t i = n; i-- > 0;) {
        FieldT cur = prefix[i] * inv;
        inv = inv * v[i];
        v[i] = cur;
    }
}

// Device geometric table tab[i] = base^i, i in [0, len). Returns device ptr.
template <typename FieldT, typename T>
static inline T *upload_geom(FieldT base, uint32_t len) {
    using HostT = decltype(FieldT::value);
    std::vector<T> h(len);
    FieldT cur = FieldT::one();
    for (uint32_t i = 0; i < len; ++i) {
        h[i] = (T)(HostT)cur.value;
        cur = cur * base;
    }
    T *d = nullptr;
    QAP_CUDA_CHECK(cudaMalloc(&d, (size_t)len * sizeof(T)));
    QAP_CUDA_CHECK(cudaMemcpy(d, h.data(), (size_t)len * sizeof(T),
                              cudaMemcpyHostToDevice));
    return d;
}

template <typename T>
static inline T *upload_vec(const std::vector<T> &h) {
    T *d = nullptr;
    QAP_CUDA_CHECK(cudaMalloc(&d, h.size() * sizeof(T)));
    QAP_CUDA_CHECK(cudaMemcpy(d, h.data(), h.size() * sizeof(T),
                              cudaMemcpyHostToDevice));
    return d;
}

// ====================== basic_radix2 GPU witness pipeline ====================
// Given device aA/aB/aC (size m, raw residues), produce coefficients_for_H in
// dH (size m+1). Mirrors r1cs_to_qap_witness_map for a basic_radix2 domain.
template <typename FieldT, typename T>
static void witness_H_basic(T *dA, T *dB, T *dC, T *dH, T *dScratch, uint32_t m,
                            FieldT omega, FieldT g, T d1, T d2, T d3, T mod) {
    const uint32_t logm = ilog2<FieldT>(m);
    const FieldT omega_inv = omega.inverse();
    const FieldT g_inv = g.inverse();
    const T inv_m = (T)(decltype(FieldT::value))FieldT((long)m).inverse().value;

    T *TWf = upload_geom<FieldT, T>(omega, m / 2);
    T *TWi = upload_geom<FieldT, T>(omega_inv, m / 2);
    T *COS = upload_geom<FieldT, T>(g, m);
    T *COSi = upload_geom<FieldT, T>(g_inv, m);

    // 1-2. iFFT(aA), iFFT(aB)
    qap_ntt_gpu<T>(dA, TWi, dScratch, logm, mod); qap_scale_gpu<T>(dA, m, inv_m, mod);
    qap_ntt_gpu<T>(dB, TWi, dScratch, logm, mod); qap_scale_gpu<T>(dB, m, inv_m, mod);

    // 3. ZK-patch coefficients_for_H = d2*aA + d1*aB; [0]-=d3; add_poly_Z(d1*d2)
    QAP_CUDA_CHECK(cudaMemset(dH, 0, (size_t)(m + 1) * sizeof(T)));
    qap_lincomb2_gpu<T>(dH, dA, dB, d2, d1, m, mod);
    qap_add_elt_gpu<T>(dH, 0, mod - d3, mod);
    const T c = (T)(((unsigned __int128)d1 * d2) % mod); // d1*d2 in field
    qap_add_elt_gpu<T>(dH, m, c, mod);
    qap_add_elt_gpu<T>(dH, 0, mod - c, mod);

    // 4-5. cosetFFT(aA), cosetFFT(aB)
    qap_mul_vec_gpu<T>(dA, COS, m, mod); qap_ntt_gpu<T>(dA, TWf, dScratch, logm, mod);
    qap_mul_vec_gpu<T>(dB, COS, m, mod); qap_ntt_gpu<T>(dB, TWf, dScratch, logm, mod);

    // 6. H_tmp = aA*aB  (in dA)
    qap_pointwise_mul_gpu<T>(dA, dB, m, mod);

    // 7. iFFT(aC) then cosetFFT(aC)
    qap_ntt_gpu<T>(dC, TWi, dScratch, logm, mod); qap_scale_gpu<T>(dC, m, inv_m, mod);
    qap_mul_vec_gpu<T>(dC, COS, m, mod); qap_ntt_gpu<T>(dC, TWf, dScratch, logm, mod);

    // 8. H_tmp -= aC
    qap_pointwise_sub_gpu<T>(dA, dC, m, mod);

    // 9. divide_by_Z_on_coset: scalar ((g^m)-1)^{-1}
    const FieldT Zinv = (field_pow(g, m) - FieldT::one()).inverse();
    qap_scale_gpu<T>(dA, m, (T)(decltype(FieldT::value))Zinv.value, mod);

    // 10. icosetFFT(H_tmp) = iFFT then multiply_by_coset(g^{-1})
    qap_ntt_gpu<T>(dA, TWi, dScratch, logm, mod); qap_scale_gpu<T>(dA, m, inv_m, mod);
    qap_mul_vec_gpu<T>(dA, COSi, m, mod);

    // 11. coefficients_for_H[0..m) += H_tmp
    qap_pointwise_add_gpu<T>(dH, dA, m, mod);

    cudaFree(TWf); cudaFree(TWi); cudaFree(COS); cudaFree(COSi);
}

// ======================= step_radix2 helpers (GPU) ==========================
// In-place step_radix2 iFFT of device buffer a (size bm+sm). Mirrors
// step_radix2_domain::iFFT exactly.
template <typename FieldT, typename T>
static void step_iFFT(T *a, T *scratch, uint32_t bm, uint32_t sm, FieldT omega,
                      FieldT big_omega, FieldT small_omega, T mod) {
    const uint32_t lbm = ilog2<FieldT>(bm), lsm = ilog2<FieldT>(sm);
    T *U0 = a;          // first big_m
    T *U1 = a + bm;     // last small_m
    const T inv_bm = (T)(decltype(FieldT::value))FieldT((long)bm).inverse().value;
    const T inv_sm = (T)(decltype(FieldT::value))FieldT((long)sm).inverse().value;
    const FieldT over_two = FieldT((long)2).inverse();

    T *TWbi = upload_geom<FieldT, T>(big_omega.inverse(), bm / 2 ? bm / 2 : 1);
    T *TWsi = upload_geom<FieldT, T>(small_omega.inverse(), sm / 2 ? sm / 2 : 1);

    qap_ntt_gpu<T>(U0, TWbi, scratch, lbm, mod); qap_scale_gpu<T>(U0, bm, inv_bm, mod);
    qap_ntt_gpu<T>(U1, TWsi, scratch, lsm, mod); qap_scale_gpu<T>(U1, sm, inv_sm, mod);

    // tmp = U0 .* (omega^i)  (over big_m)
    T *tmp = nullptr; QAP_CUDA_CHECK(cudaMalloc(&tmp, (size_t)bm * sizeof(T)));
    QAP_CUDA_CHECK(cudaMemcpy(tmp, U0, (size_t)bm * sizeof(T), cudaMemcpyDeviceToDevice));
    T *OMb = upload_geom<FieldT, T>(omega, bm);
    qap_mul_vec_gpu<T>(tmp, OMb, bm, mod);

    // U1[i] -= sum_{j=1}^{compr-1} tmp[i + j*sm]   (i in [0,sm))
    const uint32_t compr = 1u << (lbm - lsm);
    if (compr > 1) {
        // fold tmp[sm .. bm) into U1: build partial-sum buffer of the j>=1 terms
        T *fold = nullptr; QAP_CUDA_CHECK(cudaMalloc(&fold, (size_t)sm * sizeof(T)));
        // reuse k_step_fold over tmp+sm with (compr-1) groups
        qap_step_fold_gpu<T>(tmp + sm, fold, sm, compr - 1, mod);
        qap_pointwise_sub_gpu<T>(U1, fold, sm, mod);
        cudaFree(fold);
    }
    // U1[i] *= (omega^{-1})^i  (over sm)
    T *OMsi = upload_geom<FieldT, T>(omega.inverse(), sm);
    qap_mul_vec_gpu<T>(U1, OMsi, sm, mod);

    // a layout result:
    //   a[i]      = (U0[i]+U1[i])*over_two           i in [0,sm)
    //   a[bm+i]   = (U0[i]-U1[i])*over_two           i in [0,sm)
    //   a[i]      = U0[i]  (unchanged)               i in [sm,bm)
    // U0 occupies a[0..bm); we must combine the first sm of U0 with U1 without
    // clobbering U0[i] still needed. Compute into a separate small buffer.
    const T ot = (T)(decltype(FieldT::value))over_two.value;
    T *lo = nullptr, *hi = nullptr;
    QAP_CUDA_CHECK(cudaMalloc(&lo, (size_t)sm * sizeof(T)));
    QAP_CUDA_CHECK(cudaMalloc(&hi, (size_t)sm * sizeof(T)));
    // lo = (U0[0..sm)+U1)*ot ; hi = (U0[0..sm)-U1)*ot
    QAP_CUDA_CHECK(cudaMemcpy(lo, U0, (size_t)sm * sizeof(T), cudaMemcpyDeviceToDevice));
    QAP_CUDA_CHECK(cudaMemcpy(hi, U0, (size_t)sm * sizeof(T), cudaMemcpyDeviceToDevice));
    qap_pointwise_add_gpu<T>(lo, U1, sm, mod); qap_scale_gpu<T>(lo, sm, ot, mod);
    qap_pointwise_sub_gpu<T>(hi, U1, sm, mod); qap_scale_gpu<T>(hi, sm, ot, mod);
    // write back: a[bm+i] = hi ; a[i] = lo (U0[sm..bm) already correct in place)
    QAP_CUDA_CHECK(cudaMemcpy(U1, hi, (size_t)sm * sizeof(T), cudaMemcpyDeviceToDevice));
    QAP_CUDA_CHECK(cudaMemcpy(U0, lo, (size_t)sm * sizeof(T), cudaMemcpyDeviceToDevice));

    cudaFree(tmp); cudaFree(OMb); cudaFree(OMsi); cudaFree(lo); cudaFree(hi);
    cudaFree(TWbi); cudaFree(TWsi);
}

// In-place step_radix2 FFT of device buffer a (size bm+sm). Mirrors ::FFT.
template <typename FieldT, typename T>
static void step_FFT(T *a, T *scratch, uint32_t bm, uint32_t sm, FieldT omega,
                     FieldT big_omega, FieldT small_omega, T mod) {
    const uint32_t lbm = ilog2<FieldT>(bm), lsm = ilog2<FieldT>(sm);
    T *c = nullptr, *d = nullptr, *e = nullptr;
    QAP_CUDA_CHECK(cudaMalloc(&c, (size_t)bm * sizeof(T)));
    QAP_CUDA_CHECK(cudaMalloc(&d, (size_t)bm * sizeof(T)));
    QAP_CUDA_CHECK(cudaMalloc(&e, (size_t)sm * sizeof(T)));
    T *OMb = upload_geom<FieldT, T>(omega, bm);

    qap_step_c_gpu<T>(a, c, bm, sm, mod);
    qap_step_d_gpu<T>(a, OMb, d, bm, sm, mod);
    const uint32_t compr = 1u << (lbm - lsm);
    qap_step_fold_gpu<T>(d, e, sm, compr, mod);

    T *TWb = upload_geom<FieldT, T>(big_omega, bm / 2 ? bm / 2 : 1);
    T *TWs = upload_geom<FieldT, T>(small_omega, sm / 2 ? sm / 2 : 1);
    qap_ntt_gpu<T>(c, TWb, scratch, lbm, mod);
    qap_ntt_gpu<T>(e, TWs, scratch, lsm, mod);

    QAP_CUDA_CHECK(cudaMemcpy(a, c, (size_t)bm * sizeof(T), cudaMemcpyDeviceToDevice));
    QAP_CUDA_CHECK(cudaMemcpy(a + bm, e, (size_t)sm * sizeof(T), cudaMemcpyDeviceToDevice));

    cudaFree(c); cudaFree(d); cudaFree(e); cudaFree(OMb); cudaFree(TWb); cudaFree(TWs);
}

// step_radix2 divide_by_Z_on_coset on device buffer P (size bm+sm).
template <typename FieldT, typename T>
static void step_divZ(T *P, uint32_t bm, uint32_t sm, FieldT omega, FieldT g,
                      T mod) {
    const FieldT Z0 = field_pow(g, bm) - FieldT::one();
    const FieldT coset_to_sm_Z0 = field_pow(g, sm) * Z0;
    const FieldT omega_to_sm_Z0 = field_pow(omega, sm) * Z0;
    const FieldT omega_to_2sm = field_pow(omega, 2 * sm);
    // denom[i] = coset_to_sm_Z0 * omega_to_2sm^i - omega_to_sm_Z0
    std::vector<FieldT> denom(bm);
    FieldT elt = FieldT::one();
    for (uint32_t i = 0; i < bm; ++i) {
        denom[i] = coset_to_sm_Z0 * elt - omega_to_sm_Z0;
        elt = elt * omega_to_2sm;
    }
    batch_inverse(denom);
    using HostT = decltype(FieldT::value);
    std::vector<T> h(bm);
    for (uint32_t i = 0; i < bm; ++i) h[i] = (T)(HostT)denom[i].value;
    T *Dinv = upload_vec<T>(h);
    qap_mul_vec_gpu<T>(P, Dinv, bm, mod);
    cudaFree(Dinv);

    const FieldT go = g * omega;
    const FieldT Z1 = (field_pow(go, bm) - FieldT::one()) *
                      (field_pow(go, sm) - field_pow(omega, sm));
    const T Z1inv = (T)(HostT)Z1.inverse().value;
    qap_scale_gpu<T>(P + bm, sm, Z1inv, mod);
}

// step_radix2 GPU witness pipeline (produces coefficients_for_H in dH, m+1).
template <typename FieldT, typename T>
static void witness_H_step(T *dA, T *dB, T *dC, T *dH, T *dScratch, uint32_t m,
                           uint32_t bm, uint32_t sm, FieldT omega,
                           FieldT big_omega, FieldT small_omega, FieldT g, T d1,
                           T d2, T d3, T mod) {
    using HostT = decltype(FieldT::value);
    const FieldT g_inv = g.inverse();
    T *COS = upload_geom<FieldT, T>(g, m);
    T *COSi = upload_geom<FieldT, T>(g_inv, m);
    auto cosetFFT = [&](T *x) {
        qap_mul_vec_gpu<T>(x, COS, m, mod);
        step_FFT<FieldT, T>(x, dScratch, bm, sm, omega, big_omega, small_omega, mod);
    };

    // 1-2. iFFT(aA), iFFT(aB)
    step_iFFT<FieldT, T>(dA, dScratch, bm, sm, omega, big_omega, small_omega, mod);
    step_iFFT<FieldT, T>(dB, dScratch, bm, sm, omega, big_omega, small_omega, mod);

    // 3. ZK patch + add_poly_Z (step variant)
    QAP_CUDA_CHECK(cudaMemset(dH, 0, (size_t)(m + 1) * sizeof(T)));
    qap_lincomb2_gpu<T>(dH, dA, dB, d2, d1, m, mod);
    qap_add_elt_gpu<T>(dH, 0, mod - d3, mod);
    const T c = (T)(((unsigned __int128)d1 * d2) % mod);
    const T om_sm = (T)(HostT)field_pow(omega, sm).value;
    const T c_om = (T)(((unsigned __int128)c * om_sm) % mod);
    // H[m]+=c; H[bm]-=c*om_sm; H[sm]-=c; H[0]+=c*om_sm
    qap_add_elt_gpu<T>(dH, m, c, mod);
    qap_add_elt_gpu<T>(dH, bm, mod - c_om, mod);
    qap_add_elt_gpu<T>(dH, sm, mod - c, mod);
    qap_add_elt_gpu<T>(dH, 0, c_om, mod);

    // 4-5. cosetFFT(aA), cosetFFT(aB)
    cosetFFT(dA); cosetFFT(dB);

    // 6. H_tmp = aA*aB  (in dA)
    qap_pointwise_mul_gpu<T>(dA, dB, m, mod);

    // 7. iFFT(aC), cosetFFT(aC)
    step_iFFT<FieldT, T>(dC, dScratch, bm, sm, omega, big_omega, small_omega, mod);
    cosetFFT(dC);

    // 8. H_tmp -= aC
    qap_pointwise_sub_gpu<T>(dA, dC, m, mod);

    // 9. divide_by_Z_on_coset (step)
    step_divZ<FieldT, T>(dA, bm, sm, omega, g, mod);

    // 10. icosetFFT(H_tmp) = iFFT then multiply_by_coset(g^{-1})
    step_iFFT<FieldT, T>(dA, dScratch, bm, sm, omega, big_omega, small_omega, mod);
    qap_mul_vec_gpu<T>(dA, COSi, m, mod);

    // 11. coefficients_for_H[0..m) += H_tmp
    qap_pointwise_add_gpu<T>(dH, dA, m, mod);

    cudaFree(COS); cudaFree(COSi);
}

// ===================== extended_radix2_large GPU pipeline ===================
// The 60-bit bootstrap field selects extended_radix2_large for non-power-of-2
// sizes. Its m points are ncosets cosets (shift^j * H) of the nroots-th roots
// of unity H. FFT/iFFT factor into: a radix-2 transform of size nroots per
// coset (batched) and a geometric-sequence transform of size ncosets across
// cosets. extended_radix2_large always invokes the geometric domain with
// start=1, which reduces each geometric FFT/iFFT to two NTT-based polynomial
// multiplications plus elementwise scaling by constant tables (precomputed once
// here from the domain's geometric sequences). Everything is exact modular
// arithmetic, so the result is bit-identical to libfqfft.
template <typename FieldT, typename T> struct ExtLargeCtx {
    using HostT = decltype(FieldT::value);
    T mod;
    uint32_t m, nroots, ncosets, lognr, n2, logn2;
    T inv_n2;
    // size-ncosets geometric constant tables (device)
    T *C1, *Zs, *REVZ, *GT, *Tt, *TINV;          // forward geo FFT
    T *PT, *T2, *GTinv, *C3, *REVU;              // inverse geo FFT (Zs reused)
    // size-(n2) poly-mult twiddles + scratch
    T *tw2f, *tw2i, *dU, *dV, *scr2;
    // size-nroots radix2 twiddles
    T *twNf, *twNi;
    // size-m 2D scale tables + coset tables
    T *TWO2D, *TWO2D_INV, *COSET_M, *COSET_M_INV;
    // domain vanishing data
    T *VAN, *VCI;
    // per-call column scratch (size ncosets)
    T *col, *gtmp;

    void build(libfqfft::extended_radix2_large_domain<FieldT> *d) {
        mod = (T)FieldT::mod;
        m = (uint32_t)d->m;
        nroots = (uint32_t)d->nroots;
        ncosets = (uint32_t)d->ncosets;
        lognr = ilog2<FieldT>(nroots);
        const uint32_t n = ncosets;
        // poly-mult size n2 = next pow2 of (2n-1)
        n2 = 1;
        while (n2 < 2 * n - 1) n2 <<= 1;
        logn2 = ilog2<FieldT>(n2);

        const FieldT *geo = d->geo_domain->geometric_sequence.data();
        const FieldT *gt = d->geo_domain->geometric_triangular_sequence.data();

        // u[i] (unsigned), z signed, T, etc. (mirror basis_change.tcc, start=1)
        std::vector<FieldT> u(n), zsg(n), Tt_h(n), PT_h(n), T2_h(n);
        std::vector<FieldT> C1_h(n), GT_h(n), TINV_h(n), GTinv_h(n), C3_h(n);
        u[0] = FieldT::one();
        zsg[0] = FieldT::one();
        Tt_h[0] = FieldT::one();
        PT_h[0] = FieldT::one();
        T2_h[0] = gt[0]; // gt[0]=1
        C1_h[0] = u[0].inverse() * gt[0];
        GT_h[0] = gt[0];
        TINV_h[0] = Tt_h[0].inverse();
        GTinv_h[0] = gt[0].inverse();
        C3_h[0] = gt[0] * u[0].inverse();
        FieldT prevT = FieldT::one();
        for (uint32_t i = 1; i < n; ++i) {
            u[i] = u[i - 1] * geo[i] * (FieldT::one() - geo[i]).inverse();
            FieldT z = u[i] * gt[i].inverse();
            Tt_h[i] = Tt_h[i - 1] * (geo[i] - FieldT::one()).inverse();
            prevT = prevT * (geo[i] - FieldT::one()).inverse();
            PT_h[i] = prevT;
            FieldT t2 = gt[i] * prevT;
            FieldT c1 = u[i].inverse() * gt[i];
            FieldT c3 = gt[i] * u[i].inverse();
            if (i & 1) { z = -z; t2 = -t2; c1 = -c1; c3 = -c3; }
            zsg[i] = z;
            T2_h[i] = t2;
            C1_h[i] = c1;
            C3_h[i] = c3;
            GT_h[i] = gt[i];
            TINV_h[i] = Tt_h[i].inverse();
            GTinv_h[i] = gt[i].inverse();
        }
        std::vector<FieldT> revz(n), revu(n);
        for (uint32_t i = 0; i < n; ++i) { revz[i] = zsg[n - 1 - i]; revu[i] = u[n - 1 - i]; }

        auto up = [&](const std::vector<FieldT> &v) {
            std::vector<T> h(v.size());
            for (size_t i = 0; i < v.size(); ++i) h[i] = (T)(HostT)v[i].value;
            return upload_vec<T>(h);
        };
        C1 = up(C1_h); Zs = up(zsg); REVZ = up(revz); GT = up(GT_h);
        Tt = up(Tt_h); TINV = up(TINV_h);
        PT = up(PT_h); T2 = up(T2_h); GTinv = up(GTinv_h); C3 = up(C3_h);
        REVU = up(revu);

        // poly-mult twiddles for n2
        const FieldT om2 = libff::get_root_of_unity<FieldT>(n2);
        tw2f = upload_geom<FieldT, T>(om2, n2 / 2);
        tw2i = upload_geom<FieldT, T>(om2.inverse(), n2 / 2);
        inv_n2 = (T)(HostT)FieldT((long)n2).inverse().value;
        QAP_CUDA_CHECK(cudaMalloc(&dU, (size_t)n2 * sizeof(T)));
        QAP_CUDA_CHECK(cudaMalloc(&dV, (size_t)n2 * sizeof(T)));
        QAP_CUDA_CHECK(cudaMalloc(&scr2, (size_t)n2 * sizeof(T)));

        // radix2 twiddles for nroots
        twNf = upload_geom<FieldT, T>(d->omega, nroots / 2 ? nroots / 2 : 1);
        twNi = upload_geom<FieldT, T>(d->omega.inverse(),
                                      nroots / 2 ? nroots / 2 : 1);

        // 2D scale tables (size m): TWO2D[j*nr+i]=shift^(j*i),
        // TWO2D_INV[j*nr+i]=inv_nr*shiftinv^(j*i)
        const FieldT shift = d->shift, shiftinv = shift.inverse();
        const FieldT inv_nr = FieldT((long)nroots).inverse();
        std::vector<T> two(m), twoi(m);
        for (uint32_t j = 0; j < ncosets; ++j) {
            FieldT bj = field_pow(shift, j), cur = FieldT::one();
            FieldT bji = field_pow(shiftinv, j), curi = inv_nr;
            for (uint32_t i = 0; i < nroots; ++i) {
                two[(size_t)j * nroots + i] = (T)(HostT)cur.value;
                twoi[(size_t)j * nroots + i] = (T)(HostT)curi.value;
                cur = cur * bj;
                curi = curi * bji;
            }
        }
        TWO2D = upload_vec<T>(two);
        TWO2D_INV = upload_vec<T>(twoi);

        // coset tables (size m): mult_generator^t and its inverse
        const FieldT g = FieldT::multiplicative_generator;
        COSET_M = upload_geom<FieldT, T>(g, m);
        COSET_M_INV = upload_geom<FieldT, T>(g.inverse(), m);

        // vanishing data from the domain
        std::vector<T> van(d->vanishing_polynomial.size()),
            vci(d->vanishing_coset_invs.size());
        for (size_t i = 0; i < van.size(); ++i)
            van[i] = (T)(HostT)d->vanishing_polynomial[i].value;
        for (size_t i = 0; i < vci.size(); ++i)
            vci[i] = (T)(HostT)d->vanishing_coset_invs[i].value;
        VAN = upload_vec<T>(van);
        VCI = upload_vec<T>(vci);

        QAP_CUDA_CHECK(cudaMalloc(&col, (size_t)ncosets * sizeof(T)));
        QAP_CUDA_CHECK(cudaMalloc(&gtmp, (size_t)ncosets * sizeof(T)));
    }

    void destroy() {
        for (T *p : {C1, Zs, REVZ, GT, Tt, TINV, PT, T2, GTinv, C3, REVU, tw2f,
                     tw2i, dU, dV, scr2, twNf, twNi, TWO2D, TWO2D_INV, COSET_M,
                     COSET_M_INV, VAN, VCI, col, gtmp})
            cudaFree(p);
    }

    // out[0..keep) = coeffs[offset..offset+keep) of (A[0..la) * B[0..lb)).
    void full_mult(const T *A, uint32_t la, const T *B, uint32_t lb,
                   uint32_t offset, uint32_t keep, T *out) {
        QAP_CUDA_CHECK(cudaMemset(dU, 0, (size_t)n2 * sizeof(T)));
        QAP_CUDA_CHECK(cudaMemset(dV, 0, (size_t)n2 * sizeof(T)));
        QAP_CUDA_CHECK(cudaMemcpy(dU, A, (size_t)la * sizeof(T), cudaMemcpyDeviceToDevice));
        QAP_CUDA_CHECK(cudaMemcpy(dV, B, (size_t)lb * sizeof(T), cudaMemcpyDeviceToDevice));
        qap_ntt_gpu<T>(dU, tw2f, scr2, logn2, mod);
        qap_ntt_gpu<T>(dV, tw2f, scr2, logn2, mod);
        qap_pointwise_mul_gpu<T>(dU, dV, n2, mod);
        qap_ntt_gpu<T>(dU, tw2i, scr2, logn2, mod);
        qap_scale_gpu<T>(dU, n2, inv_n2, mod);
        QAP_CUDA_CHECK(cudaMemcpy(out, dU + offset, (size_t)keep * sizeof(T),
                                  cudaMemcpyDeviceToDevice));
    }

    // geometric FFT (coeffs->evals on geometric points), in place on `c` (len n).
    void geoFFT(T *c) {
        const uint32_t n = ncosets;
        qap_mul_vec_gpu<T>(c, C1, n, mod);            // c = f = in*C1
        full_mult(REVZ, n, c, n, n - 1, n, gtmp);     // wmid (middle product)
        qap_mul_vec_gpu<T>(gtmp, Zs, n, mod);         // aN
        qap_mul_vec_gpu<T>(gtmp, GT, n, mod);         // g
        full_mult(gtmp, n, Tt, n, 0, n, c);           // conv first n
        qap_mul_vec_gpu<T>(c, TINV, n, mod);          // eval
    }
    // geometric iFFT (evals->coeffs), in place on `c` (len n).
    void geoiFFT(T *c) {
        const uint32_t n = ncosets;
        qap_mul_vec_gpu<T>(c, PT, n, mod);            // W
        full_mult(c, n, T2, n, 0, n, gtmp);           // conv first n
        qap_mul_vec_gpu<T>(gtmp, GTinv, n, mod);      // aN2
        qap_mul_vec_gpu<T>(gtmp, C3, n, mod);         // w2
        full_mult(REVU, n, gtmp, n, n - 1, n, c);     // wmid2
        qap_mul_vec_gpu<T>(c, Zs, n, mod);            // out
    }

    void FFT(T *a) {
        for (uint32_t i = 0; i < nroots; ++i) {
            qap_gather_gpu<T>(a, col, ncosets, nroots, i);
            geoFFT(col);
            qap_scatter_scaled_gpu<T>(a, col, TWO2D, ncosets, nroots, i, mod);
        }
        qap_batched_ntt_gpu<T>(a, twNf, ncosets, nroots, lognr, mod);
    }
    void iFFT(T *a) {
        qap_batched_ntt_gpu<T>(a, twNi, ncosets, nroots, lognr, mod);
        for (uint32_t i = 0; i < nroots; ++i) {
            qap_gather_scaled_gpu<T>(a, TWO2D_INV, col, ncosets, nroots, i, mod);
            geoiFFT(col);
            qap_scatter_gpu<T>(a, col, ncosets, nroots, i);
        }
    }
    void cosetFFT(T *a) { qap_mul_vec_gpu<T>(a, COSET_M, m, mod); FFT(a); }
    void icosetFFT(T *a) { iFFT(a); qap_mul_vec_gpu<T>(a, COSET_M_INV, m, mod); }
    void divide_by_Z(T *P) {
        qap_divz_percoset_gpu<T>(P, VCI, (uint64_t)m, nroots, mod);
    }
    void add_poly_Z(T *H, T coeff) {
        qap_addpolyZ_gpu<T>(H, VAN, coeff, ncosets + 1, nroots, mod);
    }
};

// extended_radix2_large GPU witness pipeline (coefficients_for_H in dH, m+1).
template <typename FieldT, typename T>
static void witness_H_extlarge(ExtLargeCtx<FieldT, T> &cx, T *dA, T *dB, T *dC,
                               T *dH, uint32_t m, T d1, T d2, T d3, T mod) {
    cx.iFFT(dA);
    cx.iFFT(dB);

    QAP_CUDA_CHECK(cudaMemset(dH, 0, (size_t)(m + 1) * sizeof(T)));
    qap_lincomb2_gpu<T>(dH, dA, dB, d2, d1, m, mod);
    qap_add_elt_gpu<T>(dH, 0, mod - d3, mod);
    const T c = (T)(((unsigned __int128)d1 * d2) % mod);
    cx.add_poly_Z(dH, c);

    cx.cosetFFT(dA);
    cx.cosetFFT(dB);
    qap_pointwise_mul_gpu<T>(dA, dB, m, mod);

    cx.iFFT(dC);
    cx.cosetFFT(dC);
    qap_pointwise_sub_gpu<T>(dA, dC, m, mod);

    cx.divide_by_Z(dA);
    cx.icosetFFT(dA);

    qap_pointwise_add_gpu<T>(dH, dA, m, mod);
}

// ===================== generator QAP: Lagrange eval on device ================
// basic_radix2 Lagrange into du[0..mm): du[i] = (Z/mm)·ω^i / (t − ω^i), where
// Z = t^mm − 1. Mirrors _basic_radix2_evaluate_all_lagrange_polynomials. The
// (t−ω^i) inverses are computed per-element by Fermat (a^(p−2)); a field inverse
// is unique, so the result is bit-identical to the CPU batch inverse regardless
// of method. dg/denom are device scratch (size ≥ mm). Returns false iff t is a
// root of unity in the domain (Z==0) — rare for a random t; caller CPU-falls-back.
template <typename FieldT, typename T>
static bool lagrange_basic_gpu(uint32_t mm, FieldT omega, FieldT t, T mod,
                               unsigned long long pm2, T *du, T *dg, T *denom) {
    using HostT = decltype(FieldT::value);
    const FieldT Z = field_pow(t, mm) - FieldT::one();
    if (Z == FieldT::zero()) return false;
    const FieldT l0 = Z * FieldT((long)mm).inverse();
    const T tt = (T)(HostT)t.value, om = (T)(HostT)omega.value;
    qap_geom_gpu<T>(dg, om, mm, mod);                       // dg = ω^i
    QAP_CUDA_CHECK(cudaMemcpy(denom, dg, (size_t)mm * sizeof(T),
                              cudaMemcpyDeviceToDevice));
    qap_rsub_scalar_gpu<T>(denom, tt, mm, mod);             // denom = t − ω^i
    qap_field_pow_gpu<T>(denom, pm2, mm, mod);              // denom = 1/(t−ω^i)
    QAP_CUDA_CHECK(cudaMemcpy(du, dg, (size_t)mm * sizeof(T),
                              cudaMemcpyDeviceToDevice));
    qap_scale_gpu<T>(du, mm, (T)(HostT)l0.value, mod);      // du = l0·ω^i
    qap_pointwise_mul_gpu<T>(du, denom, mm, mod);           // du *= 1/(t−ω^i)
    return true;
}

// step_radix2 Lagrange into du[0..m): mirrors
// step_radix2_domain::evaluate_all_lagrange_polynomials (two basic evals +
// combination). dg/denom (size ≥ big_m) and dsmall (size ≥ small_m) are scratch.
template <typename FieldT, typename T>
static bool lagrange_step_gpu(uint32_t big_m, uint32_t small_m, FieldT omega,
                              FieldT big_omega, FieldT small_omega, FieldT t,
                              T mod, unsigned long long pm2, T *du, T *dg,
                              T *denom, T *dsmall) {
    using HostT = decltype(FieldT::value);
    if (!lagrange_basic_gpu<FieldT, T>(big_m, big_omega, t, mod, pm2, du, dg,
                                       denom))
        return false;
    if (!lagrange_basic_gpu<FieldT, T>(small_m, small_omega,
                                       t * omega.inverse(), mod, pm2, dsmall, dg,
                                       denom))
        return false;
    const FieldT L0 = field_pow(t, small_m) - field_pow(omega, small_m);
    const FieldT omega_to_small_m = field_pow(omega, small_m);
    const FieldT big_omega_to_small_m = field_pow(big_omega, small_m);
    const FieldT L1 = (field_pow(t, big_m) - FieldT::one()) *
                      (field_pow(omega, big_m) - FieldT::one()).inverse();
    // result[i] = inner_big[i]·L0 / (big_omega_to_small_m^i − omega_to_small_m)
    qap_geom_gpu<T>(denom, (T)(HostT)big_omega_to_small_m.value, big_m, mod);
    qap_sub_scalar_gpu<T>(denom, (T)(HostT)omega_to_small_m.value, big_m, mod);
    qap_field_pow_gpu<T>(denom, pm2, big_m, mod);
    qap_scale_gpu<T>(du, big_m, (T)(HostT)L0.value, mod);
    qap_pointwise_mul_gpu<T>(du, denom, big_m, mod);
    // result[big_m + i] = L1 · inner_small[i]
    qap_scale_gpu<T>(dsmall, small_m, (T)(HostT)L1.value, mod);
    QAP_CUDA_CHECK(cudaMemcpy(du + big_m, dsmall, (size_t)small_m * sizeof(T),
                              cudaMemcpyDeviceToDevice));
    return true;
}

} // namespace qap_gpu_detail

// ===========================================================================
// Public entry point.
//
// Computes the prover's pi vector on the GPU and returns it in device memory
// (raw FieldT::value residues). Layout matches prepare_pi_proof:
//   [ full_assignment[num_inputs ..] | d1 d2 d3 | coefficients_for_H[0..m] ]
//
// Returns true and sets *d_pi_out / *proof_dim_out on success (caller owns the
// device buffer and must cudaFree it). Returns false for unsupported domains so
// the caller can fall back to the CPU witness map.
// ===========================================================================
template <typename FieldT>
bool qap_witness_map_gpu(const r1cs_constraint_system<FieldT> &cs,
                         const std::vector<FieldT> &full_assignment,
                         const FieldT &d1, const FieldT &d2, const FieldT &d3,
                         void **d_pi_out, size_t *proof_dim_out) {
    using namespace qap_gpu_detail;
    using HostT = decltype(FieldT::value);
    using T = HostT; // device residue type (uint64_t or unsigned __int128)

    const auto domain = libfqfft::get_evaluation_domain<FieldT>(
        cs.num_constraints() + cs.num_inputs() + 1);
    const uint32_t m = (uint32_t)domain->m;

    auto *basic = dynamic_cast<libfqfft::basic_radix2_domain<FieldT> *>(domain.get());
    auto *step = dynamic_cast<libfqfft::step_radix2_domain<FieldT> *>(domain.get());
    auto *extl = dynamic_cast<libfqfft::extended_radix2_large_domain<FieldT> *>(
        domain.get());
    if (!basic && !step && !extl) return false; // unsupported -> CPU fallback

    const size_t num_inputs = cs.num_inputs();
    const size_t num_constraints = cs.num_constraints();
    const T mod = (T)FieldT::mod;

    // ---- host: evaluate A,B,C on set S (exact CPU semantics) ----
    std::vector<FieldT> aA(m, FieldT::zero()), aB(m, FieldT::zero()),
        aC(m, FieldT::zero());
    for (size_t i = 0; i <= num_inputs; ++i)
        aA[i + num_constraints] =
            (i > 0 ? full_assignment[i - 1] : FieldT::one());
    for (size_t i = 0; i < num_constraints; ++i) {
        aA[i] = aA[i] + cs.constraints[i].a.evaluate(full_assignment);
        aB[i] = aB[i] + cs.constraints[i].b.evaluate(full_assignment);
        aC[i] = aC[i] + cs.constraints[i].c.evaluate(full_assignment);
    }

    // ---- upload aA, aB, aC ----
    std::vector<T> hA(m), hB(m), hC(m);
    for (uint32_t i = 0; i < m; ++i) {
        hA[i] = (T)(HostT)aA[i].value;
        hB[i] = (T)(HostT)aB[i].value;
        hC[i] = (T)(HostT)aC[i].value;
    }
    T *dA = upload_vec<T>(hA), *dB = upload_vec<T>(hB), *dC = upload_vec<T>(hC);
    T *dScratch = nullptr; QAP_CUDA_CHECK(cudaMalloc(&dScratch, (size_t)m * sizeof(T)));
    T *dH = nullptr; QAP_CUDA_CHECK(cudaMalloc(&dH, (size_t)(m + 1) * sizeof(T)));

    const T t1 = (T)(HostT)d1.value, t2 = (T)(HostT)d2.value,
            t3 = (T)(HostT)d3.value;

    ExtLargeCtx<FieldT, T> cx{};
    if (basic) {
        witness_H_basic<FieldT, T>(dA, dB, dC, dH, dScratch, m, basic->omega,
                                   FieldT::multiplicative_generator, t1, t2, t3,
                                   mod);
    } else if (step) {
        witness_H_step<FieldT, T>(dA, dB, dC, dH, dScratch, m,
                                  (uint32_t)step->big_m, (uint32_t)step->small_m,
                                  step->omega, step->big_omega, step->small_omega,
                                  FieldT::multiplicative_generator, t1, t2, t3,
                                  mod);
    } else {
        cx.build(extl);
        witness_H_extlarge<FieldT, T>(cx, dA, dB, dC, dH, m, t1, t2, t3, mod);
    }
    QAP_CUDA_CHECK(cudaGetLastError());
    QAP_CUDA_CHECK(cudaDeviceSynchronize());
    if (extl) cx.destroy();

    // ---- assemble pi on device ----
    const size_t num_variables = cs.num_variables();
    const size_t num_ABC = num_variables - num_inputs;
    const size_t H_len = (size_t)m + 1;
    const size_t proof_dim = num_ABC + 3 + H_len;

    T *d_pi = nullptr;
    QAP_CUDA_CHECK(cudaMalloc(&d_pi, proof_dim * sizeof(T)));
    // head = witness tail + d1,d2,d3
    std::vector<T> head(num_ABC + 3);
    for (size_t i = 0; i < num_ABC; ++i)
        head[i] = (T)(HostT)full_assignment[i + num_inputs].value;
    head[num_ABC] = t1; head[num_ABC + 1] = t2; head[num_ABC + 2] = t3;
    QAP_CUDA_CHECK(cudaMemcpy(d_pi, head.data(), head.size() * sizeof(T),
                              cudaMemcpyHostToDevice));
    QAP_CUDA_CHECK(cudaMemcpy(d_pi + num_ABC + 3, dH, H_len * sizeof(T),
                              cudaMemcpyDeviceToDevice));

    cudaFree(dA); cudaFree(dB); cudaFree(dC); cudaFree(dScratch); cudaFree(dH);

    *d_pi_out = (void *)d_pi;
    *proof_dim_out = proof_dim;
    return true;
}

// ===========================================================================
// GPU QAP instance map (generator side).
//
// Mirrors the gen_q_mat loop `qap_inst = r1cs_to_qap_instance_map_with_evaluation
// (cs, t_s[i])` for every query point, filling A_qs/B_qs/C_qs/H_qs/Z_s. The
// domain-specific Lagrange vector u = evaluate_all_lagrange_polynomials(t) is
// computed on the CPU (exact for every domain), so the GPU part — the sparse
// A/B/C evaluation, which is the loop's dominant cost — is domain-INDEPENDENT:
// build CSC(A/B/C) once and gather out[v] = init[v] + Σ u[row]*coeff per column.
// Field addition is exact/commutative, so results are bit-identical to the CPU
// path. Opt-in (caller gates on HECATE_QAP_GEN_GPU). Throws on CUDA failure.
// ===========================================================================
template <typename FieldT>
bool qap_instance_map_gpu(const r1cs_constraint_system<FieldT> &cs,
                          const std::vector<FieldT> &t_s,
                          std::vector<std::vector<FieldT>> &A_qs,
                          std::vector<std::vector<FieldT>> &B_qs,
                          std::vector<std::vector<FieldT>> &C_qs,
                          std::vector<std::vector<FieldT>> &H_qs,
                          std::vector<FieldT> &Z_s) {
    using namespace qap_gpu_detail;
    using HostT = decltype(FieldT::value);
    using T = HostT;

    const auto domain = libfqfft::get_evaluation_domain<FieldT>(
        cs.num_constraints() + cs.num_inputs() + 1);
    const uint32_t m = (uint32_t)domain->m;
    const size_t num_constraints = cs.num_constraints();
    const size_t num_inputs = cs.num_inputs();
    const uint32_t num_out = (uint32_t)(cs.num_variables() + 1);
    const T mod = (T)FieldT::mod;
    const size_t query_num = t_s.size();

    // ---- build CSC(A/B/C) on host (transpose of the per-constraint term lists);
    // one column per variable index, entries carry (row=constraint, coeff). ----
    auto build_csc = [&](int which, std::vector<uint64_t> &col_ptr,
                         std::vector<uint32_t> &row_idx, std::vector<T> &coeff) {
        auto terms_of = [&](size_t i) -> const auto & {
            return which == 0 ? cs.constraints[i].a.terms
                 : which == 1 ? cs.constraints[i].b.terms
                              : cs.constraints[i].c.terms;
        };
        col_ptr.assign(num_out + 1, 0);
        for (size_t i = 0; i < num_constraints; ++i)
            for (const auto &tm : terms_of(i)) col_ptr[tm.index + 1]++;
        for (uint32_t v = 0; v < num_out; ++v) col_ptr[v + 1] += col_ptr[v];
        const uint64_t nnz = col_ptr[num_out];
        row_idx.resize(nnz);
        coeff.resize(nnz);
        std::vector<uint64_t> pos(col_ptr.begin(), col_ptr.end());
        for (size_t i = 0; i < num_constraints; ++i)
            for (const auto &tm : terms_of(i)) {
                const uint64_t p = pos[tm.index]++;
                row_idx[p] = (uint32_t)i;
                coeff[p] = (T)(HostT)tm.coeff.value;
            }
    };
    const bool prof = std::getenv("HECATE_QAP_GEN_GPU_PROF") != nullptr;
    auto now = [] { return std::chrono::steady_clock::now(); };
    auto ms = [](auto a, auto b) {
        return std::chrono::duration_cast<std::chrono::milliseconds>(b - a).count();
    };
    double t_csc = 0, t_lag = 0, t_up = 0, t_ker = 0, t_dn = 0, t_ht = 0;

    auto _c0 = now();
    std::vector<uint64_t> cpA, cpB, cpC;
    std::vector<uint32_t> riA, riB, riC;
    std::vector<T> coA, coB, coC;
    build_csc(0, cpA, riA, coA);
    build_csc(1, cpB, riB, coB);
    build_csc(2, cpC, riC, coC);
    t_csc = ms(_c0, now());

    // ---- upload CSC once + allocate per-query scratch ----
    auto up64 = [](const std::vector<uint64_t> &v) { return upload_vec<uint64_t>(v); };
    auto up32 = [](const std::vector<uint32_t> &v) {
        return v.empty() ? (uint32_t *)nullptr : upload_vec<uint32_t>(v);
    };
    auto upT = [](const std::vector<T> &v) {
        return v.empty() ? (T *)nullptr : upload_vec<T>(v);
    };
    uint64_t *dcpA = up64(cpA), *dcpB = up64(cpB), *dcpC = up64(cpC);
    uint32_t *driA = up32(riA), *driB = up32(riB), *driC = up32(riC);
    T *dcoA = upT(coA), *dcoB = upT(coB), *dcoC = upT(coC);
    T *du = nullptr, *dinit = nullptr, *dA = nullptr, *dB = nullptr, *dC = nullptr;
    QAP_CUDA_CHECK(cudaMalloc(&du, (size_t)m * sizeof(T)));
    QAP_CUDA_CHECK(cudaMalloc(&dinit, (size_t)num_out * sizeof(T)));
    QAP_CUDA_CHECK(cudaMalloc(&dA, (size_t)num_out * sizeof(T)));
    QAP_CUDA_CHECK(cudaMalloc(&dB, (size_t)num_out * sizeof(T)));
    QAP_CUDA_CHECK(cudaMalloc(&dC, (size_t)num_out * sizeof(T)));

    // GPU Lagrange dispatch: basic_radix2 / step_radix2 compute u on-device
    // (skipping the CPU evaluate_all_lagrange + the m-element upload — the real
    // bottleneck). Other domains keep the CPU path. pm2 = p-2 (Fermat inverse
    // exponent) fits u64 for the native ≤60-bit SNARK fields used here.
    auto *dbasic =
        dynamic_cast<libfqfft::basic_radix2_domain<FieldT> *>(domain.get());
    auto *dstep =
        dynamic_cast<libfqfft::step_radix2_domain<FieldT> *>(domain.get());
    const unsigned long long pm2 = (unsigned long long)FieldT::mod - 2ull;
    T *dg = nullptr, *denom = nullptr, *dsmall = nullptr;
    if (dbasic || dstep) {
        QAP_CUDA_CHECK(cudaMalloc(&dg, (size_t)m * sizeof(T)));
        QAP_CUDA_CHECK(cudaMalloc(&denom, (size_t)m * sizeof(T)));
        QAP_CUDA_CHECK(cudaMalloc(&dsmall, (size_t)m * sizeof(T)));
    }

    A_qs.resize(query_num); B_qs.resize(query_num); C_qs.resize(query_num);
    H_qs.resize(query_num); Z_s.resize(query_num);
    std::vector<T> hu(m), hinit(num_out), hA(num_out), hB(num_out), hC(num_out);
    std::vector<T> hslice(num_inputs + 1);  // At-init seed downloaded from device u

    for (size_t q = 0; q < query_num; ++q) {
        const FieldT &t = t_s[q];
        const FieldT Zt = domain->compute_vanishing_polynomial(t);
        auto _l0 = now();
        // Compute u on-device for basic/step domains (returns false only if t is
        // a domain root of unity — negligible for random t — then CPU-fall-back).
        bool gpu_lag = false;
        if (dbasic)
            gpu_lag = lagrange_basic_gpu<FieldT, T>(m, dbasic->omega, t, mod, pm2,
                                                    du, dg, denom);
        else if (dstep)
            gpu_lag = lagrange_step_gpu<FieldT, T>(
                (uint32_t)dstep->big_m, (uint32_t)dstep->small_m, dstep->omega,
                dstep->big_omega, dstep->small_omega, t, mod, pm2, du, dg, denom,
                dsmall);
        if (gpu_lag) {
            QAP_CUDA_CHECK(cudaDeviceSynchronize());
            t_lag += ms(_l0, now());
            auto _u0 = now();
            // At-init seed lives in device u at [num_constraints, +num_inputs].
            QAP_CUDA_CHECK(cudaMemcpy(hslice.data(), du + num_constraints,
                                      (size_t)(num_inputs + 1) * sizeof(T),
                                      cudaMemcpyDeviceToHost));
            for (uint32_t v = 0; v < num_out; ++v) hinit[v] = (T)0;
            for (size_t v = 0; v <= num_inputs; ++v) hinit[v] = hslice[v];
            QAP_CUDA_CHECK(cudaMemcpy(dinit, hinit.data(),
                                      (size_t)num_out * sizeof(T),
                                      cudaMemcpyHostToDevice));
            t_up += ms(_u0, now());
        } else {
            const std::vector<FieldT> u =
                domain->evaluate_all_lagrange_polynomials(t);
            for (uint32_t j = 0; j < m; ++j) hu[j] = (T)(HostT)u[j].value;
            t_lag += ms(_l0, now());
            auto _u0 = now();
            QAP_CUDA_CHECK(cudaMemcpy(du, hu.data(), (size_t)m * sizeof(T),
                                      cudaMemcpyHostToDevice));
            for (uint32_t v = 0; v < num_out; ++v) hinit[v] = (T)0;
            for (size_t v = 0; v <= num_inputs; ++v)
                hinit[v] = hu[num_constraints + v];
            QAP_CUDA_CHECK(cudaMemcpy(dinit, hinit.data(),
                                      (size_t)num_out * sizeof(T),
                                      cudaMemcpyHostToDevice));
            t_up += ms(_u0, now());
        }

        auto _k0 = now();
        qap_spmv_col_gpu<T>(dcpA, driA, dcoA, du, dinit, dA, num_out, mod);
        qap_spmv_col_gpu<T>(dcpB, driB, dcoB, du, nullptr, dB, num_out, mod);
        qap_spmv_col_gpu<T>(dcpC, driC, dcoC, du, nullptr, dC, num_out, mod);
        QAP_CUDA_CHECK(cudaGetLastError());
        QAP_CUDA_CHECK(cudaDeviceSynchronize());
        t_ker += ms(_k0, now());

        auto _d0 = now();
        QAP_CUDA_CHECK(cudaMemcpy(hA.data(), dA, (size_t)num_out * sizeof(T), cudaMemcpyDeviceToHost));
        QAP_CUDA_CHECK(cudaMemcpy(hB.data(), dB, (size_t)num_out * sizeof(T), cudaMemcpyDeviceToHost));
        QAP_CUDA_CHECK(cudaMemcpy(hC.data(), dC, (size_t)num_out * sizeof(T), cudaMemcpyDeviceToHost));
        A_qs[q].resize(num_out); B_qs[q].resize(num_out); C_qs[q].resize(num_out);
        for (uint32_t v = 0; v < num_out; ++v) {
            A_qs[q][v].value = hA[v];
            B_qs[q][v].value = hB[v];
            C_qs[q][v].value = hC[v];
        }
        t_dn += ms(_d0, now());
        // Ht = [t^0 .. t^m] (exact powers; sequential like the CPU path).
        auto _h0 = now();
        H_qs[q].resize((size_t)m + 1);
        FieldT ti = FieldT::one();
        for (uint32_t j = 0; j <= m; ++j) { H_qs[q][j] = ti; ti *= t; }
        Z_s[q] = Zt;
        t_ht += ms(_h0, now());
    }
    if (prof)
        std::fprintf(stderr,
                     "[qap_gpu] instance-map profile (ms): csc=%.0f lagrange=%.0f "
                     "upload=%.0f kernel=%.0f download=%.0f ht=%.0f (m=%u nnzA=%zu "
                     "nnzB=%zu nnzC=%zu num_out=%u)\n",
                     t_csc, t_lag, t_up, t_ker, t_dn, t_ht, m, coA.size(),
                     coB.size(), coC.size(), num_out);

    for (void *p : {(void *)dcpA, (void *)dcpB, (void *)dcpC, (void *)driA,
                    (void *)driB, (void *)driC, (void *)dcoA, (void *)dcoB,
                    (void *)dcoC, (void *)du, (void *)dinit, (void *)dA,
                    (void *)dB, (void *)dC, (void *)dg, (void *)denom,
                    (void *)dsmall})
        if (p) cudaFree(p);

    // Bit-identity self-check vs the CPU instance map (HECATE_QAP_GEN_GPU_CHECK).
    // Recomputes the reference on CPU and aborts on any divergence — the same
    // rigor as test_qap_gpu for the witness map. Opt-in (it re-does the CPU work).
    if (std::getenv("HECATE_QAP_GEN_GPU_CHECK")) {
        auto eq = [](const std::vector<FieldT> &a, const std::vector<FieldT> &b) {
            if (a.size() != b.size()) return false;
            for (size_t i = 0; i < a.size(); ++i)
                if (!(a[i] == b[i])) return false;
            return true;
        };
        for (size_t q = 0; q < query_num; ++q) {
            const auto ref =
                r1cs_to_qap_instance_map_with_evaluation(cs, t_s[q]);
            if (!eq(A_qs[q], ref.At) || !eq(B_qs[q], ref.Bt) ||
                !eq(C_qs[q], ref.Ct) || !eq(H_qs[q], ref.Ht) ||
                !(Z_s[q] == ref.Zt)) {
                std::fprintf(stderr,
                             "[qap_gpu] INSTANCE-MAP MISMATCH at query %zu "
                             "(GPU != CPU)\n",
                             q);
                std::abort();
            }
        }
        std::fprintf(stderr,
                     "[qap_gpu] instance-map self-check PASS (%zu queries, "
                     "GPU == CPU bit-identical)\n",
                     query_num);
    }
    return true;
}

} // namespace libsnark

#endif // __QAP_GPU_HPP__
