#ifndef __R1CS_LATTICE_SNARK_COMMON__
#define __R1CS_LATTICE_SNARK_COMMON__

#include "lwe/container/extension.hpp"
#include "lwe/lwe.hpp"
#include "lwe/lwe_params.hpp"
#include "lwe/randomness/aes.hpp"
#include "lwe/randomness/prg.hpp"

#include <cuda_runtime.h>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <type_traits>
#include <vector>
#include <libff/algebra/curves/public_params.hpp>
#include <libfqfft/evaluation_domain/get_evaluation_domain.hpp>
#include <libsnark/reductions/r1cs_to_qap/r1cs_to_qap.hpp>
#include <libsnark/relations/constraint_satisfaction_problems/r1cs/examples/r1cs_examples.hpp>
#include <libsnark/relations/constraint_satisfaction_problems/r1cs/r1cs.hpp>

namespace libsnark {

    /* TYPE ALIAS DEFINITONS */

    template <typename cpT> using Rq_T = typename cpT::Rq_type;

    /* TRI-STATE ENV FLAGS
     *
     * Optimizations (GPU keygen, GPU QAP, the CRS cache, the resident-key
     * scheduler) ship ENABLED; the env var exists to turn them OFF for
     * benchmarking and debugging. So the contract everywhere is:
     *
     *   unset            -> `def` (the shipped default, normally true)
     *   "0"/"false"/"off"/"no"/""  -> false
     *   anything else (incl. "1")  -> true
     *
     * Accepting "1" as ON keeps every existing script, README example and
     * keygen shell wrapper working unchanged after the defaults flipped.
     */
    inline bool vfhe_env_on(const char *name, bool def) {
        const char *e = std::getenv(name);
        if (!e || !*e) return def;
        return !(std::strcmp(e, "0") == 0 || std::strcmp(e, "false") == 0 ||
                 std::strcmp(e, "off") == 0 || std::strcmp(e, "no") == 0);
    }

    // Is a CUDA device actually usable? Probed once. The GPU paths default ON, so
    // this is what keeps a CPU-only host working without setting anything: no
    // device (or no driver) simply routes back to the CPU implementation.
    inline bool vfhe_gpu_available() {
        static const bool ok = [] {
            int n = 0;
            return cudaGetDeviceCount(&n) == cudaSuccess && n > 0;
        }();
        return ok;
    }

    /* FLAT enc_qs ACCESS
     *
     * enc_qs IS the flat layout the GPU wants -- it never needs to be built.
     * LWE::Vector<T,LEN> holds exactly one std::array<T,LEN>; Ring<u128,mod> and
     * RingBig<u256,q_log> hold exactly one `value` (mod/MASK/prg/dg are static).
     * So std::vector<LWE::Vector<Rq_T,cdim>> is byte-identical to the row-major
     * [index][cdim] buffer accumulate_c_vec{,_big} reads -- little-endian u128 is
     * (lo,hi), and uint256 is already w[0..3] in order. crs_wvec/crs_rvec write and
     * read those same bytes, so the layout is flat end to end: keygen writes into
     * it, the .crs file stores it, the loader memcpy's it back, and the prover
     * cudaMemcpy's straight out of it. Re-flattening per proof rebuilt up to 6 GB
     * of bit-identical data on the CPU for every proof sharing a key.
     *
     * The static_asserts are the guard: if a member is ever added to Vector or to
     * the ring types, this stops compiling rather than silently uploading garbage.
     */
    template <typename T, uint64_t LEN>
    inline constexpr void enc_qs_check_flat() {
        static_assert(std::is_standard_layout<LWE::Vector<T, LEN>>::value,
                      "enc_qs row must be standard-layout to alias as u64[]");
        static_assert(sizeof(LWE::Vector<T, LEN>) == LEN * sizeof(T),
                      "enc_qs row has padding -- not the flat device layout");
        static_assert(sizeof(T) == sizeof(decltype(T::value)),
                      "ring element is more than its value -- not flat");
        static_assert(sizeof(T) % sizeof(uint64_t) == 0,
                      "ring element is not a whole number of u64 limbs");
    }

    // u64 limbs per ring element (2 for the native u128 path, 4 for the u256 path).
    template <typename T>
    inline constexpr std::size_t enc_qs_limbs() {
        return sizeof(T) / sizeof(uint64_t);
    }

    template <typename T, uint64_t LEN>
    inline const uint64_t *
    enc_qs_flat(const std::vector<LWE::Vector<T, LEN>> &enc_qs) {
        enc_qs_check_flat<T, LEN>();
        return enc_qs.empty()
                   ? nullptr
                   : reinterpret_cast<const uint64_t *>(enc_qs.data());
    }

    template <typename T, uint64_t LEN>
    inline uint64_t *enc_qs_flat(std::vector<LWE::Vector<T, LEN>> &enc_qs) {
        enc_qs_check_flat<T, LEN>();
        return enc_qs.empty() ? nullptr
                              : reinterpret_cast<uint64_t *>(enc_qs.data());
    }

    template <typename ppT, uint32_t pt_dim>
    using r1cs_lattice_snark_query_matrix =
        std::vector<LWE::Vector<libff::Fr<ppT>, pt_dim>>;

    template <typename ppT, typename cpT, class Params>
    class r1cs_lattice_snark_proof {
    public:
        LWE::ciphertext<Rq_T<cpT>, libff::Fr<ppT>, Params> response;
        r1cs_lattice_snark_proof() = default;
        explicit r1cs_lattice_snark_proof(
            LWE::ciphertext<Rq_T<cpT>, libff::Fr<ppT>, Params> &&response)
            : response(std::move(response)) {}
    };

    template <typename ppT>
    std::vector<ppT> reject_sampling_S(const r1cs_constraint_system<ppT> &cs,
                                       int sample_num) {
        const auto domain = libfqfft::get_evaluation_domain<ppT>(
            cs.num_constraints() + cs.num_inputs() + 1);
        std::vector<ppT> res(sample_num);
        for (int i = 0; i < sample_num; i++) {
            ppT _res = ppT::random_element();
            while (domain->compute_vanishing_polynomial(_res) == ppT::zero())
                _res = ppT::random_element();
            res[i] = _res;
        }
        return res;
    }

    template <typename T, typename... RT>
    inline void public_params_init(LWERandomness::PseudoRandomGenerator *prg,
                                   LWERandomness::DiscreteGaussian *dg) {
        T::prg = prg;
        T::dg = dg;
        T::init_public_params();
        if constexpr (sizeof...(RT) > 0)
            public_params_init<RT...>(prg, dg);
    }

    inline void genAES_key(LWERandomness::AES_KEY *_key) {
        static std::ifstream urandom("/dev/urandom", std::ios::binary);
        LWERandomness::byte buffer[LWERandomness::AES_KEY_BYTES];
        urandom.read(reinterpret_cast<char *>(buffer),
                     LWERandomness::AES_KEY_BYTES);
        urandom.close();
        LWERandomness::AES_128_Key_Expansion(buffer, _key);
    }

    // GPU launcher (crs_gen.cu) — flat host-array interface. See crs_gen.cu.
    extern "C" int launch_crs_encrypt(const uint64_t *h_S_T, const uint64_t *h_uv,
                                      const uint32_t *h_keys, uint64_t rows,
                                      uint32_t n, uint32_t cdim, uint64_t mask_lo,
                                      uint64_t mask_hi, uint64_t *h_enc);

    // Big-int (256-bit RingBig) GPU CRS keygen launcher (crs_gen.cu).
    extern "C" int launch_crs_encrypt_big(const uint64_t *h_S_T,
                                          const uint64_t *h_uv,
                                          const uint32_t *h_keys, uint64_t rows,
                                          uint32_t n, uint32_t cdim, uint64_t m0,
                                          uint64_t m1, uint64_t m2, uint64_t m3,
                                          uint64_t *h_enc);

    // Original CPU encrypt loop (NOISELESS): every q_mat row -> c_vec. a_vec is
    // drawn sequentially from the crs_aes_key PRG (counter i*n+k for row i), which
    // the prover regenerates identically.
    template <typename ppT, typename cpT, class Params>
    static inline void encrypt_query_matrix_cpu(
        const LWE::secret_key<Rq_T<cpT>, libff::Fr<ppT>, Params> &sk,
        const r1cs_lattice_snark_query_matrix<ppT, Params::pt_dim> &q_mat,
        const LWERandomness::AES_KEY &crs_aes_key,
        std::vector<LWE::Vector<Rq_T<cpT>, Params::pt_dim + Params::tau>> &enc_qs) {
        enc_qs.resize(q_mat.size());
        auto *temp_prg = new LWERandomness::PseudoRandomGenerator(crs_aes_key);
        auto *temp_dg = new LWERandomness::DiscreteGaussian(
            Params::width, LWE::expand, *temp_prg);
        auto *original_prg = ppT::prg;
        auto *original_dg = ppT::dg;
        public_params_init<ppT, cpT>(temp_prg, temp_dg);
        int counter = 0;
        for (const auto &row : q_mat) {
            auto encrypted_query =
                LWE::encrypt<Rq_T<cpT>, libff::Fr<ppT>, Params>(sk, row, false);
            enc_qs[counter++] = std::move(encrypted_query.c_vec);
        }
        public_params_init<ppT, cpT>(original_prg, original_dg);
        delete temp_dg;
        delete temp_prg;
    }

    // GPU encrypt loop (NOISELESS): flattens S_T (cdim x n) and uv[i] = [pt[i] |
    // T_mat*pt[i]] (lift_to is a residue copy), regenerates a_vec on-GPU as
    // AES(i*n+k)&mask, computes c_vec[out] = lift(uv[out]) + S_T[out]·a_vec mod
    // 2^q_log, and unflattens into enc_qs. Native (uint128) params only.
    template <typename ppT, typename cpT, class Params>
    static inline void encrypt_query_matrix_gpu(
        const LWE::secret_key<Rq_T<cpT>, libff::Fr<ppT>, Params> &sk,
        const r1cs_lattice_snark_query_matrix<ppT, Params::pt_dim> &q_mat,
        const LWERandomness::AES_KEY &crs_aes_key,
        std::vector<LWE::Vector<Rq_T<cpT>, Params::pt_dim + Params::tau>> &enc_qs) {
        const uint32_t n = Params::n;
        const uint32_t cdim = Params::pt_dim + Params::tau;
        const uint64_t rows = q_mat.size();
        enc_qs.resize(rows);

        std::vector<uint64_t> hS((size_t)2 * cdim * n);
        for (uint32_t out = 0; out < cdim; out++)
            for (uint32_t k = 0; k < n; k++) {
                unsigned __int128 v = (unsigned __int128)sk.S_T[out][k].value;
                const size_t idx = ((size_t)out * n + k) * 2;
                hS[idx] = (uint64_t)v;
                hS[idx + 1] = (uint64_t)(v >> 64);
            }

        std::vector<uint64_t> hUV((size_t)rows * cdim);
        for (uint64_t i = 0; i < rows; i++) {
            auto Tv = sk.T_mat * q_mat[i];  // Vector<Fr, tau>
            for (uint32_t out = 0; out < Params::pt_dim; out++)
                hUV[i * cdim + out] = (uint64_t)q_mat[i][out].value;
            for (uint32_t t = 0; t < Params::tau; t++)
                hUV[i * cdim + Params::pt_dim + t] = (uint64_t)Tv[t].value;
        }

        const uint32_t *keys =
            reinterpret_cast<const uint32_t *>(crs_aes_key.rd_key);
        const uint64_t q_log = Params::q_log;
        const uint64_t mask_lo = (q_log >= 64) ? ~0ull : ((1ull << q_log) - 1);
        const uint64_t mask_hi =
            (q_log > 64) ? ((1ull << (q_log - 64)) - 1) : 0ull;

        // The kernel's h_enc layout ([row*cdim + out] -> (lo,hi)) is exactly
        // enc_qs's own memory, so it D2H's straight into the CRS -- no staging
        // buffer, no unflatten pass. enc_qs is flat from the moment it is written.
        int rc = launch_crs_encrypt(hS.data(), hUV.data(), keys, rows, n, cdim,
                                    mask_lo, mask_hi, enc_qs_flat(enc_qs));
        if (rc != 0)
            throw std::runtime_error("launch_crs_encrypt failed");
    }

    // Big-int (256-bit RingBig) GPU encrypt loop (NOISELESS). Mirrors
    // encrypt_query_matrix_gpu but every S_T / enc element is 4 uint64, the mask
    // is the 4-limb 2^q_log-1, and a_vec on-GPU is two AES blocks per element
    // (matching RingBig::random_element). Byte-exact with encrypt_query_matrix_cpu.
    template <typename ppT, typename cpT, class Params>
    static inline void encrypt_query_matrix_gpu_big(
        const LWE::secret_key<Rq_T<cpT>, libff::Fr<ppT>, Params> &sk,
        const r1cs_lattice_snark_query_matrix<ppT, Params::pt_dim> &q_mat,
        const LWERandomness::AES_KEY &crs_aes_key,
        std::vector<LWE::Vector<Rq_T<cpT>, Params::pt_dim + Params::tau>> &enc_qs) {
        const uint32_t n = Params::n;
        const uint32_t cdim = Params::pt_dim + Params::tau;
        const uint64_t rows = q_mat.size();
        enc_qs.resize(rows);

        std::vector<uint64_t> hS((size_t)4 * cdim * n);
        for (uint32_t out = 0; out < cdim; out++)
            for (uint32_t k = 0; k < n; k++) {
                const auto &v = sk.S_T[out][k].value;  // T256
                const size_t idx = ((size_t)out * n + k) * 4;
                hS[idx + 0] = v.w[0]; hS[idx + 1] = v.w[1];
                hS[idx + 2] = v.w[2]; hS[idx + 3] = v.w[3];
            }

        std::vector<uint64_t> hUV((size_t)rows * cdim);
        for (uint64_t i = 0; i < rows; i++) {
            auto Tv = sk.T_mat * q_mat[i];
            for (uint32_t out = 0; out < Params::pt_dim; out++)
                hUV[i * cdim + out] = (uint64_t)q_mat[i][out].value;
            for (uint32_t t = 0; t < Params::tau; t++)
                hUV[i * cdim + Params::pt_dim + t] = (uint64_t)Tv[t].value;
        }

        const uint32_t *keys =
            reinterpret_cast<const uint32_t *>(crs_aes_key.rd_key);
        const uint64_t q_log = Params::q_log;
        uint64_t m[4] = {0, 0, 0, 0};
        for (uint64_t b = 0; b < q_log && b < 256; b++) m[b >> 6] |= (1ull << (b & 63));

        // As in the native path: h_enc ([row*cdim + out] -> w0..w3) is enc_qs's own
        // memory, so the D2H lands directly in the CRS. No staging, no unflatten.
        int rc = launch_crs_encrypt_big(hS.data(), hUV.data(), keys, rows, n, cdim,
                                        m[0], m[1], m[2], m[3], enc_qs_flat(enc_qs));
        if (rc != 0)
            throw std::runtime_error("launch_crs_encrypt_big failed");
    }

    template <typename ppT, typename cpT, class Params>
    static inline void encrypt_query_matrix(
        const LWE::secret_key<Rq_T<cpT>, libff::Fr<ppT>, Params> &sk,
        const r1cs_lattice_snark_query_matrix<ppT, Params::pt_dim> &q_mat,
        const LWERandomness::AES_KEY &crs_aes_key,
        std::vector<LWE::Vector<Rq_T<cpT>, Params::pt_dim + Params::tau>>
            &enc_qs) {
        // Default = GPU keygen (~20x); HECATE_CRS_GPU=0 forces the CPU path.
        // HECATE_CRS_GPU_VALIDATE=1 runs BOTH (noiseless) and reports byte-exact
        // agreement, keeping the CPU result (the correctness gate before trusting
        // the GPU path on a multi-hour keygen). Both are native-params only.
        //
        // GPU is only ATTEMPTED when a usable device is present, and a launch
        // failure falls back to the CPU path instead of propagating. That safety
        // net is what makes on-by-default correct: this used to be opt-in, so a
        // throw here could only happen to someone who had explicitly asked for
        // the GPU. Now every CPU-only host reaches this code, and keygen must
        // still complete for them.
        const bool validate = std::getenv("HECATE_CRS_GPU_VALIDATE") != nullptr;
        const bool want_gpu = vfhe_env_on("HECATE_CRS_GPU", true) && vfhe_gpu_available();
        if constexpr (!cpT::is_big) {
            if (validate) {
                std::vector<
                    LWE::Vector<Rq_T<cpT>, Params::pt_dim + Params::tau>>
                    gpu_enc;
                encrypt_query_matrix_gpu<ppT, cpT, Params>(sk, q_mat, crs_aes_key,
                                                           gpu_enc);
                encrypt_query_matrix_cpu<ppT, cpT, Params>(sk, q_mat, crs_aes_key,
                                                           enc_qs);
                const uint32_t cdim = Params::pt_dim + Params::tau;
                size_t bad = 0, total = 0;
                for (size_t i = 0; i < enc_qs.size(); i++)
                    for (uint32_t out = 0; out < cdim; out++) {
                        total++;
                        if (gpu_enc[i][out].value != enc_qs[i][out].value) bad++;
                    }
                std::fprintf(stderr,
                             "[crs_gen] GPU vs CPU noiseless enc_qs: %zu / %zu "
                             "mismatch (keeping CPU)\n",
                             bad, total);
            } else if (want_gpu) {
                try {
                    encrypt_query_matrix_gpu<ppT, cpT, Params>(sk, q_mat,
                                                               crs_aes_key, enc_qs);
                } catch (const std::exception &ex) {
                    // launch_crs_encrypt failed (OOM, no driver, ...). enc_qs may
                    // be half-written, so drop it and redo on the CPU: a slower
                    // keygen beats aborting the trusted setup.
                    std::fprintf(stderr,
                                 "[crs_gen] GPU keygen failed (%s) -- falling back "
                                 "to CPU. Set HECATE_CRS_GPU=0 to skip this.\n",
                                 ex.what());
                    enc_qs.clear();
                    encrypt_query_matrix_cpu<ppT, cpT, Params>(sk, q_mat,
                                                               crs_aes_key, enc_qs);
                }
            } else {
                encrypt_query_matrix_cpu<ppT, cpT, Params>(sk, q_mat, crs_aes_key,
                                                           enc_qs);
            }
        } else {
            // Big-int (256-bit) path: GPU CRS keygen under HECATE_CRS_GPU (the
            // big-int CPU keygen is the dominant cost, ~29 min for a 60-bit-q0 op).
            if (validate) {
                std::vector<
                    LWE::Vector<Rq_T<cpT>, Params::pt_dim + Params::tau>>
                    gpu_enc;
                encrypt_query_matrix_gpu_big<ppT, cpT, Params>(sk, q_mat,
                                                               crs_aes_key, gpu_enc);
                encrypt_query_matrix_cpu<ppT, cpT, Params>(sk, q_mat, crs_aes_key,
                                                           enc_qs);
                const uint32_t cdim = Params::pt_dim + Params::tau;
                size_t bad = 0, total = 0;
                for (size_t i = 0; i < enc_qs.size(); i++)
                    for (uint32_t out = 0; out < cdim; out++) {
                        total++;
                        if (gpu_enc[i][out].value != enc_qs[i][out].value) bad++;
                    }
                std::fprintf(stderr,
                             "[crs_gen big] GPU vs CPU noiseless enc_qs: %zu / %zu "
                             "mismatch (keeping CPU)\n",
                             bad, total);
            } else if (want_gpu) {
                encrypt_query_matrix_gpu_big<ppT, cpT, Params>(sk, q_mat,
                                                               crs_aes_key, enc_qs);
            } else {
                encrypt_query_matrix_cpu<ppT, cpT, Params>(sk, q_mat, crs_aes_key,
                                                           enc_qs);
            }
        }

        // Gaussian noise pass (kept on host; uses the restored ppT::dg).
        for (auto &row : enc_qs) {
            LWE::Vector<Rq_T<cpT>, Params::pt_dim + Params::tau> ev;
            ev.discrete_gaussian();
            row += ev * Params::p_int;
        }
    }

    template <typename pdpT, typename ppT, uint32_t pt_dim>
    inline void expand_queries(
        const r1cs_lattice_snark_query_matrix<pdpT, pt_dim> &orig_query,
        r1cs_lattice_snark_query_matrix<ppT, pt_dim * 2> &expand_query) {
        size_t orig_query_len = orig_query.size();
        expand_query.resize(orig_query_len * 2);
        for (size_t i = 0; i < orig_query_len; i++) {
            for (size_t j = 0; j < pt_dim; j++) {
                expand_query[2 * i][2 * j] = orig_query[i][j].c0;
                expand_query[2 * i][2 * j + 1] = orig_query[i][j].c1;
                expand_query[2 * i + 1][2 * j] =
                    orig_query[i][j].c1 * libff::Fr<pdpT>::non_residue;
                expand_query[2 * i + 1][2 * j + 1] = orig_query[i][j].c0;
            }
        }
    }

    template <typename ppT>
    inline void fp_shrink(const std::vector<ppT> &fp_respond,
                          std::vector<libsnark::Extension<ppT>> &shrink_res) {
        shrink_res.resize(fp_respond.size() / 2);
        for (size_t i = 0; i < shrink_res.size(); i++) {
            shrink_res[i].c0 = fp_respond[i * 2];
            shrink_res[i].c1 = fp_respond[i * 2 + 1];
        }
    }
}

#endif
