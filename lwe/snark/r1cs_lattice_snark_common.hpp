#ifndef __R1CS_LATTICE_SNARK_COMMON__
#define __R1CS_LATTICE_SNARK_COMMON__

#include "lwe/container/extension.hpp"
#include "lwe/lwe.hpp"
#include "lwe/lwe_params.hpp"
#include "lwe/randomness/aes.hpp"
#include "lwe/randomness/prg.hpp"

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <libff/algebra/curves/public_params.hpp>
#include <libfqfft/evaluation_domain/get_evaluation_domain.hpp>
#include <libsnark/reductions/r1cs_to_qap/r1cs_to_qap.hpp>
#include <libsnark/relations/constraint_satisfaction_problems/r1cs/examples/r1cs_examples.hpp>
#include <libsnark/relations/constraint_satisfaction_problems/r1cs/r1cs.hpp>

namespace libsnark {

    /* TYPE ALIAS DEFINITONS */

    template <typename cpT> using Rq_T = typename cpT::Rq_type;

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

        std::vector<uint64_t> hEnc((size_t)2 * rows * cdim);
        int rc = launch_crs_encrypt(hS.data(), hUV.data(), keys, rows, n, cdim,
                                    mask_lo, mask_hi, hEnc.data());
        if (rc != 0)
            throw std::runtime_error("launch_crs_encrypt failed");

        for (uint64_t i = 0; i < rows; i++)
            for (uint32_t out = 0; out < cdim; out++) {
                const size_t idx = ((size_t)i * cdim + out) * 2;
                unsigned __int128 lo = hEnc[idx];
                unsigned __int128 hi = hEnc[idx + 1];
                enc_qs[i][out].value =
                    (decltype(enc_qs[i][out].value))((hi << 64) | lo);
            }
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

        std::vector<uint64_t> hEnc((size_t)4 * rows * cdim);
        int rc = launch_crs_encrypt_big(hS.data(), hUV.data(), keys, rows, n, cdim,
                                        m[0], m[1], m[2], m[3], hEnc.data());
        if (rc != 0)
            throw std::runtime_error("launch_crs_encrypt_big failed");

        for (uint64_t i = 0; i < rows; i++)
            for (uint32_t out = 0; out < cdim; out++) {
                const size_t idx = ((size_t)i * cdim + out) * 4;
                auto &v = enc_qs[i][out].value;  // T256
                v.w[0] = hEnc[idx + 0]; v.w[1] = hEnc[idx + 1];
                v.w[2] = hEnc[idx + 2]; v.w[3] = hEnc[idx + 3];
            }
    }

    template <typename ppT, typename cpT, class Params>
    static inline void encrypt_query_matrix(
        const LWE::secret_key<Rq_T<cpT>, libff::Fr<ppT>, Params> &sk,
        const r1cs_lattice_snark_query_matrix<ppT, Params::pt_dim> &q_mat,
        const LWERandomness::AES_KEY &crs_aes_key,
        std::vector<LWE::Vector<Rq_T<cpT>, Params::pt_dim + Params::tau>>
            &enc_qs) {
        // Default = CPU. HECATE_CRS_GPU=1 uses the GPU keygen.
        // HECATE_CRS_GPU_VALIDATE=1 runs BOTH (noiseless) and reports byte-exact
        // agreement, keeping the CPU result (the correctness gate before trusting
        // the GPU path on a multi-hour keygen). Both are native-params only.
        const bool validate = std::getenv("HECATE_CRS_GPU_VALIDATE") != nullptr;
        const bool want_gpu = std::getenv("HECATE_CRS_GPU") != nullptr;
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
                encrypt_query_matrix_gpu<ppT, cpT, Params>(sk, q_mat, crs_aes_key,
                                                           enc_qs);
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
