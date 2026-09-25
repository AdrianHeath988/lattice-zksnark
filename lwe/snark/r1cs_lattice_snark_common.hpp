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
#include <mutex>
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
            const cudaError_t e = cudaGetDeviceCount(&n);
            const bool good = (e == cudaSuccess && n > 0);
            // SAY WHY. This one predicate decides between the GPU QAP instance map
            // and a CPU fallback that materialises the dense A/B/C/H query matrices
            // in host RAM -- ~560 GB for an 81,920-constraint board, versus ~73 GB
            // and 100x faster on the GPU path. When it silently returned false the
            // only symptom was a keygen that ran for half an hour and ate the host,
            // with the GPU flags all set and looking correct.
            std::fprintf(stderr,
                         "[vfhe-gpu] cudaGetDeviceCount -> %s (devices=%d) "
                         "CUDA_VISIBLE_DEVICES=%s => GPU paths %s\n",
                         cudaGetErrorString(e), n,
                         std::getenv("CUDA_VISIBLE_DEVICES")
                             ? std::getenv("CUDA_VISIBLE_DEVICES")
                             : "(unset)",
                         good ? "ENABLED" : "DISABLED (CPU fallback)");

            // HECATE_GPU_STRICT=1 -- REFUSE TO FALL BACK.
            //
            // Reaching here with good==false means a GPU path was REQUESTED (every
            // call site is `vfhe_env_on(FLAG, true) && vfhe_gpu_available()`, and &&
            // short-circuits, so a deliberate FLAG=0 never gets here) and the device
            // probe failed. Continuing silently swaps in a CPU path that is ~15x
            // slower on the prove side and materialises far larger host buffers on
            // the keygen side -- which for a BENCHMARK quietly corrupts the number
            // being measured, and is very hard to spot after the fact.
            //
            // This is not hypothetical. On 2026-08-20 one of eight GPUs wedged
            // (RmInitAdapter failed, after a burst of prover SIGFPEs) and the run
            // kept dispatching to the now-nonexistent index 7. Those jobs silently
            // took the CPU path: measured 199s -> 3046s of prove time for the SAME
            // buffer, and they were the jobs reporting verify=FAIL. The run looked
            // like it was merely slow for hours.
            //
            // Opt-in (default off) so a genuinely CPU-only host still works.
            if (!good && vfhe_env_on("HECATE_GPU_STRICT", false)) {
                std::fprintf(stderr,
                    "[vfhe-gpu] CPU FALLBACK DISABLED (HECATE_GPU_STRICT=1).\n"
                    "  A GPU path was requested but no usable CUDA device was found.\n"
                    "  cudaGetDeviceCount: %s (devices=%d), CUDA_VISIBLE_DEVICES=%s\n"
                    "  Refusing to continue on the CPU path: it is ~15x slower and\n"
                    "  would silently misreport any benchmark taken from this run.\n"
                    "  Check `nvidia-smi -L` -- the live device COUNT, not a\n"
                    "  remembered one -- and that CUDA_VISIBLE_DEVICES names an\n"
                    "  index that still exists. Unset HECATE_GPU_STRICT to allow the\n"
                    "  CPU path.\n",
                    cudaGetErrorString(e), n,
                    std::getenv("CUDA_VISIBLE_DEVICES")
                        ? std::getenv("CUDA_VISIBLE_DEVICES")
                        : "(unset)");
                std::fflush(stderr);
                // _Exit, not throw: the keygen call site wraps its GPU attempt in a
                // try/catch that falls back to the CPU generator, so an exception
                // here would be swallowed and do the very thing this flag forbids.
                std::_Exit(3);
            }
            return good;
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

    // Page-locks the enc_qs host buffer so the per-proof host->device upload runs
    // at PCIe rate instead of through the driver's pageable staging path. Measured
    // on an L40S (PCIe 4 x16) with a 6 GB std::vector buffer: 5.5 GB/s pageable
    // vs 26.9 GB/s registered, and cudaHostRegister itself costs 0.19 s ONCE.
    // The upload was 2.2 s per proof at index 7.27M (the single largest GPU-side
    // cost in the prover), so this is ~2 s per proof for a one-off 0.2 s.
    //
    // No copy site changes: cudaMemcpy detects a registered range on its own.
    //
    // OWNED BY THE CRS, not a side table keyed on the pointer. A buffer that is
    // freed while still registered is not merely a leak: the driver keeps the old
    // physical pages pinned under that virtual range, and if a later allocation
    // lands on the same addresses a cudaMemcpy from it DMAs the OLD pages -- a
    // silently wrong proof, not a fault. Tying the registration to the vector's
    // owner (declared after enc_qs, so destroyed before it) closes that.
    // The CRS's assignment operators release() before the vector is replaced for
    // the same reason. Copies start unregistered (they own a different buffer);
    // moves carry the registration, since a moved vector keeps its buffer.
    //
    // HECATE_CRS_PIN=0 disables. Registration failure is non-fatal: the copy
    // simply stays pageable, exactly as before.
    class enc_qs_host_pin {
    public:
        enc_qs_host_pin() = default;
        enc_qs_host_pin(const enc_qs_host_pin &) noexcept {}
        enc_qs_host_pin(enc_qs_host_pin &&o) noexcept
            : ptr_(o.ptr_), bytes_(o.bytes_) {
            o.ptr_ = nullptr;
            o.bytes_ = 0;
        }
        enc_qs_host_pin &operator=(const enc_qs_host_pin &) noexcept {
            release();
            return *this;
        }
        enc_qs_host_pin &operator=(enc_qs_host_pin &&o) noexcept {
            if (this != &o) {
                release();
                ptr_ = o.ptr_;
                bytes_ = o.bytes_;
                o.ptr_ = nullptr;
                o.bytes_ = 0;
            }
            return *this;
        }
        ~enc_qs_host_pin() { release(); }

        // Register [p, p+bytes). Idempotent for the same range; re-registers if
        // the buffer moved (a reload into the same CRS object).
        void ensure(const void *p, std::size_t bytes) {
            if (!p || !bytes) return;
            // The CRS is shared by every prover thread on a shape; serialise so
            // the first proofs under a key do not race to register the same range.
            std::lock_guard<std::mutex> lk(mtx_);
            if (ptr_ == p && bytes_ == bytes) return;
            release_locked();
            if (!vfhe_env_on("HECATE_CRS_PIN", true)) return;
            const auto t0 = std::chrono::steady_clock::now();
            const cudaError_t e = cudaHostRegister(
                const_cast<void *>(p), bytes, cudaHostRegisterPortable);
            if (e != cudaSuccess) {
                cudaGetLastError();  // clear the sticky error; stay pageable
                std::fprintf(stderr,
                             "[crs-pin] cudaHostRegister(%.1f GB) failed: %s "
                             "-- enc_qs uploads stay pageable\n",
                             bytes / 1e9, cudaGetErrorString(e));
                return;
            }
            ptr_ = p;
            bytes_ = bytes;
            std::fprintf(stderr, "[crs-pin] pinned enc_qs %.1f GB in %.2f s\n",
                         bytes / 1e9,
                         std::chrono::duration<double>(
                             std::chrono::steady_clock::now() - t0)
                             .count());
        }
        void release() noexcept {
            std::lock_guard<std::mutex> lk(mtx_);
            release_locked();
        }
        bool pinned() const { return ptr_ != nullptr; }

    private:
        void release_locked() noexcept {
            if (!ptr_) return;
            // At process exit the runtime may already be torn down; the error is
            // harmless and there is nothing to do about it.
            cudaHostUnregister(const_cast<void *>(ptr_));
            cudaGetLastError();
            ptr_ = nullptr;
            bytes_ = 0;
        }
        std::mutex mtx_;  // not moved: each object guards its own registration
        const void *ptr_ = nullptr;
        std::size_t bytes_ = 0;
    };

    template <typename T, uint64_t LEN>
    inline void
    enc_qs_pin_host(enc_qs_host_pin &pin,
                    const std::vector<LWE::Vector<T, LEN>> &enc_qs) {
        pin.ensure(enc_qs_flat(enc_qs), enc_qs.size() * sizeof(LWE::Vector<T, LEN>));
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

    // Fresh AES key from /dev/urandom. The stream stays OPEN: it used to be
    // close()d after the first read, so every later call in the process (the
    // prover's per-proof masking key; the 2nd..nth key of a multi-key genonly
    // pass) read from a closed stream, failed silently, and expanded whatever
    // bytes were on the stack. Serialised so parallel callers stay safe.
    inline void genAES_key(LWERandomness::AES_KEY *_key) {
        static std::ifstream urandom("/dev/urandom", std::ios::binary);
        static std::mutex mtx;
        LWERandomness::byte buffer[LWERandomness::AES_KEY_BYTES];
        {
            std::lock_guard<std::mutex> lk(mtx);
            urandom.read(reinterpret_cast<char *>(buffer),
                         LWERandomness::AES_KEY_BYTES);
            if (!urandom || urandom.gcount() != LWERandomness::AES_KEY_BYTES)
                throw std::runtime_error("genAES_key: /dev/urandom read failed");
        }
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
        libff::enter_block("CRS encrypt host prep (resize + S_T + uv)");
        enc_qs.resize(rows);

        {  // hS/hUV scope: their teardown is timed as its own block below
        std::vector<uint64_t> hS((size_t)2 * cdim * n);
        for (uint32_t out = 0; out < cdim; out++)
            for (uint32_t k = 0; k < n; k++) {
                unsigned __int128 v = (unsigned __int128)sk.S_T[out][k].value;
                const size_t idx = ((size_t)out * n + k) * 2;
                hS[idx] = (uint64_t)v;
                hS[idx + 1] = (uint64_t)(v >> 64);
            }

        std::vector<uint64_t> hUV((size_t)rows * cdim);
        // Rows are independent (Tv is per-row, hUV writes are disjoint) and this
        // was the serial CPU phase between the GPU QAP map and the GPU encrypt:
        // ~35M rows x tau*pt_dim field mults for a 2^14 relin key, on ONE core
        // while the GPU sat idle. Plan-gen already exports OMP_NUM_THREADS per
        // child (cores/gpus); nothing here used it before.
        #pragma omp parallel for schedule(static)
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
        libff::leave_block("CRS encrypt host prep (resize + S_T + uv)");

        // The kernel's h_enc layout ([row*cdim + out] -> (lo,hi)) is exactly
        // enc_qs's own memory, so it D2H's straight into the CRS -- no staging
        // buffer, no unflatten pass. enc_qs is flat from the moment it is written.
        libff::enter_block("CRS encrypt (GPU launch)");
        int rc = launch_crs_encrypt(hS.data(), hUV.data(), keys, rows, n, cdim,
                                    mask_lo, mask_hi, enc_qs_flat(enc_qs));
        libff::leave_block("CRS encrypt (GPU launch)");
        if (rc != 0)
            throw std::runtime_error("launch_crs_encrypt failed");
        libff::enter_block("CRS encrypt host teardown (free uv/S_T)");
        }
        libff::leave_block("CRS encrypt host teardown (free uv/S_T)");
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
        // Rows are independent (Tv is per-row, hUV writes are disjoint) and this
        // was the serial CPU phase between the GPU QAP map and the GPU encrypt:
        // ~35M rows x tau*pt_dim field mults for a 2^14 relin key, on ONE core
        // while the GPU sat idle. Plan-gen already exports OMP_NUM_THREADS per
        // child (cores/gpus); nothing here used it before.
        #pragma omp parallel for schedule(static)
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

        // Gaussian noise pass (kept on host). Measured at N=2^14 (35M rows x 53):
        // 62 s of the 227 s "Generating CRS and VK" block when run serially off
        // the one global PRG -- more than the GPU kernel's host-visible overhead
        // and second only to the kernel itself. The noise is fresh randomness
        // that nothing regenerates (unlike a_vec), so each thread may draw from
        // its own /dev/urandom-seeded PRG; the Gaussian table it indexes is
        // shared read-only. Row updates are disjoint.
        libff::enter_block("CRS Gaussian noise pass");
        if (vfhe_env_on("HECATE_CRS_NOISE_SERIAL", false)) {
            // Diagnostic: the original serial loop off the global prg/dg.
            for (auto &row : enc_qs) {
                LWE::Vector<Rq_T<cpT>, Params::pt_dim + Params::tau> ev;
                ev.discrete_gaussian();
                row += ev * Params::p_int;
            }
        } else {
            const int64_t nrows = static_cast<int64_t>(enc_qs.size());
            #pragma omp parallel
            {
                LWERandomness::AES_KEY tkey;
                genAES_key(&tkey);  // serialised inside; fresh key per thread
                LWERandomness::PseudoRandomGenerator tprg(tkey);
                #pragma omp for schedule(static)
                for (int64_t i = 0; i < nrows; i++) {
                    LWE::Vector<Rq_T<cpT>, Params::pt_dim + Params::tau> ev;
                    ev.discrete_gaussian(tprg);
                    enc_qs[i] += ev * Params::p_int;
                }
            }
        }
        libff::leave_block("CRS Gaussian noise pass");
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
