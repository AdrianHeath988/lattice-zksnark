#ifndef __R1CS_LATTICE_SNARK__
#define __R1CS_LATTICE_SNARK__

#include "r1cs_lattice_snark_common.hpp"
#include "qap_gpu.hpp"
#include <cuda_runtime.h>
#include "vfhe/CudaProfile.h"  // HECATE_CUDA_PROF: time the blocking CUDA calls
#include <algorithm>
#include <cstdio>
#include <functional>
#include <mutex>
#include <stdexcept>
// --- ADD THIS AT THE TOP OF r1cs_lattice_snark.hpp ---
// Standard C++ declarations that hide the CUDA launch syntax
void launch_accumulate_c_vec_kernel(
    const void* d_enc_qs, 
    const uint64_t* d_pi, 
    void* d_out_c_vec, 
    int index, 
    int total_coeffs, 
    int blocks_c, 
    int threads_per_block);

// void copy_aes_keys_to_constant(const uint32_t* host_keys);

void launch_generate_a_vec_kernel(
    const uint64_t* d_pi,
    void* d_out_a_vec,
    int index,
    int total_coeffs_a,
    int blocks_a,
    int threads_per_block,
    uint64_t mod_mask_lo,
    uint64_t mod_mask_hi,
    const uint32_t* d_aes_round_keys);

// Big-int (256-bit) prove launchers (proof.cu).
void launch_accumulate_c_vec_big(const uint64_t* d_enc_qs, const uint64_t* d_pi,
                                 uint64_t* d_out_c_vec, int index, int total_coeffs,
                                 int blocks_c, int threads_per_block);
void launch_generate_a_vec_big(const uint64_t* d_pi, uint64_t* d_out_a_vec, int index,
                               int total_coeffs_a, int blocks_a, int threads_per_block,
                               uint64_t m0, uint64_t m1, uint64_t m2, uint64_t m3,
                               const uint32_t* d_aes_round_keys);

// Split a_vec launchers (proof.cu): materialise A once, then MAC each trace's pi
// against it. A is held TRANSPOSED, [coeff][row] -- see proof.cu. A chunk covers
// rows [row0, row0+rows). See a_matrix_cache below for why the fused launchers
// above are kept.
void launch_generate_a_matrix(void* d_A, int rows, uint64_t row0, int total_coeffs_a,
                              uint64_t mod_mask_lo, uint64_t mod_mask_hi,
                              const uint32_t* d_aes_round_keys,
                              cudaStream_t stream = 0);
void launch_accumulate_a_vec_kernel(const void* d_A, const uint64_t* d_pi,
                                    void* d_out_a_vec, int rows, int blocks_a,
                                    int threads_per_block, cudaStream_t stream = 0);
void launch_generate_a_matrix_big(uint64_t* d_A, int rows, uint64_t row0,
                                  int total_coeffs_a, uint64_t m0, uint64_t m1,
                                  uint64_t m2, uint64_t m3,
                                  const uint32_t* d_aes_round_keys,
                                  cudaStream_t stream = 0);
void launch_accumulate_a_vec_big(const uint64_t* d_A, const uint64_t* d_pi,
                                 uint64_t* d_out_a_vec, int rows, int blocks_a,
                                 int threads_per_block, cudaStream_t stream = 0);

// Forward declarations for your CUDA kernels
template <typename DataType>
extern __global__ void generate_and_accumulate_a_vec(
    const uint32_t* aes_round_keys, 
    const DataType* pi, 
    DataType* out_a_vec, 
    int index, 
    int elements_per_poly, 
    DataType modulus);

template <typename DataType>
extern __global__ void accumulate_c_vec(
    const DataType* enc_qs, 
    const DataType* pi, 
    DataType* out_c_vec, 
    int index, 
    int total_coeffs, 
    DataType modulus);

// GPU Error checking macro
#define cudaCheckError() { \
    cudaError_t e=cudaGetLastError(); \
    if(e!=cudaSuccess) { \
        printf("CUDA Error %s:%d: %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); \
        exit(EXIT_FAILURE); \
    } \
}

namespace libsnark {

    /* DATA STRUCTURE DEFINITIONS */

    template <typename ppT, typename cpT, class Params>
    class r1cs_lattice_snark_crs {
    public:
        r1cs_constraint_system<libff::Fr<ppT>> constraint_system;
        std::vector<LWE::Vector<Rq_T<cpT>, Params::pt_dim + Params::tau>>
            enc_qs;
        LWE::public_parameter<Rq_T<cpT>, Params> public_parameter;
        LWERandomness::AES_KEY crs_aes_key{};

        r1cs_lattice_snark_crs() = default;
        r1cs_lattice_snark_crs &
        operator=(const r1cs_lattice_snark_crs &) = default;
        r1cs_lattice_snark_crs(const r1cs_lattice_snark_crs &) = default;
        r1cs_lattice_snark_crs(r1cs_lattice_snark_crs &&) noexcept = default;

        explicit r1cs_lattice_snark_crs(
            const r1cs_constraint_system<libff::Fr<ppT>> &cs,
            std::vector<LWE::Vector<Rq_T<cpT>, Params::pt_dim + Params::tau>>
                &&enc_q,
            LWE::public_parameter<Rq_T<cpT>, Params> &&pp,
            const LWERandomness::AES_KEY &_key)
            : constraint_system(cs), enc_qs(std::move(enc_q)),
              public_parameter(std::move(pp)), crs_aes_key{} {
            std::copy_n(_key.rd_key, 15, this->crs_aes_key.rd_key);
        }
    };

    template <typename ppT, typename cpT, class Params>
    class r1cs_lattice_snark_verification_key {
    public:
        LWE::secret_key<Rq_T<cpT>, libff::Fr<ppT>, Params> sk;
        std::vector<libff::Fr_vector<ppT>> A_prefix, B_prefix, C_prefix;
        libff::Fr_vector<ppT> Z_s;

        r1cs_lattice_snark_verification_key() = default;
        r1cs_lattice_snark_verification_key(
            LWE::secret_key<Rq_T<cpT>, libff::Fr<ppT>, Params> &&sk_,
            std::vector<libff::Fr_vector<ppT>> &&A_prefix_,
            std::vector<libff::Fr_vector<ppT>> &&B_prefix_,
            std::vector<libff::Fr_vector<ppT>> &&C_prefix_,
            libff::Fr_vector<ppT> &&Z_s_)
            : sk(std::move(sk_)), A_prefix(std::move(A_prefix_)),
              B_prefix(std::move(B_prefix_)), C_prefix(std::move(C_prefix_)),
              Z_s(std::move(Z_s_)) {}
    };

    template <typename ppT, typename Params>
    inline void
    gen_q_mat(const r1cs_constraint_system<libff::Fr<ppT>> &cs,
              r1cs_lattice_snark_query_matrix<ppT, Params::pt_dim> &q_mat,
              std::vector<libff::Fr_vector<ppT>> &A_qs,
              std::vector<libff::Fr_vector<ppT>> &B_qs,
              std::vector<libff::Fr_vector<ppT>> &C_qs,
              std::vector<libff::Fr_vector<ppT>> &A_prefix,
              std::vector<libff::Fr_vector<ppT>> &B_prefix,
              std::vector<libff::Fr_vector<ppT>> &C_prefix,
              std::vector<libff::Fr_vector<ppT>> &H_qs,
              libff::Fr_vector<ppT> &Z_s) {

        A_qs.resize(Params::query_num);
        B_qs.resize(Params::query_num);
        C_qs.resize(Params::query_num);
        H_qs.resize(Params ::query_num);

        Z_s.resize(Params::query_num);
        A_prefix.resize(Params::query_num);
        B_prefix.resize(Params::query_num);
        C_prefix.resize(Params::query_num);

        libff::enter_block("Generating QAP queries");
        const size_t num_inputs = cs.num_inputs();

        auto t_s = reject_sampling_S(cs, Params::query_num);

        // GPU QAP instance map (ON by default; HECATE_QAP_GEN_GPU=0 disables):
        // fills A_qs/B_qs/C_qs/H_qs/Z_s bit-identically to the per-query CPU call
        // below, but runs the sparse A/B/C evaluation (this loop's dominant cost)
        // on the GPU. The Lagrange vector is still computed on the CPU (exact for
        // every domain), so it is domain-independent. qap_instance_map_gpu returns
        // false on any unsupported domain or failure, and the loop below then does
        // the CPU work -- that built-in fallback is why this is safe on by default.
        bool gpu_qap = false;
        if (vfhe_env_on("HECATE_QAP_GEN_GPU", true) && vfhe_gpu_available()) {
            gpu_qap = qap_instance_map_gpu<libff::Fr<ppT>>(
                cs, t_s, A_qs, B_qs, C_qs, H_qs, Z_s);
        }
        for (size_t i = 0; i < Params::query_num; i++) {
            if (!gpu_qap) {
                qap_instance_evaluation<libff::Fr<ppT>> qap_inst =
                    r1cs_to_qap_instance_map_with_evaluation(cs, t_s[i]);
                A_qs[i] = std::move(qap_inst.At);
                B_qs[i] = std::move(qap_inst.Bt);
                C_qs[i] = std::move(qap_inst.Ct);
                H_qs[i] = std::move(qap_inst.Ht);
                Z_s[i] = qap_inst.Zt;
            }
            A_prefix[i].reserve(num_inputs + 1);
            std::copy_n(std::begin(A_qs[i]), num_inputs + 1,
                        std::begin(A_prefix[i]));
            B_prefix[i].reserve(num_inputs + 1);
            std::copy_n(std::begin(B_qs[i]), num_inputs + 1,
                        std::begin(B_prefix[i]));
            C_prefix[i].reserve(num_inputs + 1);
            std::copy_n(std::begin(C_qs[i]), num_inputs + 1,
                        std::begin(C_prefix[i]));
        }

        const uint64_t ABC_rows = A_qs.begin()->size() - num_inputs - 1;
        const uint64_t H_rows = H_qs.begin()->size();
        const uint64_t rows = ABC_rows + 3 + H_rows;
        q_mat.resize(rows);
        for (uint64_t i = 0; i < ABC_rows; i++) {
            for (uint64_t j = 0; j < Params::query_num; j++) {
                q_mat[i][j * LWE::query_size] = A_qs[j][i + 1 + num_inputs];
                q_mat[i][j * LWE::query_size + 1] = B_qs[j][i + 1 + num_inputs];
                q_mat[i][j * LWE::query_size + 2] = C_qs[j][i + 1 + num_inputs];
            }
        }
        for (uint64_t i = 0; i < 3; i++)
            for (uint64_t j = 0; j < Params::query_num; j++)
                q_mat[ABC_rows + i][i + LWE::query_size * j] = Z_s[j];
        for (uint64_t i = 0; i < Params::query_num; i++)
            for (uint64_t j = 0; j < H_rows; j++)
                q_mat[ABC_rows + 3 + j][i * LWE::query_size + 3] = H_qs[i][j];
        libff::leave_block("Generating QAP queries");
    }

    template <typename ppT, typename cpT, class Params>
    void r1cs_lattice_snark_generator(
        const r1cs_constraint_system<libff::Fr<ppT>> &cs,
        r1cs_lattice_snark_crs<ppT, cpT, Params> &crs,
        r1cs_lattice_snark_verification_key<ppT, cpT, Params> &vk) {

        libff::enter_block("Generating LWE secret key");
        auto sk_pp = LWE::keygen<Rq_T<cpT>, libff::Fr<ppT>, Params>();
        libff::leave_block("Generating LWE secret key");

        std::vector<libff::Fr_vector<ppT>> A_qs, B_qs, C_qs, H_qs;
        libff::Fr_vector<ppT> Zs;
        std::vector<libff::Fr_vector<ppT>> A_prefix, B_prefix, C_prefix;
        r1cs_lattice_snark_query_matrix<ppT, Params::pt_dim> q_mat;
        gen_q_mat<ppT, Params>(cs, q_mat, A_qs, B_qs, C_qs, A_prefix, B_prefix,
                               C_prefix, H_qs, Zs);

        libff::enter_block("Generating CRS and VK");
        //TDOD: make this faster 

        LWERandomness::AES_KEY _crs_aes_key;
        genAES_key(&_crs_aes_key);

        std::vector<LWE::Vector<Rq_T<cpT>, Params::pt_dim + Params::tau>> dummy;
        crs = r1cs_lattice_snark_crs<ppT, cpT, Params>(
            cs, std::move(dummy), std::move(sk_pp.second), _crs_aes_key);
        vk = r1cs_lattice_snark_verification_key<ppT, cpT, Params>(
            std::move(sk_pp.first), std::move(A_prefix), std::move(B_prefix),
            std::move(C_prefix), std::move(Zs));
        encrypt_query_matrix<ppT, cpT, Params>(vk.sk, q_mat, crs.crs_aes_key,
                                               crs.enc_qs);

        libff::leave_block("Generating CRS and VK");
    }

    template <typename ppT>
    inline void prepare_pi_proof(const qap_witness<libff::Fr<ppT>> &qap_wit,
                                 libff::Fr_vector<ppT> &pi) {
        libff::enter_block("Prepare pi proof");
        size_t num_inputs = qap_wit.num_inputs();
        size_t num_ABC_coeffs =
            qap_wit.coefficients_for_ABCs.size() - num_inputs;
        size_t proof_dim =
            num_ABC_coeffs + 3 + qap_wit.coefficients_for_H.size();
        pi.resize(proof_dim);

        for (size_t i = 0; i < num_ABC_coeffs; i++)
            pi[i] = qap_wit.coefficients_for_ABCs[i + num_inputs];
        pi[num_ABC_coeffs] = qap_wit.d1;
        pi[num_ABC_coeffs + 1] = qap_wit.d2;
        pi[num_ABC_coeffs + 2] = qap_wit.d3;
        std::copy(std::begin(qap_wit.coefficients_for_H),
                  std::end(qap_wit.coefficients_for_H),
                  std::begin(pi) + num_ABC_coeffs + 3);
        libff::leave_block("Prepare pi proof");
    }

    // ---------------------------------------------------------------------
    // RESIDENT a_vec MATRIX (A)
    //
    // A[i][j] = AES_k(i*n + j) & mask is a pure function of the CRS AES key and
    // the dimensions -- no trace input enters it. The fused kernel therefore
    // re-derived all index*n AES blocks on EVERY prove call, even though the
    // prover server holds one CRS across a run of traces (the region path
    // measured 1.97-4.05 proofs per CRS load, and the per-op path more), so that
    // AES work was repeated verbatim per trace. With the kernel split in proof.cu
    // we materialise A on the first proof under a key, MAC every later trace's pi
    // straight out of it, and free it the moment the key or the shape changes.
    //
    // The constraint is size: A is index*n*sizeof(elem) bytes, which for a large
    // QAP (index ~ 7M, n ~ 1.7k) is hundreds of GB -- past any device. So the
    // cache is taken ONLY when the whole matrix fits in free device memory (and
    // under HECATE_A_CACHE_MB, if set). Otherwise we keep the original fused
    // generate+MAC kernel, which stores no A at all; for a single proof the split
    // has nothing to win there anyway (the fused kernel already generates each
    // element exactly once, one block per coefficient) and would only add a
    // global-memory round trip. HECATE_A_CACHE=0 forces the fused path.
    struct a_matrix_cache {
        std::uint32_t keys[44] = {};
        std::uint64_t mask[4] = {};
        std::size_t rows = 0, coeffs = 0, elem_bytes = 0;
        void *d_A = nullptr;
        std::size_t capacity = 0;  // bytes actually allocated at d_A

        bool matches(const std::uint32_t *k, const std::uint64_t *m, std::size_t r,
                     std::size_t c, std::size_t eb) const {
            return d_A != nullptr && rows == r && coeffs == c && elem_bytes == eb &&
                   std::memcmp(keys, k, sizeof(keys)) == 0 &&
                   std::memcmp(mask, m, sizeof(mask)) == 0;
        }
        // Forget WHAT is in the buffer, keep the buffer. A key change invalidates
        // the contents but not the allocation, and at 9 GB the cudaMalloc costs
        // ~0.29 s -- six times the refill itself -- so re-taking it per key would
        // dominate everything the cache saves.
        void invalidate() {
            rows = coeffs = elem_bytes = 0;
            std::memset(keys, 0, sizeof(keys));
            std::memset(mask, 0, sizeof(mask));
        }
        void release() {
            // SYNCHRONIZE BEFORE FREEING. release() is reachable from CRS eviction,
            // which is driven by the residency ceiling and is NOT ordered against
            // the proof that is currently reading A. Freeing under a live kernel
            // gave an "illegal memory access" at the next cudaCheckError, reported
            // against whatever proof happened to be running (see the eviction
            // immediately preceding every observed fault). The sync costs nothing
            // next to a 9 GB regeneration and makes the free safe from any caller.
            if (d_A) {
                cudaDeviceSynchronize();
                cudaFree(d_A);
            }
            d_A = nullptr;
            capacity = 0;
            invalidate();
        }
    };

    // Serialises the A-matrix section across threads that share this process's CUDA
    // context. The cache is ONE ~9 GB device buffer keyed on (keys,mask,rows,coeffs);
    // giving each worker its own would multiply that by the worker count and
    // re-create the out-of-memory that killed multi-PROCESS concurrency (measured:
    // 8 processes already peak at 37.7 of 45 GiB, and 48 processes OOM'd away 19 of
    // 114 proofs).
    //
    // Holding a lock across acquire + the a_vec launch + its sync serialises only the
    // GPU section. That is deliberate and cheap: the device is ~20% utilised, so the
    // serialised part is the small part, while the ~80% that is host work (CRS mmap,
    // witness generation, is_satisfied, artifact write) still overlaps freely across
    // threads. It also keeps device memory at roughly the single-worker footprint.
    //
    // The sync inside the guarded region is REQUIRED, not incidental: the launches are
    // asynchronous, so releasing the lock before the kernel has consumed d_A would let
    // the next thread's acquire cudaFree/regenerate the buffer under a live kernel --
    // exactly the illegal memory access this codebase has hit before.
    inline std::mutex &a_cache_mutex() {
        static std::mutex m;
        return m;
    }

    inline a_matrix_cache &a_cache() {
        // No destructor frees this: a static teardown cudaFree runs after the CUDA
        // context may already be gone. The driver reclaims it at process exit.
        static a_matrix_cache c;
        return c;
    }

    // Device A for (key, mask, rows, coeffs), generated on a miss; nullptr when A
    // cannot be held resident (caller then uses the fused kernel). Call AFTER the
    // proof's other device allocations, so cudaMemGetInfo's `free` already nets
    // them out -- a resident A has to coexist with them on every LATER proof too.
    // `reserve_bytes` is what THIS proof's own device buffers occupy; the budget
    // holds that much back again so a later, larger CRS can still allocate its
    // enc_qs mirror next to a resident A.
    inline void *a_matrix_acquire(const std::uint32_t *host_keys,
                                  const std::uint64_t *mask, std::size_t rows,
                                  std::size_t coeffs, bool big,
                                  std::size_t reserve_bytes,
                                  const std::uint32_t *d_aes_round_keys) {
        if (!vfhe_env_on("HECATE_A_CACHE", true)) return nullptr;
        const std::size_t elem_bytes = big ? 32 : 16;
        a_matrix_cache &c = a_cache();
        if (c.matches(host_keys, mask, rows, coeffs, elem_bytes)) return c.d_A;
        // Miss. Drop the identity now so a failure below cannot leave a stale hit;
        // whether the ALLOCATION survives is decided next.
        c.invalidate();

        if (rows == 0 || coeffs == 0) return nullptr;
        // The fill puts the coefficient on grid dim y, which caps at 65535. Every
        // Params::n in use is ~2-5k; a hypothetical larger one just takes the fused
        // path rather than silently generating a short A.
        if (coeffs > 65535) return nullptr;
        const std::size_t count = rows * coeffs;
        if (count / coeffs != rows) return nullptr;  // overflow
        const std::size_t bytes = count * elem_bytes;
        if (bytes / elem_bytes != count) return nullptr;

        // Keep the existing buffer when the new A fits it without wasting more than
        // half of it; that turns a key change into a refill with no cudaMalloc.
        if (!(c.d_A && bytes <= c.capacity && bytes * 2 >= c.capacity)) {
            c.release();
            std::size_t free_b = 0, total_b = 0;
            if (cudaMemGetInfo(&free_b, &total_b) != cudaSuccess) {
                cudaGetLastError();
                return nullptr;
            }
            const std::size_t slack =
                std::max(std::size_t(512) << 20, reserve_bytes);
            std::size_t budget = (free_b > slack) ? free_b - slack : 0;
            if (const char *e = std::getenv("HECATE_A_CACHE_MB")) {
                const std::size_t cap =
                    static_cast<std::size_t>(std::strtoull(e, nullptr, 10)) << 20;
                if (cap < budget) budget = cap;
            }
            if (bytes > budget) return nullptr;

            void *p = nullptr;
            if (cudaMalloc(&p, bytes) != cudaSuccess) {
                cudaGetLastError();  // clear, so the caller's cudaCheckError() stays clean
                return nullptr;
            }
            c.d_A = p;
            c.capacity = bytes;
        }
        void *d_A = c.d_A;
        const auto gen_srt = std::chrono::high_resolution_clock::now();
        // row0 = 0: the cache always holds the whole matrix, rows [0, rows).
        const int irows = static_cast<int>(rows), icoeffs = static_cast<int>(coeffs);
        if (big)
            launch_generate_a_matrix_big(reinterpret_cast<std::uint64_t *>(d_A), irows,
                                         0, icoeffs, mask[0], mask[1], mask[2],
                                         mask[3], d_aes_round_keys);
        else
            launch_generate_a_matrix(d_A, irows, 0, icoeffs, mask[0], mask[1],
                                     d_aes_round_keys);
        if (cudaDeviceSynchronize() != cudaSuccess) {
            cudaGetLastError();
            c.release();
            return nullptr;
        }
        const double gen_s =
            std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::high_resolution_clock::now() - gen_srt)
                .count() / 1e6;
        std::memcpy(c.keys, host_keys, sizeof(c.keys));
        std::memcpy(c.mask, mask, sizeof(c.mask));
        c.rows = rows;
        c.coeffs = coeffs;
        c.elem_bytes = elem_bytes;
        std::cout << "[PROFILER] a_vec matrix A materialised in " << gen_s
                  << " seconds (" << (bytes >> 20)
                  << " MB resident): every later proof under this key MACs it "
                     "instead of regenerating\n";
        return d_A;
    }

    // Drop the resident A. Called when the sink evicts a CRS -- eviction happens
    // under memory pressure, so holding A for a key that may not come back is
    // exactly the wrong trade.
    inline void release_a_matrix_cache() { a_cache().release(); }

    // CPU response generator for the big-int ring path (cpT::is_big), where the
    // SNARK modulus exceeds 128 bits and the native GPU kernels (uint128) don't
    // apply. Mirrors the GPU "Generating response" block:
    //   c_vec[j] = Σ_i enc_qs[i][j]·pi[i]
    //   a_vec[j] = Σ_i A[i][j]·pi[i]
    // A is the pseudorandom LWE matrix, NOT stored: it is regenerated by REPLAYING
    // the exact random_element_sequence draws encrypt_query_matrix used — a fresh
    // PRG seeded with crs_aes_key, n elements per query, in query order. Using the
    // identical code path guarantees the regenerated A matches the CRS (no AES
    // counter reverse-engineering). Works for any ring type (native or RingBig).
    template <typename ppT, typename cpT, class Params>
    void generate_response_cpu(
        const r1cs_lattice_snark_crs<ppT, cpT, Params> &crs,
        const libff::Fr_vector<ppT> &pi,
        LWE::ciphertext<Rq_T<cpT>, libff::Fr<ppT>, Params> &added) {
        using RingT = Rq_T<cpT>;
        const size_t index = crs.enc_qs.size();
        const size_t total_c = Params::pt_dim + Params::tau;
        for (size_t j = 0; j < total_c; j++) {
            RingT acc;  // zero
            for (size_t i = 0; i < index; i++)
                acc += pi[i].lift_ring_multiply(crs.enc_qs[i][j]);
            added.c_vec[j] = acc;
        }
        auto *a_prg = new LWERandomness::PseudoRandomGenerator(crs.crs_aes_key);
        auto *saved_prg = RingT::prg;
        RingT::prg = a_prg;
        std::array<RingT, Params::n> row;
        for (size_t i = 0; i < index; i++) {
            RingT::random_element_sequence(row);  // query i's a_vec (n elements)
            for (size_t j = 0; j < Params::n; j++)
                added.a_vec[j] += pi[i].lift_ring_multiply(row[j]);
        }
        RingT::prg = saved_prg;
        delete a_prg;
    }

    template <typename ppT, typename cpT, class Params>
    r1cs_lattice_snark_proof<ppT, cpT, Params> r1cs_lattice_snark_prove(
        const r1cs_lattice_snark_crs<ppT, cpT, Params> &crs,
        const r1cs_primary_input<libff::Fr<ppT>> &primary_input,
        const r1cs_auxiliary_input<libff::Fr<ppT>> &auxiliary_input,
        double* gpu_time_out = nullptr) {
        
        libff::enter_block("Call to r1cs lattice snark prover");

        libff::enter_block("Compute H polynomial");

        LWERandomness::AES_KEY _aes_key;
        genAES_key(&_aes_key);

        auto *temp_prg = new LWERandomness::PseudoRandomGenerator(_aes_key);
        auto *temp_dg = new LWERandomness::DiscreteGaussian(
            Params::width, LWE::expand, *temp_prg);
        auto *original_prg = ppT::prg;
        auto *original_dg = ppT::dg;
        public_params_init<ppT, cpT>(temp_prg, temp_dg);

        const libff::Fr<ppT> d1 = libff::Fr<ppT>::random_element(),
                            d2 = libff::Fr<ppT>::random_element(),
                            d3 = libff::Fr<ppT>::random_element();
        public_params_init<ppT, cpT>(original_prg, original_dg);
        delete temp_dg;
        delete temp_prg;

        // GPU QAP witness map (opt-in via HECATE_QAP_GPU). The witness map is
        // ring-independent (it is purely over the SNARK field), so both ring
        // paths can use it. Returns false (CPU fallback) for FFT domains we
        // don't replicate exactly; results are always bit-identical to the CPU
        // path. For the native uint64 ring the device pi feeds the response
        // kernels directly (no host round-trip); for the big-int ring the pi is
        // materialised to host for the CPU response generator.
        void *d_pi_qap = nullptr;
        size_t qap_dim = 0;
        bool used_gpu_qap = false;
        bool gpu_pi_on_device = false;
        libff::Fr_vector<ppT> pi;
        // ON by default; HECATE_QAP_GPU=0 disables. qap_witness_map_gpu returns
        // false for FFT domains it does not replicate exactly, so the CPU witness
        // map below still runs -- the fallback is built into the contract.
        if (vfhe_env_on("HECATE_QAP_GPU", true) && vfhe_gpu_available()) {
            std::vector<libff::Fr<ppT>> full_assignment = primary_input;
            full_assignment.insert(full_assignment.end(),
                                   auxiliary_input.begin(),
                                   auxiliary_input.end());
            used_gpu_qap = qap_witness_map_gpu<libff::Fr<ppT>>(
                crs.constraint_system, full_assignment, d1, d2, d3, &d_pi_qap,
                &qap_dim);
        }
        if (used_gpu_qap) {
            if constexpr (cpT::is_big) {
                // materialise device pi -> host pi for the CPU big-int response
                using HostT = decltype(libff::Fr<ppT>::value);
                std::vector<HostT> tmp(qap_dim);
                cudaMemcpy(tmp.data(), d_pi_qap, qap_dim * sizeof(HostT),
                           cudaMemcpyDeviceToHost);
                pi.resize(qap_dim);
                for (size_t i = 0; i < qap_dim; ++i) pi[i].value = tmp[i];
                cudaFree(d_pi_qap);
                d_pi_qap = nullptr;
            } else {
                gpu_pi_on_device = true; // response kernels consume d_pi_qap
            }
        } else {
            const qap_witness<libff::Fr<ppT>> qap_wit = r1cs_to_qap_witness_map(
                crs.constraint_system, primary_input, auxiliary_input, d1, d2,
                d3);
            prepare_pi_proof<ppT>(qap_wit, pi);
            assert(pi.size() == crs.enc_qs.size());
        }
        libff::leave_block("Compute H polynomial");

        libff::enter_block("Generating response (GPU Accelerated)");

        LWE::ciphertext<Rq_T<cpT>, libff::Fr<ppT>, Params> added;
        const int index = crs.enc_qs.size();
        double transfer_t = 0, compute_t = 0;
        // Big-int ring (q_log > 128): the native GPU kernels are uint128-only, so
        // run the equivalent response generation on the CPU. The 28-bit path keeps
        // the GPU. if constexpr discards the unused branch per instantiation, so the
        // big pp never needs the CUDA calls.
        if constexpr (cpT::is_big) {
          // Big-int (256-bit) response generation. GPU under HECATE_CRS_GPU
          // (256-bit analog of the native kernels), else CPU. Byte-exact: the
          // device MAC is uint256::mul_u64 + operator+=, a_vec is the same two-
          // AES-block element as RingBig::random_element.
          const int total_coeffs_c = Params::pt_dim + Params::tau;
          const int total_coeffs_a = Params::n;
          // ON by default; HECATE_CRS_GPU=0 forces the CPU response generator.
          // Unlike the keygen path there is no try/catch here: this block reports
          // CUDA errors via cudaCheckError(), which exit()s rather than throwing,
          // so the vfhe_gpu_available() probe is what protects a CPU-only host.
          // A device that exists but fails mid-kernel still aborts, exactly as it
          // did when this was opt-in.
          if (vfhe_env_on("HECATE_CRS_GPU", true) && vfhe_gpu_available()) {
            // CHUNKED c_vec response. The enc_qs device mirror dominates GPU memory
            // (index * total_coeffs_c * 32 B); a large QAP domain (the 2^25
            // rotate/relin key-switch is ~53 GB) exceeds an L40S's 46 GB and the
            // single-shot copy aborted with cudaErrorInvalidValue. So process enc_qs
            // in ROW-CHUNKS — the same pattern crs_gen.cu already uses for keygen:
            // accumulate_c_vec_big now adds into out_c_vec (memset to 0 once), and we
            // stream at most `chunkRows` rows through a bounded d_enc. Byte-identical
            // to the single-shot path. a_vec needs no enc_qs (it generates from AES +
            // pi), so it stays a single launch over the full index.
            std::vector<uint64_t> flat_pi(index);
            for (int i = 0; i < index; i++) flat_pi[i] = (uint64_t)pi[i].value;

            const uint64_t q_log = Params::q_log;
            uint64_t m[4] = {0, 0, 0, 0};
            for (uint64_t b = 0; b < q_log && b < 256; b++) m[b >> 6] |= (1ull << (b & 63));

            // Row budget: cap d_enc at ~8 GB so it fits any modern device with room
            // for d_pi/d_out and page-cache pressure, regardless of the domain size.
            const std::size_t bytes_per_row =
                static_cast<std::size_t>(total_coeffs_c) * 4 * sizeof(uint64_t);
            const std::size_t kEncBudget = std::size_t(8) << 30;  // 8 GB
            int chunkRows = static_cast<int>(
                std::max<std::size_t>(1, std::min<std::size_t>(
                    static_cast<std::size_t>(index), kEncBudget / bytes_per_row)));

            uint64_t *d_pi, *d_enc, *d_outc, *d_outa;
            uint32_t *d_keys;
            cudaMalloc(&d_pi, (std::size_t)index * sizeof(uint64_t));
            cudaMalloc(&d_enc, (std::size_t)chunkRows * bytes_per_row);
            cudaMalloc(&d_outc, total_coeffs_c * 4 * sizeof(uint64_t));
            cudaMalloc(&d_outa, total_coeffs_a * 4 * sizeof(uint64_t));
            cudaMalloc(&d_keys, 44 * sizeof(uint32_t));
            cudaMemset(d_outc, 0, total_coeffs_c * 4 * sizeof(uint64_t));
            cudaMemset(d_outa, 0, total_coeffs_a * 4 * sizeof(uint64_t));
            cudaMemcpy(d_pi, flat_pi.data(), (std::size_t)index * sizeof(uint64_t), cudaMemcpyHostToDevice);
            cudaMemcpy(d_keys, crs.crs_aes_key.rd_key, 44 * sizeof(uint32_t), cudaMemcpyHostToDevice);

            // crs.enc_qs already IS the flat [index][total_coeffs_c] w[0..3]
            // buffer (see enc_qs_flat), so a chunk is a contiguous byte range at
            // r0 * bytes_per_row -- no per-chunk gather, no staging copy.
            const uint64_t *enc_base = enc_qs_flat(crs.enc_qs);
            for (int r0 = 0; r0 < index; r0 += chunkRows) {
              const int rows = std::min(chunkRows, index - r0);
              cudaMemcpy(d_enc,
                         reinterpret_cast<const char *>(enc_base) +
                             static_cast<std::size_t>(r0) * bytes_per_row,
                         static_cast<std::size_t>(rows) * bytes_per_row,
                         cudaMemcpyHostToDevice);
              // pi is chunk-local via the offset pointer; enc_qs indexed [0,rows).
              launch_accumulate_c_vec_big(d_enc, d_pi + r0, d_outc, rows,
                                          total_coeffs_c, total_coeffs_c, 256);
              cudaCheckError();
              cudaDeviceSynchronize();
            }
            // a_vec: single launch over the full index (no enc_qs; d_pi = 256 MB).
            // With a resident A this proof pays only the MAC; a_matrix_acquire
            // returns null when A will not fit and the fused kernel runs instead.
            std::unique_lock<std::mutex> a_lk(a_cache_mutex());
            void *d_A = a_matrix_acquire(
                reinterpret_cast<const std::uint32_t *>(crs.crs_aes_key.rd_key), m,
                static_cast<std::size_t>(index),
                static_cast<std::size_t>(total_coeffs_a), /*big=*/true,
                /*reserve_bytes=*/static_cast<std::size_t>(chunkRows) * bytes_per_row,
                d_keys);
            if (d_A)
              launch_accumulate_a_vec_big(reinterpret_cast<const uint64_t *>(d_A), d_pi,
                                          d_outa, index, total_coeffs_a, 256);
            else
              launch_generate_a_vec_big(d_pi, d_outa, index, total_coeffs_a, total_coeffs_a, 256,
                                        m[0], m[1], m[2], m[3], d_keys);
            cudaCheckError();
            cudaDeviceSynchronize();
            a_lk.unlock();  // d_A consumed

            std::vector<uint64_t> hc(total_coeffs_c * 4), ha(total_coeffs_a * 4);
            cudaMemcpy(hc.data(), d_outc, total_coeffs_c * 4 * sizeof(uint64_t), cudaMemcpyDeviceToHost);
            cudaMemcpy(ha.data(), d_outa, total_coeffs_a * 4 * sizeof(uint64_t), cudaMemcpyDeviceToHost);
            for (int i = 0; i < total_coeffs_c; i++) {
              auto &v = added.c_vec[i].value;
              v.w[0]=hc[i*4]; v.w[1]=hc[i*4+1]; v.w[2]=hc[i*4+2]; v.w[3]=hc[i*4+3];
            }
            for (int i = 0; i < total_coeffs_a; i++) {
              auto &v = added.a_vec[i].value;
              v.w[0]=ha[i*4]; v.w[1]=ha[i*4+1]; v.w[2]=ha[i*4+2]; v.w[3]=ha[i*4+3];
            }
            cudaFree(d_pi); cudaFree(d_enc); cudaFree(d_outc); cudaFree(d_outa); cudaFree(d_keys);
          } else {
            generate_response_cpu<ppT, cpT, Params>(crs, pi, added);
          }
        } else {

        // Fix 1: The true underlying types revealed by the compiler
        using ScalarType128 = unsigned __int128;
        
        int total_coeffs_a = Params::n; 
        // Fix 2: Extracted directly from the LWE::Vector<..., 53> error
        int total_coeffs_c = 53; 

        // enc_qs needs NO flattening: crs.enc_qs already IS the row-major
        // [index][total_coeffs_c] (lo,hi) buffer the kernel reads (see
        // enc_qs_flat). This used to rebuild a bit-identical index*53*16 B copy on
        // the CPU for every proof -- 6.2 GB / ~7.5 s per proof at index = 7.27M,
        // ~32% of the prover call, all of it redundant across proofs sharing a key.
        const uint64_t* flat_qs_ptr = enc_qs_flat(crs.enc_qs);

        std::vector<uint64_t> flat_pi;
        if (!used_gpu_qap) {
            flat_pi.resize(index);
            uint64_t* pi_ptr = flat_pi.data();

            #pragma omp parallel for
            for(int i = 0; i < index; i++) {
                pi_ptr[i] = pi[i].value;
            }
        } else if (qap_dim != (size_t)index) {
            throw std::runtime_error(
                "qap_witness_map_gpu: proof_dim != enc_qs size");
        }

        // Extract Mask
        unsigned __int128 mask = crs.enc_qs[0][0].mod - 1;
        uint64_t mod_mask_lo = (uint64_t)mask;
        uint64_t mod_mask_hi = (uint64_t)(mask >> 64);

        // Allocate Device Memory (Multiplying sizes by 2 to account for 64-bit chunks)
        uint64_t *d_pi;
        void *d_out_a_vec, *d_out_c_vec, *d_enc_qs; 
        uint32_t *d_aes_round_keys;
        // uint32_t *d_aes_round_keys;
        // cudaMalloc(&d_aes_round_keys, 44 * sizeof(uint32_t));
        // cudaMemcpy(d_aes_round_keys, crs.crs_aes_key.rd_key, 44 * sizeof(uint32_t), cudaMemcpyHostToDevice);
        // When the GPU QAP path ran, pi is already resident on the device
        // (d_pi_qap); reuse it directly instead of allocating + uploading.
        if (used_gpu_qap)
            d_pi = reinterpret_cast<uint64_t *>(d_pi_qap);
        else
            cudaMalloc(&d_pi, index * sizeof(uint64_t));
        // CHECKED allocations. cudaMalloc leaves the pointer UNTOUCHED on failure,
        // so an unchecked failure launches the kernel against an uninitialized
        // pointer -- compute-sanitizer caught exactly that: accumulate_c_vec
        // reading address 0x6a00, with the nearest real allocation 21.6 GB away.
        //
        // It fails because the resident A matrix (a_matrix_cache, ~9 GB) has to
        // coexist with THIS proof's enc_qs mirror, which for a top-of-chain relin
        // board is ~21.6 GB on a 46 GB card. A's admission budget is computed from
        // the proof resident at the time it was taken, so a later, larger board can
        // still be squeezed out. When that happens, drop A and retry: A is a cache
        // and is regenerable, the proof is not.
        auto alloc_or_drop_A = [&](void **p, std::size_t bytes) -> bool {
            *p = nullptr;
            if (cudaMalloc(p, bytes) == cudaSuccess) return true;
            cudaGetLastError();                 // clear the sticky error
            {   // Visible, because this is the event that decides whether the A
                // cache can pay at all: if a big board evicts A on every proof,
                // A is re-materialised each time and the cache is pure overhead.
                std::size_t fb = 0, tb = 0;
                cudaMemGetInfo(&fb, &tb);
                std::fprintf(stderr,
                             "[a-cache] dropping resident A to fit a %.1f GB "
                             "allocation (free %.1f / %.1f GB)\n",
                             bytes / 1e9, fb / 1e9, tb / 1e9);
            }
            release_a_matrix_cache();           // give back the ~9 GB and retry once
            if (cudaMalloc(p, bytes) == cudaSuccess) return true;
            cudaGetLastError();
            *p = nullptr;
            return false;
        };
        bool alloc_ok = true;
        alloc_ok &= alloc_or_drop_A(&d_out_a_vec, total_coeffs_a * 2 * sizeof(uint64_t));
        alloc_ok &= alloc_or_drop_A(&d_out_c_vec, total_coeffs_c * 2 * sizeof(uint64_t));
        alloc_ok &= alloc_or_drop_A(&d_enc_qs, static_cast<std::size_t>(index) *
                                                   total_coeffs_c * 2 * sizeof(uint64_t));
        alloc_ok &= alloc_or_drop_A(reinterpret_cast<void **>(&d_aes_round_keys),
                                    44 * sizeof(uint32_t));
        if (!alloc_ok) {
            // Out of device memory even without A. Fail loudly here rather than
            // launching a kernel on a null pointer and reporting an illegal access
            // against whatever proof happens to be running.
            std::fprintf(stderr,
                         "[r1cs_lattice_snark] device allocation failed for a %llu-row "
                         "x %llu-coeff board (enc_qs %.1f GB); aborting this proof\n",
                         (unsigned long long)index, (unsigned long long)total_coeffs_c,
                         (double)index * total_coeffs_c * 2 * sizeof(uint64_t) / 1e9);
            if (d_out_a_vec) cudaFree(d_out_a_vec);
            if (d_out_c_vec) cudaFree(d_out_c_vec);
            if (d_enc_qs) cudaFree(d_enc_qs);
            if (d_aes_round_keys) cudaFree(d_aes_round_keys);
            if (!used_gpu_qap && d_pi) cudaFree(d_pi);
            throw std::runtime_error("r1cs_lattice_snark: device out of memory");
        }

        cudaMemset(d_out_a_vec, 0, total_coeffs_a * 2 * sizeof(uint64_t));
        cudaMemset(d_out_c_vec, 0, total_coeffs_c * 2 * sizeof(uint64_t));
        auto transfer_srt = std::chrono::high_resolution_clock::now();
        if (!used_gpu_qap)
            cudaMemcpy(d_pi, flat_pi.data(), index * sizeof(uint64_t), cudaMemcpyHostToDevice);
        cudaMemcpy(d_enc_qs, flat_qs_ptr, static_cast<std::size_t>(index) * total_coeffs_c * 2 * sizeof(uint64_t), cudaMemcpyHostToDevice);
        cudaMemcpy(d_aes_round_keys, crs.crs_aes_key.rd_key, 44 * sizeof(uint32_t), cudaMemcpyHostToDevice);
        // copy_aes_keys_to_constant(reinterpret_cast<const uint32_t*>(crs.crs_aes_key.rd_key));
        // ---------------------------------------------------------
        // 2. ADD THE END TRANSFER TIMER & START COMPUTE TIMER
        auto transfer_end = std::chrono::high_resolution_clock::now();
        auto gpu_compute_srt = std::chrono::high_resolution_clock::now();
        // ---------------------------------------------------------
        
        // Launch c_vec accumulation
        // --- SETUP TIMERS ---
        cudaEvent_t start_c, stop_c, start_a, stop_a;
        cudaEventCreate(&start_c); cudaEventCreate(&stop_c);
        cudaEventCreate(&start_a); cudaEventCreate(&stop_a);

        int threads_per_block = 256;
        
        // --- TIMING C_VEC ---
        int blocks_c = total_coeffs_c; 
        cudaEventRecord(start_c);
        
        launch_accumulate_c_vec_kernel(
            d_enc_qs, d_pi, d_out_c_vec, index, total_coeffs_c, blocks_c, threads_per_block
        );
        
        cudaEventRecord(stop_c);
        cudaEventSynchronize(stop_c); // Force CPU to wait for GPU
        
        float milliseconds_c = 0;
        cudaEventElapsedTime(&milliseconds_c, start_c, stop_c);
        std::cout << "[PROFILER] c_vec accumulation took: " << milliseconds_c / 1000.0 << " seconds\n";

        // --- TIMING A_VEC ---
        int blocks_a = total_coeffs_a;
        const std::uint64_t a_mask[4] = {mod_mask_lo, mod_mask_hi, 0, 0};
        cudaEventRecord(start_a);

        // Try for a resident A first: on a hit this proof pays only the MAC, not
        // the index*n AES blocks. a_matrix_acquire runs AFTER the allocations
        // above on purpose (see its comment), and returns null when A will not
        // fit, in which case the original fused kernel runs unchanged. It is
        // inside the timed region so the a_vec number below stays honest: the
        // first proof under a key pays the fill, later ones do not.
        std::unique_lock<std::mutex> a_lk(a_cache_mutex());
        void *d_A = a_matrix_acquire(
            reinterpret_cast<const std::uint32_t *>(crs.crs_aes_key.rd_key), a_mask,
            static_cast<std::size_t>(index), static_cast<std::size_t>(total_coeffs_a),
            /*big=*/false,
            /*reserve_bytes=*/static_cast<std::size_t>(index) * total_coeffs_c * 2 *
                sizeof(uint64_t),
            d_aes_round_keys);

        if (d_A) {
            launch_accumulate_a_vec_kernel(
                d_A, d_pi, d_out_a_vec, index, blocks_a, threads_per_block
            );
        } else {
            launch_generate_a_vec_kernel(
                d_pi, d_out_a_vec, index, total_coeffs_a, blocks_a, threads_per_block, mod_mask_lo, mod_mask_hi, d_aes_round_keys
            );
        }

        cudaEventRecord(stop_a);
        cudaEventSynchronize(stop_a); // Force CPU to wait for GPU
        a_lk.unlock();  // d_A consumed: release the A section to the next thread
        
        float milliseconds_a = 0;
        cudaEventElapsedTime(&milliseconds_a, start_a, stop_a);
        std::cout << "[PROFILER] a_vec generation took: " << milliseconds_a / 1000.0 << " seconds\n";

        // --- CLEANUP TIMERS ---
        cudaEventDestroy(start_c); cudaEventDestroy(stop_c);
        cudaEventDestroy(start_a); cudaEventDestroy(stop_a);



        
        cudaCheckError();

        cudaDeviceSynchronize();
        // ---------------------------------------------------------
        // 3. ADD THE END COMPUTE TIMER
        auto gpu_compute_end = std::chrono::high_resolution_clock::now();
        // ---------------------------------------------------------
        // Transfer Results Back to Host
        std::vector<uint64_t> host_out_a_vec(total_coeffs_a * 2);
        std::vector<uint64_t> host_out_c_vec(total_coeffs_c * 2);

        cudaMemcpy(host_out_a_vec.data(), d_out_a_vec, total_coeffs_a * 2 * sizeof(uint64_t), cudaMemcpyDeviceToHost);
        cudaMemcpy(host_out_c_vec.data(), d_out_c_vec, total_coeffs_c * 2 * sizeof(uint64_t), cudaMemcpyDeviceToHost);

        // Reconstruct the 128-bit values back into the C++ classes
        for(int i = 0; i < total_coeffs_a; i++) {
            unsigned __int128 lo = host_out_a_vec[i * 2];
            unsigned __int128 hi = host_out_a_vec[i * 2 + 1];
            added.a_vec[i].value = (hi << 64) | lo;
        }
        for(int i = 0; i < total_coeffs_c; i++) {
            unsigned __int128 lo = host_out_c_vec[i * 2];
            unsigned __int128 hi = host_out_c_vec[i * 2 + 1];
            added.c_vec[i].value = (hi << 64) | lo;
        }

        cudaFree(d_pi);
        cudaFree(d_out_a_vec);
        cudaFree(d_out_c_vec);
        cudaFree(d_enc_qs);
        cudaFree(d_aes_round_keys);

        using micro_s = std::chrono::microseconds;
        transfer_t = std::chrono::duration_cast<micro_s>(transfer_end - transfer_srt).count();
        compute_t = std::chrono::duration_cast<micro_s>(gpu_compute_end - gpu_compute_srt).count();

        if (gpu_time_out) {
            *gpu_time_out = (transfer_t + compute_t) / 1e6;  // seconds
        }
        }  // end else (GPU path)

    #ifndef NOT_PROVABLE_ZK
        // The public parameter is likely the CRS elements needed for blinding
        LWE::re_randomize(crs.public_parameter, added);
    #endif

        // =========================================================
        // DEBUG: ISOLATING THE CPU/GPU MISMATCH
        // // =========================================================
        
        // // 1. Check the Linear Math (c_vec)
        // unsigned __int128 cpu_c_vec_0 = 0;
        // for(int i = 0; i < index; i++) {
        //     cpu_c_vec_0 += (unsigned __int128)crs.enc_qs[i][0].value * (unsigned __int128)pi[i].value;
        // }
        // std::cout << "\n[DEBUG] CPU c_vec[0]: " << (uint64_t)(cpu_c_vec_0 >> 64) << " | " << (uint64_t)cpu_c_vec_0 << "\n";
        // std::cout << "[DEBUG] GPU c_vec[0]: " << (uint64_t)(added.c_vec[0].value >> 64) << " | " << (uint64_t)added.c_vec[0].value << "\n";

        // // 2. Check the AES Cryptography (a_vec)
        // auto *check_prg = new LWERandomness::PseudoRandomGenerator(crs.crs_aes_key);
        // unsigned __int128 cpu_a_vec_0 = 0;
        // for(int i = 0; i < index; i++) {
        //     // The CPU generates a block, but we only care about coeff 0
        //     unsigned __int128 rand_val = check_prg->next_prg_block();
            
        //     // Fast-forward the PRG state past the rest of the polynomial
        //     for(int j = 1; j < total_coeffs_a; j++) check_prg->next_prg_block();
            
        //     // The CPU applies a mask before multiplying!
        //     rand_val = rand_val & (crs.enc_qs[0][0].mod - 1);
            
        //     cpu_a_vec_0 += rand_val * (unsigned __int128)pi[i].value;
        // }
        // delete check_prg;
        
        // std::cout << "[DEBUG] CPU a_vec[0]: " << (uint64_t)(cpu_a_vec_0 >> 64) << " | " << (uint64_t)cpu_a_vec_0 << "\n";
        // std::cout << "[DEBUG] GPU a_vec[0]: " << (uint64_t)(added.a_vec[0].value >> 64) << " | " << (uint64_t)added.a_vec[0].value << "\n";
        // // =========================================================

        added.rescale();
        
        libff::leave_block("Generating response (GPU Accelerated)");

        libff::leave_block("Call to r1cs lattice snark prover");
        
        std::cout << "\n  * GPU Data Transfer: " + std::to_string(transfer_t / 1e6) + "s\n"
                << "  * GPU Computation: " + std::to_string(compute_t / 1e6) + "s\n"
                << "  * Linear comb size " + std::to_string(index)
                << std::endl;

        return r1cs_lattice_snark_proof<ppT, cpT, Params>(std::move(added));
    }

    // =====================================================================
    // BATCHED PROVER -- K traces against ONE key, proved together.
    //
    // WHY. The a_vec matrix A[i][j] = AES_k(i*n + j) & mask is a function of the
    // key alone, so proving K traces one at a time re-derives the SAME index*n
    // AES blocks K times. Here A is walked once in row-chunks and each chunk is
    // MAC'd against all K pi vectors before it is discarded, so the AES is paid
    // once per BATCH. Two chunk buffers on two streams overlap chunk c+1's fill
    // with chunk c's MACs, making the cost max(fill_total, K*read_total) instead
    // of their sum -- at the measured 204 GB/s fill and 678 GB/s resident read
    // that is ~3.3x on a_vec by K=5, against a 3.4x asymptote.
    //
    // The enc_qs mirror is amortised for free by the same restructuring: it is
    // uploaded ONCE and c_vec runs K times against it. That upload was measured
    // at 2.2 s per proof at index 7.27M, so on the large keys it is the bigger
    // half of the saving.
    //
    // HOST MEMORY stays flat in K: `fill` is invoked one trace at a time, so the
    // caller's protoboard (multi-GB at relin sizes) can die before the next call.
    // Only the K pi vectors survive into the batched phase, and those live on the
    // DEVICE (index * 8 B each -- 48 MB at index 6.06M).
    //
    // EXACTNESS. Identical to K separate r1cs_lattice_snark_prove calls on the
    // same witnesses: chunking A composes because the accumulators are mod 2^128
    // (mod 2^256 for the big ring) and therefore associative, and every other
    // step is per-trace and untouched.
    //
    // `fill(k, primary, auxiliary)` must populate the two vectors for trace k and
    // return false to skip it. ok_out (optional, resized to `batch`) reports which
    // traces produced a proof; skipped entries hold a default-constructed proof.
    template <typename ppT, typename cpT, class Params>
    std::vector<r1cs_lattice_snark_proof<ppT, cpT, Params>>
    r1cs_lattice_snark_prove_batch(
        const r1cs_lattice_snark_crs<ppT, cpT, Params> &crs, std::size_t batch,
        const std::function<bool(std::size_t, r1cs_primary_input<libff::Fr<ppT>> &,
                                 r1cs_auxiliary_input<libff::Fr<ppT>> &)> &fill,
        std::vector<char> *ok_out = nullptr, double *gpu_time_out = nullptr) {

        std::vector<r1cs_lattice_snark_proof<ppT, cpT, Params>> proofs(batch);
        std::vector<char> ok(batch, 0);
        auto finish = [&]() {
            if (ok_out) *ok_out = ok;
            return proofs;
        };
        if (batch == 0) return finish();

        const int index = static_cast<int>(crs.enc_qs.size());
        const int total_coeffs_a = Params::n;
        const int total_coeffs_c = Params::pt_dim + Params::tau;

        // Prove one at a time. The contract is unchanged, just unamortised.
        auto prove_serially = [&]() {
            for (std::size_t k = 0; k < batch; ++k) {
                r1cs_primary_input<libff::Fr<ppT>> prim;
                r1cs_auxiliary_input<libff::Fr<ppT>> aux;
                if (!fill(k, prim, aux)) continue;
                proofs[k] = r1cs_lattice_snark_prove<ppT, cpT, Params>(
                    crs, prim, aux, gpu_time_out);
                ok[k] = 1;
            }
        };

        // The batched response generator is the uint128 GPU path only. The big-int
        // ring runs a CPU response generator that materialises no A to share, and a
        // CPU-only host has no kernels at all: both prove one at a time, exactly as
        // before. if constexpr, so the big pp never instantiates the uint128 block.
        if constexpr (cpT::is_big) {
            prove_serially();
            return finish();
        } else {
        if (!(vfhe_env_on("HECATE_CRS_GPU", true) && vfhe_gpu_available() &&
              batch >= 2)) {
            prove_serially();
            return finish();
        }

        libff::enter_block("Call to r1cs lattice snark prover (batch)");
        const auto t_start = std::chrono::high_resolution_clock::now();

        // ---- PHASE 1: per trace, witness -> pi, straight into a device slab.
        // One protoboard at a time upstream; here one pi at a time, D2D-copied into
        // d_pi_all so the batched phase can index every trace's pi by offset.
        uint64_t *d_pi_all = nullptr;
        if (cudaMalloc(&d_pi_all, (std::size_t)batch * index * sizeof(uint64_t)) !=
            cudaSuccess) {
            cudaGetLastError();
            libff::leave_block("Call to r1cs lattice snark prover (batch)");
            // Out of room for K pi vectors: fall back rather than fail the batch.
            prove_serially();
            return finish();
        }

        libff::enter_block("Compute H polynomial (batch)");
        std::vector<std::size_t> live;  // batch slots that produced a pi
        for (std::size_t k = 0; k < batch; ++k) {
            r1cs_primary_input<libff::Fr<ppT>> prim;
            r1cs_auxiliary_input<libff::Fr<ppT>> aux;
            if (!fill(k, prim, aux)) continue;

            // Per-trace ZK blinding, exactly as the scalar prover draws it.
            LWERandomness::AES_KEY _aes_key;
            genAES_key(&_aes_key);
            auto *temp_prg = new LWERandomness::PseudoRandomGenerator(_aes_key);
            auto *temp_dg = new LWERandomness::DiscreteGaussian(
                Params::width, LWE::expand, *temp_prg);
            auto *original_prg = ppT::prg;
            auto *original_dg = ppT::dg;
            public_params_init<ppT, cpT>(temp_prg, temp_dg);
            const libff::Fr<ppT> d1 = libff::Fr<ppT>::random_element(),
                                 d2 = libff::Fr<ppT>::random_element(),
                                 d3 = libff::Fr<ppT>::random_element();
            public_params_init<ppT, cpT>(original_prg, original_dg);
            delete temp_dg;
            delete temp_prg;

            uint64_t *slot = d_pi_all + (std::size_t)k * index;
            void *d_pi_qap = nullptr;
            std::size_t qap_dim = 0;
            bool used_gpu_qap = false;
            if (vfhe_env_on("HECATE_QAP_GPU", true)) {
                std::vector<libff::Fr<ppT>> full_assignment = prim;
                full_assignment.insert(full_assignment.end(), aux.begin(), aux.end());
                used_gpu_qap = qap_witness_map_gpu<libff::Fr<ppT>>(
                    crs.constraint_system, full_assignment, d1, d2, d3, &d_pi_qap,
                    &qap_dim);
            }
            if (used_gpu_qap) {
                if (static_cast<int>(qap_dim) != index) {
                    cudaFree(d_pi_qap);
                    throw std::runtime_error(
                        "qap_witness_map_gpu: proof_dim != enc_qs size");
                }
                cudaMemcpy(slot, d_pi_qap, (std::size_t)index * sizeof(uint64_t),
                           cudaMemcpyDeviceToDevice);
                cudaFree(d_pi_qap);
            } else {
                const qap_witness<libff::Fr<ppT>> qap_wit = r1cs_to_qap_witness_map(
                    crs.constraint_system, prim, aux, d1, d2, d3);
                libff::Fr_vector<ppT> pi;
                prepare_pi_proof<ppT>(qap_wit, pi);
                assert(static_cast<int>(pi.size()) == index);
                std::vector<uint64_t> flat(index);
#pragma omp parallel for
                for (int i = 0; i < index; i++) flat[i] = (uint64_t)pi[i].value;
                cudaMemcpy(slot, flat.data(), (std::size_t)index * sizeof(uint64_t),
                           cudaMemcpyHostToDevice);
            }
            live.push_back(k);
        }
        libff::leave_block("Compute H polynomial (batch)");
        if (live.empty()) {
            cudaFree(d_pi_all);
            libff::leave_block("Call to r1cs lattice snark prover (batch)");
            return finish();
        }

        // ---- PHASE 2: one pass over the key, K accumulations per pass.
        libff::enter_block("Generating response (batch, GPU)");
        const int K = static_cast<int>(live.size());
        const int threads = 256;

        unsigned __int128 mask = crs.enc_qs[0][0].mod - 1;
        const uint64_t mod_mask_lo = (uint64_t)mask;
        const uint64_t mod_mask_hi = (uint64_t)(mask >> 64);

        void *d_enc_qs = nullptr, *d_out_a = nullptr, *d_out_c = nullptr;
        uint32_t *d_keys = nullptr;
        const std::size_t enc_bytes =
            (std::size_t)index * total_coeffs_c * 2 * sizeof(uint64_t);
        // CHECKED, for the same reason as the single-proof path: an unchecked
        // failure here leaves the pointer uninitialized and the kernel reads it.
        // The batch path is MORE exposed, not less -- it holds K witnesses plus a
        // resident A alongside enc_qs, and A at batch dimensions was measured at
        // 36.6 GB on a 47.7 GB card. Drop A (regenerable) and retry before failing.
        auto balloc = [&](void **p, std::size_t bytes) -> bool {
            *p = nullptr;
            if (cudaMalloc(p, bytes) == cudaSuccess) return true;
            cudaGetLastError();
            {
                std::size_t fb = 0, tb = 0;
                cudaMemGetInfo(&fb, &tb);
                std::fprintf(stderr,
                             "[a-cache] (batch) dropping resident A to fit a %.1f GB "
                             "allocation (free %.1f / %.1f GB)\n",
                             bytes / 1e9, fb / 1e9, tb / 1e9);
            }
            release_a_matrix_cache();
            if (cudaMalloc(p, bytes) == cudaSuccess) return true;
            cudaGetLastError();
            *p = nullptr;
            return false;
        };
        bool balloc_ok = true;
        balloc_ok &= balloc(&d_enc_qs, enc_bytes);
        balloc_ok &= balloc(&d_out_a, (std::size_t)K * total_coeffs_a * 2 * sizeof(uint64_t));
        balloc_ok &= balloc(&d_out_c, (std::size_t)K * total_coeffs_c * 2 * sizeof(uint64_t));
        balloc_ok &= balloc(reinterpret_cast<void **>(&d_keys), 44 * sizeof(uint32_t));
        if (!balloc_ok) {
            std::fprintf(stderr,
                         "[r1cs_lattice_snark] batch device allocation failed "
                         "(K=%d, enc_qs %.1f GB); aborting this batch\n",
                         (int)K, (double)enc_bytes / 1e9);
            if (d_enc_qs) cudaFree(d_enc_qs);
            if (d_out_a) cudaFree(d_out_a);
            if (d_out_c) cudaFree(d_out_c);
            if (d_keys) cudaFree(d_keys);
            throw std::runtime_error("r1cs_lattice_snark: device out of memory (batch)");
        }
        cudaMemset(d_out_a, 0, (std::size_t)K * total_coeffs_a * 2 * sizeof(uint64_t));
        cudaMemset(d_out_c, 0, (std::size_t)K * total_coeffs_c * 2 * sizeof(uint64_t));

        const auto transfer_srt = std::chrono::high_resolution_clock::now();
        // ONE upload for the whole batch -- the amortisation the scalar path could
        // not do (enc_qs is already the flat device layout, see enc_qs_flat).
        cudaMemcpy(d_enc_qs, enc_qs_flat(crs.enc_qs), enc_bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(d_keys, crs.crs_aes_key.rd_key, 44 * sizeof(uint32_t),
                   cudaMemcpyHostToDevice);
        const auto transfer_end = std::chrono::high_resolution_clock::now();
        const auto compute_srt = std::chrono::high_resolution_clock::now();

        for (int b = 0; b < K; ++b) {
            launch_accumulate_c_vec_kernel(
                d_enc_qs, d_pi_all + (std::size_t)live[b] * index,
                (uint64_t *)d_out_c + (std::size_t)b * total_coeffs_c * 2, index,
                total_coeffs_c, total_coeffs_c, threads);
        }
        cudaCheckError();

        // a_vec. A resident whole A (small keys) needs no fill at all; otherwise
        // stream A through two chunk buffers, MACing all K traces per chunk.
        const std::uint64_t a_mask[4] = {mod_mask_lo, mod_mask_hi, 0, 0};
        std::unique_lock<std::mutex> a_res_lk(a_cache_mutex());
        void *d_A_res = a_matrix_acquire(
            reinterpret_cast<const std::uint32_t *>(crs.crs_aes_key.rd_key), a_mask,
            static_cast<std::size_t>(index), static_cast<std::size_t>(total_coeffs_a),
            /*big=*/false, /*reserve_bytes=*/enc_bytes, d_keys);
        const auto a_srt = std::chrono::high_resolution_clock::now();
        if (d_A_res) {
            for (int b = 0; b < K; ++b)
                launch_accumulate_a_vec_kernel(
                    d_A_res, d_pi_all + (std::size_t)live[b] * index,
                    (uint64_t *)d_out_a + (std::size_t)b * total_coeffs_a * 2, index,
                    total_coeffs_a, threads);
            cudaCheckError();
            cudaDeviceSynchronize();
        } else {
            // Chunk rows so TWO buffers fit what is free after enc_qs/pi/outputs,
            // capped at 4 GB each (bigger buys nothing -- the fill is already
            // saturated long before that, and the cap keeps room for the next key).
            const std::size_t row_bytes = (std::size_t)total_coeffs_a * 16;
            std::size_t free_b = 0, total_b = 0;
            cudaMemGetInfo(&free_b, &total_b);
            const std::size_t slack = std::size_t(1) << 30;
            std::size_t per_buf = (free_b > slack) ? (free_b - slack) / 2 : 0;
            if (per_buf > (std::size_t(4) << 30)) per_buf = std::size_t(4) << 30;
            int chunk_rows = static_cast<int>(
                std::max<std::size_t>(1, std::min<std::size_t>(index, per_buf / row_bytes)));

            void *buf[2] = {nullptr, nullptr};
            const std::size_t buf_bytes = (std::size_t)chunk_rows * row_bytes;
            const bool two = cudaMalloc(&buf[0], buf_bytes) == cudaSuccess &&
                             cudaMalloc(&buf[1], buf_bytes) == cudaSuccess;
            if (!two) {
                // Could not double-buffer: run the fused per-trace kernel, which
                // needs no A storage. Correct, just unamortised.
                cudaGetLastError();
                if (buf[0]) cudaFree(buf[0]);
                if (buf[1]) cudaFree(buf[1]);
                for (int b = 0; b < K; ++b)
                    launch_generate_a_vec_kernel(
                        d_pi_all + (std::size_t)live[b] * index,
                        (uint64_t *)d_out_a + (std::size_t)b * total_coeffs_a * 2,
                        index, total_coeffs_a, total_coeffs_a, threads, mod_mask_lo,
                        mod_mask_hi, d_keys);
                cudaCheckError();
                cudaDeviceSynchronize();
            } else {
                cudaStream_t s_fill, s_mac;
                cudaStreamCreate(&s_fill);
                cudaStreamCreate(&s_mac);
                cudaEvent_t filled[2], maced[2];
                for (int i = 0; i < 2; ++i) {
                    cudaEventCreateWithFlags(&filled[i], cudaEventDisableTiming);
                    cudaEventCreateWithFlags(&maced[i], cudaEventDisableTiming);
                }
                int chunk = 0;
                for (int r0 = 0; r0 < index; r0 += chunk_rows, ++chunk) {
                    const int rows = std::min(chunk_rows, index - r0);
                    const int cur = chunk & 1;
                    // Buffer reuse: this fill must not start until the MACs that
                    // read buf[cur] two chunks ago have finished.
                    if (chunk >= 2) cudaStreamWaitEvent(s_fill, maced[cur], 0);
                    launch_generate_a_matrix(buf[cur], rows, (uint64_t)r0,
                                             total_coeffs_a, mod_mask_lo, mod_mask_hi,
                                             d_keys, s_fill);
                    cudaEventRecord(filled[cur], s_fill);
                    // ... and the MACs wait for the fill they consume, then every
                    // trace in the batch reads this chunk before it is overwritten.
                    cudaStreamWaitEvent(s_mac, filled[cur], 0);
                    for (int b = 0; b < K; ++b)
                        launch_accumulate_a_vec_kernel(
                            buf[cur], d_pi_all + (std::size_t)live[b] * index + r0,
                            (uint64_t *)d_out_a + (std::size_t)b * total_coeffs_a * 2,
                            rows, total_coeffs_a, threads, s_mac);
                    cudaEventRecord(maced[cur], s_mac);
                }
                cudaCheckError();
                cudaStreamSynchronize(s_mac);
                cudaStreamSynchronize(s_fill);
                for (int i = 0; i < 2; ++i) {
                    cudaEventDestroy(filled[i]);
                    cudaEventDestroy(maced[i]);
                }
                cudaStreamDestroy(s_fill);
                cudaStreamDestroy(s_mac);
                cudaFree(buf[0]);
                cudaFree(buf[1]);
                std::cout << "[PROFILER] a_vec batch: A streamed in " << chunk
                          << " chunks of " << chunk_rows << " rows, MAC'd by " << K
                          << " traces per chunk (AES paid once for the batch)\n";
            }
        }
        cudaDeviceSynchronize();
        a_res_lk.unlock();  // every A consumer (resident or chunked) has synced
        const auto a_end = std::chrono::high_resolution_clock::now();
        std::cout << "[PROFILER] a_vec generation took: "
                  << std::chrono::duration_cast<std::chrono::microseconds>(a_end - a_srt)
                             .count() / 1e6
                  << " seconds (batch of " << K << ")\n";

        // ---- PHASE 3: per trace, assemble + re-randomise.
        std::vector<uint64_t> host_a((std::size_t)K * total_coeffs_a * 2);
        std::vector<uint64_t> host_c((std::size_t)K * total_coeffs_c * 2);
        cudaMemcpy(host_a.data(), d_out_a, host_a.size() * sizeof(uint64_t),
                   cudaMemcpyDeviceToHost);
        cudaMemcpy(host_c.data(), d_out_c, host_c.size() * sizeof(uint64_t),
                   cudaMemcpyDeviceToHost);
        const auto compute_end = std::chrono::high_resolution_clock::now();

        for (int b = 0; b < K; ++b) {
            LWE::ciphertext<Rq_T<cpT>, libff::Fr<ppT>, Params> added;
            const uint64_t *ha = host_a.data() + (std::size_t)b * total_coeffs_a * 2;
            const uint64_t *hc = host_c.data() + (std::size_t)b * total_coeffs_c * 2;
            for (int i = 0; i < total_coeffs_a; i++)
                added.a_vec[i].value = ((unsigned __int128)ha[i * 2 + 1] << 64) |
                                       (unsigned __int128)ha[i * 2];
            for (int i = 0; i < total_coeffs_c; i++)
                added.c_vec[i].value = ((unsigned __int128)hc[i * 2 + 1] << 64) |
                                       (unsigned __int128)hc[i * 2];
    #ifndef NOT_PROVABLE_ZK
            LWE::re_randomize(crs.public_parameter, added);
    #endif
            added.rescale();
            proofs[live[b]] =
                r1cs_lattice_snark_proof<ppT, cpT, Params>(std::move(added));
            ok[live[b]] = 1;
        }

        cudaFree(d_pi_all);
        cudaFree(d_enc_qs);
        cudaFree(d_out_a);
        cudaFree(d_out_c);
        cudaFree(d_keys);
        libff::leave_block("Generating response (batch, GPU)");
        libff::leave_block("Call to r1cs lattice snark prover (batch)");

        using micro_s = std::chrono::microseconds;
        const double transfer_t =
            std::chrono::duration_cast<micro_s>(transfer_end - transfer_srt).count();
        const double compute_t =
            std::chrono::duration_cast<micro_s>(compute_end - compute_srt).count();
        if (gpu_time_out) *gpu_time_out = (transfer_t + compute_t) / 1e6;
        std::cout << "\n  * GPU Data Transfer: " << (transfer_t / 1e6) << "s (one "
                  << "upload for " << K << " proofs)\n"
                  << "  * GPU Computation: " << (compute_t / 1e6) << "s\n"
                  << "  * Linear comb size " << index << " x batch " << K << "\n"
                  << "  * Batch wall: "
                  << std::chrono::duration_cast<micro_s>(
                         std::chrono::high_resolution_clock::now() - t_start)
                             .count() / 1e6
                  << "s" << std::endl;
        return finish();
        }  // end else (native uint128 ring)
    }

    template <typename ppT, typename cpT, class Params>
    bool r1cs_lattice_snark_verify(
        const r1cs_lattice_snark_verification_key<ppT, cpT, Params> &vk,
        const r1cs_primary_input<libff::Fr<ppT>> &primary_input,
        const r1cs_lattice_snark_proof<ppT, cpT, Params> &proof) {
        bool res = true;
        libff::enter_block("Call to r1cs lattice snark verifier");

        libff::enter_block("Decrypting proof");
        auto decrypted = LWE::decrypt(vk.sk, proof.response, Params::rescale_q);
        libff::Fr_vector<ppT> Ap(Params::query_num), Bp(Params::query_num),
            Cp(Params::query_num), Hp(Params::query_num);

        // VERIFY IS THE ONE PART OF THIS FILE A CLIENT RUNS, AND IT WAS SERIAL.
        //
        // The work is query_num x |statement| x 3 field multiply-accumulates, and
        // iteration i touches only Ap[i]/Bp[i]/Cp[i] -- distinct elements of vectors
        // sized query_num, already allocated above. So the outer loop is independent
        // with no reduction and no shared writes; only `decrypted`, `primary_input`
        // and the vk prefix rows are read, all const here.
        //
        // Measured single-threaded: ~15.2 M statement-elements/s (Atapoor k0=100,
        // 172.7 M elements in 11.33 s). query_num is 9-11, so this caps at ~10x --
        // which is the whole gap: it takes the inline verify from 11.33 s to ~1 s,
        // against the 925 ms Atapoor reports for a circuit 63x smaller.
        //
        // Deliberately NOT parallelising over j: that needs a reduction per (i,
        // matrix), and the outer loop already saturates what query_num offers. And
        // this is r1cs_lattice_snark_VERIFY -- the prover calls a different function,
        // so nothing here can move the proving numbers.
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
        for (uint i = 0; i < Params::query_num; i++) {
            // ACCUMULATE IN LOCALS, store once.
            //
            // Not a micro-optimisation -- it is what makes the loop above
            // parallelisable at all. Ap/Bp/Cp are vectors of only query_num (9-11)
            // field elements, so Ap[i] and Ap[i+1] sit in the same cache line.
            // Accumulating directly into them means every one of the millions of +=
            // in the inner loop is a read-modify-write on a line another thread is
            // also writing: measured, that took the inline verify from 11.3 s to
            // 93 s. Bounding the thread pool did not help, because the cost is
            // coherence traffic, not oversubscription.
            libff::Fr<ppT> a = decrypted[i * LWE::query_size];
            libff::Fr<ppT> b = decrypted[i * LWE::query_size + 1];
            libff::Fr<ppT> c = decrypted[i * LWE::query_size + 2];
            Hp[i] = decrypted[i * LWE::query_size + 3];

            a += vk.A_prefix[i][0];
            b += vk.B_prefix[i][0];
            c += vk.C_prefix[i][0];

            for (uint64_t j = 0; j < primary_input.size(); j++) {
                a += primary_input[j] * vk.A_prefix[i][j + 1];
                b += primary_input[j] * vk.B_prefix[i][j + 1];
                c += primary_input[j] * vk.C_prefix[i][j + 1];
            }
            Ap[i] = a;
            Bp[i] = b;
            Cp[i] = c;
        }
        libff::leave_block("Decrypting proof");

        libff::enter_block("Checking QAP divisibility");
        for (uint i = 0; i < Params::query_num; i++) {
            if (Ap[i] * Bp[i] != Hp[i] * vk.Z_s[i] + Cp[i]) {
                if (!libff::inhibit_profiling_info) {
                    libff::print_indent();
                    printf("QAP divisibility check failed.\n");
                }
                res = false;
            }
        }
        libff::leave_block("Checking QAP divisibility");

        libff::leave_block("Call to r1cs lattice snark verifier");
        return res;
    }

    template <typename ppT, typename cpT, class Params>
    bool run_r1cs_lattice_snark(const r1cs_example<libff::Fr<ppT>> &example) {
        libff::enter_block("Call to R1CS lattice SNARK");

        libff::print_header("R1CS lattice SNARK Generator");
        r1cs_lattice_snark_crs<ppT, cpT, Params> crs;
        r1cs_lattice_snark_verification_key<ppT, cpT, Params> vk;
        r1cs_lattice_snark_generator<ppT, cpT, Params>(
            example.constraint_system, crs, vk);
        printf("\n");
        libff::print_indent();
        libff::print_mem("after generator");

        libff::print_header("R1CS lattice SNARK Prover");
        r1cs_lattice_snark_proof<ppT, cpT, Params> proof =
            r1cs_lattice_snark_prove<ppT, cpT, Params>(
                crs, example.primary_input, example.auxiliary_input);
        printf("\n");
        libff::print_indent();
        libff::print_mem("after prover");

        libff::print_header("R1CS lattice SNARK Verifier");
        const bool ans = r1cs_lattice_snark_verify<ppT, cpT>(
            vk, example.primary_input, proof);
        printf("\n");
        libff::print_indent();
        libff::print_mem("after verifier");
        printf("* The verification result is: %s\n", (ans ? "PASS" : "FAIL"));

        libff::leave_block("Call to R1CS lattice SNARK");
        return ans;
    }

    template <typename ppT, typename cpT, class Params>
    void test_r1cs_lattice_snark(size_t num_constraints, size_t input_size) {
        libff::print_header("(enter) Test R1CS lattice SNARK");
        r1cs_example<libff::Fr<ppT>> example =
            generate_r1cs_example_with_field_input<libff::Fr<ppT>>(
                num_constraints, input_size);
        const bool res = run_r1cs_lattice_snark<ppT, cpT, Params>(example);
        if (!res)
            libff::print_header("TEST FAILED");

        libff::print_header("(leave) Test R1CS lattice SNARK");
    }
}

#endif
