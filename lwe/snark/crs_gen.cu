// crs_gen.cu — GPU acceleration of the CRS keygen hot loop.
//
// The lattice-SNARK generator's "Generating CRS and VK" phase is dominated by
// encrypt_query_matrix (r1cs_lattice_snark_common.hpp): it LWE-encrypts every row
// of the query matrix q_mat, ~25M rows for the Lattigo relin circuit, ~111 min on
// CPU. Each row is independent, so it maps perfectly onto the GPU.
//
// Per row i, noiseless encryption (encrypt_query_matrix calls encrypt(..., /*noise*/false),
// then adds the Gaussian noise in a separate pass which stays on the host):
//
//     c_vec[out] = lift(uv[i][out]) + sum_{k=0}^{n-1} S_T[out][k] * a_vec[i][k]   (mod q)
//

#include <cstdint>
#include <cstdio>
#include <cuda_runtime.h>
#include <chrono>

namespace {

// ---- AES-128 (copied verbatim from proof.cu; static = TU-local) ----

__constant__ uint8_t crs_sbox[256] = {
    0x63, 0x7c, 0x77, 0x7b, 0xf2, 0x6b, 0x6f, 0xc5, 0x30, 0x01, 0x67, 0x2b, 0xfe, 0xd7, 0xab, 0x76,
    0xca, 0x82, 0xc9, 0x7d, 0xfa, 0x59, 0x47, 0xf0, 0xad, 0xd4, 0xa2, 0xaf, 0x9c, 0xa4, 0x72, 0xc0,
    0xb7, 0xfd, 0x93, 0x26, 0x36, 0x3f, 0xf7, 0xcc, 0x34, 0xa5, 0xe5, 0xf1, 0x71, 0xd8, 0x31, 0x15,
    0x04, 0xc7, 0x23, 0xc3, 0x18, 0x96, 0x05, 0x9a, 0x07, 0x12, 0x80, 0xe2, 0xeb, 0x27, 0xb2, 0x75,
    0x09, 0x83, 0x2c, 0x1a, 0x1b, 0x6e, 0x5a, 0xa0, 0x52, 0x3b, 0xd6, 0xb3, 0x29, 0xe3, 0x2f, 0x84,
    0x53, 0xd1, 0x00, 0xed, 0x20, 0xfc, 0xb1, 0x5b, 0x6a, 0xcb, 0xbe, 0x39, 0x4a, 0x4c, 0x58, 0xcf,
    0xd0, 0xef, 0xaa, 0xfb, 0x43, 0x4d, 0x33, 0x85, 0x45, 0xf9, 0x02, 0x7f, 0x50, 0x3c, 0x9f, 0xa8,
    0x51, 0xa3, 0x40, 0x8f, 0x92, 0x9d, 0x38, 0xf5, 0xbc, 0xb6, 0xda, 0x21, 0x10, 0xff, 0xf3, 0xd2,
    0xcd, 0x0c, 0x13, 0xec, 0x5f, 0x97, 0x44, 0x17, 0xc4, 0xa7, 0x7e, 0x3d, 0x64, 0x5d, 0x19, 0x73,
    0x60, 0x81, 0x4f, 0xdc, 0x22, 0x2a, 0x90, 0x88, 0x46, 0xee, 0xb8, 0x14, 0xde, 0x5e, 0x0b, 0xdb,
    0xe0, 0x32, 0x3a, 0x0a, 0x49, 0x06, 0x24, 0x5c, 0xc2, 0xd3, 0xac, 0x62, 0x91, 0x95, 0xe4, 0x79,
    0xe7, 0xc8, 0x37, 0x6d, 0x8d, 0xd5, 0x4e, 0xa9, 0x6c, 0x56, 0xf4, 0xea, 0x65, 0x7a, 0xae, 0x08,
    0xba, 0x78, 0x25, 0x2e, 0x1c, 0xa6, 0xb4, 0xc6, 0xe8, 0xdd, 0x74, 0x1f, 0x4b, 0xbd, 0x8b, 0x8a,
    0x70, 0x3e, 0xb5, 0x66, 0x48, 0x03, 0xf6, 0x0e, 0x61, 0x35, 0x57, 0xb9, 0x86, 0xc1, 0x1d, 0x9e,
    0xe1, 0xf8, 0x98, 0x11, 0x69, 0xd9, 0x8e, 0x94, 0x9b, 0x1e, 0x87, 0xe9, 0xce, 0x55, 0x28, 0xdf,
    0x8c, 0xa1, 0x89, 0x0d, 0xbf, 0xe6, 0x42, 0x68, 0x41, 0x99, 0x2d, 0x0f, 0xb0, 0x54, 0xbb, 0x16};

union Block128 {
  uint64_t u64[2];
  uint8_t u8[16];
};

__device__ inline uint8_t galois_mul2(uint8_t v) { return (v << 1) ^ ((v >> 7) * 0x1b); }

__device__ inline void ShiftRows(uint8_t *s) {
  uint8_t t;
  t = s[1]; s[1] = s[5]; s[5] = s[9]; s[9] = s[13]; s[13] = t;
  t = s[2]; s[2] = s[10]; s[10] = t;
  t = s[6]; s[6] = s[14]; s[14] = t;
  t = s[15]; s[15] = s[11]; s[11] = s[7]; s[7] = s[3]; s[3] = t;
}

__device__ inline void MixColumns(uint8_t *s) {
  uint8_t tmp, tm, t;
  for (int i = 0; i < 4; i++) {
    t = s[i * 4];
    tmp = s[i * 4] ^ s[i * 4 + 1] ^ s[i * 4 + 2] ^ s[i * 4 + 3];
    tm = galois_mul2(s[i * 4] ^ s[i * 4 + 1]); s[i * 4] ^= tm ^ tmp;
    tm = galois_mul2(s[i * 4 + 1] ^ s[i * 4 + 2]); s[i * 4 + 1] ^= tm ^ tmp;
    tm = galois_mul2(s[i * 4 + 2] ^ s[i * 4 + 3]); s[i * 4 + 2] ^= tm ^ tmp;
    tm = galois_mul2(s[i * 4 + 3] ^ t); s[i * 4 + 3] ^= tm ^ tmp;
  }
}

__device__ inline void AddRoundKey(uint8_t *s, const uint32_t *rk, int round) {
  for (int i = 0; i < 4; i++) {
    uint32_t k = rk[round * 4 + i];
    s[i * 4 + 0] ^= (k >> 0) & 0xFF;
    s[i * 4 + 1] ^= (k >> 8) & 0xFF;
    s[i * 4 + 2] ^= (k >> 16) & 0xFF;
    s[i * 4 + 3] ^= (k >> 24) & 0xFF;
  }
}

__device__ inline void SubBytes(uint8_t *s, const uint8_t *sbox) {
#pragma unroll
  for (int i = 0; i < 16; i++) s[i] = sbox[s[i]];
}

__device__ inline void AES_ecb(Block128 *blk, const uint32_t *rk, const uint8_t *sbox) {
  uint8_t *s = blk->u8;
  AddRoundKey(s, rk, 0);
  for (int round = 1; round < 10; round++) {
    SubBytes(s, sbox); ShiftRows(s); MixColumns(s); AddRoundKey(s, rk, round);
  }
  SubBytes(s, sbox); ShiftRows(s); AddRoundKey(s, rk, 10);
}

// a_vec element at global counter c = the FULL 128-bit AES(c). The CPU keygen's
// Ring(T) stores the raw AES output UNMASKED (ring arithmetic is raw uint128; the
// 2^q_log modulus is logical, applied later by the prover/verify), so for a
// byte-exact CRS we must NOT mask here. block = [c_lo, 0] (counter < 2^64).
struct u128 { uint64_t lo, hi; };

__device__ inline u128 aes_a_element(uint64_t counter, const uint32_t *rk,
                                     const uint8_t *sbox) {
  Block128 b;
  b.u64[0] = counter;
  b.u64[1] = 0;
  AES_ecb(&b, rk, sbox);
  u128 r;
  r.lo = b.u64[0];
  r.hi = b.u64[1];
  return r;
}

__device__ inline void add128(u128 *acc, u128 b) {
  asm volatile("add.cc.u64 %0, %0, %2;\n\t addc.u64 %1, %1, %3;\n\t"
               : "+l"(acc->lo), "+l"(acc->hi)
               : "l"(b.lo), "l"(b.hi));
}

// (a*b) mod 2^128 (low 128 bits of the full product). Caller masks hi to q_log-64
// bits for the q=2^q_log reduction. Cross terms beyond bit 127 do not affect the low
// q_log<=128 bits, so this is exact mod 2^q_log.
__device__ inline u128 mul128_lo(u128 a, u128 b) {
  u128 r;
  uint64_t hi0, cross;
  asm volatile(
      "mul.lo.u64 %0, %2, %3;\n\t"   // r.lo = a.lo*b.lo (low)
      "mul.hi.u64 %1, %2, %3;\n\t"   // hi0  = a.lo*b.lo (high)
      : "=l"(r.lo), "=l"(hi0)
      : "l"(a.lo), "l"(b.lo));
  cross = a.lo * b.hi + a.hi * b.lo;  // contributes to bits [64,128)
  r.hi = hi0 + cross;
  return r;
}

// One block per query row. blockDim should be a power of two (e.g. 256). The n a-vec
// elements are generated cooperatively into DYNAMIC shared memory (n can be ~4580 =>
// 71KB, beyond the 48KB static limit), then reused across all cdim outputs (a_vec is
// shared by every output of a row). The launcher opts into the larger shared limit and
// passes n*sizeof(u128) as the dynamic shared size.
__global__ void crs_encrypt_kernel(const u128 *__restrict__ S_T,  // [cdim*n]
                                   const uint64_t *__restrict__ uv, // [rows*cdim]
                                   const uint32_t *__restrict__ aes_keys, // [44]
                                   u128 *__restrict__ enc_qs,       // [rows*cdim]
                                   uint64_t rows, uint64_t row_offset,
                                   uint32_t n, uint32_t cdim,
                                   uint64_t mask_lo, uint64_t mask_hi) {
  extern __shared__ u128 a_sh[];
  __shared__ uint32_t s_keys[44];
  __shared__ uint8_t s_sbox[256];
  const int tid = threadIdx.x;
  if (tid < 44) s_keys[tid] = aes_keys[tid];
  if (tid < 256) s_sbox[tid] = crs_sbox[tid];
  __syncthreads();

  for (uint64_t row = blockIdx.x; row < rows; row += gridDim.x) {
    // AES a_vec counter keys off the ABSOLUTE row index so row-chunked launches
    // (row_offset>0, uv/enc_qs indexed chunk-locally) are byte-identical to a
    // single-shot launch (row_offset=0). Mirrors crs_encrypt_kernel_big.
    const uint64_t base = (row_offset + row) * (uint64_t)n;
    for (uint32_t k = tid; k < n; k += blockDim.x)
      a_sh[k] = aes_a_element(base + k, s_keys, s_sbox);
    __syncthreads();

    for (uint32_t out = tid; out < cdim; out += blockDim.x) {
      u128 acc;
      acc.lo = uv[row * cdim + out];  // lift(uv[out]) (field value < p < 2^64)
      acc.hi = 0;
      const u128 *srow = S_T + (uint64_t)out * n;
      for (uint32_t k = 0; k < n; k++) {
        u128 p = mul128_lo(srow[k], a_sh[k]);
        add128(&acc, p);
      }
      // NO final mask: the CPU ring stores the raw uint128 (mod 2^128) accumulation
      // and reduces mod 2^q_log only later (prover/verify). a_vec is already masked
      // (matching the prover's generate_a_vec_element). Masking c_vec here would zero
      // bits q_log..127 that the CPU keeps -> byte mismatch.
      enc_qs[row * cdim + out] = acc;
    }
    __syncthreads();
  }
}

// ============================================================================
// Big-int (256-bit) variant — GPU CRS keygen for the 60-bit-q0 RingBig path.
//
// Mirrors crs_encrypt_kernel but the ring is RingBig<uint256, q_log>: S_T and
// enc_qs are 256-bit (4x u64), and each a_vec element is TWO 128-bit AES blocks
// (RingBig::random_element: lo=AES(c), hi=AES(c+1), value = lo | hi<<128, masked
// to q_log). The CPU consumes 2 PRG blocks per element, so for row r element k
// the counters are base+2k and base+2k+1 with base = r*2n. All 256-bit ops use
// device `unsigned __int128` carry chains IDENTICAL to the CPU uint256 ops
// (operator+=/operator*=), so the GPU CRS is byte-exact with the CPU keygen.
//
// a_vec (n=8192 -> 256KB of u256) exceeds the shared-memory limit, so it is
// TILED: each pass loads a tile of a_vec into shared and every owned output
// accumulates its partial sum over the tile (acc persists in registers across
// tiles). cdim outputs map one-per-thread (cdim<=blockDim required).

struct u256 { uint64_t w[4]; };

__device__ inline void add256(u256 &a, const u256 &b) {
  unsigned __int128 carry = 0;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    unsigned __int128 s = (unsigned __int128)a.w[i] + b.w[i] + carry;
    a.w[i] = (uint64_t)s;
    carry = s >> 64;
  }
}

// Low 256 bits of a*b (schoolbook), identical to uint256::operator*=.
__device__ inline u256 mul256_lo(const u256 &a, const u256 &b) {
  uint64_t res[4] = {0, 0, 0, 0};
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    if (!a.w[i]) continue;
    unsigned __int128 carry = 0;
    for (int j = 0; i + j < 4; ++j) {
      unsigned __int128 cur = (unsigned __int128)res[i + j] +
                              (unsigned __int128)a.w[i] * b.w[j] + carry;
      res[i + j] = (uint64_t)cur;
      carry = cur >> 64;
    }
  }
  u256 r;
  r.w[0] = res[0]; r.w[1] = res[1]; r.w[2] = res[2]; r.w[3] = res[3];
  return r;
}

// a_vec element at element-counter c: two AES blocks (c, c+1) -> 256 bits, masked
// to q_log. Matches RingBig::random_element (next_prg_block twice).
__device__ inline u256 aes_a_element_big(uint64_t c, const uint32_t *rk,
                                         const uint8_t *sbox, const u256 &mask) {
  Block128 b0, b1;
  b0.u64[0] = c;     b0.u64[1] = 0;
  b1.u64[0] = c + 1; b1.u64[1] = 0;
  AES_ecb(&b0, rk, sbox);
  AES_ecb(&b1, rk, sbox);
  u256 v;
  v.w[0] = b0.u64[0] & mask.w[0];
  v.w[1] = b0.u64[1] & mask.w[1];
  v.w[2] = b1.u64[0] & mask.w[2];
  v.w[3] = b1.u64[1] & mask.w[3];
  return v;
}

__global__ void crs_encrypt_kernel_big(const u256 *__restrict__ S_T,   // [cdim*n]
                                       const uint64_t *__restrict__ uv, // [rows*cdim]
                                       const uint32_t *__restrict__ aes_keys,
                                       u256 *__restrict__ enc_qs,        // [rows*cdim]
                                       uint64_t rows, uint64_t row_offset,
                                       uint32_t n, uint32_t cdim,
                                       uint32_t tile,
                                       uint64_t m0, uint64_t m1, uint64_t m2,
                                       uint64_t m3) {
  extern __shared__ u256 a_tile[];  // `tile` u256 elements
  __shared__ uint32_t s_keys[44];
  __shared__ uint8_t s_sbox[256];
  const int tid = threadIdx.x;
  if (tid < 44) s_keys[tid] = aes_keys[tid];
  if (tid < 256) s_sbox[tid] = crs_sbox[tid];
  __syncthreads();
  u256 mask; mask.w[0] = m0; mask.w[1] = m1; mask.w[2] = m2; mask.w[3] = m3;

  const bool own = (uint32_t)tid < cdim;
  for (uint64_t row = blockIdx.x; row < rows; row += gridDim.x) {
    // AES a_vec counter keys off the ABSOLUTE row index so row-chunked launches
    // (row_offset>0, uv/enc_qs indexed chunk-locally) are byte-identical to a
    // single-shot launch (row_offset=0).
    const uint64_t base = (row_offset + row) * (uint64_t)n * 2;  // 2 AES blocks per a_vec element
    u256 acc;
    if (own) {
      acc.w[0] = uv[row * cdim + tid];  // lift(uv[out]) (field value < 2^64)
      acc.w[1] = 0; acc.w[2] = 0; acc.w[3] = 0;
    }
    const u256 *srow = own ? (S_T + (uint64_t)tid * n) : nullptr;
    for (uint32_t t0 = 0; t0 < n; t0 += tile) {
      const uint32_t tl = (n - t0 < tile) ? (n - t0) : tile;
      for (uint32_t k = tid; k < tl; k += blockDim.x)
        a_tile[k] = aes_a_element_big(base + 2 * (uint64_t)(t0 + k), s_keys,
                                      s_sbox, mask);
      __syncthreads();
      if (own) {
        for (uint32_t k = 0; k < tl; k++) {
          u256 p = mul256_lo(srow[t0 + k], a_tile[k]);
          add256(acc, p);
        }
      }
      __syncthreads();
    }
    if (own) enc_qs[row * cdim + tid] = acc;  // raw mod 2^256 (CPU masks later)
  }
}

}  // namespace

// Host launcher. All pointers are HOST arrays; device memory is managed internally.
//   h_S_T : cdim*n   * 2 uint64 (lo,hi per element), row-major [out*n + k]
//   h_uv  : rows*cdim    uint64 (field residue per (row,out))
//   h_keys: 44          uint32  (AES-128 expanded round keys, == CRS aes key)
//   h_enc : rows*cdim * 2 uint64 (output, [row*cdim + out] -> (lo,hi))
// Returns 0 on success, nonzero cudaError on failure.
extern "C" int launch_crs_encrypt(const uint64_t *h_S_T, const uint64_t *h_uv,
                                  const uint32_t *h_keys, uint64_t rows, uint32_t n,
                                  uint32_t cdim, uint64_t mask_lo, uint64_t mask_hi,
                                  uint64_t *h_enc) {
  const size_t st_elems = (size_t)cdim * n;          // u128 each (row-independent)

  // Chunk over ROWS so the per-chunk device working set (d_enc = rows*cdim*16B +
  // d_uv = rows*cdim*8B) stays bounded. A single-shot d_enc OOMs a 46GB GPU once the
  // key-switch circuits grow (blueprint Lq_pad=7 => ~15.6M constraints), which is
  // exactly what the u256 path already guards against. Budget the enc+uv working set
  // to ~10GB/chunk; d_S_T (cdim*n, row-independent) stays resident. The AES a_vec
  // counter keys off the ABSOLUTE row index (row_offset), so the chunked result is
  // byte-identical to a single-shot launch.
  const size_t row_bytes = (size_t)cdim * (sizeof(u128) + sizeof(uint64_t));
  size_t chunk_rows = (size_t)((10ULL << 30) / (row_bytes ? row_bytes : 1));
  if (chunk_rows == 0) chunk_rows = 1;
  if ((uint64_t)chunk_rows > rows) chunk_rows = (size_t)rows;

  u128 *d_S_T = nullptr, *d_enc = nullptr;
  uint64_t *d_uv = nullptr;
  uint32_t *d_keys = nullptr;
  cudaError_t e;
#define CK(call) do { e = (call); if (e != cudaSuccess) { \
    fprintf(stderr, "[crs_gen] %s: %s\n", #call, cudaGetErrorString(e)); goto fail; } } while (0)

  CK(cudaMalloc(&d_S_T, st_elems * sizeof(u128)));
  CK(cudaMalloc(&d_uv, (size_t)chunk_rows * cdim * sizeof(uint64_t)));
  CK(cudaMalloc(&d_keys, 44 * sizeof(uint32_t)));
  CK(cudaMalloc(&d_enc, (size_t)chunk_rows * cdim * sizeof(u128)));

  CK(cudaMemcpy(d_S_T, h_S_T, st_elems * sizeof(u128), cudaMemcpyHostToDevice));
  CK(cudaMemcpy(d_keys, h_keys, 44 * sizeof(uint32_t), cudaMemcpyHostToDevice));

  {
    const int threads = 256;
    // a_vec lives in DYNAMIC shared memory (n u128). Opt into the larger per-block
    // shared limit (L40S allows up to ~99KB); n=4580 => ~72KB.
    const size_t shmem = (size_t)n * sizeof(u128);
    CK(cudaFuncSetAttribute(crs_encrypt_kernel,
                            cudaFuncAttributeMaxDynamicSharedMemorySize, (int)shmem));
    // Per-phase wall clock, so a keygen log says where a 2^14 key's minutes go
    // (H2D of uv, kernel, D2H into enc_qs) instead of one opaque block time.
    double t_h2d = 0, t_kern = 0, t_d2h = 0;
    auto now = [] { return std::chrono::duration<double>(
        std::chrono::steady_clock::now().time_since_epoch()).count(); };
    // One row-chunk at a time; uv/enc are indexed chunk-locally, row_offset makes
    // the AES stream absolute so the output matches an unchunked run bit-for-bit.
    for (uint64_t off = 0; off < rows; off += chunk_rows) {
      const uint64_t this_rows =
          (rows - off < (uint64_t)chunk_rows) ? (rows - off) : (uint64_t)chunk_rows;
      double t0 = now();
      CK(cudaMemcpy(d_uv, h_uv + off * cdim,
                    (size_t)this_rows * cdim * sizeof(uint64_t),
                    cudaMemcpyHostToDevice));
      double t1 = now(); t_h2d += t1 - t0;
      // Cap grid; the kernel grid-strides over rows.
      const int blocks = (int)((this_rows < 65535) ? this_rows : 65535);
      crs_encrypt_kernel<<<blocks, threads, shmem>>>(d_S_T, d_uv, d_keys, d_enc,
                                                     this_rows, off, n, cdim,
                                                     mask_lo, mask_hi);
      CK(cudaGetLastError());
      CK(cudaDeviceSynchronize());
      double t2 = now(); t_kern += t2 - t1;
      CK(cudaMemcpy(h_enc + off * cdim * 2, d_enc,
                    (size_t)this_rows * cdim * sizeof(u128),
                    cudaMemcpyDeviceToHost));
      t_d2h += now() - t2;
    }
    fprintf(stderr,
            "[crs_gen] encrypt rows=%llu n=%u cdim=%u chunks=%llu: H2D %.1fs, kernel %.1fs, D2H %.1fs (%.1f GB out)\n",
            (unsigned long long)rows, n, cdim,
            (unsigned long long)((rows + chunk_rows - 1) / chunk_rows), t_h2d, t_kern, t_d2h,
            (double)rows * cdim * sizeof(u128) / 1e9);
  }
  e = cudaSuccess;
fail:
  if (d_S_T) cudaFree(d_S_T);
  if (d_uv) cudaFree(d_uv);
  if (d_keys) cudaFree(d_keys);
  if (d_enc) cudaFree(d_enc);
  return (int)e;
#undef CK
}

// Big-int (256-bit) host launcher. Layout mirrors launch_crs_encrypt but every
// S_T / enc element is 4 uint64 (u256). mask is the 4-limb 2^q_log-1 mask.
//   h_S_T : cdim*n  * 4 uint64 (w0..w3), row-major [out*n + k]
//   h_uv  : rows*cdim   uint64 (field residue, lifted into the ring low limb)
//   h_keys: 44          uint32 (AES round keys = crs aes key)
//   h_enc : rows*cdim * 4 uint64 (output [row*cdim + out] -> w0..w3)
extern "C" int launch_crs_encrypt_big(const uint64_t *h_S_T, const uint64_t *h_uv,
                                      const uint32_t *h_keys, uint64_t rows,
                                      uint32_t n, uint32_t cdim, uint64_t m0,
                                      uint64_t m1, uint64_t m2, uint64_t m3,
                                      uint64_t *h_enc) {
  const size_t st_elems = (size_t)cdim * n;     // u256 each (fixed, not chunked)

  // Chunk over ROWS so the per-chunk device working set (d_enc = rows*cdim*32B +
  // d_uv = rows*cdim*8B) stays small. The full d_enc OOMs a 46GB GPU for the large
  // key-switch circuits (60-bit q0 relin/rotate). Budget the enc+uv working set to
  // ~10GB/chunk; d_S_T (cdim*n, row-independent) stays resident. The AES a_vec
  // counter keys off the ABSOLUTE row index (row_offset), so the chunked result is
  // byte-identical to a single-shot launch.
  const size_t row_bytes = (size_t)cdim * (sizeof(u256) + sizeof(uint64_t));
  size_t chunk_rows = (size_t)((10ULL << 30) / (row_bytes ? row_bytes : 1));
  if (chunk_rows == 0) chunk_rows = 1;
  if ((uint64_t)chunk_rows > rows) chunk_rows = (size_t)rows;

  u256 *d_S_T = nullptr, *d_enc = nullptr;
  uint64_t *d_uv = nullptr;
  uint32_t *d_keys = nullptr;
  cudaError_t e;
#define CKB(call) do { e = (call); if (e != cudaSuccess) { \
    fprintf(stderr, "[crs_gen big] %s: %s\n", #call, cudaGetErrorString(e)); goto failb; } } while (0)

  if (cdim > 256) {
    fprintf(stderr, "[crs_gen big] cdim=%u > 256 unsupported (one output per thread)\n", cdim);
    return -1;
  }
  CKB(cudaMalloc(&d_S_T, st_elems * sizeof(u256)));
  CKB(cudaMalloc(&d_keys, 44 * sizeof(uint32_t)));
  CKB(cudaMalloc(&d_uv, (size_t)chunk_rows * cdim * sizeof(uint64_t)));
  CKB(cudaMalloc(&d_enc, (size_t)chunk_rows * cdim * sizeof(u256)));

  CKB(cudaMemcpy(d_S_T, h_S_T, st_elems * sizeof(u256), cudaMemcpyHostToDevice));
  CKB(cudaMemcpy(d_keys, h_keys, 44 * sizeof(uint32_t), cudaMemcpyHostToDevice));

  {
    const int threads = 256;
    // Tile a_vec to fit shared. Cap the tile so tile*32 bytes <= ~96KB; also no
    // larger than n.
    uint32_t tile = n;
    const uint32_t max_tile = 3072;  // 3072 * 32 = 96KB
    if (tile > max_tile) tile = max_tile;
    const size_t shmem = (size_t)tile * sizeof(u256);
    CKB(cudaFuncSetAttribute(crs_encrypt_kernel_big,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, (int)shmem));
    // One row-chunk at a time; uv/enc are indexed chunk-locally, row_offset makes
    // the AES stream absolute so the output matches an unchunked run bit-for-bit.
    for (uint64_t off = 0; off < rows; off += chunk_rows) {
      const uint64_t this_rows =
          (rows - off < (uint64_t)chunk_rows) ? (rows - off) : (uint64_t)chunk_rows;
      CKB(cudaMemcpy(d_uv, h_uv + off * cdim,
                     (size_t)this_rows * cdim * sizeof(uint64_t),
                     cudaMemcpyHostToDevice));
      const int blocks = (int)((this_rows < 65535) ? this_rows : 65535);
      crs_encrypt_kernel_big<<<blocks, threads, shmem>>>(
          d_S_T, d_uv, d_keys, d_enc, this_rows, off, n, cdim, tile, m0, m1, m2, m3);
      CKB(cudaGetLastError());
      CKB(cudaDeviceSynchronize());
      CKB(cudaMemcpy(h_enc + off * cdim * 4, d_enc,
                     (size_t)this_rows * cdim * sizeof(u256),
                     cudaMemcpyDeviceToHost));
    }
  }
  e = cudaSuccess;
failb:
  if (d_S_T) cudaFree(d_S_T);
  if (d_uv) cudaFree(d_uv);
  if (d_keys) cudaFree(d_keys);
  if (d_enc) cudaFree(d_enc);
  return (int)e;
#undef CKB
}
