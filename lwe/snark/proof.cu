#include <cuda_runtime.h>
#include <stdint.h>

#include <cuda_runtime.h>
#include <stdint.h>
// __constant__ uint32_t d_const_aes_keys[44];
// CUDA constant memory for the AES S-box (cached and ultra-fast for parallel reads)
__constant__ uint8_t d_sbox[256] = {
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
    0x8c, 0xa1, 0x89, 0x0d, 0xbf, 0xe6, 0x42, 0x68, 0x41, 0x99, 0x2d, 0x0f, 0xb0, 0x54, 0xbb, 0x16
};

// 128-bit block structure matching your PRG logic
union Block128 {
    uint64_t u64[2];
    uint32_t u32[4];
    uint8_t  u8[16];
};

__device__ inline uint8_t galois_mul2(uint8_t value) {
    return (value << 1) ^ ((value >> 7) * 0x1b);
}

__device__ void ShiftRows(uint8_t* state) {
    uint8_t temp;
    // Row 1
    temp = state[1]; state[1] = state[5]; state[5] = state[9]; state[9] = state[13]; state[13] = temp;
    // Row 2
    temp = state[2]; state[2] = state[10]; state[10] = temp;
    temp = state[6]; state[6] = state[14]; state[14] = temp;
    // Row 3
    temp = state[15]; state[15] = state[11]; state[11] = state[7]; state[7] = state[3]; state[3] = temp;
}

__device__ void MixColumns(uint8_t* state) {
    uint8_t tmp, tm, t;
    for (int i = 0; i < 4; i++) {
        t   = state[i * 4];
        tmp = state[i * 4] ^ state[i * 4 + 1] ^ state[i * 4 + 2] ^ state[i * 4 + 3];
        tm  = state[i * 4] ^ state[i * 4 + 1]; tm = galois_mul2(tm); state[i * 4] ^= tm ^ tmp;
        tm  = state[i * 4 + 1] ^ state[i * 4 + 2]; tm = galois_mul2(tm); state[i * 4 + 1] ^= tm ^ tmp;
        tm  = state[i * 4 + 2] ^ state[i * 4 + 3]; tm = galois_mul2(tm); state[i * 4 + 2] ^= tm ^ tmp;
        tm  = state[i * 4 + 3] ^ t;            tm = galois_mul2(tm); state[i * 4 + 3] ^= tm ^ tmp;
    }
}

__device__ void AddRoundKey(uint8_t* state, const uint32_t* round_key, int round) {
    for (int i = 0; i < 4; i++) {
        // Extract bytes from the 32-bit round key word
        uint32_t k = round_key[round * 4 + i];
        state[i * 4 + 0] ^= (k >> 0)  & 0xFF;
        state[i * 4 + 1] ^= (k >> 8)  & 0xFF;
        state[i * 4 + 2] ^= (k >> 16) & 0xFF;
        state[i * 4 + 3] ^= (k >> 24) & 0xFF;
    }
}

__device__ inline void SubBytes(uint8_t* state, const uint8_t* shared_sbox) {
    #pragma unroll
    for (int i = 0; i < 16; i++) {
        // Now reading from the multi-banked L1 cache instead of Constant Memory!
        state[i] = shared_sbox[state[i]]; 
    }
}
// =====================================================================
// You will need to drop in a standard CUDA AES single-block encryptor here.
// There are many open-source lightweight implementations available.
// It takes a 16-byte plaintext (the counter) and encrypts it in-place.
// =====================================================================
// This is the direct replacement for your CPU's AES_ecb_encrypt_blk
// Update the signature to accept the shared S-Box
__device__ void AES_ecb_encrypt_blk_gpu(Block128* block, const uint32_t* round_keys, const uint8_t* shared_sbox) {
    uint8_t* state = block->u8;

    AddRoundKey(state, round_keys, 0);

    for (int round = 1; round < 10; round++) {
        SubBytes(state, shared_sbox); // Pass the shared cache here!
        ShiftRows(state);
        MixColumns(state);
        AddRoundKey(state, round_keys, round);
    }

    SubBytes(state, shared_sbox); // And here!
    ShiftRows(state);
    AddRoundKey(state, round_keys, 10);
}
#include <cuda_runtime.h>
#include <stdint.h>

// CUDA representation of __int128
struct uint128_cuda {
    uint64_t lo;
    uint64_t hi;
};

// 128-bit addition using raw PTX hardware carry flags
__device__ inline void add128(uint128_cuda* accum, uint128_cuda b) {
    asm volatile(
        "add.cc.u64 %0, %0, %2;\n\t"  // Add lower 64 bits and set carry flag (.cc)
        "addc.u64 %1, %1, %3;\n\t"    // Add upper 64 bits plus the carry flag (addc)
        : "+l"(accum->lo), "+l"(accum->hi) 
        : "l"(b.lo), "l"(b.hi)
    );
}
// 128-bit Multiply-Accumulate using raw PTX hardware instructions
__device__ inline void mac128(uint128_cuda* accum, uint128_cuda a, uint64_t b) {
    uint64_t r0, r1, hi_prod;
    asm volatile(
        "mul.lo.u64 %0, %5, %7;\n\t"       // r0 = a.lo * b
        "mul.hi.u64 %1, %5, %7;\n\t"       // r1 = umulhi(a.lo, b)
        "mul.lo.u64 %2, %6, %7;\n\t"       // hi_prod = a.hi * b
        "add.u64 %1, %1, %2;\n\t"          // r1 += hi_prod
        "add.cc.u64 %3, %3, %0;\n\t"       // accum.lo += r0 (set carry flag)
        "addc.u64 %4, %4, %1;\n\t"         // accum.hi += r1 (add with carry flag)
        
        // --- OUTPUTS (Modifiers '=' and '+' go here) ---
        : "=l"(r0), "=l"(r1), "=l"(hi_prod), "+l"(accum->lo), "+l"(accum->hi)
        
        // --- INPUTS (No modifiers allowed here) ---
        : "l"(a.lo), "l"(a.hi), "l"(b)
    );
}

__device__ void generate_a_vec_element(
    uint64_t absolute_counter_start, 
    uint128_cuda* out_element, 
    uint64_t mod_mask_lo, 
    uint64_t mod_mask_hi,
    const uint8_t* shared_sbox,
    const uint32_t* shared_aes_keys) 
{
    uint64_t block_u64[2];
    block_u64[0] = absolute_counter_start; 
    block_u64[1] = 0; 
    
    // Hand the ultra-fast L1 cache pointer down to the AES hardware
    AES_ecb_encrypt_blk_gpu((Block128*)block_u64, shared_aes_keys, shared_sbox);
    
    // Apply the modulo mask
    out_element->lo = block_u64[0] & mod_mask_lo;
    out_element->hi = block_u64[1] & mod_mask_hi; 
}
// -----------------------------------------------------------------

__global__ void accumulate_c_vec(
    const ulonglong2* __restrict__ enc_qs_raw, 
    const uint64_t* __restrict__ pi, 
    ulonglong2* __restrict__ out_c_vec_raw, 
    int index, int total_coeffs) 
{
    // Each BLOCK calculates exactly one coefficient
    int coeff_idx = blockIdx.x; 
    int tid = threadIdx.x;

    // Fast L1 cache to hold the sub-totals for the 256 threads
    __shared__ uint128_cuda sdata[256];

    uint128_cuda accum = {0, 0};

    // Grid-Stride Loop: 256 threads work together to chew through the 700k elements
    for (int i = tid; i < index; i += blockDim.x) {
        uint64_t qs_idx = ((uint64_t)i * total_coeffs) + coeff_idx;
        
        ulonglong2 val = enc_qs_raw[qs_idx];
        uint128_cuda enc_val = {val.x, val.y};

        mac128(&accum, enc_val, pi[i]);
    }

    // Save thread's partial sum to shared memory
    sdata[tid] = accum;
    __syncthreads();

    // Parallel Reduction: Collapse the 256 partial sums into 1 final answer
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s && (tid + s) < blockDim.x) {
            add128(&sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
    }

    // Thread 0 writes the final consolidated answer to Global Memory
    if (tid == 0) {
        out_c_vec_raw[coeff_idx] = make_ulonglong2(sdata[0].lo, sdata[0].hi);
    }
}

// Assuming you have the S-Box defined globally somewhere like this:
// __constant__ uint8_t d_const_sbox[256] = { ... };

__global__ void generate_and_accumulate_a_vec(
    const uint32_t* __restrict__ global_aes_keys,
    const uint64_t* __restrict__ pi, 
    ulonglong2* __restrict__ out_a_vec_raw, 
    int index, int elements_per_poly,
    uint64_t mod_mask_lo, uint64_t mod_mask_hi) 
{
    int coeff_idx = blockIdx.x;
    int tid = threadIdx.x;

    // 1. Allocate Shared Memory for the math and the S-Box
    __shared__ uint128_cuda sdata[256];
    __shared__ uint8_t s_sbox[256];
    __shared__ uint32_t s_aes_keys[44];

    // 2. Collaborative Cache Load: Every thread loads exactly 1 byte
    if (tid < 256) {
        s_sbox[tid] = d_sbox[tid]; // Read from constant once!
    }
    if (tid < 44) {
        s_aes_keys[tid] = global_aes_keys[tid]; // First 44 threads load the keys
    }
    __syncthreads(); // Wait for the S-Box to finish loading into L1

    uint128_cuda accum = {0, 0};

    #pragma unroll 4
    for (int i = tid; i < index; i += blockDim.x) {
        uint64_t absolute_counter = ((uint64_t)i * elements_per_poly) + coeff_idx;
        
        // ... Inside your element generator, ensure it passes s_sbox down to AES
        uint128_cuda random_element;
        generate_a_vec_element(absolute_counter, &random_element, mod_mask_lo, mod_mask_hi, s_sbox, s_aes_keys);
        
        mac128(&accum, random_element, pi[i]);
    }
    sdata[tid] = accum;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s && (tid + s) < blockDim.x) {
            add128(&sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
    }

    if (tid == 0) {
        out_a_vec_raw[coeff_idx] = make_ulonglong2(sdata[0].lo, sdata[0].hi);
    }
}

// Ensure the C++ wrappers cast the void* to uint64_t*
// Ensure the C++ wrappers cast the void* to ulonglong2* for vectorized memory!
void launch_accumulate_c_vec_kernel(
    const void* d_enc_qs, const uint64_t* d_pi, void* d_out_c_vec, 
    int index, int total_coeffs, int blocks_c, int threads_per_block) 
{
    accumulate_c_vec<<<blocks_c, threads_per_block>>>(
        (const ulonglong2*)d_enc_qs, d_pi, (ulonglong2*)d_out_c_vec, index, total_coeffs
    );
}

void launch_generate_a_vec_kernel(
    const uint64_t* d_pi, void* d_out_a_vec, 
    int index, int total_coeffs_a, int blocks_a, int threads_per_block,
    uint64_t mod_mask_lo, uint64_t mod_mask_hi, const uint32_t* d_aes_round_keys) 
{
    generate_and_accumulate_a_vec<<<blocks_a, threads_per_block>>>(
        d_aes_round_keys, d_pi, (ulonglong2*)d_out_a_vec, index, total_coeffs_a, mod_mask_lo, mod_mask_hi
    );
}

// Add this at the bottom of proof.cu
// void copy_aes_keys_to_constant(const uint32_t* host_keys) {
//     // proof.cu can "see" d_const_aes_keys perfectly
//     cudaMemcpyToSymbol(d_const_aes_keys, host_keys, 44 * sizeof(uint32_t));
// }