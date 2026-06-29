#ifndef __UINT256__
#define __UINT256__

// Software 256-bit unsigned integer for the CPU big-int proof path.
//
// Why this exists: bootstrapped CKKS needs a ~60-bit bottom prime q0, which is a
// CRT-load-bearing limb of every operand, so proving any op in F_{q0} is required
// for soundness. A 60-bit limb forces the SNARK ring modulus to q_log ~140 > 128,
// which the native __uint128_t Ring can't hold. RingBig<uint256, Q_LOG> (ring_big.hpp)
// uses this type; the existing 28-bit GPU path is untouched.
//
// The SNARK ring modulus is ALWAYS a power of two (2^Q_LOG), so reduction is a mask
// (no division on the hot path) and multiply only needs the low 256 bits. The hot
// proof op is `accum += enc * scalar` where scalar = a field value < 2^64. divmod
// (by a <=128-bit divisor) is only used on the cold verify-side rescale.
//
// Little-endian limbs: w[0] is least significant.

#include <cstdint>
#include <cstdio>
#include <ostream>
#include <istream>

namespace LWE {

struct uint256 {
    uint64_t w[4];

    constexpr uint256() : w{0, 0, 0, 0} {}
    // Single small-int constructor: covers uint64 and integer literals (an int
    // literal converts to __uint128 unambiguously). Avoids the uint64-vs-__int128
    // overload ambiguity on `uint256(1)`.
    constexpr uint256(unsigned __int128 v)
        : w{(uint64_t)v, (uint64_t)(v >> 64), 0, 0} {}
    constexpr uint256(uint64_t w0, uint64_t w1, uint64_t w2, uint64_t w3)
        : w{w0, w1, w2, w3} {}

    // --- conversions ---
    explicit operator uint64_t() const { return w[0]; }
    explicit operator unsigned __int128() const {
        return ((unsigned __int128)w[1] << 64) | w[0];
    }
    explicit operator bool() const { return w[0] | w[1] | w[2] | w[3]; }

    // --- comparisons ---
    bool operator==(const uint256 &o) const {
        return w[0] == o.w[0] && w[1] == o.w[1] && w[2] == o.w[2] &&
               w[3] == o.w[3];
    }
    bool operator!=(const uint256 &o) const { return !(*this == o); }
    bool operator<(const uint256 &o) const {
        for (int i = 3; i >= 0; --i)
            if (w[i] != o.w[i]) return w[i] < o.w[i];
        return false;
    }
    bool operator>(const uint256 &o) const { return o < *this; }
    bool operator<=(const uint256 &o) const { return !(o < *this); }
    bool operator>=(const uint256 &o) const { return !(*this < o); }

    // --- add / sub (mod 2^256, wrapping) ---
    uint256 &operator+=(const uint256 &o) {
        unsigned __int128 carry = 0;
        for (int i = 0; i < 4; ++i) {
            unsigned __int128 s = (unsigned __int128)w[i] + o.w[i] + carry;
            w[i] = (uint64_t)s;
            carry = s >> 64;
        }
        return *this;
    }
    uint256 &operator-=(const uint256 &o) {
        unsigned __int128 borrow = 0;
        for (int i = 0; i < 4; ++i) {
            unsigned __int128 d =
                (unsigned __int128)w[i] - o.w[i] - borrow;
            w[i] = (uint64_t)d;
            borrow = (d >> 64) & 1;  // 1 if underflow
        }
        return *this;
    }
    uint256 operator+(const uint256 &o) const {
        uint256 r(*this);
        r += o;
        return r;
    }
    uint256 operator-(const uint256 &o) const {
        uint256 r(*this);
        r -= o;
        return r;
    }

    // --- bitwise AND (used for power-of-two modular masking) ---
    uint256 &operator&=(const uint256 &o) {
        for (int i = 0; i < 4; ++i) w[i] &= o.w[i];
        return *this;
    }
    uint256 operator&(const uint256 &o) const {
        uint256 r(*this);
        r &= o;
        return r;
    }

    // --- shifts ---
    uint256 operator<<(unsigned s) const {
        uint256 r;
        if (s >= 256) return r;
        unsigned word = s >> 6, bit = s & 63;
        for (int i = 3; i >= 0; --i) {
            uint64_t v = 0;
            int src = i - (int)word;
            if (src >= 0) {
                v = w[src] << bit;
                if (bit && src - 1 >= 0) v |= w[src - 1] >> (64 - bit);
            }
            r.w[i] = v;
        }
        return r;
    }
    uint256 operator>>(unsigned s) const {
        uint256 r;
        if (s >= 256) return r;
        unsigned word = s >> 6, bit = s & 63;
        for (int i = 0; i < 4; ++i) {
            uint64_t v = 0;
            int src = i + (int)word;
            if (src < 4) {
                v = w[src] >> bit;
                if (bit && src + 1 < 4) v |= w[src + 1] << (64 - bit);
            }
            r.w[i] = v;
        }
        return r;
    }

    // --- multiply (low 256 bits only; modulus is 2^Q_LOG so high bits are masked away) ---
    // Scalar fast path: the proof hot loop is accum += enc * (field value < 2^64).
    uint256 mul_u64(uint64_t s) const {
        uint256 r;
        unsigned __int128 carry = 0;
        for (int i = 0; i < 4; ++i) {
            unsigned __int128 p =
                (unsigned __int128)w[i] * s + carry;
            r.w[i] = (uint64_t)p;
            carry = p >> 64;
        }
        return r;  // carry out of w[3] is dropped (mod 2^256)
    }
    uint256 &operator*=(const uint256 &o) {
        // Schoolbook, keeping only the low 4 limbs (mod 2^256).
        uint64_t res[4] = {0, 0, 0, 0};
        for (int i = 0; i < 4; ++i) {
            if (!w[i]) continue;
            unsigned __int128 carry = 0;
            for (int j = 0; i + j < 4; ++j) {
                unsigned __int128 cur = (unsigned __int128)res[i + j] +
                                        (unsigned __int128)w[i] * o.w[j] +
                                        carry;
                res[i + j] = (uint64_t)cur;
                carry = cur >> 64;
            }
        }
        for (int i = 0; i < 4; ++i) w[i] = res[i];
        return *this;
    }
    uint256 operator*(const uint256 &o) const {
        uint256 r(*this);
        r *= o;
        return r;
    }

    // bit length (index of highest set bit + 1; 0 for zero)
    unsigned bit_length() const {
        for (int i = 3; i >= 0; --i)
            if (w[i]) return (unsigned)(i * 64) + (64 - __builtin_clzll(w[i]));
        return 0;
    }

    // divmod by a <=128-bit divisor (cold path: verify-side rescale only).
    // Returns quotient; writes remainder to *rem. Simple long division by bit.
    uint256 divmod(unsigned __int128 d, unsigned __int128 *rem) const {
        uint256 q, r;
        const uint256 D(d);
        for (int b = (int)bit_length() - 1; b >= 0; --b) {
            // r = (r << 1) | bit b of *this  (full 256-bit, no truncation)
            r = r << 1;
            if ((w[b >> 6] >> (b & 63)) & 1u) r.w[0] |= 1;
            if (r >= D) {
                r -= D;
                q.w[b >> 6] |= (uint64_t)1 << (b & 63);
            }
        }
        if (rem) *rem = (unsigned __int128)r;
        return q;
    }
};

inline std::ostream &operator<<(std::ostream &o, const uint256 &v) {
    // hex, most-significant word first
    o << "0x";
    for (int i = 3; i >= 0; --i) {
        char buf[17];
        std::snprintf(buf, sizeof(buf), "%016llx", (unsigned long long)v.w[i]);
        o << buf;
    }
    return o;
}
inline std::istream &operator>>(std::istream &i, uint256 &v) {
    // read a single 64-bit word into the low limb (sufficient for current call sites)
    unsigned long long x = 0;
    i >> x;
    v = uint256((uint64_t)x);
    return i;
}

}  // namespace LWE

#endif
