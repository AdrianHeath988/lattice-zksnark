#ifndef __RING_BIG__
#define __RING_BIG__

// RingBig<T256, Q_LOG>: the SNARK outer ring Z_{2^Q_LOG} for the CPU big-int proof
// path, used when the FHE limb prime is too large (~60-bit q0) for the native
// __uint128_t Ring (which caps q_log < 128). Mirrors the libsnark::Ring<T,modulus>
// interface (ring_base.hpp) so the proof/LWE templates instantiate unchanged.
//
// C++17 forbids a struct as a non-type template parameter, so the modulus is keyed
// on the integral Q_LOG (the ring modulus is always a power of two). Because
// 2^Q_LOG divides 2^256, arithmetic may wrap mod 2^256 (natural uint256 overflow)
// and only be masked to Q_LOG bits at the points the native Ring masks (a-vec
// generation, rescale, decrypt) — the residue mod 2^Q_LOG is preserved either way.

#include "lwe/container/uint256.hpp"
#include "lwe/randomness/prg.hpp"
#include <array>
#include <cstdint>
#include <istream>
#include <ostream>

namespace libsnark {

    template <typename T256, unsigned Q_LOG> class RingBig {
    public:
        T256 value;

        static const T256 mod;   // 2^Q_LOG
        static const T256 MASK;  // 2^Q_LOG - 1

        static LWERandomness::PseudoRandomGenerator *prg;
        static LWERandomness::DiscreteGaussian *dg;

        RingBig() : value() {}
        explicit RingBig(const T256 &v) : value(v) {}
        RingBig(const RingBig &o) : value(o.value) {}

        inline RingBig &operator=(const RingBig &o) {
            this->value = o.value;
            return *this;
        }

        inline bool operator==(const RingBig &o) const { return value == o.value; }
        inline bool operator!=(const RingBig &o) const { return value != o.value; }

        // Arithmetic wraps mod 2^256 (uint256 overflow); correct mod 2^Q_LOG.
        inline RingBig &operator+=(const RingBig &o) {
            value += o.value;
            return *this;
        }
        inline RingBig &operator-=(const RingBig &o) {
            value -= o.value;
            return *this;
        }
        inline RingBig &operator*=(const RingBig &o) {
            value *= o.value;
            return *this;
        }
        inline RingBig operator+(const RingBig &o) const {
            RingBig r(*this);
            r += o;
            return r;
        }
        inline RingBig operator-(const RingBig &o) const {
            RingBig r(*this);
            r -= o;
            return r;
        }
        inline RingBig operator*(const RingBig &o) const {
            RingBig r(*this);
            r *= o;
            return r;
        }
        inline RingBig squared() const {
            RingBig r(*this);
            r *= r;
            return r;
        }

        static RingBig zero() { return RingBig(); }
        static RingBig one() { return RingBig(T256((unsigned __int128)1)); }

        // Reduce value into [0, 2^Q_LOG).
        inline RingBig &reduce() {
            value &= MASK;
            return *this;
        }

        // Divide-and-round rescale: value (mod 2^Q_LOG) is scaled down by
        // modulus_scale (the rescale_q, <= ~64-bit) with rounding, keeping the
        // residue mod p_prime fixed. 256-bit port of Ring::rescale (ring_base.hpp).
        inline RingBig &rescale(unsigned __int128 modulus_scale, uint64_t p_prime) {
            // Sentinel: modulus_scale == 0 means "no rescale" (the big param uses
            // this; mirrors the native B*C15 params where rescale_q == q_int makes
            // div_interval == 1, i.e. rescale is the identity).
            if (modulus_scale == 0) return *this;
            T256 v = value & MASK;
            unsigned __int128 rem128;
            v.divmod(p_prime, &rem128);  // rem128 = v % p_prime (< p_prime, fits u64)
            uint64_t mod_res = (uint64_t)rem128;

            // div_interval = 2^Q_LOG / modulus_scale  (fits 128 bits for our params)
            unsigned __int128 dummy;
            T256 div_interval256 = mod.divmod(modulus_scale, &dummy);
            unsigned __int128 div_interval = (unsigned __int128)div_interval256;

            unsigned __int128 res1_128;
            T256 res0_256 = v.divmod(div_interval, &res1_128);
            unsigned __int128 res_0 = (unsigned __int128)res0_256;  // < modulus_scale
            unsigned __int128 res_1 = res1_128;
            unsigned __int128 rescale =
                res_0 + (res_1 > (div_interval >> 1) ? 1 : 0);
            unsigned __int128 p_prime_round = rescale / p_prime;
            unsigned __int128 p_pivot = p_prime_round * p_prime + mod_res;
            unsigned __int128 diff_pivot =
                p_pivot > rescale ? p_pivot - rescale : rescale - p_pivot;
            if (diff_pivot > p_prime / 2) {
                if (p_pivot < rescale && p_pivot + p_prime < modulus_scale)
                    p_pivot += p_prime;
                else if (p_pivot > rescale && p_pivot >= p_prime)
                    p_pivot -= p_prime;
            }
            value = T256(p_pivot);
            return *this;
        }

        // --- randomness ---
        // Construct from a signed value, reduced into [0, 2^Q_LOG).
        static RingBig from_signed(long long s) {
            if (s >= 0) return RingBig(T256((unsigned __int128)(unsigned long long)s));
            T256 v = mod - T256((unsigned __int128)(unsigned long long)(-s));
            return RingBig(v & MASK);
        }

        // Uniform mod 2^Q_LOG: two 128-bit AES blocks -> 256 bits -> mask.
        static RingBig random_element() {
            unsigned __int128 lo = prg->next_prg_block();
            unsigned __int128 hi = prg->next_prg_block();
            T256 v(lo);
            v += T256(hi) << 128;
            return RingBig(v & MASK);
        }

        template <uint64_t LENGTH>
        static void random_element_sequence(std::array<RingBig, LENGTH> &dest) {
            for (uint64_t i = 0; i < LENGTH; i++) dest[i] = random_element();
        }

        // Uniform in [0, bound). bound < 2^128 path forwards to the native PRG.
        static RingBig bounded(unsigned __int128 bound) {
            return RingBig(T256(prg->bounded(bound)));
        }
        // 256-bit bound via rejection (cold; used only if a >128-bit bound appears).
        static RingBig bounded(const T256 &bound) {
            if ((unsigned __int128)(bound & (bound - T256((unsigned __int128)1))) == 0 &&
                bound.w[2] == 0 && bound.w[3] == 0)
                return bounded((unsigned __int128)bound);
            for (;;) {
                RingBig r = random_element();
                if (r.value < bound) return r;
            }
        }
        template <uint64_t LENGTH>
        static void bounded_sequence(unsigned __int128 bound,
                                     std::array<RingBig, LENGTH> &dest) {
            for (uint64_t i = 0; i < LENGTH; i++) dest[i] = bounded(bound);
        }
        template <uint64_t LENGTH>
        static void bounded_sequence(const T256 &bound,
                                     std::array<RingBig, LENGTH> &dest) {
            for (uint64_t i = 0; i < LENGTH; i++) dest[i] = bounded(bound);
        }

        // Signed bound in (-bound, bound), reduced mod 2^Q_LOG. bound < 2^128.
        static RingBig pm_bounded(unsigned __int128 bound) {
            return from_signed((long long)(__int128_t)prg->pm_bounded(bound));
        }
        template <uint64_t LENGTH>
        static void pm_bounded_sequence(unsigned __int128 bound,
                                        std::array<RingBig, LENGTH> &dest) {
            for (uint64_t i = 0; i < LENGTH; i++) dest[i] = pm_bounded(bound);
        }

        static RingBig discrete_gaussian() { return from_signed(dg->sample()); }
        template <uint64_t LENGTH>
        static void
        discrete_gaussian_sequence(std::array<RingBig, LENGTH> &dest) {
            alignas(16) std::array<uint64_t, LENGTH> rnd_src;
            prg->prg_mem_randomize(rnd_src);
            for (uint64_t i = 0; i < LENGTH; i++) {
                auto bucket = --dg->probability_interval.lower_bound(rnd_src[i]);
                dest[i] = from_signed(bucket->second);
            }
        }

        friend std::ostream &operator<<(std::ostream &o, const RingBig &p) {
            o << (p.value & MASK);
            return o;
        }
        friend std::istream &operator>>(std::istream &i, RingBig &p) {
            i >> p.value;
            return i;
        }
    };

    template <typename T256, unsigned Q_LOG>
    const T256 RingBig<T256, Q_LOG>::mod =
        T256((unsigned __int128)1) << Q_LOG;
    template <typename T256, unsigned Q_LOG>
    const T256 RingBig<T256, Q_LOG>::MASK =
        (T256((unsigned __int128)1) << Q_LOG) - T256((unsigned __int128)1);

    template <typename T256, unsigned Q_LOG>
    LWERandomness::PseudoRandomGenerator *RingBig<T256, Q_LOG>::prg;
    template <typename T256, unsigned Q_LOG>
    LWERandomness::DiscreteGaussian *RingBig<T256, Q_LOG>::dg;
}

#endif
