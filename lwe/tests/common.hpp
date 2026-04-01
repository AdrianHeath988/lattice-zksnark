#ifndef __LWE_TEST_COMMON__
#define __LWE_TEST_COMMON__

#define CURVE_BN128

#include "lwe/container/extension.hpp"
#include "lwe/container/field_base.hpp"
#include "lwe/container/ring_base.hpp"
#include "lwe/lwe_params.hpp"
#include "lwe/tests/circ_lattice_params.hpp"

#include <libff/algebra/fields/fp.hpp>
#include <libff/common/profiling.hpp>
#include <libsnark/common/default_types/r1cs_gg_ppzksnark_pp.hpp>
#include <libsnark/reductions/r1cs_to_qap/r1cs_to_qap.hpp>
#include <libsnark/relations/constraint_satisfaction_problems/r1cs/examples/r1cs_examples.hpp>
#include <libsnark/zk_proof_systems/ppzksnark/r1cs_gg_ppzksnark/examples/run_r1cs_gg_ppzksnark.hpp>
#include <libsnark/zk_proof_systems/ppzksnark/r1cs_gg_ppzksnark/r1cs_gg_ppzksnark.hpp>

class Fp2_b13_pp {
public:
    using Fp = libsnark::Field<uint32_t, LWE::B13Fp2ParamsBase::p_int>;
    using Fp_type = libsnark::Extension<Fp>;

    static LWERandomness::PseudoRandomGenerator *prg;
    static LWERandomness::DiscreteGaussian *dg;

    static void init_public_params() {
        Fp::prg = Fp2_b13_pp::prg;
        Fp::dg = Fp2_b13_pp::dg;
        Fp::multiplicative_generator = Fp(3);
        Fp::s = 1;
        Fp::root_of_unity = Fp(8190);

        Fp_type::non_residue = Fp(3);
        Fp_type::multiplicative_generator = Fp_type(3, 1);
        Fp_type::s = 14;
        Fp_type::root_of_unity = Fp_type(8127, 64);
    }
};

LWERandomness::PseudoRandomGenerator *Fp2_b13_pp::prg;
LWERandomness::DiscreteGaussian *Fp2_b13_pp::dg;

class Fp2_b19_pp {
public:
    using Fp = libsnark::Field<uint64_t, LWE::B19Fp2ParamsBase::p_int>;
    using Fp_type = libsnark::Extension<Fp>;

    static LWERandomness::PseudoRandomGenerator *prg;
    static LWERandomness::DiscreteGaussian *dg;

    static void init_public_params() {
        Fp::prg = Fp2_b19_pp::prg;
        Fp::dg = Fp2_b19_pp::dg;
        Fp::multiplicative_generator = Fp(3);
        Fp::s = 1;
        Fp::root_of_unity = Fp(524286);

        Fp_type::non_residue = Fp(3);
        Fp_type::multiplicative_generator = Fp_type(3, 1);
        Fp_type::s = 20;
        Fp_type::root_of_unity = Fp_type(512, 523775);
    }
};

LWERandomness::PseudoRandomGenerator *Fp2_b19_pp::prg;
LWERandomness::DiscreteGaussian *Fp2_b19_pp::dg;

template <LWE::uint128_t q_int = LWE::B19C20::q_int> class Ring2_common_pp {
public:
    using Rq = libsnark::Ring<LWE::uint128_t, q_int>;
    using Rq_type = libsnark::Extension<Rq>;

    static LWERandomness::PseudoRandomGenerator *prg;
    static LWERandomness::DiscreteGaussian *dg;

    static void init_public_params() {
        Rq::prg = prg;
        Rq::dg = dg;

        Rq_type::non_residue = Rq(3);
    }
};

template <LWE::uint128_t q_int>
LWERandomness::PseudoRandomGenerator *Ring2_common_pp<q_int>::prg;
template <LWE::uint128_t q_int>
LWERandomness::DiscreteGaussian *Ring2_common_pp<q_int>::dg;
// Compile-time modular exponentiation
constexpr uint64_t ct_modpow(uint64_t base, uint64_t exp, uint64_t mod) {
    uint64_t res = 1;
    base %= mod;
    while (exp > 0) {
        if (exp & 1) res = (static_cast<unsigned __int128>(res) * base) % mod;
        base = (static_cast<unsigned __int128>(base) * base) % mod;
        exp >>= 1;
    }
    return res;
}

// Compile-time primitive generator finder
constexpr uint64_t get_generator(uint64_t p, uint64_t f1, uint64_t f2, uint64_t f3) {
    for (uint64_t g = 2; g < p; g++) {
        bool is_gen = true;
        if (f1 != 0 && ct_modpow(g, (p - 1) / f1, p) == 1) is_gen = false;
        if (f2 != 0 && ct_modpow(g, (p - 1) / f2, p) == 1) is_gen = false;
        if (f3 != 0 && ct_modpow(g, (p - 1) / f3, p) == 1) is_gen = false;
        if (is_gen) return g;
    }
    return 0;
}

template <typename ParamsBase>
class Fp_b28_template_pp {
public:
    using Fp_type = libsnark::Field<uint64_t, ParamsBase::p_int>;

    static LWERandomness::PseudoRandomGenerator *prg;
    static LWERandomness::DiscreteGaussian *dg;

    static void init_public_params() {
        Fp_type::prg = Fp_b28_template_pp<ParamsBase>::prg;
        Fp_type::dg = Fp_b28_template_pp<ParamsBase>::dg;
        Fp_type::s = 25; 
        
        if constexpr (ParamsBase::p_int == 167772161) { 
            constexpr uint64_t gen = get_generator(167772161, 2, 5, 0);
            Fp_type::multiplicative_generator = Fp_type(gen);
            Fp_type::root_of_unity = Fp_type(ct_modpow(gen, 5, 167772161));
        } 
        else if constexpr (ParamsBase::p_int == 469762049) { 
            constexpr uint64_t gen = get_generator(469762049, 2, 7, 0);
            Fp_type::multiplicative_generator = Fp_type(gen);
            Fp_type::root_of_unity = Fp_type(ct_modpow(gen, 14, 469762049));
        }
        else if constexpr (ParamsBase::p_int == 1107296257) { 
            constexpr uint64_t gen = get_generator(1107296257, 2, 3, 11);
            Fp_type::multiplicative_generator = Fp_type(gen);
            Fp_type::root_of_unity = Fp_type(ct_modpow(gen, 33, 1107296257));
        }
        else if constexpr (ParamsBase::p_int == 1711276033) { 
            constexpr uint64_t gen = get_generator(1711276033, 2, 3, 17);
            Fp_type::multiplicative_generator = Fp_type(gen);
            Fp_type::root_of_unity = Fp_type(ct_modpow(gen, 51, 1711276033));
        }
        else if constexpr (ParamsBase::p_int == 1811939329) { 
            constexpr uint64_t gen = get_generator(1811939329, 2, 3, 0);
            Fp_type::multiplicative_generator = Fp_type(gen);
            Fp_type::root_of_unity = Fp_type(ct_modpow(gen, 54, 1811939329));
        }
        // ==========================================
        // NEW P LIMB GENERATORS (With 64-bit safety)
        // ==========================================
        else if constexpr (ParamsBase::p_int == 2013265921ULL) { // k = 60
            constexpr uint64_t gen = get_generator(2013265921ULL, 2, 3, 5);
            Fp_type::multiplicative_generator = Fp_type(gen);
            Fp_type::root_of_unity = Fp_type(ct_modpow(gen, 60, 2013265921ULL));
        }
        else if constexpr (ParamsBase::p_int == 2113929217ULL) { // k = 63
            constexpr uint64_t gen = get_generator(2113929217ULL, 2, 3, 7);
            Fp_type::multiplicative_generator = Fp_type(gen);
            Fp_type::root_of_unity = Fp_type(ct_modpow(gen, 63, 2113929217ULL));
        }
        else if constexpr (ParamsBase::p_int == 2281701377ULL) { // k = 68
            constexpr uint64_t gen = get_generator(2281701377ULL, 2, 17, 0);
            Fp_type::multiplicative_generator = Fp_type(gen);
            Fp_type::root_of_unity = Fp_type(ct_modpow(gen, 68, 2281701377ULL));
        }
    }
};

template <typename ParamsBase>
class Fp_b60_template_pp {
public:
    // THE ULTIMATE FIX: 128-bit storage prevents multiplication truncation!
    using Fp_type = libsnark::Field<unsigned __int128, ParamsBase::p_int>;

    static LWERandomness::PseudoRandomGenerator *prg;
    static LWERandomness::DiscreteGaussian *dg;

    static void init_public_params() {
        Fp_type::prg = Fp_b60_template_pp<ParamsBase>::prg;
        Fp_type::dg = Fp_b60_template_pp<ParamsBase>::dg;
        
        // Initialize the SNARK backend mathematically for your P prime
        if constexpr (ParamsBase::p_int == 1152921504942391297ULL) { 
            Fp_type::s = 60; // Because P = 2^60 + 1
            Fp_type::multiplicative_generator = Fp_type(5);
            Fp_type::root_of_unity = Fp_type(5);
        }
    }
};

// Ensure static members are allocated (just like they likely are for B28)
template <typename ParamsBase> 
LWERandomness::PseudoRandomGenerator* Fp_b60_template_pp<ParamsBase>::prg = nullptr;
template <typename ParamsBase> 
LWERandomness::DiscreteGaussian* Fp_b60_template_pp<ParamsBase>::dg = nullptr;

template <typename ParamsBase>
LWERandomness::PseudoRandomGenerator* Fp_b28_template_pp<ParamsBase>::prg = nullptr;

template <typename ParamsBase>
LWERandomness::DiscreteGaussian* Fp_b28_template_pp<ParamsBase>::dg = nullptr;

class Fp_b28_pp {
public:
    using Fp_type = libsnark::Field<uint64_t, LWE::B28FpParamsBase::p_int>;

    static LWERandomness::PseudoRandomGenerator *prg;
    static LWERandomness::DiscreteGaussian *dg;

    static void init_public_params() {
        Fp_type::prg = Fp_b28_pp::prg;
        Fp_type::dg = Fp_b28_pp::dg;
        Fp_type::s = 25;
        Fp_type::multiplicative_generator = Fp_type(3);
        Fp_type::root_of_unity = Fp_type(243);
    }
};


class Fp_b23_pp {
public:
    using Fp_type = libsnark::Field<uint64_t, LWE::B23FpParamsBase::p_int>;

    static LWERandomness::PseudoRandomGenerator *prg;
    static LWERandomness::DiscreteGaussian *dg;

    static void init_public_params() {
        Fp_type::prg = Fp_b23_pp::prg;
        Fp_type::dg = Fp_b23_pp::dg;
        Fp_type::s = 20;
        Fp_type::multiplicative_generator = Fp_type(3);
        Fp_type::root_of_unity = Fp_type(2187);
    }
};

LWERandomness::PseudoRandomGenerator *Fp_b23_pp::prg;
LWERandomness::DiscreteGaussian *Fp_b23_pp::dg;

class Fp_b19_pp {
public:
    using Fp_type = libsnark::Field<uint64_t, LWE::B19FpParamsBase::p_int>;

    static LWERandomness::PseudoRandomGenerator *prg;
    static LWERandomness::DiscreteGaussian *dg;

    static void init_public_params() {
        Fp_type::prg = Fp_b19_pp::prg;
        Fp_type::dg = Fp_b19_pp::dg;
        Fp_type::s = 1;
        Fp_type::multiplicative_generator = Fp_type(3);
        Fp_type::root_of_unity = Fp_type(524286);
    }
};

LWERandomness::PseudoRandomGenerator *Fp_b19_pp::prg;
LWERandomness::DiscreteGaussian *Fp_b19_pp::dg;

class Fp_b13_pp {
public:
    using Fp_type = libsnark::Field<uint32_t, LWE::B13FpParamsBase::p_int>;

    static LWERandomness::PseudoRandomGenerator *prg;
    static LWERandomness::DiscreteGaussian *dg;

    static void init_public_params() {
        Fp_type::prg = Fp_b13_pp::prg;
        Fp_type::dg = Fp_b13_pp::dg;
        Fp_type::multiplicative_generator = Fp_type(3);
        Fp_type::s = 1;
        Fp_type::root_of_unity = Fp_type(8190);
    }
};

LWERandomness::PseudoRandomGenerator *Fp_b13_pp::prg;
LWERandomness::DiscreteGaussian *Fp_b13_pp::dg;

template <LWE::uint128_t q_int = LWE::B19C20::q_int> class Ring_common_pp {
public:
    using Rq_type = libsnark::Ring<LWE::uint128_t, q_int>;
    using Rq = Rq_type;

    static LWERandomness::PseudoRandomGenerator *prg;
    static LWERandomness::DiscreteGaussian *dg;

    static void init_public_params() {
        Rq_type::prg = prg;
        Rq_type::dg = dg;
    }
};

template <LWE::uint128_t q_int>
LWERandomness::PseudoRandomGenerator *Ring_common_pp<q_int>::prg;
template <LWE::uint128_t q_int>
LWERandomness::DiscreteGaussian *Ring_common_pp<q_int>::dg;

namespace LWE {
    constexpr const double width_default = 64.0;
}

#endif
