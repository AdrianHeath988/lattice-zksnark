// Determinism test for the GPU QAP witness map.
//
// Builds a random R1CS example, computes the prover's `pi` vector both on the
// CPU (r1cs_to_qap_witness_map + prepare_pi_proof) and on the GPU
// (qap_witness_map_gpu), and asserts they are bit-identical. Same d1,d2,d3 are
// fed to both paths. Exercises basic_radix2 (power-of-2 domain) and step_radix2
// (2^k + 2^r domain) on both field widths:
//   - 64-bit storage  (Fp_b23, a <32-bit prime; the native B*C* prover path)
//   - 128-bit storage (Fp_b60, a 60-bit prime; the big-int-ring / bootstrap path)
//
//   usage: ./test_qap_gpu [<num_constraints> <input_size>]   (64-bit field only)

#include "lwe/snark/qap_gpu.hpp"
#include "lwe/snark/r1cs_lattice_snark.hpp"
#include "lwe/tests/circ_lattice_params.hpp"
#include "lwe/tests/common.hpp"

#include <iostream>
#include <vector>

using namespace libsnark;
using namespace LWE;

template <typename ppT>
static bool run_case(size_t num_constraints, size_t input_size) {
    using FieldT = typename ppT::Fp_type;
    r1cs_example<FieldT> example =
        generate_r1cs_example_with_field_input<FieldT>(num_constraints,
                                                       input_size);
    const auto &cs = example.constraint_system;

    const FieldT d1 = FieldT::random_element();
    const FieldT d2 = FieldT::random_element();
    const FieldT d3 = FieldT::random_element();

    // ---- CPU reference pi ----
    const qap_witness<FieldT> qap_wit = r1cs_to_qap_witness_map(
        cs, example.primary_input, example.auxiliary_input, d1, d2, d3);
    libff::Fr_vector<ppT> cpu_pi;
    prepare_pi_proof<ppT>(qap_wit, cpu_pi);

    // ---- GPU pi ----
    std::vector<FieldT> full_assignment = example.primary_input;
    full_assignment.insert(full_assignment.end(),
                           example.auxiliary_input.begin(),
                           example.auxiliary_input.end());
    void *d_pi = nullptr;
    size_t proof_dim = 0;
    bool ok = qap_witness_map_gpu<FieldT>(cs, full_assignment, d1, d2, d3, &d_pi,
                                          &proof_dim);
    if (!ok) {
        std::cout << "  [skip] unsupported domain (CPU fallback path)\n";
        return true;
    }

    using HostT = decltype(FieldT::value);
    std::vector<HostT> gpu_pi(proof_dim);
    cudaMemcpy(gpu_pi.data(), d_pi, proof_dim * sizeof(HostT),
               cudaMemcpyDeviceToHost);
    cudaFree(d_pi);

    if (proof_dim != cpu_pi.size()) {
        std::cout << "  [FAIL] size mismatch: cpu=" << cpu_pi.size()
                  << " gpu=" << proof_dim << "\n";
        return false;
    }
    size_t bad = 0, first_bad = (size_t)-1;
    for (size_t i = 0; i < proof_dim; ++i)
        if (gpu_pi[i] != cpu_pi[i].value) {
            if (first_bad == (size_t)-1) first_bad = i;
            ++bad;
        }
    if (bad) {
        std::cout << "  [FAIL] " << bad << " / " << proof_dim
                  << " mismatch (first at " << first_bad << ")\n";
        return false;
    }
    std::cout << "  [PASS] pi identical (" << proof_dim << " elems)\n";
    return true;
}

// Domain is field-dependent: a non-power-of-2 size is step_radix2 on the 64-bit
// field but extended_radix2_large on the high-2-adicity 60-bit field.
struct CaseSpec { size_t nc, ni; const char *tag; };
static const CaseSpec kCases[] = {
    {1015, 8, "m=1024  (pow2 -> basic_radix2)"},
    {4087, 8, "m=4096  (pow2 -> basic_radix2)"},
    {16375, 8, "m=16384 (pow2 -> basic_radix2)"},
    {1528, 7, "m=1536  (step / ext-large)"},
    {6135, 8, "m=6144  (step / ext-large)"},
    {3063, 8, "m=3072  (step / ext-large)"},
    {65271, 8, "m=65280 (ext-large, ncosets=255)"},
};

template <typename ppT> static bool run_all(const char *label) {
    std::cout << "==== field: " << label << " ====\n";
    bool all = true;
    for (auto &c : kCases) {
        std::cout << "case [" << c.tag << "] nc=" << c.nc << " ni=" << c.ni
                  << ":\n";
        all &= run_case<ppT>(c.nc, c.ni);
    }
    return all;
}

// Force instantiation of the prover for both ring paths so this TU
// compile-checks the GPU-QAP wiring inside r1cs_lattice_snark_prove (the native
// uint64 device-pi path and the big-int-ring host-materialise path), even though
// we don't run it here (the determinism test above validates the QAP math).
static void compile_check_prove() {
    using NRing = Ring_common_pp<LWE::B23C15::q_int>;
    using BRing = Ring_common_big_pp<LWE::B60Cbig::q_log>;
    using BField = Fp_b60_template_pp<LWE::B60FpParamsBase>;
    volatile auto p_native =
        &r1cs_lattice_snark_prove<Fp_b23_pp, NRing, LWE::B23C15>;
    volatile auto p_big =
        &r1cs_lattice_snark_prove<BField, BRing, LWE::B60Cbig>;
    (void)p_native;
    (void)p_big;
}

int main(int argc, char *argv[]) {
    using ring_pp = Ring_common_pp<LWE::B28C15::q_int>;
    using Fp60pp = Fp_b60_template_pp<LWE::B60FpParamsBase>;
    (void)&compile_check_prove;

    auto *prg = new LWERandomness::PseudoRandomGenerator();
    auto *dg = new LWERandomness::DiscreteGaussian(18.0, LWE::expand, *prg);
    // Initialise both SNARK fields' static params (root of unity, s, prg/dg).
    public_params_init<Fp_b23_pp, ring_pp>(prg, dg);
    public_params_init<Fp60pp>(prg, dg);

    bool all = true;
    if (argc >= 3) {
        size_t nc = std::stoul(argv[1]), ni = std::stoul(argv[2]);
        // 3rd arg "60" selects the 128-bit/60-bit-prime field, else Fp_b23.
        bool big = (argc >= 4 && std::string(argv[3]) == "60");
        std::cout << "case nc=" << nc << " ni=" << ni
                  << (big ? " (Fp_b60)" : " (Fp_b23)") << ":\n";
        if (big) all &= run_case<Fp60pp>(nc, ni);
        else all &= run_case<Fp_b23_pp>(nc, ni);
    } else {
        all &= run_all<Fp_b23_pp>("Fp_b23 (64-bit storage, <32-bit prime)");
        all &= run_all<Fp60pp>("Fp_b60 (128-bit storage, 60-bit prime)");
    }
    std::cout << (all ? "ALL PASS" : "FAILURES") << std::endl;
    delete prg;
    delete dg;
    return all ? 0 : 1;
}
