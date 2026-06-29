#ifndef __FIELD_BASE__
#define __FIELD_BASE__

#include "ring_base.hpp"

namespace libsnark {
    // Forward declaration for the CPU big-int ring (ring_big.hpp): the Field
    // needs project_from / lift_to / lift_ring_multiply overloads against it,
    // but must not pull in ring_big.hpp (include cycle). Bodies are templates,
    // so RingBig only needs to be complete at instantiation (the call site).
    template <typename T256, unsigned Q_LOG> class RingBig;

    template <typename T, T modulus> class Field {
    public:
        T value;
        static Field multiplicative_generator;
        static Field root_of_unity;
        static size_t s;
        static LWERandomness::PseudoRandomGenerator *prg;
        static LWERandomness::DiscreteGaussian *dg;

        Field() : value{} {}
        explicit Field(const long &x)
            : value(x % modulus + ((x < 0) ? modulus : 0)) {}
        Field(const Field &o) : value(o.value) {}

        inline Field &operator=(const Field &o) {
            this->value = o.value % modulus + ((o.value < 0) ? modulus : 0);
            return *this;
        }

        inline Field &operator=(const T &o) {
            this->value = o % modulus + ((o < 0) ? modulus : 0);
            return *this;
        }

        inline bool operator==(const Field &other) const {
            return this->value == other.value;
        }

        inline bool operator!=(const Field &other) const {
            return this->value != other.value;
        }

        inline Field &operator+=(const Field &other) {
            this->value += other.value;
            this->value %= modulus;
            return *this;
        }

        inline Field operator+(const Field &other) const {
            Field what(*this);
            what += other;
            return what;
        }

        inline Field &operator-=(const Field &other) {
            if (this->value >= other.value)
                this->value -= other.value;
            else
                this->value = modulus - other.value + this->value;
            return *this;
        }

        inline Field operator-(const Field &other) const {
            Field what(*this);
            what -= other;
            return what;
        }

        inline Field &operator*=(const Field &other) {
            this->value *= other.value;
            this->value %= modulus;
            return *this;
        }

        inline Field operator*(const Field &other) const {
            Field what(*this);
            what *= other;
            return what;
        }

        inline Field &operator^=(const unsigned long &pwr) {
            if (pwr == 0) {
                this->value = T(1);
                return *this;
            }
            unsigned long _pwr = pwr;
            Field _res(1);
            while (_pwr > 0) {
                if (_pwr & 1u)
                    _res *= *this;
                _pwr >>= 1u;
                *this *= Field(*this);
            }
            this->value = _res.value;
            return *this;
        }

        inline Field &operator^=(const Field &other) {
            *this ^= ((unsigned long) other.value);
            return *this;
        }

        inline Field &operator^=(const libff::bigint<1> &pwr) {
            *this ^= pwr.as_ulong();
            return *this;
        }

        inline Field operator^(const Field &pwr) const {
            Field what(*this);
            what ^= pwr;
            return what;
        }

        inline Field operator^(const libff::bigint<1> &pwr) const {
            Field what(*this);
            what ^= pwr;
            return what;
        }

        inline Field operator^(unsigned long pwr) const {
            Field what(*this);
            what ^= pwr;
            return what;
        }

        inline Field squared() const {
            Field f(*this);
            f *= f;
            return f;
        }

        inline Field &invert() {
            this->value = LWE::modular_inverse(this->value, modulus);
            return *this;
        }

        inline Field inverse() const {
            Field f(*this);
            return (f.invert());
        }

        inline Field operator-() const {
            Field what(*this);
            what.value = modulus - what.value;
            return what;
        }

        static Field zero() { return Field(0); }
        static Field one() { return Field(1); }
        static Field geometric_generator() {
            return Field::multiplicative_generator;
        }
        static Field arithmetic_generator() { return Field::one(); }

        template <typename nT, nT o_mod>
        Field &project_from(const Ring<nT, o_mod> &o, nT modulus_q = o_mod) {
            __int128_t o_v;
            if (modulus_q & (modulus_q - 1)) {
                o_v = o.value > o_mod >> 1 ? (o.value & (o_mod - 1)) - o_mod
                                           : o.value;
                o_v %= __int128_t(modulus_q);
                if (o_v < 0)
                    o_v += modulus_q;
            } else
                o_v = __int128_t(o.value & (modulus_q - 1));
            if (o_v > __int128_t(modulus_q >> 1))
                o_v -= modulus_q;
            o_v %= __int128_t(modulus);
            if (o_v < 0)
                o_v += modulus;
            this->value = o_v;
            return *this;
        }

        template <typename nT, nT o_mod>
        void lift_to(Ring<nT, o_mod> &other) const {
            other.value = nT(this->value);
        }

        template <typename nT, nT o_mod>
        Ring<nT, o_mod> lift_ring_multiply(const Ring<nT, o_mod> &other) const {
            Ring<nT, o_mod> res(other);
            res.value *= this->value;
            return res;
        }

        // --- CPU big-int ring (RingBig<T256,Q_LOG>) bridges ---
        // The big ring's modulus 2^Q_LOG is a power of two held in a software
        // 256-bit value; reductions use uint256::divmod (the field prime <=~60 bit
        // fits __int128). modulus_q is ignored: the true modulus is RingBig::mod.
        template <typename T256, unsigned Q_LOG>
        Field &project_from(const RingBig<T256, Q_LOG> &o,
                            unsigned __int128 /*modulus_q (ignored)*/ = 0) {
            const T256 masked = o.value & RingBig<T256, Q_LOG>::MASK;
            const bool neg = masked > (RingBig<T256, Q_LOG>::mod >> 1);
            unsigned __int128 rem;
            masked.divmod((unsigned __int128)modulus, &rem);  // masked mod p
            __int128_t o_v = (__int128_t)rem;
            if (neg) {  // centered: subtract 2^Q_LOG (mod p)
                unsigned __int128 qmod;
                RingBig<T256, Q_LOG>::mod.divmod((unsigned __int128)modulus, &qmod);
                o_v -= (__int128_t)qmod;
            }
            o_v %= (__int128_t)modulus;
            if (o_v < 0) o_v += modulus;
            this->value = (T)o_v;
            return *this;
        }

        template <typename T256, unsigned Q_LOG>
        void lift_to(RingBig<T256, Q_LOG> &other) const {
            other.value = T256((unsigned __int128)this->value);
        }

        template <typename T256, unsigned Q_LOG>
        RingBig<T256, Q_LOG>
        lift_ring_multiply(const RingBig<T256, Q_LOG> &other) const {
            RingBig<T256, Q_LOG> res(other);
            res.value *= T256((unsigned __int128)this->value);
            return res;
        }

        static Field random_element() {
            return Field(prg->bounded((__uint128_t) modulus));
        }

        template <uint64_t LENGTH>
        static void
        random_element_sequence(std::array<Field<T, modulus>, LENGTH> &_dest) {
            prg->prg_mem_randomize(_dest);
            auto &T_dest = reinterpret_cast<std::array<T, LENGTH> &>(_dest);
            for (uint64_t i = 0; i < LENGTH; i++)
                _dest[i] = Field<T, modulus>(T_dest[i]);
        }

        friend std::ostream &operator<<(std::ostream &o,
                                        const Field<T, modulus> &p) {
            o << (p.value % modulus);
            return o;
        }
        friend std::istream &operator>>(std::istream &i, Field<T, modulus> &p) {
            i >> p.value;
            return i;
        }
    };

    template <typename T, T modulus>
    Field<T, modulus> Field<T, modulus>::multiplicative_generator;

    template <typename T, T modulus>
    Field<T, modulus> Field<T, modulus>::root_of_unity;

    template <typename T, T modulus> size_t Field<T, modulus>::s;

    template <typename T, T modulus>
    LWERandomness::PseudoRandomGenerator *Field<T, modulus>::prg;

    template <typename T, T modulus>
    LWERandomness::DiscreteGaussian *Field<T, modulus>::dg;
}

#endif
