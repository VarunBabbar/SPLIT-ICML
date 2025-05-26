#include "bitset.hpp"

#include <memory>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <cassert>
#include <bit>


const static mp_limb_t FULL_BLOCK = ~0UL;
const static mp_limb_t EMPTY_BLOCK = 0UL;

#define MASK_WORD(size) (FULL_BLOCK >> (mp_bits_per_limb - ((size) % mp_bits_per_limb)))
#define NUMBER_OF_WORDS(size) ((size) / mp_bits_per_limb + ((size) % mp_bits_per_limb != 0))
#define FULL_CHAR 0xFF

Bitset::Bitset(size_t size) : m_size(size), m_data(new mp_limb_t[NUMBER_OF_WORDS(size)]) {}

Bitset::Bitset(const Bitset &other) : m_size(other.m_size), m_data(new mp_limb_t[NUMBER_OF_WORDS(other.m_size)]) {
    mpn_copyi(m_data, other.m_data, NUMBER_OF_WORDS(other.m_size));
}

Bitset::Bitset(Bitset &&other) noexcept {
    // Instantiate this object
    m_size = other.m_size;
    m_data = other.m_data;

    // Reset the other object
    other.m_size = 0;
    other.m_data = nullptr;
}

Bitset::~Bitset() { delete[] m_data; }

Bitset Bitset::create_full(size_t size) {
    Bitset result(size);
    size_t n_words = NUMBER_OF_WORDS(size);
    std::memset(result.m_data, FULL_CHAR, n_words * sizeof(mp_limb_t));
    // Mask the last word
    result.m_data[n_words - 1] &= MASK_WORD(size);
    return result;
}

Bitset Bitset::create_empty(size_t size) {
    Bitset result(size);
    mpn_zero(result.m_data, NUMBER_OF_WORDS(size));
    return result;
}

Bitset &Bitset::operator=(const Bitset &other) {
    if (this == &other) {
        return *this;
    }
    delete[] m_data;
    m_data = new mp_limb_t[NUMBER_OF_WORDS(other.m_size)];
    m_size = other.m_size;
    mpn_copyi(m_data, other.m_data, NUMBER_OF_WORDS(other.m_size));
    return *this;
}

Bitset &Bitset::operator=(Bitset &&other) noexcept {
    delete[] m_data;
    m_data = other.m_data;
    m_size = other.m_size;

    other.m_data = nullptr;
    other.m_size = 0;

    return *this;
}

bool Bitset::get(size_t index) const {
    // TODO We should not have this check!
    if (index >= m_size) {
        throw std::out_of_range("Bitset index out of range");
    }

    auto word = index / mp_bits_per_limb;
    auto bit = index % mp_bits_per_limb;
    auto block = m_data[word];
    return (block >> bit) % 2;
}

void Bitset::set(size_t index, bool value) const {
    // TODO We should not have this check!
    if (index >= m_size) {
        throw std::out_of_range("Bitset index out of range");
    }

    auto word = index / mp_bits_per_limb;
    auto bit = index % mp_bits_per_limb;
    auto mask = 1UL << bit;
    if (value) {
        m_data[word] |= mask;
    } else {
        m_data[word] &= ~mask;
    }
}

size_t Bitset::count() const { return mpn_popcount(m_data, NUMBER_OF_WORDS(m_size)); }

bool operator==(const Bitset &left, const Bitset &right) {
    if (left.m_size != right.m_size) {
        return false;
    }
    return mpn_cmp(left.m_data, right.m_data, NUMBER_OF_WORDS(left.m_size)) == 0;
}

void Bitset::bit_and(const Bitset &other) {
    assert(m_size == other.m_size);
    mpn_and_n(m_data, m_data, other.m_data, NUMBER_OF_WORDS(m_size));
}

void Bitset::bit_xor(const Bitset &other) {
    assert(m_size == other.m_size);
    mpn_xor_n(m_data, m_data, other.m_data, NUMBER_OF_WORDS(m_size));
}

void Bitset::bit_xnor(const Bitset &other) {
    assert(m_size == other.m_size);
    mpn_xnor_n(m_data, m_data, other.m_data, NUMBER_OF_WORDS(m_size));
    // Mask the last word. TODO check if this is right.
    m_data[NUMBER_OF_WORDS(m_size) - 1] &= MASK_WORD(m_size);
}

void Bitset::bit_flip() {
    mpn_nior_n(m_data, m_data, m_data, NUMBER_OF_WORDS(m_size));
    // Mask the last word. TODO check if this is right.
    m_data[NUMBER_OF_WORDS(m_size) - 1] &= MASK_WORD(m_size);
}

void Bitset::bit_and(const Bitset &left, const Bitset &right, Bitset &result) {
    assert(left.m_size == right.m_size);
    assert(left.m_size == result.m_size);
    mpn_and_n(result.m_data, left.m_data, right.m_data, NUMBER_OF_WORDS(left.m_size));
}

void Bitset::bit_xor(const Bitset &left, const Bitset &right, Bitset &result) {
    assert(left.m_size == right.m_size);
    assert(left.m_size == result.m_size);
    mpn_xor_n(result.m_data, left.m_data, right.m_data, NUMBER_OF_WORDS(left.m_size));
}

void Bitset::bit_xnor(const Bitset &left, const Bitset &right, Bitset &result) {
    assert(left.m_size == right.m_size);
    assert(left.m_size == result.m_size);
    mpn_xnor_n(result.m_data, left.m_data, right.m_data, NUMBER_OF_WORDS(left.m_size));
    // Mask the last word. TODO check if this is right.
    result.m_data[NUMBER_OF_WORDS(left.m_size) - 1] &= MASK_WORD(result.m_size);
}

void Bitset::bit_flip(const Bitset &left, Bitset &result) {
    assert(left.m_size == result.m_size);
    mpn_nior_n(result.m_data, left.m_data, left.m_data, NUMBER_OF_WORDS(left.m_size));
    // Mask the last word. TODO check if this is right.
    result.m_data[NUMBER_OF_WORDS(result.m_size) - 1] &= MASK_WORD(result.m_size);
}

void Bitset::for_each(const std::function<void(size_t)> &fn) const {
    mp_limb_t bitset;
    for (size_t i = 0; i < NUMBER_OF_WORDS(m_size); i++) {
        bitset = m_data[i];
        while (bitset != 0) {
            uint64_t t = bitset & -bitset;
            // int r = __builtin_ctzl(bitset);
            int r = std::countr_zero(bitset);
            fn(i * 64 + r);
            bitset ^= t;
        }
    }
}

size_t Bitset::size() const { return m_size; }

bool Bitset::empty() const { return count() == 0; }
