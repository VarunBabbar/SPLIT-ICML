#ifndef LIBGOSDT_BITSET_HPP
#define LIBGOSDT_BITSET_HPP

#include <gmp.h>

#include <cstddef>
#include <cstdint>
#include <functional>
#include <vector>

class Bitset {
public:
    /// Copy constructor for Bitset.
    Bitset(const Bitset &other);

    /// Move constructor for Bitset.
    Bitset(Bitset &&other) noexcept;

    /// Destructor for Bitset.
    ~Bitset();

    /// Create a Bitset with the given size. Initializes the underlying memory to fill the Bitset.
    static Bitset create_full(size_t size);

    /// Create a Bitset with the given size. Initializes the underlying memory to empty the Bitset.
    static Bitset create_empty(size_t size);

    /// Copy assignment operator for Bitset.
    Bitset &operator=(const Bitset &other);

    /// Move assignment operator for Bitset.
    Bitset &operator=(Bitset &&other) noexcept;

    /// Returns whether the index bit is set.
    [[nodiscard]] bool get(size_t index) const;

    /// Sets the index bit to the given value.
    void set(size_t index, bool value) const;

    /// Returns the number of set bits in the Bitset.
    [[nodiscard]] size_t count() const;

    /// Returns the number of potential elements in the Bitset.
    [[nodiscard]] size_t size() const;

    /// Returns whether the Bitset is empty.
    [[nodiscard]] bool empty() const;

    /// Returns the hash of the Bitset.
    [[nodiscard]] size_t hash_value() const;

    /// Inplace bitwise AND with the other Bitset.
    void bit_and(const Bitset &other);

    /// Inplace bitwise XOR with the other Bitset.
    void bit_xor(const Bitset &other);

    /// Inplace bitwise XNOR with the other Bitset.
    void bit_xnor(const Bitset &other);

    /// Inplace bitwise NOT.
    void bit_flip();

    /// Performs bitwise AND with the left and right Bitsets and stores the result in the result Bitset.
    static void bit_and(const Bitset &left, const Bitset &right, Bitset &result);

    /// Performs bitwise XOR with the left and right Bitsets and stores the result in the result Bitset.
    static void bit_xor(const Bitset &left, const Bitset &right, Bitset &result);

    /// Performs bitwise XNOR with the left and right Bitsets and stores the result in the result Bitset.
    static void bit_xnor(const Bitset &left, const Bitset &right, Bitset &result);

    /// Performs bitwise NOT with the left Bitset and stores the result in the result Bitset.
    static void bit_flip(const Bitset &left, Bitset &result);

    /// Applies the given function to each set bit in the Bitset.
    void for_each(const std::function<void(size_t)> &fn) const;

private:
    /// Create a Bitset with the given size. Does not initialize the underlying memory.
    explicit Bitset(size_t size);

    /// The number of potential elements in the bitset.
    size_t m_size;

    /// Pointer to the Bitset data.
    mp_ptr m_data;

    /// Equality operator for Bitset.
    bool friend operator==(const Bitset &left, const Bitset &right);
};

#endif  // LIBGOSDT_BITSET_HPP
