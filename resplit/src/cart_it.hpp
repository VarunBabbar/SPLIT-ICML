
#ifndef CART_IT
#define CART_IT

#include <algorithm>
#include <iostream>
#include <iterator>
#include <map>
#include <memory>
#include <set>
#include <vector>

using namespace std;

// typedef int T;

// Compute the cartesian product of input
// {S_1, S_2, ..., S_n} --> S_1 x S_2 x ... S_n
// For a {L_1, L_2, ..., L_n} vector where L_i is the legnth of the ith inner
// vector, generate L_1 * ... * L_n vectors of length n
template <typename T> class CartIt {
  public:
    typedef vector<T> value_type;
    typedef vector<T> &reference;
    typedef vector<T> *pointer;

    // Should really do a const iterator but whatever lol
    class iterator {
      public:
        typedef iterator self_type;
        typedef std::forward_iterator_tag iterator_category;
        typedef int difference_type;
        iterator(CartIt *base) : base(base) { idx = 0; }
        iterator(CartIt *base, long long idx) : base(base), idx(idx) {}
        self_type operator++() {
            self_type i = *this;
            idx++;
            return i;
        }
        self_type operator++(int junk) {
            idx++;
            return *this;
        }
        value_type operator*() { return base->access_idx(idx); }
        // pointer operator->() { return ptr_; }
        bool operator==(const self_type &rhs) { return idx == rhs.idx; }
        bool operator!=(const self_type &rhs) { return idx != rhs.idx; }

      private:
        CartIt *base;
        long long int idx;
    };

    CartIt(const vector<vector<T>> &item) : v(item) {
        auto product = [](long long int a, const vector<T> &b) {
            return a * b.size();
        };
        N = accumulate(v.begin(), v.end(), 1LL, product);
    }

    ~CartIt() {}

    // size_type size() const { return size_; }

    // T& operator[](size_type index)
    // {
    //     assert(index < size_);
    //     return data_[index];
    // }

    // const T& operator[](size_type index) const
    // {
    //     assert(index < size_);
    //     return data_[index];
    // }

    iterator begin() { return iterator(this); }

    iterator end() { return iterator(this, N); }

    // int size() {
    //     if (clipped_size < 0) {
    //         return keys.size();
    //     } else {
    //         return clipped_size;
    //     }
    // }

    // float scope;

  private:
    const vector<vector<T>> &v;
    long long N;
    // From
    // https://stackoverflow.com/questions/5279051/how-can-i-create-cartesian-product-of-vector-of-vectors
    value_type access_idx(const long long n) {
        value_type u(v.size());
        lldiv_t q{n, 0};
        for (long long i = v.size() - 1; 0 <= i; --i) {
            q = div(q.quot, v[i].size());
            u[i] = v[i][q.rem];
        }
        return u;
    }
};
#endif