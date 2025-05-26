#pragma once

#include <cassert>
#include <cstddef>
#include <cstring>
#include <fstream>
#include <istream>
#include <ostream>
#include <stdexcept>

template<class T>
class Matrix {
public:
    Matrix();

    Matrix(size_t n_rows, size_t n_columns);

    Matrix(size_t n_rows, size_t n_columns, const T&initial_value);

    Matrix(const Matrix&other);

    Matrix(Matrix&&other) noexcept;

    ~Matrix();

    Matrix<T>& operator=(const Matrix<T>&other);

    Matrix<T>& operator=(Matrix<T>&&other) noexcept;

    T operator()(size_t row_index, size_t column_index) const;

    T& operator()(size_t row_index, size_t column_index);

    // Performs bound checking
    T at(size_t row_index, size_t column_index) const;

    T& at(size_t row_index, size_t column_index);

    [[nodiscard]] size_t n_rows() const;

    [[nodiscard]] size_t n_columns() const;

    T* data();

    friend std::ostream & operator<<(std::ostream &os, const Matrix&m) {
        os << m.m_rows << " " << m.m_columns << std::endl;
        for (size_t i = 0; i < m.n_rows(); i++) {
            for (size_t j = 0; j < m.n_columns(); j++) {
                os << m(i, j) << " ";
            }
            os << std::endl;
        }
        return os;
    }

    friend std::istream & operator>>(std::istream &is, Matrix &m) {
        size_t rows, columns;
        is >> rows >> columns;
        m = Matrix<T>(rows, columns);
        for (size_t i = 0; i < m.n_rows(); i++) {
            for (size_t j = 0; j < m.n_columns(); j++) {
                is >> m(i, j);
            }
        }
        return is;
    }

private:
    size_t m_rows;
    size_t m_columns;
    T* m_data;
};

template<class T>
Matrix<T>::Matrix() : m_rows(0), m_columns(0), m_data(nullptr) {
}

template<class T>
Matrix<T>::Matrix(size_t n_rows, size_t n_columns)
    : m_rows(n_rows), m_columns(n_columns), m_data(new T[m_rows * m_columns]) {
}

template<class T>
Matrix<T>::Matrix(size_t n_rows, size_t n_columns, const T&initial_value)
    : m_rows(n_rows), m_columns(n_columns), m_data(new T[m_rows * m_columns]) {
    for (size_t i = 0; i < m_rows; i++) {
        for (size_t j = 0; j < m_columns; j++) {
            m_data[i * m_columns + j] = initial_value;
        }
    }
}

template<class T>
Matrix<T>::Matrix(const Matrix<T>&other)
    : m_rows(other.m_rows), m_columns(other.m_columns), m_data(new T[m_rows * m_columns]) {
    memcpy(m_data, other.m_data, m_rows * m_columns * sizeof(T));
}

template<class T>
Matrix<T>::Matrix(Matrix<T>&&other) noexcept : m_rows(other.m_rows), m_columns(other.m_columns), m_data(other.m_data) {
    other.m_rows = 0;
    other.m_columns = 0;
    other.m_data = nullptr;
}

template<class T>
Matrix<T>::~Matrix() {
    delete[] m_data;
}

template<class T>
Matrix<T>& Matrix<T>::operator=(const Matrix<T>&other) {
    if (this == &other) return *this;
    delete[] m_data;
    m_rows = other.m_rows;
    m_columns = other.m_columns;
    m_data = new T[m_rows * m_columns];
    memcpy(m_data, other.m_data, m_rows * m_columns * sizeof(T));
    return *this;
}

template<class T>
Matrix<T>& Matrix<T>::operator=(Matrix<T>&&other) noexcept {
    delete[] m_data;
    m_rows = other.m_rows;
    m_columns = other.m_columns;
    m_data = other.m_data;

    other.m_rows = 0;
    other.m_columns = 0;
    other.m_data = nullptr;
    return *this;
}

template<class T>
T Matrix<T>::operator()(size_t row_index, size_t col_index) const {
    return m_data[row_index * m_columns + col_index];
}

template<class T>
T& Matrix<T>::operator()(size_t row_index, size_t col_index) {
    return m_data[row_index * m_columns + col_index];
}

template<class T>
T Matrix<T>::at(size_t row_index, size_t column_index) const {
    if (row_index >= m_rows || column_index >= m_columns) {
        throw std::out_of_range("[Matrix] attempted an out of bounds access.");
    }
    return m_data[row_index * m_columns + column_index];
}

template<class T>
T& Matrix<T>::at(size_t row_index, size_t column_index) {
    if (row_index >= m_rows || column_index >= m_columns) {
        throw std::out_of_range("[Matrix] attempted an out of bounds access.");
    }
    return m_data[row_index * m_columns + column_index];
}

template<class T>
size_t Matrix<T>::n_rows() const {
    return m_rows;
}

template<class T>
size_t Matrix<T>::n_columns() const {
    return m_columns;
}

template<class T>
T* Matrix<T>::data() {
    return m_data;
}
