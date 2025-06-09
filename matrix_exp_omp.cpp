#include <cmath>
#include <iostream>
#include <vector>
#include <omp.h>

constexpr int SIZE = 3;
constexpr int N_TERMS = 5000;

class Matrix {
public:
    double data[SIZE][SIZE];

    Matrix() {
        #pragma omp parallel for collapse(2)
        for (int i = 0; i < SIZE; ++i)
            for (int j = 0; j < SIZE; ++j)
                data[i][j] = 0.0;
    }

    Matrix(double value) {
        #pragma omp parallel for collapse(2)
        for (int i = 0; i < SIZE; ++i)
            for (int j = 0; j < SIZE; ++j)
                data[i][j] = value;
    }

    static Matrix Identity() {
        Matrix m;
        #pragma omp parallel for
        for (int i = 0; i < SIZE; ++i)
            m.data[i][i] = 1.0;
        return m;
    }

    void print(const std::string& label) const {
        std::cout << label << " (" << SIZE << "x" << SIZE << "):\n";
        for (int i = 0; i < SIZE; ++i) {
            std::cout << "[ ";
            for (int j = 0; j < SIZE; ++j) {
                printf("%8.4f ", data[i][j]);
            }
            std::cout << "]\n";
        }
    }

    Matrix operator+(const Matrix& other) const {
        Matrix result;
        #pragma omp parallel for collapse(2)
        for (int i = 0; i < SIZE; ++i)
            for (int j = 0; j < SIZE; ++j)
                result.data[i][j] = data[i][j] + other.data[i][j];
        return result;
    }

    Matrix operator*(const Matrix& other) const {
        Matrix result;
        #pragma omp parallel for collapse(2)
        for (int i = 0; i < SIZE; ++i) {
            for (int j = 0; j < SIZE; ++j) {
                double sum = 0.0;
                #pragma omp simd reduction(+:sum)
                for (int k = 0; k < SIZE; ++k) {
                    sum += data[i][k] * other.data[k][j];
                }
                result.data[i][j] = sum;
            }
        }
        return result;
    }

    Matrix operator*(double scalar) const {
        Matrix result;
        #pragma omp parallel for collapse(2)
        for (int i = 0; i < SIZE; ++i)
            for (int j = 0; j < SIZE; ++j)
                result.data[i][j] = data[i][j] * scalar;
        return result;
    }
};

Matrix calculateTaylorSum(const Matrix& A) {
    Matrix taylor_sum = Matrix::Identity(); // Start with identity matrix (k=0 term)
    Matrix current_term = Matrix::Identity(); // Current term (A^k / k!)
    
    for (long k = 1; k <= N_TERMS; ++k) {
        current_term = current_term * A * (1.0 / k);
        
        // Parallel reduction for matrix addition
        #pragma omp parallel for collapse(2)
        for (int i = 0; i < SIZE; ++i) {
            for (int j = 0; j < SIZE; ++j) {
                taylor_sum.data[i][j] += current_term.data[i][j];
            }
        }
    }
    
    return taylor_sum;
}

int main(int argc, char* argv[]) {
    Matrix A;
    A.data[0][0] = 1; A.data[0][1] = -1; A.data[0][2] = -1;
    A.data[1][0] = 1; A.data[1][1] = 1; A.data[1][2] = 0;
    A.data[2][0] = 3; A.data[2][1] = 0; A.data[2][2] = 1;

    double start_time = omp_get_wtime();

    Matrix taylor_sum = calculateTaylorSum(A);

    double end_time = omp_get_wtime();

    std::cout << "Matrix size: " << SIZE << "x" << SIZE << "\n";
    std::cout << "Number of terms: " << N_TERMS << " (+ Identity)\n";
    
    A.print("A (Initial)");
    std::cout << "Calculation finished.\n";
    printf("Execution time: %f seconds\n", end_time - start_time);
    taylor_sum.print("e^A (Result)");

    return 0;
}