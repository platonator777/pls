#include <cmath>
#include <iostream>
#include <vector>
#include <mpi.h>

constexpr int SIZE = 3;
constexpr int N_TERMS = 5000;

class Matrix {
public:
    double data[SIZE][SIZE];

    Matrix() {
        for (int i = 0; i < SIZE; ++i)
            for (int j = 0; j < SIZE; ++j)
                data[i][j] = 0.0;
    }

    Matrix(double value) {
        for (int i = 0; i < SIZE; ++i)
            for (int j = 0; j < SIZE; ++j)
                data[i][j] = value;
    }

    static Matrix Identity() {
        Matrix m;
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
        for (int i = 0; i < SIZE; ++i)
            for (int j = 0; j < SIZE; ++j)
                result.data[i][j] = data[i][j] + other.data[i][j];
        return result;
    }

    Matrix operator*(const Matrix& other) const {
        Matrix result;
        for (int i = 0; i < SIZE; ++i) {
            for (int j = 0; j < SIZE; ++j) {
                for (int k = 0; k < SIZE; ++k) {
                    result.data[i][j] += data[i][k] * other.data[k][j];
                }
            }
        }
        return result;
    }

    Matrix operator*(double scalar) const {
        Matrix result;
        for (int i = 0; i < SIZE; ++i)
            for (int j = 0; j < SIZE; ++j)
                result.data[i][j] = data[i][j] * scalar;
        return result;
    }
};

void matrixSumMpiOp(void* invec, void* inoutvec, int* len, MPI_Datatype* datatype) {
    double* in_matrix = static_cast<double*>(invec);
    double* inout_matrix = static_cast<double*>(inoutvec);

    if (*len != SIZE * SIZE) {
        std::cerr << "Error in matrixSumMpiOp: len mismatch! Expected " 
                  << SIZE * SIZE << ", got " << *len << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    for (int i = 0; i < SIZE * SIZE; ++i) {
        inout_matrix[i] += in_matrix[i];
    }
}

Matrix calculateTaylorSum(const Matrix& A, int rank, int num_procs) {
    Matrix local_taylor_sum;
    Matrix temp_matrix;

    Matrix A_P = A;
    if (num_procs == 0) {
        MPI_Abort(MPI_COMM_WORLD, 1);
        return local_taylor_sum;
    }

    if (num_procs > 1) {
        for (int i = 1; i < num_procs; ++i) {
            A_P = A_P * A;
        }
    }

    long k_first_for_proc = rank + 1;

    if (k_first_for_proc <= N_TERMS) {
        Matrix A_power_k_first = Matrix::Identity();

        for (long p = 1; p <= k_first_for_proc; ++p) {
            A_power_k_first = A_power_k_first * A;
        }

        double factorial_k_first = 1.0;
        for (long p = 1; p <= k_first_for_proc; ++p) {
            factorial_k_first *= static_cast<double>(p);
        }

        Matrix current_term = A_power_k_first * (1.0 / factorial_k_first);
        local_taylor_sum = local_taylor_sum + current_term;

        for (long k_current = k_first_for_proc; k_current <= N_TERMS - num_procs; k_current += num_procs) {
            long k_next = k_current + num_procs;

            current_term = current_term * A_P;

            double product_for_denominator = 1.0;
            for (long val = k_current + 1; val <= k_next; ++val) {
                product_for_denominator *= static_cast<double>(val);
            }

            if (product_for_denominator == 0) {
                std::cerr << "Rank " << rank << ": product_for_denominator is zero at k_current=" 
                          << k_current << ", k_next=" << k_next << std::endl;
                MPI_Abort(MPI_COMM_WORLD, 1);
            }

            current_term = current_term * (1.0 / product_for_denominator);
            local_taylor_sum = local_taylor_sum + current_term;
        }
    }

    return local_taylor_sum;
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    Matrix A;
    A.data[0][0] = 1; A.data[0][1] = -1; A.data[0][2] = -1;
    A.data[1][0] = 1; A.data[1][1] = 1; A.data[1][2] = 0;
    A.data[2][0] = 3; A.data[2][1] = 0; A.data[2][2] = 1;

    double start_time = MPI_Wtime();

    Matrix local_taylor_sum = calculateTaylorSum(A, rank, num_procs);

    MPI_Op matrix_sum_op;
    MPI_Op_create(matrixSumMpiOp, 1, &matrix_sum_op);

    Matrix global_taylor_sum;
    MPI_Reduce(&local_taylor_sum.data[0][0], &global_taylor_sum.data[0][0], 
               SIZE * SIZE, MPI_DOUBLE, matrix_sum_op, 0, MPI_COMM_WORLD);

    MPI_Op_free(&matrix_sum_op);

    double end_time = MPI_Wtime();

    if (rank == 0) {
        global_taylor_sum = global_taylor_sum + Matrix::Identity();

        std::cout << "Matrix size: " << SIZE << "x" << SIZE << "\n";
        std::cout << "Number of terms: " << N_TERMS << " (+ Identity)\n";
        std::cout << "Number of processors: " << num_procs << "\n";

        A.print("A (Initial)");
        std::cout << "Calculation finished.\n";
        printf("Execution time: %f seconds\n", end_time - start_time);
        global_taylor_sum.print("e^A (Result)");
    }

    MPI_Finalize();
    return 0;
}