#include <omp.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define SIZE 3
#define N_TERMS 5000000000

typedef struct {
    double data[SIZE][SIZE];
} Matrix;

void matrix_print(const char* label, const double matrix[SIZE][SIZE]) {
    printf("%s:\n", label);
    for (int i = 0; i < SIZE; i++) {
        for (int j = 0; j < SIZE; j++) {
            printf("%8.4f ", matrix[i][j]);
        }
        printf("\n");
    }
    printf("\n");
}

Matrix matrix_multiply(const double matrix1[SIZE][SIZE], const double matrix2[SIZE][SIZE]) {
    Matrix result;
    for (int i = 0; i < SIZE; i++) {
        for (int j = 0; j < SIZE; j++) {
            result.data[i][j] = 0.0;
            for (int k = 0; k < SIZE; k++) {
                result.data[i][j] += matrix1[i][k] * matrix2[k][j];
            }
        }
    }
    return result;
}

Matrix matrix_add(const double matrix1[SIZE][SIZE], const double matrix2[SIZE][SIZE]) {
    Matrix result;
    for (int i = 0; i < SIZE; i++) {
        for (int j = 0; j < SIZE; j++) {
            result.data[i][j] = matrix1[i][j] + matrix2[i][j];
        }
    }
    return result;
}

void matrix_identity(double matrix[SIZE][SIZE]) {
    for (int i = 0; i < SIZE; i++) {
        for (int j = 0; j < SIZE; j++) {
            matrix[i][j] = (i == j) ? 1.0 : 0.0;
        }
    }
}

Matrix matrix_scalar_multiply(const double matrix[SIZE][SIZE], double scalar) {
    Matrix result;
    for (int i = 0; i < SIZE; i++) {
        for (int j = 0; j < SIZE; j++) {
            result.data[i][j] = matrix[i][j] * scalar;
        }
    }
    return result;
}

int main() {
    double A[SIZE][SIZE] = {
        {0.1, 0.4, 0.2},
        {0.3, 0.0, 0.5},
        {0.6, 0.2, 0.1}
    };

    double global_taylor_sum[SIZE][SIZE] = {0.0};
    double identity_matrix[SIZE][SIZE];
    matrix_identity(identity_matrix);

    double start_time = omp_get_wtime();

    // Используем редукцию по матрице через приватные переменные
    #pragma omp parallel
    {
        Matrix local_sum;
        for (int i = 0; i < SIZE; ++i)
            for (int j = 0; j < SIZE; ++j)
                local_sum.data[i][j] = 0.0;

        Matrix current_term;
        for (int i = 0; i < SIZE; ++i)
            for (int j = 0; j < SIZE; ++j)
                current_term.data[i][j] = identity_matrix[i][j];

        #pragma omp for schedule(static)
        for (long k = 1; k <= N_TERMS; ++k) {
            current_term = matrix_multiply(current_term.data, A);
            Matrix term = matrix_scalar_multiply(current_term.data, 1.0 / (double)k);
            local_sum = matrix_add(local_sum.data, term.data);
        }

        // Критическая секция для аккумулирования глобального результата
        #pragma omp critical
        {
            for (int i = 0; i < SIZE; ++i)
                for (int j = 0; j < SIZE; ++j)
                    global_taylor_sum[i][j] += local_sum.data[i][j];
        }
    }

    // Добавляем единичную матрицу (T_0 = I)
    for (int i = 0; i < SIZE; ++i)
        for (int j = 0; j < SIZE; ++j)
            global_taylor_sum[i][j] += identity_matrix[i][j];

    double end_time = omp_get_wtime();

    matrix_print("Matrix A", A);
    matrix_print("Result e^A (Taylor approximation)", global_taylor_sum);
    printf("Time taken: %f seconds\n", end_time - start_time);

    return 0;
}
