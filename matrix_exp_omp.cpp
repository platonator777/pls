#include <cmath>
#include <cstdio>
#include <omp.h>

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
    double taylor_sum[SIZE][SIZE] = {0};
    double current_term[SIZE][SIZE];
    double temp_matrix[SIZE][SIZE];

    // Calculate optimal M_jump_size
    long M_jump_size = (long)sqrt((double)N_TERMS);
    if (M_jump_size == 0) M_jump_size = 1;

    double start_time = omp_get_wtime();

    // Precompute A^M
    Matrix A_pow_M_struct;
    if (M_jump_size == 1) {
        for(int r=0; r<SIZE; ++r) 
            for(int c=0; c<SIZE; ++c) 
                A_pow_M_struct.data[r][c] = A[r][c];
    } else {
        matrix_identity(A_pow_M_struct.data);
        for (long i = 0; i < M_jump_size; ++i) {
            A_pow_M_struct = matrix_multiply(A_pow_M_struct.data, A);
        }
    }

    // Parallel computation of Taylor series
    #pragma omp parallel
    {
        double local_sum[SIZE][SIZE] = {0};
        double local_current_term[SIZE][SIZE];
        double local_temp_matrix[SIZE][SIZE];

        // Determine each thread's chunk of work
        int thread_id = omp_get_thread_num();
        int num_threads = omp_get_num_threads();
        
        long chunk_size = N_TERMS / num_threads;
        long remainder = N_TERMS % num_threads;
        
        long start_k = thread_id * chunk_size + ((thread_id < remainder) ? thread_id : remainder) + 1;
        long end_k = start_k + chunk_size - 1 + (thread_id < remainder ? 1 : 0);
        
        if (end_k > N_TERMS) end_k = N_TERMS;
        if (start_k > end_k) start_k = end_k + 1; // No work for this thread

        // Compute initial term for this thread's chunk
        Matrix current_T_val_struct;
        double my_initial_term_T_km1[SIZE][SIZE];
        
        long target_k_for_precalc = start_k - 1;
        
        if (target_k_for_precalc <= 0) {
            matrix_identity(my_initial_term_T_km1);
        } else {
            matrix_identity(current_T_val_struct.data);
            long current_k_val = 0;

            // Process in jumps of M_jump_size
            for (long j = 0; j < target_k_for_precalc / M_jump_size; ++j) {
                double scalar_prod_inv = 1.0;
                for (long l = 1; l <= M_jump_size; ++l) {
                    scalar_prod_inv /= (double)(current_k_val + l);
                }
                Matrix temp_prod_struct = matrix_multiply(current_T_val_struct.data, A_pow_M_struct.data);
                current_T_val_struct = matrix_scalar_multiply(temp_prod_struct.data, scalar_prod_inv);
                current_k_val += M_jump_size;
            }

            // Process remaining terms
            for (long l = 0; l < target_k_for_precalc % M_jump_size; ++l) {
                current_k_val += 1;
                Matrix temp_prod_struct = matrix_multiply(current_T_val_struct.data, A);
                current_T_val_struct = matrix_scalar_multiply(temp_prod_struct.data, 1.0 / (double)current_k_val);
            }
            
            for(int r=0; r<SIZE; ++r) 
                for(int c=0; c<SIZE; ++c) 
                    my_initial_term_T_km1[r][c] = current_T_val_struct.data[r][c];
        }

        // Initialize current term
        for(int r=0; r<SIZE; ++r) 
            for(int c=0; c<SIZE; ++c) 
                local_current_term[r][c] = my_initial_term_T_km1[r][c];

        // Compute terms in this thread's chunk
        for (long k = start_k; k <= end_k; ++k) {
            Matrix term_A_prod_struct = matrix_multiply(local_current_term, A);
            for (int r = 0; r < SIZE; ++r) 
                for (int c = 0; c < SIZE; ++c) 
                    local_temp_matrix[r][c] = term_A_prod_struct.data[r][c];
                    
            Matrix actual_term_k_struct = matrix_scalar_multiply(local_temp_matrix, 1.0 / (double)k);
            Matrix new_sum_struct = matrix_add(local_sum, actual_term_k_struct.data);
            
            for (int r = 0; r < SIZE; ++r) 
                for (int c = 0; c < SIZE; ++c) {
                    local_sum[r][c] = new_sum_struct.data[r][c];
                    local_current_term[r][c] = actual_term_k_struct.data[r][c];
                }
        }

        // Combine results from all threads
        #pragma omp critical
        {
            for (int r = 0; r < SIZE; ++r) 
                for (int c = 0; c < SIZE; ++c) 
                    taylor_sum[r][c] += local_sum[r][c];
        }
    }

    // Add the identity matrix (k=0 term)
    matrix_identity(temp_matrix);
    Matrix final_sum_struct = matrix_add(taylor_sum, temp_matrix);
    for (int r = 0; r < SIZE; ++r) 
        for (int c = 0; c < SIZE; ++c) 
            taylor_sum[r][c] = final_sum_struct.data[r][c];

    double end_time = omp_get_wtime();

    matrix_print("Matrix A", A);
    matrix_print("Result e^A (Taylor approximation)", taylor_sum);
    printf("Time taken: %f seconds\n", end_time - start_time);

    return 0;
}