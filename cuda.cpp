#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <cmath>
#include <iomanip>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define CUDA_CHECK(api_call) \
    do { \
        cudaError_t error_status = (api_call); \
        if (error_status != cudaSuccess) { \
            std::cerr << "CUDA error in " << __FILE__ << ":" << __LINE__ \
                      << " for call '" << #api_call << "': " \
                      << cudaGetErrorString(error_status) << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

__device__ double device_f(double x) {
    // Обработка границ и особых точек
    if (x <= 1e-12 || x >= M_PI - 1e-12) return 0.0;
    
    double sin_x = sin(x);
    if (fabs(sin_x) < 1e-12) return 0.0;  // Избегаем cot(x) = cos(x)/sin(x)
    
    double cot_x = cos(x) / sin_x;
    double sin_term = 1.0 + sin_x;
    
    // Логарифм от отрицательного числа или нуля не определен
    if (sin_term <= 1e-12) return 0.0;
    double ln_term = log(sin_term);
    
    // Проверка знаменателя
    double sin_ln_term = sin(sin_term);
    if (fabs(ln_term * sin_ln_term) < 1e-100) return 0.0;
    
    return cot_x / (ln_term * sin_ln_term);
}

__global__ void integrate_kernel(double a, double delta_x, int n_steps, double* partial_sums_d) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n_steps) {
        double x_i = a + (idx + 0.5) * delta_x;  // Метод средних прямоугольников
        partial_sums_d[idx] = device_f(x_i) * delta_x;
    }
}

int main() {
    // Параметры интегрирования
    double a = 1e-6;       // Начало интервала (не 0 из-за особенности)
    double b = M_PI - 1e-6; // Конец интервала (не Pi из-за особенности)
    const int N = 1000000;  // Число шагов (уменьшено для стабильности)
    const int threads_per_block = 256;

    double delta_x = (b - a) / N;

    std::cout << "Integrating f(x) = cot(x) / (ln(1+sin(x)) * sin(1+sin(x)))" << std::endl;
    std::cout << "Interval: [" << a << ", " << b << "]" << std::endl;
    std::cout << "Number of steps: " << N << std::endl;
    std::cout << "Delta x: " << delta_x << std::endl;

    // Выделение памяти на GPU и CPU
    double *partial_sums_d;
    std::vector<double> partial_sums_h(N);
    size_t size = N * sizeof(double);
    
    CUDA_CHECK(cudaMalloc(&partial_sums_d, size));

    // Создание событий для замера времени
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // Запуск ядра
    int blocks = (N + threads_per_block - 1) / threads_per_block;
    CUDA_CHECK(cudaEventRecord(start));
    integrate_kernel<<<blocks, threads_per_block>>>(a, delta_x, N, partial_sums_d);
    CUDA_CHECK(cudaEventRecord(stop));
    
    // Копирование результатов и синхронизация
    CUDA_CHECK(cudaMemcpy(partial_sums_h.data(), partial_sums_d, size, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaDeviceSynchronize());

    // Вычисление времени выполнения
    float time_ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&time_ms, start, stop));

    // Суммирование результатов
    double total_sum = 0.0;
    for (int i = 0; i < N; ++i) {
        total_sum += partial_sums_h[i];
    }

    // Вывод результата
    std::cout << std::fixed << std::setprecision(15);
    std::cout << "Integral result: " << total_sum << std::endl;
    std::cout << "Kernel time: " << time_ms << " ms" << std::endl;

    // Освобождение ресурсов
    CUDA_CHECK(cudaFree(partial_sums_d));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return 0;
}