import matplotlib.pyplot as plt
import numpy as np

# Данные для MPI
mpi_procs = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13])
mpi_time  = np.array([172.8, 87.4, 59.1, 46.6, 37, 31.5, 27.4, 24.3, 22.2, 20.0, 18.6, 17.4, 17.1])

# Данные дляOpenMP
omp_threads= np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13])
omp_time= np.array([111.5, 56.3, 37.7, 28.6, 23.2, 19.4, 16.7, 14.7, 13.1, 11.9, 10.8, 10, 9.2])

# --- График времени выполнения ---
plt.figure(figsize=(10, 6))
plt.plot(omp_threads, omp_time, marker='o', linestyle='-', color='blue', label='OpenMP - Время выполнения')
plt.plot(mpi_procs, mpi_time, marker='s', linestyle='--', color='red', label='MPI - Время выполнения')

plt.title('Сравнение времени выполнения OpenMP и MPI')
plt.xlabel('Количество потоков / процессов')
plt.ylabel('Время выполнения (секунды)')
plt.xticks(np.arange(1, 15, 1)) # Метки на оси X от 1 до 14
plt.grid(True, which="both", ls="--", alpha=0.5)
plt.legend()
plt.tight_layout()
plt.savefig("performance_time.png") # Сохраняем график
plt.show()

# --- График ускорения (Speedup) ---
# Ускорение = Время_на_1_потоке_процессе / Время_на_N_потоках_процессах
omp_speedup = omp_time[0] / omp_time
mpi_speedup = mpi_time[0] / mpi_time

plt.figure(figsize=(10, 6))
plt.plot(omp_threads, omp_speedup, marker='o', linestyle='-', color='blue', label='OpenMP - Ускорение')
plt.plot(mpi_procs, mpi_speedup, marker='s', linestyle='--', color='red', label='MPI - Ускорение')
# Линия идеального ускорения
plt.plot(omp_threads, omp_threads, linestyle=':', color='gray', label='Идеальное ускорение')


plt.title('Сравнение ускорения OpenMP и MPI')
plt.xlabel('Количество потоков / процессов')
plt.ylabel('Ускорение (Speedup)')
plt.xticks(np.arange(1, 15, 1)) # Метки на оси X от 1 до 14
plt.yticks(np.arange(0, max(np.max(omp_speedup), np.max(mpi_speedup), omp_threads[-1]) + 2, 1)) # Метки на оси Y
plt.grid(True, which="both", ls="--", alpha=0.5)
plt.legend()
plt.tight_layout()
plt.savefig("performance_speedup.png") # Сохраняем график
plt.show()

print("Графики сохранены как performance_time.png и performance_speedup.png в директории /home/Haipovich/dev/Projects/study/mpi/")