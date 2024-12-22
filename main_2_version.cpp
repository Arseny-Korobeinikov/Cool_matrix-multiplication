#include <algorithm>
#include <chrono>
#include <iostream>
#include <random>
#include <string>
#include <immintrin.h>

#define MKL
#ifdef MKL
#include "mkl.h"
#endif

using namespace std;

void generation(double * mat, size_t size)
{
	random_device rd;
	mt19937 gen(rd());
	uniform_real_distribution<double> uniform_distance(-2.001, 2.001);
	for (size_t i = 0; i < size * size; i++)
		mat[i] = uniform_distance(gen);
}

// Функция транспонирования матрицы
void transpose(const double* src, double* dst, int size) {
#pragma omp parallel for schedule(static)
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            dst[j * size + i] = src[i * size + j];
        }
    }
}

// Оптимизированная функция умножения матриц для типа double
void matrix_multiply_blocked(const double* a, const double* b, double* res, int size) {
    // Инициализация результирующей матрицы нулями
#pragma omp parallel for schedule(static)
    for (int i = 0; i < size * size; i++) {
        res[i] = 0.0;
    }

    // Транспонирование матрицы b для улучшения локальности доступа
    double* b_transposed = new double[size * size];
    transpose(b, b_transposed, size);

    const size_t block_size = 64;

    // Блочное умножение матриц с использованием SIMD и OpenMP
#pragma omp parallel for schedule(static)
    for (int ii = 0; ii < size; ii += block_size) {
        for (int jj = 0; jj < size; jj += block_size) {
            for (int kk = 0; kk < size; kk += block_size) {
                for (int i = ii; i < ii + block_size && i < size; i++) {
                    for (int j = jj; j < jj + block_size && j < size; j++) {
                        __m256d sum = _mm256_setzero_pd(); // Инициализация SIMD-регистра для накопления
                        int k = kk;
                        // Обработка блоков по 4 элемента (так как __m256d обрабатывает 4 double)
                        for (; k <= kk + block_size - 4 && k + 3 < size; k += 4) {
                            __m256d vec_a = _mm256_loadu_pd(&a[i * size + k]);                // Загрузка 4 элементов из a
                            __m256d vec_b = _mm256_loadu_pd(&b_transposed[j * size + k]);      // Загрузка 4 элементов из транспонированного b
                            sum = _mm256_fmadd_pd(vec_a, vec_b, sum);                         // Fused Multiply-Add
                        }
                        // Суммирование элементов из SIMD-регистра
                        double tmp[4];
                        _mm256_storeu_pd(tmp, sum);
                        double total = tmp[0] + tmp[1] + tmp[2] + tmp[3];

                        // Обработка оставшихся элементов, если размер блока не кратен 4
                        for (; k < kk + block_size && k < size; k++) {
                            total += a[i * size + k] * b_transposed[j * size + k];
                        }

                        res[i * size + j] += total;
                    }
                }
            }
        }
    }

    delete[] b_transposed; // Освобождение памяти транспонированной матрицы
}

int main()
{
	double *mat, *mat_mkl, *a, *b, *a_mkl, *b_mkl;
	size_t size = 1000;
	chrono::time_point<chrono::system_clock> start, end;

	mat = new double[size * size];
	a = new double[size * size];
	b = new double[size * size];
	generation(a, size);
	generation(b, size);
	memset(mat, 0, size*size * sizeof(double));

#ifdef MKL     
    mat_mkl = new double[size * size];
	a_mkl = new double[size * size];
	b_mkl = new double[size * size];
	memcpy(a_mkl, a, sizeof(double)*size*size);
    memcpy(b_mkl, b, sizeof(double)*size*size);
	memset(mat_mkl, 0, size*size * sizeof(double));
#endif

	start = chrono::system_clock::now();
	matrix_mult(a, b, mat, size);
	end = chrono::system_clock::now();
    
   
	int elapsed_seconds = chrono::duration_cast<chrono::milliseconds>
		(end - start).count();
	cout << "Total time: " << elapsed_seconds/1000.0 << " sec" << endl;

#ifdef MKL 
	start = chrono::system_clock::now();
	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                size, size, size, 1.0, a_mkl, size, b_mkl, size, 0.0, mat_mkl, size);
    end = chrono::system_clock::now();
    
    elapsed_seconds = chrono::duration_cast<chrono::milliseconds>
		(end - start).count();
	cout << "Total time mkl: " << elapsed_seconds/1000.0 << " sec" << endl;
     
    int flag = 0;
    for (unsigned int i = 0; i < size * size; i++)
        if(abs(mat[i] - mat_mkl[i]) > size*1e-14){
		    flag = 1;
        }
    if (flag)
        cout << "fail" << endl;
    else
        cout << "correct" << endl; 
    
    delete (a_mkl);
    delete (b_mkl);
    delete (mat_mkl);
#endif

    delete (a);
    delete (b);
    delete (mat);

	//system("pause");
	
	return 0;
}