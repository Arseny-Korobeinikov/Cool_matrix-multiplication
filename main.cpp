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

void transpose(const double* src, double* dst, int size) {
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            dst[j * size + i] = src[i * size + j];
        }
    }
}

void matrix_mult(double * a, double * b, double * res, size_t size)
{
	double* b_transposed = new double[size * size];
    transpose(b, b_transposed, size);

    const size_t block_size = 64;

 #pragma omp parallel for schedule(static)
    for (int ii = 0; ii < size; ii += block_size) {
        for (int jj = 0; jj < size; jj += block_size) {
            for (int kk = 0; kk < size; kk += block_size) {
                for (int i = ii; i < ii + block_size && i < size; i++) {
                    for (int j = jj; j < jj + block_size && j < size; j++) {
                        __m256 sum = _mm256_setzero_ps(); // SIMD-регистры для накопления
                        for (int k = kk; k < kk + block_size && k < size; k += 8) {
                            __m256 vec_a = _mm256_loadu_pd(&a[i * size + k]);       // Загружаем 8 элементов из `a`
                            __m256 vec_b = _mm256_loadu_pd(&b_transposed[j * size + k]); // Загружаем 8 элементов из транспонированного `b`
                            sum = _mm256_fmadd_ps(vec_a, vec_b, sum);              // Fused Multiply-Add
                        }
                        // Суммируем элементы из sum и обновляем res
                        float tmp[8];
                        _mm256_storeu_ps(tmp, sum);
                        for (int t = 0; t < 8; t++) {
                            res[i * size + j] += tmp[t];
                        }
                    }
                }
            }
        }
    }

    delete[] b_transposed; // Удаляем транспонированную матрицу
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