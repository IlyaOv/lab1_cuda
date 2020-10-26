#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h> 
#include <chrono>

#define BLOCK_SIZE 16

//функция ядра
__global__ void matrixMult(const double *A, const double *B, double *C, int n)
{
	int ai = n * (blockDim.y * blockIdx.y + threadIdx.y);
	int bj = blockDim.x * blockIdx.x + threadIdx.x;
	double sum = 0;
	for (int k = 0; k < n; k++)
		sum += A[ai + k] * B[k * n + bj];
	int index = n * (blockDim.y * blockIdx.y + threadIdx.y) + blockDim.x * blockIdx.x + threadIdx.x;
	C[index] = sum;
}

//функция ядра с разделяемой памятью
__global__ void matrixMultShared(double* A, double* B, double* C, int n) {
	int bx = blockIdx.x;
	int by = blockIdx.y;
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int aBegin = n * BLOCK_SIZE * by;
	int aEnd = aBegin + n - 1;
	int aStep = BLOCK_SIZE;
	int bBegin = BLOCK_SIZE * bx;
	int bStep = BLOCK_SIZE * n; 
	double Csub = 0; 
	for (int a = aBegin, b = bBegin; a <= aEnd; a += aStep, b += bStep) {
		__shared__ double As[BLOCK_SIZE][BLOCK_SIZE];
		__shared__ double Bs[BLOCK_SIZE][BLOCK_SIZE];
		As[ty][tx] = A[a + n * ty + tx];
		Bs[ty][tx] = B[b + n * ty + tx];
		__syncthreads(); 
		for (int k = 0; k < BLOCK_SIZE; ++k)
			Csub += As[ty][k] * Bs[k][tx];
		__syncthreads();
	}
	int c = n * BLOCK_SIZE * by + BLOCK_SIZE * bx;
	C[c + n * ty + tx] = Csub;
}

// генерация матриц
double * generateRandMatrix(int n, size_t sizeMatrix) {
	double * matrix = (double *)malloc(sizeMatrix);
	for (int i = 0; i < n * n; i++) {
		matrix[i] = (double)rand() / (double)RAND_MAX;
	}
	return matrix;
}

void printMatrix(double * matrix, int n) {
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < n; j++) {
			printf("%4.1lf ", matrix[i*n + j]);
		}
		printf("\n");
	}
}

// функция для последовательного варианта умножения матриц
void matrixMultCPU(double* A, double* B, double * C, int n) {
	for (int i = 0; i<n; i++) {
		for (int j = 0; j<n; j++) {
			for (int k = 0; k<n; k++) {
				C[i*n + j] += A[i*n + k] * B[k*n + j];
			}
		}
	}
}

// проверка результатов умножения
bool checkMult(double * C1, double * C2, int n) {
	double accuracy = 1.e-6;
	for (int i = 0; i < n*n; i++) {
		if (abs(C1[i] - C2[i]) >= accuracy)
			return false;
	}
	return true;
}

int main(int argc, char *argv[])
{
    int N = atoi(argv[1]);
	int flag_s = atoi(argv[2]);
	if (N % 16 != 0) {
		printf("The number is not a multiple of the block size. The program will be closed.\n");
		system("pause");
		exit(1);
	}
	
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	srand(time(NULL));
	size_t sizeMatrix = sizeof(double) * N * N;

	double * h_A = generateRandMatrix(N, sizeMatrix);
	double * h_B = generateRandMatrix(N, sizeMatrix);
	double * h_C = (double *)malloc(sizeMatrix);
	double * h_C_seq = (double *)malloc(sizeMatrix);
	for (int i = 0; i<N*N; i++) {
		h_C_seq[i] = 0;
	}

	using namespace std::chrono;
	high_resolution_clock::time_point t1 = high_resolution_clock::now();
	matrixMultCPU(h_A, h_B, h_C_seq, N);
	high_resolution_clock::time_point t2 = high_resolution_clock::now();
	duration<double, std::milli> time_span = t2 - t1;
	double cpu_time = time_span.count();
	printf("The time: %f milliseconds\n", cpu_time);
	
	double *d_A;
	cudaMalloc((void **)&d_A, sizeMatrix);
	double *d_B;
	cudaMalloc((void **)&d_B, sizeMatrix);
	double * d_C;
	cudaMalloc((void **)&d_C, sizeMatrix);

	cudaMemcpy(d_A, h_A, sizeMatrix, cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, h_B, sizeMatrix, cudaMemcpyHostToDevice);

   	dim3 threadsPerBlock = dim3(BLOCK_SIZE, BLOCK_SIZE);
	dim3 blocksPerGrid = dim3(N / BLOCK_SIZE, N / BLOCK_SIZE);
    
	if (flag_s) {
	    cudaEventRecord(start, 0);
	    matrixMultShared<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
	    cudaEventRecord(stop, 0);
	    cudaEventSynchronize(stop);
	}
	else {
	    cudaEventRecord(start, 0);
	    matrixMult<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
	    cudaEventRecord(stop, 0);
	    cudaEventSynchronize(stop);
	}
	float KernelTime;
	cudaEventElapsedTime(&KernelTime, start, stop);
	printf("KernelTime: %f milliseconds\n", KernelTime);

	double S = cpu_time / KernelTime;

	printf("Acceleration: %f\n", S);

	cudaMemcpy(h_C, d_C, sizeMatrix, cudaMemcpyDeviceToHost);


	if (checkMult(h_C, h_C_seq, N))
		printf("The multiplication results are correct.\n");
	else
		printf("Multiplication results are NOT correct.\n");

	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);
	free(h_A);
	free(h_B);
	free(h_C);
	free(h_C_seq);
	return 0;
}
