// Low level matrix multiplication on GPU using CUDA with CURAND and CUBLAS
// C(m,n) = A(m,k) * B(k,n)
#include <iostream>
#include <cstdlib>
#include <ctime>
//#include <sys/time.h>      
#include <cublas_v2.h>
#include <curand.h>
#include <time.h>
#include <fstream>
#include <cuda_runtime.h>
#include <stdlib.h>
#include <stdio.h>

using namespace std;




__global__ void
col_detect_correct_kernel(double * A, int lda, int B, double E,
	double * checksum_update, int checksum_update_ld,
	double * hrz_recal_chk, int hrz_recal_chk_ld)
{
	//determin the block to process
	A = A + blockIdx.x * B + blockIdx.y * B * lda;

	checksum_update = checksum_update + blockIdx.x * 2 + blockIdx.y * B * checksum_update_ld;
	hrz_recal_chk = hrz_recal_chk + blockIdx.x * 2 + blockIdx.y * B * hrz_recal_chk_ld;

	//determine the specific colum to process
	A = A + threadIdx.x * lda;
	checksum_update = checksum_update + threadIdx.x * checksum_update_ld;
	hrz_recal_chk = hrz_recal_chk + threadIdx.x * hrz_recal_chk_ld;

	double d1 = (*checksum_update) - (*hrz_recal_chk);
	double d2 = (*(checksum_update + 1)) - (*hrz_recal_chk + 1);

	//error detected
	if (fabs(d1) > E) {
		//locate the error
		int loc = round(d2 - d1) - 1;
		printf("[col check]error detected:%f---%d \n", d1, loc);

		//the sum of the rest correct number except the error one
		double sum = 0.0;
		for (int i = 0; i < B; i++) {
			if (i != loc) {
				sum += *(A + i);
			}
		}
		//correct the error
		*(A + loc) = *checksum_update - sum;
	}
}


__global__ void
row_detect_correct_kernel(double * A, int lda, int B, double E,
	double * checksum_update, int checksum_update_ld,
	double * vrt_recal_chk, int vrt_recal_chk_ld)
{
	//determin the block to process
	A = A + blockIdx.x * B + blockIdx.y * B * lda;

	checksum_update = checksum_update + blockIdx.x * B + blockIdx.y * 2 * checksum_update_ld;
	vrt_recal_chk = vrt_recal_chk + blockIdx.x * B + blockIdx.y * 2 * vrt_recal_chk_ld;

	//determine the specific colum to process
	A = A + threadIdx.x;
	checksum_update = checksum_update + threadIdx.x;
	vrt_recal_chk = vrt_recal_chk + threadIdx.x;

	double d1 = (*checksum_update) - (*vrt_recal_chk);
	double d2 = (*(checksum_update + checksum_update_ld)) - (*(vrt_recal_chk + vrt_recal_chk_ld));

	//error detected
	if (fabs(d1) > E) {
		//locate the error
		int loc = round(d2 - d1) - 1;
		printf("[row check]error detected:%f---%d \n", d1, loc);

		//the sum of the rest correct number except the error one
		double sum = 0.0;
		for (int i = 0; i < B; i++) {
			if (i != loc) {
				sum += *(A + i * lda);
			}
		}
		//correct the error
		*(A + loc * lda) = *checksum_update - sum;
	}
}

/*
* B: block size
* m: # of row
* n: # of column
*/
    void col_detect_correct(double * A, int lda, int B, int m, int n,
	double * checksum_update, int checksum_update_ld,
	double * hrz_recal_chk, int hrz_recal_chk_ld)
{
	//printf("col_detect_correct called \n");
	//error threshold 
	double E = 1e-10;

	col_detect_correct_kernel <<<dim3(m / B, n / B), dim3(B) >>>(A, lda, B, E, //dim3(m / B, n / B) -> dim3(m / B, n / B)
		checksum_update, checksum_update_ld,
		hrz_recal_chk, hrz_recal_chk_ld);
}


/*
* B: block size
* m: # of row
* n: # of column
*/
void row_detect_correct(double * A, int lda, int B, int m, int n,
	double * checksum_update, int checksum_update_ld,
	double * vrt_recal_chk, int vrt_recal_chk_ld)
{
	//printf("row_detect_correct called \n");
	//error threshold 

	double E = 1e-10;

	row_detect_correct_kernel <<<dim3(m / B, n / B), dim3(B) >>> (A, lda, B, E,
		checksum_update, checksum_update_ld,
		vrt_recal_chk, vrt_recal_chk_ld);

}





/* return value :
* -2 : failure, too many faults
* -1 : failure, unable to recover
*  0 : pass
*  1 : failure, recovered
* */
/*void gpu_checksum(double *result, double *matrix, int matrixSize) {

	cublasStatus_t ret;
	cublasHandle_t handle;
	ret = cublasCreate(&handle);

	ret = cublasSasum(handle, matrixSize, matrix, 1, result);

	cublasDestroy(handle);

}
*/
using namespace std;
//dgemm with ABFT
//
void col_detect_correct(double * A, int lda, int B, int m, int n,
	double * checksum_update, int checksum_update_ld,
	double * hrz_recal_chk, int hrz_recal_chk_ld);


void row_detect_correct(double * A, int lda, int B, int m, int n,
	double * checksum_update, int checksum_update_ld,
	double * vrt_recal_chk, int vrt_recal_chk_ld);

void col_checksum_kernel(cublasHandle_t handle,
	int m, int n, int nb,
	double * A, int lda,
	double * vd, int vd_ld,
	double * col_chk, int col_chk_ld) {
	double one = 1;
	double zero = 0;
	for (int i = 0; i < m; i += nb) {
		cublasDgemm(handle,
			CUBLAS_OP_N, CUBLAS_OP_N,
			m, n, nb,// 2->m & nb -> 
			&one,
			vd, vd_ld,
			A + i, lda,
			&zero,
			col_chk + (i / nb) * 2, col_chk_ld);
	}

}



void row_checksum_kernel(cublasHandle_t handle,
	int m, int n, int nb,
	double * A, int lda,
	double * vd, int vd_ld,
	double * row_chk, int row_chk_ld) {

	double one = 1;
	double zero = 0;
	for (int i = 0; i < n; i += nb) {
		cublasDgemm(handle,
			CUBLAS_OP_N, CUBLAS_OP_T,
			m, n, nb, // 2->n 
			&one,
			A + i * lda, lda,
			vd, vd_ld,
			&zero,
			row_chk + (i / nb) * 2 * row_chk_ld, row_chk_ld);
	}

}



void dgemmFT(cublasHandle_t handle,
	cublasOperation_t transa, cublasOperation_t transb,
	int m, int n, int k,
	double * alpha,
	double * A, int lda,
	double * B, int ldb,
	double * beta,
	double * C, int ldc, double * h_C,
	int nb) {

	// nb is the matrix block size

	double * col_chkA;
	int col_chkA_ld;
	
	
	double * row_chkB;
	int row_chkB_ld;

	double * col_chkC;
	int col_chkC_ld;// ? initialization extra

	double * row_chkC;
	int row_chkC_ld;// ? initialization extra


	// Checksum weight vectors
	// [1,1,...,1]
	// [1,2,...,nb]
	double * v = new double[2 * nb];
	int v_ld = 2;
	for (int i = 0; i < nb; ++i) {
		*(v + i * v_ld) = 1;
	}
	for (int i = 0; i < nb; ++i) {
		*(v + i * v_ld + 1) = 1 + i;
	}

	// Copy checksum weight vectors to GPU
	// pitch memory is used for efficiency
	double * vd;
	int vd_ld;
	size_t vd_pitch;
	cudaMallocPitch((void**)&vd, &vd_pitch, 2, nb);
	vd_ld = vd_pitch / sizeof(double);


	// Original Computation
	cublasDgemm(handle,
		transa, transb,
		m, n, k,
		alpha,
		A, lda, B, ldb,
		beta,
		C, ldc);

		cudaMemcpy(h_C, C, ldc*ldc * sizeof(double), cudaMemcpyDeviceToHost);
	//	print_matrix(h_C, m , n );

	// Update column checksum of C
	// if transa == N (No transpose), then we use: col_chk(A) * B to update col_chk(C)
	// id transa == T (Transpose), then we use: row_chk(A) * B to update col_chk(C)
	if (transa == CUBLAS_OP_N) {
		size_t col_chkA_pitch;
		size_t col_chkC_pitch;
		cudaMallocPitch((void**)&col_chkA, &col_chkA_pitch, (m / nb) * 2, k);
		cudaMallocPitch((void**)&col_chkC, &col_chkC_pitch, (m / nb) * 2, n);

		col_chkA_ld = col_chkA_pitch / sizeof(double); //modified: added initialisation for col_chkC
		col_chkC_ld = col_chkC_pitch / sizeof(double); //modified: added initialisation for col_chkC

		col_checksum_kernel(handle, m, k, nb,
			A, lda,
			vd, vd_ld,
			col_chkA, col_chkA_ld);

		cublasDgemm(handle,
			transa, transb,
			(m / nb) * m, n, k, //2->m
			alpha,
			col_chkA, col_chkA_ld, B, ldb,
			beta,
			col_chkC, col_chkA_ld);
	}
	else {

		size_t col_chkA_pitch;
		cudaMallocPitch((void**)&col_chkA, &col_chkA_pitch, m, (k / nb) * 2);
		col_chkA_ld = col_chkA_pitch / sizeof(double);

		row_checksum_kernel(handle, m, k, nb,
			A, lda,
			vd, vd_ld,
			col_chkA, col_chkA_ld);
		cublasDgemm(handle,
			transa, transb,
			(m / nb) * nb, n, k,//2->nb
			alpha,
			col_chkA, col_chkA_ld, B, ldb,
			beta,
			col_chkC, col_chkC_ld);
	}



	// Update row checksum of C
	// if transa == N (No transpose), then we use: A * row_chk(B) to update row_chk(C)
	// id transa == T (Transpose), then we use: A * col_chk(B) to update row_chk(C)     
	if (transb == CUBLAS_OP_N) {
		size_t row_chkB_pitch;
		cudaMallocPitch((void**)&row_chkB, &row_chkB_pitch, k, (n / nb) * 2);
		row_chkB_ld = row_chkB_pitch / sizeof(double);

		size_t row_chkC_pitch;
		cudaMallocPitch((void**)&row_chkC, &row_chkC_pitch, m, (n / nb) * 2);
		row_chkC_ld = row_chkC_pitch / sizeof(double);

		row_checksum_kernel(handle, k, n, nb,
			B, ldb,
			vd, vd_ld,
			row_chkB, row_chkB_ld);
		cublasDgemm(handle,
			transa, transb,
			m, (n / nb) * nb, k,// 2->nb
			alpha,
			A, lda,
			row_chkB, row_chkB_ld,
			beta,
			row_chkC, row_chkC_ld);
	}
	else {

		size_t row_chkB_pitch;
		cudaMallocPitch((void**)&row_chkB, &row_chkB_pitch, (k / nb) * 2, n);
		row_chkC_ld = row_chkB_ld = row_chkB_pitch / sizeof(double); //modified:added chkC_ld

		col_checksum_kernel(handle, k, n, nb,
			B, ldb,
			vd, vd_ld,
			row_chkB, row_chkB_ld);

		cublasDgemm(handle,
			transa, transb,
			m, (n / nb) * nb, k,//2->nb
			alpha,
			A, lda,
			row_chkB, row_chkB_ld,
			beta,
			row_chkC, row_chkC_ld);
	}



	double * recal_col_chk;
	int recal_col_chk_ld;
	size_t recal_col_chk_pitch;
	cudaMallocPitch((void**)&recal_col_chk, &recal_col_chk_pitch, (m / nb) * 2, n);
	recal_col_chk_ld = recal_col_chk_pitch / sizeof(double);


	// Recalculate col_chk(C) and then check for error
	col_checksum_kernel(handle, m, n, nb,
		C, ldc,
		vd, vd_ld,
		recal_col_chk, recal_col_chk_ld);

	col_detect_correct(C, ldc, nb, m, n,
		col_chkC, col_chkC_ld,
		recal_col_chk, recal_col_chk_ld);




	double * recal_row_chk;
	int recal_row_chk_ld;
	size_t recal_row_chk_pitch;
	cudaMallocPitch((void**)&recal_row_chk, &recal_row_chk_pitch, m, (n / nb) * 2);
	recal_row_chk_ld = recal_row_chk_pitch / sizeof(double);


	// Recalculate row_chk(C) and then check for error
	row_checksum_kernel(handle, m, n, nb,
		C, ldc,
		vd, vd_ld,
		recal_row_chk, recal_row_chk_ld);


	row_detect_correct(C, ldc, nb, m, n,
		row_chkC, row_chkC_ld,
		recal_row_chk, recal_row_chk_ld);


}


bool verify(double a, double b) {
	bool test_result = 0;
	const double relativeTolerance = 1e-1;

	double relativeError = ((a - b) / a);
	if (relativeError > relativeTolerance
		|| relativeError < -relativeTolerance) {
		test_result = 0;
		std::cout << "TEST FAILED" << std::endl;

		exit(0);
	}
	test_result = 1;
	std::cout << "TEST PASSED" << std::endl;
	return test_result;

}



double build_checksum(double *cs_A_rows, double *A, double *cs_B_cols, double *B, int nr_rows_A, int nr_cols_A, int nr_rows_B, int nr_cols_B)//is not working correctly
{
	double checksum_sum = 0;
	for (int i = 0; i < nr_rows_A; ++i) {
		for (int j = 0; j < nr_cols_A; ++j)
		{
			cs_A_rows[i] += A[i * nr_cols_A + j];
		}
	}

	for (int j = 0; j < nr_cols_B; ++j) {
		for (int i = 0; i < nr_rows_B; ++i)
		{
			cs_B_cols[j] += B[i * nr_cols_B + j];
		}
	}
	for (int i = 0; i< nr_cols_A;i++) {
		checksum_sum += cs_A_rows[i] * cs_B_cols[i];
	}
	return checksum_sum;


}



void matrix_input_generator(double *h_a, int m, int n) {
	// random initialize matrix A
	for (int i = 0; i < m; ++i) {
		for (int j = 0; j < n; ++j) {
			h_a[i * n + j] = rand() % 4;
		}
	}

}
// Fill the array A(nr_rows_A, nr_cols_A) with random numbers on GPU
/*void GPU_fill_rand(double *A, int nr_rows_A, int nr_cols_A) {
	// Create a pseudo-random number generator
	curandGenerator_t prng;
	curandCreateGenerator(&prng, CURAND_RNG_PSEUDO_DEFAULT);

	// Set the seed for the random number generator using the system clock
	curandSetPseudoRandomGeneratorSeed(prng, (unsigned long long) clock());

	// Fill the array with random numbers on the device
	curandGenerateUniform(prng, A, nr_rows_A * nr_cols_A);
}
*/
// Multiply the arrays A and B on GPU and save the result in C
// C(m,n) = A(m,k) * B(k,n)
/*void gpu_blas_mmul(const double *A, const double *B, double *C, const int m, const int k, const int n) {
	int lda = m, ldb = k, ldc = m;
	const double alf = 1;
	const double bet = 0;
	const double *alpha = &alf;
	const double *beta = &bet;

	// Create a handle for CUBLAS
	cublasHandle_t handle;
	cublasCreate(&handle);

	// Do the actual multiplication
	cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);

	// Destroy the handle
	cublasDestroy(handle);
}
*/
//Print matrix A(nr_rows_A, nr_cols_A) storage in column-major format
void print_matrix(const double *A, int nr_rows_A, int nr_cols_A) {

	for (int i = 0; i < nr_rows_A; ++i) {
		for (int j = 0; j < nr_cols_A; ++j) {
			std::cout << A[j * nr_rows_A + i] << " ";
		}
		std::cout << std::endl;
	}
	std::cout << std::endl;
}

int main(int argc, char ** argv) {
//added by me extra start

	//added by me extra end
	// Create a handle for CUBLAS
	cublasHandle_t handle;
	cublasCreate(&handle);


	ofstream myfile;
	myfile.open("Exe_Time_Online_offline_abft_mm.txt", ios::app);
	

	//struct timeval t1, t2;
	double elapsedTime;
	//FILE *fpo = fopen("exe_time_bfs.txt\n\n\n\n","a");

	srand(time(NULL));
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	int nb;//*nr_cols_A;
	int nr_rows_A,
		nr_cols_A, 
		nr_rows_B, 
		nr_cols_B, 
		nr_rows_C, 
		nr_cols_C;

	cout << "please insert M * N* K And Block Size" << endl;
	
	cin >> nr_rows_A;
	cin >> nr_cols_A;
	cin >> nr_cols_B;
	cin >> nb;

	nr_rows_B = nr_cols_A;
	nr_rows_C = nr_rows_A;
	nr_cols_C = nr_cols_B;



	// Allocate 3 arrays on CPU

	
	//nr_rows_A= atoi(argv[1]);
	//nr_rows_A = 2;
	
	
	//lda = m, ldb = k, ldc = m;
	int lda = nr_rows_A;// *nr_cols_A;//*nr_cols_A;
	int ldb = nr_cols_B;// *nr_cols_B; // * nr_cols_B;
	int ldc = nr_rows_A;//cols_C;; //*nr_cols_B;

	double alf = 1;
	double bet = 0;
	double *alpha = &alf;
	double *beta = &bet;

	//cublasOperation_t transa='n';

	// for simplicity we are going to use square arrays
	//nr_rows_A = nr_cols_A = nr_rows_B = nr_cols_B = nr_rows_C = nr_cols_C = 10000;

	double *h_A = (double *)malloc(nr_rows_A * nr_cols_A * sizeof(double));
	for (unsigned int i = 0; i < nr_rows_A * nr_cols_A; i++) { h_A[i] = 0; }

	double *h_B = (double *)malloc(nr_rows_B * nr_cols_B * sizeof(double));
	for (unsigned int i = 0; i < nr_rows_B * nr_cols_B; i++) { h_B[i] = 0; }

	double *h_C = (double *)malloc(nr_rows_C * nr_cols_C * sizeof(double));
	for (unsigned int i = 0; i < (nr_rows_C) * (nr_cols_C); i++) { h_C[i] = 0; }

	double *A_row_checksum = (double *)malloc(nr_cols_A * 1 * sizeof(double));
	for (unsigned int i = 0; i < nr_cols_C; i++) { A_row_checksum[i] = 0; }



	double *B_col_checksum = (double *)malloc(nr_rows_B * sizeof(double));
	for (unsigned int i = 0; i < nr_cols_B; i++) { B_col_checksum[i] = 0; }


	// Allocate 3 arrays on GPU
	double *d_A, *d_B, *d_C;

	cudaMalloc(&d_A, nr_rows_A * nr_cols_A * sizeof(double));
	cudaMalloc(&d_B, nr_rows_B * nr_cols_B * sizeof(double));
	cudaMalloc(&d_C, nr_rows_C * nr_cols_C * sizeof(double));


	// If you already have useful values in A and B you can copy them in GPU:


	// Fill the arrays A and B on GPU with random numbers

	//GPU_fill_rand(d_A, nr_rows_A, nr_cols_A);
	//GPU_fill_rand(d_B, nr_rows_B, nr_cols_B);

	matrix_input_generator(h_A, nr_rows_A, nr_cols_A);
	matrix_input_generator(h_B, nr_rows_B, nr_cols_B);


	//print_matrix(h_A, nr_rows_A, nr_cols_A);
	
//	print_matrix(h_B, nr_rows_B, nr_cols_B);

	//print_matrix(h_C, nr_rows_C, nr_cols_C);

	//	double matrix_checksum = 0;

	

	cudaMemcpy(d_A, h_A, nr_rows_A * nr_cols_A * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, h_B, nr_rows_B * nr_cols_B * sizeof(double), cudaMemcpyHostToDevice);

	clock_t start2 = clock();
	// Execuatable code
	
	dgemmFT(handle,
		CUBLAS_OP_N, CUBLAS_OP_N,
		nr_rows_A, nr_cols_A, nr_cols_B,
		alpha, d_A, lda, d_B, ldb,
		beta,
		d_C, ldc, h_C
		, nb);

	clock_t stop2 = clock();

	double elapsed = (double)(stop2 - start2) * 1000.0 / CLOCKS_PER_SEC;
	printf("Time elapsed in ms: %f", elapsed);


//	print_matrix(h_C, nr_rows_C, nr_cols_C);

	
	//gettimeofday(&t1, NULL);

	//matrix_checksum = build_checksum(A_row_checksum, h_A, B_col_checksum, h_B, nr_rows_A, nr_cols_A, nr_rows_B, nr_cols_B);

	//gettimeofday(&t2, NULL);


	//elapsedTime = (t2.tv_sec - t1.tv_sec) * 1000.0;      // sec to ms
	//elapsedTime += (t2.tv_usec - t1.tv_usec) / 1000.0;   // us to ms



														 //	fclose(fpo);


					/*									 
														 std::cout << "A in cpu side=" << std::endl;
														 print_matrix(h_A, nr_rows_A, nr_cols_A);

														 std::cout << "Row checksum of Matrix A=" << std::endl;//for test
														 print_matrix(A_row_checksum, nr_rows_A, 1);//for test

														 std::cout << "B in cpu side=" << std::endl;
														 print_matrix(h_B, nr_rows_B, nr_cols_B);

														 std::cout << "Col checksum of Matrix B=" << std::endl;//for test
														 print_matrix(B_col_checksum, 1, nr_cols_B);//for test
														 
//	std::cout << "A and B checksum summation=" << matrix_checksum << std::endl;


	// Multiply A and B on GPU
	double milliseconds = 0;
	double milliseconds2 = 0;
	double cb_result = 0;
	int counter = 0;
	for (int i = 0;i<1;i++) {
		cudaEventRecord(start);

		gpu_blas_mmul(d_A, d_B, d_C, nr_rows_A, nr_cols_A, nr_cols_B);
		cudaEventRecord(stop);
		cudaEventSynchronize(stop);

		cudaEventElapsedTime(&milliseconds, start, stop);



		cudaEventRecord(start);
		gpu_checksum(&cb_result, d_C, (nr_rows_C) * (nr_cols_C));
		cudaEventRecord(stop);
		cudaEventSynchronize(stop);

		counter++;



		cudaEventElapsedTime(&milliseconds2, start, stop);
		std::cout << "Multiplication Execution Time without checksum = " << milliseconds << " " << "ms" << std::endl;
	}
	std::cout << "The counter is =" << counter << std::endl;
	std::cout << "the result matrix sum in GPU is =" << cb_result << std::endl;

	verify(matrix_checksum, cb_result);

	myfile << "Execution Time for Matrix M * N * K = " << nr_rows_A << " * " << nr_cols_A << " * " << nr_cols_B << " * " << std::endl << std::endl;
//	myfile << "Multiplication Execution Time without checksum = " << milliseconds << " " << "ms" << std::endl;
//	myfile << "Checksum calculation in CPU                    = " << elapsedTime << " " << "ms" << std::endl;
//	myfile << "Checksum Calculation Execution Time in GPU     = " << milliseconds2 << " " << "ms" << std::endl;
//	myfile << "Total Kernel Execution Time with checksum      = " << milliseconds + milliseconds2 << " " << "ms" << std::endl;

//	std::cout << "Multiplication Execution Time without checksum = " << milliseconds << " " << "ms" << std::endl;
//	std::cout << "Checksum calculation in CPU                    = " << elapsedTime << " " << "ms" << std::endl;
//	std::cout << "Checksum Calculation Execution Time in GPU     = " << milliseconds2 << " " << "ms" << std::endl;
//	std::cout << "Total Kernel Execution Time with checksum      = " << milliseconds + milliseconds2 << " " << "ms" << std::endl << std::endl << std::endl;

	// Copy (and print) the result on host memory

	cudaMemcpy(h_C, d_C, nr_rows_C * nr_cols_C * sizeof(double), cudaMemcpyDeviceToHost);
	//std::cout << "C =" << std::endl;
	//print_matrix(h_C, nr_rows_C, nr_cols_C);



	// added_end
	//	 std::cout << "MMUL_1 Execution completed. Elapsed Time = " << milliseconds << std::endl;
	//std::cout << "MMUL_1 Done" << std::endl;
	//Free GPU memory
	*/
	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);
	//cudaFree(d_c_A);
	//cudaFree(d_c_B);
	//cudaFree(d_c_C);	
	// Free CPU memory
	
	free(h_A);
	free(h_B);
	free(h_C);
	free(A_row_checksum);
	//free(A_col_checksum);
	//free(B_row_checksum);
	free(B_col_checksum);
	//free(C_row_checksum);
	//free(C_col_checksum);
	myfile.close();
	
	system("pause");
	return 0;
	
}
