#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <iostream>
#include "abft.cu"
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
			 double * col_chk, int col_chk_ld){
  double one = 1;
  double zero = 0;
  for (int i = 0; i < m; i += nb) {
    cublasDgemm(handle,
		CUBLAS_OP_N, CUBLAS_OP_N,
		2, n, nb,
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
		m, 2, nb,
		&one, 
		A + i * lda, lda,
		vd, vd_ld,
		&zero, 
		row_chk + (i / nb) * 2 * row_chk_ld, row_chk_ld);
  }

}



void dgemmFT( cublasHandle_t handle,
	      cublasOperation_t transa, cublasOperation_t transb,
	      int m, int n, int k, 
	      double * alpha, 
	      double * A, int lda,
	      double * B, int ldb, 
	      double * beta, 
	      double * C, int ldc,
	      int nb){
  
  // nb is the matrix block size

  double * col_chkA;
  int col_chkA_ld; 
  
  double * row_chkB;
  int row_chkB_ld; 
    
  double * col_chkC;
  int col_chkC_ld;  
  
  double * row_chkC;
  int row_chkC_ld;
     

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
	      C, ldc );
  
  
  // Update column checksum of C
  // if transa == N (No transpose), then we use: col_chk(A) * B to update col_chk(C)
  // id transa == T (Transpose), then we use: row_chk(A) * B to update col_chk(C)
  if (transa == CUBLAS_OP_N) {
    size_t col_chkA_pitch;
    cudaMallocPitch((void**)&col_chkA, &col_chkA_pitch, (m / nb) * 2, k);
    col_chkA_ld = col_chkA_pitch / sizeof(double);

    col_checksum_kernel(handle, m, k, nb,
			A, lda,
			vd, vd_ld,
			col_chkA, col_chkA_ld);
    
    cublasDgemm(handle,
		transa, transb,
		(m / nb) * 2, n, k,
		alpha,
		col_chkA, col_chkA_ld, B, ldb,
		beta,
		col_chkC, col_chkC_ld );
  } else {

    size_t col_chkA_pitch;
    cudaMallocPitch((void**)&col_chkA, &col_chkA_pitch, m, (k / nb) * 2);
    col_chkA_ld = col_chkA_pitch / sizeof(double);

    row_checksum_kernel(handle, m, k, nb,
                        A, lda,
                        vd, vd_ld,
                        col_chkA, col_chkA_ld);
    cublasDgemm(handle,
		transa, transb,
		(m / nb) * 2, n, k,
		alpha,
		col_chkA, col_chkA_ld, B, ldb,
		beta,
		col_chkC, col_chkC_ld );
  }
  


  // Update row checksum of C
  // if transa == N (No transpose), then we use: A * row_chk(B) to update row_chk(C)
  // id transa == T (Transpose), then we use: A * col_chk(B) to update row_chk(C)     
  if (transb == CUBLAS_OP_N) {
    size_t row_chkB_pitch;
    cudaMallocPitch((void**)&row_chkB, &row_chkB_pitch, k, (n / nb) * 2);
    row_chkB_ld = row_chkB_pitch / sizeof(double);

    row_checksum_kernel(handle, k, n, nb,
                        B, ldb,
                        vd, vd_ld,
                        row_chkB, row_chkB_ld);
    cublasDgemm(handle,
		transa, transb,
		m , (n / nb) * 2, k,
		alpha,
		A, lda,
		row_chkB, row_chkB_ld,
		beta,
		row_chkC, row_chkC_ld );
  } else {

    size_t row_chkB_pitch;
    cudaMallocPitch((void**)&row_chkB, &row_chkB_pitch, (k / nb) * 2, n);
    row_chkB_ld = row_chkB_pitch / sizeof(double);

    col_checksum_kernel(handle, k, n, nb,
                        B, ldb,
                        vd, vd_ld,
                        row_chkB, row_chkB_ld);
    cublasDgemm(handle,
		transa, transb,
		m, (n / nb) * 2, k,
		alpha,
		A, lda,
		row_chkB, row_chkB_ld,
		beta,
		row_chkC, row_chkC_ld );
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
