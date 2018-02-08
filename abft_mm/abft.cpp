/*
    Enhanced Online ABFT
    UC Riverside
    Jieyang Chen
*/
#include <stdlib.h>
#include <stdio.h>

__global__ void
col_detect_correct_kernel(double * A, int lda, int B, double E,
				double * checksum_update, int checksum_update_ld,
				double * hrz_recal_chk, int hrz_recal_chk_ld)
{
    //determin the block to process
    A = A + blockIdx.x * B + blockIdx.y * B * lda;
	
    checksum_update = checksum_update + blockIdx.x * 2  + blockIdx.y * B * checksum_update_ld;
    hrz_recal_chk = hrz_recal_chk + blockIdx.x * 2+ blockIdx.y * B * hrz_recal_chk_ld;
    
    //determine the specific colum to process
    A = A + threadIdx.x * lda;
    checksum_update = checksum_update + threadIdx.x * checksum_update_ld;
    hrz_recal_chk = hrz_recal_chk + threadIdx.x * hrz_recal_chk_ld;
	
    double d1 = (*checksum_update) - (*hrz_recal_chk);
    double d2 = (*(checksum_update + 1)) - (*hrz_recal_chk + 1);
	
    //error detected
    if(fabs(d1) > E) {
    	//locate the error
	int loc = round(d2 - d1) - 1;
	printf("[col check]error detected:%f---%d \n",d1,loc);
		
	//the sum of the rest correct number except the error one
	double sum = 0.0;
	for (int i = 0; i < B; i++) {
		if (i != loc) {
			sum +=	*(A + i); 
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
    if(fabs(d1) > E) {
	//locate the error
	int loc = round(d2 - d1) - 1;
	printf("[row check]error detected:%f---%d \n",d1,loc);
		
	//the sum of the rest correct number except the error one
	double sum = 0.0;
	for (int i = 0; i < B; i++) {
	    if (i != loc) {
		sum +=	*(A + i * lda); 
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
	
	col_detect_correct_kernel<<<dim3(m/B, n/B), dim3(B)>>>(A, lda, B, E,
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
	
	row_detect_correct_kernel<<<dim3(m/B, n/B), dim3(B)>>>(A, lda, B, E,
					checksum_update, checksum_update_ld,
					vrt_recal_chk, vrt_recal_chk_ld);
					
}




