/*
 * magmamath.c
 *
 *  Created on: Nov 9, 2015
 *      Author: nfo
 */

#include "magmamath.h"
#include <magma.h>
#include <stdlib.h>

void magma_calculate_correction(
		const mat& Ho,
		const mat& ro,
		const mat& P,
		const double imageNoiseVar,
		mat& dstates,
		mat& Pnew
		)
{

	//------GPU substitute for calc_rn_R() func in fastmath.cpp------

	magma_init ();// initialize Magma
	const magma_int_t m = Ho.n_rows, n = Ho.n_cols ,  o=ro.n_rows, p=ro.n_cols;
	magma_int_t ldwork, info, min_mn,dTsize,nb; //,ldda,lddc
	dstates.zeros(n,p);
	Pnew.zeros(n,n);
	double ones = 1;
	double *tau, *tau2 ; // scalars defiing the elementary reflectors
	min_mn = std::min (m, n);
	double *h_Ho, *h_ro, *h_rn, *wa, *h_R, *h_P, *h_Rn, *h_M, *h_dstates, *h_Pnew; //h_Ho, h_ro - mxn matrices on the host
	int *ipiv;
	magmaDouble_ptr d_Ho, d_ro, d_R, d_P, d_Rn, d_M, d_dstates, d_work, d_T, d_Ho2;//d_rn; // d_Ho mxn matrix on the device
//	magma_int_t lddHo=magma_roundup(m,32), lddro=magma_roundup(o,32), lddR=magma_roundup(m,32), lddP=magma_roundup(n,32), lddRn=magma_roundup(m,32), lddM=magma_roundup(n,32);
	magma_int_t lddHo=m, lddro=o, lddR=m, lddP=n, lddRn=m, lddM=n;
	nb=magma_get_dgeqrf_nb(m);
	dTsize=nb*(2*min_mn+magma_roundup(n,32));

	magma_imalloc_pinned(&ipiv,m);
	//Allocate memory for tau. h_Ho, h_ro (on Host) and d_Ho on device.
	magma_dmalloc_pinned(&tau ,m ); // host memory for tau
	magma_dmalloc_pinned(&tau2 ,m ); // host memory for tau
	magma_dmalloc_pinned(&h_Ho,m*n ); // host memory for a
	magma_dmalloc_pinned(&h_ro,o*p ); // host memory for r
	magma_dmalloc_pinned(&h_rn,m*p);
	magma_dmalloc_pinned(&wa , o*o); // device memory for d_a
	magma_dmalloc_pinned(&h_R,m*n);
	magma_dmalloc_pinned(&h_P,n*n);
	magma_dmalloc_pinned(&h_Rn,m*m);
	magma_dmalloc_pinned(&h_M,n*n);
	magma_dmalloc_pinned(&h_Pnew,n*n);

	magma_dmalloc(&d_Ho , m*n); // device memory for d_a
	magma_dmalloc(&d_Ho2 , m*n); // device memory for d_a
	magma_dmalloc(&d_ro , o*p); // device memory for d_a
	magma_dmalloc(&d_R , m*n); // device memory for d_a
	magma_dmalloc(&d_P , n*n); // device memory for d_a
	magma_dmalloc(&d_Rn , m*m); // device memory for d_a
	magma_dmalloc(&d_M , n*n); // device memory for d_a
	magma_dmalloc(&d_T, dTsize);
//	magma_dmalloc(&d_Q, m*m);
//	magma_dmalloc(&d_rn , o*p); // device memory for d_a

/* Initialize  allocated memory on CPU to the armadillo matrices h_A and h_B*/

	//1. h_Ho
     	for (int j=0;j<n;j++){
     	for (int i=0;i<m;i++){
        	h_Ho[m*j+i] = Ho.at(m*j+i);
      	}
     	}

	//2. h_ro
     	for (int j=0;j<p;j++){
     	for (int i=0;i<o;i++){
        	h_ro[o*j+i] = ro.at(o*j+i);
     	}
     	}

	//3. h_Rn
	for(int j=0;j<m;j++){
		for(int i=0;i<m;i++){
			if(i==j) h_Rn[m*j+i]=imageNoiseVar;
			else h_Rn[m*j+i]=0;
		}
	}


	//4. h_P
	for (int j=0;j<n;j++){
     	for (int i=0;i<n;i++){
        	h_P[n*j+i] = P.at(n*j+i);
     	}
     	}
	//5. h_M
	for(int j=0;j<n;j++){
		for(int i=0;i<n;i++){
			if(i==j)h_M[n*j+i]=ones;
			else h_M[n*j+i]=0;
		}
	}
	//6. h_R
	for(int i=0; i<n*m; i++) h_R[i]=0;

	//MAGMA
	// Get size for workspace
	cout<<"starting magma qr"<<endl;
	magma_dsetmatrix ( m, n, h_Ho, m, d_Ho , lddHo); // copy h_Ho -> d_Ho --last m is ldda
	magma_dsetmatrix ( o, p, h_ro, o, d_ro , lddro ); // copy h_Ho -> d_Ho --last m is ldda
	magma_dsetmatrix ( m, m, h_Rn, m, d_Rn , lddRn); // copy h_Ho -> d_Ho --last m is ldda
	magma_dsetmatrix ( n, n, h_P, n, d_P , lddP ); // copy h_Ho -> d_Ho --last m is ldda
	magma_dsetmatrix ( n, n, h_M, n, d_M , lddM ); // copy h_Ho -> d_Ho --last m is ldda
	magma_dsetmatrix ( m, n, h_R, m, d_R , lddR ); // copy h_Ho -> d_Ho --last m is ldda
	magmablas_dlacpy(MagmaFull,m,n,d_Ho,lddHo,d_Ho2,lddHo);
	// compute a QR factorization of a real mxn matrix d_Ho. d_Ho =Q*R, Q - orthogonal , R - upper triangular
	magma_dgeqrf2_gpu( m, n, d_Ho, lddHo, tau, &info); //thirdlast m is ldda
//	magma_dgeqrf_gpu(m,n,d_Ho,lddHo,tau,d_T,&info);

	//Calculate R
	//R is the upper triangular part of d_Ho
	magmablas_dlacpy(MagmaUpper,m,n,d_Ho,lddHo,d_R,lddR);
	double *hwork;
	magma_int_t lwork;
	lwork=nb*(2*n+nb);
	magma_dmalloc_pinned(&hwork,lwork);

	magma_dgeqrf_gpu(m,n,d_Ho2,lddHo,tau2,d_T,&info);
	magma_dormqr_gpu(MagmaLeft,MagmaTrans,o,p,o,d_Ho2,lddHo,tau2,d_ro,lddro,hwork,-1,d_T,nb,&info);
	lwork=(magma_int_t)MAGMA_D_REAL(hwork[0]);
	magma_dormqr_gpu(MagmaLeft,MagmaTrans,o,p,o,d_Ho2,lddHo,tau2,d_ro,lddro,hwork,lwork,d_T,nb,&info);
//	magma_dormqr2_gpu(MagmaLeft,MagmaTrans,o,p,o,d_Ho2,lddHo,tau2,d_ro,lddro,hwork,lwork,&info);

	//Method 2: Find Q
	//magmablas_dlacpy(MagmaLower,m,n,d_Ho,lddHo,d_Q,m);
	//magma_dorgqr_gpu(m,m,min_mn,d_Q,m,tau,d_T,min_mn*magma_get_dgeqrf_nb(m),&info);
	//magma_dgemm(MagmaTrans,MagmaNoTrans,o,p,o,1,d_Ho,lddHo,d_ro,lddro,0,d_rn,o);


	//Q is derived from (1-tau_1*v_1*v_1.t())*(1-tau_2*v_2*v_2.t())*...*(1-tau_min_nm*v_mi_mn*v_min_mn.t())
	//and where v_1 .... v_k-1 is 0, v_k is 1, v_k/tau
	//+1 ... m is in the lower traingular matrix of d_Ho
	//magma_dprint_gpu(o,p,d_ro,o);
	//magma_dprint_gpu(m,n,d_Ho,m);

	//Calculte rn = Q.t()*d_ro (on GPU)
//	magma_dsytrd_gpu(MagmaLower,o,d_ho,lddHo,d,e,tau,A,);
//	magma_dormqr2_gpu(MagmaLeft,MagmaTrans,o,p,o,d_Ho,lddHo,tau,d_ro,lddro,wa,o,&info);
//	magma_dormqr_gpu_2stages(MagmaLeft,MagmaTrans,o,p,o,d_Ho,lddHo,d_ro,lddro,d_T,magma_get_dgeqrf_nb(m),&info);
	//double *hwork;
	//magma_dmalloc_pinned(&hwork,1);
	//magma_dormqr_gpu(MagmaLeft,MagmaTrans,o,p,o,d_Ho,lddHo,tau,d_ro,lddro,hwork,-1,d_T,m*magma_get_dgeqrf_nb(m),&info);
	//rn in d_ro
	//So, d_R is the R matrix
	//d_ro is Q.t()*ro or d_ro is rn
	//magma_dprint_gpu(o,p,d_ro,o);

	magma_free_pinned(tau); // free host memory
	magma_free_pinned(tau2); // free host memory
	magma_free_pinned(h_Ho); // free host memory
	magma_free_pinned(h_ro); // free host memory
	magma_free_pinned(h_rn); // free host memory
	magma_free_pinned(h_R); // free host memory
	magma_free_pinned(wa); // free host memory
	magma_free_pinned(h_P); // free host memory
	magma_free_pinned(h_M); // free host memory
	magma_free(d_Ho); // free device memory
	magma_free(d_Ho2); // free device memory
	magma_free(d_T); // free device memory
	//magma_free(d_ro); // free device memory

	magmaDouble_ptr d_temp1,d_temp2,d_temp3,d_temp4,d_temp5,d_temp6,d_temp7;
	magma_dmalloc(&d_temp1,n*m);
	magma_dmalloc(&d_temp2,m*m);
	magma_dmalloc(&d_temp4,n*m);
	magma_dmalloc(&d_temp3,n*n);
	magma_dmalloc(&d_temp5,n*n);
	magma_dmalloc(&d_temp6,m*n);
	magma_dmalloc(&d_temp7,n*n);

	magma_dgemm(MagmaNoTrans,MagmaTrans,n,m,n,1,d_P,lddP,d_R,lddR,0,d_temp1,n);
	magma_dgemm(MagmaNoTrans,MagmaNoTrans,m,m,n,1,d_R,lddR,d_temp1,n,0,d_temp2,m);
//	magma_dprint_gpu(m,n,d_R,m);
//	magma_dprint_gpu(n,m,d_temp1,n);
//	magma_dprint_gpu(m,m,d_temp2,m);
	magmablas_dgeadd(m,m,1,d_Rn,lddRn,d_temp2,m);

	//inverse
	ldwork=m*magma_get_dgetri_nb(m);
	magma_dmalloc(&d_work, ldwork);
	//magma_dprint_gpu(m,m,d_temp2,m);

	magma_dgetrf_gpu(m,m,d_temp2,m,ipiv,&info);
	//magma_dprint_gpu(m,m,d_temp2,m);

	magma_dgetri_gpu(m,d_temp2,m,ipiv,d_work,m*magma_get_dgetri_nb(m),&info);
	//magma_dprint_gpu(m,m,d_temp2,m);

	magma_dgemm(MagmaNoTrans,MagmaNoTrans,n,m,m,1,d_temp1,n,d_temp2,m,0,d_temp4,n);
	//d_temp4 is K=P*R'*inv(R*P*R'+Rn)

	//dstates in d_dstates
	magma_dmalloc_pinned(&h_dstates,n*p);
	magma_dmalloc(&d_dstates,n*p);
	magma_dgemm(MagmaNoTrans,MagmaNoTrans,n,p,m,1,d_temp4,n,d_ro,lddro,0,d_dstates,n);
	magma_dgetmatrix ( n, p, d_dstates, n, h_dstates , n );

	//output in arma::mat dstates
	for (int j=0;j<p;j++){
     	for (int i=0;i<n;i++){
        	dstates.at(n*j+i)=h_dstates[n*j+i];
     	}
     	}

	//M
	magma_dgemm(MagmaNoTrans,MagmaNoTrans,n,n,m,-1,d_temp4,n,d_R,lddR,1,d_M,lddM);

	//Pnew
	magma_dgemm(MagmaNoTrans,MagmaTrans,n,n,n,1,d_P,lddP,d_M,lddM,0,d_temp3,n);
	magma_dgemm(MagmaNoTrans,MagmaNoTrans,n,n,n,1,d_M,lddM,d_temp3,n,0,d_temp5,n);

	magma_dgemm(MagmaNoTrans,MagmaTrans,m,n,m,1,d_Rn,lddRn,d_temp4,n,0,d_temp6,m);
	magma_dgemm(MagmaNoTrans,MagmaNoTrans,n,n,m,1,d_temp4,n,d_temp6,m,0,d_temp7,n);

	magmablas_dgeadd(n,n,1,d_temp7,n,d_temp5,n);
	magma_dgetmatrix ( n, n, d_temp5, n, h_Pnew , n );

	//output in arma::mat Pnew
	for (int j=0;j<n;j++){
     	for (int i=0;i<n;i++){
        	Pnew.at(n*j+i)=h_Pnew[n*j+i];
     	}
     	}




	magma_free(d_ro);
	magma_free(d_R);
	magma_free(d_Rn);
	magma_free(d_P);
	magma_free(d_temp1);
	magma_free(d_temp2);
	magma_free(d_temp3);
	magma_free(d_temp4);
	magma_free(d_temp5);
	magma_free(d_temp7);
	magma_free(d_work); // free device memory
	magma_free(d_dstates); // free device memory
	magma_free_pinned(h_dstates); // free device memory
	magma_free_pinned(h_Pnew); // free device memory
	magma_free_pinned(h_Rn); // free device memory
	magma_free_pinned(ipiv); // free device memory


	magma_finalize(); // finalize Magma

	//Convert to dstates and Pnew
}
