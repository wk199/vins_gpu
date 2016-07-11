/*
 * qr.cpp
 *
 *  Created on: Sep 17, 2015
 *      Author: nfo
 */

#include "fastmath.h"
#include <cblas.h>
#include <lapacke.h>

using namespace arma;

//#if !defined(ARMA_BLAS_CAPITALS)
//#define arma_dormqr dormqr_
//#else
//#define arma_dormqr DORMQR_
//#endif
//
//extern "C" {
//	void arma_dormqr(
//			char* side,
//			char* trans,
//			blas_int* m,
//			blas_int* n,
//			blas_int* k,
//			double* a,
//			blas_int* lda,
//			double* tau,
//			double* c,
//			blas_int* ldc,
//			double* work,
//			blas_int* lwork,
//			blas_int* info);
//}

void calc_K(const mat& R, const mat& P, const vec& Rn, mat& K)
{
	mat PRt;
	mat RPRt;

	if (R.n_elem != P.n_elem) {
		// TODO: Can we utilize the fact that Rn is a diagonal matrix
		// and does it help that P is symmetric?
		// Also, see if we can make use of dtrmm even though
		// R and P are not the same length.
		PRt = P*R.t();
		RPRt = R*PRt;
		RPRt.diag() += Rn;
		K = PRt*inv_sympd(RPRt);
		return;
	}

	dtrmm(R, P, fastMathRight, fastMathUpper, fastMathTrans, PRt);
	dtrmm(R, PRt, fastMathLeft, fastMathUpper, fastMathNoTrans, RPRt);

	RPRt.diag() += Rn;
	K = PRt*inv(RPRt);  // TODO: Check if inv_sympd works faster on target platform. It does not nfos computer.
}

bool calc_rn_R_full(const mat& ro, const mat& Ho, mat& rn, mat& R)
{
	R = Ho.get_ref();

	const uword R_n_rows = R.n_rows;
	const uword R_n_cols = R.n_cols;

	blas_int m         = static_cast<blas_int>(R_n_rows);
	blas_int n         = static_cast<blas_int>(R_n_cols);
	blas_int lwork     = 0;
	blas_int lwork2     = 0;
	blas_int lwork_min = (std::max)(blas_int(1), (std::max)(m,n));
	blas_int k         = (std::min)(m,n);
	blas_int info      = 0;

	podarray<double> tau( static_cast<uword>(k) );

	double work_query[2];
	blas_int lwork_query = -1;

	// Calculate optimal work array size
	info = LAPACKE_dgeqrf_work(LAPACK_COL_MAJOR, m, n, R.memptr(), m, tau.memptr(), &work_query[0], lwork_query);

	if (info == 0) {
		const blas_int lwork_proposed = static_cast<blas_int>( access::tmp_real(work_query[0]) );

		lwork = (lwork_proposed > lwork_min) ? lwork_proposed : lwork_min;
	} else {
		return false;
	}

	podarray<double> work( static_cast<uword>(lwork) );

	// Do the QR factorization

	info = LAPACKE_dgeqrf_work(LAPACK_COL_MAJOR, m, n, R.memptr(), m, tau.memptr(), work.memptr(), lwork);

	mat Q;
	Q.set_size(R_n_rows, R_n_rows);

	arrayops::copy( Q.memptr(), R.memptr(), (std::min)(Q.n_elem, R.n_elem) );

    for(uword col = 0; col < R_n_cols; ++col)
      for(uword row = (col + 1); row < R_n_rows; ++row)
        R.at(row, col) = 0.0;

	rn = ro;

	lwork_query = -1;

	char L = 'L';
	char T = 'T';
	blas_int one = 1;
	// Calculate optimal work array size
	info = LAPACKE_dormqr_work(LAPACK_COL_MAJOR, L, T, m, one, k, Q.memptr(), m, tau.memptr(), rn.memptr(), m, &work_query[0], lwork_query);
	if (info == 0) {
		const blas_int lwork_proposed = static_cast<blas_int>( access::tmp_real(work_query[0]) );

		lwork2 = (lwork_proposed > lwork_min) ? lwork_proposed : lwork_min;
	} else {
		return false;
	}
	podarray<double> work2( static_cast<uword>(lwork) );

	info = LAPACKE_dormqr_work(LAPACK_COL_MAJOR, L, T, m, one, k, Q.memptr(), m, tau.memptr(), rn.memptr(), m, work2.memptr(), lwork2);

	return (info == 0);
}

bool calc_rn_R(const mat& ro, const mat& Ho, mat& rn, mat& R)
{

//	cout << "Ho: " << Ho.n_rows << "x" << Ho.n_cols << endl;
//
//	int numZeroes = 0;
//	for (int i = 0; i < Ho.n_rows; i++) {
//		int j;
//		if (i > Ho.n_cols - 1)
//			j = Ho.n_cols - 1;
//		else
//			j = i;
//
//		for (; j >=0 ; j--)
//			if (Ho(i, j) == 0.0)
//				numZeroes++;
//	}
//	cout << "num zeroes in lower diagonal: " << numZeroes << endl;
//	cout << "total number to zero: " << Ho.n_cols*(Ho.n_rows - Ho.n_cols) + (Ho.n_cols*Ho.n_cols / 2)<< endl;

	if (Ho.n_rows < Ho.n_cols)
		return calc_rn_R_full(ro, Ho, rn, R);

    mat Q = Ho.get_ref();

    const uword Q_n_rows = Q.n_rows;
    const uword Q_n_cols = Q.n_cols;

    if( Q_n_rows <= Q_n_cols )
		return calc_rn_R_full(ro, Ho, rn, R);

	blas_int m         = static_cast<blas_int>(Q_n_rows);
	blas_int n         = static_cast<blas_int>(Q_n_cols);
	blas_int lwork     = 0;
	blas_int lwork2     = 0;
	blas_int lwork_min = (std::max)(blas_int(1), (std::max)(m,n));
	blas_int k         = (std::min)(m,n);
	blas_int info      = 0;

	podarray<double> tau( static_cast<uword>(k) );

	double work_query[2];
	blas_int lwork_query = -1;

	// Calculate optimal work array size
	info = LAPACKE_dgeqrf_work(LAPACK_COL_MAJOR, m, n, Q.memptr(), m, tau.memptr(), &work_query[0], lwork_query);

	if (info == 0) {
		const blas_int lwork_proposed = static_cast<blas_int>( access::tmp_real(work_query[0]) );

		lwork = (lwork_proposed > lwork_min) ? lwork_proposed : lwork_min;
	} else {
		return false;
	}

	podarray<double> work( static_cast<uword>(lwork) );

	// Do the QR factorization

	info = LAPACKE_dgeqrf_work(LAPACK_COL_MAJOR, m, n, Q.memptr(), m, tau.memptr(), work.memptr(), lwork);

	R.set_size(Q_n_cols, Q_n_cols);

	for (uword col = 0; col < Q_n_cols; ++col) {
		for (uword row = 0; row <= col; ++row)
			R.at(row, col) = Q.at(row, col);

		for (uword row = (col + 1); row < Q_n_cols; ++row)
			R.at(row, col) = 0.0;
	}

	rn = ro;

	lwork_query = -1;

	char L = 'L';
	char T = 'T';
	blas_int one = 1;
	// Calculate optimal work array size
	info = LAPACKE_dormqr_work(LAPACK_COL_MAJOR, L, T, m, one, k, Q.memptr(), m, tau.memptr(), rn.memptr(), m, &work_query[0], lwork_query);
	if (info == 0) {
		const blas_int lwork_proposed = static_cast<blas_int>( access::tmp_real(work_query[0]) );

		lwork2 = (lwork_proposed > lwork_min) ? lwork_proposed : lwork_min;
	} else {
		return false;
	}
	podarray<double> work2( static_cast<uword>(lwork) );

	info = LAPACKE_dormqr_work(LAPACK_COL_MAJOR, L, T, m, one, k, Q.memptr(), m, tau.memptr(), rn.memptr(), m, work2.memptr(), lwork2);

	rn.reshape(Q_n_cols, 1);
	return (info == 0);
}

void dtrmm(const mat& m1, mat& m2, fastMathSide side, fastMathUpLo uplo, fastMathTranspose transa)
{
	int n_rows = -1;
	int n_cols = -1;
	int lda = -1;
	int ldb = -1;
	CBLAS_SIDE side_blas;
	CBLAS_UPLO uplo_blas;
	CBLAS_TRANSPOSE transa_blas;
	CBLAS_DIAG diag_blas;

	if (side == fastMathLeft) {
		n_rows = m1.n_rows;
		n_cols = m2.n_cols;
		lda = m1.n_cols;
		ldb = m2.n_rows;
		side_blas = CblasLeft;
	} else if (side == fastMathRight) {
		n_rows = m2.n_rows;
		n_cols = m1.n_cols;
		lda = n_cols;
		ldb = n_rows;
		side_blas = CblasRight;
	}

	if (transa == fastMathNoTrans)
		transa_blas = CblasNoTrans;
	else if (transa == fastMathTrans)
		transa_blas = CblasTrans;
	else if (transa == fastMathConjTrans)
		transa_blas = CblasConjTrans;

	if (uplo == fastMathUpper)
		uplo_blas = CblasUpper;
	else if (uplo == fastMathLower)
		uplo_blas = CblasLower;

	diag_blas = CblasNonUnit;

//	cublasDtrmm(
//			side, uplo, transa, 'N',
//			n_rows, n_cols,
//			1.0,
//			m1.memptr(), lda,
//			m2.memptr(), ldb);

	cblas_dtrmm(
			CblasColMajor, side_blas, uplo_blas, transa_blas, diag_blas,
			n_rows, n_cols,
			1.0,
			m1.memptr(), lda,
			m2.memptr(), ldb);
}

void dtrmm(const mat& m1, const mat& m2, fastMathSide side, fastMathUpLo uplo, fastMathTranspose transa, mat& out)
{
	int n_rows = -1;
	int n_cols = -1;
	int lda = -1;
	int ldb = -1;
	CBLAS_SIDE side_blas;
	CBLAS_UPLO uplo_blas;
	CBLAS_TRANSPOSE transa_blas;
	CBLAS_DIAG diag_blas;

	if (side == fastMathLeft) {
		n_rows = m1.n_rows;
		n_cols = m2.n_cols;
		lda = m1.n_cols;
		ldb = m2.n_rows;
		side_blas = CblasLeft;
	} else if (side == fastMathRight) {
		n_rows = m2.n_rows;
		n_cols = m1.n_cols;
		lda = n_cols;
		ldb = n_rows;
		side_blas = CblasRight;
	}

	if (transa == fastMathNoTrans)
		transa_blas = CblasNoTrans;
	else if (transa == fastMathTrans)
		transa_blas = CblasTrans;
	else if (transa == fastMathConjTrans)
		transa_blas = CblasConjTrans;

	if (uplo == fastMathUpper)
		uplo_blas = CblasUpper;
	else if (uplo == fastMathLower)
		uplo_blas = CblasLower;

	diag_blas = CblasNonUnit;

	out = m2;

//	cublasDtrmm(
//			side, uplo, transa, 'N',
//			n_rows, n_cols,
//			1.0,
//			m1.memptr(), lda,
//			out.memptr(), ldb);
	cblas_dtrmm(
			CblasColMajor, side_blas, uplo_blas, transa_blas, diag_blas,
			n_rows, n_cols,
			1.0,
			m1.memptr(), lda,
			out.memptr(), ldb);
}

void sparsemult_left(const mat& M2, const mat& M1, mat& out)
{
	// Calculate M2 * M1 where M2 is a Hxj sparse matrix
	out = mat(M2.n_rows, M1.n_cols);
	for (uword m = 0; m < M2.n_rows/2; m++) {
		const uword base = 15 + 6*m;
		const uword row = m*2;
		const double val000 = M2(row, base);
		const double val010 = M2(row, base + 1);
		const double val020 = M2(row, base + 2);
		const double val030 = M2(row, base + 3);
		const double val040 = M2(row, base + 4);
		const double val050 = M2(row, base + 5);

		const double val100 = M2(row+1, base);
		const double val110 = M2(row+1, base + 1);
		const double val120 = M2(row+1, base + 2);
		const double val130 = M2(row+1, base + 3);
		const double val140 = M2(row+1, base + 4);
		const double val150 = M2(row+1, base + 5);

		for (uword n = 0; n < M1.n_cols; n++) {
			const double val01 = M1(base, n);
			const double val11 = M1(base + 1, n);
			const double val21 = M1(base + 2, n);
			const double val31 = M1(base + 3, n);
			const double val41 = M1(base + 4, n);
			const double val51 = M1(base + 5, n);

			out(row,n) = val000*val01 + val010*val11 + val020*val21 + val030*val31 + val040*val41 + val050*val51;
			out(row+1,n) = val100*val01 + val110*val11 + val120*val21 + val130*val31 + val140*val41 + val150*val51;
		}
	}
}

void sparsemultsym_rightt(const mat& M1, const mat& M2, mat& out)
{
	// Calculate M1*M2.t() where M2 is a Hxj sparse matrix and M1 is a symmetric matrix.
	// Note, the first 15 rows of the result are not calculated as it is not used by
	// the algorithms.

	out = mat(M1.n_rows, M2.n_rows);
	for (uword m = 0; m < M2.n_rows/2; m++) {
		const uword base = 15 + 6*m;
		const uword row = m*2;
		const double val000 = M2(row, base);
		const double val010 = M2(row, base + 1);
		const double val020 = M2(row, base + 2);
		const double val030 = M2(row, base + 3);
		const double val040 = M2(row, base + 4);
		const double val050 = M2(row, base + 5);

		const double val100 = M2(row+1, base);
		const double val110 = M2(row+1, base + 1);
		const double val120 = M2(row+1, base + 2);
		const double val130 = M2(row+1, base + 3);
		const double val140 = M2(row+1, base + 4);
		const double val150 = M2(row+1, base + 5);
		for (uword n = 15; n < M1.n_rows; n++) {
			const double* __restrict__ M1ptr = &M1.mem[base + n*M1.n_rows];
			const double val01 = *M1ptr++;
			const double val11 = *M1ptr++;
			const double val21 = *M1ptr++;
			const double val31 = *M1ptr++;
			const double val41 = *M1ptr++;
			const double val51 = *M1ptr++;

			out(n,row) = val000*val01 + val010*val11 + val020*val21 + val030*val31 + val040*val41 + val050*val51;
			out(n,row+1) = val100*val01 + val110*val11 + val120*val21 + val130*val31 + val140*val41 + val150*val51;
		}
	}
}
