/*
 * qr.h
 *
 *  Created on: Sep 17, 2015
 *      Author: nfo
 */

#ifndef QR_H_
#define QR_H_

#include <armadillo>
using namespace arma;

enum fastMathSide {fastMathRight = 'R', fastMathLeft = 'L'};
enum fastMathUpLo {fastMathUpper = 'U', fastMathLower = 'L'};
enum fastMathTranspose {fastMathNoTrans = 'N', fastMathTrans = 'T', fastMathConjTrans = 'C'};

void calc_K(const mat& R, const mat& P, const vec& Rn, mat& K);

bool calc_rn_R(const mat& ro, const mat& Ho, mat& rn, mat& R);
bool calc_rn_R_full(const mat& ro, const mat& Ho, mat& rn, mat& R);

void dtrmm(const mat& m1, const mat& m2, fastMathSide side, fastMathUpLo uplo, fastMathTranspose transa, mat& out);
void dtrmm(const mat& m1, mat& m2, fastMathSide side, fastMathUpLo uplo, fastMathTranspose transa);

void sparsemultsym_rightt(const mat& M1, const mat& M2, mat& out);
void sparsemult_left(const mat& M2, const mat& M1, mat& out);

#endif /* QR_H_ */
