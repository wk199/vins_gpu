/*
 * magmamath.h
 *
 *  Created on: Nov 9, 2015
 *      Author: nfo
 */

#ifndef MAGMAMATH_H_
#define MAGMAMATH_H_

#include <armadillo>

using namespace arma;

void magma_calculate_correction(
		const mat& Ho,
		const mat& ro,
		const mat& P,
		const double imageNoiseVar,
		mat& dstates,
		mat& Pnew
);


#endif /* MAGMAMATH_H_ */
