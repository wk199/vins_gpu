/*
 * 
 * Jim Aldon D'Souza
 *
 */
#include "kalmanfilter.h"
#include <stdio.h>
#include <iostream>
#include <math.h>
#include <algorithm>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "magma.h"
#include <vector>
#include <chrono>

#include "../triangulate/triangulate_est.h"
#include "../statehandler/statehandler.h"
#include "../navigation/navigation.h"
#include "../util/util.h"
#include "fastmath.h"
#include "magmamath.h"
#include <QtCore/QElapsedTimer>
#include <nvToolsExt.h>

void augmentState(const States& states, const vec3& p_ci_c, const mat33& Rci, mat& P)
{
	nvtxRangePushA(__FUNCTION__);
	mat33 Roc = states.Rbo.t();
	vec3 p_ci_o = Roc*p_ci_c;
	mat33 p_ci_o_skew = skew(p_ci_o);
	mat33 eye3 = eye<mat>(3,3);

	//mat row1 = join_horiz(Rci, zeros<mat>(3, P.n_cols - 3));
	//mat row2 = join_horiz(p_ci_o_skew, join_horiz(zeros<mat>(3,3), join_horiz(eye<mat>(3,3), zeros<mat>(3, P.n_cols - 9))));
	//mat J = join_vert(eye<mat>(P.n_cols, P.n_rows), join_vert(row1, row2));

	umat J_locations;
	mat J_values;

	J_locations.set_size(2, 27 + P.n_cols);
	J_values.set_size(1, 27 + P.n_cols);

	for (int row = 0; row < 3; row++) {
		for (int col = 0; col < 3; col++) {
			J_locations(0,row*3+col) = row + P.n_cols;
			J_locations(1,row*3+col) = col;
			J_values(row*3+col) = Rci(row, col);

			J_locations(0,row*3+col+9) = row + P.n_cols + 3;
			J_locations(1,row*3+col+9) = col;
			J_values(row*3+col+9) = p_ci_o_skew(row, col);

			J_locations(0,row*3+col+18) = row + P.n_cols + 3;
			J_locations(1,row*3+col+18) = col+6;
			J_values(row*3+col+18) = eye3(row, col);
		}
	}

	for (int i = 0; i < P.n_cols; i++) {
		J_locations(0,27 + i) = i;
		J_locations(1,27 + i) = i;
		J_values(27 + i) = 1.0;
	}

	SpMat<double> Jsp = SpMat<double>(J_locations, J_values, P.n_cols+6, P.n_cols);

	P = Jsp*P*Jsp.t();
	nvtxRangePop();
}

void covarianceUpdate(
        const States& states,
        const vec3& wibb,
        const vec3& ab,
        const vec3& wieo,
        const double imuTs,
        const double gNoiseVar,
        const double gBiasVar,
        const double aNoiseVar,
        const double aBiasVar,
        mat& P)
{
    static const mat Q =
    {
            {gNoiseVar, 0,         0,         0,                   0,                    0,                   0,                          0,                          0,                          0, 0, 0, 0, 0, 0},
            {0,         gNoiseVar, 0,         0,                   0,                    0,                   0,                          0,                          0,                          0, 0, 0, 0, 0, 0},
            {0,         0,         gNoiseVar, 0,                   0,                    0,                   0,                          0,                          0,                          0, 0, 0, 0, 0, 0},
            {0,         0,         0,         aNoiseVar,           0,                    0,                   0.5*imuTs*aNoiseVar,        0,                          0,                          0, 0, 0, 0, 0, 0},
            {0,         0,         0,         0,                   aNoiseVar,            0,                   0,                          0.5*imuTs*aNoiseVar,        0,                          0, 0, 0, 0, 0, 0},
            {0,         0,         0,         0,                   0,                    aNoiseVar,           0,                          0,                          0.5*imuTs*aNoiseVar,        0, 0, 0, 0, 0, 0},
            {0,         0,         0,         0.5*imuTs*aNoiseVar, 0,                    0,                   0.25*imuTs*imuTs*aNoiseVar, 0,                          0,                          0, 0, 0, 0, 0, 0},
            {0,         0,         0,         0,                   0.5*imuTs*aNoiseVar,  0,                   0,                          0.25*imuTs*imuTs*aNoiseVar, 0,                          0, 0, 0, 0, 0, 0},
            {0,         0,         0,         0,                   0,                    0.5*imuTs*aNoiseVar, 0,                          0,                          0.25*imuTs*imuTs*aNoiseVar, 0, 0, 0, 0, 0, 0},
            {0,         0,         0,         0,                   0,                    0,                   0,                          0,                          0,                          gBiasVar, 0, 0, 0, 0, 0},
            {0,         0,         0,         0,                   0,                    0,                   0,                          0,                          0,                          0, gBiasVar, 0, 0, 0, 0},
            {0,         0,         0,         0,                   0,                    0,                   0,                          0,                          0,                          0, 0, gBiasVar, 0, 0, 0},
            {0,         0,         0,         0,                   0,                    0,                   0,                          0,                          0,                          0, 0, 0, aBiasVar, 0, 0},
            {0,         0,         0,         0,                   0,                    0,                   0,                          0,                          0,                          0, 0, 0, 0, aBiasVar, 0},
            {0,         0,         0,         0,                   0,                    0,                   0,                          0,                          0,                          0, 0, 0, 0, 0, aBiasVar}
    };

    mat33 Rbo = states.Rbo;

    mat row1 = join_horiz(zeros<mat>(3, 9), join_horiz(-Rbo.t(), zeros<mat>(3,3)));
    mat row2 = join_horiz(-skew(Rbo.t()*ab), join_horiz(zeros<mat>(3,9), -Rbo.t()));
    mat row3 = join_horiz(-skew(Rbo.t()*0.5*imuTs*ab), join_horiz(eye<mat>(3,3), join_horiz(zeros(3,6), -0.5*imuTs*Rbo.t())));
    mat F = join_vert(row1, join_vert(row2, join_vert(row3, join_vert(zeros<mat>(3,15), zeros<mat>(3,15)))));


    magma_int_t n=F.n_rows, o=P.n_rows, p=P.n_cols, lddP=o, lddF=n, lddQ=n, lddPhi=n,lddPII=n, lddPIC=n;
    double *h_F, *h_Q, *h_P;
    double x=1.0, y=0.0;
    magmaDouble_ptr d_F, d_Q, d_Phi, d_PII, d_P, d_PIC, d_temp1;

    magma_dmalloc_pinned(&h_F,n*n);
    magma_dmalloc_pinned(&h_Q,n*n);
    magma_dmalloc_pinned(&h_P,o*p);

    magma_dmalloc(&d_F,n*n);
    magma_dmalloc(&d_Q,n*n);
    magma_dmalloc(&d_Phi,n*n);
    magma_dmalloc(&d_PII,n*n);
    magma_dmalloc(&d_PIC,n*(p-15));
    magma_dmalloc(&d_P,o*p);
    magma_dmalloc(&d_temp1,n*n);

	/* Initialize  allocated memory on CPU to the armadillo matrices h_A and h_B*/

	//1. h_F
    std::memcpy(h_F, F.memptr(), n*n*sizeof(double));

	//2. h_Q
    std::memcpy(h_Q, Q.memptr(), n*n*sizeof(double));

    //3. h_P
    std::memcpy(h_P, P.memptr(), o*p*sizeof(double));

    magma_dsetmatrix(n,n,h_F,n,d_F,lddF);
    magma_dsetmatrix(n,n,h_Q,n,d_Q,lddQ);
    magma_dsetmatrix(o,p,h_P,o,d_P,lddP);

    //Computing d_Phi
	magmablas_dlaset(MagmaFull,n,n,y,x,d_Phi,lddPhi);
    magmablas_dgeadd(n,n,imuTs,d_F,lddF,d_Phi,lddPhi);

    //Computing d_PII
    magmablas_dlacpy(MagmaFull,n,n,d_P,lddP,d_PII,lddPII); //dlacpy can be used as a substitute for arma::P.submat()
 
    //Computing d_P(15,15) 
    magma_dgemm(MagmaNoTrans,MagmaTrans,n,n,n,1,d_PII,lddPII,d_Phi,lddPhi,0,d_temp1,lddP);
    magma_dgemm(MagmaNoTrans,MagmaNoTrans,n,n,n,1,d_Phi,lddPhi,d_temp1,lddP,0,d_P,lddP);
    magmablas_dgeadd(n,n,imuTs,d_Q,lddQ,d_P,lddP);

    if (p > 15) {
        magma_dgemm(MagmaNoTrans,MagmaNoTrans,n,p-15,n,1,d_Phi,lddPhi,&d_P[0+15*lddP],lddP,0,&d_P[0+15*lddP],lddP);
        magmablas_dlacpy(MagmaFull,n,p-15,&d_P[0+15*lddP],lddP,d_PIC,lddPIC);
        magmablas_dtranspose(n,p-15,d_PIC,lddPIC,&d_P[15+0*lddP],lddP);
    }

    magma_dgetmatrix(o,p,d_P,lddP,h_P,lddP);
    std::memcpy(P.memptr(), h_P, o*p*sizeof(double));

	magma_free_pinned(h_F);
	magma_free_pinned(h_Q);
	magma_free_pinned(h_P);
	magma_free(d_F);
	magma_free(d_Q);
	magma_free(d_Phi);
	magma_free(d_PII);
	magma_free(d_PIC);
	magma_free(d_P);
	magma_free(d_temp1);


}
void covarianceUpdate_cpu(
		const States& states,
		const vec3& wibb,
		const vec3& ab,
		const vec3& wieo,
		const double imuTs,
		const double gNoiseVar,
		const double gBiasVar,
		const double aNoiseVar,
		const double aBiasVar,
		mat& P)
{
	nvtxRangePushA(__FUNCTION__);
	static const mat Q =
	{
			{gNoiseVar, 0,         0,         0,                   0,                    0,                   0,                          0,                          0,                          0, 0, 0, 0, 0, 0},
			{0,         gNoiseVar, 0,         0,                   0,                    0,                   0,                          0,                          0,                          0, 0, 0, 0, 0, 0},
			{0,         0,         gNoiseVar, 0,                   0,                    0,                   0,                          0,                          0,                          0, 0, 0, 0, 0, 0},
			{0,         0,         0,         aNoiseVar,           0,                    0,                   0.5*imuTs*aNoiseVar,        0,                          0,                          0, 0, 0, 0, 0, 0},
			{0,         0,         0,         0,                   aNoiseVar,            0,                   0,                          0.5*imuTs*aNoiseVar,        0,                          0, 0, 0, 0, 0, 0},
			{0,         0,         0,         0,                   0,                    aNoiseVar,           0,                          0,                          0.5*imuTs*aNoiseVar,        0, 0, 0, 0, 0, 0},
			{0,         0,         0,         0.5*imuTs*aNoiseVar, 0,                    0,                   0.25*imuTs*imuTs*aNoiseVar, 0,                          0,                          0, 0, 0, 0, 0, 0},
			{0,         0,         0,         0,                   0.5*imuTs*aNoiseVar,  0,                   0,                          0.25*imuTs*imuTs*aNoiseVar, 0,                          0, 0, 0, 0, 0, 0},
			{0,         0,         0,         0,                   0,                    0.5*imuTs*aNoiseVar, 0,                          0,                          0.25*imuTs*imuTs*aNoiseVar, 0, 0, 0, 0, 0, 0},
			{0,         0,         0,         0,                   0,                    0,                   0,                          0,                          0,                          gBiasVar, 0, 0, 0, 0, 0},
			{0,         0,         0,         0,                   0,                    0,                   0,                          0,                          0,                          0, gBiasVar, 0, 0, 0, 0},
			{0,         0,         0,         0,                   0,                    0,                   0,                          0,                          0,                          0, 0, gBiasVar, 0, 0, 0},
			{0,         0,         0,         0,                   0,                    0,                   0,                          0,                          0,                          0, 0, 0, aBiasVar, 0, 0},
			{0,         0,         0,         0,                   0,                    0,                   0,                          0,                          0,                          0, 0, 0, 0, aBiasVar, 0},
			{0,         0,         0,         0,                   0,                    0,                   0,                          0,                          0,                          0, 0, 0, 0, 0, aBiasVar}
	};

	mat33 Rbo = states.Rbo;

	mat row1 = join_horiz(zeros<mat>(3, 9), join_horiz(-Rbo.t(), zeros<mat>(3,3)));
	mat row2 = join_horiz(-skew(Rbo.t()*ab), join_horiz(zeros<mat>(3,9), -Rbo.t()));
	mat row3 = join_horiz(-skew(Rbo.t()*0.5*imuTs*ab), join_horiz(eye<mat>(3,3), join_horiz(zeros(3,6), -0.5*imuTs*Rbo.t())));
	mat F = join_vert(row1, join_vert(row2, join_vert(row3, join_vert(zeros<mat>(3,15), zeros<mat>(3,15)))));

//    F = vstack((
//        hstack(( zeros((3,3)),               zeros((3,3)), zeros((3,3)), -Rbo.T,         zeros((3,3)) )), #attitude
//        hstack((-skew(Rbo.T*ab),             zeros((3,3)), zeros((3,3)),  zeros((3,3)), -Rbo.T )), # velocity
//        hstack((-skew(Rbo.T*0.5*imuTs*ab),   eye(3),       zeros((3,3)),  zeros((3,3)), -0.5*imuTs*Rbo.T )), # position
//        zeros((3,15)), # gbias
//        zeros((3,15)) # abias
//        ))

	mat Phi = eye<mat>(15,15) + F*imuTs;

	mat PII = P.submat(0, 0, 14, 14);
	P.submat(0, 0, 14, 14) = Phi*PII*Phi.t() + Q*imuTs; // TODO: Does it make sense to use sparse matrix tricks here, or is BLAS fast enough?

	if (P.n_cols > 15) {
		mat PIC = P.submat(0, 15, 14, P.n_cols - 1);
		PIC = Phi*PIC;
		P.submat(0, 15, 14, P.n_cols - 1) = PIC;
		P.submat(15, 0, P.n_rows - 1, 14) = PIC.t();
	}
	nvtxRangePop();
}

mat hstack(std::initializer_list<mat> list)
{
	mat out;
	for (mat elem : list)
		out = join_horiz(out, elem);
	return out;
}

mat vstack(std::initializer_list<mat> list)
{
	mat out;
	for (mat elem : list)
		out = join_vert(out, elem);
	return out;
}

void covarianceUpdate_firstestimates(
		const States& states,
		const States& oldstates,
		const vec3& go,
		const double imuTs,
		const double gNoiseVar,
		const double gBiasVar,
		const double aNoiseVar,
		const double aBiasVar,
		mat& P)
{
	nvtxRangePushA(__FUNCTION__);
	static const mat Q =
	{
			{gNoiseVar, 0,         0,         0,                   0,                    0,                   0,                          0,                          0,                          0, 0, 0, 0, 0, 0},
			{0,         gNoiseVar, 0,         0,                   0,                    0,                   0,                          0,                          0,                          0, 0, 0, 0, 0, 0},
			{0,         0,         gNoiseVar, 0,                   0,                    0,                   0,                          0,                          0,                          0, 0, 0, 0, 0, 0},
			{0,         0,         0,         aNoiseVar,           0,                    0,                   0.5*imuTs*aNoiseVar,        0,                          0,                          0, 0, 0, 0, 0, 0},
			{0,         0,         0,         0,                   aNoiseVar,            0,                   0,                          0.5*imuTs*aNoiseVar,        0,                          0, 0, 0, 0, 0, 0},
			{0,         0,         0,         0,                   0,                    aNoiseVar,           0,                          0,                          0.5*imuTs*aNoiseVar,        0, 0, 0, 0, 0, 0},
			{0,         0,         0,         0.5*imuTs*aNoiseVar, 0,                    0,                   0.25*imuTs*imuTs*aNoiseVar, 0,                          0,                          0, 0, 0, 0, 0, 0},
			{0,         0,         0,         0,                   0.5*imuTs*aNoiseVar,  0,                   0,                          0.25*imuTs*imuTs*aNoiseVar, 0,                          0, 0, 0, 0, 0, 0},
			{0,         0,         0,         0,                   0,                    0.5*imuTs*aNoiseVar, 0,                          0,                          0.25*imuTs*imuTs*aNoiseVar, 0, 0, 0, 0, 0, 0},
			{0,         0,         0,         0,                   0,                    0,                   0,                          0,                          0,                          gBiasVar, 0, 0, 0, 0, 0},
			{0,         0,         0,         0,                   0,                    0,                   0,                          0,                          0,                          0, gBiasVar, 0, 0, 0, 0},
			{0,         0,         0,         0,                   0,                    0,                   0,                          0,                          0,                          0, 0, gBiasVar, 0, 0, 0},
			{0,         0,         0,         0,                   0,                    0,                   0,                          0,                          0,                          0, 0, 0, aBiasVar, 0, 0},
			{0,         0,         0,         0,                   0,                    0,                   0,                          0,                          0,                          0, 0, 0, 0, aBiasVar, 0},
			{0,         0,         0,         0,                   0,                    0,                   0,                          0,                          0,                          0, 0, 0, 0, 0, aBiasVar}
	};

	mat33 Rbo = states.Rbo;
	const mat33 eye3 = eye(3,3);
	const mat33 zeros3 = zeros(3,3);
	mat row1 = hstack({eye3,                                                                                  zeros3,     zeros3, -Rbo.t()*imuTs,  zeros3});
	mat row2 = hstack({skew(states.v_bo_o - oldstates.v_bo_o - go*imuTs),                                    eye3,       zeros3,  zeros3,        -Rbo.t()*imuTs});
	mat row3 = hstack({skew(states.p_bo_o - oldstates.p_bo_o - oldstates.v_bo_o*imuTs - 0.5*go*imuTs*imuTs), eye3*imuTs, eye3,    zeros3,        -0.5*Rbo.t()*imuTs*imuTs});
	mat row4 = hstack({zeros3,                                                                                zeros3,     zeros3,  eye3,           zeros3});
	mat row5 = hstack({zeros3,                                                                                zeros3,     zeros3,  zeros3,         eye3});
	mat Phi = vstack({row1, row2, row3, row4, row5});

//    F = vstack((
//        hstack(( zeros((3,3)),               zeros((3,3)), zeros((3,3)), -Rbo.T,         zeros((3,3)) )), #attitude
//        hstack((-skew(Rbo.T*ab),             zeros((3,3)), zeros((3,3)),  zeros((3,3)), -Rbo.T )), # velocity
//        hstack((-skew(Rbo.T*0.5*imuTs*ab),   eye(3),       zeros((3,3)),  zeros((3,3)), -0.5*imuTs*Rbo.T )), # position
//        zeros((3,15)), # gbias
//        zeros((3,15)) # abias
//        ))

	//mat Phi = eye<mat>(15,15) + F*imuTs;

	mat PII = P.submat(0, 0, 14, 14);
	P.submat(0, 0, 14, 14) = Phi*PII*Phi.t() + Q*imuTs; // TODO: Does it make sense to use sparse matrix tricks here, or is BLAS fast enough?

	if (P.n_cols > 15) {
		mat PIC = P.submat(0, 15, 14, P.n_cols - 1);
		PIC = Phi*PIC;
		P.submat(0, 15, 14, P.n_cols - 1) = PIC;
		P.submat(15, 0, P.n_rows - 1, 14) = PIC.t();
	}
	nvtxRangePop();
}

void calcPsAndQs(
		const TCameraPoses& cameraPoses,
		const vector<vec2>& trackCoords,
		const unsigned long firstImageIndex,
		const mat33& cameraMatrix,
		vector<mat34>& Ps,
		vector<vec2>& qs,
		vector<mat33>& Rcos,
		vector<vec3>& p_oc_cs,
		vector<mat33>& Rcos_first,
		vector<vec3>& p_oc_os_first)
{
	nvtxRangePushA(__FUNCTION__);
	for (unsigned long i = 0; i < trackCoords.size(); i++) {
		vec2 q = trackCoords.at(i);
		qs.push_back(q);
	}

	long firstPoseIndex = getPoseIndexFromImageIndex(cameraPoses, firstImageIndex);
	for (unsigned long i = firstPoseIndex; i < firstPoseIndex + trackCoords.size(); i++) {
		if ((i >= cameraPoses.size()) || (i < 0)) {
			cout << "i: " << i << endl;
			cout << "firstImageIndex: " << firstImageIndex << endl;
			cout << "firstPoseIndex: " << firstPoseIndex << endl;
			cout << "trackCoords.size():" << trackCoords.size() << endl;
			cout << "index of first pose: " << cameraPoses.at(0).imageIndex <<endl;
		}
		struct pose_t pose = cameraPoses.at(i);
		mat33 Rco = pose.Rco;
		vec3 p_oc_c = -Rco*pose.p_co_o;

		Rcos.push_back(Rco);
		p_oc_cs.push_back(p_oc_c);

		Ps.push_back(cameraMatrix*join_rows(Rco, p_oc_c));

		Rcos_first.push_back(pose.Rco_first);
		p_oc_os_first.push_back(-pose.p_co_o_first);
	}
	nvtxRangePop();
}

void calcPsAndQs_first(
		const TCameraPoses& cameraPoses,
		const vector<vec2>& trackCoords,
		const unsigned long firstImageIndex,
		const mat33& cameraMatrix,
		vector<mat34>& Ps,
		vector<vec2>& qs,
		vector<mat33>& Rcos,
		vector<vec3>& p_oc_cs,
		vector<mat34>& Ps_first,
		vector<mat33>& Rcos_first,
		vector<vec3>& p_oc_os_first)
{
	nvtxRangePushA(__FUNCTION__);
	for (unsigned long i = 0; i < trackCoords.size(); i++) {
		vec2 q = trackCoords.at(i);
		qs.push_back(q);
	}

	long firstPoseIndex = getPoseIndexFromImageIndex(cameraPoses, firstImageIndex);
	for (unsigned long i = firstPoseIndex; i < firstPoseIndex + trackCoords.size(); i++) {
		if ((i >= cameraPoses.size()) || (i < 0)) {
			cout << "i: " << i << endl;
			cout << "firstImageIndex: " << firstImageIndex << endl;
			cout << "firstPoseIndex: " << firstPoseIndex << endl;
			cout << "trackCoords.size():" << trackCoords.size() << endl;
			cout << "index of first pose: " << cameraPoses.at(0).imageIndex <<endl;
		}
		struct pose_t pose = cameraPoses.at(i);
		mat33 Rco = pose.Rco;
		vec3 p_oc_c = -Rco*pose.p_co_o;

		Rcos.push_back(Rco);
		p_oc_cs.push_back(p_oc_c);

		Ps.push_back(cameraMatrix*join_rows(Rco, p_oc_c));

		Rcos_first.push_back(pose.Rco_first);
		p_oc_os_first.push_back(-pose.p_co_o_first);

		mat33 Rco_first = pose.Rco_first;
		vec3 p_oc_c_first = -Rco_first*pose.p_co_o_first;
		Ps_first.push_back(cameraMatrix*join_rows(Rco_first, p_oc_c_first));

	}
	nvtxRangePop();
}

void calc_residual(
		const vector<mat33>& Rcos,
		const vector<vec3>& p_oc_cs,
		const vector<mat33>& Rcos_first,
		const vector<vec3>& p_oc_os_first,
		const vec3& p_fo_o,
		const vector<vec2>& trackCoords,
		const unsigned long firstPoseIndex,
		const unsigned long numPoses,
		const mat33& cameraMatrix,
		mat& rj,
		sp_mat& Hxj,
		mat& Hfj
		)
{
	nvtxRangePushA(__FUNCTION__);
	unsigned long trackLength = trackCoords.size();
	double dx = cameraMatrix(0,2);
	double dy = cameraMatrix(1,2);
	double fx = cameraMatrix(0,0);
	double fy = cameraMatrix(1,1);
	umat Hxj_locations;
	mat Hxj_values;

	rj.set_size(trackLength*2,1);
	Hfj.set_size(trackLength*2,3);

	Hxj_locations.set_size(2, trackLength*12);
	Hxj_values.set_size(1, trackLength*12);
	unsigned int Hxj_elemIdx = 0;

	for (unsigned long k = firstPoseIndex; k < trackLength + firstPoseIndex; k++) {
		mat33 Rco = Rcos[k - firstPoseIndex];
		vec3 p_oc_o = Rco.t()*p_oc_cs[k - firstPoseIndex];

		vec3 p_fc_o = p_fo_o + p_oc_o;
		vec3 p_fc_c = Rco*p_fc_o;

		double X = p_fc_c(0);
		double Y = p_fc_c(1);
		double Z = p_fc_c(2);

		vec2 zhat;
		zhat(0) = X/Z;
		zhat(1) = Y/Z;

		mat J;
		J << 1/Z << 0   << -X/(Z*Z) << endr << 0   << 1/Z << -Y/(Z*Z) << endr;

		mat Hxjk = join_rows(J*Rco*skew(p_fc_o), -J*Rco);
		mat Hfjk = J*Rco;

		vec2 z;
		z(0) = (trackCoords.at(k - firstPoseIndex)(0) - dx)/fx;
		z(1) = (trackCoords.at(k - firstPoseIndex)(1) - dy)/fy;

		vec2 rjk = z - zhat;

		// TODO Create the individual Hxjk matrices and do custom sparse matrix multiplication
		unsigned int startCol = 15 + 6*k;
		unsigned int startRow = (k - firstPoseIndex)*2;
		for (unsigned int row = startRow; row < startRow + Hxjk.n_rows; row++) {
			for (unsigned int col = startCol; col < startCol + Hxjk.n_cols; col++) {
				Hxj_locations(0, Hxj_elemIdx) = row;
				Hxj_locations(1, Hxj_elemIdx) = col;
				Hxj_values(Hxj_elemIdx) = Hxjk(row - startRow, col - startCol);
				Hxj_elemIdx++;
			}
		}

		Hfj.rows((k - firstPoseIndex)*2, (k-firstPoseIndex)*2 + 1) = Hfjk;
		rj((k - firstPoseIndex)*2) = rjk(0);
		rj((k - firstPoseIndex)*2 + 1) = rjk(1);
	}

	Hxj = SpMat<double>(Hxj_locations, Hxj_values, trackLength*2, 15+6*numPoses);
	nvtxRangePop();
}

void calc_residual_firstestimates(
		const vector<mat33>& Rcos,
		const vector<vec3>& p_oc_cs,
		const vector<mat33>& Rcos_first,
		const vector<vec3>& p_oc_os_first,
		const vec3& p_fo_o,
		const vec3& p_fo_o_first,
		const vector<vec2>& trackCoords,
		const unsigned long firstPoseIndex,
		const unsigned long numPoses,
		const mat33& cameraMatrix,
		mat& rj,
		sp_mat& Hxj,
		mat& Hfj
		)
{
	nvtxRangePushA(__FUNCTION__);
	unsigned long trackLength = trackCoords.size();
	double dx = cameraMatrix(0,2);
	double dy = cameraMatrix(1,2);
	double fx = cameraMatrix(0,0);
	double fy = cameraMatrix(1,1);
	umat Hxj_locations;
	mat Hxj_values;

	rj.set_size(trackLength*2,1);
	Hfj.set_size(trackLength*2,3);

	Hxj_locations.set_size(2, trackLength*12);
	Hxj_values.set_size(1, trackLength*12);
	unsigned int Hxj_elemIdx = 0;

	for (unsigned long k = firstPoseIndex; k < trackLength + firstPoseIndex; k++) {
		mat33 Rco = Rcos[k - firstPoseIndex];
		vec3 p_oc_o = Rco.t()*p_oc_cs[k - firstPoseIndex];

		vec3 p_fc_o = p_fo_o + p_oc_o;
		vec3 p_fc_c = Rco*p_fc_o;

		double X = p_fc_c(0);
		double Y = p_fc_c(1);
		double Z = p_fc_c(2);

		vec2 zhat;
		zhat(0) = X/Z;
		zhat(1) = Y/Z;

		// Now use first estimates for Jacobians
		Rco = Rcos_first[k - firstPoseIndex];
		p_oc_o = p_oc_os_first[k - firstPoseIndex];

		p_fc_o = p_fo_o_first + p_oc_o;
		p_fc_c = Rco*p_fc_o;

		X = p_fc_c(0);
		Y = p_fc_c(1);
		Z = p_fc_c(2);

		mat J;
		J << 1/Z << 0   << -X/(Z*Z) << endr <<
			 0   << 1/Z << -Y/(Z*Z) << endr;

		mat Hxjk = J*Rco*join_rows(skew(p_fc_o), -eye(3,3));
		mat Hfjk = J*Rco;

		vec2 z;
		z(0) = (trackCoords.at(k - firstPoseIndex)(0) - dx)/fx;
		z(1) = (trackCoords.at(k - firstPoseIndex)(1) - dy)/fy;

		vec2 rjk = z - zhat;

		// TODO Create the individual Hxjk matrices and do custom sparse matrix multiplication
		unsigned int startCol = 15 + 6*k;
		unsigned int startRow = (k - firstPoseIndex)*2;
		for (unsigned int row = startRow; row < startRow + Hxjk.n_rows; row++) {
			for (unsigned int col = startCol; col < startCol + Hxjk.n_cols; col++) {
				Hxj_locations(0, Hxj_elemIdx) = row;
				Hxj_locations(1, Hxj_elemIdx) = col;
				Hxj_values(Hxj_elemIdx) = Hxjk(row - startRow, col - startCol);
				Hxj_elemIdx++;
			}
		}

		Hfj.rows((k - firstPoseIndex)*2, (k-firstPoseIndex)*2 + 1) = Hfjk;
		rj((k - firstPoseIndex)*2) = rjk(0);
		rj((k - firstPoseIndex)*2 + 1) = rjk(1);
	}

	Hxj = SpMat<double>(Hxj_locations, Hxj_values, trackLength*2, 15+6*numPoses);
	nvtxRangePop();
}

bool is_inlier
(
const sp_mat& Hxj,
const mat& Aj,
const mat& P,
const mat& roj,
const unsigned long trackLength,
const double imageNoiseVar,
const double chiMult
)
{
mat res1, res2;
mat Hxj2 = mat(Hxj);
sparsemultsym_rightt(P, Hxj2, res1);
sparsemult_left(Hxj2, res1, res2);

static double chitable[] = {
    -1.00,-1.00,3.84,7.81,11.07,14.07,16.92,19.68,22.36,25.00,27.59,30.14,32.67,
    35.17,37.65,40.11,42.56,44.99,47.40,49.80,52.19,54.57,56.94,59.30,61.66,64.00,
    66.34,68.67,70.99,73.31,75.62,77.93,80.23,82.53,84.82,87.11,89.39,91.67,93.95,
    96.22,98.48,100.75,103.01,105.27,107.52,109.77,112.02,114.27,116.51,118.75,
    120.99,123.23,125.46,127.69,129.92,132.14,134.37,136.59,138.81,141.03,143.25,
    145.46,147.67,149.88,152.09,154.30,156.51,158.71,160.91,163.12,165.32,167.51,
    169.71,171.91,174.10,176.29,178.49,180.68,182.86,185.05,187.24,189.42,191.61,
    193.79,195.97,198.15,200.33,202.51,204.69,206.87,209.04,211.22,213.39,215.56,
    217.73,219.91,222.08,224.24,226.41,228.58,230.75,232.91,235.08,237.24,239.40,
    241.57,243.73,245.89,248.05,250.21,252.37,254.52,256.68,258.84,260.99,263.15,
    265.30,267.45,269.61,271.76,273.91,276.06,278.21,280.36,282.51,284.66,286.81,
    288.96,291.10,293.25,295.39,297.54,299.68,301.83,303.97,306.11,308.25,310.40,
    312.54,314.68,316.82,318.96,321.10,323.24,325.37,327.51,329.65,331.79,333.92,
    336.06,338.19,340.33,342.46,344.60,346.73,348.86,351.00,353.13,355.26,357.39,
    359.52,361.65,363.78,365.91,368.04,370.17,372.30,374.43,376.55,378.68,380.81,
    382.94,385.06,387.19,389.31,391.44,393.56,395.69,397.81,399.94,402.06,404.18,
    406.30,408.43,410.55,412.67,414.79,416.91,419.03,421.15,423.27,425.39,427.51,
    429.63,431.75,433.87,435.99,438.11,440.22,442.34,444.46,446.57,448.69,450.81,
    452.92,455.04,457.15,459.27,461.38,463.50,465.61,467.73,469.84,471.95,474.07,
    476.18,478.29,480.40,482.51,484.63,486.74,488.85,490.96,493.07,495.18,497.29,
    499.40,501.51,503.62,505.73,507.84,509.95,512.06,514.16,516.27,518.38,520.49,
    522.60,524.70,526.81,528.92,531.02,533.13,535.23,537.34,539.45,541.55,543.66,
    545.76,547.87,549.97,552.07,554.18,556.28,558.39,560.49,562.59,564.70,566.80,
    568.90,571.00,573.11,575.21,577.31,579.41,581.51,583.61,585.72,587.82,589.92,
    592.02,594.12,596.22,598.32,600.42,602.52,604.62,606.72,608.82,610.91,613.01,
    615.11,617.21,619.31,621.41,623.50,625.60,627.70,629.80,631.89,633.99,636.09,
    638.18,640.28,642.38,644.47,646.57,648.66,650.76,652.86,654.95,657.05,659.14,
    661.24,663.33,665.43,667.52,669.61,671.71,673.80,675.90,677.99,680.08,682.18,
    684.27,686.36,688.45,690.55,692.64,694.73,696.82,698.92,701.01,703.10,705.19,
    707.28,709.38,711.47,713.56,715.65,717.74,719.83,721.92,724.01,726.10,728.19,
    730.28,732.37,734.46,736.55,738.64,740.73,742.82,744.91,747.00,749.09,751.18,
    753.26,755.35,757.44,759.53,761.62,763.70,765.79,767.88,769.97,772.06,774.14,
    776.23,778.32,780.40,782.49,784.58,786.66,788.75,790.84,792.92,795.01,797.10,
    799.18,801.27,803.35,805.44,807.52,809.61,811.69,813.78,815.86,817.95,820.03,
    822.12,824.20,826.29,828.37,830.46,832.54,834.62,836.71,838.79,840.87,842.96,
    845.04,847.13,849.21,851.29,853.37,855.46,857.54,859.62,861.71,863.79,865.87,
    867.95,870.03,872.12,874.20,876.28,878.36,880.44,882.53,884.61,886.69,888.77,
    890.85,892.93,895.01,897.09,899.17,901.26,903.34,905.42,907.50,909.58,911.66,
    913.74,915.82,917.90,919.98,922.06,924.14,926.22,928.30,930.37,932.45,934.53,
    936.61,938.69,940.77,942.85,944.93,947.01,949.08,951.16,953.24,955.32,957.40,
    959.48,961.55,963.63,965.71,967.79,969.86,971.94,974.02,976.10,978.17,980.25,
    982.33,984.41,986.48,988.56,990.64,992.71,994.79,996.87,998.94};


magma_int_t  n=Hxj.n_rows, p=Aj.n_rows, q=roj.n_cols, info, ldwork, lddres2=n, lddAj=p, lddroj=p, lddM=p;
magmaDouble_ptr d_temp3, d_temp4, d_Aj, d_roj, d_M, d_chi, d_work, d_imageNoiseVar, d_res2;
int *ipiv;
double *h_Aj, *h_roj, *h_chi, *h_res2;

magma_imalloc_pinned(&ipiv,p);
magma_dmalloc_pinned(&h_Aj,p*n);
magma_dmalloc_pinned(&h_chi,q*q);
magma_dmalloc_pinned(&h_roj,p*q);
magma_dmalloc_pinned(&h_res2,n*n);

magma_dmalloc(&d_Aj,p*n);
magma_dmalloc(&d_roj,p*q);
magma_dmalloc(&d_temp3,n*p);
magma_dmalloc(&d_temp4,p*q);
magma_dmalloc(&d_chi,q*q);
magma_dmalloc(&d_M,n*n);
magma_dmalloc(&d_imageNoiseVar,p*p);
magma_dmalloc(&d_res2,n*n);

//1. h_res2
std::memcpy(h_res2, res2.memptr(), n*n*sizeof(double));
//2. h_Aj
std::memcpy(h_Aj, Aj.memptr(), p*n*sizeof(double));
//3. h_roj
std::memcpy(h_roj, roj.memptr(), p*q*sizeof(double));

magma_dsetmatrix(n,n,h_res2,n,d_res2,lddres2);
magma_dsetmatrix(p,n,h_Aj,p,d_Aj,lddAj);    
magma_dsetmatrix(p,q,h_roj,p,d_roj,lddroj); 

//d_M=d_Aj*d_res2()*d_Aj.t();
magma_dgemm(MagmaNoTrans,MagmaTrans,n,p,n,1,d_res2,n,d_Aj,lddAj,0,d_temp3,n);
magma_dgemm(MagmaNoTrans,MagmaNoTrans,p,p,n,1,d_Aj,lddAj,d_temp3,n,0,d_M,lddM);

//Adding imageNoiseVar to diagonals of M
magmablas_dlaset(MagmaFull,p,p,0,imageNoiseVar,d_imageNoiseVar,p);
magmablas_dgeadd(p,p,1,d_imageNoiseVar,p,d_M,lddM); 

//inverse of d_M
ldwork=p*magma_get_dgetri_nb(p);
magma_dmalloc(&d_work, ldwork);
magma_dgetrf_gpu(p,p,d_M,lddM,ipiv,&info);
magma_dgetri_gpu(p,d_M,p,ipiv,d_work,ldwork,&info);
//magma_dprint_gpu(p,p,d_M,p);

//chi=roj.t()*M.inv()*roj
magma_dgemm(MagmaNoTrans,MagmaNoTrans,p,q,p,1,d_M,lddM,d_roj,lddroj,0,d_temp4,p);
magma_dgemm(MagmaTrans,MagmaNoTrans,q,q,p,1,d_roj,lddroj,d_temp4,p,0,d_chi,q);

//Retrieve matrix chi from d_chi    
magma_dgetmatrix(q,q,d_chi,q,h_chi,q);

//Compare first element of chi  
double resu;
resu =  h_chi[0] < chitable[trackLength]*chiMult;

//free memory
magma_free_pinned(h_Aj);
magma_free_pinned(h_chi);
magma_free_pinned(h_roj);
magma_free_pinned(ipiv);
magma_free_pinned(h_res2);

magma_free(d_work);
magma_free(d_Aj);
magma_free(d_roj);
magma_free(d_temp3);
magma_free(d_temp4);
magma_free(d_chi);
magma_free(d_M);
magma_free(d_imageNoiseVar);
magma_free(d_res2);


return resu;
}

bool is_inlier_cpu(
		const sp_mat& Hxj,
		const mat& Aj,
		const mat& P,
		const mat& roj,
		const unsigned long trackLength,
		const double imageNoiseVar,
		const double chiMult
		)
{
	nvtxRangePushA(__FUNCTION__);
	QElapsedTimer tmr;
	static double chitable[] = {
			-1.00,-1.00,3.84,7.81,11.07,14.07,16.92,19.68,22.36,25.00,27.59,30.14,32.67,
			35.17,37.65,40.11,42.56,44.99,47.40,49.80,52.19,54.57,56.94,59.30,61.66,64.00,
			66.34,68.67,70.99,73.31,75.62,77.93,80.23,82.53,84.82,87.11,89.39,91.67,93.95,
			96.22,98.48,100.75,103.01,105.27,107.52,109.77,112.02,114.27,116.51,118.75,
			120.99,123.23,125.46,127.69,129.92,132.14,134.37,136.59,138.81,141.03,143.25,
			145.46,147.67,149.88,152.09,154.30,156.51,158.71,160.91,163.12,165.32,167.51,
			169.71,171.91,174.10,176.29,178.49,180.68,182.86,185.05,187.24,189.42,191.61,
			193.79,195.97,198.15,200.33,202.51,204.69,206.87,209.04,211.22,213.39,215.56,
			217.73,219.91,222.08,224.24,226.41,228.58,230.75,232.91,235.08,237.24,239.40,
			241.57,243.73,245.89,248.05,250.21,252.37,254.52,256.68,258.84,260.99,263.15,
			265.30,267.45,269.61,271.76,273.91,276.06,278.21,280.36,282.51,284.66,286.81,
			288.96,291.10,293.25,295.39,297.54,299.68,301.83,303.97,306.11,308.25,310.40,
			312.54,314.68,316.82,318.96,321.10,323.24,325.37,327.51,329.65,331.79,333.92,
			336.06,338.19,340.33,342.46,344.60,346.73,348.86,351.00,353.13,355.26,357.39,
			359.52,361.65,363.78,365.91,368.04,370.17,372.30,374.43,376.55,378.68,380.81,
			382.94,385.06,387.19,389.31,391.44,393.56,395.69,397.81,399.94,402.06,404.18,
			406.30,408.43,410.55,412.67,414.79,416.91,419.03,421.15,423.27,425.39,427.51,
			429.63,431.75,433.87,435.99,438.11,440.22,442.34,444.46,446.57,448.69,450.81,
			452.92,455.04,457.15,459.27,461.38,463.50,465.61,467.73,469.84,471.95,474.07,
			476.18,478.29,480.40,482.51,484.63,486.74,488.85,490.96,493.07,495.18,497.29,
			499.40,501.51,503.62,505.73,507.84,509.95,512.06,514.16,516.27,518.38,520.49,
			522.60,524.70,526.81,528.92,531.02,533.13,535.23,537.34,539.45,541.55,543.66,
			545.76,547.87,549.97,552.07,554.18,556.28,558.39,560.49,562.59,564.70,566.80,
			568.90,571.00,573.11,575.21,577.31,579.41,581.51,583.61,585.72,587.82,589.92,
			592.02,594.12,596.22,598.32,600.42,602.52,604.62,606.72,608.82,610.91,613.01,
			615.11,617.21,619.31,621.41,623.50,625.60,627.70,629.80,631.89,633.99,636.09,
			638.18,640.28,642.38,644.47,646.57,648.66,650.76,652.86,654.95,657.05,659.14,
			661.24,663.33,665.43,667.52,669.61,671.71,673.80,675.90,677.99,680.08,682.18,
			684.27,686.36,688.45,690.55,692.64,694.73,696.82,698.92,701.01,703.10,705.19,
			707.28,709.38,711.47,713.56,715.65,717.74,719.83,721.92,724.01,726.10,728.19,
			730.28,732.37,734.46,736.55,738.64,740.73,742.82,744.91,747.00,749.09,751.18,
			753.26,755.35,757.44,759.53,761.62,763.70,765.79,767.88,769.97,772.06,774.14,
			776.23,778.32,780.40,782.49,784.58,786.66,788.75,790.84,792.92,795.01,797.10,
			799.18,801.27,803.35,805.44,807.52,809.61,811.69,813.78,815.86,817.95,820.03,
			822.12,824.20,826.29,828.37,830.46,832.54,834.62,836.71,838.79,840.87,842.96,
			845.04,847.13,849.21,851.29,853.37,855.46,857.54,859.62,861.71,863.79,865.87,
			867.95,870.03,872.12,874.20,876.28,878.36,880.44,882.53,884.61,886.69,888.77,
			890.85,892.93,895.01,897.09,899.17,901.26,903.34,905.42,907.50,909.58,911.66,
			913.74,915.82,917.90,919.98,922.06,924.14,926.22,928.30,930.37,932.45,934.53,
			936.61,938.69,940.77,942.85,944.93,947.01,949.08,951.16,953.24,955.32,957.40,
			959.48,961.55,963.63,965.71,967.79,969.86,971.94,974.02,976.10,978.17,980.25,
			982.33,984.41,986.48,988.56,990.64,992.71,994.79,996.87,998.94};

	static unsigned long long mult_time = 0;
	static unsigned long long inv_time = 0;
	tmr.start();

	mat res1, res2;
	mat Hxj2 = mat(Hxj);
	sparsemultsym_rightt(P, Hxj2, res1);
	sparsemult_left(Hxj2, res1, res2);
	mat M = Aj*res2*Aj.t();
	//	mat M = Aj*Hxj*P*Hxj.t()*Aj.t();

	mult_time += tmr.nsecsElapsed();

	vec imageNoise(M.n_rows);
	imageNoise.fill(imageNoiseVar);
	M.diag() += imageNoise;

	tmr.restart();
	mat chi = roj.t()*inv(M)*roj; // TODO: Check if inv_sympd works faster on target platform. It does not nfos computer.
	inv_time += tmr.nsecsElapsed();

//	cout << "inlier: mult: " << mult_time / 1000000 << " inv: " << inv_time/1000000 << endl;
	nvtxRangePop();
	return chi[0] < chitable[trackLength]*chiMult;
}

void calculate_correction(
		const mat& Ho,
		const mat& ro,
		const mat& P,
		const double imageNoiseVar,
		mat& dstates,
		mat& Pnew
		)
{
	//------GPU substitute for calc_rn_R() func in fastmath.cpp------

	const magma_int_t m = Ho.n_rows, n = Ho.n_cols ,  o=ro.n_rows, p=ro.n_cols;
	magma_int_t ldwork, info, min_mn,dTsize,nb, max_mn; 
	dstates.zeros(n,p);
	Pnew.zeros(n,n);
	double *tau, *tau2 ; 
	min_mn = min (m, n);
	max_mn = max (m, n);
	double *h_Ho, *h_ro, *wa, *h_P, *h_Rn, *h_M, *h_dstates, *h_Pnew; 
	int *ipiv;
	magmaDouble_ptr d_Ho, d_ro, d_R, d_P, d_Rn, d_M, d_dstates, d_work, d_T, d_Ho2;
	magma_int_t lddHo=m, lddro=o, lddR=min_mn, lddP=n, lddRn=min_mn, lddM=n;
	nb=magma_get_dgeqrf_nb(m);
	dTsize=nb*(2*min_mn+magma_roundup(max_mn,32));
	
	magma_imalloc_pinned(&ipiv,min_mn);
	magma_dmalloc_pinned(&tau ,m ); 
	magma_dmalloc_pinned(&tau2 ,m ); 
	magma_dmalloc_pinned(&h_Ho,m*n ); 
	magma_dmalloc_pinned(&h_ro,o*p ); 
	magma_dmalloc_pinned(&wa , o*o);
	magma_dmalloc_pinned(&h_P,n*n);
	magma_dmalloc_pinned(&h_Rn,min_mn*min_mn);
	magma_dmalloc_pinned(&h_M,n*n);
	magma_dmalloc_pinned(&h_Pnew,n*n);

	magma_dmalloc(&d_Ho , m*n); 
	magma_dmalloc(&d_Ho2 , m*n);
	magma_dmalloc(&d_ro , o*p); 
	magma_dmalloc(&d_R , min_mn*n);
	magma_dmalloc(&d_P , n*n);
	magma_dmalloc(&d_Rn , min_mn*min_mn); 
	magma_dmalloc(&d_M , n*n); 
	magma_dmalloc(&d_T, dTsize);

	/* Initialize  allocated memory on CPU to the armadillo matrices h_A and h_B*/

	//1. h_Ho
	memcpy(h_Ho, Ho.memptr(), m*n*sizeof(double));

	//2. h_ro
	memcpy(h_ro, ro.memptr(), p*o*sizeof(double));

	//3. h_P
	memcpy(h_P, P.memptr(), n*n*sizeof(double));

	//4. h_M
	memset(h_M, 0, n*n*sizeof(double));

	for (int j=0; j<n; j++)
		h_M[n*j + j] = 1.0;
	
	//MAGMA
	// Get size for workspace
	//cout<<"starting magma qr"<<endl;	
	magma_dsetmatrix ( m, n, h_Ho, m, d_Ho , lddHo); 
	magma_dsetmatrix ( o, p, h_ro, o, d_ro , lddro );
	magma_dsetmatrix ( n, n, h_P, n, d_P , lddP ); 
	magma_dsetmatrix ( n, n, h_M, n, d_M , lddM );
	
	magmablas_dlaset(MagmaFull,min_mn,n,0,0,d_R,lddR);
	magmablas_dlaset(MagmaFull,min_mn,min_mn,0,imageNoiseVar,d_Rn,lddRn);
	magmablas_dlacpy(MagmaFull,m,n,d_Ho,lddHo,d_Ho2,lddHo);
	
	magma_dgeqrf2_gpu( m, n, d_Ho, lddHo, tau, &info); 

	//Calculate R
	magmablas_dlacpy(MagmaUpper,min_mn,n,d_Ho,lddHo,d_R,lddR);
	double *hwork;
	double *hwork_query;
	magma_int_t lwork;
	lwork=nb*(2*n+nb);
	//if(m>500){magma_dmalloc_pinned(&hwork,4*lwork);}
	//else{magma_dmalloc_pinned(&hwork,lwork);}
	magma_dmalloc_pinned(&hwork,4*lwork);
	magma_dmalloc_pinned(&hwork_query,2);

	magma_dgeqrf_gpu(m,n,d_Ho2,lddHo,tau2,d_T,&info);
	magma_dormqr_gpu(MagmaLeft,MagmaTrans,o,p,min_mn,d_Ho2,lddHo,tau2,d_ro,lddro,hwork_query,-1,d_T,nb,&info);
	lwork=(magma_int_t)MAGMA_D_REAL(hwork_query[0]);
	magma_dormqr_gpu(MagmaLeft,MagmaTrans,o,p,min_mn,d_Ho2,lddHo,tau2,d_ro,lddro,hwork,lwork,d_T,nb,&info);
	
	magma_free_pinned(tau);
	magma_free_pinned(tau2); 
	magma_free_pinned(h_Ho);
	magma_free_pinned(h_ro);
	magma_free_pinned(wa);
	magma_free_pinned(h_P);
	magma_free_pinned(h_M);
	magma_free_pinned(hwork);
	magma_free_pinned(hwork_query);
	magma_free(d_Ho);
	magma_free(d_Ho2);
	magma_free(d_T);

	magmaDouble_ptr d_temp1,d_temp2,d_temp3,d_temp4,d_temp5,d_temp6,d_temp7;
	magma_dmalloc(&d_temp1,n*m);
	magma_dmalloc(&d_temp2,m*m);
	magma_dmalloc(&d_temp4,n*m);
	magma_dmalloc(&d_temp3,n*n);
	magma_dmalloc(&d_temp5,n*n);
	magma_dmalloc(&d_temp6,m*n);
	magma_dmalloc(&d_temp7,n*n);
	
	magma_dgemm(MagmaNoTrans,MagmaTrans,n,min_mn,n,1,d_P,lddP,d_R,lddR,0,d_temp1,n);
	magma_dgemm(MagmaNoTrans,MagmaNoTrans,min_mn,min_mn,n,1,d_R,lddR,d_temp1,n,0,d_temp2,min_mn);
	magmablas_dgeadd(min_mn,min_mn,1,d_Rn,lddRn,d_temp2,min_mn);

	//inverse
	ldwork=min_mn*magma_get_dgetri_nb(min_mn);
	magma_dmalloc(&d_work, ldwork);
	magma_dgetrf_gpu(min_mn,min_mn,d_temp2,min_mn,ipiv,&info);
	magma_dgetri_gpu(min_mn,d_temp2,min_mn,ipiv,d_work,ldwork,&info);

	magma_dgemm(MagmaNoTrans,MagmaNoTrans,n,min_mn,min_mn,1,d_temp1,n,d_temp2,min_mn,0,d_temp4,n);
	//d_temp4 is K=P*R'*inv(R*P*R'+Rn)

	//dstates in d_dstates	
	magma_dmalloc_pinned(&h_dstates,n*p);
	magma_dmalloc(&d_dstates,n*p);
	magma_dgemm(MagmaNoTrans,MagmaNoTrans,n,p,min_mn,1,d_temp4,n,d_ro,lddro,0,d_dstates,n);
	magma_dgetmatrix ( n, p, d_dstates, n, h_dstates , n ); 

	//output in arma::mat dstates
	for (int j=0;j<p;j++){
     	for (int i=0;i<n;i++){
        	dstates.at(n*j+i)=h_dstates[n*j+i];
     	}
     	}

	//M
	magma_dgemm(MagmaNoTrans,MagmaNoTrans,n,n,min_mn,-1,d_temp4,n,d_R,lddR,1,d_M,lddM);

	//Pnew
	magma_dgemm(MagmaNoTrans,MagmaTrans,n,n,n,1,d_P,lddP,d_M,lddM,0,d_temp3,n);
	magma_dgemm(MagmaNoTrans,MagmaNoTrans,n,n,n,1,d_M,lddM,d_temp3,n,0,d_temp5,n);

	magma_dgemm(MagmaNoTrans,MagmaTrans,min_mn,n,min_mn,1,d_Rn,lddRn,d_temp4,n,0,d_temp6,min_mn);
	magma_dgemm(MagmaNoTrans,MagmaNoTrans,n,n,min_mn,1,d_temp4,n,d_temp6,min_mn,0,d_temp7,n);

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
	magma_free(d_M);
	magma_free(d_temp1);
	magma_free(d_temp2);
	magma_free(d_temp3);
	magma_free(d_temp4);
	magma_free(d_temp5);
	magma_free(d_temp6);
	magma_free(d_temp7);
	magma_free(d_work); 
	magma_free(d_dstates);
	magma_free_pinned(h_dstates);
	magma_free_pinned(h_Pnew);
	magma_free_pinned(h_Rn); 
	magma_free_pinned(ipiv); 
}

void calculate_correction_cpu(
		const mat& Ho,
		const mat& ro,
		const mat& P,
		const double imageNoiseVar,
		mat& dstates,
		mat& Pnew
		)
{
	nvtxRangePushA(__FUNCTION__);
	static long long calc_rn_R_time = 0;
	static long long Ktime = 0;
	static long long KRnKttime = 0;
	static long long Ptime = 0;
	static long long Mtime = 0;
	static long long totaltime = 0;
	QElapsedTimer tmr;
	QElapsedTimer tmr2;
	mat rn;
	mat R;
	tmr2.start();
	tmr.start();
	calc_rn_R(ro, Ho, rn, R);
	calc_rn_R_time += tmr.nsecsElapsed();

	// Express Rn as a vector although it is a diagonal matrix
	vec Rn(rn.n_rows);
	Rn.fill(imageNoiseVar);

	mat K;
	tmr.restart();
	calc_K(R, P, Rn, K);
	Ktime += tmr.nsecsElapsed();

	tmr.restart();
	mat M;
	M.eye(P.n_rows, P.n_rows);
	M = M - K*R;
	Mtime += tmr.nsecsElapsed();


	tmr.restart();
	// TODO: Optimize this as Rn is diagonal. Use DSYRK for K*K.t() when possible (do the
	// matrices need to be square for K*Rn*K.t() = Rn*K*K.t()?). Also, use optimized blas
	// when we can (P is symmetric).
	mat RnKt = K.t();
	RnKt.each_col() %= Rn;
	mat KRnKt = K*RnKt;
	KRnKttime += tmr.nsecsElapsed();

	//cout << "R: " << R.n_rows << "x" << R.n_cols << " " << norm(R-R.t(), "fro") << endl;
	//cout << "K: " << K.n_rows << "x" << K.n_cols << " " << norm(K-K.t(), "fro") << endl;
	//cout << "M: " << M.n_rows << "x" << M.n_cols << " " << norm(M-M.t(), "fro") << endl;

	tmr.restart();
	Pnew = M*P*M.t() + KRnKt;
	Ptime += tmr.nsecsElapsed();

	dstates = K*rn;

	totaltime += tmr2.nsecsElapsed();

	cout << "update_time - " <<
			"calc_rn_R: " << calc_rn_R_time/1000000 << " (" << (calc_rn_R_time*100)/totaltime << "%)" <<
			" K: " << Ktime/1000000 << " (" << (Ktime*100)/totaltime << "%)" <<
			" M: " << Mtime/1000000 << " (" << (Mtime*100)/totaltime << "%)" <<
			" KRnKt: " << KRnKttime/1000000 << " (" << (KRnKttime*100)/totaltime << "%)" <<
			" P: " << Ptime/1000000 << " (" << (Ptime*100)/totaltime << "%)" <<
			" tot: " << totaltime/1000000 << endl;
	nvtxRangePop();
}

void remove_feature_dependency(
		const mat& Hfj,
		const sp_mat& Hxj,
		const mat& rj,
		mat& Aj)
{
	// TODO: According to the article, we do not need to explicitly calculate Aj
	if (!null(Aj, Hfj.t())) {
		cout << "Unable to find nullspace of matrix Hfj: " << endl;
		cout << Hfj << endl;
	}
	Aj = Aj.t();
}

void correct_states(const mat& dstates, const unsigned long numPoses, States& states, TCameraPoses& camPoses)
{
	states.abias = correct_vector(states.abias, get_dabias(dstates));
	states.gbias = correct_vector(states.gbias, -get_dgbias(dstates));
	states.Rbo = correct_rot(states.Rbo, get_dq_bo(dstates));
	states.v_bo_o = correct_vector(states.v_bo_o, get_dv_bo_o(dstates));
	states.p_bo_o = correct_vector(states.p_bo_o, get_dp_bo_o(dstates));

	for (unsigned long i = 0; i < numPoses; i++) {
		camPoses[i].Rco = correct_rot(camPoses[i].Rco, get_dq_co(dstates, i));
		camPoses[i].p_co_o = correct_vector(camPoses[i].p_co_o, get_dp_co_o(dstates, i));
	}
}

void update(
		mat& P,
		struct States* states,
		Tracks2D& tracks,
		TCameraPoses& cameraPoses,
		const mat33& cameraMatrix,
		const double imageNoiseVar,
		const unsigned long maxTrackLength,
		const unsigned long maxEkfTracks,
		const double minBaseline,
		const double minBaselineMult,
		const double minDistance,
		const double maxDistance,
		const double chiMult)
{
	static long counter = 0;
	nvtxRangePushA(__FUNCTION__);
	static long long PsAndQs_time = 0;
	static long long triang_time = 0;
	static long long residual_time = 0;
	static long long featuredep_time = 0;
	static long long outlier_time = 0;
	static long long correction_time = 0;
	static long long total_time = 0;
	QElapsedTimer tmr;
	QElapsedTimer tmr2;

	tmr2.start();

	unsigned long numPoses = cameraPoses.size();

	cout << "Number of poses: " << numPoses << endl;

	vector<mat*> Hojs;
	vector<mat*> rojs;
	vector<CTrack2D*> potentialNewEkfTracks;
	unsigned long numEkfTracks = 0;
	for (unsigned long trackIdx = 0; trackIdx < tracks.size(); trackIdx++) {
		CTrack2D* track = tracks.at(trackIdx);

		track->setDelete(false); // TODO: This will not delete the track, just color it green

		if (track->isEkfTracked()) {
			numEkfTracks++;
			continue;
		}

		// Process track if it is lost, or it has been seen for maxTrackLength (additional) frames
		if (!(track->isLost() || (track->nFrames() % maxTrackLength == 0)))
			continue;

		// If the track is not lost, it should be marked as a potential EKF track
		if (!track->isLost())
			potentialNewEkfTracks.push_back(track);

		// Reset track status
		track->setOutlier(false);
		track->setTooShort(false);
		track->setTooShortBaseline(false);

		if (track->nFrames() < 2) {
			track->setTooShort(true);
			continue;
		}

		track->setDelete(true); // TODO: This will not delete the track, just color it green

		// Extract most recent part of track (at most maxTrackLength coords)
		const unsigned long trackLength = min(track->nFrames(), (size_t)maxTrackLength);
		const unsigned long firstImageIndex = track->getFirstImageIndex() + track->nFrames() - trackLength;
		vector<vec2> coords(track->getCoords().end() - trackLength, track->getCoords().end());

		//cout << "id" << track->getID() << " lost: " << track->isLost() << " Tot len: " << track->nFrames() << " part: " << trackLength << " coords: " << coords.size() << " FIIndex: " << firstImageIndex << endl;

		vector<mat34> Ps;
		vector<vec2> qs;
		vector<mat33> Rcos;
		vector<vec3> poc_cs;
		vector<mat34> Ps_first;
		vector<mat33> Rcos_first;
		vector<vec3> p_oc_os_first;

		tmr.start();
		calcPsAndQs_first(cameraPoses, coords, firstImageIndex, cameraMatrix, Ps, qs, Rcos, poc_cs, Ps_first, Rcos_first, p_oc_os_first);
		PsAndQs_time += tmr.nsecsElapsed();

		long firstPoseIndex = getPoseIndexFromImageIndex(cameraPoses, firstImageIndex);
		struct pose_t firstPose = cameraPoses.at(firstPoseIndex);
		struct pose_t lastPose = cameraPoses.at(firstPoseIndex + coords.size() - 1);

		double baseline = norm(lastPose.p_co_o - firstPose.p_co_o);
		if (baseline < minBaseline) {
			track->setTooShortBaseline(true);
			continue;
		}

		tmr.restart();
		vec3 p_fo_o_est = triangulate_est(Ps, qs);
		vec3 p_fo_o_est_first = triangulate_est(Ps_first, qs);
		triang_time += tmr.nsecsElapsed();
		if (p_fo_o_est(0) == datum::inf) {
			track->setOutlier(true);
			continue;
		}

		// Merge with previous 3d point estimate.
		// TODO: We might be able to get better estimates by doing triangulation over
		// the full track length. For that, we need to save rotations and translations
		// for the entire track.
		//const int numOldEstimates = track->nFrames() / maxTrackLength - 1;
		//if (numOldEstimates > 0) {
			//cout << "We have " << numOldEstimates << " old estimates: " << track->getCoord3d().t() << ". Merging with " << p_fo_o_est.t() << endl;
		//	p_fo_o_est = (track->getCoord3d()*numOldEstimates + p_fo_o_est)/(numOldEstimates + 1);
		//}

		vec3 p_c0o_o = firstPose.p_co_o;
		mat33 Roc0 = firstPose.Rco.t();

		vec3 p_c0o_o_first = firstPose.p_co_o_first;
		mat33 Roc0_first = firstPose.Rco_first.t();

		vec3 p_fc0_c0 = Roc0.t()*(p_fo_o_est - p_c0o_o);
		vec3 p_fc0_c0_first = Roc0_first.t()*(p_fo_o_est_first - p_c0o_o_first);

		if (baseline*minBaselineMult < norm(p_fc0_c0)) {
			track->setTooShortBaseline(true);
			continue;
		}

		// Sanity checking
		double x = p_fc0_c0(0);
		double y = p_fc0_c0(1);
		double z = p_fc0_c0(2);

        if (z < minDistance) {
            track->setOutlier(true);
            continue;
        }
        if (z > maxDistance) {
            track->setOutlier(true);
            continue;
        }

        double pixelcoordx = x/z*cameraMatrix(0,0) + cameraMatrix(0,2);
        double pixelcoordy = y/z*cameraMatrix(1,1) + cameraMatrix(1,2);

        if ((pixelcoordx < 0) || (pixelcoordx > 1920)) {
            track->setOutlier(true);
            continue;
        }
        if ((pixelcoordy < 0) || (pixelcoordy > 1440)) {
            track->setOutlier(true);
            continue;
        }
        vec3 p_fo_o = Roc0*p_fc0_c0 + p_c0o_o;
        vec3 p_fo_o_first = Roc0_first*p_fc0_c0_first + p_c0o_o_first;

		track->setCoord3d(p_fo_o);

		mat rj;
		sp_mat Hxj;
		mat Hfj;
		tmr.restart();
		//calc_residual_firstestimates(Rcos, poc_cs, Rcos_first, p_oc_os_first, p_fo_o, p_fo_o_first, coords, firstPoseIndex, numPoses, cameraMatrix, rj, Hxj, Hfj);
		calc_residual(Rcos, poc_cs, Rcos_first, p_oc_os_first, p_fo_o, coords, firstPoseIndex, numPoses, cameraMatrix, rj, Hxj, Hfj);
		residual_time += tmr.nsecsElapsed();

		mat* Hoj = new mat();
		mat* roj = new mat();
		mat Aj;

		tmr.restart();
		Aj = null(Hfj.t()).t();
		*roj = Aj*rj;
//		remove_feature_dependency(Hfj, Hxj, rj, *Hoj, *roj, Aj);
		featuredep_time += tmr.nsecsElapsed();


		// Parse Hoj as Hxj and Aj to keep sparsity of Hxj intact.
		tmr.restart();

		//code to save files for testing is_inlier
		//Hxj.save("../resources/JDS/cuda/inlier/testcase/Hxj" + std::to_string(counter) + ".bin");
		//Aj.save("../resources/JDS/cuda/inlier/testcase/Aj" + std::to_string(counter) + ".bin");
		//P.save("../resources/JDS/cuda/inlier/testcase/P" + std::to_string(counter) + ".bin");
		//roj->save("../resources/JDS/cuda/inlier/testcase/roj" + std::to_string(counter) + ".bin");
		//ofstream myfile1 ("../resources/JDS/cuda/inlier/testcase/trackLength" + std::to_string(counter) + ".bin");
		//if (myfile1.is_open()){myfile1 <<  coords.size();        	
		//myfile1.close();}   
		//ofstream myfile2 ("../resources/JDS/cuda/inlier/testcase/imageNoiseVar" + std::to_string(counter) + ".bin");
		//if (myfile2.is_open()){myfile2 << imageNoiseVar;        	
		//myfile2.close();}
		//ofstream myfile3 ("../resources/JDS/cuda/inlier/testcase/chiMult" + std::to_string(counter) + ".bin");
		//if (myfile3.is_open()){myfile3 << chiMult;        	
		//myfile3.close();} 
		
		bool inlier = is_inlier(Hxj, Aj, P, *roj, coords.size(), imageNoiseVar, chiMult);
		
		//ofstream myfile4 ("../resources/JDS/cuda/inlier/testcase/inlier" + std::to_string(counter) + ".bin");
		//if (myfile4.is_open()){myfile4 << std::fixed << inlier;        	
		//myfile4.close();} 	

		outlier_time += tmr.nsecsElapsed();
		counter++;
		
			
			if (inlier) {
//			if (track->nFrames() < 25) {
//				cout << "Track " << track->getID() << " coord " << track->getCoord3d()(0) << "," << track->getCoord3d()(1) << "," << track->getCoord3d()(2) << endl;
//				cout << "At Rco:" << Rcos.at(0) << endl;
//				cout << "At poc_c:" << poc_cs.at(0) << endl;
//				cout << "P:" << Ps.at(0) << endl;
//				cout << "2d point:" << track->getCoordAt(0) << endl;
//				cout << "camera Matrix: " << cameraMatrix << endl;
//				if (track->getID() == 3001)
//					abort();
//			}

			*Hoj = Aj*Hxj; // TODO: can we do custom matrix multiplication to gain speed here?
			// We use Aj*Hxj in the inlier detection algorithm. Does it make sense to use the result
			// we get there? It changes the order of the matrix multiplication a bit.
			Hojs.push_back(Hoj);
			rojs.push_back(roj);
		} else {
			track->setOutlier(true);
			continue;
		}
	}

	if (rojs.size() > 0) {
		// Stack the matrices
		unsigned long rows = 0;
		for (const mat* roj : rojs)
			rows += roj->n_rows;

		mat ro = mat(rows, 1);
		mat Ho = mat(rows, Hojs.at(0)->n_cols);

		unsigned long ro_row_ptr = 0;
		for (const mat* roj : rojs) {
			ro.rows(ro_row_ptr, ro_row_ptr + roj->n_rows - 1) = *roj;
			ro_row_ptr += roj->n_rows;
			delete roj;
		}
		unsigned long Ho_row_ptr = 0;
		for (const mat* Hoj : Hojs) {
			Ho.rows(Ho_row_ptr, Ho_row_ptr + Hoj->n_rows - 1) = *Hoj;
			Ho_row_ptr += Hoj->n_rows;
			delete Hoj;
		}

		mat dstates;
		tmr.restart();
		mat dstates2;
		mat P2;

//		Ho.save("Ho" + std::to_string(counter) + ".bin");
//		ro.save("ro" + std::to_string(counter) + ".bin");
//		P.save("Pin" + std::to_string(counter) + ".bin");
		calculate_correction(Ho, ro, P, imageNoiseVar, dstates, P);
		//dstates.save("dstates" + std::to_string(counter) + ".bin");
//		P.save("Pout" + std::to_string(counter) + ".bin");

		//magma_calculate_correction(Ho, ro, P, imageNoiseVar, dstates2, P2);

		//cout << "P difference: " << norm(P - P2, "fro") << endl;
		//cout << "dstates diff: " << norm(dstates - dstates2, "fro") << endl;

		correction_time += tmr.nsecsElapsed();

		correct_states(dstates, numPoses, *states, cameraPoses);
	}

	total_time += tmr2.nsecsElapsed();

	for (CTrack2D* potentialNewEkfTrack : potentialNewEkfTracks) {
		if (numEkfTracks < maxEkfTracks) {
			potentialNewEkfTrack->setEkfTracked(true);
			numEkfTracks++;
		}
	}
	nvtxRangePop();

	cout << "Update time - psandqs: " << PsAndQs_time/1000000 << " (" << (PsAndQs_time*100)/total_time << "%)" <<
			" tri: " << triang_time/1000000 << " (" << (triang_time*100)/total_time << "%)" <<
			" res: " << residual_time/1000000 << " (" << (residual_time*100)/total_time << "%)" <<
			" fd: " << featuredep_time/1000000 << " (" << (featuredep_time*100)/total_time << "%)" <<
			" out: " << outlier_time/1000000 << " (" << (outlier_time*100)/total_time << "%)" <<
			" cor: " << correction_time/1000000 << " (" << (correction_time*100)/total_time << "%)" <<
			" tot: " << total_time/1000000 << endl;
}
