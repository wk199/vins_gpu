#ifndef __KALMANFILTER_H
#define __KALMANFILTER_H

#include <vector>
#include <armadillo>

#include "../featuretracker/track2d.h"
#include "../camposehandler/camposehandler.h"
#include "../navigation/navigation.h"

using namespace std;
using namespace arma;

#ifndef mat34
typedef mat::fixed<3, 4> mat34;
#endif

void calcPsAndQs(
		const TCameraPoses& camPoses,
		const vector<vec2>& trackCoords,
		const unsigned long firstImageIndex,
		const mat33& cameraMatrix,
		vector<mat34>& Ps,
		vector<vec2>& qs,
		vector<mat33>& Rcos,
		vector<vec3>& p_oc_cs,
		vector<mat33>& Rcos_first,
		vector<vec3>& p_oc_os_first);

void calc_residual(
		const vector<mat33>& Rcos,
		const vector<vec3>& p_oc_cs,
		const vector<mat33>& Rcos_first,
		const vector<vec3>& p_oc_os_first,
		const vec3& p_fo_o,
		const CTrack2D& track,
		const unsigned long firstPoseIndex,
		const unsigned long numPoses,
		const mat33& cameraMatrix,
		mat& rj,
		sp_mat& Hxj,
		mat& Hfj
		);

void calc_residual_firstestimates(
		const vector<mat33>& Rcos,
		const vector<vec3>& p_oc_cs,
		const vector<mat33>& Rcos_first,
		const vector<vec3>& p_oc_os_first,
		const vec3& p_fo_o,
		const CTrack2D& track,
		const unsigned long firstPoseIndex,
		const unsigned long numPoses,
		const mat33& cameraMatrix,
		mat& rj,
		sp_mat& Hxj,
		mat& Hfj
		);

bool is_inlier(
		const sp_mat& Hxj,
		const mat& Aj,
		const mat& P,
		const mat& roj,
		const unsigned long trackLength,
		const double imageNoiseVar
		);

bool is_inlier_cpu(
		const sp_mat& Hxj,
		const mat& Aj,
		const mat& P,
		const mat& roj,
		const unsigned long trackLength,
		const double imageNoiseVar
		);

void calculate_correction(
		const mat& Ho,
		const mat& ro,
		const mat& P,
		const double imageNoiseVar,
		mat& dstates,
		mat& Pnew
		);

void calculate_correction_cpu(
		const mat& Ho,
		const mat& ro,
		const mat& P,
		const double imageNoiseVar,
		mat& dstates,
		mat& Pnew
		);

void remove_feature_dependency(
		const mat& Hfj,
		const sp_mat& Hxj,
		const mat& rj,
		mat& Hoj,
		mat& roj,
		mat& Aj);

void correct_states(
		const mat& dstates,
		const unsigned long numPoses,
		States& states,
		TCameraPoses& camPoses);

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
		const double chiMult);

void augmentState(
		const States& states,
		const vec3& p_ci_c,
		const mat33& Rci,
		mat& P);

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
		mat& P);

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
		mat& P);

void covarianceUpdate_firstestimates(
		const States& states,
		const States& oldstates,
		const vec3& go,
		const double imuTs,
		const double gNoiseVar,
		const double gBiasVar,
		const double aNoiseVar,
		const double aBiasVar,
		mat& P);

#endif
