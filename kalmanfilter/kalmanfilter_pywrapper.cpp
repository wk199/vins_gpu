#include <Python.h>
#define NO_IMPORT_ARRAY
#include "../util/numpy.h"

#include <vector>
#include <string.h>
#include <stdio.h>

#include "../camposehandler/camposehandler.h"
#include "../featuretracker/track2d.h"
#include "../featuretracker/conversions.h"
#include "../util/pywrapper.h"

#include "kalmanfilter.h"

using namespace std;

static mat33* convertCameraMatrix(PyObject* cameraMatrixArg_py)
{
	PyObject* cameraMatrix_py = PyArray_FROM_OTF(cameraMatrixArg_py, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
	if (cameraMatrix_py == NULL)
		return NULL;

    mat33* cameraMatrix = new mat33();

	pyArrayToArmadillo(*cameraMatrix, cameraMatrix_py);
	Py_DECREF(cameraMatrix_py);

	return cameraMatrix;
}

static TCameraPoses* cameraPosesPyToC(PyObject* cameraPoses_py)
{
    TCameraPoses* cameraPoses = new TCameraPoses();

    const unsigned int numPoses = PyList_Size(cameraPoses_py);
	//cout << "number of poses " << numPoses << endl;
    for (unsigned int i = 0; i < numPoses; i++) {
    	struct pose_t pose;

    	//cout << "pose number " << i << endl;

    	PyObject* pose_py = PyList_GET_ITEM(cameraPoses_py, i);

        pose.imageIndex = PyLong_AsLong(PyObject_GetAttrString(pose_py, "imageIndex"));

        //cout << "Image index: " << pose.imageIndex << endl;

    	// Convert Rco to armadillo matrix
    	PyObject* Rco_py = PyArray_FROM_OTF(PyObject_GetAttrString(pose_py, "Rco"), NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    	if (Rco_py == NULL) {
    		delete cameraPoses;
    		return NULL;
    	}

    	pyArrayToArmadillo(pose.Rco, Rco_py);
    	Py_DECREF(Rco_py);
    	//cout << "Rco:" << endl;
    	//cout << pose.Rco;

    	// Convert p_co_o to armadillo matrix
    	PyObject* p_co_o_py = PyArray_FROM_OTF(PyObject_GetAttrString(pose_py, "p_co_o"), NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    	if (p_co_o_py == NULL) {
    		delete cameraPoses;
    		return NULL;
    	}

    	pyArrayToArmadillo(pose.p_co_o, p_co_o_py);
    	Py_DECREF(p_co_o_py);
    	//cout << "p_co_o:" << endl;
    	//cout << pose.p_co_o;

    	// Convert Rco to armadillo matrix
    	PyObject* Rco_first_py = PyArray_FROM_OTF(PyObject_GetAttrString(pose_py, "Rco_first"), NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    	if (Rco_first_py == NULL) {
    		delete cameraPoses;
    		return NULL;
    	}

    	pyArrayToArmadillo(pose.Rco_first, Rco_first_py);
    	Py_DECREF(Rco_first_py);
    	//cout << "Rco:" << endl;
    	//cout << pose.Rco;

    	// Convert p_co_o to armadillo matrix
    	PyObject* p_co_o_first_py = PyArray_FROM_OTF(PyObject_GetAttrString(pose_py, "p_co_o_first"), NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    	if (p_co_o_first_py == NULL) {
    		delete cameraPoses;
    		return NULL;
    	}

    	pyArrayToArmadillo(pose.p_co_o_first, p_co_o_first_py);
    	Py_DECREF(p_co_o_first_py);


    	cameraPoses->push_back(pose);
    }

    return cameraPoses;
}

PyObject* updateCameraPoses(const TCameraPoses& cameraPoses, PyObject* cameraPoses_py)
{
	for (unsigned long i = 0; i < cameraPoses.size(); i++) {
		PyObject* pose_py = PyList_GetItem(cameraPoses_py, i);
		if (pose_py == NULL)
			return NULL;

	    PyObject_SetAttrString(pose_py, "Rco", armadilloToPyArray(cameraPoses.at(i).Rco));
	    PyObject_SetAttrString(pose_py, "p_co_o", armadilloToPyArray(cameraPoses.at(i).p_co_o));
	    PyObject_SetAttrString(pose_py, "Rco_first", armadilloToPyArray(cameraPoses.at(i).Rco_first));
	    PyObject_SetAttrString(pose_py, "p_co_o_first", armadilloToPyArray(cameraPoses.at(i).p_co_o_first));
	}

	return cameraPoses_py;
}


static PyObject* null_pywrapper(PyObject *self, PyObject *args)
{
    PyObject *A_pyarg = NULL;

    //cout << "entering null_pywrapper" << endl;

    if (!PyArg_ParseTuple(args, "O", &A_pyarg))
        return NULL;

    mat* A = armaMatFromPyObject(A_pyarg);
    if (A == NULL)
    	return NULL;

	mat nullspace;
	nullspace = null(*A).t();
	delete A;

	PyObject* nullspace_py = armadilloToPyArray(nullspace);

    return Py_BuildValue("N", nullspace_py);
}

static PyObject* remove_feature_dependency_pywrapper(PyObject *self, PyObject *args)
{
    PyObject *Hfj_pyarg = NULL;
    PyObject *Hxj_pyarg = NULL;
    PyObject *rj_pyarg = NULL;

    //cout << "entering remove_feature_dependency_pywrapper" << endl;

    if (!PyArg_ParseTuple(args, "OOO", &Hfj_pyarg, &Hxj_pyarg, &rj_pyarg))
        return NULL;

    //cout << "Hfj" << endl;

    // *** Convert Hfj ***
	PyObject* Hfj_py = PyArray_FROM_OTF(Hfj_pyarg, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
	if (Hfj_py == NULL)
		return NULL;

	mat Hfj;
	pyArrayToArmadillo(Hfj, Hfj_py);
	Py_DECREF(Hfj_py);

    //cout << "Hxj" << endl;
    // *** Convert Hxj ***
	PyObject* Hxj_py = PyArray_FROM_OTF(Hxj_pyarg, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
	if (Hxj_py == NULL)
		return NULL;

	mat Hxj;
	pyArrayToArmadillo(Hxj, Hxj_py);
	Py_DECREF(Hxj_py);

	//cout << "rj" << endl;
    // *** Convert rj ***
	PyObject* rj_py = PyArray_FROM_OTF(rj_pyarg, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
	if (rj_py == NULL)
		return NULL;

	mat rj;
	pyArrayToArmadillo(rj, rj_py);
	Py_DECREF(rj_py);

	//cout << "call function" << endl;

	sp_mat Hxj_sparse = sp_mat(Hxj);
	mat Hoj;
	mat roj;
	mat Aj;
	remove_feature_dependency(Hfj, Hxj_sparse, rj, Hoj, roj, Aj);

	PyObject* Hoj_py = armadilloToPyArray(Hoj);
	PyObject* roj_py = armadilloToPyArray(roj);
	PyObject* Aj_py = armadilloToPyArray(Aj);

    return Py_BuildValue("NNN", Hoj_py, roj_py, Aj_py);
}

static PyObject* calculate_correction_pywrapper(PyObject *self, PyObject *args)
{
    PyObject *Ho_pyarg = NULL;
    PyObject *ro_pyarg = NULL;
    PyObject *P_pyarg = NULL;
    double imageNoiseVar;


    if (!PyArg_ParseTuple(args, "OOOd", &Ho_pyarg, &ro_pyarg, &P_pyarg, &imageNoiseVar))
        return NULL;

    //cout << "Hoj" << endl;

    // *** Convert Ho ***
	PyObject* Ho_py = PyArray_FROM_OTF(Ho_pyarg, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
	if (Ho_py == NULL)
		return NULL;

	mat Ho;
	pyArrayToArmadillo(Ho, Ho_py);
	Py_DECREF(Ho_py);

    //cout << "ro" << endl;
    // *** Convert ro ***
	PyObject* ro_py = PyArray_FROM_OTF(ro_pyarg, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
	if (ro_py == NULL)
		return NULL;

	mat ro;
	pyArrayToArmadillo(ro, ro_py);
	Py_DECREF(ro_py);

	//cout << "P" << endl;
    // *** Convert P ***
	PyObject* P_py = PyArray_FROM_OTF(P_pyarg, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
	if (P_py == NULL)
		return NULL;

	mat P;
	pyArrayToArmadillo(P, P_py);
	Py_DECREF(P_py);

	//cout << "call function" << endl;

	mat dstates;
	mat Pout;
	calculate_correction(Ho, ro, P, imageNoiseVar, dstates, Pout);

	PyObject* Pout_py = armadilloToPyArray(Pout);
	PyObject* dstates_py = armadilloToPyArray(dstates);

    return Py_BuildValue("NN", dstates_py, Pout_py);
}

static PyObject* is_inlier_pywrapper(PyObject *self, PyObject *args)
{
    PyObject *Hxj_pyarg = NULL;
    PyObject *Aj_pyarg = NULL;
    PyObject *P_pyarg = NULL;
    PyObject *roj_pyarg = NULL;
    unsigned long trackLength;
    double imageNoiseVar;

    //cout << "entering is_inlier_pywrapper" << endl;

    if (!PyArg_ParseTuple(args, "OOOOld", &Hxj_pyarg, &Aj_pyarg, &P_pyarg, &roj_pyarg, &trackLength, &imageNoiseVar))
        return NULL;

    //cout << "Hxj" << endl;

    // *** Convert Hxj ***
	PyObject* Hxj_py = PyArray_FROM_OTF(Hxj_pyarg, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
	if (Hxj_py == NULL)
		return NULL;

	mat Hxj;
    //cout << "To armadillo" << endl;
	pyArrayToArmadillo(Hxj, Hxj_py);
    //cout << "Done" << endl;
	Py_DECREF(Hxj_py);

    // *** Convert Aj ***
	PyObject* Aj_py = PyArray_FROM_OTF(Aj_pyarg, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
	if (Aj_py == NULL)
		return NULL;

	mat Aj;
    //cout << "To armadillo" << endl;
	pyArrayToArmadillo(Aj, Aj_py);
    //cout << "Done" << endl;
	Py_DECREF(Aj_py);

    //cout << "P" << endl;
    // *** Convert P ***
	PyObject* P_py = PyArray_FROM_OTF(P_pyarg, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
	if (P_py == NULL)
		return NULL;

	mat P;
	pyArrayToArmadillo(P, P_py);
	Py_DECREF(P_py);

    //cout << "roj" << endl;
    // *** Convert roj ***
	PyObject* roj_py = PyArray_FROM_OTF(roj_pyarg, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
	if (roj_py == NULL)
		return NULL;

	mat roj;
	pyArrayToArmadillo(roj, roj_py);
	Py_DECREF(roj_py);

	//cout << "call function" << endl;

	//cout << "Hoj" << endl;
	//cout << Hoj << endl;

	//cout << "P" << endl;
	//cout << P << endl;

	//cout << "roj" << endl;
	//cout << roj << endl;

	//cout << "tracklength" << endl;
	//cout << trackLength << endl;

	//cout << "imageNoiseVar" << endl;
	//cout << imageNoiseVar << endl;

	sp_mat Hxj_sparse = sp_mat(Hxj);

	if (is_inlier(Hxj_sparse, Aj, P, roj, trackLength, imageNoiseVar)) {
		Py_INCREF(Py_True);
		return Py_True;
	} else {
		Py_INCREF(Py_False);
		return Py_False;
	}
}

static PyObject* calc_residual_pywrapper(PyObject *self, PyObject *args)
{
	PyObject* Rcos_py;
	PyObject* p_oc_cs_py;
	PyObject* Rcos_first_py;
	PyObject* p_oc_os_first_py;
	PyObject* p_fo_o_py_arg;
	PyObject* track_py;
	unsigned long firstPoseIndex;
	unsigned long numPoses;
	PyObject* cameraMatrix_py_arg;

    //cout << "entering calc_residual_pywrapper" << endl;

    if (!PyArg_ParseTuple(args, "O!O!O!O!OOllO", &PyList_Type, &Rcos_py, &PyList_Type, &p_oc_cs_py, &PyList_Type, &Rcos_first_py, &PyList_Type, &p_oc_os_first_py, &p_fo_o_py_arg, &track_py, &firstPoseIndex, &numPoses, &cameraMatrix_py_arg))
        return NULL;

    vector<mat33> Rcos;

    const unsigned int numRcos = PyList_Size(Rcos_py);
	//cout << "number of Rcos " << numRcos << endl;
    for (unsigned int i = 0; i < numRcos; i++) {
    	// Convert Rco to armadillo matrix
    	PyObject* Rco_py = PyArray_FROM_OTF(PyList_GetItem(Rcos_py, i), NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    	if (Rco_py == NULL)
    		return NULL;

    	mat33 Rco;
    	pyArrayToArmadillo(Rco, Rco_py);
    	Py_DECREF(Rco_py);

    	Rcos.push_back(Rco);
    }

    // *** Convert p_oc_cs ***
    vector<vec3> p_oc_cs;

    const unsigned int nump_oc_cs = PyList_Size(p_oc_cs_py);
	//cout << "number of p_oc_cs " << nump_oc_cs << endl;
    for (unsigned int i = 0; i < nump_oc_cs; i++) {
    	// Convert p_oc_c to armadillo matrix
    	PyObject* p_oc_c_py = PyArray_FROM_OTF(PyList_GetItem(p_oc_cs_py, i), NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    	if (p_oc_c_py == NULL)
    		return NULL;

    	vec3 p_oc_c;
    	pyArrayToArmadillo(p_oc_c, p_oc_c_py);
    	Py_DECREF(p_oc_c_py);

    	p_oc_cs.push_back(p_oc_c);
    }

    vector<mat33> Rcos_first;

    const unsigned int numRcos_first = PyList_Size(Rcos_first_py);
	//cout << "number of Rcos_first " << numRcos_first << endl;
    for (unsigned int i = 0; i < numRcos_first; i++) {
    	// Convert Rco to armadillo matrix
    	PyObject* Rco_first_py = PyArray_FROM_OTF(PyList_GetItem(Rcos_first_py, i), NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    	if (Rco_first_py == NULL)
    		return NULL;

    	mat33 Rco_first;
    	pyArrayToArmadillo(Rco_first, Rco_first_py);
    	Py_DECREF(Rco_first_py);

    	Rcos_first.push_back(Rco_first);
    }

    // *** Convert p_oc_cs ***
    vector<vec3> p_oc_os_first;

    const unsigned int nump_oc_os_first = PyList_Size(p_oc_os_first_py);
	//cout << "number of p_oc_os_first " << nump_oc_os_first << endl;
    for (unsigned int i = 0; i < nump_oc_os_first; i++) {
    	// Convert p_oc_c to armadillo matrix
    	PyObject* p_oc_o_first_py = PyArray_FROM_OTF(PyList_GetItem(p_oc_os_first_py, i), NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    	if (p_oc_o_first_py == NULL)
    		return NULL;

    	vec3 p_oc_o_first;
    	pyArrayToArmadillo(p_oc_o_first, p_oc_o_first_py);
    	Py_DECREF(p_oc_o_first_py);

    	p_oc_os_first.push_back(p_oc_o_first);
    }

    //cout << "converting p_fo_o" << endl;

    // *** Convert p_fo_o ***
	PyObject* p_fo_o_py = PyArray_FROM_OTF(p_fo_o_py_arg, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
	if (p_fo_o_py == NULL)
		return NULL;

	vec3 p_fo_o;
	pyArrayToArmadillo(p_fo_o, p_fo_o_py);
	Py_DECREF(p_fo_o_py);

    //cout << "converting track" << endl;

    // *** Convert track ***
	CTrack2D* track = trackPyToC(track_py);
	if (track == NULL)
		return NULL;

	// *** Convert cameraMatrix ***
	mat33* cameraMatrix = convertCameraMatrix(cameraMatrix_py_arg);
	if (cameraMatrix == NULL)
		return NULL;

	//cout << "calculating residual" << endl;
	mat rj;
	sp_mat Hxj;
	mat Hfj;
	calc_residual(Rcos, p_oc_cs, p_fo_o, *track, firstPoseIndex, numPoses, *cameraMatrix, rj, Hxj, Hfj);
	delete track;
	delete cameraMatrix;

	//cout << "Convetring values" <<endl;

	mat Hxj_dense = mat(Hxj);

	PyObject* rj_py = armadilloToPyArray(rj);
	PyObject* Hxj_py = armadilloToPyArray(Hxj_dense);
	PyObject* Hfj_py = armadilloToPyArray(Hfj);

	//cout << "rj" << endl;
	//cout << rj << endl;

    return Py_BuildValue("NNN", rj_py, Hxj_py, Hfj_py);
}

static PyObject* calcPsAndQs_pywrapper(PyObject *self, PyObject *args)
{
    PyObject *cameraPoses_py = NULL;
    PyObject *track_py = NULL;
    PyObject *cameraMatrixArg_py = NULL;

    //cout << "entering calcPsAndQs" << endl;

    if (!PyArg_ParseTuple(args, "O!OO", &PyList_Type, &cameraPoses_py, &track_py, &cameraMatrixArg_py))
        return NULL;

    //cout << "Parsed args" << endl;

    // *** Convert camposehandler ***
    TCameraPoses* cameraPoses = cameraPosesPyToC(cameraPoses_py);

    // *** Convert track ***
    CTrack2D* track = trackPyToC(track_py);

    // *** Convert camera matrix ***
    mat33* cameraMatrix = convertCameraMatrix(cameraMatrixArg_py);
	//cout << "camera matrix " << cameraMatrix << endl;

	//cout << "Running function... " << endl;
    // Run actual function
	vector<mat34> Ps;
	vector<vec2> qs;
	vector<mat33>Rcos;
	vector<vec3>p_oc_cs;
	vector<mat33>Rcos_first;
	vector<vec3>p_oc_os_first;
    calcPsAndQs(*cameraPoses, *track, *cameraMatrix, Ps, qs, Rcos, p_oc_cs, Rcos_first, p_oc_os_first);
    delete track;
    delete cameraMatrix;
    delete cameraPoses;
	//cout << "Done... " << endl;

	//cout << qs.size() << endl;

	//cout << "New lists... " << endl;
    // Convert results to python lists
    unsigned long numCoords = qs.size();
    PyObject* Ps_py = PyList_New(numCoords);
    PyObject* qs_py = PyList_New(numCoords);
    PyObject* Rcos_py = PyList_New(numCoords);
    PyObject* p_oc_cs_py = PyList_New(numCoords);
    PyObject* Rcos_first_py = PyList_New(numCoords);
    PyObject* p_oc_os_first_py = PyList_New(numCoords);

    for (unsigned long i = 0; i < numCoords; i++) {
    	//cout << "Loop number " << i << endl;

    	//cout << "P_py " << endl;
    	PyObject* P_py = armadilloToPyArray(Ps[i]);
    	//cout << "q_py " << qs[i] << endl;
    	PyObject* q_py = armadilloToPyArray(qs[i]);
    	//cout << "Rco_py " << endl;
    	PyObject* Rco_py = armadilloToPyArray(Rcos[i]);
    	//cout << "p_oc_c_py " << endl;
    	PyObject* p_oc_c_py = armadilloToPyArray(p_oc_cs[i]);

    	PyObject* Rco_first_py = armadilloToPyArray(Rcos_first[i]);
    	//cout << "p_oc_c_py " << endl;
    	PyObject* p_oc_o_first_py = armadilloToPyArray(p_oc_os_first[i]);

    	//cout << "Setting items " << endl;
    	PyList_SetItem(Ps_py, i, P_py);
    	PyList_SetItem(qs_py, i, q_py);
    	PyList_SetItem(Rcos_py, i, Rco_py);
    	PyList_SetItem(p_oc_cs_py, i, p_oc_c_py);
    	PyList_SetItem(Rcos_first_py, i, Rco_first_py);
    	PyList_SetItem(p_oc_os_first_py, i, p_oc_o_first_py);
    }

	//cout << "Building values and returning... " << endl;

    return Py_BuildValue("NNNNNN", Ps_py, qs_py, Rcos_py, p_oc_cs_py, Rcos_first_py, p_oc_os_first_py);
}

States* statesPyToC(PyObject* states_py)
{
	States* states = new States();

    PyObject* states_abias_py = PyObject_GetAttrString(states_py, "abias");
    PyObject* states_gbias_py = PyObject_GetAttrString(states_py, "gbias");
    PyObject* states_p_bo_o_py = PyObject_GetAttrString(states_py, "p_bo_o");
    PyObject* states_Rbo_py = PyObject_GetAttrString(states_py, "Rbo");
    PyObject* states_v_bo_o_py = PyObject_GetAttrString(states_py, "v_bo_o");

    mat* Rbo = armaMatFromPyObject(states_Rbo_py);
    if (Rbo == NULL)
    	return NULL;
    states->Rbo = *Rbo;
    delete Rbo;

    mat* abias = armaMatFromPyObject(states_abias_py);
    if (abias == NULL)
    	return NULL;
    states->abias = *abias;
    delete abias;

    mat* gbias = armaMatFromPyObject(states_gbias_py);
    if (gbias == NULL)
    	return NULL;
    states->gbias = *gbias;
    delete gbias;

    mat* p_bo_o = armaMatFromPyObject(states_p_bo_o_py);
    if (p_bo_o == NULL)
    	return NULL;
    states->p_bo_o = *p_bo_o;
    delete p_bo_o;

    mat* v_bo_o = armaMatFromPyObject(states_v_bo_o_py);
    if (v_bo_o == NULL)
    	return NULL;
    states->v_bo_o = *v_bo_o;
    delete v_bo_o;

    return states;
}

PyObject* updateStates(const States& states, PyObject* states_py)
{
    PyObject* states_Rbo_py = armadilloToPyArray(states.Rbo);
    PyObject* states_abias_py = armadilloToPyArray(states.abias);
    PyObject* states_gbias_py = armadilloToPyArray(states.gbias);
    PyObject* states_p_bo_o_py = armadilloToPyArray(states.p_bo_o);
    PyObject* states_v_bo_o_py = armadilloToPyArray(states.v_bo_o);

    // FIXME: Does this remove the reference to the old object?
    if (PyObject_SetAttrString(states_py, "Rbo", states_Rbo_py ) == -1)
    	return NULL;
    if (PyObject_SetAttrString(states_py, "abias", states_abias_py) == -1)
    	return NULL;
    if (PyObject_SetAttrString(states_py, "gbias", states_gbias_py) == -1)
    	return NULL;
	if (PyObject_SetAttrString(states_py, "p_bo_o", states_p_bo_o_py) == -1)
		return NULL;
	if (PyObject_SetAttrString(states_py, "v_bo_o", states_v_bo_o_py) == -1)
		return NULL;

    return states_py;
}

static PyObject* correct_states_pywrapper(PyObject *self, PyObject *args)
{
    PyObject *dstates_pyarg = NULL;
    unsigned long numPoses;
    PyObject *states_py = NULL;
    PyObject *cameraPoses_py = NULL;

    //cout << "correct_states_pywrapper" << endl;

    if (!PyArg_ParseTuple(args, "OlOO!", &dstates_pyarg, &numPoses, &states_py, &PyList_Type, &cameraPoses_py))
        return NULL;

    // Convert dstates
    mat* dstates = armaMatFromPyObject(dstates_pyarg);
    if (dstates == NULL)
    	return NULL;

    // Convert states
    States* states = statesPyToC(states_py);
    if (states == NULL)
    	return NULL;

    // Convert camera poses
    TCameraPoses* cameraPoses = cameraPosesPyToC(cameraPoses_py);
    if (cameraPoses == NULL) {
    	delete states;
    	return NULL;
    }

    //cout << "Running function" << endl;
    // Run the function
    correct_states(*dstates, numPoses, *states, *cameraPoses);

    //cout << "Converting to python" << endl;

    //cout << "Camera poses" << endl;
    cameraPoses_py = updateCameraPoses(*cameraPoses, cameraPoses_py);
    if (cameraPoses_py == NULL) {
    	delete states;
    	return NULL;
    }
    delete cameraPoses;

    //cout << "states" << endl;
    states_py = updateStates(*states, states_py);
    if (states_py == NULL)
    	return NULL;

    delete states;

    //cout << "Building return" << endl;

    PyObject* ret = Py_BuildValue("OO", states_py, cameraPoses_py);

    //cout << "Returning" << endl;

    return ret;
}

static PyObject* updateTracks(const Tracks2D& tracks, PyObject* tracks_py)
{
	for (unsigned long i = 0; i < tracks.size(); i++) {
		PyObject* track_py = PyList_GetItem(tracks_py, i);
		if (track_py == NULL)
			return NULL;

		if (tracks.at(i)->doDelete()) {
			Py_INCREF(Py_True);
			PyObject_SetAttrString(track_py, "delete", Py_True);
		} else {
			Py_INCREF(Py_False);
			PyObject_SetAttrString(track_py, "delete", Py_False);
		}

		if (tracks.at(i)->isOutlier()) {
			Py_INCREF(Py_True);
			PyObject_SetAttrString(track_py, "outlier", Py_True);
		} else {
			Py_INCREF(Py_False);
			PyObject_SetAttrString(track_py, "outlier", Py_False);
		}

		if (tracks.at(i)->isTooShort()) {
			Py_INCREF(Py_True);
			PyObject_SetAttrString(track_py, "tooShort", Py_True);
		} else {
			Py_INCREF(Py_False);
			PyObject_SetAttrString(track_py, "tooShort", Py_False);
		}

		if (tracks.at(i)->isTooShortBaseline()) {
			Py_INCREF(Py_True);
			PyObject_SetAttrString(track_py, "tooShortBaseline", Py_True);
		} else {
			Py_INCREF(Py_False);
			PyObject_SetAttrString(track_py, "tooShortBaseline", Py_False);
		}
	}

	return tracks_py;
}

static PyObject* update_pywrapper(PyObject *self, PyObject *args)
{
    PyObject *P_py = NULL;
    PyObject *states_py = NULL;
    PyObject *tracks_py = NULL;
    PyObject *cameraPoses_py = NULL;
    PyObject *cameraMatrix_py = NULL;
    double imageNoiseVar;
    unsigned long maxTrackLength;

    //cout << "parsing args" << endl;

    if (!PyArg_ParseTuple(args, "OOO!O!Odl", &P_py, &states_py, &PyList_Type, &tracks_py, &PyList_Type, &cameraPoses_py, &cameraMatrix_py, &imageNoiseVar, &maxTrackLength))
        return NULL;

    // Convert P
    //cout << "P" << endl;
    mat* P = armaMatFromPyObject(P_py);
    if (P == NULL)
    	return NULL;

    // Convert states
    //cout << "states" << endl;
    States* states = statesPyToC(states_py);
    if (states == NULL)
    	return NULL;

    //cout << "tracks" << endl;
    Tracks2D* tracks = tracksPyToC(tracks_py);
    if (tracks == NULL) {
    	delete states;
    	return NULL;
    }

    // Convert camera poses
    //cout << "cameraPoses" << endl;
    TCameraPoses* cameraPoses = cameraPosesPyToC(cameraPoses_py);
    if (cameraPoses == NULL) {
        for (unsigned long i = 0; i < tracks->size(); i++)
        	delete tracks->at(i);
        delete tracks;
    	delete states;
    	return NULL;
    }

    // Convert cameraMatrix
    //cout << "cameraMatrix" << endl;
    mat* cameraMatrix = armaMatFromPyObject(cameraMatrix_py);
    if (cameraMatrix == NULL) {
        for (unsigned long i = 0; i < tracks->size(); i++)
        	delete tracks->at(i);
        delete tracks;
    	delete states;
    	return NULL;
    }

    //cout << "CameraMatrix: " << endl;
    //cout << *cameraMatrix << endl;

    //cout << "Running function" << endl;
    // Run the function
    vector<vec3> points3d;
    update(*P, states, *tracks, *cameraPoses, points3d, *cameraMatrix, imageNoiseVar, maxTrackLength);

    //cout << "Converting to python" << endl;

    //cout << "Camera poses" << endl;
    cameraPoses_py = updateCameraPoses(*cameraPoses, cameraPoses_py);
    if (cameraPoses_py == NULL)
    	return NULL;
    delete cameraPoses;

    //cout << "states" << endl;
    states_py = updateStates(*states, states_py);
    if (states_py == NULL)
    	return NULL;
    delete states;

    //cout << "tracks" << endl;
    tracks_py = updateTracks(*tracks, tracks_py);
    for (unsigned long i = 0; i < tracks->size(); i++)
    	delete tracks->at(i);
    delete tracks;
    //cout << "Building return" << endl;

    PyObject* Pout_py = armadilloToPyArray(*P);
    delete P;

    PyObject* ret = Py_BuildValue("NOOO", Pout_py, states_py, tracks_py, cameraPoses_py);

    //cout << "Returning" << endl;

    return ret;
}

static PyObject* augmentState_pywrapper(PyObject *self, PyObject *args)
{
    PyObject *P_py = NULL;
    PyObject *states_py = NULL;

    if (!PyArg_ParseTuple(args, "OO", &P_py, &states_py))
        return NULL;

    // Convert P
    mat* P = armaMatFromPyObject(P_py);
    if (P == NULL)
    	return NULL;

    // Convert states
    States* states = statesPyToC(states_py);
    if (states == NULL)
    	return NULL;

    //cout << "Running function" << endl;
    // Run the function
    augmentState(*states, *P);
    delete states;

    //cout << "Converting to python" << endl;
    PyObject* Pout_py = armadilloToPyArray(*P);
    delete P;

    return Pout_py;
}

static PyObject* covarianceUpdate_pywrapper(PyObject *self, PyObject *args)
{
    PyObject *P_py = NULL;
    PyObject *states_py = NULL;
    PyObject *wibb_py = NULL;
    PyObject *ab_py = NULL;
    PyObject *wieo_py = NULL;

    if (!PyArg_ParseTuple(args, "OOOOO", &P_py, &states_py, &wibb_py, &ab_py, &wieo_py))
        return NULL;

    // Convert P
    mat* P = armaMatFromPyObject(P_py);
    if (P == NULL)
    	return NULL;

    // Convert states
    States* states = statesPyToC(states_py);
    if (states == NULL)
    	return NULL;

    // Convert wibb
    mat* wibb = armaMatFromPyObject(wibb_py);
    if (wibb == NULL) {
        delete states;
    	return NULL;
    }

    // Convert ab
    mat* ab = armaMatFromPyObject(ab_py);
    if (ab == NULL)
    	return NULL;

    // Convert wieo
    mat* wieo = armaMatFromPyObject(wieo_py);
    if (wieo == NULL)
    	return NULL;

    //cout << "Running function" << endl;
    // Run the function
    covarianceUpdate(*states, *wibb, *ab, *wieo, *P);
    delete states;

    //cout << "Converting to python" << endl;
    PyObject* Pout_py = armadilloToPyArray(*P);
    delete P;

    return Pout_py;
}


///***PYTHON BOILERPLATE FROM HERE***

static struct PyMethodDef methods[] = {
	{"calcPsAndQs_pywrapper", calcPsAndQs_pywrapper, METH_VARARGS, "Calculate Ps and qs"},
	{"calc_residual_pywrapper", calc_residual_pywrapper, METH_VARARGS, "Calculate residuals"},
	{"is_inlier_pywrapper", is_inlier_pywrapper, METH_VARARGS, "Determine if point is inlier"},
	{"calculate_correction_pywrapper", calculate_correction_pywrapper, METH_VARARGS, "Calculate KF correction and new P"},
	{"remove_feature_dependency_pywrapper", remove_feature_dependency_pywrapper, METH_VARARGS, "Calculate KF correction and new P"},
	{"null_pywrapper", null_pywrapper, METH_VARARGS, "Calculate null-space"},
	{"correct_states_pywrapper", correct_states_pywrapper, METH_VARARGS, "Correct states"},
	{"update_pywrapper", update_pywrapper, METH_VARARGS, "Update kalman filter"},
	{"augmentState_pywrapper", augmentState_pywrapper, METH_VARARGS, "Augment state matrix"},
	{"covarianceUpdate_pywrapper", covarianceUpdate_pywrapper, METH_VARARGS, "Covariance update"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef PYMODULE_NAME_pywrapper_module = {
   PyModuleDef_HEAD_INIT,
   "kalmanfilter_pywrapper",   /* name of module */
   NULL, /* module documentation, may be NULL */
   -1,
   methods
};

PyMODINIT_FUNC
PyInit_kalmanfilter_pywrapper(void)
{
	init_pywrapper();

	PyObject* module = PyModule_Create(&PYMODULE_NAME_pywrapper_module);
	return module;
}
