import sys
import os
import time

sys.path.append(os.path.join(os.path.dirname(__file__), "../triangulate"))
sys.path.append(os.path.join(os.path.dirname(__file__), "../camposehandler"))
sys.path.append(os.path.join(os.path.dirname(__file__), "../featuretracker"))
sys.path.append(os.path.join(os.path.dirname(__file__), "../statehandler"))

from featuretracker import Track
from camposehandler import CamPoseHandler

import kalmanfilter_pywrapper
import kalmanfilter
from numpy import matrix, mat
from numpy import linalg, count_nonzero
from numpy.random import rand
import numpy
import random

def test_calcPsAndQs_pywrapper():
    for i in range(1000):
        #dx = 948.149
        #dy = 766.599
        #fx = 1099.99
        #fy = 1103.92
    
        #cameraMatrix = matrix([
        #[fx, 0,  dx],
        #[0,  fy, dy],
        #[0,  0,  1]])
        
        cameraMatrix = rand(3,3)
    
        cph = CamPoseHandler()
        #Rbo = matrix([
        #[1, 0, 0],
        #[0, 1, 0],
        #[0, 0, 1]])
        #p_bo_o = matrix([
        #[1],
        #[2],
        #[3]])
        for j in range(100):
            Rbo = rand(3,3)
            p_bo_o = rand(3,1)
            cph.appendPose(j, Rbo, p_bo_o)
            
        #Rbo = matrix([
        #[0, 0, 1],
        #[0, 1, 0],
        #[1, 0, 0]])
        #p_bo_o = matrix([
        #[4],
        #[5],
        #[6]])
        
        track = Track()
    
        track.id = 100
        track.firstImageIndex = random.randint(0, 98)
        for j in range(100 - track.firstImageIndex):
            track.coords.append((random.random()*100, random.random()*100))
    
        Ps_pywrapper, qs_pywrapper, Rcos_pywrapper, p_oc_cs_pywrapper = kalmanfilter_pywrapper.calcPsAndQs_pywrapper(cph, track, cameraMatrix)

        Ps, qs, Rcos, p_oc_cs = kalmanfilter.calcPsAndQs(cph, track, cameraMatrix)

        for i in range(len(Ps)):
            assert (linalg.norm(Ps[i] - Ps_pywrapper[i]) < 10e-15)
            assert (linalg.norm(qs[i] - qs_pywrapper[i]) < 10e-15)
            assert (linalg.norm(Rcos[i] - Rcos_pywrapper[i]) < 10e-15)
            assert (linalg.norm(p_oc_cs[i] - mat(p_oc_cs_pywrapper[i]).T) < 10e-15)

def test_null_pywrapper():
    A = matrix([
        [2,3,5],
        [-4,2,3]
        ])  
    #	A = rand(10,10)
    nullspace_wrap = kalmanfilter_pywrapper.null_pywrapper(A)
    nullspace = kalmanfilter.null(A)
    if len(nullspace_wrap.shape) == 1:
        nullspace_wrap = numpy.asmatrix(nullspace_wrap).T
    else:
        nullspace_wrap = numpy.asmatrix(nullspace_wrap)
    print(nullspace)
    print(nullspace_wrap)
    assert (linalg.norm(nullspace - nullspace_wrap) < 10e-12)


test_null_pywrapper()

test_calcPsAndQs_pywrapper()

