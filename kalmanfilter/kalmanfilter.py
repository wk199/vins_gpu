from numpy import *
import numpy
import scipy
import triangulate_est
import triangulate

try:
    import triangulate_est_pywrapper
except ImportError:
    print("Unable to find triangulate_est C wrapper")

try:
    import statehandler_pywrapper
except ImportError:
    print("Unable to find statehandler C wrapper")

try:
    import kalmanfilter_pywrapper
except ImportError:
    print("Unable to find kalmanfilter C wrapper")

from statehandler import *
import camposehandler
from scipy.stats import chi2
import time
from copy import deepcopy

chitable = [-1, -1]
for i in range(2, 1000):
    chitable.append(chi2.ppf(0.95, 2*i - 3))

def skew(v):
    v0 = v.item(0,0)
    v1 = v.item(1,0)
    v2 = v.item(2,0)
    return matrix([
        [ 0, -v2, v1],
        [ v2, 0, -v0],
        [-v1, v0, 0]])

#@profile
def augmentState(P, states, unused=None):
    # Camera and body frame are equal
    Rci = mat(eye(3,3))
    p_ci_c = mat([0, 0, 0]).T
    Roc = states.Rbo.T
    p_ci_o = Roc*p_ci_c
    J = vstack((
        hstack((Rci,          mat(zeros((3,3))), mat(zeros((3,3))), mat(zeros((3,6))), mat(zeros((3, P.shape[0] - 15))))),
        hstack((skew(p_ci_o), mat(zeros((3,3))), mat(eye(3)),       mat(zeros((3,6))), mat(zeros((3, P.shape[0] - 15)))))
        ))
        
    J2 = vstack((
            mat(eye(P.shape[0])),
            J))
    P = J2*P*J2.T
    
    #PqV = P[:,0:3]
    #PpV = P[:,6:9]
    #PqH = P[0:3,:]
    #PpH = P[6:9,:]
    
    #Pqq = P[0:3,0:3]
    #Ppq = P[6:9,0:3]
    #Ppp = P[6:9,6:9]
    #Pqp = P[0:3,6:9]
    
    #P = hstack((P, PqV, PpV))
    
    #Pq = hstack((PqH, Pqq, Pqp))
    #Pp = hstack((PpH, Ppq, Ppp))
    
    #P = vstack((P, Pq, Pp))
    
    return P

def calculateNullspace(states, go, camPoses):
    Rbo = states.Rbo
    v_bo_o = states.v_bo_o
    p_bo_o = states.p_bo_o
    
    # Calculate robot null-space
    nullspace_robot = vstack((
        hstack((zeros((3,3)), Rbo*go)),
        hstack((zeros((3,3)), -skew(v_bo_o)*go)),
        hstack((zeros((3,3)), -skew(p_bo_o)*go)),
        zeros((3,4)),
        zeros((3,4))
    ))

    # Calculate camera null-space
    nullspace_poses = []
    for camPose in camPoses:
        if nullspace_poses == []:
            nullspace_poses = vstack((
            hstack((zeros((3,3)), camPose.Rbo_true*go)),
            hstack((eye(3,3), -skew(camPose.p_bo_o_true)*go))
            ))
        else :
            nullspace_poses = vstack((
                nullspace_poses,
                hstack((zeros((3,3)), camPose.Rbo_true*go)),
                hstack((eye(3,3), -skew(camPose.p_bo_o_true)*go))
            ))
        
    if len(nullspace_poses) == 0:
        return nullspace_robot
    else:
        return vstack((nullspace_robot, nullspace_poses))

def correctNullSpace(A, u, w):
    # TODO: nfo: Use solve instead
    return A - (A*u - w)*linalg.inv(u.T*u)*u.T

#@profile
def covarianceUpdate(P, states, wibb, ab, wieo, conf, nullspace, nullspace_new, oldstates, go, unused=None, unused2=None):
    global Q
    imuTs = 0.01
    
    Rbo = states.Rbo
    
    # Covariance propagation
    F = vstack((
         hstack((-skew(wibb),     zeros((3,3)), zeros((3,3)),  eye(3,3),      zeros((3,3)) )), #attitude
         hstack((-Rbo.T*skew(ab), zeros((3,3)), zeros((3,3)),   zeros((3,3)), -Rbo.T )), # velocity
         hstack(( zeros((3,3)),   eye(3),       zeros((3,3)),   zeros((3,3)),  zeros((3,3)) )), # position
         zeros((3,15)), # gbias
         zeros((3,15)) # abias
         ))

    # F = vstack((
    #     hstack(( zeros((3,3)),               zeros((3,3)), zeros((3,3)), -Rbo.T,         zeros((3,3)) )), #attitude
    #     hstack((-skew(Rbo.T*ab),             zeros((3,3)), zeros((3,3)),  zeros((3,3)), -Rbo.T )), # velocity
    #     hstack((-skew(Rbo.T*0.5*imuTs*ab),   eye(3),       zeros((3,3)),  zeros((3,3)), -0.5*imuTs*Rbo.T )), # position
    #     zeros((3,15)), # gbias
    #     zeros((3,15)) # abias
    #     ))

    if not 'Q' in globals():
        # according to report number 1
        #gnoisevar = 0.25/180*pi
        #gbiasvar = 1E-5
        #anoisevar = 9e-3
        #abiasvar = 1E-5
    
        # according to report number 2
        gnoisevar = conf.gnoisevar
        gbiasvar = conf.gbiasvar
        anoisevar = conf.anoisevar
        abiasvar = conf.abiasvar
    
        #gnoisevar = (1E-5)**2
        #gbiasvar = (1E-9)**2
        #anoisevar = (1E-5)**2
        #abiasvar = (1E-9)**2
        
        #From measurements
        #gnoisevar = 0.0065
        #anoisevar = 0.0087
        #gbiasvar = (1E-5)**2
        #abiasvar = (1E-5)**2
        

        Q = matrix([
            [gnoisevar, 0,         0,         0,                   0,                    0,                   0,                          0,                          0,                          0, 0, 0, 0, 0, 0],
            [0,         gnoisevar, 0,         0,                   0,                    0,                   0,                          0,                          0,                          0, 0, 0, 0, 0, 0],
            [0,         0,         gnoisevar, 0,                   0,                    0,                   0,                          0,                          0,                          0, 0, 0, 0, 0, 0],
            [0,         0,         0,         anoisevar,           0,                    0,                   0.5*imuTs*anoisevar,        0,                          0,                          0, 0, 0, 0, 0, 0],
            [0,         0,         0,         0,                   anoisevar,            0,                   0,                          0.5*imuTs*anoisevar,        0,                          0, 0, 0, 0, 0, 0],
            [0,         0,         0,         0,                   0,                    anoisevar,           0,                          0,                          0.5*imuTs*anoisevar,        0, 0, 0, 0, 0, 0],
            [0,         0,         0,         0.5*imuTs*anoisevar, 0,                    0,                   0.25*imuTs*imuTs*anoisevar, 0,                          0,                          0, 0, 0, 0, 0, 0],
            [0,         0,         0,         0,                   0.5*imuTs*anoisevar,  0,                   0,                          0.25*imuTs*imuTs*anoisevar, 0,                          0, 0, 0, 0, 0, 0],
            [0,         0,         0,         0,                   0,                    0.5*imuTs*anoisevar, 0,                          0,                          0.25*imuTs*imuTs*anoisevar, 0, 0, 0, 0, 0, 0],
            [0,         0,         0,         0,                   0,                    0,                   0,                          0,                          0,                          gbiasvar, 0, 0, 0, 0, 0],
            [0,         0,         0,         0,                   0,                    0,                   0,                          0,                          0,                          0, gbiasvar, 0, 0, 0, 0],
            [0,         0,         0,         0,                   0,                    0,                   0,                          0,                          0,                          0, 0, gbiasvar, 0, 0, 0],
            [0,         0,         0,         0,                   0,                    0,                   0,                          0,                          0,                          0, 0, 0, abiasvar, 0, 0],
            [0,         0,         0,         0,                   0,                    0,                   0,                          0,                          0,                          0, 0, 0, 0, abiasvar, 0],
            [0,         0,         0,         0,                   0,                    0,                   0,                          0,                          0,                          0, 0, 0, 0, 0, abiasvar]
            ])*imuTs
        #linalg.cholesky(Q) # Check for positive definiteness
    
        # Gc = vstack((
        #    hstack((eye(3),       zeros((3,3)), zeros((3,3)),    zeros((3,3)) )), # attitude
        #    hstack((zeros((3,3)), zeros((3,3)), Rbo.T,           zeros((3,3)) )), # velocite
        #    hstack((zeros((3,3)), zeros((3,3)), 0.5*imuTs*Rbo.T, zeros((3,3)) )), # position
        #    hstack((zeros((3,3)), eye(3),       zeros((3,3)),    zeros((3,3)) )), # gbias
        #    hstack((zeros((3,3)), zeros((3,3)), zeros((3,3)),    eye(3) ))        # abias
        #    ))

       ##   gnoise = matrix([gnoisevar, gnoisevar, gnoisevar]).T
        # gbiasrw = matrix([gbiasvar, gbiasvar, gbiasvar]).T
        # anoise = matrix([anoisevar, anoisevar, anoisevar]).T
        # abiasrw = matrix([abiasvar, abiasvar, abiasvar]).T
        # 
        # Qc = diag(vstack((gnoise, gbiasrw, anoise, abiasrw)).A.squeeze())
        # Q = Gc*Qc*Gc.T*imuTs
        # 
        # Q = Q + eye(len(Q))*10e-20 # Add a slight noise because of roundoff errors to keep matrix positive definite
    
    #A = vstack((
    #    hstack((-F, Gc*Qc*Gc.T)),
    #    hstack((zeros(F.shape), F.T))
    #    ))*imuTs
    #B = scipy.linalg.expm(A)
    #phi = B[15:,15:].T
    #Q = phi*B[15:,0:15]

    Phi = eye(15) + F*imuTs

    # Ensure observability properties
    nullspace_new_calculated = Phi*nullspace[0:15,:]

    Phi[0:3,0:3] = states.Rbo*oldstates.Rbo.T
    Phi[3:6,0:3] = correctNullSpace(Phi[3:6,0:3], oldstates.Rbo*go, skew(oldstates.v_bo_o - states.v_bo_o)*go)
    Phi[6:9,0:3] = correctNullSpace(Phi[6:9,0:3], oldstates.Rbo*go, skew(oldstates.v_bo_o*imuTs + oldstates.p_bo_o - states.p_bo_o)*go)
    # 
    nullspace_new_calculated_fixed = Phi*nullspace[0:15,:]
    

    #Nc = Gc*Qc*Gc.T
    #Q = imuTs/2*(Phi*Nc*Phi.T + Nc)

    
    PII  = P[0:15,0:15]
    PII = Phi*PII*Phi.T + Q
    
    if len(P) > 15:
        PIC  = P[0:15,15:]
        PIC = Phi*PIC
        PCC  = P[15:,15:]
    
        P = vstack((
            hstack((PII, PIC)),
            hstack((PIC.T, PCC))
            ))
    else:
        P = PII
    
    return P

def calcPsAndQs(camPoses, track, cameraMatrix):
    # Get the camera rotations and positions for this track
    qs = track.coords
    Rcos = []
    p_oc_cs = []
    Rcos_first = []
    p_oc_os_first = []
    Ps = []
    firstPoseIndex = camposehandler.getPoseIndexFromImageIndex(camPoses, track.firstImageIndex)
    #print("firstposeindex", firstPoseIndex)
    for i in range(firstPoseIndex, firstPoseIndex + len(track.coords)):
        pose = camPoses[i]
        # Calculate rotation from camera 0 to camera k
        Rco = pose.Rco
        # Calculate translation from camera 0 to camera k seen in the camera frame
        p_oc_c = -Rco*pose.p_co_o
        Rcos.append(Rco)
        p_oc_cs.append(p_oc_c)
        
        Ps.append(cameraMatrix*hstack((Rco, p_oc_c)))

        Rcos_first.append(pose.Rco_first)
        p_oc_os_first.append(-pose.p_co_o_first)
    
    assert len(qs) == len(Ps)
    
    return Ps, qs, Rcos, p_oc_cs, Rcos_first, p_oc_os_first

def calc_residual(Rcos, p_oc_cs, Rcos_first, p_oc_os_first, p_fo_o, track, firstPoseIndex, numPoses, cameraMatrix):
    
    rj = [] # Residual
    Hxj = [] # Measurement matrix
    Hfj = []
    
    dx = cameraMatrix[0,2]
    dy = cameraMatrix[1,2]
    fx = cameraMatrix[0,0]
    fy = cameraMatrix[1,1]
    
    # Now calculate the residual
    for k in range(firstPoseIndex, len(track.coords) + firstPoseIndex):
        Rco = Rcos[k - firstPoseIndex]
        p_oc_o = Rco.T*p_oc_cs[k - firstPoseIndex]
    
        p_fc_o = p_fo_o + p_oc_o
        p_fc_c = Rco*p_fc_o
        
        X = p_fc_c.item(0)
        Y = p_fc_c.item(1)
        Z = p_fc_c.item(2)
    
        zhat = matrix([[X/Z],[Y/Z]])

        J = matrix([[1/Z, 0, -X/(Z**2)],
                    [0, 1/Z, -Y/(Z**2)]])
                    
                
        # Hxjk = concatenate((zeros((2,15 + 6*k)), J*skew(p_fc_c), -J*Rco, zeros((2,6*(numPoses - k - 1)))), 1)
        # Hfjk = J*Rco
                    
        # Hxjk = concatenate((zeros((2,15 + 6*k)), J*Rco*skew(p_fc_o), -J*Rco, zeros((2,6*(numPoses - k - 1)))), 1)
        # Hfjk = J*Rco

        Hfjk = J*Rco
        Hxjk = hstack((zeros((2,15 + 6*k)), J*skew(p_fc_c), -J*(Rco), zeros((2,6*(numPoses - k - 1)))))

        # Ensure observability properties
        #A = J*hstack((Rco, skew(p_fc_c)))
        #u = vstack((
        #    Rco*go,
        #    skew(p_fc_o)*go));
        #Astar = correctNullspace(A, u, zeros(len(u),1))
        
        z = matrix([(track.coords[k - firstPoseIndex][0] - dx)/fx, (track.coords[k - firstPoseIndex][1] - dy)/fy]).T
        
        rjk = z - zhat
        
        if Hxj == []:
            Hxj = Hxjk
            Hfj = Hfjk
            rj = rjk
        else:
            Hxj = concatenate([Hxj, Hxjk])
            Hfj = concatenate([Hfj, Hfjk])
            rj = concatenate([rj, rjk])
    
    return rj, Hxj, Hfj

#@profile
def is_inlier(Hxj, Aj, P, roj, trackLength, imageNoiseVar):
    # Do outlier detection
    
    M = Aj*Hxj*P*Hxj.T*Aj.T
    chi = roj.T*linalg.inv(M + imageNoiseVar*eye(len(M)))*roj

    return chi < chitable[trackLength]*10
    

def calculate_correction(Ho, ro, P, imageNoiseVar):
    Q_1, R_1 = numpy.linalg.qr(Ho, mode='reduced')
        
    rn = Q_1.T*ro
        
    Rn = eye(len(Q_1.T))*imageNoiseVar

    # Calculate the kalman gain and do the aiding
    K = P*R_1.T*linalg.inv(R_1*P*R_1.T + Rn)

    M = (eye(len(P)) - K*R_1)
    P = M*P*M.T + K*Rn*K.T

    dstates = K*rn
    return dstates, P

def null(A, eps=1e-12):
    # Find the nullspace of the matrix
    u, s, vh = linalg.svd(A)
    padding = max(0, shape(A)[1] - shape(s)[0])
    null_mask = concatenate(((s <= eps), ones((padding,), dtype=bool)), axis=0)
    null_space = scipy.compress(null_mask, vh, axis=0)
    return null_space

def remove_feature_dependency(Hfj, Hxj, rj):
    Aj = null(Hfj.T)
    Hoj = Aj*Hxj
    roj = Aj*rj
    return Hoj, roj, Aj

def remove_feature_dependency_wrap(Hfj, Hxj, rj, test = False, useC = False):
    if not useC:
        return remove_feature_dependency(Hfj, Hxj, rj)
        
    if test:
        Hoj_copy, roj_copy, Aj_copy = remove_feature_dependency(Hfj, Hxj, rj)
        
    Hoj, roj, Aj = kalmanfilter_pywrapper.remove_feature_dependency_pywrapper(Hfj, Hxj, rj)
    
    Hoj = asmatrix(Hoj)
    roj = asmatrix(roj).T
    
    if len(Aj.shape) == 1:
        Aj = numpy.asmatrix(Aj).T
    else:
        Aj = numpy.asmatrix(Aj)
    
    if test:
        assert (linalg.norm(Aj - Aj_copy) < 10e-7)
        assert (linalg.norm(Hoj - Hoj_copy) < 10e-7)
        assert (linalg.norm(roj - roj_copy) < 10e-7)
    
    return Hoj, roj, Aj
        

def correct_states(dstates, numPoses, states, camPoses):
    states.abias = correct_vector(states.abias, get_dabias(dstates))
    states.gbias = correct_vector(states.gbias, get_dgbias(dstates))
    states.Rbo = correct_rotation(states.Rbo, get_dq_bo(dstates))
    states.v_bo_o = correct_vector(states.v_bo_o, get_dv_bo_o(dstates))
    states.p_bo_o = correct_vector(get_dp_bo_o(dstates), states.p_bo_o)

    for i in range(0, numPoses):
        pose = camPoses[i]
        pose.Rco = correct_rotation(pose.Rco, get_dq_co(dstates, i))
        pose.p_co_o = correct_vector(get_dp_co_o(dstates, i), pose.p_co_o)
        camPoses[i] = pose
    
    return states, camPoses

#@profile
def update(P, states, tracks, camPoses, cameraMatrix, imageNoiseVar, maxTrackLength, unused=None):
    numPoses = len(camPoses)
    
    ro = []
    Ho = []
    
    for trackidx in range(0, len(tracks)):
        track = tracks[trackidx]
        if not (track.lost or len(track.coords) > maxTrackLength):
            continue

        # Mark track for deletion after we use it
        tracks[trackidx].delete = True

        # Use only tracks that are longer than 2.
        if len(track.coords) < 2:
            tracks[trackidx].tooShort = True
            print("Discarding track because it is shorter than 2 images")
            continue
            
        Ps, qs, Rcos, p_oc_cs, Rcos_first, p_oc_os_first = calcPsAndQs(camPoses, track, cameraMatrix) # Wrap
        
        # Check baseline
        firstPoseIndex = camposehandler.getPoseIndexFromImageIndex(camPoses, track.firstImageIndex)
        firstPose = camPoses[firstPoseIndex]
        lastPose = camPoses[firstPoseIndex + len(track.coords) - 1]
        baseline = linalg.norm(lastPose.p_co_o - firstPose.p_co_o)
        #if (baseline < 0.5):
           #print('We do not have a large enough baseline for track ' + str(track.id) + '. Baseline = ' + str(baseline))
        #   tracks[trackidx].tooShortBaseline = True
        #   continue

        # Triangulate to find the point in 3d
        p_fo_o_est = triangulate_est.triangulate_est(Ps, qs) #Wrap
        
        # If we did not get a good estimate, we might have too short a baseline
        if numpy.isinf(p_fo_o_est[0,0]):
            print('No triangulation result for track ' + str(track.id))
            tracks[trackidx].outlier = True
            continue

        # Frame c0 is the first camera pose this track was seen in
        p_c0o_o = camposehandler.getPoseFromImageIndex(camPoses, track.firstImageIndex).p_co_o
        Roc0 = camposehandler.getPoseFromImageIndex(camPoses, track.firstImageIndex).Rco.T
        
        p_fc0_c0_est = Roc0.T*(p_fo_o_est - p_c0o_o)
        
        # Refine estimate
        #p_fc0_c0 = triangulate.gaussnewtontriang(p_fc0_c0_est, Rcos, p_oc_cs, qs)
        p_fc0_c0 = p_fc0_c0_est
        # If we did not get a good estimate, we might have too short a baseline
        if numpy.isinf(p_fc0_c0[0,0]):
            print('No refined triangulation result for track ' + str(track.id))
            tracks[trackidx].outlier = True
            continue

        # We want enough baseline to be able to estimate the point correctly
        #if (baseline*20 < linalg.norm(p_fc0_c0)):
        #   tracks[trackidx].tooShortBaseline = True
        #   continue
           
        #Sanity check
        X = p_fc0_c0.item(0)
        Y = p_fc0_c0.item(1)
        Z = p_fc0_c0.item(2)

        if Z < 1.0:
            print("Point", str(track.id) ,"too close, or behind camera. Discarding. Z=",Z);
            tracks[trackidx].outlier = True
            continue
        if Z > 100:
            print("Point", str(track.id) ," too far away. Discarding. Z=", Z);
            tracks[trackidx].outlier = True
            continue

        dx = cameraMatrix.item(0,2);
        dy = cameraMatrix.item(1,2);
        fx = cameraMatrix.item(0,0);
        fy = cameraMatrix.item(1,1);

        pixelcoordx = X/Z*fx + dx;
        pixelcoordy = Y/Z*fy + dy;

        if pixelcoordx < 0 or pixelcoordx > 1920:
            print("Point", str(track.id) ," out of camera view. Discarding. X=", pixelcoordx, "Y=", pixelcoordy);
            tracks[trackidx].outlier = True
            continue
        if pixelcoordy < 0 or pixelcoordy > 1440:
            print("Point", str(track.id) ," out of camera view. Discarding. X=", pixelcoordx, "Y=", pixelcoordy);
            tracks[trackidx].outlier = True
            continue

        p_fo_o = Roc0*p_fc0_c0 + p_c0o_o
        
        rj, Hxj, Hfj = calc_residual(Rcos, p_oc_cs, Rcos_first, p_oc_os_first, p_fo_o, track, firstPoseIndex, numPoses, cameraMatrix) #Wrap

        #print(trackidx, "Hxj.shape: ", Hxj.shape, "norm:", linalg.norm(Hxj))
        #print(trackidx, "Hfj.shape: ", Hfj.shape, "norm:", linalg.norm(Hfj))
        #print(trackidx, "rj.shape: ", rj.shape, "norm:", linalg.norm(rj))
        # Do the null-space trick
        Hoj, roj, Aj = remove_feature_dependency_wrap(Hfj, Hxj, rj) #Wrap

        if is_inlier(Hxj, Aj, P, roj, len(track.coords), imageNoiseVar): #Wrap
            #print("Track ", trackidx, " is inlier")
            if Ho == []:
                Ho = Hoj
                ro = roj
            else:
                Ho = concatenate([Ho, Hoj])
                ro = concatenate([ro, roj])
                
            with open("world.csv", "a") as text_file:
                text_file.write("%f,%f,%f\n" % (p_fo_o[0], p_fo_o[1], p_fo_o[2]))

        else:
            tracks[trackidx].outlier = True

    if len(ro) > 0:
        #print("Ho.shape: ", Ho.shape, "norm:", linalg.norm(Ho))
        #print("ro.shape: ", ro.shape, "norm:", linalg.norm(ro))

        dstates, P = calculate_correction_wrap(Ho, ro, P, imageNoiseVar) #Wrap

        #Update x and camposehandler poses
        states, camPoses = correct_states(dstates, numPoses, states, camPoses) #Wrap
        
    return P, states, tracks, camPoses

## Wrapper functions
def augmentState_wrap(P, states, test=False, useC=False):
    if not useC:
        return augmentState(P, states)
        
    if test:
        P_copy = deepcopy(P)
        P_copy = augmentState(P_copy, states)
        
    P = kalmanfilter_pywrapper.augmentState_pywrapper(P, states)
    P = numpy.asmatrix(P)

    if test:
        assert(linalg.norm(P - P_copy) < 10e-12)
        
    return P
        
def covarianceUpdate_wrap(P, states, wibb, ab, wieo, conf, test=False, useC=False):
    if not useC:
        return covarianceUpdate(P, states, wibb, ab, wieo, conf)
        
    if test:
        P_copy = deepcopy(P)
        P_copy = covarianceUpdate(P_copy, states, wibb, ab, wieo, conf)
    
    P = kalmanfilter_pywrapper.covarianceUpdate_pywrapper(P, states, wibb, ab, wieo, conf)
    P = asmatrix(P)
    
    if test:
        assert(linalg.norm(P - P_copy) < 10e-12)
        
    return P
    
def update_wrap(P, states, tracks, camPoses, points3d, cameraMatrix, imageNoiseVar, maxTrackLength, test=False, useC=False):
    if not useC:
        return update(P, states, tracks, camPoses, cameraMatrix, imageNoiseVar, maxTrackLength)
    
    if test:
        P_copy = deepcopy(P)
        states_copy = deepcopy(states)
        tracks_copy = deepcopy(tracks)
        camPoses_copy = deepcopy(camPoses)
        P_copy, states_copy, tracks_copy, camPoses_copy = update(P_copy, states_copy, tracks_copy, camPoses_copy, cameraMatrix, imageNoiseVar, maxTrackLength)
        
    #print("cameraMatrix:", cameraMatrix)
        
    P, states, tracks, camPoses = kalmanfilter_pywrapper.update_pywrapper(P, states, tracks, camPoses, cameraMatrix, imageNoiseVar, maxTrackLength);
    states.Rbo = asmatrix(states.Rbo)
    states.v_bo_o = asmatrix(states.v_bo_o).T
    states.p_bo_o = asmatrix(states.p_bo_o).T
    states.gbias = asmatrix(states.gbias).T
    states.abias = asmatrix(states.abias).T
    P = asmatrix(P)
    for camPose in camPoses:
        camPose.Rco = asmatrix(camPose.Rco)
        camPose.p_co_o = asmatrix(camPose.p_co_o).T
        camPose.Rco_first = asmatrix(camPose.Rco_first)
        camPose.p_co_o_first = asmatrix(camPose.p_co_o_first).T

    if test:
        assert(linalg.norm(P_copy - P) < 10e-10)
        
        assert(linalg.norm(states_copy.Rbo - states.Rbo) < 10e-10)
        assert(linalg.norm(states_copy.v_bo_o - states.v_bo_o) < 10e-10)
        assert(linalg.norm(states_copy.p_bo_o - states.p_bo_o) < 10e-10)
        assert(linalg.norm(states_copy.gbias - states.gbias) < 10e-10)
        assert(linalg.norm(states_copy.abias - states.abias) < 10e-10)
        for j in range(len(camPoses)):
            assert(linalg.norm(camPoses[j].Rco - camPoses_copy[j].Rco) < 10e-10)
            assert(linalg.norm(camPoses[j].p_co_o - camPoses_copy[j].p_co_o) < 10e-10)
            assert(linalg.norm(camPoses[j].Rco_first - camPoses_copy[j].Rco_first) < 10e-10)
            assert(linalg.norm(camPoses[j].p_co_o_first - camPoses_copy[j].p_co_o_first) < 10e-10)

        for j in range(len(tracks)):
            assert(tracks[j].delete == tracks_copy[j].delete)
            assert(tracks[j].outlier == tracks_copy[j].outlier)
            assert(tracks[j].tooShort == tracks_copy[j].tooShort)
            assert(tracks[j].tooShortBaseline == tracks_copy[j].tooShortBaseline)

    return P, states, tracks, camPoses
    
def calcPsAndQs_wrap(camPoses, track, cameraMatrix, test=False, useC=False):
    if not useC:
        return calcPsAndQs(camPoses, track, cameraMatrix)

    if test:
        Ps_copy, qs_copy, Rcos_copy, p_oc_cs_copy, Rcos_first_copy, p_oc_os_first_copy = calcPsAndQs(camPoses, track, cameraMatrix)

    Ps, qs, Rcos, p_oc_cs, Rcos_first, p_oc_os_first = kalmanfilter_pywrapper.calcPsAndQs_pywrapper(camPoses, track, cameraMatrix)

        
    Ps = [numpy.asmatrix(P) for P in Ps]
    qs = [numpy.asmatrix(q).getT() for q in qs]
    Rcos = [numpy.asmatrix(Rco) for Rco in Rcos]
    p_oc_cs = [numpy.asmatrix(p_oc_c).getT() for p_oc_c in p_oc_cs]
    
    if test:
        for i in range(len(Ps)):
            if (linalg.norm(qs[i] - matrix(qs_copy[i]).getT()) >= 10e-15):
                print("qs:\n")
                print(qs[i])
                print("qs_:\n")
                print(qs_copy[i])
                assert(linalg.norm(qs[i] - matrix(qs_copy[i]).getT()) < 10e-15)
            
            if (linalg.norm(Rcos[i] - Rcos_copy[i]) >= 10e-15):
                print("Rcos:\n")
                print(Rcos[i])
                print("Rcos_:\n")
                print(Rcos_copy[i])
                assert(linalg.norm(Rcos[i] - Rcos_copy[i]) < 10e-15)
            
            if (linalg.norm(Ps[i] - Ps_copy[i]) >= 10e-12):
                print("Ps is different from:")
                print(Ps[i])
                print(Ps_copy[i])
                assert(linalg.norm(Ps[i] - Ps_copy[i]) < 10e-12)
            
            if (linalg.norm(p_oc_cs[i] - p_oc_cs_copy[i]) >= 10e-15):
                print(p_oc_cs[i])
                print("p_oc_cs is different from")
                print(p_oc_cs_copy[i])
                assert(linalg.norm(p_oc_cs[i] - p_oc_cs_copy[i]) < 10e-15)

    return Ps, qs, Rcos, p_oc_cs, Rcos_first, p_oc_os_first

def calc_residual_wrap(Rcos, p_oc_cs, Rcos_first, p_oc_os_first, p_fo_o, track, firstPoseIndex, numPoses, cameraMatrix, test = False, useC = False):
    if not useC:
        return calc_residual(Rcos, p_oc_cs, Rcos_first, p_oc_os_first, p_fo_o, track, firstPoseIndex, numPoses, cameraMatrix)
        
    if test:
        rj_copy, Hxj_copy, Hfj_copy = calc_residual(Rcos, p_oc_cs, Rcos_first, p_oc_os_first, p_fo_o, track, firstPoseIndex, numPoses, cameraMatrix)
    
    rj, Hxj, Hfj = kalmanfilter_pywrapper.calc_residual_pywrapper(Rcos, p_oc_cs, Rcos_first, p_oc_os_first, p_fo_o, track, firstPoseIndex, numPoses, cameraMatrix)
    rj = asmatrix(rj).getT();
    Hxj = asmatrix(Hxj);
    Hfj = asmatrix(Hfj);
    
    if test:
        assert(linalg.norm(rj - rj_copy) < 10e-12)
        assert(linalg.norm(Hxj - Hxj_copy) < 10e-12)
        assert(linalg.norm(Hfj - Hfj_copy) < 10e-12)
        
    return rj, Hxj, Hfj

def calculate_correction_wrap(Ho, ro, P, imageNoiseVar, test = False, useC = False):
    if not useC:
        return calculate_correction(Ho, ro, P, imageNoiseVar)
        
    if test:
        P_copy = deepcopy(P)
        dstates_copy, P_copy = calculate_correction(Ho, ro, P_copy, imageNoiseVar)

    dstates, P = calculate_correction(Ho, ro, P, imageNoiseVar)
    P = asmatrix(P)

    if test:
        assert(linalg.norm(P - P_copy) < 10e-12)
        assert(linalg.norm(dstates - dstates_copy) < 10e-12)
        
    return dstates, P
    
def correct_states_wrap(dstates, numPoses, states, camPoses, test = False, useC = False):
    if not useC:
        return correct_states(dstates, numPoses, states, camPoses)
        
    if test:
        states_copy = deepcopy(states)
        camPoses_copy = deepcopy(camPoses)
        states_copy, camPoses_copy = correct_states(dstates, numPoses, states_copy, camPoses_copy)

    states, camPoses = kalmanfilter_pywrapper.correct_states_pywrapper(dstates, numPoses, states, camPoses)
    states.abias = numpy.asmatrix(states.abias).T
    states.gbias = numpy.asmatrix(states.gbias).T
    states.Rbo = numpy.asmatrix(states.Rbo)
    states.v_bo_o = numpy.asmatrix(states.v_bo_o).T
    states.p_bo_o = numpy.asmatrix(states.p_bo_o).T
    
    for camPose in camPoses:
        camPose.Rco = numpy.asmatrix(camPose.Rco)
        camPose.p_co_o = numpy.asmatrix(camPose.p_co_o).T
        camPose.Rco_first = numpy.asmatrix(camPose.Rco_first)
        camPose.p_co_o_first = numpy.asmatrix(camPose.p_co_o_first).T

    if test:
        assert(linalg.norm(states_copy.Rbo - states.Rbo) < 10e-12)
        assert(linalg.norm(states_copy.v_bo_o - states.v_bo_o) < 10e-12)
        assert(linalg.norm(states_copy.p_bo_o - states.p_bo_o) < 10e-12)
        assert(linalg.norm(states_copy.gbias - states.gbias) < 10e-12)
        assert(linalg.norm(states_copy.abias - states.abias) < 10e-12)

        for j in range(len(camPoses)):
            assert(linalg.norm(camPoses[j].Rco - camPoses_copy[j].Rco) < 10e-10)
            assert(linalg.norm(camPoses[j].p_co_o - camPoses_copy[j].p_co_o) < 10e-10)
            assert(linalg.norm(camPoses[j].Rco_first - camPoses_copy[j].Rco_first) < 10e-10)
            assert(linalg.norm(camPoses[j].p_co_o_first - camPoses_copy[j].p_co_o_first) < 10e-10)
                        
    return states, camPoses

def is_inlier_wrap(Hxj, Aj, P, roj, trackLength, imageNoiseVar, test = False, useC = False):
    if not useC:
        return is_inlier(Hxj, Aj, P, roj, trackLength, imageNoiseVar)

    if test:
        inlier_copy = is_inlier(Hxj, Aj, P, roj, trackLength, imageNoiseVar)
        
    inlier = kalmanfilter_pywrapper.is_inlier_pywrapper(Hxj, Aj, P, roj, trackLength, imageNoiseVar)
    
    if test:
        assert (inlier == inlier_copy)
    
    return inlier
    
    
