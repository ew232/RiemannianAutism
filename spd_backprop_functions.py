
import tensorflow as tf
import numpy as np
import os, glob
import uuid

from scipy.linalg import fractional_matrix_power
from scipy.linalg import expm
from scipy.linalg import logm
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix

####################################################################
### helper functions

def regularizeCorMats (allCorMats, threshold=0.5): # default is 1.0, or 0.75, 0.5
    nSubjs = allCorMats.shape[0]
    for i in range(nSubjs):
        corMat = allCorMats[i]
        u, s, v = np.linalg.svd(corMat)
        s[s<threshold] = threshold
        allCorMats[i] = u.dot(np.diag(s).dot(u.T))
    return allCorMats

def check_symmetric(a, tol=1e-8):
    return np.allclose(a, a.T, atol=tol)

def cov2cor (M):
    D = np.diag(1/np.sqrt(np.diag(M)))
    return D.dot(M.dot(D))
    
def np_logEigFrechet_preproc(X, Xmu): ### AIM with specified basepoint
    Umu, Smu, Vmu = np.linalg.svd(Xmu)
    sqrtSmuInv = np.diag(1/np.sqrt(Smu))
    sqrtBInv = Umu.dot(sqrtSmuInv.dot(Vmu))
    output = X.copy()
    _nTrain = X.shape[0]
    for i in range(_nTrain):
        X_i = X[i]
        BAB = sqrtBInv.dot(X_i).dot(sqrtBInv)
        U,S,V = np.linalg.svd(BAB)
        logevals = np.diag(np.log(S))
        UlogSV = U.dot(logevals.dot(V))
        output[i] = UlogSV
    return output

def np_logEig_preproc(X):
    output = X.copy()
    _nTrain = X.shape[0]
    for i in range(_nTrain):
        X_i = X[i]
        U,S,V = np.linalg.svd(X_i)
        logevals = np.diag(np.log(S))
        UlogSV = U.dot(logevals.dot(V))
        output[i] = UlogSV
    return output

def vect2upperTri(spdDim, corMat, offset=0):
    output = np.zeros([spdDim,spdDim])
    output[np.triu_indices_from(output, offset)] = corMat
    np.fill_diagonal(output,0) # clear the diagonal and transpose again
    output = output.T
    output[np.triu_indices_from(output, offset)] = corMat
    return output

def vect2upperTriAll(spdDim, allCorMats, offset=0):
    allCorMatsU = np.zeros([allCorMats.shape[0], spdDim, spdDim])
    for i in range(allCorMats.shape[0]):
        allCorMatsU[i] = vect2upperTri(spdDim, allCorMats[i], offset)
    return allCorMatsU

def upperTriAll(spdDim, allCorMats, offset=0):
    if (offset==0):
        allCorMatsU = np.zeros([allCorMats.shape[0], int(spdDim*(spdDim+1)/2)])
    elif (offset==1):
        allCorMatsU = np.zeros([allCorMats.shape[0], int(spdDim*(spdDim+1)/2)-spdDim])
    for i in range(allCorMats.shape[0]):
        allCorMatsU[i] = spd.upperTri_linearized(allCorMats[i], offset)
    return allCorMatsU

def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
    return tf.random_normal(shape=size, stddev=xavier_stddev)

def genOrthog(genDim1, genDim2): # genDim1 >= genDim2 for long orthogonal
    randMat = np.random.normal(0,1,[genDim1,genDim2])
    Q = np.linalg.qr(randMat)[0]
    return Q

def genRandomSPD (spdDim, minEigval=0.1, maxEigval=2.0):
    Q = genOrthog(spdDim, spdDim)
    Sigma = np.diag(np.random.uniform(minEigval, maxEigval, spdDim))
    return Q.T.dot(Sigma.dot(Q))

def upperTri_linearized(x, offset=0):
    return x[np.triu_indices_from(x, offset)]

def upperTri_matrixfy(x, offset=0, size=111):
    output = np.zeros([size, size])
    output[np.triu_indices_from(output, 0)] = x
    output = output.T + output
    np.fill_diagonal(output, np.diag(output)/2)
    return output

#########################################################
### spd functions


def tfMatPow(M,pow):
    eigval, eigu = tf.linalg.eigh(M, name='eigh')
    if (pow==0.5):
        return tf.matmul(eigu, tf.matmul(tf.linalg.diag(tf.real(tf.sqrt(eigval))), eigu, adjoint_b=True))
    elif (pow==-0.5):
        return tf.matmul(eigu, tf.matmul(tf.linalg.diag(tf.real(1/tf.sqrt(eigval))), eigu, adjoint_b=True))
    else:
        return 'error'

def tfMatExp(M):
    eigval, eigu = tf.linalg.eigh(M, name='eigh')
    return tf.matmul(eigu, tf.matmul(tf.linalg.diag(tf.real(tf.exp(eigval))), eigu, adjoint_b=True))

def tfMatLog(M):
    eigval, eigu = tf.linalg.eigh(M, name='eigh')
    return tf.matmul(eigu, tf.matmul(tf.linalg.diag(tf.real(tf.log(eigval))), eigu, adjoint_b=True))

def tfExpMap(M,gradStep): # gradstep needs to be tangent vector, symmetric matrix
    MHalf = tfMatPow(M,0.5)
    MNegHalf = tfMatPow(M,-0.5)
    inside = tf.matmul(MNegHalf, tf.matmul(gradStep, MNegHalf))
    expInside = tfMatExp(inside)
    output = tf.matmul(MHalf, tf.matmul(expInside, MHalf))
    return output

def tfLogMap(M,P):
    MHalf = tfMatPow(M,0.5)
    MNegHalf = tfMatPow(M,-0.5)
    inside = tf.matmul(MNegHalf, tf.matmul(P, MNegHalf))
    logInside = tfMatLog(inside)
    output = tf.matmul(MHalf, tf.matmul(logInside, MHalf))
    return output

def calcFrechetTFPenalty (nTime, nDims, C, frechetMean):
    estXmu = tfcomputeXmufromC(C)
    frechetMeanPenalty = nTime/2*tf.linalg.logdet(frechetMean) - (nTime+nDims+1)/2*tf.linalg.logdet(estXmu) - 1/2*tf.trace(tf.matmul(frechetMean, tf.linalg.inv(estXmu))) - (nTime*nDims)/2*tf.math.log(2.0) - tf.math.log(np.float32(sc.special.multigammaln(nTime/2,nDims)))
    return -1*frechetMeanPenalty 

def calcFrechetTFPenalty2 (nTime, nDims, lmda, K, C, frechetMean):
    estXmu = tfcomputeXmufromC(C)
    frechetMeanPenalty2 = K*(nTime+nDims+1)*tf.linalg.logdet(estXmu) + lmda*tf.trace(tf.matmul(frechetMean, tf.linalg.inv(estXmu)))
    return frechetMeanPenalty2

def computeC(Xmu):
    ### compute C from the frechet mean of covariance from whole dataset
    Umu, Smu, Vmu = np.linalg.svd(Xmu)
    sqrtSmuInv = np.diag(1/np.sqrt(Smu))
    sqrtBInv = Umu.dot(sqrtSmuInv.dot(Vmu))
    return sqrtBInv

def computeXmufromC(C):
    ### get C into covariance space for the penalty comparison
    Umu, Smu, Vmu = np.linalg.svd(Xmu)
    sqSmuInv = np.diag(1/np.square(Smu))
    sqCInv = Umu.dot(sqSmuInv.dot(Vmu))
    return sqCInv

def tfcomputeXmufromC(C):
    ### get C into covariance space for the penalty comparison
    eigval, eigu, eigv = tf.linalg.svd(C, name='svd')
    output = tf.matmul(eigu, tf.matmul(tf.linalg.diag(1/tf.square(eigval)), eigv, adjoint_b=True))
    return output


def py_func(func, inp, Tout, stateful=True, name=None, grad=None):
    # Need to generate a unique name to avoid duplicates:
    rnd_name = 'PyFuncGrad' + str(np.random.randint(0, 1E+8))
    tf.RegisterGradient(rnd_name)(grad)  # see _MySquareGrad for grad example
    g = tf.get_default_graph()
    with g.gradient_override_map({"PyFunc": rnd_name}):
        return tf.py_func(func, inp, Tout, stateful=stateful, name=name)

def getStiefelParamIdx (grads_and_vars, searchName='BiMapW'):
    WBiMapIdx = []
    for i in range(len(grads_and_vars)):
        splitName = str.split(grads_and_vars[i][1].name, searchName)
        if len(splitName) > 1:
            WBiMapIdx.append(i)
    return WBiMapIdx

def stiefelRetract(grads_and_vars, j):
    W = grads_and_vars[j][1]
    Q = tf.qr(W)[0]
    return (grads_and_vars[j][1].assign(Q))

### BiMap

def biMapGrad(op, grad):
    X = op.inputs[0]; W = op.inputs[1]
    dim_ori, dim_tar = W.shape
    dim_ori = int(dim_ori); dim_tar = int(dim_tar)
    def np_biMap_dzdw (X,W,d_t):
        dzdw = 0*W
        for ix in range(X.shape[0]):
            dzdw = dzdw + 2*X[ix].dot(W.dot(d_t[ix]))
        return dzdw
    
    def tf_biMap_dzdw_stiefelTangent (W,dzdw):
        Wdgrad = tf.matmul(tf.transpose(W), dzdw)
        return dzdw - tf.matmul(W, (Wdgrad+tf.transpose(Wdgrad))/2)
    
    def tf_biMap_dzdw (X,W,d_t):
        dzdw = tf.py_func(np_biMap_dzdw, [X,W,d_t], tf.float32)
        return tf_biMap_dzdw_stiefelTangent (W, dzdw)
    
    X = tf.map_fn(lambda y: tf.reshape(y, [dim_ori, dim_ori]), X)
    d_t = tf.map_fn(lambda y: tf.reshape(y, [dim_tar, dim_tar]), grad)
    WtW = tf.map_fn(lambda y: tf.matmul(W, tf.matmul(y, tf.transpose(W))), d_t)
    dzdx = tf.map_fn(lambda y: tf.reshape(y,[-1]), WtW)
    dzdw = tf_biMap_dzdw (X,W,d_t)
    return dzdx, dzdw

def np_biMap (X, W):
    D = int(np.sqrt(int(X.shape[1])))
    output = np.zeros([X.shape[0], int(W.shape[1])**2],dtype=np.float32)
    for i in range(X.shape[0]):
        inputMat = X[i]
        inputMat = np.reshape(inputMat, [D,D])
        WXW = W.T.dot(inputMat.dot(W))
        output[i] = np.reshape(WXW, [-1])
    return output.astype(np.float32)

def tf_biMap(X,W, name=None):
    with tf.name_scope(name, 'bimap', [X,W]) as name:
        z = py_func(np_biMap, [X,W], [tf.float32], name=name, grad=biMapGrad)
        z[0].set_shape([X.get_shape()[0], int(W.get_shape()[1])**2])
        return z[0]

### AIM

def np_biMapAIM (X, W):
    D = int(np.sqrt(int(X.shape[1])))
    output = np.zeros([X.shape[0], int(W.shape[1])**2],dtype=np.float32)
    for i in range(X.shape[0]):
        inputMat = X[i]
        inputMat = np.reshape(inputMat, [D,D])
        WXW = W.T.dot(inputMat.dot(W))
        output[i] = np.reshape(WXW, [-1])
    return output.astype(np.float32)

def tf_biMapAIM(X,W, name=None):
    with tf.name_scope(name, 'bimapAIM', [X,W]) as name:
        z = py_func(np_biMapAIM, [X,W], [tf.float32], name=name, grad=biMapAIMGrad)
        z[0].set_shape([X.get_shape()[0], int(W.get_shape()[1])**2])
        return z[0]

def biMapAIMGrad(op, grad, smallStep=True):
    X = op.inputs[0]; W = op.inputs[1]
    dim_ori = W.shape[0]
    dim_ori = int(dim_ori)
    
    def np_biMapAIM_dzdw(X,W,d_t):
        dzdw = 0*W
        for ix in range(X.shape[0]):
            # dzdw = dzdw + 2*X[ix].dot(W.dot(d_t[ix])) ## wrong
            # dzdw = dzdw + (X[ix].dot(W) + W.dot(X[ix])).dot(d_t[ix]) ## wrong
            dzdw = dzdw + X[ix].dot(W.dot(d_t[ix])) + d_t[ix].dot(W.dot(X[ix]))
        dzdw = (dzdw + dzdw.T)/2 # make symmetric anyway
        return dzdw
    
    def tf_biMapAIM_dzdw (X,W,d_t):
        dzdw = tf.py_func(np_biMapAIM_dzdw, [X,W,d_t], tf.float32)
        return dzdw
    
    X = tf.map_fn(lambda y: tf.reshape(y, [dim_ori, dim_ori]), X)
    d_t = tf.map_fn(lambda y: tf.reshape(y, [dim_ori, dim_ori]), grad)
    WtW = tf.map_fn(lambda y: tf.matmul(W, tf.matmul(y, tf.transpose(W))), d_t)
    dzdx = tf.map_fn(lambda y: tf.reshape(y,[-1]), WtW)
    dzdw = tf_biMapAIM_dzdw (X,W,d_t)
    
    if (smallStep):
        dzdw = 0.01*dzdw
    
    return dzdx, dzdw

def np_expMap_sym (Cest, gradC, epsilon=1e-4):
    Cesthalf = fractional_matrix_power(Cest, 0.5)
    gradCexp = expm(-gradC*epsilon) # gradient descent in negative direction
    CestNew = Cesthalf.dot(gradCexp.dot(Cesthalf))
    CestNew = (CestNew + CestNew.T)/2 # enforce symmetry again
    return CestNew.astype(np.float32)

def tf_expMap_sym(Cest, gradC, epsilon, name=None): # should be faster without py_func
    CestSVD = tf.svd(Cest)
    CestEvalSqrt = tf.diag(tf.sqrt(CestSVD[0]))
    Cesthalf = tf.matmul(CestSVD[1], tf.matmul(CestEvalSqrt, tf.transpose(CestSVD[2])))
    gradCExp = tf.linalg.expm(-gradC*epsilon)
    CestNew = tf.matmul(Cesthalf, tf.matmul(gradCExp, Cesthalf))
    CestNew = (CestNew + tf.transpose(CestNew))/2
    return CestNew

def AIMMeanOptimize(X, Ys):
    N = Ys.shape[0]
    XHalf = fractional_matrix_power(X, 0.5)
    XNegHalf = fractional_matrix_power(X, -0.5)
    logMOutputs = np.zeros(Ys.shape)
    for i in range(N):
        Y = Ys[i]
        logMTemp = logm(XNegHalf.dot(Y.dot(XNegHalf)))
        logMOutputs[i] = logMTemp
    
    logMMean = sum(logMOutputs,0)/N
    expM = expm(logMMean)
    output = XHalf.dot(expM.dot(XHalf))
    return output.astype(np.float32)

### ReEig

def reEigGrad(op, grad):
    X = op.inputs[0]
    def np_reEigGrad(X, grad):
        np.seterr(divide='ignore')
        epsilon = 1e-4
        D = int(np.sqrt(X[1].shape))
        dzdx = np.zeros([X.shape[0], D**2])
        for i in range(X.shape[0]):
            inputMat = np.reshape(X[i], [D,D])
            eigOut = np.linalg.svd(inputMat)
            evals = eigOut[1]
            U = eigOut[0]
            dLdC = np.reshape(grad[i], [D,D])
            dLdC = (dLdC + dLdC.T)/2
            
            L_I = (evals<epsilon)
            evals[L_I] = epsilon
            maxS = np.diag(evals)
            maxI = L_I
            dLdV = 2*dLdC.dot(U.dot(maxS));
            dLdS = np.diag(np.float32(~maxI)).dot(U.T.dot(dLdC.dot(U)))
            
            Kdenom = np.reshape(evals,[D,1]).dot(np.ones([1,D]))
            K = 1/(Kdenom - Kdenom.T)
            np.fill_diagonal(K,0)
            K[np.isinf(K)] = 0
            
            KUdLdV = K.T*(U.T.dot(dLdV))
            KUdLdV = (KUdLdV + KUdLdV.T)/2
            dDiag = np.diag(np.diag(dLdS))
            dzdx_mat = np.dot(U,np.dot(KUdLdV+dDiag, U.T))
            dzdx[i] = np.reshape(dzdx_mat, [-1])
        return dzdx.astype(np.float32)
    return tf.py_func(np_reEigGrad, [X, grad], tf.float32)

def np_reEig (X):
    epsilon = 1e-4
    D = int(np.sqrt(int(X.shape[1])))
    output = np.zeros([X.shape[0], D**2])
    for i in range(X.shape[0]):
        inputMat = X[i]
        inputMat = np.reshape(inputMat, [D,D])
        eigOut = np.linalg.svd(inputMat)
        evals = eigOut[1]
        evects = eigOut[0]
        evals[evals<epsilon] = epsilon
        epsevals = np.diag(evals)
        UReSigmaU = evects.dot(epsevals.dot(evects.T))
        output[i] = np.reshape(UReSigmaU,[-1])
    return output.astype(np.float32)

def tf_reEig(X, name=None):
    with tf.name_scope(name, 'reeig', [X]) as name:
        z = py_func(np_reEig, [X], [tf.float32], name=name, grad=reEigGrad)
        z[0].set_shape([X.get_shape()[0], X.get_shape()[1]])
        return z[0]

### LogEig

def logEigGrad(op, grad):
    X = op.inputs[0]
    def np_logEigGrad(X, grad):
        np.seterr(divide='ignore')
        output = np.zeros(X.shape, dtype=np.float32)
        D = int(np.sqrt(X.shape[1]))
        for i in range (X.shape[0]):
            inputMat = X[i]
            dzdy_i = grad[i]
            inputMat = np.reshape(inputMat, [D,D])
            eigOut = np.linalg.svd(inputMat)
            diagS = eigOut[1] # vector of eigenvalues
            S = np.diag(diagS); U = eigOut[0]; V = eigOut[2]
            eps = np.finfo(np.float32).eps
            cond = diagS>D*eps
            ind = np.where(cond)[0]
            Dmin = min(sum(cond), D)
            S = S[:,ind]; U = U[:,ind]
            dLdC = np.reshape(dzdy_i, [D,D]); dLdC = (dLdC.T + dLdC)/2
            diagLogS = np.diag(np.log(np.diag(S)))
            dLdV = 2*dLdC.dot(U.dot(diagLogS))
            diagInvS = np.diag(1/np.diag(S))
            UdLdCU = U.T.dot(dLdC.dot(U))
            dLdS = diagInvS.dot(UdLdCU)
            diagSOnes = np.dot(np.reshape(np.diag(S),[D,1]), np.ones([1,D]))
            K = 1/(diagSOnes-diagSOnes.T)
            cond = np.isinf(K)
            K[cond] = 0
            if all(diagS==1):
                dzdx = tf.zeros([D,D])
            else:
                KUdLdV = K.T*(U.T.dot(dLdV))
                KUdLdV = (KUdLdV + KUdLdV.T)/2
                dDiag = np.diag(np.diag(dLdS))
                dzdx = np.dot(U,np.dot(KUdLdV+dDiag, U.T))
            output[i] = np.reshape(dzdx,[-1])
        return output.astype(np.float32)
    return tf.py_func(np_logEigGrad, [X, grad], tf.float32)

def np_logEig(X):
    D = int(np.sqrt(int(X.shape[1])))
    output = X.copy()
    _nTrain = X.shape[0]
    for i in range(_nTrain):
        X_i = np.reshape(X[i], [D, D])
        U,S,V = np.linalg.svd(X_i)
        logevals = np.diag(np.log(S))
        UlogSV = U.dot(logevals.dot(V))
        output[i] = np.reshape(UlogSV, D**2) # vectorized
    return output

def tf_logEig(X, name=None):
    with tf.name_scope(name, 'logeig', [X]) as name:
        z = py_func(np_logEig, [X], [tf.float32], name=name, grad=logEigGrad)
        z[0].set_shape([X.get_shape()[0], X.get_shape()[1]])
        return z[0]
