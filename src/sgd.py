#!/apollo/env/MLEnvImprovement/bin/python

'''
Created on Jul 22, 2014

@author: galena
'''

import math
import numpy as np
import scipy.sparse as sp
import numpy.linalg as linalg

from scipy.stats import logistic

import sys, getopt, re, gzip

from sklearn.linear_model import LogisticRegression
from sklearn.datasets import fetch_mldata, fetch_20newsgroups_vectorized

import cProfile, pstats, StringIO

'''
This code implements a variety of stochastic gradient descent algorithms.

Update types:
  1. SGD: the update is based on the gradient on a single instance at a time
  2. SVRG: the variance-reduced gradient is used
  3. SAG: the stochastic average gradient is used
  
Regularization types:
  1. None: no regularization
  2. Proximal: Proximal updates are used
  
Step sizes:
  1. Constant
  2. AdaGrad
  
All combinations of the above are supported. In addition, when proximal
regularization is used, the first epoch can be performed using only basic SGD
(with or without AdaGrad) via the --initSGD and --initAdaGrad options.

In all cases, the weights are computed only as needed for the active features
of the current instance. The updates require a number of operations
proportional to the number of non-zero features.
'''

def printStats(x):
    print "max: " + str(np.amax(x)) + "  min: " + str(np.amin(x)) + "  mean: " + str(np.mean(x)) + "  median: " + str(np.median(x))

def matDot(A,B):
    assert A.shape == B.shape
    return np.dot(A.reshape(A.size), B.reshape(B.size))

def nnz(A):
    nr, nc = A.shape
    return nr * nc - list(A.reshape(A.size)).count(0)

 # Get the next instance, either drawn uniformly at random
 # or looping over the data. The sparse representation is returned
 # X is assumed to be a csr_matrix
def getSample(X, t):
    if usePerm or useStrat:
        row = perm[t % nr]
    elif randomSamples:
        row = np.random.randint(nr)
    else:
        row = t % nr
        
    startRow = X.indptr[row]
    endRow = X.indptr[row+1]
    xInd = X.indices[startRow:endRow]
    xVal = X.data[startRow:endRow]
    
    return (row, xInd, xVal)

def prox_vec(w, l1, l2):
    if l1 > 0:
        v = np.abs(w) - l1
        v = np.clip(v, 0, np.inf)
        v *= np.sign(w) / (1 + l2)
        return v
    else:
        return w / (1 + l2)

# elastic net proximal operator
def prox(w, s, l1, l2):
    w += s
    if w > l1:
        return (w - l1) / (1 + l2)
    elif w < -l1:
        return (w + l1) / (1 + l2)
    else:
        return 0

# compute the composed elastic net proximal operator
# prox(prox(...prox(w + s) ...+ s) + s)
# assuming that w >= 0 and s is not too negative 
def iteratedProx_pos_nobounds(w, k, s, l1, l2):
    if k == 0:
        return w
    
    assert w >= 0
    
    if l2 == 0:
        return max(w+k*(s-l1),0.0)
    
    a = 1.0 / (1.0 + l2)
    aK = a ** k
    result = aK * w + a * (s - l1) * (1 - aK) / (1 - a)
    
    # this method is valid in case s >= -l1 OR s < -l1 but k is
    # small enough that the result remains positive, so assert here
    assert result > 0 or s >= -l1
    return max(result, 0.0)

# compute the composed elastic net proximal operator
# prox(prox(...prox(w + s) ...+ s) + s)
# assuming that w >= 0 and s is not too negative 
def iteratedProx_pos_nobounds_vec(w, k, s, l1, l2):
    if l2 == 0:
        return np.clip(w+k*(s-l1),0.0,np.inf)
    
    a = 1.0 / (1.0 + l2)
    aK = a ** k
    result = aK * w + a * (s - l1) * (1 - aK) / (1 - a)
    
    return np.clip(result, 0.0, np.inf)

# compute the composed elastic net proximal operator
# prox(prox(...prox(w + s) ...+ s) + s)
# in the naive way, for testing 
def iteratedProx_slow(w, k, s, l1, l2):
    if k == 0:
        return w

    w = prox(w, s, l1, l2)
    return iteratedProx_slow(w, k-1, s, l1, l2)
        
# compute the composed elastic net proximal operator
# prox(prox(...prox(w + s) ...+ s) + s)
def iteratedProx(w, k, s, l1, l2):
    if k == 0:
        return w
    elif k == 1:
        return prox(w, s, l1, l2)
    
    if w < 0:
        return -iteratedProx(-w, k, -s, l1, l2)
    
    if s >= -l1:
        return iteratedProx_pos_nobounds(w, k, s, l1, l2)
    
    if l2 > 0:
        dReal = math.log(1 + (l2 * w)/(l1 - s)) / math.log(1 + l2)
    else:
        dReal = w / (l1 - s)
    d = int(math.floor(dReal))
    
    if d >= k:
        return iteratedProx_pos_nobounds(w, k, s, l1, l2)
    
    w_d = iteratedProx_pos_nobounds(w, d, s, l1, l2)
    w_d1 = prox(w_d, s, l1, l2)
    return -iteratedProx_pos_nobounds(-w_d1, k-d-1, -s, l1, l2)

def iteratedProx_vec(w, k, s, l1, l2):
    res = np.ndarray(w.shape)

    neg = w < 0
    s[neg] *= -1
    w[neg] *= -1
    
    i = s >= -l1
    res[i] = iteratedProx_pos_nobounds_vec(w[i], k[i], s[i], l1, l2)
    
    i = np.logical_not(i)
    
    if i.sum() > 0:
        d = np.ndarray(w.shape)
        
        if l2 > 0:
            d[i] = np.log(1 + (l2 * w[i])/(l1 - s[i])) / math.log(1 + l2)
        else:
            d[i] = w[i] / (l1 - s[i])
        d = np.floor(d)
        
        j = np.logical_and(i, d >= k)
        res[j] = iteratedProx_pos_nobounds_vec(w[j], k[j], s[j], l1, l2)
        
        j = np.logical_and(i, np.logical_not(j))
        
        if j.sum() > 0:
            res[j] = iteratedProx_pos_nobounds_vec(w[j], d[j], s[j], l1, l2)
            res[j] = prox_vec(res[j] + s[j], l1, l2)
            res[j] = -iteratedProx_pos_nobounds_vec(-res[j], k[j]-d[j]-1, -s[j], l1, l2)
    
    res[neg] *= -1
    
    return res

def iteratedProx_pos_nobounds_vec_AdaGrad(w, k, l1, l2):
    result = np.ndarray(w.shape)
    
    i = l2 > 0
    
    if i.sum() > 0:
        a = 1.0 / (1.0 + l2[i])
        aK = a ** k[i]
        result[i] = aK * w[i] - a * l1[i] * (1 - aK) / (1 - a)

    i = np.logical_not(i)
    result[i] = w[i]-k[i]*l1[i]
    
    return np.clip(result, 0.0, np.inf)

def prox_vec_AdaGrad(w, l1, l2):
    if (l1 > 0).sum() > 0:
        v = np.abs(w) - l1
        v = np.clip(v, 0, np.inf)
        v *= np.sign(w) / (1 + l2)
        return v
    else:
        return w / (1 + l2)

def iteratedProx_vec_AdaGrad(w, k, l1, l2):
    neg = w < 0
    w[neg] *= -1    
    res = iteratedProx_pos_nobounds_vec_AdaGrad(w, k, l1, l2)  
    res[neg] *= -1    
    return res

def trainProx(w, b, G, p, X, y, epochs, eta, l1, l2):
    nr,nc = X.shape
    nl = y.shape[1]
    assert y.shape[0] == nr
    assert w.shape == G.shape == (nl,nc)
    assert p.shape == (nr,nl)
    assert b.size == nl
    
    # vector of time step at which each coordinate is up-to-date
    tVec = np.zeros(nc, dtype=np.int64)
    
    for t in range(epochs * nr):
        (row, xInds, xVals) = getSample(X, t)

        kVec = np.tile(t - tVec[xInds], (nl,1))
        if useSVRG or useSAG or useSAGA:
            sVec = -eta*G[:,xInds]
        else:
            sVec = np.zeros((nl, xInds.size))
        tempW = iteratedProx_vec(w[:,xInds], kVec, sVec, l1*eta, l2*eta)

        scores = xVals.dot(tempW.T)
        scores += b
        
        if not useSqErr:
            scores = logistic.cdf(scores)
        
        if useSAG:               
            w[:,xInds] = tempW        
            G[:,xInds] += np.outer(scores - p[row,:], xVals) / nr
            p[row,:] = scores
            
            tVec[xInds] = t
        else:
            if useSVRG:
                g = np.outer(scores - p[row,:], xVals) + G[:,xInds]
            elif useSAGA:
                tempG = np.outer(scores - p[row,:], xVals)
                G[:,xInds] += tempG / nr
                p[row,:] = scores                
                g = tempG + G[:,xInds]
            else:
                g = np.outer(scores - y[row,:], xVals)

            w[:,xInds] = prox_vec(tempW - eta*g, l1 * eta, l2 * eta)
    
            tVec[xInds] = t + 1

    kVec = np.tile(epochs * nr - tVec, (nl,1))
    if useSVRG or useSAG or useSAGA:
        sVec = -eta*G
    else:
        sVec = np.zeros((nl, nc))
    np.copyto(w,iteratedProx_vec(w, kVec, sVec, l1*eta, l2*eta))
                 
def trainProx_AdaGrad(w, n, b, X, y, epochs, eta, l1, l2):
    nr,nc = X.shape
    nl = y.shape[1]
    assert y.shape[0] == nr
    assert n.shape == w.shape == (nl,nc)
    assert b.size == nl
    
    # vector of time step at which each coordinate is up-to-date
    tVec = np.zeros(nc, dtype=np.int64)
    
    for t in range(epochs * nr):
        (row, xInds, xVals) = getSample(X, t)

        kVec = np.tile(t - tVec[xInds], (nl,1))
        etaVec = eta/(1+np.sqrt(n[:,xInds]))        
        tempW = iteratedProx_vec_AdaGrad(w[:,xInds], kVec, l1*etaVec, l2*etaVec)
        
        scores = xVals.dot(tempW.T)
        scores += b
        
        if not useSqErr:
            scores = logistic.cdf(scores)
        
        g = np.outer(scores - y[row,:], xVals)
            
        n[:,xInds] += np.square(g)
        etaVec = eta/(1+np.sqrt(n[:,xInds]))
        w[:,xInds] = prox_vec_AdaGrad(tempW - etaVec*g, l1*etaVec, l2*etaVec)
        
        tVec[xInds] = t + 1

    kVec = np.tile(epochs * nr - tVec, (nl,1))
    etaVec = eta/(1+np.sqrt(n))
    w = iteratedProx_vec_AdaGrad(w.copy(), kVec, l1*etaVec, l2*etaVec)

def trainSGD_AdaGrad(w, n, b, X, y, epochs, eta):
    nr,nc = X.shape
    nl = y.shape[1]
    assert y.shape[0] == nr
    assert n.shape == w.shape == (nl,nc)
    assert b.size == nl
    
    for t in range(epochs * nr):
        (row, xInd, xVal) = getSample(X, t)
        
        scores = xVal.dot(w[:,xInd].T)
        scores += b
        
        if not useSqErr:
            scores = logistic.cdf(scores)
        
        g = np.outer(scores - y[row,:], xVal)
        n[:,xInd] += np.square(g)
        eta_new = eta / (1 + np.sqrt(n[:,xInd]))

        w[:,xInd] -= eta_new * g
                
def trainSGD(w, b, X, y, epochs, eta):
    nr,nc = X.shape
    nl = y.shape[1]
    assert w.shape == (nl,nc)
    assert y.shape[0] == nr
    
    for t in range(epochs * nr):
        (row, xInd, xVal) = getSample(X, t)
        
        scores = xVal.dot(w[:,xInd].T)
        scores += b
        
        if not useSqErr:
            scores = logistic.cdf(scores)
        
        w[:,xInd] -= eta * np.outer(scores - y[row,:], xVal)

def getRegGrad(G, w, l1, l2):
    """Adjust loss gradient to include regularization
    If G is the gradient of the unregularized loss function
    the result is the minimum norm subgradient of the regularized loss.
    """
    gNew = np.zeros(G.shape)
    
    sign = np.sign(w)
    i = (sign != 0)
    gNew[i] = G[i] + l1 * sign[i] + l2 * w[i]
    sign = np.sign(G)
    i = np.logical_and(np.logical_not(i), np.abs(G) > l1)
    gNew[i] = G[i] - l1 * sign[i]
    
    return gNew

def getRegLoss(l, w, l1, l2):
    """Adjust loss to include regularization
    """
    nl = w.shape[0]
    for lab in range(nl):
        wLab = w[lab,:]
        l += l1 * linalg.norm(wLab,1)
        l += l2 / 2 * wLab.dot(wLab)
    return l

def getGPl(X, w, b, y):
    """Compute average grad G, vector of predictions p, and loss
    """
    nr,nc = X.shape
    assert w.shape == (nl,nc)
    assert y.shape == (nr,nl)
    scores = X.dot(np.transpose(w)) + b
    if useSqErr:
        p = scores
    else:
        p = logistic.cdf(scores)
    d = p-y
    if isinstance(d, np.matrix):
        d = d.getA()
    G = np.transpose(d) * X / nr
    
    if useSqErr:
        l = matDot(d, d) / (2 * nr)
    else:
        neg = logistic.logcdf(-scores)
        scores = logistic.logcdf(scores)
        scores -= neg

        if isinstance(y, sp.csr_matrix):
            l = 0
            for r in range(nr):
                startRow, endRow = y.indptr[r], y.indptr[r+1]
                indices = y.indices[startRow:endRow]
                vals = y.data[startRow:endRow]
                l -= scores[r][indices].dot(vals)
            l -= neg.sum()
            l /= nr
        else:
            l = (-matDot(y, scores)-neg.sum()) / nr
    return G, p, l

def getLoss(X, w, b, y):
    """Compute loss
    """
    nr,nc = X.shape
    assert w.shape == (nl,nc)
    assert y.shape == (nr,nl)

    if isinstance(y, sp.csr_matrix):
        loss = 0
        scores = np.ndarray(nl)
        for r in range(nr):
            startRow, endRow = X.indptr[r], X.indptr[r+1]
            indices = X.indices[startRow:endRow]
            vals = X.data[startRow:endRow]
            for l in range(nl):
                scores[l] = w[l,indices].dot(vals)
            scores += b
                
            startRow, endRow = y.indptr[r], y.indptr[r+1]
            indices = y.indices[startRow:endRow]
            vals = y.data[startRow:endRow]

            if useSqErr:
                scores[indices] -= vals
                loss += 0.5 * np.dot(scores, scores)
            else:
                pos = logistic.logcdf(scores)
                neg = logistic.logcdf(-scores)
                pos -= neg
                                
                loss += (-pos[indices].dot(vals)-neg.sum())
        loss /= nr
    else:
        scores = X.dot(np.transpose(w)) + b
        
        if useSqErr:
            scores -= y
            loss = matDot(scores, scores) / (2 * nr)
        else:
            pos = logistic.logcdf(scores)
            neg = logistic.logcdf(-scores)
        #    return (-np.dot(y,pos)-np.dot(1-y, neg)) / nr
            pos -= neg
            loss = (-matDot(y, pos)-neg.sum()) / nr
    return loss

def getStratPerm(nr, k):
    assert 1 < k < nr
    perm = np.ndarray(nr, dtype='int64')
    if usePerm:
        outerPerm = np.random.permutation(k)
    dstStart = 0
    for i in range(k):
        if usePerm:
            j = outerPerm[i]
            srcStart = long(float(nr)*j/k)
            len = long(float(nr)*(j+1)/k) - srcStart
        else:
            len = nr / k
            srcStart = np.random.randint(nr)
        innerPerm = np.random.permutation(len)
        perm[dstStart:dstStart+len] = (innerPerm + srcStart) % nr
        dstStart += len
        
    return perm
        

def makeArtificialDataMulti(nl, nr, nc, spFactor, seed):
    np.random.seed(seed)
    truth = np.random.randn(nl, nc)
    X = np.random.randn(nr, nc)
    y = np.ndarray((nr, nl))
    for r in range(nr):
        for l in range(nl):
            y[r,l] = 1 if np.random.rand() < logistic.cdf(np.dot(X[r,:],truth[l,:])) else 0 
        for c in range(nc):
            if np.random.rand() > spFactor:
                X[r,c] = 0
    return sp.csr_matrix(X), y
    
# use builtin dataset
def makeMNISTdata(max, seed):
    mnist = fetch_mldata('MNIST original')
    X = mnist.data / 255.
    nr, nc = X.shape
    y = np.zeros((nr, 10))
    for i in range(nr):
        y[i,mnist.target[i]] = 1

    nr = 60000
    testX = X[nr:,:]
    testY = y[nr:,:]
    X = X[:nr,:]
    y = y[:nr,:]

    np.random.seed(seed)
    perm = np.random.permutation(nr)
    X = X[perm]
    y = y[perm]
    
    if nr > max:
        X = X[0:max,:]
        y = y[0:max,:]
        
    return sp.csr_matrix(X), y, sp.csr_matrix(testX), testY


def makeBioASQData(dataFilename, max, seed, testFrac):
    if dataFilename.endswith(".gz"):
        datafile = gzip.open(dataFilename)
    else:
        datafile = open(dataFilename)
    nr = 0
    numVals = 0
    numLabVals = 0
    for line in datafile:
        splitLine = line.split('\t')
        assert (len(splitLine) == 2)

        feats = set(splitLine[0].split(' '))        
        numVals += len(feats)
        
        numLabVals += splitLine[1].count(' ') + 1
        
        nr += 1
        
        if nr == max: break
    datafile.close()

    Xdata = np.ndarray(numVals)
    Xindices = np.ndarray(numVals, dtype='int64')
    Xindptr = np.ndarray(nr+1, dtype="int64")
    Xindptr[0] = 0
    
    Ydata = np.ndarray(numLabVals)
    Yindices = np.ndarray(numLabVals, dtype='int64')
    Yindptr = np.ndarray(nr+1, dtype="int64")
    Yindptr[0] = 0
    
    if dataFilename.endswith(".gz"):
        datafile = gzip.open(dataFilename)
    else:
        datafile = open(dataFilename)
    insNum = 0
    featIdx = 0
    labIdx = 0
    for line in datafile:
        splitLine = line.split('\t')
        assert (len(splitLine) == 2)

        # extract feats as integers and sort
        splitFeats = splitLine[0].split(' ')
        intFeats = []
        for strFeat in splitFeats:
            intFeats.append(int(strFeat))
        intFeats.sort()
        
        # add feats, using log(1+count) as feature value
        count = 0
        currFeat = -1
        for feat in intFeats:
            if feat != currFeat and currFeat >= 0:
                Xindices[featIdx] = currFeat
                Xdata[featIdx] = math.log1p(count)
                featIdx +=1
                count = 1
            else:
                count += 1
            currFeat = feat
        if currFeat >= 0:
            Xindices[featIdx] = currFeat
            Xdata[featIdx] = math.log1p(count)
            featIdx += 1
        Xindptr[insNum+1] = featIdx
        
        # same stuff with labels (here there should be only 1 per        
        splitLabels = splitLine[1].split(' ')
        intLabels = []
        for strLab in splitLabels:
            intLabels.append(int(strLab))
        intLabels.sort()
        numLabels = len(intLabels)
        endLabIdx = labIdx + numLabels

        Yindices[labIdx:endLabIdx] = intLabels
        Ydata[labIdx:endLabIdx] = np.ones(numLabels)
        Yindptr[insNum+1] = endLabIdx
        labIdx = endLabIdx
        insNum += 1
        
        if insNum == max: break
    datafile.close()
                                
    assert insNum == nr
                
    X = sp.csr_matrix((Xdata, Xindices, Xindptr))
    y = sp.csr_matrix((Ydata, Yindices, Yindptr))
    
    np.random.seed(seed)
    perm = np.random.permutation(nr)
    beginTest = int(nr * (1.0 - testFrac))
    X = X[perm]
    y = y[perm]

    return X[:beginTest,:], y[:beginTest,:], X[beginTest:,:], y[beginTest:,:]
    

def learn_liblinear(X,y,l1,l2,tol):
    assert (l1 == 0) or (l2 == 0)
    if l2 > 0:
        LR = LogisticRegression(C=1.0/l2, penalty='l2', tol=tol, fit_intercept=useBias)
    elif l1 > 0:
        LR = LogisticRegression(C=1.0/l1, penalty='l1', tol=tol, fit_intercept=useBias)
    else:
        LR = LogisticRegression(C=1000, penalty='l2', tol=tol, fit_intercept=useBias)
    LR.fit(X,y)
    if useBias:
        bias = LR.intercept_[0]
    else:
        bias = 0
    return LR.coef_[0], bias

# set default values before reading command line
l1 = 0
l2 = 0

useSAG = False
useSAGA = False
useSVRG = False
useBias = False
useAdaGrad = False
useProx = False

initAdaGrad = False
initSGD = False
profile = False
nonRandom = False
usePerm = False
useStrat = False
useSqErr = False

eta = 1e-1
innerEpochs=1
outerEpochs=100
randomSamples=False
stratK=0

dataFilename = ""

n=np.inf

usage = """options:
    update type: (default is basic, unregularized update) 
        -p: use proximal updating
    
    stochastic gradient types: (default is basic SGD)
        -s: use stochastic average gradient (SAG)
        -g: use SAGA (only implemented for Proximal, constant step-size)
        -v: use stochastic variance reduced gradient (SVRG)
        
    other options:
        -a: use AdaGrad (not allowed with SAG/SVRG)
        -r: use random samples (not looping over the data)
        -d: data file (tsv format, may be gzipped)
        -b: add fixed bias term based on base rates for each label
        -q: use squared error (default is logistic)

long options:
    --l1: weight for l1 regularization (default: 0)
    --l2: weight for l2 regularization (default: 0)
    --eta: step size (default: 1e-1)
    --innerEpochs: number of epochs between full gradient computation (default: 1)
    --outerEpochs: number of epochs before termination (default: 100)
    --initAdaGrad: use initial epoch of AdaGrad before Prox (only allowed with Prox)
    --initSGD: use initial epoch of SGD before SVRG or SAG (only allowed with Prox)
    --nonRandom: turn off default random (with replacement) sampling for SAG or SAGA
    --perm: sample without replacement at each epoch
    --strat: use stratified sampling without replacement, given number of sections k
    --profile: turn on profiling
"""

try:
    opts, args = getopt.getopt(sys.argv[1:], 
                               "svapqgrbn:d:",
                               ["l1=","l2=","eta=","innerEpochs=","outerEpochs=","profile",
                                "initAdaGrad","initSGD","nonRandom","perm","strat="])
except getopt.GetoptError:
    print usage
    sys.exit(2)
for opt, arg in opts:
    if opt in ('-h', '--help'):
        print usage
        sys.exit()
    elif opt == '-s':
        useSAG = True
    elif opt == '-v':
        useSVRG = True
    elif opt == '-a':
        useAdaGrad = True
    elif opt == '-g':
        useSAGA = True
    elif opt == '-r':
        randomSamples = True
    elif opt == '-q':
        useSqErr = True
    elif opt == '--initAdaGrad':
        initAdaGrad = True
    elif opt == '--perm':
        usePerm = True
    elif opt == '--initSGD':
        initSGD = True
    elif opt == '--nonRandom':
        nonRandom = True
    elif opt == '--strat':
        useStrat = True
        stratK = int(arg)
    elif opt == '-p':
        useProx = True
    elif opt == '-b':
        useBias = True
    elif opt == '-d':
        dataFilename = arg
    elif opt == '-n':
        n = int(arg)
        assert 0 < n
    elif opt == '--l1':
        l1 = float(arg)
        assert 0 <= l1
    elif opt == '--l2':
        l2 = float(arg)
        assert 0 <= l2
    elif opt == '--eta':
        eta = float(arg)
        assert 0 < eta
    elif opt == '--innerEpochs':
        innerEpochs = int(arg)
        assert 0 < innerEpochs
    elif opt == '--outerEpochs':
        outerEpochs = int(arg)
        assert 0 < outerEpochs
    elif opt == '--profile':
        profile = True

#only one gradient type allowed
assert (useSAG + useSVRG + useSAGA <= 1)

# SAGA only implemented for prox, non-AdaGrad
if useSAGA:
    assert useProx and not useAdaGrad

# AdaGrad not supported with advanced gradients
assert not ((useSVRG or useSAG or useSAGA) and useAdaGrad)

# initialization epochs only implemented for prox
assert not (initAdaGrad or initSGD) or useProx

if not useProx:
    # if no regularization method is used, SVRG and SAG are not implemented
    assert (not (useSVRG or useSAG or useSAGA))
    # if no regularization method is chosen, make sure reg constants are 0
    assert l1 == 0 and l2 == 0

# can't try to turn on random and non-random at the same time!
assert not (nonRandom and randomSamples)

if (useSAG or useSAGA) and not randomSamples and not nonRandom:
    print "Turning on random samples for SAG"
    randomSamples = True
    
print "Running with options:"
if len(dataFilename) > 0:
    print "data filename: " + dataFilename
print "useSAG: " + str(useSAG)
print "useSAGA: " + str(useSAGA)
print "useSVRG: " + str(useSVRG)
print "useProx: " + str(useProx)
print "useAdaGrad: " + str(useAdaGrad)
print "initAdaGrad: " + str(initAdaGrad)
print "initSGD: " + str(initSGD)
print "useSqErr: " + str(useSqErr)
print "use fixed bias: " + str(useBias)
print "randomSamples: " + str(randomSamples)
print "nonRandom: " + str(nonRandom)
print "usePerm: " + str(usePerm)
print "useStrat: " + str(useStrat)
print "n: " + str(n)
print "l1: %e" % l1
print "l2: %e" % l2
print "eta: %e" % eta
print "innerEpochs: %d" % innerEpochs
print "outerEpochs: %d" % outerEpochs
print

# X, y = makeArtificialDataMulti(3, 200, 20, 0.2, 123)
# haveTestData = False

# X, y, testX, testY = makeMNISTdata(n, 123)
# haveTestData = True

X, y, testX, testY = makeBioASQData(dataFilename, n, 123, 0.1)
haveTestData = True

nr,nc = X.shape
nl = y.shape[1]

print str(nr) + " instances, " + str(nc) + " features, " + str(nl) + " labels."
posFrac = y.sum() / (nr * nl)
print "%f nnz feats, " % (1. * X.size / (nr * nc)),
print "%f nnz labels" % posFrac
mem = X.size + y.size + (nl * nc)
if useAdaGrad or useSVRG or useSAG or useSAGA:
    mem += (nl * nc) # full gradient, stochastic gradient, or sum squared weights
if useSAG or useSAGA:
    mem += (nr * nl) # probability table
mem *= 8 # eight bytes per double
mem /= (1024 * 1024) # in MB
print "%d MB required memory estimate (lower bound)" % mem
print 

# havetrueLoss = False
# if l1 == 0 or l2 == 0:
#     lr_w = np.ndarray((nl, nc))
#     lr_b = np.zeros(nl)
# 
#     for l in range(nl):
#         lr_w[l,:], lr_b[l] = learn_liblinear(X,y[:,l],l1 * nr,l2*nr,1e-8)
# 
#     trueLoss = getLoss(X,lr_w,lr_b,y)
#     unregTrueLoss = trueLoss
#     trueLoss = getRegLoss(trueLoss,lr_w,l1,l2)
#     print "True l: %e" % unregTrueLoss,
#     print " r: %e" % (trueLoss - unregTrueLoss)
#     havetrueLoss = True

# w represents the weight vector
w = np.zeros((nl,nc))
# b is the bias
b = np.zeros(nl)

if useBias:
    if useSqErr:
        b = y.sum(0) / nr
    else:
        # set bias using base rate with add-one smoothing
        b = (y.sum(0) + 1.) / (nr + 2.)
        b = np.log(b/(1-b))
    if isinstance(b,np.matrix):
        b = b.getA1()

if useSAG or useSAGA:
    # G is the stochastic averaged gradient
    G = np.zeros((nl, nc))
    # p is the current vector of probabilities of all instances in the dataset
    # it is initialized to y, corresponding to an average gradient of zero
    # this is slightly different from the paper, but seems to work very well 
    if isinstance(y, sp.csr_matrix):
        p = y.todense().getA()
    else:
        p = y.copy()

print "max p:", (nr * nl), " max w/G:", w.size

if useAdaGrad:
    # n is the sum of squared gradients, used by AdaGrad
    n = np.zeros((nl,nc))

if profile:
    pr = cProfile.Profile()
    pr.enable()
    
T=0
for epoch in range(outerEpochs+1):
    print epoch,
    
    if useStrat:
        perm = getStratPerm(nr, stratK)
    elif usePerm:
        perm = np.random.permutation(nr)
    
    if not (useSAG or useSAGA):
        G, p, l = getGPl(X,w,b,y)
    else:
        # SAG maintains its own G and p
        l = getLoss(X,w,b,y)
        
    unregLoss = l
    l = getRegLoss(l,w,l1,l2)
        
#     if havetrueLoss:
#         print "e: %.15e " % (l - trueLoss),
#     else:
    print "f: %.15f" % l,

    if haveTestData:
        testLoss = getLoss(testX, w, b, testY)
        print "tl: %f" % testLoss,
#     print "lg: %e " % linalg.norm(G, 'fro'),
    
    if l1 > 0 or l2 > 0:
        regGrad = getRegGrad(G, w, l1, l2)
        print "rg: %e " % linalg.norm(regGrad, 'fro'),
#         print "l: %e" % unregLoss,
#         print "r: %e" % (l - unregLoss),        
    
    if l1 > 0:
        print "nnz_p: %d" % nnz(p),
        print "nnz_w: %d" % nnz(w),
        print "nnz_G: %d" % nnz(G),

#     if havetrueLoss:
#         v = lr_w - w
#         diff = linalg.norm(v, 'fro')
#         print "diff: %e" % diff,

    print
    
    if epoch == outerEpochs:
        break
    elif epoch == 0 and (initSGD or initAdaGrad):
#         n = np.zeros((nl,nc))
#         trainSGD_AdaGrad(w, n, b, X, y, innerEpochs, eta)
#         trainSGD(w, b, X, y, innerEpochs, eta)
#         eta = eta / (1 + np.sqrt(n))
        oldSAG, oldSVRG = useSAG, useSVRG
        useSAG, useSVRG = False, False
        if initSGD:
            trainProx(w, b, G, p, X, y, innerEpochs, eta, l1, l2)
        else:
            n = np.zeros((nl,nc))
            trainProx_AdaGrad(w, n, b, X, y, innerEpochs, eta, l1, l2)
        useSAG, useSVRG = oldSAG, oldSVRG
        continue
    
    if useProx:
        if useAdaGrad:
            trainProx_AdaGrad(w, n, b, X, y, innerEpochs, eta, l1, l2)
        else:
            trainProx(w, b, G, p, X, y, innerEpochs, eta, l1, l2)
    else:
        if useAdaGrad:
            trainSGD_AdaGrad(w, n, b, X, y, innerEpochs, eta)
        else:
            trainSGD(w, b, X, y, innerEpochs, eta)

    T += innerEpochs * nr

print "done"

if profile:
    pr.disable()
    s = StringIO.StringIO()
    sortby = 'cumulative'
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    ps.print_stats()
    print s.getvalue()

# compare iterated prox implementation to naive iteration
def testIteratedProx():
    np.random.seed(123)
    
    for j in range(10000):
        w = np.random.randn()
        l1 = math.exp(np.random.randn())
        if np.random.rand() < 0.5:
            l1 = 0
        l2 = math.exp(np.random.randn())
        if np.random.rand() < 0.5:
            l2 = 0
        s = np.random.randn()
        
        print w, s, l1, l2
        
        w1 = w
        for i in range(10):
            w2 = iteratedProx(w,i,s,l1,l2)
            print w1, w2
            assert(np.allclose(w1, w2))
            w1 = prox(w1, s, l1, l2)
        
        print "************"
        print

'''
        Here is some code to generate pseudorandom permutations
        not currently used
def isPrime(n):
    if n < 3:
        return False
    
    d = 3
    while d < math.sqrt(n):
        if n % d == 0:
            return False
        d += 2

    return True

def nextGoodPrime(n):
    while n % 3 != 2: n+= 1
    if n % 2 == 0: n += 3
    while not isPrime(n):
        n = n+6
    return n

def getPermVal(i, n, p, a, b):
    p = long(p)
    i = long(i) % p
    i = (i * a) % p
    i = (i + b) % p
    i2 = (i * i) % p
    res = int((i * i2) % p)
    if res < n:
        return res
    return getPermVal(i+1, n, p, a, b) #tail recursion but rare case

def getPermConstants(n):
    p = nextGoodPrime(n)
    a = np.random.randint(p)
    b = np.random.randint(p)
    return (p, a, b)

def getSamplePerm(X, t, p, a, b):
    row = getPermVal(t, X.shape[0], p, a, b)
        
    startRow = X.indptr[row]
    endRow = X.indptr[row+1]
    xInd = X.indices[startRow:endRow]
    xVal = X.data[startRow:endRow]
    
    return (row, xInd, xVal)
'''