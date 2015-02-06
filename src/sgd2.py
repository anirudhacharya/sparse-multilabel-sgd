#!/apollo/env/MLEnvImprovement/bin/python

'''
Created on Jul 22, 2014

@author: galena

This code implements proximal stochastic gradient descent with AdaGrad for very large, sparse, multilabel problems.

The weights are stored in a sparse matrix structure that permits changing of the sparsity pattern on the fly, via lazy computation of the iterated proximal operator.

AdaGrad can be applied with step-sizes shared for each feature over all labels (appropriate for large problems) or with individual step sizes for each feature/label
'''

import math
import numpy as np
import scipy.sparse as sp
import numpy.linalg as linalg

from scipy.stats import logistic

import sys, getopt, re, gzip

import cProfile, pstats, StringIO
import cPickle

# Print some information about a vector
def printStats(x):
    print "max: " + str(np.amax(x)) + "  min: " + str(np.amin(x)) + "  mean: " + str(np.mean(x)) + "  median: " + str(np.median(x))

# Compute nnz for a matrix
def nnz(A):
    nr, nc = A.shape
    return nr * nc - list(A.reshape(A.size)).count(0)

# Print the online loss "ol", test loss "tl", test f1 for each label set, and nnz(W)
def printOutputLine(subEpoch, wRows, wData, b, testX, testY, l1, l2, loss):
    print str(epoch) + '-' + str(subEpoch),
    
    loss = loss + getRegLoss(wData,l1,l2)
    print "ol: %.15f" % loss,

    if haveTestData:
        testLoss, f1 = getLoss(testX, wRows,wData, b, testY)
        print "tl: %f" % testLoss,
        print "f1: %f" % f1,
        macroF1 = getLossMacro(testX, wRows, wData, b, testY)
        print "mf1: %f" % macroF1,
 
    if l1 > 0:
        nnz = sum([len(x) for x in wRows])
        print "nnz_w: %d" % nnz,

    print

 # Get the next instance, either drawn uniformly at random
 # or looping over the data. The sparse representation is returned
 # X is assumed to be a csr_matrix
def getSample(X, t):
    if usePerm:
        row = perm[t % nr]
    elif sampleWithReplacement:
        row = np.random.randint(nr)
    else:
        row = t % nr
        
    startRow = X.indptr[row]
    endRow = X.indptr[row+1]
    xInd = X.indices[startRow:endRow]
    xVal = X.data[startRow:endRow]
    
    return (row, xInd, xVal)

# vectorized computation of the iterated proximal map
# under the assumption that w is positive
# l1 and l2 may be arrays of the same dimensions as w,
# in which case k may also be an array, or it can be a constant
# if l1 and l2 are constants, it is assumed k is constant
def iteratedProx_pos(w, k, l1, l2):
    result = np.ndarray(w.shape)
    
    if isinstance(l2, np.ndarray):
        i = l2 > 0
        if i.sum() > 0:
            a = 1.0 / (1.0 + l2[i])
            if isinstance(k, np.ndarray):
                aK = a ** k[i]
            else:
                aK = a ** k
            result[i] = aK * w[i] - a * l1[i] * (1 - aK) / (1 - a)
        i = ~i
        if isinstance(k, np.ndarray):
            result[i] = w[i]-k[i]*l1[i]
        else:
            result[i] = w[i]-k*l1[i]
    else:
        if l2 > 0:
            a = 1.0 / (1.0 + l2)
            aK = a ** k
            result = aK * w - a * l1 * (1 - aK) / (1 - a)
        else:
            result = w - k*l1
    
    return np.clip(result, 0.0, np.inf)

# vectorized computation of the proximal map
def prox(w, l1, l2):
    if isinstance(l1, np.ndarray):
        useL1 = (l1 > 0).sum() > 0
    else:
        useL1 = (l1 > 0)

    if useL1:
        v = np.abs(w) - l1
        v = np.clip(v, 0, np.inf)
        v *= np.sign(w) / (1 + l2)
        return v
    else:
        return w / (1 + l2)

# vectorized computation of iterated proximal map
def iteratedProx(w, k, l1, l2):
    neg = w < 0
    w[neg] *= -1    
    res = iteratedProx_pos(w, k, l1, l2)  
    res[neg] *= -1    
    return res

# take dense "result" and store it sparsely in W
def reassignToConvertedW(wRows, wData, xInds, result):
    for i in range(xInds.size):
        xInd = xInds[i]
        row = result[i,:]
        wRows[xInd] = np.flatnonzero(row)
        wData[xInd] = row[wRows[xInd]]

# update all rows of W to incorporate proximal mappings        
def bringAllUpToDate(wRows, wData, tVec, t):
    nc = tVec.size
    for feat in range(nc):
        k = t - tVec[feat]
        if useAdaGrad:
            if useSharedStep:
                etaVec = eta/(1+np.sqrt(n[feat]))
            else:
                etaVec = eta/(1+np.sqrt(n[feat,:]))
        else:
            etaVec = eta
        
        wData[feat] = iteratedProx(wData[feat], k, l1*etaVec, l2*etaVec)
        
        #sparsify
        nz = np.flatnonzero(wData[feat])
        wRows[feat] = wRows[feat][nz]
        wData[feat] = wData[feat][nz]

# train weights with proximal stochastic gradient (optionally AdaGrad)
def trainProx(wRows, wData, n, b, X, y, eta, l1, l2, outputFreq):
    nr,nc = X.shape
    nl = y.shape[1]
    assert y.shape[0] == nr
    assert b.size == nl
    
    if useAdaGrad:
        if useSharedStep:
            assert n.size == nc
        else:
            assert n.shape == (nc,nl)

    # vector of time step at which each coordinate is up-to-date
    tVec = np.zeros(nc, dtype=np.int64)
    
    onlineLoss = 0
    totalOnlineLoss = 0
    
    subEpoch = 0    
    for t in range(nr):
    
        if t % 100 == 0:
            print "training row: " + str(t)

        (row, xInds, xVals) = getSample(X, t)
        
        if xInds.size == 0:
            continue

        # 1. Lazily update relevant rows of w, storing them in tempW        
        totalNnzW = sum(wRows[xInd].size for xInd in xInds)
            
        tempW = np.ndarray(totalNnzW)
        kVec = np.ndarray(totalNnzW, dtype=np.int64)

        if useAdaGrad:
            etaVec = np.ndarray(totalNnzW)
        else:
            etaVec = eta

        pos = 0
        for xInd in xInds:
            numW = wRows[xInd].size
            endPos = pos+numW
            kVec[pos:endPos] = t - tVec[xInd]
            if useAdaGrad:
                if useSharedStep:
                    etaVec[pos:endPos] = eta / (1 + math.sqrt(n[xInd]))
                else:
                    etaVec[pos:endPos] = eta / (1 + np.sqrt(n[xInd,wRows[xInd]]))
            tempW[pos:endPos] = wData[xInd]
            pos = endPos

        tempW = iteratedProx(tempW, kVec, l1*etaVec, l2*etaVec)
        tVec[xInds] = t
        
        # 2. Compute scores
        scores = b.copy()
        pos = 0
        for (xInd, xVal) in zip(xInds, xVals):
            numW = wRows[xInd].size
            endPos = pos+numW
            scores[wRows[xInd]] += tempW[pos:endPos] * xVal
            pos = endPos

        # 3. Compute loss and subtract labels from (transformed) scores for gradient        
        (startY, endY) = y.indptr[row], y.indptr[row+1]
        yCols = y.indices[startY:endY]
        yVals = y.data[startY:endY]
        
        if useSqErr:
            # linear probability model
            # quadratic loss for incorrect prediction, no penalty for invalid (out of range) correct prediction
            scores[yCols] = yVals - scores[yCols]
            scores = np.clip(scores, 0, np.inf)
            scores[yCols] *= -1
            loss = 0.5 * np.dot(scores, scores)
            onlineLoss += loss
            totalOnlineLoss += loss
        else:
            pos = logistic.logcdf(scores)
            neg = logistic.logcdf(-scores)
            pos -= neg
            
            scores = logistic.cdf(scores)
            loss = -np.dot(pos[yCols], yVals)-neg.sum()
            scores[yCols] -= yVals

            onlineLoss += loss
            totalOnlineLoss += loss
 
        # 4. Compute gradient as outer product
        # this will be dense in general, unfortunately       
        g = np.outer(xVals, scores)

        # 5. Compute updated point (store it in g)            
        if useAdaGrad:
            if useSharedStep:
                n[xInds] += np.square(g).sum(1)
                etaVec = np.tile(eta/(1+np.sqrt(n[xInds])), (nl,1)).T
            else:
                n[xInds,:] += np.square(g)
                etaVec = eta/(1+np.sqrt(n[xInds,:]))
        else:
            etaVec = eta
            
        g *= -etaVec

        pos = 0
        for xI in range(xInds.size):
            xInd = xInds[xI]
            numW = wRows[xInd].size
            endPos = pos+numW
            g[xI,wRows[xInd]] += tempW[pos:endPos]
            pos = endPos

        # 6. Sparsify updated point and store it back to W
        # now g holds dense (over labels) W - eta*g        
        reassignToConvertedW(wRows, wData, xInds, g)

        # Print output periodically        
        if (t+1) % outputFreq == 0:
            bringAllUpToDate(wRows, wData, tVec, t+1)
            tVec = np.tile(t+1, nc)
            printOutputLine(subEpoch, wRows, wData, b, testX, testY, l1, l2, onlineLoss / outputFreq)
            subEpoch += 1
            onlineLoss = 0

    # print output for whole epoch
    if nr % outputFreq != 0: # otherwise we are already up to date
        bringAllUpToDate(wRows, wData, tVec, nr)
    printOutputLine("*", wRows, wData, b, testX, testY, l1, l2, totalOnlineLoss / nr)
    print

# Compute regularization value 
def getRegLoss(wData, l1, l2):
    val = 0
    for row in wData:
        val += l1 * linalg.norm(row,1)
        val += l2 / 2 * np.dot(row,row)
    return val

# compute the loss and example-based F1
def getLoss(X, wRows, wData, b, y):
    nr,nc = X.shape
    assert y.shape == (nr,nl)
    assert wRows.size == wData.size == nc
    
    loss = 0
    scores = np.ndarray(nl)
    classes = np.ndarray(nl)
    
    if useSqErr:
        thresh = 0.3
    else:
        thresh = math.log(0.3 / 0.7)
        
    totalF1 = 0
    
    for r in range(nr):
        startRow, endRow = X.indptr[r], X.indptr[r+1]
        xInds = X.indices[startRow:endRow]
        xVals = X.data[startRow:endRow]
        rowLen = endRow - startRow

        scores = np.zeros(nl)
        for (ind, val) in zip(xInds, xVals):
            weightVals = wData[ind]
            weightInds = wRows[ind]
            scores[weightInds] += val * weightVals
        scores += b
        
        positives = scores > thresh

        startRow, endRow = y.indptr[r], y.indptr[r+1]
        yInds = y.indices[startRow:endRow]
        yVals = y.data[startRow:endRow]

        if useSqErr:
            scores[yInds] = yVals - scores[yInds]
            scores = np.clip(scores, 0, np.inf)
            scores[yInds] *= -1
            loss += 0.5 * np.dot(scores, scores)
        else:
            pos = logistic.logcdf(scores)
            neg = logistic.logcdf(-scores)
            pos -= neg
                            
            loss += (-pos[yInds].dot(yVals)-neg.sum())
            
        tp = positives[yInds].sum()
        fn = (~positives)[yInds].sum()
        fp = positives.sum() - tp # tp + fp = p

        if tp > 0:
            totalF1 += (2.0 * tp) / (2.0 * tp + fn + fp)
        elif fn + fp == 0:
            totalF1 += 1

    loss /= nr
    f1Arr = totalF1 / nr

    return loss, f1Arr

# Get macro F1 and optionally output per-label F1 and label frequencies to file
def getLossMacro(X, wRows, wData, b, y, outputFilename=""):
    nr,nc = X.shape
    assert y.shape == (nr,nl)
    assert wRows.size == wData.size == nc
    
    if useSqErr:
        thresh = 0.3
    else:
        thresh = math.log(0.3 / 0.7)

    tp = np.zeros(nl, dtype="int")
    fp = np.zeros(nl, dtype="int")
    fn = np.zeros(nl, dtype="int")
    
    sZeros = 0
   
    for r in range(nr):
        startRow, endRow = X.indptr[r], X.indptr[r+1]
        xInds = X.indices[startRow:endRow]
        xVals = X.data[startRow:endRow]
        rowLen = endRow - startRow

        scores = np.zeros(nl)
        for (ind, val) in zip(xInds, xVals):
            weightVals = wData[ind]
            weightInds = wRows[ind]
            scores[weightInds] += val * weightVals
            
        sZeros = (scores == 0).sum()
        scores += b
        
        positives = scores > thresh

        startRow, endRow = y.indptr[r], y.indptr[r+1]
        yVals = y.indices[startRow:endRow]
        
        truth = np.zeros(nl, dtype="bool")
        truth[yVals] = True
        
        tps = np.logical_and(truth, positives) 
        tp[tps] += 1
        fps = np.logical_and(~truth, positives)
        fp[fps] += 1
        fns = np.logical_and(truth, ~positives)
        fn[fns] += 1
    
    nonZeros = tp > 0
    f1 = np.zeros(nl)
    f1[nonZeros] = (2.0 * tp[nonZeros]) / (2.0 * tp[nonZeros] + fp[nonZeros] + fn[nonZeros])
    goodZeros = np.logical_and(tp == 0, np.logical_and(fp == 0, fn == 0))
    f1[goodZeros] = 1
    macroF1 = np.average(f1)
    
    if outputFilename != "":        
        labFreq = y.sum(0).getA1() / nr    
        with open(outputFilename, "w") as outputFile:
            for (freq, f1val) in zip(labFreq, f1):
                outputFile.write(str(freq) + "\t" + str(f1val) + "\n")
    
    return macroF1

# split a csr_matrix into two
def split(indptr, indices, data, splitPoint):
    nc = indices.max() + 1
    nr = indptr.size - 1

    testIndptr = indptr[splitPoint:].copy()
    beginTestIdx = testIndptr[0]
    testIndices = indices[beginTestIdx:]
    testData = data[beginTestIdx:]
    testIndptr -= beginTestIdx
    indptr = indptr[:splitPoint+1]
    indices = indices[:beginTestIdx]
    data = data[:beginTestIdx]
        
    train = sp.csr_matrix((data, indices, indptr), (splitPoint, nc))
    test = sp.csr_matrix((testData, testIndices, testIndptr), (nr - splitPoint, nc))

    return train, test

# read data formatted for bioASQ
def makeBioASQData(dataFilename, testDataFilename, trainN, trainFrac, labelFrac, testN):
    assert 0 <= trainFrac <= 1
    assert not ((testDataFilename == "") and (testN == 0))
    
    if dataFilename.endswith(".gz"):
        datafile = gzip.open(dataFilename)
    else:
        datafile = open(dataFilename)
    nr = 0
    numVals = 0
    numLabVals = 0
    keeperCounter = 0
    featCounts = {}
    
    line_process_counter = 0
    
    for line in datafile:
        line_process_counter += 1
        if line_process_counter % 100 == 0:
            print "pass 1 of 4: " + str(line_process_counter)
        keeperCounter += trainFrac
        if keeperCounter < 1:
            continue
        else:
            keeperCounter -= 1
            
        splitLine = line.split('\t')
        assert (len(splitLine) == 2)

        feats = set(splitLine[0].split(' '))        
        numVals += len(feats)
        
        for feat in feats:
            intFeat = int(feat)
            if intFeat in featCounts:
                featCounts[intFeat] += 1
            else:
                featCounts[intFeat] = 1
        
        numLabVals += splitLine[1].count(' ') + 1
        
        nr += 1        
        if nr == trainN: break
    datafile.close()
   
    print "Made it past reading data file"

    Xdata = np.ndarray(numVals)
    Xindices = np.ndarray(numVals, dtype='int64')
    Xindptr = np.ndarray(nr+1, dtype="int64")
    Xindptr[0] = 0
    
    Ydata = np.ndarray(numLabVals)
    Yindices = np.ndarray(numLabVals, dtype='int64')
    Yindptr = np.ndarray(nr+1, dtype="int64")
    Yindptr[0] = 0
            
    insNum = 0
    featIdx = 0
    labIdx = 0
    keeperCounter = 0

    def addFeat(indices, data, idx, feat, count):
        indices[idx] = feat
        adjCount = featCounts[feat] - 0.5 #absolute discounting
        data[idx] = math.log1p(count) * math.log(float(nr) / adjCount)

    def addIns(splitFeats, idx, indices, data):
        intFeats = []
        for strFeat in splitFeats:
            intFeats.append(int(strFeat))
        intFeats.sort()

        startIdx = idx
        
        # add feats, using log(1+count) * log(nr/totalCount) as feature value
        count = 0
        currFeat = -1
        for feat in intFeats:
            if feat != currFeat:
                if currFeat in featCounts:
                    addFeat(indices, data, idx, currFeat, count)
                    idx +=1
                count = 1
            else:
                count += 1
            currFeat = feat
        if currFeat in featCounts:
            addFeat(indices, data, idx, currFeat, count)
            idx += 1
        
        # normalize to unit 2-norm
        xVec = data[startIdx:idx]
        xVec /= linalg.norm(xVec)
        
        return idx
        
    if dataFilename.endswith(".gz"):
        datafile = gzip.open(dataFilename)
    else:
        datafile = open(dataFilename)

    print "second datafile loop"
    second_line_counter = 0
    for line in datafile:
        second_line_counter += 1
        if second_line_counter % 100 == 0:
            print "pass 2 of 4: " + str(second_line_counter)
        keeperCounter += trainFrac
        if keeperCounter < 1:
            continue
        else:
            keeperCounter -= 1

        splitLine = line.split('\t')
        assert (len(splitLine) == 2)

        # extract feats as integers and sort
        splitFeats = splitLine[0].split(' ')

        featIdx = addIns(splitFeats, featIdx, Xindices, Xdata)
        Xindptr[insNum+1] = featIdx

        # same stuff with labels (here there should be only 1 per line) 
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
        if insNum == trainN: break
    datafile.close()
                                
    assert insNum == nr

    if testDataFilename != "":
        if testDataFilename.endswith(".gz"):
            datafile = gzip.open(testDataFilename)
        else:
            datafile = open(testDataFilename)

        testNumVals = 0
        testNumLabVals = 0
        testNR = 0
        
        third_line_counter = 0
        for line in datafile:
            third_line_counter += 1
            if third_line_counter % 100 == 0:
                print "pass 3 of 4: " + str(third_line_counter)

            splitLine = line.split('\t')
            assert (len(splitLine) == 2)
    
            feats = set(splitLine[0].split(' '))
            for feat in feats:
                if int(feat) in featCounts:
                    testNumVals += 1
            
            testNumLabVals += splitLine[1].count(' ') + 1
            
            testNR += 1        
            if testNR == testN: break
        datafile.close()
    
        testXdata = np.ndarray(testNumVals)
        testXindices = np.ndarray(testNumVals, dtype='int64')
        testXindptr = np.ndarray(testNR+1, dtype="int64")
        testXindptr[0] = 0
        
        testYdata = np.ndarray(testNumLabVals)
        testYindices = np.ndarray(testNumLabVals, dtype='int64')
        testYindptr = np.ndarray(testNR+1, dtype="int64")
        testYindptr[0] = 0
                
        insNum = 0
        featIdx = 0
        labIdx = 0
        
        if testDataFilename.endswith(".gz"):
            datafile = gzip.open(testDataFilename)
        else:
            datafile = open(testDataFilename)

        fourth_line_count = 0
        for line in datafile:
            fourth_line_count += 1
            if fourth_line_count % 100 == 0:
                print "pass 4 of 4: " +  str(fourth_line_count)

            splitLine = line.split('\t')
            assert (len(splitLine) == 2)
    
            # extract feats as integers and sort
            splitFeats = splitLine[0].split(' ')
    
            featIdx = addIns(splitFeats, featIdx, testXindices, testXdata)
            testXindptr[insNum+1] = featIdx
    
            # same stuff with labels (here there should be only 1 per line) 
            splitLabels = splitLine[1].split(' ')
            intLabels = []
            for strLab in splitLabels:
                intLabels.append(int(strLab))
            intLabels.sort()
            numLabels = len(intLabels)
            endLabIdx = labIdx + numLabels
    
            testYindices[labIdx:endLabIdx] = intLabels
            testYdata[labIdx:endLabIdx] = np.ones(numLabels)
            testYindptr[insNum+1] = endLabIdx
            labIdx = endLabIdx
    
            insNum += 1
            if insNum == testN: break
        datafile.close()
        
        assert insNum == testNR
        
        numFeats = max(featCounts.keys()) + 1
        
        print "setting CSR matrices before returning"

        X = sp.csr_matrix((Xdata, Xindices, Xindptr), (nr, numFeats))
        testX = sp.csr_matrix((testXdata, testXindices, testXindptr), (testNR, numFeats))
        
        numLab = max(Yindices.max(), testYindices.max()) + 1
        y = sp.csr_matrix((Ydata, Yindices, Yindptr), (nr, numLab))
        testY = sp.csr_matrix((testYdata, testYindices, testYindptr), (testNR, numLab))

    else:
        beginTest = nr - testN
        
        X, testX = split(Xindptr, Xindices, Xdata, beginTest)
        y, testY = split(Yindptr, Yindices, Ydata, beginTest)

    if trainN < np.inf:
        # compact to remove all zero features and labels
        # for testing only
        featTotals = X.sum(0).getA1() + testX.sum(0).getA1()
        nonZero = featTotals > 0
        nzCount = nonZero.sum()
        print "Removing %d zero features" % (nonZero.size - nzCount)
        X = sp.csr_matrix(X.todense()[:,nonZero])
        testX = sp.csr_matrix(testX.todense()[:,nonZero])
        
        labTotals = y.sum(0).getA1() + testY.sum(0).getA1()
        nonZero = labTotals > 0
        nzCount = nonZero.sum()
        print "Removing %d zero labels" % (nonZero.size - nzCount)
        y = sp.csr_matrix(y.todense()[:,nonZero])
        testY = sp.csr_matrix(testY.todense()[:,nonZero])
        
    # remove infrequent labels
    if labelFrac < 1:
        labCounts = y.sum(0).getA1()
        percentile = np.percentile(labCounts, (1-labelFrac)*100)
        keepLabs = np.where(labCounts > percentile)[0]
        y = y[:,keepLabs] 
        testY = testY[:,keepLabs]
                    
    return X, y, testX, testY
    
# set default values before reading command line
l1 = 0
l2 = 0

useBias = False
useAdaGrad = False
useSharedStep = False

profile = False
sampleWithReplacement = False
useSqErr = False
usePerm = False

useScaledAdaGrad = False

eta = 1
epochs=10

dataFilename = ""
testDataFilename = ""
modelOutputFile = ""
modelInputFile = ""

maxN=np.inf
testN=0
outFreq=np.inf
trainFrac=1
labelFrac=1

usage = """options:
    -a: use AdaGrad
    -r: sample with replacement (not looping over the data)
    -p: choose new permutation for each pass
    -d: data file (tsv format, may be gzipped, based on extension)
    -b: add fixed bias term based on base rates for each label
    -q: use squared error (default is logistic)
    -s: use shared AdaGrad step sizes for all labels
    -n: use prefix of data of this size
    -t: read at most this many test instances
    -T: number of training epochs
    -o: output frequency (if smaller than one epoch)

long options:
    --l1: weight for l1 regularization (default: 0)
    --l2: weight for l2 regularization (default: 0)
    --eta: step size (default: 1e-1)
    --profile: turn on profiling
    --trainFrac: fraction of train instances to keep
    --labelFrac: fraction of labels to keep
    --testD: test data file
    --outputFile: file to write model to
    --inputFile: file to read model from (no model will be trained)
    --scaledAdaGrad: scale AdaGrad step by sqrt(# labels)
"""

try:
    opts, args = getopt.getopt(sys.argv[1:], 
                               "arqt:n:T:bpsd:o:",
                               ["l1=","l2=","eta=","profile",
                                "trainFrac=", "labelFrac=", "testD=",
                                "outputFile=", "inputFile=", "scaledAdaGrad"])
except getopt.GetoptError:
    print usage
    sys.exit(2)
for opt, arg in opts:
    if opt in ('-h', '--help'):
        print usage
        sys.exit()
    elif opt == '-s':
        useSharedStep = True
    elif opt == '-a':
        useAdaGrad = True
    elif opt == '-r':
        sampleWithReplacement = True
    elif opt == '-q':
        useSqErr = True
    elif opt == '-p':
        usePerm = True
    elif opt == '-b':
        useBias = True
    elif opt == '-d':
        dataFilename = arg
    elif opt == '--testD':
        testDataFilename = arg
    elif opt == '-n':
        maxN = int(arg)
        assert 0 < maxN
    elif opt == '-t':
        testN = int(arg)
        assert 0 <= testN
    elif opt == '-o':
        outFreq = int(arg)
        assert 0 < outFreq
    elif opt == '-T':
        epochs = int(arg)
        assert 0 < epochs
    elif opt == '--l1':
        l1 = float(arg)
        assert 0 <= l1
    elif opt == '--l2':
        l2 = float(arg)
        assert 0 <= l2
    elif opt == '--scaledAdaGrad':
        useScaledAdaGrad = True
    elif opt == '--eta':
        eta = float(arg)
        assert 0 < eta
    elif opt == '--trainFrac':
        trainFrac = float(arg)
        assert 0 < trainFrac
    elif opt == '--outputFile':
        modelOutputFile = arg
    elif opt == '--inputFile':
        modelInputFile = arg
    elif opt == '--labelFrac':
        labelFrac = float(arg)
        assert 0 < labelFrac
    elif opt == '--profile':
        profile = True

# can't turn on shared step without AdaGrad
assert useAdaGrad or not useSharedStep

# can't turn on scaled AdaGrad without shared step
assert useSharedStep or not useScaledAdaGrad

# can't both train a model and read pre-trained model
assert not (modelOutputFile and modelInputFile)
        
print "Running with options:"
if len(dataFilename) > 0:
    print "data filename: " + dataFilename
print "useAdaGrad: " + str(useAdaGrad)
print "useSharedStep: " + str(useSharedStep)
print "useScaledAdaGrad: " + str(useScaledAdaGrad)
print "sampleWithReplacement: " + str(sampleWithReplacement)
print "useSqErr: " + str(useSqErr)
print "use fixed bias: " + str(useBias)
print "usePerm: " + str(usePerm)
print "epochs: " + str(epochs)
if maxN < np.inf:
    print "n: " + str(maxN)
if testN < np.inf:
    print "testN: " + str(testN)
if outFreq < np.inf:
    print "outputFreq: " + str(outFreq)
print "l1: %e" % l1
print "l2: %e" % l2
print "eta: %e" % eta
if trainFrac < 1:
    print "trainFrac: %e" % trainFrac
if labelFrac < 1:
    print "labelFrac: %e" % labelFrac
if modelOutputFile != "":
    print "modelOutputFile: " + modelOutputFile
if modelInputFile != "":
    print "modelInputFile: " + modelInputFile
print

# X, y, testX, testY = makeArtificialDataMulti(3, maxN, 50, 0.2, 123, testN)
# haveTestData = True

# X, y, testX, testY = makeMNISTdata(maxN, 123)
# haveTestData = True

np.random.seed(123)

X, y, testX, testY = makeBioASQData(dataFilename, testDataFilename, maxN, trainFrac, labelFrac, testN)
haveTestData = True

print ("pre-processing returned")

f_X = open("X.pickle", "w")
f_y = open("y.pickle", "w")
f_testX = open("testX.pickle", "w")
f_testY = open("testY.pickle", "w")

cPickle.dump(X, f_X)
cPickle.dump(y, f_y)
cPickle.dump(testX, f_testX)
cPickle.dump(testY, f_testY )

f_X.close()
f_y.close()
f_testX.close()
f_testY.close()

print ("wrote files to disk")

nr,nc = X.shape
nl = y.shape[1]

print str(nr) + " train instances, " + str(testX.shape[0]) + " test instances, " + str(nc) + " features, " + str(nl) + " labels."
print str(nc * nl) + " total weights."
posFrac = y.sum() / (nr * nl)
print "%f nnz feats, " % (1. * X.size / (nr * nc)),
print "%f nnz labels" % posFrac
 

# w represents the weight vector
wRows, wData = np.ndarray(nc, dtype=object), np.ndarray(nc, dtype=object)
for c in range(nc):
    wRows[c] = np.ndarray(0, np.dtype(int))
    wData[c] = np.ndarray(0, np.dtype(float))

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

if useAdaGrad:
    # n is the sum of squared gradients, used by AdaGrad
    if useSharedStep:
        n = np.zeros(nc)
    else:
        n = np.zeros((nc,nl))

if useScaledAdaGrad:
    eta *= math.sqrt(nl)

if profile:
    pr = cProfile.Profile()
    pr.enable()
    
if modelInputFile == "":
    for epoch in range(epochs+1):
        if epoch == epochs:
            break
        
        if usePerm:
            perm = np.random.permutation(nr)
       
        print "beginning traning"

        trainProx(wRows, wData, n, b, X, y, eta, l1, l2, outFreq)
    print "done training"
            
    if modelOutputFile != "":
        np.savez_compressed(modelOutputFile, b=b, wRows=wRows, wData=wData)
else:
    print "Loading input file: ", modelInputFile
    
    data = np.load(modelInputFile)
    b = data['b']
    wRows = data['wRows']
    wData = data['wData']

    print "Training set:"    
    testLoss, f1 = getLoss(X, wRows, wData, b, y)
    print "loss: %f" % testLoss
    print "per-example f1: %f" % f1
    f1 = getLossMacro(X, wRows, wData, b, y, "trainMacroF1")
    print "macro F1: ", f1

    print "Test set:"
    testLoss, f1 = getLoss(testX, wRows, wData, b, testY)
    print "loss: %f" % testLoss
    print "per-example f1: %f" % f1
    f1 = getLossMacro(testX, wRows, wData, b, testY, "testMacroF1")
    print "Test macro F1: ", f1
#     testLoss, testF1 = getLoss(testX, wRows, wData, b, testY)
#     
#     print "Test loss: ", testLoss
#     print "Test F1: ", testF1

if profile:
    pr.disable()
    s = StringIO.StringIO()
    sortby = 'cumulative'
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    ps.print_stats()
    print s.getvalue()
