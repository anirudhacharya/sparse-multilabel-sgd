
#!/apollo/env/MLEnvImprovement/bin/python

'''
Created on Jul 22, 2014

@author: zack

This code runs sgd2.py's implementation of proximal stochastic gradient descent with AdaGrad for very large, sparse, multilabel problems.

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


haveTestData = True



f_X = open("../Data/X.pickle")
f_y = open("../Data/y.pickle")
f_testX = open("../Data/testX.pickle")
f_testY = open("../Data/testY.pickle")

X = cPickle.load(f_X)
y = cPickle.load(f_y)
testX = cPickle.load(f_testX)
testY = cPickle.load(f_testY )

f_X.close()
f_y.close()
f_testX.close()
f_testY.close()

print ("Loaded data into main memory cand closed files")

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

