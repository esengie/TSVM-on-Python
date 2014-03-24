import numpy as np
import time
from svm_trn import svm_trn

def svm_ova(X, Y, N, C1, C2, k):
      #% SVM_OVA Solves the multiclass problem with SVMs (one vs all).
      #%
      #% Input:
      #%       X(n,d)  = Input data points
      #%       Y(n,1)  = Input data labels
      #%       N(~,d)  = Input data points (neutral)
      #%       C1(1,1) = Misclassification penalty (labeled data)
      #%       C2(1,1) = Misclassification penalty (neutral data)
      #%       k = Kernel function handle
      #%
      #% Output:
      #%       f = Classification function handle
      #%       SX(s,d) = Support vector points
      #%       SY(s,1) = Support vector labels
      #%       SA(s,1) = Support vector alphas (Lagrange multipliers)
      #%       t(1,1)  = Computation time in seconds

    #%%%%%%%%%%%%%%%%%%%%%%
    #% Auxiliary function %
    #%%%%%%%%%%%%%%%%%%%%%%
    def dsep(y):
        #% Prepare separation
        c0 = np.sum(Y == y)
        X1 = np.zeros((c0,d))
        c1 = 0
        X2 = np.zeros((n-c0,d))
        c2 = 0
        #% Perform separation
        for q in range(n):
            if Y[q] == y:
		X1[c1] = X[q]
                c1 += 1
            else:
		X2[c2] = X[q]
                c2 += 1
	return X1, X2    
    #% Start timer
    t2 = time.time()
    #% Get input size
    n, d = np.shape(X)
    #% Prepare algorithm
    c_list = np.unique(Y)
    c_size = np.shape(c_list)[0]
    f_list = [None]*c_size
    
    for i in range(c_size):
	#% Prepare data
	X1,X2 = dsep(c_list[i])
	    
	#% Learn classifier
	f,SX_new,SY_new,SA_new,useless = svm_trn(X1,X2,N,C1,C2,k)
	#% Store classifier
	f_list[i] = f
	if i == 0:
	    SX = SX_new    #% Support vector points
	    SY = SY_new    #% Support vector labels
	    SA = SA_new    #% Support vector alphas
	#% Add support vectors
	else:
	    SX = np.vstack([SX,SX_new])
	    SY = np.hstack([SY,SY_new])
	    SA = np.hstack([SA,SA_new])
	    
    #%%%%%%%%%%%%%%%%%%%%%
    #% Define classifier %
    #%%%%%%%%%%%%%%%%%%%%%
       
    def f_fun(x):
        #% Reset maximum
        max_j = 0
        max_v = f_list[0](x)

        #% Fetch maximum
        for j in range(1,c_size):
            #% Fetch value
            v = f_list[j](x)

            #% Check value
            if v > max_v:
                max_j = j
                max_v = v
                
        #% Assign class
        y = c_list[max_j]
        return y

    #% Assign classifier
    f = f_fun

    #% Stop timer
    t = time.time() - t2   
    
    return f, SX, SY, SA, t
