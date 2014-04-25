import numpy as np
import time
from svm_trn import svm_trn

def svm_ovo(X, Y, N, C1, C2, k):
    # Local Variables: SY, SX, d, f, t_s, k, f_size, m, N, f_list, t, C2, Y, X, SA, n, c_list, C1, c_size
    # Function calls: svm_ovo, cell, unique, size, tic
    #% SVM_OVO Solves the multiclass problem with SVMs (one vs one).
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
    def dext(y1,y2):
        #% Prepare extraction
        X1 = np.zeros((np.sum(Y == y1),d))
        c1 = 0
        X2 = np.zeros((np.sum(Y == y2),d))
        c2 = 0
        #% Perform extraction
        for q in range(n):
            if Y[q] == y1:
                X1[c1] = X[q]
                c1 += 1
            elif Y[q] == y2:
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
    f_size = (c_size * (c_size-1))/2
    f_list1 = [None]*f_size
    f_list2 = [None]*f_size
    f_list3 = [None]*f_size
    SX = X
    SY = Y
    SA = Y
    m = 0
    for i in range(c_size - 1):
	for j in range(i+1, c_size):
	    #% Prepare data
	    X1,X2 = dext(c_list[i],c_list[j])
	    
	    #% Learn classifier
	    f,SX_new,SY_new,SA_new,useless = svm_trn(X1,X2,N,C1,C2,k)
	    #% Store classifier
	    f_list1[m] = i
	    f_list2[m] = j
	    f_list3[m] = f
	    m += 1
	    if m == 1:
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
        #% Reset votes
        votes = np.zeros(c_size)

        #% Count votes
        for q in range(f_size):
            #% Fetch value
            v = f_list3[q](x)
            #% Check value
            if v > 0:
                v_index = f_list1[q]
            else:
                v_index = f_list2[q]

            #% Add voting
            votes[v_index] += 1
        #% Check votes
        max_q = 0
        max_v = votes[0]

        for q in range(1,c_size):
            if votes[q] > max_v:
                max_v = votes[q]
                max_q = q
                
        #% Assign class
        y = c_list[max_q]
        return y


    #% Assign classifier
    f = f_fun

    #% Stop timer
    t = time.time() - t2   
    
    return f, SX, SY, SA, t
