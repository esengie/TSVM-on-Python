#import cvxopt as cv
#import cvxopt.solvers
from svm_ind import svm_ind
import numpy as np
import time

def svm_trn(X1, X2, X3, C1, C2, k):
    #%%%%%%%%%%%%%%%%%%%%%%
    #% Auxiliary function %
    #%%%%%%%%%%%%%%%%%%%%%%
    def is_switched(i,j):
        #% Prepare search
        s = 0
        #% Perform search
        if cnt > 0:
            for q in np.arange(0, cnt):
                if switched_ij[q,0] == i and switched_ij[q,1] == j:
                    s = 1
                    break
        return s
    # Local Variables: SY, SX, Y, f, k, C, n3, n2, n1, t, X, C2, X2, X3, SA, X1, C1
    # Function calls: svm_trn, ones, svm_ind, size
    #% SVM_TRN Computes a separating hyperplane (transductive SVM).
    #%
    #% Input:
    #%       X1(~,d) = Input data points (class 1, d-dimensional)
    #%       X2(~,d) = Input data points (class 2, d-dimensional)
    #%       X3(~,d) = Input data points (class 3, d-dimensional, neutral)
    #%       C1(1,1) = Misclassification penalty factor (for labeled data)
    #%       C2(1,1) = Misclassification penalty factor (for neutral data)
    #%       k = Kernel function handle
    #%
    #% Output:
    #%       f = Classification function handle (f(x) = k(w,x)+b)
    #%       SX(s,d) = Support vector points
    #%       SY(s,1) = Support vector labels
    #%       SA(s,1) = Support vector alphas (Lagrange multipliers)
    #%       t(1,1)  = Computation time in seconds
    #% Start timer
    t2 = time.time()
    #% Get input size
    n1 = np.shape(X1)[0]
    n2 = np.shape(X2)[0]
    n3 = np.shape(X3)[0]
    #% Get input data
    X = np.vstack([X1,X2])
    Y = np.hstack([np.ones(n1), -np.ones(n2)])
    C = C1 * np.ones(n1+n2)
    #% Learn initial model
    f, SX, SY, SA, t = svm_ind(X, Y, C, k)
   
    if n3 > 0:
	#% Sign neutral data
	Y3 = np.zeros(n3)
	for i in np.arange(0, n3):
	    Y3[i] = f(X3[i])
	#% Sign neutral data
	Y3 = np.sign(Y3)
	#% Save neutral data
	X = np.vstack([X,X3])
	Y = np.hstack([Y, Y3])
	
	#%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	#% Learn transductive model %
	#%%%%%%%%%%%%%%%%%%%%%%%%%%%%

	min_i = n1 + n2
	max_i = n1 + n2 + n3 - 1

	switched_ij = np.zeros((1,2))

	for c in np.arange(-5, 1):
	    #% Compute penalty factor
	    C3 = (C2* 10**c) * np.ones(n3)

	    #% Perform learning algorithm
	    switched = 1
	    cnt = 0

	    while switched:
		#% Get new classifier
		f,SX,SY,SA,useless = svm_ind(X,Y,np.hstack([C,C3]),k);

		#% Switch data labels
		switched = 0

		for i in np.arange(min_i, max_i):
		    for j in np.arange(i + 1, max_i+1):
			#% Fetch labels
			y1 = Y[i]
			y2 = Y[j]

			#% Check labels
			if y1 != y2 and not is_switched(i,j):
			    #% Fetch values
			    f1 = f(X[i])
			    f2 = f(X[j])

			    #% Check values
			    if max([0,1-y1*f1]) + max([0,1-y2*f2]) > max([0,1-y2*f1]) + max([0,1-y1*f2]):
				#% Switch labels
				Y[i] = y2
				Y[j] = y1

				#% Store indices
				switched_ij[cnt] = [i,j]
				cnt = cnt + 1
				switched_ij = np.concatenate((switched_ij,np.zeros((1,2))), axis=0)
				#% Leave loop
				switched = 1
				break
                        
		    if switched:
			break
    #% Stop timer
    t = time.time() - t2
    
    return f,SX,SY,SA,t
