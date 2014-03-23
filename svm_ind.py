import cvxopt as cv
import cvxopt.solvers
import numpy as np 
import time

def svm_ind(X, Y, C, k):
  
    # Local Variables: SY, SX, f, k, C, t, Y, X, SA
    # Function calls: svm_ind
    #% SVM_IND Computes a separating hyperplane (inductive SVM).
    #%
    #% Input:
    #%       X(n,d) = Input data points (d-dimensional)
    #%       Y(n,1) = Input data labels (data labeling)
    #%       C(n,1) = Misclassification penalty factors
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
    #%%%%%%%%%%%%%%%%%%%%%%%%
    #% Perform optimization %
    #%%%%%%%%%%%%%%%%%%%%%%%%
    n, d = np.shape(X)
    Z = np.zeros(n)
    H = np.zeros((n, n))
    for i in np.arange(0, n):
	for j in np.arange(i, n):
	    H[i,j] = k(X[i, :], X[j, :])
    Q = np.outer(Y,Y) * H
    G = np.vstack([-np.eye(n), np.eye(n)])
    h = np.hstack([-Z, C])
    Am = cv.matrix(Y, (1,n))  #+np.triu(H, 1).conj().T????
    cvxopt.solvers.options['show_progress'] = False
    sol = cvxopt.solvers.qp(cv.matrix(Q+np.triu(Q, 1).conj().T), cv.matrix(-np.ones(n)), cv.matrix(G), cv.matrix(h), Am, cv.matrix(.0)) # np.array([]), optimset('Display', 'off', 'LargeScale', 'off'))
    A = np.ravel(sol['x'])
    #%%%%%%%%%%%%%%%%%%%%%%%
    #% Get support vectors %
    #%%%%%%%%%%%%%%%%%%%%%%%
    sv = A > 1e-5
    ind = np.arange(len(A))[sv]
    SA = A[sv]
    SX = X[sv]
    SY = Y[sv]
    
    #%%%%%%%%%%%%%
    #% Compute b %
    #%%%%%%%%%%%%%
    b = 0.
    for n in range(len(SA)):
	b += SY[n]
	b -= np.sum(SA * SY * H[ind[n],sv])
    
    b /= len(SA)
    
    def f_fun(x):

	# Local Variables: y, x, q
	# Function calls: s_range, b, f_fun, k, SY, SX, SA
	#% Compute class
	y = b
	for q in range(len(SA)):
	    y += SA[q] * SY[q] * k(SX[q, :], x)
        
	return y
	
    #% Assign classifier
    f = f_fun
    #% Stop timer
    t = time.time() - t2
    
    return f, SX, SY, SA, t