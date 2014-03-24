import numpy as np
import time

def svm_hps(X1, Y1, N, X2, Y2, mul, gen, arg, gsize, depth):

    # Local Variables: acc, gsize, Y2, prm, N, X2, depth, t, arg, Y1, mul, X1, gen
    # Function calls: svm_hps
    #% SVM_HPS Learnes a TSVM model based on the given training data,
    #% multi-mode, kernel, and kernel parameters and determines its
    #% accuracy by classifying the given test data.
    #%
    #% If size(arg,2) == 1, then arg specifies the parameters
    #% C1 = arg(1) (misclassification penalty for labeled data)
    #% C2 = arg(2) (misclassification penalty for neutral data)
    #% and the kernel function parameters in arg(3:end,1).
    #%
    #% If size(arg,2) == 2, then arg specifies the parameter ranges with
    #% min_i = arg(i,1) and max_i = arg(i,2) for each parameter p_i where
    #% p_1 = C1, p_2 = C2 and p_3... specify the kernel parameters. An
    #% optimal parameter setting will be searched with a grid-based
    #% heuristic search.
    #%
    #% Input:
    #%       X1(n1,2) = Training data points
    #%       Y1(n1,1) = Training data labels
    #%       N (n2,2) = Training data points (neutral)
    #%       X2(n3,2) = Test data points
    #%       Y2(n3,1) = Test data labels
    #%       mul = Multiclass SVM function handle
    #%       gen = Kernel generator function
    #%       arg(d1,d2) = Kernel generator parameters (or ranges)
    #%       gsize(1,1) = Grid size
    #%       depth(1,1) = Search depth
    #%
    #% Output:
    #%       acc(1,1) = Accuracy (0.0 to 1.0)
    #%       prm(d1,d2) = Parameter setting
    #%       t(1,1) = Computation time in seconds
    
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    #% Define accuracy function %
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    def accuracy(f):
        #% Prepare counter
        c = 0
        #% Check test data
        for q in t_range:
            if f(X2[q]) == Y2[q]:
                c += 1
        #% Determine accuracy
        a = c / float(n3)
	return a
                    
    #% Prepare accuracy function
    n3 = np.shape(X2)[0]
    t_range = range(n3)

    #% Check arguments
    d1,d2 = np.shape(arg)			

    if d2 == 1:
      #% Generate kernel
      k = gen(arg[2:d1-1])

      #% Learn classifier
      f,usel,usel2,usel3,t = mul(X1,Y1,N,arg[0],arg[1],k)

      #% Determine accuracy
      acc = accuracy(f)
      prm = arg
    else:
      #% Start timer
      t2 = time.time()

      #%%%%%%%%%%%%%%%%%%%%%%%%%%%%
      #% Prepare parameter search %
      #%%%%%%%%%%%%%%%%%%%%%%%%%%%%

      #% Verify unlabeled data
      if np.shape(N)[0] == 0:
        karg = 2
        arg = np.vstack([arg[0],arg[2:]])
        d1 -= 1
      else:
        karg = 3
    
      #% Set search variables
      prmgrd = np.zeros([d1,gsize])
      optprm = np.zeros(d1)
      tmpprm = optprm
      optacc = -1e-10

      prm = np.ones(d1)
      
      #%%%%%%%%%%%%%%%%%%%%%%%%%%%
      #% Define optimum function %
      #%%%%%%%%%%%%%%%%%%%%%%%%%%%

      def optimum(pos, optacc, tmpprm, optprm):
	  if pos >= d1:
	      if max(tmpprm != prm):
		  #% Learn classifier
		  C1 = tmpprm[0]

		  if karg == 3:
		      C2 = tmpprm[1]
		  else:
		      C2 = 0
		  f, us1, us2, us3, us4 = mul(X1,Y1,N,C1,C2,gen(tmpprm[karg-1:]))

		  #% Determine accuracy
		  tmpacc = accuracy(f)
		  if tmpacc > optacc:
		      optacc = tmpacc
		      optprm = tmpprm
                
            
	  else:
	      #% Recursive callback
	      for q in range(gsize):
		  #% Set this parameter
		  tmpprm[pos] = prmgrd[pos,q]

		  #% Set next parameter
		  optacc, optprm = optimum(pos+1, optacc, tmpprm, optprm)

		  #% Check accuracy
		  if optacc == 1:
		      break
	  return optacc, optprm
    
      #%%%%%%%%%%%%%%%%%%%%%%%%%%%%
      #% Perform parameter search %
      #%%%%%%%%%%%%%%%%%%%%%%%%%%%%

      for d in range(depth):

        #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        #% Step 1: Instantiate parameter grid %
        #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        for i in range(d1):
            #% Prepare instantiation
            range1 = (arg[i,1] - arg[i,0]) / 2.
            start = arg[i,0] + range1 / 2.
            width = range1 / float(gsize-1)

            #% Perform instantiation
            for j in range(gsize):
                prmgrd[i,j] = start + j * width
            
        #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        #% Step 2: Search optimal parameters %
        #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        

        #% Perform search
        acc, prm = optimum(0, optacc, tmpprm, optprm)   

        #% Check accuracy
        if acc == 1:
            break


        #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        #% Step 3: Update parameter ranges %
        #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        for i in range(d1):
            #% Prepare centering
            bias = (arg[i,1] - arg[i,0]) / 4.

            #% Perform centering
            arg[i,0] = optprm[i] - bias
            arg[i,1] = optprm[i] + bias

      #% Include C2 value
      if karg == 2:
         prm = np.hstack([prm[0], 0 ,prm[1:]])
    
      #% Stop timer
      t = time.time() - t2    
    
    return acc, prm, t
