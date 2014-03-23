
import numpy as np

def kernel_gen_rbf(sigma):

    # Local Variables: k, sigma
    # Function calls: kernel_gen_rbf
    #% KERNEL_GEN_RBF Generates a radial basis function kernel.
    #%
    #% Input:
    #%       sigma(1,1) = Kernel argument (see code)
    #%
    #% Output:
    #%       k = Kernel function handle
    #%%%%%%%%%%%%%%%%%
    #% Define kernel %
    #%%%%%%%%%%%%%%%%%
    def kernel_rbf(x1, x2):

	# Local Variables: x2, x1, d
	# Function calls: norm, sigma, exp, kernel_rbf
	d = np.exp((- np.linalg.norm((x1-x2))**2.)/ (2.*sigma**2.))
	return d 
    #%%%%%%%%%%%%%%%%%
    #% Return kernel %
    #%%%%%%%%%%%%%%%%%
    k = kernel_rbf
    return k