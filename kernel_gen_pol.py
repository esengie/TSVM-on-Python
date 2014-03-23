
import numpy as np

def kernel_gen_pol(args):
    # Local Variables: a, k, args, b
    # Function calls: kernel_gen_pol
    #% KERNEL_GEN_POL Generates a polynomial kernel.
    #%
    #% Input:
    #%       args(2,1) or args(1,2) = Kernel arguments (see code)
    #%
    #% Output:
    #%       k = Kernel function handle
    #%%%%%%%%%%%%%%%%%
    #% Define kernel %
    #%%%%%%%%%%%%%%%%%
    a = args[0]
    b = args[1]
    #%%%%%%%%%%%%%%%%%
    #% Return kernel %
    #%%%%%%%%%%%%%%%%%
    def kernel_pol(x1, x2):

	# Local Variables: x2, x1, d
	# Function calls: a, kernel_pol, b, dot
	d = np.power(np.dot(x1, x2)+a, b)
	return d
    k = kernel_pol
    return k