
import numpy as np

def kernel_gen_sig(args):

    # Local Variables: a, k, args, b
    # Function calls: kernel_gen_sig
    #% KERNEL_GEN_SIG Generates a sigmoidal kernel.
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
    def kernel_sig(x1, x2):
      
	# Local Variables: x2, x1, d
	# Function calls: a, tanh, b, dot, kernel_sig
	d = np.tanh((np.dot(a, np.dot(x1, x2))-b))
	return d
    #%%%%%%%%%%%%%%%%%
    #% Return kernel %
    #%%%%%%%%%%%%%%%%%
    k = kernel_sig
    return k
