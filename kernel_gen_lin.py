
import numpy as np

def kernel_gen_lin():
    def kernel_lin(x1, x2):

      # Local Variables: x2, x1, d
      # Function calls: kernel_lin, dot
      d = np.dot(x1, x2)
      #%%%%%%%%%%%%%%%%%
      #% Return kernel %
      #%%%%%%%%%%%%%%%%%
      return d
    # Local Variables: k
    # Function calls: kernel_gen_lin
    #% KERNEL_GEN_LIN Generates a linear kernel.
    #%
    #% Output:
    #%       k = Kernel function handle
    k = kernel_lin
    #%%%%%%%%%%%%%%%%%
    #% Define kernel %
    #%%%%%%%%%%%%%%%%%
    return k
