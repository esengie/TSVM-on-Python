
#svm_trn
import numpy as np
import svm_trn as tt
import svm_ind as ind
import kernel_gen_lin as ff

nnm = ff.kernel_gen_lin()
x1 = np.zeros((4,1))
x2 = np.zeros((4,1))
x3 = np.zeros((4,1))
x1[0]=1
x1[1]=2
x1[2]=4
x1[3]=4
x2[0]=50
x2[1]=29
x2[2]=42
x2[3]=40
x3[0]=100
x3[1]=1
x3[2]=12
x3[3]=46
C1 = 10
C2 = 5
g, h, i, o, io = tt.svm_trn(x1,x2,x3,C1,C2,nnm)
 

#svm_ind
n1 = np.shape(x1)[0]
n2 = np.shape(x2)[0]

X= np.vstack([x1,x2])
Y = np.hstack([np.ones(n1), -np.ones(n2)])
C = C1 * np.ones(n1+n2)

f, SX, SY, SA, t = ind.svm_ind(X, Y, C, nnm)
 