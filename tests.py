
#svm_trn
import numpy as np
import svm_trn as tt
import svm_ind as ind
import kernel_gen_lin as ff
import svm_ovo

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
g, h, i, o, io = tt.svm_trn(x1,x2,[],C1,C2,nnm)
 

#svm_ind
n1 = np.shape(x1)[0]
n2 = np.shape(x2)[0]

X= np.vstack([x1,x2])
Y = np.hstack([np.ones(n1), -np.ones(n2)])
C = C1 * np.ones(n1+n2)

f, SX, SY, SA, t = ind.svm_ind(X, Y, C, nnm)


#svm_ovo
import numpy as np
import kernel_gen_lin as ff
import svm_ovo as svm

nnm = ff.kernel_gen_lin()
n = np.array([])
x1 = np.zeros((12,1))
y1 = np.zeros(12)
C1 = 2
C2 = 0

x1[0]=1
x1[1]=2
x1[2]=4
x1[3]=4
x1[4]=50
x1[5]=29
x1[6]=42
x1[7]=40
x1[8]=-100
x1[9]=-25
x1[10]=-14
x1[11]=-21

y1[0]=1
y1[1]=1
y1[2]=1
y1[3]=1
y1[4]=2
y1[5]=2
y1[6]=2
y1[7]=2
y1[8]=-1
y1[9]=-1
y1[10]=-1
y1[11]=-1

f, SX, SY, SA, t = svm.svm_ovo(x1, y1, n, C1, C2, nnm)