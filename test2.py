
#svm_ovo
import numpy as np
import kernel_gen_pol as ff
import svm_ovo as svm1
import svm_ova as svm2

nnm = ff.kernel_gen_pol([2,5])
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

x1 = np.array([[3,3],[1,7],[2,9], [1,6]])
y1 = np.array([1,2,3,2])


f, SX, SY, SA, t = svm1.svm_ovo(x1, y1, n, C1, C2, nnm)
f, SX, SY, SA, t = svm2.svm_ova(x1, y1, n, C1, C2, nnm) 
