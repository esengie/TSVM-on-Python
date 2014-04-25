from svm_hps import *
from svm_ovo import *
from kernel_gen_pol import *
from kernel_gen_lin import *
from F1 import *

from pandas import *
import numpy as np

x2 = np.array(read_csv("../data/titanic/test_samples.csv"))
y2 = np.array(read_csv("../data/titanic/test_answers.csv"))

x1 = np.array(read_csv("../data/titanic/train_samples.csv"))
y1 = np.array(read_csv("../data/titanic/train_answers.csv"))

n = np.array([])

acc, prm, t = svm_hps(x1,y1,n,x2,y2,svm_ovo,kernel_gen_pol, np.array([[0.1,100.],[0.,0.],[0.,10.],[0.,5.]]),3,5)
 
nnm = kernel_gen_pol(prm[2:])

f, SX, SY, SA, t = svm_ovo(x1, y1, n, prm[0], prm[1], nnm)

preds = map(f, x2)

y2 = pd.DataFrame(y2)
preds = pd.DataFrame(preds)

y2.to_csv("test_answers.csv", index = False)
preds.to_csv("pred_answers.csv", index = False)
