import numpy as np
from hmmlearn import hmm
import pandas as pd
import glob
import matplotlib.pyplot as plt

eps = 1e-6
N_train = 40     # Number of training signatures
N_imit = 20     # Number of imitated signatures

path = r'C:\Users\ahmad\Desktop\Ahmad\PA\exe04\pa_mobisig\user_0' # Use your path
all_files = glob.glob(path + "/*.csv")
li = []
leng = []
for filename in all_files:
    df = pd.read_csv(filename, index_col=None, header=0)
    li.append(df)
    leng.append(len(df))
lengths = np.asarray(leng)
t_leng = np.copy(lengths[len(lengths)-N_train:])    # Training lengths
frame = pd.concat(li, axis=0, ignore_index=True)
Data = frame.to_numpy()
t_r = np.sum(lengths) - np.sum(t_leng)       # Training rows
X_vector = np.zeros((np.sum(lengths), 7))
X_vector[:,0] = np.copy(Data[:,0])
X_vector[:,1] = np.copy(Data[:,1])
X_vector[:,2] = np.copy(Data[:,2])
X_vector[:,3] = np.arctan(Data[:,6] / (Data[:,5] + eps))
X_vector[:,4] = np.sqrt(np.power(Data[:,5], 2) + np.power(Data[:,6], 2))
X_vector[:,5] = np.log(abs(np.power(np.power(Data[:,5], 2) + np.power(Data[:,6], 2), 1.5) / (Data[:,8] * Data[:,5] - Data[:,7] * Data[:,6] + eps)) + eps)
X_vector[:,6] = np.sqrt(np.power(Data[:,7], 2) + np.power(Data[:,8], 2))
training_X = np.copy(X_vector[t_r:,])

N_states = 2
init_prob = np.zeros(N_states)
init_prob[0] = 1
model_1 = hmm.GaussianHMM(n_components=N_states, covariance_type='diag', startprob_prior=init_prob, params='mc', init_params='mc')
model_2 = hmm.GMMHMM(n_components=N_states, n_mix=32, covariance_type='diag', startprob_prior=init_prob, params='mcw', init_params='mcw')
model_1.fit(X_vector[t_r:,], t_leng)
model_2.fit(X_vector[t_r:,], t_leng)
scores_1 = np.empty((0, 0))
scores_2 = np.empty((0, 0))
l = 0
for ll in lengths:
    scores_1 = np.append(scores_1, model_1.score(X_vector[l:l+ll-1,]))
    scores_2 = np.append(scores_2, model_2.score(X_vector[l:l+ll-1,]))
    l = l + ll

N_signs = np.arange(1, len(lengths)+1, 1)
print()
print('******************* GaussianHMM ******************')
plt.figure(figsize=(15,10))
plt.scatter(N_signs[:N_imit], scores_1[:N_imit], c='red')
plt.scatter(N_signs[N_imit:len(lengths)-N_train], scores_1[N_imit:len(lengths)-N_train], c='limegreen')
plt.scatter(N_signs[len(lengths)-N_train:], scores_1[len(lengths)-N_train:], c='blue')
plt.show()
plt.close() 
print()
print('******************* GMMHMM ******************')
plt.figure(figsize=(15,10))
plt.scatter(N_signs[:N_imit], scores_2[:N_imit], c='red')
plt.scatter(N_signs[N_imit:len(lengths)-N_train], scores_2[N_imit:len(lengths)-N_train], c='limegreen')
plt.scatter(N_signs[len(lengths)-N_train:], scores_2[len(lengths)-N_train:], c='blue')
plt.show()
plt.close() 
print() 

rates = []
N_Hs = []
N_Ms = []
for i in range(3):
    for j in range(7):
        N_H = 2**i   # Number of states
        N_M = 2**j   # Number of Gaussian mixtures per state
        init_probs = np.zeros(N_H)
        init_probs[0] = 1
        model = hmm.GMMHMM(n_components=N_H, n_mix=N_M, covariance_type='diag', startprob_prior=init_probs, params='mcw', init_params='mcw')
        model.fit(X_vector[t_r:,], t_leng)
        scores = np.empty((0, 0))
        l = 0
        for ll in lengths:
            scores = np.append(scores, model.score(X_vector[l:l+ll-1,]))
            l = l + ll
        rate = np.mean(scores[N_imit:len(lengths)-N_train]) / np.mean(scores[:N_imit])
        print()
        print('H =', N_H, '   M =', N_M, '   Rate =', rate)
        N_Hs.append(N_H)
        N_Ms.append(N_M)
        rates.append(rate)
print()
print('*************************************************')
print('The best HMM parameters are : ', 'H =', N_Hs[rates.index(max(rates))], '   M =', N_Ms[rates.index(max(rates))])        
print('*************************************************')     

