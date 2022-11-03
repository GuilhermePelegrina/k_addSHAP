" Importing packages "
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from math import comb
from scipy.special import bernoulli

from itertools import chain, combinations
import itertools

import scipy.special
import random
import sklearn.datasets 


def nParam_kAdd(kAdd,nAttr):
    '''Return the number of parameters in a k-additive model'''
    aux_numb = 1
    for ii in range(kAdd):
        aux_numb += comb(nAttr,ii+1)
    return aux_numb

    
def powerset(iterable,k_add):
    '''Return the powerset of a set of m attributes
    powerset([1,2,..., m],m) --> () (1,) (2,) (3,) ... (m,) (1,2) (1,3) ... (1,m) ... (m-1,m) ... (1, ..., m)
    powerset([1,2,..., m],2) --> () (1,) (2,) (3,) ... (m,) (1,2) (1,3) ... (1,m) ... (m-1,m)
    '''
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(k_add+1))


def tr_shap2game(nAttr, k_add):
    '''Return the transformation matrix from Shapley interaction indices, given a k_additive model, to game'''
    nBern = bernoulli(k_add) #Números de Bernoulli
    k_add_numb = nParam_kAdd(k_add,nAttr)
    
    coalit = np.zeros((k_add_numb,nAttr))
    
    for i,s in enumerate(powerset(range(nAttr),k_add)):
        s = list(s)
        coalit[i,s] = 1
        
    matrix_shap2game = np.zeros((k_add_numb,k_add_numb))
    for i in range(coalit.shape[0]):
        for i2 in range(k_add_numb):
            aux2 = int(sum(coalit[i2,:]))
            aux3 = int(sum(coalit[i,:] * coalit[i2,:]))
            aux4 = 0
            for i3 in range(int(aux3+1)):
                aux4 += comb(aux3, i3) * nBern[aux2-i3]
            matrix_shap2game[i,i2] = aux4
    return matrix_shap2game

 
def vector_shap2game(x,k_add,nAttr,coal_shap):
    nBern = bernoulli(k_add) #Números de Bernoulli  
    k_add_numb = nParam_kAdd(k_add,nAttr)
    vector_shap2game = np.zeros((k_add_numb,))
    
    for ii in range(coal_shap.shape[0]):
        aux1 = int(sum(coal_shap[ii,:]))
        aux2 = int(sum(x * coal_shap[ii,:]))
        aux3 = 0
        for jj in range(int(aux2+1)):
            aux3 += comb(aux2, jj) * nBern[aux1-jj]
        vector_shap2game[ii] = aux3
    return vector_shap2game

def coalition_shap_kadd(k_add,nAttr):
    k_add_numb = nParam_kAdd(k_add,nAttr)
    coal_shap = np.zeros((k_add_numb,nAttr))
    
    for i,s in enumerate(powerset(range(nAttr),k_add)):
        s = list(s)
        coal_shap[i,s] = 1
    return coal_shap

def shapley_kernel(M,s):
    if s == 0 or s == M:
        return 100000
    return (M-1)/(scipy.special.binom(M,s)*s*(M-s))

 
" Defining and reading the dataset - Choose one of them"
# Red wine quality dataset
data = pd.read_csv('data_wine_quality_red.csv')
X_orig = data.drop('quality', axis=1)
y_orig = data['quality']
y_orig = (y_orig>5)*1
#k_add, nEval = 2, np.append(np.arange(120, 2001, 60),2**11)
k_add, nEval = 3, np.append(np.arange(300, 2001, 60),2**11)
nSimul = 501 # Number of simulations

# Diabetes dataset
#X_orig, y_orig = sklearn.datasets.load_diabetes(return_X_y=True, as_frame=True)
#k_add, nEval = 2, np.append(np.arange(110, 1001, 30),2**10)
#k_add, nEval = 3, np.append(np.arange(200, 1001, 30),2**10)
#nSimul = 501 # Number of simulations

" Spliting and scalling train and test datasets "
X_train_orig, X_test_orig, y_train, y_test = train_test_split(X_orig, y_orig, test_size=0.2, random_state=42)

X_train, X_test = np.array(X_train_orig), np.array(X_test_orig)

nSamp_train, nAttr = X_train.shape # Number of samples (train) and attributes
nSamp_test = X_test.shape[0] # Number of samples (test)

" Creating the regressor - Choose one of them "
#model = RandomForestRegressor(n_estimators=1000, max_depth=None, min_samples_split=2)
#model = RandomForestRegressor()
#model = RandomForestClassifier(n_estimators=1000, max_depth=None, min_samples_split=2)

#model = MLPRegressor(max_iter=10**6)
#model = MLPRegressor()
model = MLPClassifier(max_iter=10**6)
#model = MLPClassifier()

model.fit(X_train, y_train)
score = model.score(X_test, y_test)

" Basic elements"
k_add_numb = 1
for ii in range(k_add):
    k_add_numb += comb(nAttr,ii+1)
    
opt_data_all = tr_shap2game(nAttr,nAttr) # Matrix that transform all interaction indices to game
results = np.zeros((nSamp_test,len(nEval),nSimul)) # Matrix that will stores the obtained errors
results_shap = np.zeros((nSamp_test,len(nEval),nSimul)) # Matrix that will stores the obtained errors
shapley_ci = np.zeros((nSamp_test,len(nEval),nSimul,k_add_numb-1))
shapley_shap = np.zeros((nSamp_test,len(nEval),nSimul,nAttr))
shapley_val_ci = np.zeros((nAttr+1,nSamp_test))

exp_payoffs_ci = np.zeros((2**nAttr,nSamp_test))
exp_payoffs_shap = np.zeros((2**nAttr,nSamp_test))

weights_shap = np.zeros(2**nAttr)
X = np.zeros((2**nAttr,nAttr+1))
X[:,-1] = 1
for i,s in enumerate(powerset(range(nAttr),nAttr)):
    s = list(s)
    X[i,s] = 1
    weights_shap[i] = shapley_kernel(nAttr,len(s))
shapley_val_ci_aux = np.linalg.inv(X.T @ np.diag(weights_shap) @ X) @ X.T @ np.diag(weights_shap)

" Providing local explanations "
for jj in range(nSimul):
    
    for ii in range(len(nEval)): # For in the number of function's evaluations
        ''' By selecting weighted random samples (without replacement)'''
        aux_det = 0
        while aux_det == 0:
            selec_data = np.random.choice(np.arange(int(2**nAttr-2))+1, size=nEval[ii]-2, replace=False, p=weights_shap[1:-1]/sum(weights_shap[1:-1]))
            opt_data = np.concatenate((opt_data_all[0,0:k_add_numb].reshape(1,-1),opt_data_all[selec_data,0:k_add_numb],opt_data_all[-1,0:k_add_numb].reshape(1,-1)),axis=0)
            weights = np.eye(opt_data.shape[0])
            weights[0,0], weights[-1,-1] = 10**6, 10**6
            X_shap = X[selec_data,:]
            X_shap = np.concatenate((X[0,:].reshape(1,-1),X_shap),axis=0)
            X_shap = np.concatenate((X_shap,X[-1,:].reshape(1,-1)),axis=0)
            weights_shap_aux = weights_shap[selec_data]
            weights_shap_aux = np.append(weights_shap[0],weights_shap_aux)
            weights_shap_aux = np.append(weights_shap_aux,weights_shap[-1])
            
            if np.linalg.det(opt_data.T @ weights @ opt_data) != 0 and np.linalg.det(X_shap.T @ np.diag(weights_shap_aux) @ X_shap) != 0:
                aux_det += 1
        
        inter_val_aux = np.linalg.inv(opt_data.T @ weights @ opt_data) @ opt_data.T @ weights
        shapley_val_ci_shap_aux = np.linalg.inv(X_shap.T @ np.diag(weights_shap_aux) @ X_shap) @ X_shap.T @ np.diag(weights_shap_aux)
        
        for kk in range(nSamp_test):
            
            if jj == 0 and ii == 0:
                # For all samples
                exp_payoffs_ci_aux = np.zeros((2**nAttr))
                
                for i,s in enumerate(powerset(range(nAttr),nAttr)):
                    X_train_aux = np.array(X_train)
                    X_train_aux[:,s] = X_test[kk,s]
                    exp_payoffs_ci_aux[i] = model.predict(X_train_aux).mean()
                    
                exp_payoffs_shap[:,kk] = exp_payoffs_ci_aux
                exp_payoffs_ci[:,kk] = exp_payoffs_ci_aux - exp_payoffs_ci_aux[0]
                
                shapley_val_ci[:,kk] = shapley_val_ci_aux @ exp_payoffs_shap[:,kk]
                
           # For the Choquet integral
            exp_payoffs = exp_payoffs_ci[selec_data,kk]
            exp_payoffs = np.append(exp_payoffs_ci[0,kk],exp_payoffs)
            exp_payoffs = np.append(exp_payoffs,exp_payoffs_ci[-1,kk])
            
            # For the Kernel SHAP
            exp_payoffs_shap_aux = exp_payoffs_shap[selec_data,kk]
            exp_payoffs_shap_aux = np.append(exp_payoffs_shap[0,kk],exp_payoffs_shap_aux)
            exp_payoffs_shap_aux = np.append(exp_payoffs_shap_aux,exp_payoffs_shap[-1,kk])          
            
            # For weighted random samples
            inter_val = inter_val_aux @ exp_payoffs
            shapley_ci[kk,ii,jj,:] = inter_val[1:]
            shapley_val = inter_val[1:nAttr+1]
            results[kk,ii,jj] = sum((shapley_val_ci[0:-1,kk]-shapley_val)**2)/nAttr
            shapley_val_ci_shap = shapley_val_ci_shap_aux @ exp_payoffs_shap_aux
            shapley_shap[kk,ii,jj,:] = shapley_val_ci_shap[0:-1]
            results_shap[kk,ii,jj] = sum((shapley_val_ci[0:-1,kk]-shapley_val_ci_shap[0:-1])**2)/nAttr
                
            print(' Simulation: ', jj, ' from ', nSimul, 'Number of evaluations : ', ii, ' from ', len(nEval), 'Test sample : ', kk, ' from ', nSamp_test)                

" Plots "             
names=X_orig.columns
cut_in = 0
        
med_add, q25_add, q75_add = np.quantile(results[:,cut_in:,:].mean(0),0.5,axis=1), np.quantile(results[:,cut_in:,:].mean(0),0.1,axis=1), np.quantile(results[:,cut_in:,:].mean(0),0.9,axis=1)
med_shap, q25_shap, q75_shap = np.quantile(results_shap[:,cut_in:,:].mean(0),0.5,axis=1), np.quantile(results_shap[:,cut_in:,:].mean(0),0.1,axis=1), np.quantile(results_shap[:,cut_in:,:].mean(0),0.9,axis=1)

plt.show()
plt.plot(nEval[cut_in:],np.maximum(med_add,10**(-8)), label='$3_{ADD}$-SHAP',linewidth=2.5, color='blue')
plt.fill_between(nEval[cut_in:], np.maximum(q25_add,10**(-8)), np.maximum(np.minimum(q75_add,med_shap[0]+10**(-4)/2),10**(-8)),facecolor='b',alpha=0.3)
plt.plot(nEval[cut_in:],np.maximum(med_shap,10**(-8)), label='Kernel SHAP',linewidth=2.5, color='red')
plt.fill_between(nEval[cut_in:], np.maximum(q25_shap,10**(-8)), np.maximum(np.minimum(q75_shap,med_shap[0]+10**(-4)/2),10**(-8)),facecolor='r',alpha=0.3)

plt.xlabel('# of expected prediction evaluations ($n_{\mathcal{M}})$',fontsize=15)
plt.ylabel(r'Average errors ($\bar{\varepsilon}$)',fontsize=15)
plt.legend(prop={'size': 16});
