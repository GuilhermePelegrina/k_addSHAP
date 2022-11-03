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
import xgboost

import seaborn as sns


def nParam_kAdd(kAdd,nAttr):
    '''Return the number of parameters in a k-additive model'''
    aux_numb = 1
    for ii in range(kAdd):
        aux_numb += comb(nAttr,ii+1)
    return aux_numb

    
def powerset(iterable,k_add):
    '''Return the powerset (for coalitions until k_add players) of a set of m attributes
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

def coalition_shap_kadd(k_add,nAttr):
    ''' Return the matrix whose rows represent coalitions of players for cardinality at most k_add '''
    k_add_numb = nParam_kAdd(k_add,nAttr)
    coal_shap = np.zeros((k_add_numb,nAttr))
    
    for i,s in enumerate(powerset(range(nAttr),k_add)):
        s = list(s)
        coal_shap[i,s] = 1
    return coal_shap

def vector_shap2game(x,k_add,nAttr,coal_shap):
    '''Return the transformation vector, associated with the coalition represented by x, from Shapley interaction indices, given a k_additive model, to game'''
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

def opt_Xbinary_wrand_allMethods(nEval, nAttr, k_add, coal_shap, X_train):
    ''' Return the matrix of coalitions used in Kernel SHAP (X), the transformation matrix used in the proposal (opt_data) '''
    # Select at random, but with probability distributions based on the SHAP weights
    weights_shap = np.zeros((nEval))
    k_add_numb = nParam_kAdd(k_add,nAttr)
    aux = []
    aux2 = np.ones((nAttr,))
    for ii in range(1,nAttr):
        aux = np.append(aux,comb(nAttr,ii)*shapley_kernel(nAttr,ii))
        aux2[ii] = aux2[ii-1]+comb(nAttr,ii)
   
    selec_data_aux = np.zeros((nEval,))
    p_aux = aux/sum(aux)
    for ii in range(nEval):
        selec_data_aux[ii] = np.random.choice(np.arange(nAttr-1)+1, size=1, replace=False, p=p_aux)
        aux[int(selec_data_aux[ii]-1)] -= shapley_kernel(nAttr,int(selec_data_aux[ii]))
        aux = np.maximum(aux,np.zeros((len(aux),)))
        p_aux = aux/sum(aux)
        
    unique, counts = np.unique(selec_data_aux, return_counts=True)
    X = np.zeros((nEval,nAttr))
    opt_data = np.zeros((nEval,k_add_numb))
    counter = 0
    for ii in range(len(unique)):
        aux = np.sort(random.sample(range(nAttr), int(unique[ii]))).reshape(1,-1)
        while np.unique(aux,axis=0).shape[0] < counts[ii]:
            aux = np.append(aux,np.sort(random.sample(range(nAttr), int(unique[ii]))).reshape(1,-1),axis=0)
        aux = np.unique(aux,axis=0)
        
        for jj in range(aux.shape[0]):
            X[counter,aux[jj,:]] = 1
            opt_data[counter,:] = vector_shap2game(X[counter,:],k_add,nAttr,coal_shap)
            
            weights_shap[counter] = shapley_kernel(nAttr,sum(X[counter,:]))
            
            counter += 1
            
    X = np.concatenate((np.concatenate((np.zeros((1,nAttr)),X),axis=0),np.ones((1,nAttr))),axis=0)
    X = np.concatenate((X,np.ones((nEval+2,1))),axis=1)
    opt_data = np.concatenate((vector_shap2game(np.zeros((nAttr,)),k_add,nAttr,coal_shap).reshape(1,-1),opt_data),axis=0)
    opt_data = np.concatenate((opt_data,vector_shap2game(np.ones((nAttr,)),k_add,nAttr,coal_shap).reshape(1,-1)),axis=0)
    weights_shap = np.append(10**6,weights_shap)
    weights_shap = np.append(weights_shap,10**6)
    
    return X, opt_data, weights_shap

def shapley_kernel(M,s):
    ''' Return the Kernel SHAP weight '''
    if s == 0 or s == M:
        return 100000
    return (M-1)/(scipy.special.binom(M,s)*s*(M-s))

def plot_waterfall(nAttr,nAttr_plot,exp_pred,values,names):
    
    if nAttr > nAttr_plot:
        values_argsort = np.abs(values[0:nAttr]).argsort()[::-1]
        values_others = np.sum(values[values_argsort[nAttr_plot:]])
        values_sort = values[values_argsort[0:nAttr_plot]]
        values_sort = np.hstack([values_sort,values_others])
        values_sort = np.hstack([exp_pred,values_sort])
        names_plot = names[values_argsort[0:nAttr_plot]]
        names_plot = names_plot.insert(0,'EXPECTED PREDICTION ($\phi_0$)')
        names_plot = names_plot.insert(len(names_plot),'others')
        nAttr = nAttr_plot + 1
    else:
        values_argsort = np.abs(values[0:nAttr]).argsort()[::-1]
        values_sort = values[values_argsort[0:nAttr]]
        values_sort = np.hstack([exp_pred,values_sort])
        names_plot = names[values_argsort[0:nAttr]]
        names_plot = names_plot.insert(0,'EXPECTED PREDICTION ($\phi_0$)')
        
    increment = np.zeros((nAttr+2,))
    increment[0:nAttr+1] = values_sort
    increment[-1] = sum(increment)
    start_point = np.zeros((len(increment)))
    position = np.zeros((nAttr+2,))
    position[0] = increment[0]
    position[-1] = sum(increment[0:-1])
    for ii in range(len(increment)-2):
        start_point[ii+1] = start_point[ii] + increment[ii]
        position[ii+1] = position[ii] + increment[ii+1]

    names_plot = names_plot.insert(len(names_plot),'PREDICTED OUTCOME')

    colors_bar = ["black"]
    for ii in increment[1:-1]:
        if ii >= 0:
            colors_bar.append("green")
        else:
            colors_bar.append("red")
    colors_bar.append("blue")

    width = 0.75
    fig = plt.figure()
    ax = fig.add_axes([0,0,1,1])
    bar_plot = ax.bar(names_plot, increment, width,bottom=start_point, color=colors_bar, edgecolor = 'black', capsize = 7)
    plt.xticks(rotation=90, fontsize=14)
    ax.set_ylim([-0.05, min(1.1, max(position)+0.08)])
    ax.set_ylabel('Feature contribution',fontsize=15)
    ii = 0
    for rect in bar_plot:
        if ii == 0:
            plt.text(rect.get_x() + rect.get_width()/2., 0.005+position[ii],'%.2f' % increment[ii],ha='center', va='bottom')
        else:
            plt.text(rect.get_x() + rect.get_width()/2., 0.005+max(position[ii],position[ii-1]),'%.2f' % increment[ii],ha='center', va='bottom')
        ii += 1
    plt.show()

 
" Defining and reading the dataset - Choose one of them"
# Red wine quality dataset
data = pd.read_csv('data_wine_quality_red.csv')
X_orig = data.drop('quality', axis=1)
y_orig = data['quality']
y_orig = (y_orig>5)*1
k_add, nEval = 3, 500

# Diabetes dataset
#X_orig, y_orig = sklearn.datasets.load_diabetes(return_X_y=True, as_frame=True)
#k_add, nEval = 3, 500

" Spliting and scalling train and test datasets "
X_train_orig, X_test_orig, y_train, y_test = train_test_split(X_orig, y_orig, test_size=0.2, random_state=42)

X_train, X_test = np.array(X_train_orig), np.array(X_test_orig)

nSamp_train, nAttr = X_train.shape # Number of samples (train) and attributes
nSamp_test = 18 # Sample of interest

" Creating the regressor - Choose one of them "
#model = RandomForestRegressor(n_estimators=1000, max_depth=None, min_samples_split=2)
model = RandomForestClassifier(n_estimators=1000, max_depth=None, min_samples_split=2)
model.fit(X_train, y_train)
score = model.score(X_test, y_test)

#model = MLPRegressor(max_iter=10**6)
#model = MLPClassifier(max_iter=10**6)
#model.fit(X_train, y_train)
#score = model.score(X_test, y_test)

" Basic elements"
k_add_numb = 1
for ii in range(k_add):
    k_add_numb += comb(nAttr,ii+1)
    
" Providing local explanations "
coal_shap = coalition_shap_kadd(k_add,nAttr)
names = X_orig.columns
nAttr_plot = 10
        
" By selecting weighted random samples (without replacement) "
X, opt_data, weights_shap = opt_Xbinary_wrand_allMethods(nEval-2, nAttr, k_add, coal_shap, X_train)

weights = np.eye(nEval)
weights[0,0], weights[-1,-1] = 10**6, 10**6
      
# For all samples
exp_payoffs_ci = np.zeros((nEval))
weights_shap_ci = np.zeros(nEval)

for ll in range(nEval):
    X_train_aux = np.array(X_train)
    X_train_aux[:,np.where(X[ll,0:-1]==1)] = X_test[nSamp_test,np.where(X[ll,0:-1]==1)]
    exp_payoffs_ci[ll] = model.predict(X_train_aux).mean()
    weights_shap_ci[ll] = shapley_kernel(nAttr,sum(X[ll,0:-1]))
    
exp_payoffs_shap = exp_payoffs_ci 
exp_payoffs_ci = exp_payoffs_ci - exp_payoffs_ci[0]

# For weighted random samples
explain_matrix = np.linalg.inv(opt_data.T @ weights @ opt_data) @ opt_data.T @ weights
inter_val = explain_matrix @ exp_payoffs_ci
shapley_ci = inter_val[1:]
shapley_val = inter_val[1:nAttr+1]
shapley_val_ci_shap = np.linalg.inv(X.T @ np.diag(weights_shap) @ X) @ X.T @ np.diag(weights_shap) @ exp_payoffs_shap
shapley_shap = shapley_val_ci_shap[0:-1]

" Plots "
# Shapley values
plt.show()
plot_waterfall(nAttr,nAttr_plot,model.predict(X_train).mean(),shapley_ci,names)

# Interaction indices
combinat = np.zeros((np.int(nAttr*(nAttr-1)/2),2))
count = 0
indices = np.zeros((nAttr,nAttr))
for ii in range(nAttr-1):
    for jj in range(ii+1,nAttr):
        combinat[count,0], combinat[count,1] = ii, jj
        indices[ii,jj] = shapley_ci[nAttr+count]
        count += 1

indices = indices + indices.T

ax = sns.heatmap(
    indices, 
    vmin=np.min(indices), vmax=np.max(indices), center=0,
    cmap=sns.diverging_palette(20, 220, n=200),
    square=True
)
plt.figure(figsize = (25,5))
ax.set_xticklabels(
    ax.get_xticklabels(),
    rotation=45,
    horizontalalignment='right'
)