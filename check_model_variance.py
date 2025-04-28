#!/usr/bin/env python

import numpy as np
from data_utils import split_train_test
from qcnico.plt_utils import histogram
import matplotlib.pyplot as plt
from ase.db import connect
import lightgbm as lgb
from models import test_lgb


path_to_db = 'data/train2.db'
db = connect(path_to_db)

model_sizes = [10,40,100,500]
nmodels = 50

for n in model_sizes:
    print(f'*** num_estimators = {n} ***')
    corr_coefs = np.zeros(nmodels)
    for k in range(nmodels):
        print(k, end = ' ')
        Xtrain, ytrain, Xtest, ytest, _ = split_train_test(db,normalize=True)
        corr_coefs[k] = test_lgb(Xtrain,ytrain,Xtest,ytest,model_type='test',num_estimators=n)
    
    fig, ax = plt.subplots()
    fig, ax = histogram(corr_coefs,bins=50,plt_objs=(fig,ax),show=False, usetex=False)
    ax.set_title(f'# of estimators = {n}')
    plt.savefig(f'/Users/nico/Desktop/figures_worth_saving/OPVGCN/r_distrib_{n}_estimators.png')
    plt.show()
        

