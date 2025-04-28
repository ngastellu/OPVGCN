#!/usr/bin/env python


from data_utils import make_dataset_ecfp4, split_dataset
from models import test_lgb
import numpy as np
import matplotlib.pyplot as plt


csvpath = '/Users/nico/Desktop/simulation_outputs/OPV_matls/AustinT_harvard_cep/moldata-filtered.csv'

X, y = make_dataset_ecfp4(csvpath,N=1000)
Xtr, ytr, Xte, yte = split_dataset(X,y,train_frac=0.5)

r, yte_preds = test_lgb(Xtr, ytr, Xte, yte, model_type='paper',return_preds=True)
# yte_preds = yte_preds.flatten()

print(f'***  r = {r} ***')

ref_min = np.min(yte)
ref_max = np.max(yte)
ref_pts = np.linspace(ref_min, ref_max, 20)

fig, ax = plt.subplots()

ax.plot(yte, yte_preds,'o', ms=5.0)
ax.plot(ref_pts,ref_pts, 'k--', lw=0.8)
ax.set_title('LightGBM PCE predictions from ECFP4-2048 fingerprints')
ax.set_xlabel('Target PCE')
ax.set_ylabel('Predicted PCE')
ax.set_aspect('equal')
plt.show()