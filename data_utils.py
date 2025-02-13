from ase.db import connect
import pandas as pd
import numpy as np

def get_row_data(df,mol_type,mol_name):
    if mol_name == 'PC61BM':
        return {'LUMO': -3.70, 'LUMO+1': -3.70 + 0.077824564}
    elif mol_name == 'PC71BM':
        return {'LUMO': -3.91, 'LUMO+1': 0.033470005}

    filtered = df[df[mol_type] == mol_name]

    if filtered.empty:
        print(f'{mol_type} {mol_name} is missing from Excel sheet')
        return None
    else:
        return filtered.iloc[0].to_dict()


def get_don_acc_combined_features(don_homo,don_lumo,acc_lumo):
    edahl = acc_lumo - don_homo
    edall = don_lumo - acc_lumo
    return edahl, edall


def make_indoor_data(device_xlsx, donor_csv, acceptor_csv, normalize=False, minmax_scale=True):
    don_df = pd.read_csv(donor_csv)
    acc_df = pd.read_csv(acceptor_csv)
    df = pd.read_excel(device_xlsx)

    ndevices = len(df)
    X = np.zeros((ndevices,8))
    y = np.ones(ndevices) * -1.0

    seen_devices = set() #set of pairs of acceptor-donor pairs; avoids conflicting data points

    for k, row in df.iterrows():
        donor = row['Donor'].strip()
        acceptor = row['Acceptor'].strip()
        device = (donor,acceptor)
        
        if device in seen_devices:
            print('Already saw ', device)
            continue
        
        seen_devices.add(device)

        don_data = get_row_data(don_df,'Donors', donor)
        acc_data = get_row_data(acc_df,'Acceptors', acceptor)

        if don_data is None or acc_data is None:
            continue

        don_homo = don_data['HOMO']
        don_lumo = don_data['LUMO']
        don_dhomo = don_homo - don_data['HOMO-1']
        don_dlumo = don_data['LUMO+1'] - don_lumo
        don_et1 = don_data['DET1']
        don_nd = don_data['Nd']
        
        acc_lumo = acc_data['LUMO']
        # adl = acc_data['LUMO+1'] - acc_lumo

        edahl, edall = get_don_acc_combined_features(don_homo,don_lumo,acc_lumo)

        X[k,0] = don_homo
        X[k,1] = don_lumo
        X[k,2] = don_et1
        X[k,3] = don_dhomo
        X[k,4] = don_dlumo
        X[k,5] = float(don_nd)
        X[k,6] = edahl
        X[k,7] = edall
        # X[k,8] = adl

        y[k] = row['PCE(%)']

    good_row_filter = y>=0 # True for rows with no missing data
    X = X[good_row_filter,:]
    y = y[good_row_filter]
    
    # Rescales data to be between 0 and 1
    # Rescale before standardizing?
    if minmax_scale:
         X = (X - np.min(X,axis=0)) / (np.max(X,axis=0) - np.min(X, axis=0))

    # Standardize (zero-mean, unit varianc)
    if normalize:
        X -= X.mean(axis=0)
        X /= X.std(axis=0)
    

    return X,y


def cal_prop_cc(molo,tag):
    if molo.data.Acceptor=='PC61BM':
        al= -3.70
        adl= 0.077824564
    elif molo.data.Acceptor=='PC71BM':
        al= -3.91
        adl= 0.033470005
    if tag=='edahl':
        prop=al-float(molo.homo)
    if tag=='edall':
        prop=float(molo.lumo)-al
    if tag=='adlumo':
        prop=adl
    if tag=='nd':
        # prop=cal_nd(moln)
        prop = float(molo.nd)
    return prop


def make_cc_data(path_to_db, normalize=False, minmax_scale=True):
    db = connect(path_to_db)
    nmols = db.count()
    X = np.zeros((nmols,8))
    y = np.zeros(nmols)
    
    for k, row in enumerate(db.select()):
        print(row.data)
        X[k,0] = row.homo
        X[k,1] = row.lumo
        X[k,2] = row.et1
        X[k,3] = row.dh
        X[k,4] = row.dl
        X[k,5] = row.nd
        X[k,6] = cal_prop_cc(row,'edahl') 
        X[k,7] = cal_prop_cc(row,'edall') 

        y[k] = row.data.PCE
    

    # Rescales data to be between 0 and 1
    # Rescale before standardizing?
    if minmax_scale:
         X = (X - np.min(X,axis=0)) / (np.max(X,axis=0) - np.min(X, axis=0))

    # Standardize (zero-mean, unit varianc)
    if normalize:
        X -= X.mean(axis=0)
        X /= X.std(axis=0)
 
    return X, y