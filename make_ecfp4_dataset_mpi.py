#!/usr/bin/env python 

import numpy as np
import pandas as pd
from rdkit.Chem import AllChem, rdFingerprintGenerator
from tqdm import tqdm
from mpi4py import MPI


def ecfp4_generator(L=2048, use_chirality=True):
    return rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=L, includeChirality=use_chirality)

def make_dataset_ecfp4_mpi(datapath, rank, nprocs, L=2048, N=None, smiles_col='SMILES_str', pce_col='pce'):

    filetype = datapath.strip().split('.')[-1]
    if filetype == 'csv':
        df = pd.read_csv(datapath, usecols=[smiles_col, pce_col])
    elif filetype == 'xlsx':
        df = pd.read_excel(datapath, usecols=[smiles_col, pce_col])
    else:
        if rank == 0:
            print(f'File type .{filetype} not supported.')
        return None, None

    if N is None:
        N = len(df)

    # Setup local data
    if rank == nprocs - 1:
        inds_local = np.arange(rank*(N//nprocs),N)
    else:
        inds_local = np.arange(rank*(N//nprocs), (rank+1)*(N//nprocs))

    nrows_local = inds_local.shape[0]
    X_local = np.zeros((nrows_local , L), dtype=np.uint8)
    y_local = np.zeros(nrows_local)

    fp_gen = ecfp4_generator(L=L)

    if rank == 0:
        pbar = tqdm(total=nrows_local)  # only approximate progress for rank 0

    for k, i in enumerate(inds_local):
        dfrow = df.iloc[i]
        try:
            smiles = dfrow[smiles_col]
            molecule = AllChem.MolFromSmiles(smiles)
            if molecule is not None:
                X_local[k, :] = fp_gen.GetCountFingerprintAsNumPy(molecule)
                y_local[k] = dfrow[pce_col]
            else:
                print(f"Rank {rank}: Invalid molecule at row {k}.")
                y_local[k] = -np.inf
        except Exception as e:
            print(f"Rank {rank}: Error at row {k} -> {e}")
        if rank == 0:
            pbar.update(1)

    if rank == 0:
        pbar.close()

    invalid = (y_local == -np.inf) # mask which selects all rows that threw an exxeption
    print(f'Rank {rank}: found {invalid.sum()} invalid rows.')
    return X_local[~invalid,:], y_local[~invalid]


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nprocs = comm.Get_size()

csvpath = 'moldata-filtered.csv'

X, y = make_dataset_ecfp4_mpi(csvpath, rank, nprocs)

np.save(f'ecfp4-{rank}.npy', X)
np.save(f'pces-{rank}.npy', y)