#!/usr/bin/env python

from ase.db import connect
import numpy as np
from qcnico.plt_utils import histogram




db = connect('data/train2.db')
feature = np.array([row.data.et1 for row in db.select()])

print(np.unique(feature).shape)

histogram(feature,bins=50)