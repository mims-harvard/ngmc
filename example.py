"""
Copyright (C) 2014  Marinka Zitnik <marinka.zitnik@fri.uni-lj.si>

This file is part of NG-MC.

NG-MC is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

NG-MC is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with NG-MC.  If not, see <http://www.gnu.org/licenses/>.
"""

import numpy as np
import scipy.stats as spt

from ngmc import Ngmc
from data import loader


def eval(ngmc):
    G_hat = ngmc.imputed()
    true = np.array(zip(*test_gi)[2])
    pred = np.array([(G_hat[i, j]+G_hat[j, i])/2. for i, j, _ in test_gi])
    nrmse = np.sqrt(np.mean(np.multiply(true-pred, true-pred))/np.var(true))
    cr = spt.pearsonr(true, pred)[0]
    print "NRMSE[test] = %5.4f" % nrmse
    print "Corr[test] = %5.4f" % cr


np.random.seed(42)
Gma, gene2idx = loader.load_surma_emap('data/surma-mmc7/S-Scores-lipid-E-MAP.csv')

print "Size: %d rows, %d cols" % (len(gene2idx), len(gene2idx))
print "Missing data (before test set): %5.4f" % (np.sum(Gma.mask)/float(Gma.size))

# Construct test set and hide interaction measurements
n_measur = (Gma.size-np.sum(Gma.mask))/2.
n_rem = int(0.05*n_measur)
print 'Test size: %d' % n_rem
nan_ridx, nan_cidx = np.where(Gma.mask == False)
cnds = list(set([frozenset([r, c]) for r,c in zip(nan_ridx, nan_cidx)]))
test_gi = []
for i in np.random.choice(xrange(len(cnds)-1), n_rem, replace=False):
    idx1, idx2 = cnds[i]
    test_gi.append((idx1, idx2, Gma[idx1, idx2]))
    Gma[idx1, idx2] = np.nan
    Gma[idx2, idx1] = np.nan
print "Missing data (after test set): %5.4f" % (np.sum(Gma.mask)/float(Gma.size))

ngmc = Ngmc(Gma, c=60, callback=eval)
ngmc.fit()
print "::Final evaluation after learning:"
nrmse, pearson = eval(ngmc)

