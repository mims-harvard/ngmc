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


def load_surma_emap(path):
    print "Reading: %s" % path
    f = open(path)
    f.readline()
    names = np.array(f.readline().strip().split(',')[1:])
    f.close()
    names = [g.strip().split()[0] for g in names]
    g2idx = {g: i for i, g in enumerate(names)}
    nc = len(names)
    uc = list(xrange(1, nc+1))
    conv = {i: lambda x: float(x.strip()) if x else np.nan for i in xrange(1, nc+1)}
    G = np.loadtxt(path, delimiter=',', skiprows=2, usecols=uc, converters=conv)
    G = np.ma.masked_array(G, np.isnan(G))
    Gma = G-np.ma.min(G)
    Gma = np.ma.multiply(Gma, 1./np.ma.max(Gma))
    return Gma, g2idx