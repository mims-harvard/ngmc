NG-MC
===========

Epistatic miniarray profile (E-MAP) is a popular large-scale gene interaction discovery platform. E-MAPs benefit from quantitative output, which makes it possible to detect subtle interactions. However, due to the limits of biotechnology, E-MAP studies fail to measure genetic interactions for up to 40% of gene pairs in an assay. 

Network-guided matrix completion (NG-MC) is a knowledge-assisted method for imputing and predicting interactions in E-MAP-like data sets. The core part of NG-MC is low-rank probabilistic matrix completion that considers additional knowledge presented as a collection of gene networks. 

NG-MC assumes that interactions are transitive, such that latent gene interaction profiles inferred by NG-MC depend on the profiles of their direct neighbors in gene networks. As the NG-MC inference algorithm progresses, it propagates latent interaction profiles through each of the networks and updates gene network weights towards improved prediction. 

This repository contains supplementary material for *Data imputation in epistatic MAPs by network-guided matrix completion* by Marinka Zitnik and Blaz Zupan. 

Usage 
-----
	
Fitting network-guided matrix completion with default parameters::

	>>> from ngmc import Ngmc
	>>> from data import loader
	>>> G, gene2idx = loader.load_surma_emap("data/surma-mmc7/S-Scores-lipid-E-MAP.csv")
	>>> ngmc = Ngmc(G, c=60)
	>>> ngmc.fit()

For complete example see ``example.py`` or run::

    $ python example.py

Citing
------

::

    @article{Zitnik2015,
      title     = {Data imputation in epistatic {MAP}s by network-guided matrix completion},
      author    = {{\v{Z}}itnik, Marinka and Zupan, Bla{\v{z}}},
      journal   = {Journal of Computational Biology},
      volume    = {22},
      number    = {6},
      pages     = {595-608},
      year      = {2015},
      publisher = {Mary Ann Liebert, Inc}
    }
