from __future__ import annotations
import numpy as np


class PairSampler:
    def __init__(self,n:int,pairs:int,seed:int):
        if n<2 or pairs<1: raise ValueError("Need n>=2 and pairs>=1")
        self.n,self.pairs,self.seed=n,min(pairs,n*(n-1)),seed
    def sample(self,epoch:int=0):
        rng=np.random.default_rng(self.seed+epoch); flat=rng.choice(self.n*(self.n-1),self.pairs,replace=False)
        i=flat//(self.n-1);j=flat%(self.n-1);j=j+(j>=i)
        return i.astype(np.int64),j.astype(np.int64)
    def diagnostics(self,epoch=0):
        i,j=self.sample(epoch);return {"pairs":len(i),"unit_coverage":float(len(np.unique(np.r_[i,j]))/self.n),"self_pairs":int(np.sum(i==j))}
