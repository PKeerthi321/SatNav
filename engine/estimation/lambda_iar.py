import numpy as np
from dataclasses import dataclass
from typing import Optional

@dataclass
class LambdaResult:
    a_fixed:      np.ndarray
    a_fixed_2nd:  Optional[np.ndarray]
    sq_norm_1:    float
    sq_norm_2:    Optional[float]
    ratio:        Optional[float]
    success_rate: float
    accepted:     bool
    n_candidates: int

def _ldl(Q):
    n = Q.shape[0]; L = np.eye(n); d = np.zeros(n); A = Q.copy().astype(float)
    for j in range(n-1,-1,-1):
        d[j] = A[j,j]
        if d[j] <= 0: raise ValueError(f'Q not PD at {j}')
        L[j,:j] = A[j,:j]/d[j]
        for i in range(j): A[i,:i+1] -= L[j,i]*A[j,:i+1]
    return L, d

def _decorrelate(Q):
    L, d = _ldl(Q); n = L.shape[0]; Z = np.eye(n,dtype=int)
    for j in range(n-2,-1,-1):
        for i in range(j+1,n):
            mu = round(L[i,j])
            if mu != 0:
                L[i,j:i] -= mu*L[j,j:i]; Z[:,j] -= mu*Z[:,i]
    order = np.argsort(d); Z = Z[:,order]
    Qz = 0.5*(Z.T@Q@Z + (Z.T@Q@Z).T)
    Lz, dz = _ldl(Qz)
    return Z, Lz, dz

def _search(az, L, d, n_best=2):
    n = len(az); cands = []; chi2 = np.inf; ai = np.zeros(n,dtype=int)
    def _rec(k, psq):
        nonlocal chi2
        if k < 0:
            cands.append((psq, ai.copy()))
            cands.sort(key=lambda x:x[0])
            if len(cands)>n_best*3: del cands[n_best*3:]
            if len(cands)>=n_best: chi2=cands[n_best-1][0]
            return
        cf = az[k] - sum(L[j,k]*(ai[j]-az[j]) for j in range(k+1,n))
        for dlt in range(-6,7):
            c = round(cf)+dlt; diff = c-cf; sq = psq+diff**2/d[k]
            if sq >= chi2: continue
            ai[k] = c; _rec(k-1, sq)
    _rec(n-1, 0.0)
    return sorted(cands, key=lambda x:x[0])[:n_best]

def _sr(d):
    from math import erf, sqrt
    r = 1.0
    for di in d: r *= erf(1.0/(2.0*sqrt(di)*sqrt(2.0)))
    return float(r)

def lambda_search(a_float, Q_aa, ratio_threshold=3.0):
    a = np.asarray(a_float,dtype=float).ravel()
    Q = np.asarray(Q_aa,dtype=float)
    Z, L, d = _decorrelate(Q)
    Zi   = np.round(np.linalg.inv(Z)).astype(int)
    az   = Zi @ a
    sr   = _sr(d)
    cands= _search(az, L, d)
    if not cands:
        return LambdaResult(np.round(a).astype(float),None,0.,None,None,sr,False,0)
    af = Z @ cands[0][1].astype(float); sq1 = cands[0][0]
    sq2=None; a2=None; ratio=None
    if len(cands)>=2:
        sq2=cands[1][0]; a2=Z@cands[1][1].astype(float)
        ratio = sq2/sq1 if sq1>1e-12 else np.inf
    accepted = ratio is not None and ratio>=ratio_threshold
    return LambdaResult(af,a2,sq1,sq2,ratio,sr,accepted,len(cands))

def recover_n1_n2_from_wl_nl(n_wl, n_nl):
    n1=(n_nl+n_wl)/2.; n2=(n_nl-n_wl)/2.
    if abs(n1-round(n1))>0.01 or abs(n2-round(n2))>0.01:
        raise ValueError(f'Non-integer: N_WL={n_wl} N_NL={n_nl}')
    return int(round(n1)), int(round(n2))
