'''
python script to do a big-ass calculation
of the probability of finding stable ice edges as a function of obliquity

This version uses automated 2D numerical quadrature to integrate over xs and alpha
but manual numerical integration over arrays of delta values.

We do the calculation with 3 different sets of pdfs for the planetary parameters
and for each set, we do two different calculations:
- probability computed over all possible stable ice edges
- probability computed only over accessible stable ice edges

Need to start up the ipyparallel session before running this...

salloc -p snow-1 -O -n 64 --exclusive
[ssh to appropriate snow node]
cd /network/rit/lab/roselab_rit/ebm_analytical
ipcluster start -n 63 &
python compute_ice_probability_v3.py
'''
import numpy as np
import scipy.integrate as integrate
import ebm_analytical as ebm
import ipyparallel as ipp


def compute_Lice_parallel_0_all(thiss2):
    import numpy as np
    import ebm_analytical as ebm
    import scipy.integrate as integrate
        
    #  Romberg's rule for integration
    #  requires 2**k + 1 equally spaced samples
    k = 5
    delta = np.linspace(0.01, 3., 2**k+1)
    res = np.zeros_like(delta)
    #  PDF set 0
    h_delta, h_q, h_alpha = ebm.PDF_set[0]
    #  all stable states
    condition_func = ebm.alpha_crit
    for nd, d in enumerate(delta):
        result = integrate.dblquad(lambda alpha, xs: h_q(ebm.q(xs,d,thiss2,alpha)) * h_alpha(alpha), 
                                       0, 1, lambda xs: 0, lambda xs: condition_func(xs,d,thiss2))
        res[nd] = result[0]
    # Take probability-weighted integral across all values of delta
    return integrate.romb(res* h_delta(delta), dx=np.diff(delta)[0])

def compute_Lice_parallel_0_acc(thiss2):
    import numpy as np
    import ebm_analytical as ebm
    import scipy.integrate as integrate
        
    #  Romberg's rule for integration
    #  requires 2**k + 1 equally spaced samples
    k = 5
    delta = np.linspace(0.01, 3., 2**k+1)
    res = np.zeros_like(delta)
    #  PDF set 0
    h_delta, h_q, h_alpha = ebm.PDF_set[0]
    #  accessible stable states only
    condition_func = ebm.alpha_max
    for nd, d in enumerate(delta):
        result = integrate.dblquad(lambda alpha, xs: h_q(ebm.q(xs,d,thiss2,alpha)) * h_alpha(alpha), 
                                       0, 1, lambda xs: 0, lambda xs: condition_func(xs,d,thiss2))
        res[nd] = result[0]
    # Take probability-weighted integral across all values of delta
    return integrate.romb(res* h_delta(delta), dx=np.diff(delta)[0])

def compute_Lice_parallel_1_all(thiss2):
    import numpy as np
    import ebm_analytical as ebm
    import scipy.integrate as integrate
        
    #  Romberg's rule for integration
    #  requires 2**k + 1 equally spaced samples
    k = 5
    delta = np.linspace(0.01, 3., 2**k+1)
    res = np.zeros_like(delta)
    #  PDF set 1
    h_delta, h_q, h_alpha = ebm.PDF_set[1]
    #  all stable states
    condition_func = ebm.alpha_crit
    for nd, d in enumerate(delta):
        result = integrate.dblquad(lambda alpha, xs: h_q(ebm.q(xs,d,thiss2,alpha)) * h_alpha(alpha), 
                                       0, 1, lambda xs: 0, lambda xs: condition_func(xs,d,thiss2))
        res[nd] = result[0]
    # Take probability-weighted integral across all values of delta
    return integrate.romb(res* h_delta(delta), dx=np.diff(delta)[0])

def compute_Lice_parallel_2_acc(thiss2):
    import numpy as np
    import ebm_analytical as ebm
    import scipy.integrate as integrate
        
    #  Romberg's rule for integration
    #  requires 2**k + 1 equally spaced samples
    k = 5
    delta = np.linspace(0.01, 3., 2**k+1)
    res = np.zeros_like(delta)
    #  PDF set 2
    h_delta, h_q, h_alpha = ebm.PDF_set[2]
    #  accessible stable states only
    condition_func = ebm.alpha_max
    for nd, d in enumerate(delta):
        result = integrate.dblquad(lambda alpha, xs: h_q(ebm.q(xs,d,thiss2,alpha)) * h_alpha(alpha), 
                                       0, 1, lambda xs: 0, lambda xs: condition_func(xs,d,thiss2))
        res[nd] = result[0]
    # Take probability-weighted integral across all values of delta
    return integrate.romb(res* h_delta(delta), dx=np.diff(delta)[0])

def compute_Lice_parallel_2_all(thiss2):
    import numpy as np
    import ebm_analytical as ebm
    import scipy.integrate as integrate
        
    #  Romberg's rule for integration
    #  requires 2**k + 1 equally spaced samples
    k = 5
    delta = np.linspace(0.01, 3., 2**k+1)
    res = np.zeros_like(delta)
    #  PDF set 2
    h_delta, h_q, h_alpha = ebm.PDF_set[2]
    #  all stable states
    condition_func = ebm.alpha_crit
    for nd, d in enumerate(delta):
        result = integrate.dblquad(lambda alpha, xs: h_q(ebm.q(xs,d,thiss2,alpha)) * h_alpha(alpha), 
                                       0, 1, lambda xs: 0, lambda xs: condition_func(xs,d,thiss2))
        res[nd] = result[0]
    # Take probability-weighted integral across all values of delta
    return integrate.romb(res* h_delta(delta), dx=np.diff(delta)[0])

def compute_Lice_parallel_1_acc(thiss2):
    import numpy as np
    import ebm_analytical as ebm
    import scipy.integrate as integrate
        
    #  Romberg's rule for integration
    #  requires 2**k + 1 equally spaced samples
    k = 5
    delta = np.linspace(0.01, 3., 2**k+1)
    res = np.zeros_like(delta)
    #  PDF set 1
    h_delta, h_q, h_alpha = ebm.PDF_set[1]
    #  accessible stable states only
    condition_func = ebm.alpha_max
    for nd, d in enumerate(delta):
        result = integrate.dblquad(lambda alpha, xs: h_q(ebm.q(xs,d,thiss2,alpha)) * h_alpha(alpha), 
                                       0, 1, lambda xs: 0, lambda xs: condition_func(xs,d,thiss2))
        res[nd] = result[0]
    # Take probability-weighted integral across all values of delta
    return integrate.romb(res* h_delta(delta), dx=np.diff(delta)[0])


c = ipp.Client()
dview = c[:]
#  Block execution until results are complete
dview.block = True
obliquity = np.linspace(0., 90., 63)
s2array = ebm.s2(np.deg2rad(obliquity))

denom = []
Lice_all = []
for n, parallel_function in enumerate([compute_Lice_parallel_0_all, 
                                       compute_Lice_parallel_1_all, 
                                       compute_Lice_parallel_2_all]):
    #  PDF set
    h_delta, h_q, h_alpha = ebm.PDF_set[n]
    #  The denominator 
    denom.append(integrate.tplquad(lambda alpha, d, q: h_q(q) * h_delta(d) * h_alpha(alpha), 
                  0, np.infty, lambda q: 0, lambda q: np.inf,
                  lambda q,d: 0, lambda q,d: 1))

    result_all = dview.map(parallel_function, s2array)
    #  Pack result in a numpy array
    Lice_all.append(np.array(result_all[:]))
#  Save to file
np.savez('Lice_result_all.npz', denom=denom, Lice_all=Lice_all, obliquity=obliquity)

Lice_accessible = []
for n, parallel_function in enumerate([compute_Lice_parallel_0_acc, 
                                       compute_Lice_parallel_1_acc, 
                                       compute_Lice_parallel_2_acc]):
    
    result_accessible = dview.map(parallel_function, s2array)
    #  Pack result in a numpy array
    Lice_accessible.append(np.array(result_accessible[:]))
#  Save to file
np.savez('Lice_result_accessible.npz', Lice_accessible=Lice_accessible, obliquity=obliquity)
