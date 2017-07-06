'''
python script to do many parameter sweeps on the seasonal EBM
to compute hysteresis diagrams for the seasonal model

Need to start up the ipyparallel session before running this...

salloc -p snow-1 -O -n 64 --exclusive
[ssh to appropriate snow node]
cd /network/rit/lab/roselab_rit/ebm_analytical
ipcluster start -n 63 &
python seasonal_hysteresis.py
'''
import ipyparallel as ipp
import itertools
import pickle


def sweep_wrap(params):
    from seasonal_ebm import do_full_sweep
    '''Convenience wrapper so I can parallelize the sweep over a tuple of parameters'''
    ice_area_down, q_down, ice_area_up, q_up, ice_area_stable_up, q_stable_up = do_full_sweep(*params, 
                      num_lat=360, q_initial=2.0, dq=0.002, years=600., coarse_factor=1., verbose=False)
    return ice_area_down, q_down, ice_area_up, q_up, ice_area_stable_up, q_stable_up

#  Same set of parameters as we used for the annual mean model
obl = [23.45, 90.]
alpha = [0.2, 0.44, 0.7]
delta = [0.04, 0.08, 0.16, 0.32, 0.64, 2.56]
#delta = [0.16, 0.64]

#  And do a shallow and deep water version
gamma = [5., 50.]
#  Need different equilibration times for these two
#years = [50., 200.]
years = 200.

#  Make a list of parameter combinations
params = [p for p in itertools.product(obl,alpha,delta,gamma)]

c = ipp.Client()
dview = c[:]
#  Block execution until results are complete
dview.block = True

results = dview.map(sweep_wrap, params)

# Pack the results in Python pickle file
outfile = open('seasonal_results.p', 'wb')
pickle.dump((params, results), outfile)
outfile.close()
