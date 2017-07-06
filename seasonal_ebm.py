'''
Code to run the seasonal EBM in non-dimensional terms.

The seasonal model is fully implemented in climlab. Here we just need some wrappers 
to express the non-dimensional parameters in dimensional terms
and some logic to do the parameter sweeps over radiative forcing to build hysteresis curves
for the seasonal model.
'''

import numpy as np
import climlab
from climlab import constants as const


dimensional_ref = {'Tf': -2, 
                   'A': 200.,
                   'B': 2.,
                   'a0': 1. - 0.68,
                   'a2': 0.}

def S0_from_q(q):
    return 4 * q * (dimensional_ref['A'] + dimensional_ref['B'] * dimensional_ref['Tf']) / (1-dimensional_ref['a0'])

def q_from_S0(S0):
    return S0/4*(1-dimensional_ref['a0'])/(dimensional_ref['A'] + dimensional_ref['B'] * dimensional_ref['Tf'])

def D_from_delta(delta):
    return delta * dimensional_ref['B']

def depth_from_gamma(gamma):
    return dimensional_ref['B'] * gamma * const.seconds_per_year /2/np.pi/const.cw / const.rho_w

def ai_from_alpha(alpha):
    return 1 - (1-dimensional_ref['a0'])*(1-alpha)

def dimensional_params(q, delta, alpha, gamma):
    '''Translate non-dimensional seasonal EBM parameters to the numerical model'''
    param =    {}
    for key, value in dimensional_ref.iteritems():
        param[key] = value
    param['ai'] = ai_from_alpha(alpha)
    param['D'] = D_from_delta(delta)
    param['S0'] = S0_from_q(q)
    param['water_depth'] = depth_from_gamma(gamma)
    return param    

def equivalent_ice_latitude_cap(ice_area):
    return np.rad2deg(np.arcsin(1-ice_area))

def equivalent_ice_latitude_belt(ice_area):
    return np.rad2deg(np.arcsin(ice_area))

def equivalent_ice_latitude(ice_area, belt=False):
    if belt:
        return equivalent_ice_latitude_belt(ice_area)
    else:
        return equivalent_ice_latitude_cap(ice_area)

def do_full_sweep(obliquity=23.45, alpha=0.44, delta=0.31, gamma=50., 
                  num_lat=90, q_initial=1.25, dq=0.01, years=200., coarse_factor=10., verbose=False):
    model = initialize_model(num_lat, delta, alpha, gamma, obliquity, q_initial)
    ice_area_down, q_down, model_last_stable_ice_edge = sweep_down(model, dq, years, verbose)
    #  Use a coarser step for the snowball branch
    ice_area_up, q_up = sweep_up(model, dq*coarse_factor, years, verbose)
    #  Sweep the stable branch if any
    if model_last_stable_ice_edge:
        ice_area_stable_up, q_stable_up = sweep_up(model_last_stable_ice_edge, dq, years, verbose)
    else:
        ice_area_stable_up = None
        q_stable_up = None
    return ice_area_down, q_down, ice_area_up, q_up, ice_area_stable_up, q_stable_up

def initialize_model(num_lat=90, delta=0.31, alpha=0.44, gamma=50., obliquity=23.45, q=1.2):
    orb = {'ecc':0., 'long_peri':0., 'obliquity':obliquity}
    param = dimensional_params(q, delta, alpha, gamma)
    param['orb'] = orb
    model = climlab.EBM_seasonal(num_lat=num_lat, **param)
    model.integrate_years(20, verbose=False)
    return model

def sweep_down(model, dq, years, verbose=False):
    q = q_from_S0(model.subprocess['insolation'].S0)
    if verbose:
        print 'Starting sweep down from initial q = {}'.format(q)
    snowball = False
    last_stable_ice_edge = None
    previous_stable_ice_edge = None
    qlist = []
    ice_area_list = []
    while not snowball:
        qlist.append(q)
        model.subprocess['insolation'].S0 = S0_from_q(q)
        model.integrate_years(years, verbose=False)
        model.integrate_years(1., verbose=False)
        ice_area_list.append(model.timeave['ice_area'])
        if verbose:
            print 'q = {}'.format(q)
            print 'ice_area = {}'.format(model.timeave['ice_area'])
        if (model.timeave['ice_area'] == 1.):
            snowball = True
        else:
            try:
                previous_stable_ice_edge = climlab.process_like(last_stable_ice_edge)
            except:
                pass
            last_stable_ice_edge = climlab.process_like(model)
            q -= dq
    if previous_stable_ice_edge is None:
        return ice_area_list, qlist, last_stable_ice_edge
    else:
        return ice_area_list, qlist, previous_stable_ice_edge

def sweep_up(model, dq, years, verbose=False):
    q = q_from_S0(model.subprocess['insolation'].S0)
    if verbose:
        print 'Starting sweep up from initial q = {}'.format(q)
    ice_free = False
    qlist = []
    ice_area_list = []
    while not ice_free:
        qlist.append(q)
        model.subprocess['insolation'].S0 = S0_from_q(q)
        model.integrate_years(years, verbose=False)
        model.integrate_years(1., verbose=False)
        ice_area_list.append(model.timeave['ice_area'])
        if verbose:
            print 'q = {}'.format(q)
            print 'ice_area = {}'.format(model.timeave['ice_area'])
        if (model.timeave['ice_area'] == 0.):
            ice_free = True
        else:
            q += dq
    return ice_area_list, qlist
