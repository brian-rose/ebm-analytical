'''
ebm_analytical.py

Python implementation of the analytical solution to the
non-dimensional annual-mean diffusive Energy Balance Model
as described in 
Rose, Cronin and Bitz (Astrophys. J.)
'''

import numpy as np
import scipy
from mpmath import mp, fp   # needed for complex special functions
from numpy.lib.scimath import sqrt as csqrt  # complex square root
import scipy.stats as stats
import scipy.integrate as integrate
from scipy.optimize import brentq

### Functions to describe the insolation and its dependence on obliquity
def P2(x):
    return 1/2. *(3*x**2-1)

def P2prime(x):
    return 3.*x

def P2primeprime(x):
    return 3. * np.ones_like(x)

def sbar(beta, x):
    return 1+s20(beta)*P2(x)

def s20(beta):
    return 5./16.*(3*np.sin(beta)**2 - 2.)

def s11(beta):
    return -2*np.sin(beta)

def s22(beta):
    return 15./16. * np.sin(beta)**2

def s_seasonal(tau, x, beta):
    return 1 + s11(beta)*np.cos(tau)*x + (s20(beta) + s22(beta)*np.cos(2*tau)) * P2(x)

def s2(beta):
    return s20(beta)

def beta_from_s2(s2):
    return arcsin(sqrt(2./3. + 16./15.*s2))

def s11_from_s2(s2):
    return sqrt(8./3. + 64./15.*s2)

def s22_from_s2(s2):
    return s2 + 5./8.

###  The seasonal temperature solutions of the linear model
##     with Fourier-series expansion of the seasonal insolation

def Phi11(gamma, delta):
    return np.arctan(gamma/(1+2*delta))

def T11(gamma, delta, q, beta):
    return q*s11(beta)/(1+2*delta)/np.sqrt(1+(gamma/(1+2*delta))**2)

def Phi22(gamma, delta):
    return np.arctan(2*gamma/(1+6*delta))

def T22(gamma, delta, q, beta):
    return q*s22(beta)/(1+6*delta)/np.sqrt(1+(2*gamma/(1+6*delta))**2)

###  Functions for the effective coalbedo and freezing threshold
###   of the annual mean model

def abar(Tstar, alpha, chi):
    return 1 - chi * alpha / np.pi

def chi(Tstar, T11x):
    return np.arccos((1-Tstar)/T11x)

def atilde(Tstar, T11x, alpha, delta, q, sbar):
    return 1 - alpha/np.pi*(chi(Tstar, T11x) - (1+2*delta)/q/sbar*np.sqrt(T11x**2 - (1-Tstar)**2))

def Tfstar(T11x, delta, q, sbar):
    return 1- (1+2*delta)*T11x**2/2/q/sbar

####  Functions describing the limits of the linear (constant albedo) solutions
####   of the annual mean model (ice-free and Snowball)

def Twarm(x, q, delta, s2):
    return q*(1+s2*P2(x)/(1+6*delta))

def Twarmprime(x, q, delta, s2):
    return q*s2*P2prime(x)/(1+6*delta)

def qwarm(delta, s2):
    return np.where(s2<0, 1./(1 +s2/(1+6*delta) ), 1./(1-s2/(1+6*delta)/2))

def qsnow(delta, s2, alpha):
    return np.where(s2<0, 1./(1-alpha)/(1-s2/(1+6*delta)/2),
        1./(1-alpha)/(1+s2/(1+6*delta)))

###  Conversion between from annual mean to seasonal nondimensional q values
###   using the effective freezing threshold temperature from the seasonal solution

def qseas(q, xs, delta, beta, gamma):
    return 1./(1./q + (s11(beta))**2 * xs**2 * (1+2*delta)/sbar(beta,xs)/((1+2*delta)**2+gamma**2))

###  Functions required for the solution with interior ice edge

def nu(delta):
    return -1/2.*(1+sqrt(1.-4./delta))

def P(x, delta):
    return hyp2f1((1+nu(delta))/2., -nu(delta)/2., 1, 1-x**2)

def f(x, delta):
    return hyp2f1(-nu(delta)/2., (1+nu(delta))/2, 1./2., x**2)

def Pprime(x, delta):
    return -x/2/delta * hyp2f1((3+nu(delta))/2., 1-nu(delta)/2., 2, 1-x**2)

def fprime(x, delta):
    return x/delta * hyp2f1(1-nu(delta)/2., (3+nu(delta))/2., 3./2., x**2)

def GP(x, delta):
    return P(x,delta) / Pprime(x,delta)

def Gf(x, delta):
    return f(x,delta) / fprime(x,delta)

def F0(x, delta, s2):
    PoverPprime = GP(x,delta)
    foverfprime = Gf(x,delta)
    joe = P2(x) - P2prime(x)*PoverPprime
    return (1./(1.-PoverPprime/foverfprime))*(1+s2/(1+6*delta)*joe)

def F1(x, delta, s2):
    PoverPprime = GP(x,delta)
    foverfprime = Gf(x,delta)
    joe = P2(x) - P2prime(x)*foverfprime
    return (1./(1.-foverfprime/PoverPprime))*(1+s2/(1+6*delta)*joe)

def q(xs, delta, s2, alpha):
    return np.where(s2<0, real(1./(1+s2/(1+6*delta)*P2(xs)-alpha*F0(xs,delta,s2))),
             #  These are actually identical ways to give the high obliquity solution
             # real(1./((1-alpha)*(1+s2/(1+6*delta)*P2(xs))+alpha*F0(xs,delta,s2)))
              real(1./(1+s2/(1+6*delta)*P2(xs)-alpha*F1(xs,delta,s2))))

###  Functions required for the ice edge stability condition

def dqinvdx(xs, delta, s2, alpha):
    return np.where(s2<0,
            real(s2*P2prime(xs)/(1+6*delta) - alpha * F0prime(xs,delta,s2)),
            real((1-alpha)*s2*P2prime(xs)/(1+6*delta) + alpha * F0prime(xs,delta,s2))
                   )

def F0prime(x, delta, s2):
    gp = GP(x,delta)
    gf = Gf(x,delta)
    gpprime = GPprime(x,delta)
    gfprime = Gfprime(x,delta)
    return (s2/(1+6*delta)*(P2prime(x)*(1-gpprime)-3*gp)*(gf**2-gp*gf) +
            (1+s2/(1+6*delta)*(P2(x)-P2prime(x)*gp))*(gpprime*gf-gp*gfprime))/(gf - gp)**2

def GPprime(x, delta):
    return 1 - P(x,delta)*Pprimeprime(x,delta)/Pprime(x,delta)**2

def Gfprime(x, delta):
    return 1 - f(x,delta)*fprimeprime(x,delta)/fprime(x,delta)**2

def Pprimeprime(x, delta):
    return (Pprime(x,delta)/x + x**2*(1+6*delta)/8/delta**2 *
            hyp2f1((5+nu(delta))/2., 2-nu(delta)/2., 3, 1-x**2))

def fprimeprime(x, delta):
    return (fprime(x,delta)/x + x**2*(1+6*delta)/3/delta**2 *
            hyp2f1(2-nu(delta)/2., (5+nu(delta))/2., 5./2., x**2))

def stable(xs, delta, s2, alpha):
    '''Heaviside function for ice edge stability'''
    return np.where(s2*dqinvdx(xs,delta,s2,alpha) > 0, 1., 0.)

def dqdx(xs, delta, s2, alpha):
    return -q(xs,delta,s2,alpha)**2 * dqinvdx(xs,delta,s2,alpha)

###  Functions for the critical alpha value
###  which defines the edge of the stable region of parameter space
def alpha_crit_low(xs, delta, s2):
    return real(s2/(1+6*delta)*P2prime(xs)/ F0prime(xs,delta,s2))

def alpha_crit_high(xs, delta, s2):
    return 1/(1-1/alpha_crit_low(xs,delta,s2))

def alpha_crit(xs, delta, s2):
    return np.where(s2<0, alpha_crit_low(xs,delta,s2), alpha_crit_high(xs,delta,s2))

def q_crit_low(xs, delta, s2):
    return real(1./(1+s2/(1+6*delta)*(P2(xs)-P2prime(xs)*F0(xs,delta,s2)/
                                      F0prime(xs,delta,s2))))

def q_crit_high(xs, delta, s2):
    return real(1./((F0(xs,delta,s2)-
                     (1+6*delta)/s2/P2prime(xs)*F0prime(xs,delta,s2)-
                     P2(xs)/P2prime(xs)*F0prime(xs,delta,s2))/
                    (1-(1+6*delta)/s2*F0prime(xs,delta,s2)/P2prime(xs))))

def q_crit(xs, delta, s2):
    return np.where(s2<0, q_crit_low(xs,delta,s2), q_crit_high(xs,delta,s2))

def alpha_stab_warm(delta, s2):
    '''Find the value of alpha for which q(alpha_crit) == q_free.'''
    a = 0.00001
    b = 0.99999
    try:
        xcrit = brentq(lambda x: (  q(x,delta,s2,alpha_crit(x,delta,s2)) 
                                  - qwarm(delta,s2)), a, b)
        return alpha_crit(xcrit, delta, s2)
    except:
        return None
    
def alpha_stab_cold(delta, s2):
    ''' Find the value of alpha for which q(alpha_crit) == q_snow'''
    a = 0.00001
    b = 0.99999
    try:
        xcrit = brentq(lambda x: ( q(x,delta,s2,alpha_crit(x,delta,s2)) 
                                  - qsnow(delta,s2,alpha_crit(x,delta,s2))), a, b)
        return alpha_crit(xcrit, delta, s2)
    except:
        return None

def alpha_stab(delta, s2):
    '''Find the larger of alpha_stab_cold and alpha_stab_warm'''
    return np.max([alpha_stab_warm(delta,s2), alpha_stab_cold(delta,s2)])

def alpha_max(xs, delta, s2):
    '''Find the largest alpha value that permits accessible stable ice edge solutions'''
    return np.minimum(alpha_stab(delta, s2)*np.ones_like(xs), alpha_crit(xs,delta,s2))

###  To implement the required complex hypergeometric function,
###  use the mpmath library
#  Don't actually need the high precision of the mp methods here
#  But if we want them, just change fp to mp below
sqrt = csqrt
sin = np.sin
arcsin = np.arcsin
##  Here we wrap the mpmath fp.hyp2f1 function with numpy so it is vectorized
hyp2f1 = np.frompyfunc(fp.hyp2f1, 4, 1)
### When we take the real part, return the result as a numpy array of type float
def real(z):
    return np.asarray(np.frompyfunc(fp.re, 1, 1)(z)).astype('float')

###  Implementation of the probability calculation for stable ice edges
###  Need to define some PDFs for model parameters

def h_delta_0(delta):
    '''Lognormal distribution with shape parameter 1.0, scale parameter 1.0 and location parameter 0.
    (mode at delta = 0.37, median at delta = 1.)'''
    return stats.lognorm.pdf(delta, 1.)

def h_delta_1(delta):
    '''Lognormal distribution with shape parameter 2.0, scale parameter e and location parameter 0.
    (mode at delta = exp(-3), median at delta = exp(1))'''
    return stats.lognorm.pdf(delta, 2., scale=np.exp(1.))

def h_delta_2(delta):
    return h_delta_1(delta)

def h_q_0(q):
    '''Lognormal distribution with shape parameter 0.5, scale parameter 1.0 and location parameter 0.
    (mode at q=0.78, median at q=1)'''
    return stats.lognorm.pdf(q, 0.5)

def h_q_1(q):
    return h_q_0(q)

def h_q_2(q):
    return h_q_1(q)

def h_alpha_0(alpha):
    '''Uniform distribution between 0 and 1'''
    return np.ones_like(alpha)

def h_alpha_1(alpha):
    return h_alpha_0(alpha)

def h_alpha_2(alpha):
    '''Parabolic beta distribution centered at 0.5'''
    a = 2.
    b = 2.
    return stats.beta.pdf(alpha,a,b)

#  Three sets of assumptions about the PDFs
PDF_set = [(h_delta_0, h_q_0, h_alpha_0),
           (h_delta_1, h_q_1, h_alpha_1),
           (h_delta_2, h_q_2, h_alpha_2)]
