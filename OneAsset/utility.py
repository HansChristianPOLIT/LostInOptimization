from numba import njit

@njit
def utility(c,par):
    """ CRRA utility function. """
    
    return c**(1-par.rho)/(1-par.rho)

@njit
def marg_utility(c,par):
    """ Returns marginal utility given consumption level. """
    
    return c**(-par.rho)

@njit
def inv_marg_utility(q,par):
    """ Returns c, given inverse marginal utility. """
    
    return q**(-1/par.rho)