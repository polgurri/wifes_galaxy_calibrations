 
# -- Velocity finder -- #

import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from astropy.io import fits
import os
from scipy.special import erf
import scipy.optimize as opt

from scipy.ndimage import uniform_filter
import numdifftools as nd

from time import time
import pickle

import calibration as cal




def velocity_map( cube, z0):

    cube.z0 = z0 # cube.z0 = 0.073

    #t0 = time()

    initial_z = cross_corr_z_finder( cube )
    # find an initial guess for z at every spaxel

    vel_, vel_corr_, vel_unc_, sn_, vel0, width_final_ = fine_z_fitting( cube, initial_z )

    #print("Vel Map done! (%.2f sec.)" % (time() - t0))

    vel = np.reshape(vel_,cube.rawdata.shape[1:])
    vel_unc = np.reshape(vel_unc_,cube.rawdata.shape[1:])
    sn = np.reshape(sn_,cube.rawdata.shape[1:])
    width_final = np.reshape(width_final_,cube.rawdata.shape[1:])

    return vel, vel_unc, sn, width_final



# -- FINE FITTING FUNCTIONS ---------------------------------------------------

def fine_z_fitting( cube, initial_z ):

    z0 = cube.z0

    data_contsub_stack = cube.stack - cube.continuum
    # stack of signal to noise after removing the continuum
    # shape like [ fib_num, wl ]

    ivar_stack = 1. / cube.varstack
    # stack of inverse variance
    # shape like [ fib_num, wl ]

    wl = cube.wl
    # cube wavelenght

    z_final     = np.zeros(initial_z.shape)
    width_final = np.zeros(initial_z.shape)
    vel_unc     = np.zeros(initial_z.shape)
    sn          = np.zeros(initial_z.shape)

    for ii, zi in enumerate( initial_z ):

        if np.isfinite( zi ):

            data_spec = data_contsub_stack[ : , ii ]
            ivar_spec = ivar_stack[ : , ii ]

            lnL = lambda x : total_neg_loglh( x, data_spec, ivar_spec, wl, z0 )

            kms = 0.00033356409519815205 # 100kms in zi space
            bnds = ((zi - kms, zi + kms), (.5, 5.))

            res = opt.minimize( lnL, [ zi , 1.5 ], 
                tol = 1e-10, bounds = bnds) 
                #,method = 'Nelder-Mead', )

            sn_threshold = combined_line_sn( data_spec, ivar_spec, wl, res.x[0], res.x[1])

            sn[ ii ] = sn_threshold

            if sn_threshold > 1.5:

                # -- find hessian to find uncertanties --
                if np.isfinite( res.x[0] ):

                    try:
                        hessian = nd.Hessian( lnL, step = [ 1e-4 ] )( res.x )
                        fisher = np.linalg.inv( hessian )
                        uncerts = np.sqrt( np.diag( np.abs( fisher ) ) )
                        
                        if np.isfinite(uncerts[ 0 ]):
                            vel_unc[ ii ] = uncerts[ 0 ] * 299792458 / 1000
                            z_final[ ii ], width_final[ ii ] = res.x
                        else:
                            z_final[ ii ], width_final[ ii ], vel_unc[ ii ] = np.NaN, np.NaN,np.NaN

                    except RuntimeWarning:
                        print('problem with uncertanties')

            else:
                z_final[ ii ], width_final[ ii ], vel_unc[ ii ] = np.NaN, np.NaN,np.NaN



    vel = z_final * 299792458 / 1000
    vel_diff = vel - z0 * 299792458 / 1000

    mean_vel = np.mean( vel_diff[ np.isfinite( vel_diff ) ] )
    vel0 = z0 * 299792458 / 1000 - mean_vel
    vel_corr = vel_diff - mean_vel

    return vel, vel_corr, vel_unc, sn, vel0, width_final






# -- LIKELIHOOD FUNCTIONS  ----------------------------------------------------

def total_neg_loglh( param, data_spec, ivar_spec, wl, z0 ):


    z, width = param # unpack parameters of fit


    lp = lnprior( z, width, z0 )
    # compute the priors of the parameters
    # in case of flat priors, lp will be either 0 or -inf

    if not np.isfinite( lp ):
        return np.inf

    return lp - neg_loglh( data_spec, ivar_spec, wl, z, width )
    # if priors are ok, compute the negative-loglikelihood


def lnprior( z, width, z0 ):
   # set priors on fitting parameters
    
    if  width  < 0 :
        # Assume flat prior of possitive width 
        return -np.inf

    if  np.abs(z - z0) > 0.002 :
        # Assume flat prior in velocity space => vel +- 6e5 km/h 
        return -np.inf
    
    #if z < -1000:
        #print('a') 

    return 0.0

def neg_loglh( data, ivar, wl, z, width ):
    # negative log-likelihood
    # without modeling outliers
    
    prob = np.nansum( 
        -.5 * ( data - fitting_model( data, ivar, wl, z, width ) )**2 * ivar  
        )

    return prob



def fitting_model( data_spec, ivar_spec, wl, z, width ):


    wl_v, weight = emission_lines( wl, z, False )
    # find red_shifted emission lines within wl_range

    model = np.zeros( len( wl ) )
    # final model, with dimensions of ( wl )
    

    for i in range( len( wl_v )):

        norm_model = gaussian_line_profile( wl , wl_v[ i ], width )
        # generate a normalised gaussian model g( wl, center, sigma )

        A = np.nansum( data_spec * norm_model * ivar_spec ) / np.nansum( norm_model**2 * ivar_spec )

        # compute analytic amplitude of model based on data
        if (A > 0) & (weight[i] > 0):
            model +=  A * norm_model

        if (A < 0) & (weight[i] < 0):
            model +=  A * norm_model 

        # add contribution of each line

    return model













def combined_line_sn( data_spec, ivar_spec, wl, z, width ):
    
    f = data_spec
    isig2 = ivar_spec

    wl_v, weight = emission_lines( wl, z, False )

    sn = 0.0

    for i in range( len( wl_v )):

        g = gaussian_line_profile( wl , wl_v[ i ], width )
        # generate a normalised gaussian model g( wl, center, sigma )

        A = np.nansum( g * f * isig2 ) / np.nansum( g * g * isig2 )

        #plt.plot(A * g)

        A_sig = np.sqrt( np.nansum( g ) / np.nansum( g * g * isig2 ) )


        if (A > 0) & (weight[i] > 0):
            sn += (A / A_sig)**2

        if (A < 0) & (weight[i] < 0):
            sn += (A / A_sig)**2 

    return np.sqrt( sn )















def cross_corr_z_finder( cube ):

    z_grid = np.linspace( cube.z0 - 0.005, cube.z0 + 0.005, 100)
    # redshift values that we will try
    # in velocity space => vel +- 1500 km/h

    data_contsub_stack = cube.stack - cube.continuum
    # stack of signal to noise after removing the continuum
    # shape like [ fib_num, wl ]

    ivar_stack = 1. / cube.varstack
    # stack of inverse variance
    # shape like [ fib_num, wl ]

    line_width = 3.
    # we assume line to have a width of 3 wl units ( 3 * 1.253418 Amstrongs )


    # -- do a cross_correlation between mock redshifted spectra and our data

    cross_corr = np.zeros( [ len( z_grid ),
        cube.rawdata.shape[1]*cube.rawdata.shape[2] ] )
    # initialize a zeros array




    for zi, z in enumerate( z_grid ):
        
        mock_lines = emission_templates( z, cube.wl, line_width, True )
        # mock_lines is an array of shape [ num_wave ]
        
        cross_corr[zi] = mock_lines.dot( 
            ( data_contsub_stack * np.sqrt( ivar_stack ) )
            )
        

    z_corr = z_grid[ np.argmax( cross_corr, axis = 0 ) ]

    return z_corr

































# -----------------------------------------------------------------------------
# -- MOCK EMISSION-LINE SPECTRUM GENERATOR ------------------------------------
# -----------------------------------------------------------------------------


def emission_templates( z, wl, line_width, weighted ):

    wl_v, weights = emission_lines( wl, z, weighted )
    # find emission lines within wl_range

    line_profiles = np.array([
        weights[i] * gaussian_line_profile(wl, wl_v[i], line_width)
        for i in range( len( wl_v ) ) ])
    # create a gaussian emission line for each wl according to given weight

    templates = np.sum( line_profiles, axis = 0)
    # add all lines into one spectrum

    return templates




def emission_lines( wl, z, weighted ):

    d_wl = ( wl[1] - wl[0] ) / 2
    wledges = np.append( wl - d_wl, wl[-1] + d_wl )
    # create edges from cube wavelenghts

    # -- vacuum emission lines according SDSS -- 
    wl_vac = np.array([ 

        2799.117, # Mg_II
        3727.092, # O_II
        3934.777, # K (absortion)
        3969.588, # H (absortion)
        4102.890, # H_delt
        4305.610, # G (absortion)
        4341.680, # H_gam
        4862.680, # H_bet
        4960.295, # O_III
        5008.240, # O_III
        5176.700, # Mg (absortion)
        5895.600, # Na (absortion)
        6549.860, # N_II
        6564.610, # H_alp
        6585.270, # N_II
        6718.290, # S_II
        6732.670, # S_II
        8500.360, # CaII (absortion)
        8544.440, # CaII (absortion)
        8664.520  # CaII (absortion)


        ])

    wl_air = wl_vac / ( 1.0 + 2.735182e-4 + 131.4182 / wl_vac**2 + 
        2.76249e8 / wl_vac**4 )
    # transform vacuum wl to air wl with the IAU standard convetion

    z_wl_emission = wl_air * ( 1 + z )
    # shift wl to proper z

    weights = np.ones(20)
    if weighted:
        # -- weights of lines according to SDSS --
        weights = np.array([

            1., # Mg_II
            5., # O_II
            -1., # K (absortion)
            -1., # H (absortion)
            .5, # H_delt
            -1., # G (absortion)
            1., # H_gam
            2., # H_bet
            2., # O_III
            3., # O_III
            -1., # Mg (absortion)
            -1., # Na (absortion)
            3., # N_II
            8., # H_alp
            3., # N_II
            3., # S_II
            3., # S_II
            -1., # CaII (absortion)
            -1., # CaII (absortion)
            -1.  # CaII (absortion)

            ])
        
    if np.isnan(z_wl_emission).any():
        wl_range = [np.zeros(20)>1]
    else:
        wl_range =  (( z_wl_emission > wledges[0] ) & ( z_wl_emission < wledges[-1] ))

    # check which lines fall into our wl_range
    
    emission_waves = z_wl_emission[ wl_range ]
    final_weights = weights[ wl_range ]
    # return only wl and weights of lines that fall within our data wl

    return emission_waves, final_weights




def gaussian_line_profile( wl, center, width ):

    d_wl = ( wl[1] - wl[0] ) / 2
    wledges = np.append( wl - d_wl, wl[-1] + d_wl )
    # create edges from cube wavelenghts
    
    integrated_line = 0.5 * erf( (wledges - center) / (np.sqrt(2) * width ) )
    # this is analytic for a gaussian with total area = 1
    # and so should remain accurate for marginally/barely resolved lines
    
    return np.diff( integrated_line )



















