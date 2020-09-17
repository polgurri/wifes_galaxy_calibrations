

# ---------------------------------------------------------------------
# Imports

import numpy as np 
from astropy.io import fits

from scipy.interpolate import interp1d
import scipy.optimize as opt
import os



# ---------------------------------------------------------------------
# -- Perform flux calibration -----------------------------------------
# ---------------------------------------------------------------------


def stds_corr( stds_path ):


    # -- fit a gaussian to every wl to get a proper flux measurement 
    std_data_flux, std_data_wl = standard_flux( stds_path )


    ESO_star = fits.open( stds_path )[0].header['OBJECT'].replace(' ','').split('-')[0]
    ESO_path = stds_path + '/std_stars/' + ESO_star + '.dat'

    with open( ESO_path ) as hdulist:

        # -- open ESO data from txt to nparray
        _wl, _fl = zip( *[ ( float( i.split()[ 0 ] ), float( i.split()[ 1 ] ) ) 
            for i in hdulist.readlines() ] )
        eso_wl_raw = np.array( _wl )
        eso_flux_raw = np.array( _fl ) # in ( ergs/cm/cm/s/A * 10**16 )

    # -- find the ESO wl that coincides with our cube
    start = np.argmax( eso_wl_raw >= std_data_wl[ 0 ] ) - 1
    end = np.argmax( 1 - ( eso_wl_raw <= std_data_wl[ -1 ] ) ) + 1
    eso_wl = eso_wl_raw[ start:end ]
    eso_flux = eso_flux_raw[ start:end ]


    # -- calculate calibration ------------------------------------------------

    # rebin our data to eso std_data_wl
    reb_std_data_flux = rebin_transmission(eso_wl, std_data_wl, std_data_flux )
    
    # compensate for rebin error in first and last bin
    reb_std_data_flux[ 0 ] = reb_std_data_flux[ 1 ]
    reb_std_data_flux[ -1 ] = reb_std_data_flux[ -2 ]

    # compute a flux calibrator in eso std_data_wl
    eso_calb = eso_flux / reb_std_data_flux

    # interpolate calibration in ESO wl to data wl
    interp_cal = interp1d( eso_wl, eso_calb )
    data_calb = interp_cal( std_data_wl )

    # -- erase telluric -------------------------------------------------------
    wl_cal = telluric( eso_wl, std_data_wl, data_calb, eso_flux, std_data_flux)

    return wl_cal




# ---------------------------------------------------------------------
# -- Fit a Gaussian to flux -------------------------------------------
# ---------------------------------------------------------------------

def standard_flux( object_path ):
    # fit a gaussian to each wl to determine our std star observed flux

    hdulist = fits.open( object_path ) 
    size = hdulist[ 1 ].data.shape[ 0 ]

    x = np.linspace( 0.25, 37.75, size )
    y = np.linspace( 0.5, 24.5, 25 )
    xv , yv = np.meshgrid( x, y )

    header = hdulist[ 1 ].header
    wl_range = header[ 'CRVAL1' ] + (  np.arange( header[ 'NAXIS1' ], dtype='d' )- 0.  ) * header[ 'CDELT1' ]
    exp_time = header[ 'EXPTIME' ] 
    # air_mass = header['AIRMASS']


    # combine 25 slitlets into a matrix
    med_flux = np.zeros( [ 25, size ] )
    for i in range( len( hdulist ) ):
        if "SCI" in hdulist[ i ].name:
            med_flux[ i-1, : ] = np.median( hdulist[ i ].data, axis = 1 ) 


    # constrain center using the median of all cube and fiting a Gaussian
    # find intial guess for minimizer function
    y0, x0 = np.unravel_index( med_flux.argmax(), med_flux.shape )
    x0 = x0/2 # x coordinate has 0.5 arcseconds units

    # obtain median values for, gaussian center and sigma
    # assumption here is sig_x = sig_y
    med = opt.minimize( log_lh, [ x0, y0, 10, 0.5 ], args = (xv,yv,med_flux,1 ))
    med_x, med_y, med_sigma, med_b = med.x[0], med.x[1], med.x[2], med.x[3]


    # do a gaussian fitting for every wl frame 
    total_flux = [ ]

    for wl_n in range( hdulist[ 1 ].data.shape[ 1 ] ):
        
        # initialize empty flux array
        flux = np.zeros( [ 25, size ] )
        var = np.zeros( [ 25, size ] )
        
        for i in range( len( hdulist ) ):
            if "SCI" in hdulist[ i ].name:
                flux[ i-1, : ] = hdulist[ i ].data[ :, wl_n ] / exp_time
                var[ i-1, : ] = hdulist[ 25 + i ].data[ :, wl_n ] / exp_time

        # Amplitude of Gaussian
        A = f_A( xv, yv, flux, np.sqrt(var), med_x, med_y, med_sigma, med_b )
        total_flux.append( A )

    return  np.array(total_flux), wl_range


def f_A( x, y, flux, sig_f, x0, y0, sig_g, b ):
    # Model of the gaussian using analytic amplitude

    # Normalised 2D Mofat function
    gg = m( x, y, x0, y0, sig_g, b )

    # Compute analytic amplitude
    A_up   = np.nansum( flux * gg / sig_f**2 )
    A_down = np.nansum( gg**2 / sig_f**2 )

    A = 0
    if ( A_down > 0.0 ): A = A_up / A_down

    return A

def g( x, y, x0, y0, sig_g ):
    # 2D normalized gaussian
    g = 1 / ( 2 * np.pi * sig_g**2 ) * np.exp( 
        -(  ( x-x0 )**2 + ( y-y0 )**2  ) / ( 2 * sig_g**2 ) )

    return g

def m( x, y, x0, y0, sig_g, b ):
    # 2D normalized Mofat
    m = (b-1)/(np.pi*sig_g**2)*(1 + (( x-x0 )**2 + ( y-y0 )**2) / sig_g**2 )**(-b)

    return m



def log_lh( param, x, y, flux, sig_f ):
    # log likelihood funciton

    x0, y0, sig_g, b = param
    l_lh = 0

    # mofat model
    model = f_A(x, y, flux, sig_f, x0, y0, sig_g, b) * m(x, y, x0, y0, sig_g, b)
    
    # negative log likelihood ( negative because we want to minimize )
    l_lh = + 0.5 * np.nansum( ( flux - model )**2 / sig_f**2 ) + np.log( sig_g**2 )

    return l_lh


def rebin_transmission(  xout, xin, yin, return_sum=False  ):
    # get bin edges for xin and xout
    xout_edges, xin_edges = pabin_edges(  xout  ), pabin_edges(  xin  )
    
    # cumulative sum to integrate up the transmission function
    y_integral = np.cumsum(  yin * np.diff(  xin_edges  )  )
    
    # interpolate to get the integral on the new wl grid
    y_interpol = np.interp(  xout_edges, xin, y_integral  )
    
    # use diff to de-integrate
    yout = np.diff(  y_interpol  ) / np.diff(  xout_edges  )
    return yout


def pabin_edges(  bin_centres  ):

    bin_width = np.diff(  bin_centres  )
    bin_edges = np.zeros(  len( bin_centres ) + 1  )
    bin_edges[ :2 ] = bin_centres[ :2 ] - bin_width[ :2 ] / 2.
    bin_edges[ 2: ] = bin_centres[ 1: ] + bin_width / 2.

    return bin_edges



def telluric( eso_wl, data_wl, data_calb, eso_flux, data_flux ):
    bands = [ [ 6844, 6972 ], 
              [ 7148, 7340 ], 
              [ 7570, 7692 ], 
              [ 8024, 8348 ] ]
              #[ 9292, 9555 ] ]

    data_calb2 = data_calb.copy()
    correction = data_calb.copy()

    if bands[ 0 ][ 0 ] < data_wl[ -1 ]:
        for band in bands:
            wl_inter = [ eso_wl[ eso_wl < band[ 0 ] ][ -1 ], 
                eso_wl[ eso_wl > band[ 1 ] ][ 0 ] ]
            flux_inter = [ eso_flux[ eso_wl < band[ 0 ] ][ -1 ], 
                eso_flux[ eso_wl > band[ 1 ] ][ 0 ] ]

            interp_tel = interp1d( wl_inter, flux_inter )
            eso_flux[ ( eso_wl > band[ 0 ] ) & ( eso_wl < band[ 1 ] ) ] = interp_tel( 
                eso_wl[ ( eso_wl > band[ 0 ] ) & ( eso_wl < band[ 1 ] ) ] )

            interp_eso = interp1d( 
                eso_wl[ ( eso_wl > band[ 0 ] ) & ( eso_wl < band[ 1 ] ) ], 
                eso_flux[ ( eso_wl > band[ 0 ] ) & ( eso_wl < band[ 1 ] ) ] )

            tell_region = [ 
            ( data_wl >= eso_wl[ ( eso_wl >= band[ 0 ] ) & ( eso_wl <= band[ 1 ] ) ][ 0 ] ) & 
            ( data_wl <= eso_wl[ ( eso_wl >= band[ 0 ] ) & ( eso_wl <= band[ 1 ] ) ][ -1 ] ) ]
            
            data_calb2[ tell_region ] = ( interp_eso( data_wl[ tell_region ]
                 ) / data_flux[ tell_region ] )

            corr_f = interp1d([data_wl[ tell_region ][0],data_wl[ tell_region ][-1]], [data_calb[ tell_region ][0],data_calb[ tell_region ][-1]])
            correction[ tell_region ] = corr_f(data_wl[ tell_region ])


    else:
        data_calb2 = data_calb
    return data_calb2
















