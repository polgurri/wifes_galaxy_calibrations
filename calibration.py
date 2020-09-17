
#WIFES VEL FIELD CALIBRATION P.GURRI '2019'
#------------------------------------
#Complete calibration of velocity fields of WIFES data ANU 2.3m

# Calibration includes flux-calibration and skysubtraction (PCA based)

# Simply need to point to a pywifes calibrated cube with corresponding standard star

# Final velocity map will be created for the galaxy

# Please use this as a reference only, as many parameters have been tuned to suit this particular research case



# ---------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------

import os
from time import time
import pickle
import numpy as np 
from astropy.io import fits
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.ndimage import median_filter
from sklearn.decomposition import PCA


import flux_calib
import vel


# ---------------------------------------------------------------------
# -- Master Function --------------------------------------------------
# ---------------------------------------------------------------------



def run_all( path ):

    # -- get all files in position --------------------------------------------

    # -- RED CUBE --

    red_path = path + '/reduc_r/'

    red_cubePaths = [ red_path + i for i in os.listdir( red_path ) 
        if ( '.p11.fits' in i ) and ( 'T2m3wr' in i ) ]
    
    red_std_path = [ red_path + i for i in os.listdir( red_path ) 
        if ( '.p08.fits' in i ) and ('T2m3wr' in i) ][0]
    print(red_std_path)
    
    # -- BLUE CUBE --

    blue_path = path + '/reduc_b/'

    blue_cubePaths = [ blue_path + i for i in os.listdir( blue_path ) 
       if ( '.p11.fits' in i ) and ( 'T2m3wb' in i ) ]
    
    blue_std_path = [ blue_path + i for i in os.listdir( blue_path ) 
       if ( '.p08.fits' in i ) and ('T2m3wb' in i) ][0]

    # -- Run calibrations -----------------------------------------------------

    for iii in red_cubePaths:
        
        red_cube = calibrate( iii, red_std_path)
        
        # You can add fixes with with .replace() if strings need to be changed in anyway 
        obj = red_cube.header['OBJECT']

        plot_me( red_cube )
        plt.savefig(path + obj + '_r_vel.png')

        # create new fits
        hdul = savefits( red_cube )
        hdul_vel = savefits_vel ( red_cube )

        if os.path.isfile(path + '/' + obj + '_r.fits'):

            hdul.writeto(path + '/' + obj + '_r_2.fits')
            hdul_vel.writeto(path + '/' + obj + '_r_2_vel.fits')
        
        else:
            hdul.writeto(path + '/' + obj + '_r.fits')
            hdul_vel.writeto(path + '/' + obj + '_r_vel.fits')

    
    for iii in blue_cubePaths:

        blue_cube = calibrate( iii, blue_std_path)
        
        # You can add fixes with with .replace() if strings need to be changed in anyway 
        obj = blue_cube.header['OBJECT']

        plot_me( blue_cube )
        plt.savefig(path + obj + '_b_vel.png')

        # create new fits
        hdul = savefits( blue_cube )
        hdul_vel = savefits_vel ( blue_cube )

        if os.path.isfile(path + '/' + obj + '_b.fits'):

            hdul.writeto(path + '/' + obj + '_b_2.fits')
            hdul_vel.writeto(path + '/' + obj + '_b_2_vel.fits')
        
        else:
            hdul.writeto(path + '/' + obj + '_b.fits')
            hdul_vel.writeto(path + '/' + obj + '_b_vel.fits')


# ---------------------------------------------------------------------
# -- Calibration Function ---------------------------------------------
# ---------------------------------------------------------------------

def calibrate( cubepath, stds_path = None):
    
    # start a measurement of time
    t0 = time()

    # create a cube object -- this includes all the data and calibrations
    cube = Cube( cubepath )

    print( 'calibrating - ' + cube.header['OBJECT'])

    # remove all NaN values by interpolating neighboring values
    _nanclean( cube )


    # --- Flux Calibration -----------------------------------------------------
     
    flux_cal = flux_calib.stds_corr( stds_path )
    # in [10**(-16)ergs * cm-2 * s-1 * A-1 ]

    # apply flux_calibration
    cube.flux_cal = flux_cal
    for ii in np.arange( cube.stack.shape[1] ):
        cube.stack[ :, ii ] *= flux_cal[:len(cube.stack[ :, ii ])]
        cube.varstack[ :, ii ] *= flux_cal[:len(cube.stack[ :, ii ])]**2


    # --- Sky Substraction -----------------------------------------------------

    # perform a sky substraction based on a PCA method
    continuum = median_filter( cube.stack, [300,1] ) #bottleneck
    X = cube.stack - continuum

    n_components = 10

    pca = PCA( n_components = n_components)
    pca.fit( X.T )
    X_pca = pca.transform( X.T )
    A = pca.inverse_transform( X_pca )
    cube.stack = X - A.T + continuum
    cube.sky = A.T + continuum
    cube.continuum = continuum


    # --- Velocity Maps -----------------------------------------------

    obj = cube.header['OBJECT']


    with open(cubepath + 'wifes/z0s.txt', 'r') as f:
        red_shifts = f.readlines()

    print(obj)
    z0 = [float(iii.split(',')[1].replace('\n','')) for iii in red_shifts if obj[:10] in iii][0]
    #print(red_shifts)
    #z0 = [print(iii) for iii in red_shifts if obj[:10] in iii][0]

    
    cube.vel, cube.vel_unc, cube.sn, cube.width_final = vel.velocity_map( cube, z0 )


    print( "Calibd! ( %.2f sec. )" % ( time() - t0 ) )



    return cube




def savefits( cube ):

    cube_calib_data = fits.PrimaryHDU(np.reshape(cube.stack,cube.rawdata.shape),header=cube.header)
    cube_calib_var = fits.ImageHDU(np.reshape(cube.varstack,cube.rawvar.shape))
    cube_rawdata = fits.ImageHDU(cube.rawdata)
    cube_rawvar = fits.ImageHDU(cube.rawvar)
    cube_flux_cal = fits.ImageHDU(cube.flux_cal)
    cube_sky = fits.ImageHDU(np.reshape(cube.sky,cube.rawdata.shape))
    cube_continuum = fits.ImageHDU(np.reshape(cube.continuum,cube.rawdata.shape))
    cube_vel = fits.ImageHDU(cube.vel)
    cube_vel_unc = fits.ImageHDU(cube.vel_unc)
    cube_sn = fits.ImageHDU(cube.sn)
    cube_width_final = fits.ImageHDU(cube.width_final)

    hdul = fits.HDUList([cube_calib_data, cube_calib_var,cube_rawdata,cube_rawvar,cube_flux_cal,cube_sky,cube_continuum,cube_vel,cube_vel_unc,cube_sn,cube_width_final])

    return hdul

def savefits_vel( cube ):

    cube_vel = fits.PrimaryHDU(cube.vel,header=cube.header)
    cube_vel_unc = fits.ImageHDU(cube.vel_unc)
    cube_sn = fits.ImageHDU(cube.sn)
    cube_width_final = fits.ImageHDU(cube.width_final)

    hdul = fits.HDUList([cube_vel,cube_vel_unc,cube_sn,cube_width_final])

    return hdul



def plot_me( cube ):
    
    fig = plt.figure(11)

    plt.clf()
    plt.cla()

    obj = cube.header['OBJECT'].replace(' ','').replace('_','').replace('GGL','GGL_')


    if len(cube.sn) > 0 :
        ax1 = plt.subplot(221)
        
        plt.title(obj + ' -- sn [lines]')
        plt.imshow(cube.sn.T, origin="lower", interpolation="none",
            aspect = 'equal', extent = [0,35,0,25])
        plt.colorbar()
        plt.xlim([0,35])
        plt.contour(cube.sn.T, colors=['y','r'], linewidths=[3], levels = [1.5,3], extent = [0,35,0,25])
        
        if len(cube.vel[cube.sn > 3]) > 0 :
            systemic_vel = np.nanmedian(cube.vel[cube.sn > 3])

            ax2 = plt.subplot(223, sharex = ax1, sharey =ax1)

            plt.title('velocity [km/s]')

            plt.imshow((cube.vel - systemic_vel).T, aspect = 'equal', extent = [0,35,0,25], cmap = 'coolwarm', origin="lower", vmin = -200, vmax = 200);
            plt.colorbar()
            plt.xlim([0,35])

        

        # ---------
            if len(cube.vel_unc[cube.sn > 3]) > 0:
                systemic_vel_unc = np.nanmean(cube.vel_unc[cube.sn > 3])

                ax3 = plt.subplot(222, sharex = ax1, sharey =ax1)

                plt.title('vel_unc [km/s]')

                plt.imshow(cube.vel_unc.T, origin="lower", interpolation="none", aspect = 'equal', extent = [0,35,0,25], vmin = 0, vmax = 60 )#,vmax = 5*systemic_vel_unc)
                plt.xlim([0,35])
                plt.colorbar()

                systemic_width = np.nanmean(cube.width_final)

                ax4 = plt.subplot(224, sharex = ax1, sharey =ax1)

                plt.title('line_width [A]')

                plt.imshow( cube.width_final.T,
                    aspect = 'equal', extent = [0,35,0,25], cmap = 'coolwarm', origin="lower", vmin = 0, vmax = 3 )
                plt.xlim([0,35])
                plt.colorbar()


                fig.tight_layout()

                plt.subplots_adjust(left=None, bottom=None, right=None, top=0.95, wspace=0.05, hspace=None)
        else:
            plt.plot([1,2,3],[1,2,3])
            plt.title('Anything above sn of 3')
    #plt.savefig(obj + '_r_vel.png')

# ---------------------------------------------------------------------
# -- Main Class -------------------------------------------------------
# ---------------------------------------------------------------------

class Cube( object ):
    """ 
    Attributes

    -- Intitial products:
    rawdata : 3D array -- Full raw cube flux f( x, y, wl )
    rawvar : 3D array -- Full raw clube variance v( x, y, wl )
    header : str -- Original fits header
    wl : 1D array --  Wavelenght coming from the raw cube 
    stack : 2D array -- Full raw cube stacked on fiber num. f( fib_n, wl )

    -- Intermediate products:
    nancube : 3D array -- Cube with value = 1 on NaN values that have been 
              erased f( x, y, wl )
    median_spec : 1D array -- Median of all rawdata at every wl f( wl )
    prep_stack : 2D array -- stack after median spec and continuum removal
                 this is feed into the PCA routine f( fib_n, wl )
    
    -- Final products:
    data : 3D array -- Full corrected cube flux f( x, y, wl )
    var : 3D array -- Full corrected cube variance v( x, y, wl )
    corr_data : 3D array -- Corrected cube flux before skysub f( x, y, wl )
    corr_var : 3D array -- Full corrected cube variance v( x, y, wl )
    twi_corr : 2D array -- Twilight correction c( x, y )
    stds_corr : 1D array -- Standard Star Flux correction c( wl )
    continuum: 2D array -- Stack of continum values f( fib_n, wl )
    sky: 3D array -- Cube of the sky model f( x, y, wl )
    pcastack : 2D array -- stack after PCA
    """

    def __init__( self, cubepath ):
        
        self.cubepath = cubepath
        with fits.open( cubepath ) as hdul:
            self.rawdata = hdul[ 0 ].data
            self.rawvar = hdul[ 1 ].data
            self.header = hdul[ 0 ].header
            self.wl =  self.header[ 'CRVAL3' ] + (  
                np.arange( self.header[ 'NAXIS3' ], dtype='f' ) - 0. 
                                      ) * self.header[ 'CDELT3' ]

        self.stack = np.reshape( np.copy(self.rawdata), 
            [ self.rawdata.shape[0], 
            self.rawdata.shape[1] * self.rawdata.shape[2] ] )
        
        self.varstack = np.reshape( np.copy(self.rawvar), 
            [ self.rawvar.shape[0], 
            self.rawvar.shape[1] * self.rawvar.shape[2] ] )




# ---------------------------------------------------------------------
# -- Save fits --------------------------------------------------------
# ---------------------------------------------------------------------

def save_fits( self, outcubefits, overrite ):
        outcubefits = outcubefits.split( '.fits' )[ 0 ] + '.fits'

        if ( os.path.isfile( outcubefits ) and overrite ) or ( 
            not os.path.isfile( outcubefits ) ):

            if os.path.isfile( outcubefits ): os.remove( outcubefits )


            with fits.open( self.cubepath ) as hdu:
                hdu[ 0 ].header = self.header
                hdu[ 0 ].data = self.data
                hdu[ 1 ].data = self.var

                
                hdu.writeto( outcubefits )


# ---------------------------------------------------------------------
# -- Clean Cube of NaNs -----------------------------------------------
# ---------------------------------------------------------------------

def _nanclean( self, rejectratio=0.25, boxsz=1 ):
    """
    Detects NaN values in cube and removes them by replacing them with an
    interpolation of the nearest neighbors in the data cube. The positions in
    the cube are retained in nancube for later remasking.
    """
    cleancube = np.copy(self.rawdata)
    badcube = np.logical_not( np.isfinite( cleancube ) )        # find NaNs
    badmap = badcube.sum( axis=0 )  # map of total nans in a spaxel

    # choose some maximum number of bad pixels in the spaxel and extract
    # positions
    badmask = badmap > ( rejectratio * cleancube.shape[ 0 ] )

    # make cube mask of bad spaxels
    badcube &= ( ~badmask[ np.newaxis, :, : ] )
    z, y, x = np.where( badcube )

    neighbor = np.zeros( ( z.size, ( 2 * boxsz + 1 )**3 ) )
    icounter = 0

    # loop over samplecubes
    nz, ny, nx = cleancube.shape
    for j in range( -boxsz, boxsz + 1, 1 ):
        for k in range( -boxsz, boxsz + 1, 1 ):
            for l in range( -boxsz, boxsz + 1, 1 ):
                iz, iy, ix = z + l, y + k, x + j
                outsider = ( ( ix <= 0 ) | ( ix >= nx - 1 ) |
                            ( iy <= 0 ) | ( iy >= ny - 1 ) |
                            ( iz <= 0 ) | ( iz >= nz - 1 ) )
                ins = ~outsider
                neighbor[ ins, icounter ] = cleancube[ iz[ ins ], iy[ ins ], ix[ ins ] ]
                neighbor[ outsider, icounter ] = np.nan
                icounter = icounter + 1

    cleancube[ z, y, x ] = np.nanmean( neighbor, axis=1 )
    self.rawdata, self.nancube =  cleancube, badcube



# ---------------------------------------------------------------------
# -- Save cube as pickle ----------------------------------------------
# ---------------------------------------------------------------------


def save_pickle( obj, filename ):
    with open( filename, 'wb' ) as output:  # Overwrites any existing file.
        pickle.dump( obj, output, pickle.HIGHEST_PROTOCOL )



# ---------------------------------------------------------------------
# -- Visualization ----------------------------------------------------
# ---------------------------------------------------------------------

def visualization(  b , r  ):

    x = np.linspace( 0.25, 37.75, r.data.shape[ 1 ] )
    y = np.linspace( 0.5, 24.5, 25 )
    xv , yv = np.meshgrid( x, y )


    # -- find snr ----------------------------------------------------

    red_snr = np.nansum( r.data/np.sqrt( r.var ), axis = 0 ).T
    blue_snr = np.nansum( b.data/np.sqrt( b.var ), axis = 0 ).T
    

    # -- find top brightest points snr --------------------------------

    # -- red --
    num_bright = 15

    red_max = np.dstack( 
        np.unravel_index( 
            np.argsort( red_snr[ 2:-2, 2:-2 ].ravel() ), 
            ( red_snr[ 2:-2, 2:-2 ].shape ) ) )[ -1, -num_bright: ] + [ 2, 2 ]

    red_max_spec = np.zeros( [ num_bright, r.wl.shape[ 0 ] ] )
    for i in range( num_bright ):
        red_max_spec[ i, : ] = r.data[ :, red_max[ i ][ 1 ], red_max[ i ][ 0 ] ]

    red_med_max = np.nanmedian( red_max_spec, axis = 0 )*num_bright

    # -- blue --

    blue_max = np.dstack( 
        np.unravel_index( 
            np.argsort( blue_snr[ 2:-2, 2:-2 ].ravel() ), 
            ( blue_snr[ 2:-2, 2:-2 ].shape ) ) )[ -1, -num_bright: ] + [ 2, 2 ]


    blue_max_spec = np.zeros( [ num_bright, b.wl.shape[ 0 ] ] )
    for i in range( num_bright ):
        blue_max_spec[ i, : ] = b.data[ :, blue_max[ i ][ 1 ], blue_max[ i ][ 0 ] ]
        #ax_sp_b.plot( b.wl, blue_max_spec[ i, : ], c = 'gray', alpha = 0.2 )

    blue_med_max = np.nanmedian( blue_max_spec, axis = 0 )*num_bright

    
    # -- find 50% brightest points snr --------------------------------
    
    # -- red --
    num_bright = 15

    rdstack =  np.dstack( 
        np.unravel_index( 
            np.argsort( red_snr.ravel() ), ( red_snr.shape ) ) )

    red_min = rdstack[ -1, int(rdstack.shape[1] / 2) : int(rdstack.shape[1] / 2) + num_bright ]


    red_min_spec = np.zeros( [ num_bright, r.wl.shape[ 0 ] ] )

    for i in range( num_bright ):
        red_min_spec[ i, : ] = r.data[ :, red_min[ i ][ 1 ], red_min[ i ][ 0 ] ]
        #ax_sp_r.plot( wl_r, red_max_spec[ i, : ], c = 'gray', alpha = 0.2 )

    red_med_min = np.nanmedian( red_min_spec, axis = 0 )*num_bright

    # -- blue --

    bdstack =  np.dstack( 
        np.unravel_index( 
            np.argsort( blue_snr.ravel() ), ( blue_snr.shape ) ) )

    blue_min = bdstack[ -1, int(bdstack.shape[1] / 2) : int(bdstack.shape[1] / 2) + num_bright ]



    blue_min_spec = np.zeros( [ num_bright, b.wl.shape[ 0 ] ] )
    for i in range( num_bright ):
        blue_min_spec[ i, : ] = b.data[ :, blue_min[ i ][ 1 ], blue_min[ i ][ 0 ] ]
        #ax_sp_b.plot( b.wl, blue_max_spec[ i, : ], c = 'gray', alpha = 0.2 )

    blue_med_min = np.nanmedian( blue_min_spec, axis = 0 )*num_bright


    # -- plot'em --------------------------------------------------------------

    plt.figure( 1 ); plt.clf()

    f, ( ( ax_b, ax_r ), ( ax_sp_b, ax_sp_r ) ) = plt.subplots( 2, 2, 
        gridspec_kw = {'height_ratios':[ 2, 1 ]}, num =1 )

    ax_sp_r.plot( r.wl, red_med_min, 'gray' )
    ax_sp_r.plot( r.wl, red_med_max, 'k' )
    ax_sp_r.plot( r.wl, median_filter(  red_med_max, ( 10 )  ), 'r'  )
    ax_sp_r.set_ylim( [ -2000, 2*max( median_filter(  red_med_max, ( 10 )  ) ) ] ); 
    ax_sp_r.set_xlim( [ r.wl[ 0 ], r.wl[ -1 ] ] )
    
    
    ax_sp_b.plot( b.wl, blue_med_min, 'gray' )
    ax_sp_b.plot( b.wl, blue_med_max, 'k' )
    ax_sp_b.plot( b.wl, median_filter(  blue_med_max, ( 10 )  ), 'r'  )
    ax_sp_b.set_ylim( [ -2000, 2*max( median_filter(  blue_med_max, ( 10 )  ) ) ] ); 
    ax_sp_b.set_xlim( [ b.wl[ 0 ], b.wl[ -1 ] ] )

    
    v_val= 0.2#9
    h_val= 0.1#2.8
    verts = list( zip( [ -h_val, h_val, h_val, -h_val, -h_val ], [ -v_val, -v_val, v_val, v_val, -v_val ] ) )

    ax_b.scatter( xv, yv, c = np.sinh( blue_snr/3323 ), edgecolor='black', 
        marker=( verts, 0 ), s = 190, vmin = 0, vmax = np.percentile( np.sinh( blue_snr/3233 ), 98 ) );

    ax_b.scatter( [ i[ 1 ]/2 + 0.25 for i in blue_max ], [ i[ 0 ] + 0.5 for i in blue_max ], 
        c = 'none', edgecolor='red', 
        marker=( verts, 0 ), s = 190, vmin = 0, vmax = 1 );

    ax_b.scatter( [ i[ 1 ]/2 + 0.25 for i in blue_min ], [ i[ 0 ] + 0.5 for i in blue_min ], 
        c = 'none', edgecolor='white', 
        marker=( verts, 0 ), s = 190, vmin = 0, vmax = 1 );


    ax_r.scatter( xv, yv, c = np.sinh( red_snr/3323 ), edgecolor='black', 
        marker=( verts, 0 ), s = 190, vmin = 0, vmax = np.percentile( np.sinh( red_snr/3233 ), 98 ) );

    ax_r.scatter( [ i[ 1 ]/2 + 0.25 for i in red_max ], [ i[ 0 ] + 0.5 for i in red_max ], 
        c = 'none', edgecolor='red', 
        marker=( verts, 0 ), s = 190, vmin = 0, vmax = 1 );

    ax_r.scatter( [ i[ 1 ]/2 + 0.25 for i in red_min ], [ i[ 0 ] + 0.5 for i in red_min ], 
        c = 'none', edgecolor='white', 
        marker=( verts, 0 ), s = 190, vmin = 0, vmax = 1 );
    

    ax_r.set_xlim( [ 0, 38 ] ); ax_b.set_xlim( [ 0, 38 ] )
    ax_r.set_ylim( [ 0, 25 ] ); ax_b.set_ylim( [ 0, 25 ] ); 
    ax_r.set_facecolor( 'black' ); ax_b.set_facecolor( 'black' )
    
    ax_r.yaxis.tick_right() #ax_r.get_yaxis().set_visible( False )
    ax_sp_r.yaxis.tick_right()

    ax_r.tick_params( labelsize=16 )
    ax_b.tick_params( labelsize=16 )
    ax_sp_r.tick_params( labelsize=16 )
    ax_sp_b.tick_params( labelsize=16 )

    ax_r.set_title( 'R_m-snr', fontsize=20 )
    ax_b.set_title( 'B_m-snr', fontsize=20 )


    plt.subplots_adjust( wspace=0.01 )
    
    #plt.show()
    plt.savefig( r.cubepath.split('T2m3')[0] + '/' + r.cubepath.split( '/' )[ -2 ] + '.png' )

