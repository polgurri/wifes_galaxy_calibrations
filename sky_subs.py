
import numpy as np 
from astropy.io import fits

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


from scipy.interpolate import interp1d
from scipy.interpolate import UnivariateSpline
from scipy.ndimage import median_filter
from scipy.ndimage import uniform_filter
import scipy.optimize as opt


# == PCA ==============================================================

def pca_corr( self, n_components ):

    smoothing = False
    improved_continuum_subtraction( self, smoothing )

    X = self.prep_stack

    # perform PCA keeping only top components with SKLEARN
    pca = PCA( n_components = n_components )
    pca.fit( X.T )
    X_pca = pca.transform( X.T )
    A = pca.inverse_transform( X_pca )
    X_skysub = X - A.T


    self.sky = np.reshape( A.T, 
        [ self.corr_data.shape[ 0 ], self.corr_data.shape[ 1 ], self.corr_data.shape[ 2 ] ] )
    self.data = np.reshape( X_skysub.T + self.continuum, 
        [ self.corr_data.shape[ 0 ], self.corr_data.shape[ 1 ], self.corr_data.shape[ 2 ] ] )
    self.var = self.corr_var
    self.pcastack = X_skysub + self.continuum.T
    self.sky_stack = A + self.continuum.T




def improved_continuum_subtraction( self, smoothing ):

    X = self.stack

    self.median_spec = np.median( X, axis =1 ) 

    #median_spec shape
    median_spec_filtered = median_filter( self.median_spec, 300 )

    """
    Removes a 'zero' level from each spectral plane.
    This zero level is currently calculated with a median.
    Experimental operations -
        - exclude top quartile
        - run in an iterative sigma clipped mode
    """

    # -- initialize variables for later use
    continuum = np.zeros( X.shape )
    prep_stack = np.zeros( X.shape )

    # -- continuum substraction
    for ii in range( X.shape[ 1 ] ):
        
        # set every spectrum to 'zero' ( this is spatial zero )
        prep_stack[ :, ii ] = X[ :, ii ] - self.median_spec
        
        # big filter to get large scale structure ( continuum )
        continuum[ :, ii ] = median_filter( prep_stack[ :, ii ], 300 )

        # uniform filter to ease pca afterwards
        # small, less than line spread function, uniform filter
        if smoothing:
            prep_stack[ :, ii ] = uniform_filter( 
                prep_stack[ :, ii ] - continuum[ :, ii ], 3 )
        if not smoothing:
            prep_stack[ :, ii ] = prep_stack[ :, ii ] - continuum[ :, ii ]

        # true continum has the median_spec shape added to it
        continuum[ :, ii ]  += median_spec_filtered



    self.continuum = continuum
    self.prep_stack = prep_stack













