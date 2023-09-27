# Tools to compute power spectra
import jax.numpy as jnp

def calculate_2d_spectrum(Map1, Map2, delta_ell, ell_max, pix_size, N):
    "calcualtes the power spectrum of a 2d map by FFTing, squaring, and azimuthally averaging"
    # Map1
    # Map2
    # delta_ell
    # ell_max
    # pix_size  [each pixel in arcminutes]
    # N
    
    N = int(N)
    # make a 2d ell coordinate system
    ones  = jnp.ones(N)
    inds  = (jnp.arange(N)+.5 - N/2.) /(N-1.)
    kX    = jnp.outer(ones,inds) / (pix_size/60. * jnp.pi/180.)
    kY    = jnp.transpose(kX)
    K     = jnp.sqrt(kX**2. + kY**2.)
    ell2d = K * 2. * jnp.pi 
    
    # get the 2d fourier transform of the map
    F1    = jnp.fft.ifft2(jnp.fft.fftshift(Map1))
    F2    = jnp.fft.ifft2(jnp.fft.fftshift(Map2))
    PSD   = jnp.fft.fftshift(jnp.real(jnp.conj(F1) * F2))

    # make an array to hold the power spectrum results
    N_bins = int(ell_max/delta_ell)
    ells   = jnp.arange(N_bins)
    cls    = jnp.zeros(N_bins)

    # fill out the spectra
    '''
    i = 0
    while (i < N_bins):
        ell_array[i] = (i + 0.5) * delta_ell
        inds_in_bin = ((ell2d >= (i* delta_ell)) * (ell2d < ((i+1)* delta_ell))).nonzero()
        CL_array[i] = np.mean(PSMap[inds_in_bin])
        #print i, ell_array[i], inds_in_bin, CL_array[i]
        i = i + 1
    '''
    for i in nrage(0,Nbins):
        ells[i] = (i + 0.5) * delta_ell
        idx     = ((ell2d >= (i * delta_ell)) * (ell2d < ((i+1)* delta_ell))).nonzero()
        cls[i]  = jnp.mean(PSMap[idx])
        
    # return the power spectrum and ell bins
    return ells, cls*jnp.sqrt(pix_size /60.* jnp.pi/180.)*2.


## make a power spectrum
binned_ell, binned_spectrum = calculate_2d_spectrum(appodized_map,appodized_map,delta_ell,ell_max,pix_size,N)

#print binned_ell
plt.semilogy(binned_ell,binned_spectrum* binned_ell * (binned_ell+1.)/2. / np.pi)
plt.semilogy(ell,DlTT)
plt.ylabel('$D_{\ell}$ [$\mu$K$^2$]')
plt.xlabel('$\ell$')
plt.show()
