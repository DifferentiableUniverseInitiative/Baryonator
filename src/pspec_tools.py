import jax.numpy as jnp

def pspec(map1, map2, mask1, mask2, delta_ell, ell_max, pix_size, N):
    
    N = int(N)
    # make a 2d ell coordinate system
    ones  = jnp.ones(N)
    inds  = (jnp.arange(N)+.5 - N/2.) /(N-1.)
    kX    = jnp.outer(ones,inds) / (pix_size/60. * jnp.pi/180.)
    kY    = jnp.transpose(kX)
    K     = jnp.sqrt(kX**2. + kY**2.)
    ell2d = K * 2. * jnp.pi
    
    # get the 2d fourier transform of the map
    F1    = jnp.fft.ifft2(jnp.fft.fftshift(map1*mask1))
    F2    = jnp.fft.ifft2(jnp.fft.fftshift(map2*mask2))
    PSD   = jnp.fft.fftshift(jnp.real(jnp.conj(F1) * F2))

    # make an array to hold the power spectrum results
    N_bins = int(ell_max/delta_ell)
    ells   = jnp.zeros(N_bins)
    tmp    = jnp.arange(N_bins)
    cls    = jnp.zeros(N_bins)

    # fill out the spectra
    for i in range(0,N_bins):
        ells    = jnp.where(tmp == i,  (i + 0.5) * delta_ell, ells)
        idx     = jnp.where( (ell2d >= (i * delta_ell)) & (ell2d < ((i+1)* delta_ell)) ) #.nonzero())
        u       = jnp.mean(PSD[idx])
        cls     = jnp.where(tmp == i,  u, cls)

    nrm = jnp.mean(mask1*mask2)**0.5 / (N*N)**0.5 / (pix_size/60/180*jnp.pi)

    return ells, cls/nrm**2
