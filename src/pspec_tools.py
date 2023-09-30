import jax.numpy as jnp

def pspec(map1, map2, mask1, mask2, delta_ell, ell_max, pix_size, N):
    "calcualtes the power spectrum of a 2d map by FFTing, squaring, and azimuthally averaging"

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
    #for i in range(0,N_bins):
    #    ells    = jnp.where(tmp == i,  (i + 0.5) * delta_ell, ells)
    #    idx     = jnp.where( (ell2d >= (i * delta_ell)) & (ell2d < ((i+1)* delta_ell)) ) #.nonzero())
    #    u       = jnp.mean(PSD[idx])
    #    cls     = jnp.where(tmp == i,  u, cls)

    i_vals = jnp.arange(N_bins)[:, None]  # shape (N_bins, 1)
    masks = (tmp == i_vals).T  # shape (len(tmp), N_bins)
    updated_values = (i_vals + 0.5) * delta_ell
    ells = jnp.where(masks, updated_values, ells[:, None]).sum(axis=-1)
    #print(ells)
    #sys.exit()

    i_vals = jnp.arange(N_bins)[:, None, None]
    lower_bounds = i_vals * delta_ell
    upper_bounds = (i_vals + 1) * delta_ell

    masks = (ell2d >= lower_bounds) & (ell2d < upper_bounds)  # Broadcasting should work now

    u_values = jnp.sum(PSD * masks, axis=(-1, -2)) / jnp.sum(masks, axis=(-1, -2))  # compute mean using masks

    cls = jnp.where(jnp.arange(N_bins)[:, None] == i_vals[:, 0, 0], u_values[:, None], cls).sum(axis=0)

    '''
    def update_ells_and_cls(i, inputs):
        tmp, ells, ell2d, PSD, cls, delta_ell = inputs
        mask = (tmp == i)
        print(mask.shape,ells)
        updated_cls = lax.select(mask, jnp.ones_like(cls) * u, cls)

        idx = jnp.logical_and(ell2d >= (i * delta_ell), ell2d < ((i+1)* delta_ell))
        u   = jnp.mean(jnp.extract(idx, PSD))

        updated_cls = lax.select(mask, u, cls)

        return (tmp, updated_ells, ell2d, PSD, updated_cls, delta_ell)

    def main_loop(N_bins, tmp, ells, ell2d, PSD, cls, delta_ell):
        _, ells_final, _, _, cls_final, _ = lax.fori_loop(0, N_bins, update_ells_and_cls, (tmp, ells, ell2d, PSD, cls, delta_ell))
        return ells_final, cls_final

    # Call main_loop with required arguments to get ells and cls
    ells_final, cls_final = main_loop(N_bins, tmp, ells, ell2d, PSD, cls, delta_ell)
    '''

    nrm = jnp.mean(mask1*mask2)**0.5 / (N*N)**0.5 / (pix_size/60/180*jnp.pi)

    return ells, cls/nrm**2
