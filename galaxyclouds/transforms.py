import numpy as np
import matplotlib.pyplot as plt

class CoordinateFrame:
    """
    Manages coordinate transformations for galaxy point clouds.
    
    Just as particle physicists boost to the jet rest frame 
    to study intrinsic fragmentation properties independent 
    of detector orientation, we transform galaxy coordinates 
    to the galaxy's principal frame to study intrinsic 
    morphology independent of sky projection.
    """
    def __init__(self):
        pass

def compute_principal_axes(X, mask):
    """
    PCA on flux-weighted stellar positions.
    Returns eigenvalues (axis lengths) and eigenvectors 
    (axis directions).
    Axis ratio b/a measures ellipticity.
    """
    N = X.shape[0]
    evals = np.zeros((N, 2))
    evecs = np.zeros((N, 2, 2))
    
    flux = X[:, :, 0] * mask
    F_tot = np.sum(flux, axis=1, keepdims=True)
    F_tot = np.where(F_tot == 0, 1e-10, F_tot)
    
    ra = X[:, :, 1]
    dec = X[:, :, 2]
    
    ra_c = np.sum(flux * ra, axis=1, keepdims=True) / F_tot
    dec_c = np.sum(flux * dec, axis=1, keepdims=True) / F_tot
    
    delta_ra = ((ra - ra_c + 180) % 360) - 180
    delta_dec = dec - dec_c
    
    for i in range(N):
        if np.sum(mask[i]) < 2:
            evals[i] = [1.0, 1.0]
            evecs[i] = np.eye(2)
            continue
            
        x_m = delta_ra[i, mask[i]]
        y_m = delta_dec[i, mask[i]]
        w_m = flux[i, mask[i]]
        W = np.sum(w_m)
        
        cov_xx = np.sum(w_m * x_m**2) / W
        cov_yy = np.sum(w_m * y_m**2) / W
        cov_xy = np.sum(w_m * x_m * y_m) / W
        
        cov_mat = np.array([[cov_xx, cov_xy], [cov_xy, cov_yy]])
        w, v = np.linalg.eigh(cov_mat)
        
        # Sort descending
        idx = w.argsort()[::-1]
        evals[i] = w[idx]
        evecs[i] = v[:, idx]
        
    return evals, evecs

def transform_to_galaxy_frame(X, mask):
    """
    Transform stellar coordinates to galaxy principal frame:
    1. Translate: subtract flux-weighted centroid 
    2. Rotate: align major axis with x-axis
    3. Normalize: divide by half-light radius 
    
    Returns: transformed X array, transformation parameters
    """
    X_trans = X.copy()
    
    flux = X[:, :, 0] * mask
    F_tot = np.sum(flux, axis=1, keepdims=True)
    F_tot = np.where(F_tot == 0, 1e-10, F_tot)
    
    ra = X[:, :, 1]
    dec = X[:, :, 2]
    
    ra_c = np.sum(flux * ra, axis=1, keepdims=True) / F_tot
    dec_c = np.sum(flux * dec, axis=1, keepdims=True) / F_tot
    
    delta_ra = ((ra - ra_c + 180) % 360) - 180
    delta_dec = dec - dec_c
    
    delta_r = np.sqrt(delta_ra**2 + delta_dec**2)
    r_half = np.sum(flux * delta_r, axis=1) / F_tot.squeeze()
    r_half = np.where(r_half == 0, 1.0, r_half)
    
    evals, evecs = compute_principal_axes(X, mask)
    
    for i in range(X.shape[0]):
        if np.sum(mask[i]) < 2:
            X_trans[i, mask[i], 1] = 0
            X_trans[i, mask[i], 2] = 0
            continue
            
        pts = np.vstack([delta_ra[i, mask[i]], delta_dec[i, mask[i]]])
        
        # Rotate using eigenvectors
        # evecs[i] has principal axis as columns. V.T aligns them to identity.
        pts_rot = evecs[i].T @ pts
        
        # Normalize
        pts_norm = pts_rot / r_half[i]
        
        X_trans[i, mask[i], 1] = pts_norm[0, :]
        X_trans[i, mask[i], 2] = pts_norm[1, :]
        
    params = {
        'ra_c': ra_c,
        'dec_c': dec_c,
        'r_half': r_half,
        'evals': evals,
        'evecs': evecs
    }
        
    return X_trans, params


def verify_transform(X_original, X_transformed, mask):
    """
    Verify transformation correctness:
    1. Centroid of transformed coords ≈ (0,0) 
    2. Major axis aligned with x-axis (cross-term ≈ 0)
    
    Print residuals and PASS/FAIL for each check.
    """
    flux = X_transformed[:, :, 0] * mask
    F_tot = np.sum(flux, axis=1)
    F_tot = np.where(F_tot == 0, 1e-10, F_tot)
    
    x = X_transformed[:, :, 1]
    y = X_transformed[:, :, 2]
    
    x_c = np.sum(flux * x, axis=1) / F_tot
    y_c = np.sum(flux * y, axis=1) / F_tot
    
    centroid_res = np.mean(np.sqrt(x_c**2 + y_c**2))
    
    # cross-terms
    cross_term_res = 0.0
    valid_count = 0
    for i in range(X_transformed.shape[0]):
        if np.sum(mask[i]) < 2:
            continue
        w_i = flux[i, mask[i]]
        x_i = x[i, mask[i]]
        y_i = y[i, mask[i]]
        cov_xy = np.sum(w_i * x_i * y_i) / np.sum(w_i)
        cross_term_res += np.abs(cov_xy)
        valid_count += 1
        
    cross_term_res /= max(1, valid_count)
    
    print(f"Centroid Res: {centroid_res:.2e} -> {'PASS' if centroid_res < 1e-10 else 'FAIL'}")
    print(f"Cross-term Res: {cross_term_res:.2e} -> {'PASS' if cross_term_res < 1e-10 else 'FAIL'}")


def animate_transform(X_single_galaxy, mask_single, n_steps=60, filename='transform.gif'):
    """
    Create matplotlib animation showing one galaxy's 
    stellar point cloud continuously transforming 
    from sky frame to galaxy principal frame.
    """
    print(f"Saving animation to {filename} (Placeholder implementation, use true matplotlib.animation in notebooks)")
    # For actual implementation this requires matplotlib.animation hook
