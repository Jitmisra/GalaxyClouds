import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp

def attach_metadata(**kwargs):
    def decorator(func):
        func.meta = kwargs
        return func
    return decorator


@attach_metadata(
    name='Stellar Multiplicity',
    symbol='N_*',
    units='dimensionless',
    category='structural',
    expected_discriminating_power='high',
    reference='Conselice 2003, ApJS 147'
)
def stellar_multiplicity(X, mask):
    """
    Count of detected stellar sources per galaxy.
    
    Formula: N = sum(mask, axis=1)
    
    Physical interpretation: Higher multiplicity indicates 
    more resolved stellar populations. Spiral and irregular 
    galaxies typically show higher multiplicity than 
    ellipticals due to ongoing star formation regions.
    
    Expected values:
    - Elliptical: 20-40
    - Spiral: 40-80  
    - Irregular: 60-100
    
    Analogous to: constituent multiplicity in jet physics,
    where gluon jets have higher multiplicity than quark jets
    due to enhanced soft radiation.
    """
    return np.sum(mask, axis=1).astype(float)


@attach_metadata(
    name='Total Flux',
    symbol='F_tot',
    units='counts',
    category='photometric',
    expected_discriminating_power='medium',
    reference='Fundamental Observable'
)
def total_flux(X, mask):
    """
    Sum of flux from all stellar sources.
    
    Formula: F_total = sum(flux_i * mask_i)
    
    Analogous to: total pT of a jet.
    """
    flux = X[:, :, 0]
    return np.sum(flux * mask, axis=1)


@attach_metadata(
    name='Half Light Radius',
    symbol='r_half',
    units='degrees',
    category='structural',
    expected_discriminating_power='high',
    reference='Conselice 2003, ApJS 147'
)
def half_light_radius(X, mask):
    """
    Flux-weighted angular RMS spread of stellar sources.
    
    Formula: 
    r_half = sum(flux_i * DeltaR_i) / sum(flux_i)
    
    where DeltaR_i = sqrt((ra_i - ra_center)^2 + (dec_i - dec_center)^2)
    and ra_center, dec_center are the flux-weighted centroid.
    
    Physical interpretation: Measures how concentrated or 
    extended a galaxy's stellar distribution is. Compact 
    ellipticals have small r_half; extended spirals larger.
    
    Note on angular wrapping: RA coordinates wrap at 360 
    degrees. Always compute Delta_RA = ((ra_i - ra_c + 180) % 360) - 180 
    to handle this periodicity correctly.
    
    Analogous to: jet width in particle physics, where 
    w = sum(pT_i * DeltaR_i) / sum(pT_i)
    """
    flux = X[:, :, 0] * mask
    F_tot = np.sum(flux, axis=1, keepdims=True)
    
    # Avoid division by zero
    F_tot = np.where(F_tot == 0, 1e-10, F_tot)
    
    ra = X[:, :, 1]
    dec = X[:, :, 2]
    
    # Calculate centroids
    ra_c = np.sum(flux * ra, axis=1, keepdims=True) / F_tot
    dec_c = np.sum(flux * dec, axis=1, keepdims=True) / F_tot
    
    # Handle RA wrapping
    delta_ra = ((ra - ra_c + 180) % 360) - 180
    delta_dec = dec - dec_c
    
    delta_r = np.sqrt(delta_ra**2 + delta_dec**2)
    
    r_half = np.sum(flux * delta_r, axis=1) / F_tot.squeeze()
    return r_half


@attach_metadata(
    name='Flux Dispersion',
    symbol='G',
    units='dimensionless',
    category='photometric',
    expected_discriminating_power='high',
    reference='Conselice 2003, ApJS 147'
)
def flux_dispersion(X, mask):
    """
    Normalized RMS flux, measuring how evenly distributed 
    flux is among stellar sources.
    
    Formula:
    G = sqrt(sum(flux_i^2)) / sum(flux_i)
    
    Physical interpretation: 
    G → 1/sqrt(N) when all stars have equal flux (uniform)
    G → 1 when one star dominates (concentrated)
    
    Elliptical galaxies: high G (central concentration)
    Spiral/Irregular: lower G (distributed star formation)
    
    Analogous to: pT dispersion in jet physics,
    pT^D = sqrt(sum(pT_i^2)) / sum(pT_i)
    """
    flux = X[:, :, 0] * mask
    F_tot = np.sum(flux, axis=1)
    F_tot = np.where(F_tot == 0, 1e-10, F_tot)
    
    rms_flux = np.sqrt(np.sum(flux**2, axis=1))
    return rms_flux / F_tot


@attach_metadata(
    name='Asymmetry Index',
    symbol='A',
    units='dimensionless',
    category='structural',
    expected_discriminating_power='high',
    reference='Abraham et al. 1996'
)
def asymmetry_index(X, mask):
    """
    Measures deviation from rotational symmetry.
    
    Formula:
    A = sum(flux_i * |theta_i - pi|) / (pi * sum(flux_i))
    
    where theta_i is the azimuthal angle of source i 
    around the galaxy centroid.
    
    A = 0: perfectly symmetric (elliptical)
    A = 1: maximally asymmetric (irregular)
    """
    flux = X[:, :, 0] * mask
    F_tot = np.sum(flux, axis=1, keepdims=True)
    F_tot = np.where(F_tot == 0, 1e-10, F_tot)
    
    ra = X[:, :, 1]
    dec = X[:, :, 2]
    
    ra_c = np.sum(flux * ra, axis=1, keepdims=True) / F_tot
    dec_c = np.sum(flux * dec, axis=1, keepdims=True) / F_tot
    
    delta_ra = ((ra - ra_c + 180) % 360) - 180
    delta_dec = dec - dec_c
    
    # Azimuthal angle around centroid
    theta = np.arctan2(delta_dec, delta_ra)
    # Map to [0, 2*pi]
    theta = np.mod(theta, 2*np.pi)
    
    A = np.sum(flux * np.abs(theta - np.pi), axis=1) / (np.pi * F_tot.squeeze())
    return A


@attach_metadata(
    name='Concentration Index',
    symbol='C',
    units='dimensionless',
    category='structural',
    expected_discriminating_power='high',
    reference='Conselice 2003, ApJS 147'
)
def concentration_index(X, mask, fraction=0.8):
    """
    Ratio of radius containing fraction of total flux 
    to radius containing all flux.
    
    C = r_80 / r_total
    
    High C: concentrated (elliptical)
    Low C: extended (spiral/irregular)
    """
    flux = X[:, :, 0] * mask
    F_tot = np.sum(flux, axis=1, keepdims=True)
    F_tot = np.where(F_tot == 0, 1e-10, F_tot)
    
    ra = X[:, :, 1]
    dec = X[:, :, 2]
    
    ra_c = np.sum(flux * ra, axis=1, keepdims=True) / F_tot
    dec_c = np.sum(flux * dec, axis=1, keepdims=True) / F_tot
    
    delta_ra = ((ra - ra_c + 180) % 360) - 180
    delta_dec = dec - dec_c
    r = np.sqrt(delta_ra**2 + delta_dec**2)
    
    C = np.zeros(X.shape[0])
    for i in range(X.shape[0]):
        if np.sum(mask[i]) <= 1:
            C[i] = 1.0
            continue
            
        r_i = r[i, mask[i]]
        f_i = flux[i, mask[i]]
        
        sort_idx = np.argsort(r_i)
        cum_flux = np.cumsum(f_i[sort_idx]) / F_tot[i, 0]
        
        # radius containing 80%
        idx_80 = np.searchsorted(cum_flux, fraction)
        if idx_80 >= len(r_i):
            idx_80 = len(r_i) - 1
            
        r_80 = r_i[sort_idx][idx_80]
        r_total = r_i[sort_idx][-1]
        
        if r_total == 0:
            C[i] = 1.0
        else:
            C[i] = r_80 / r_total
            
    return C


@attach_metadata(
    name='Gini Coefficient',
    symbol='Gini',
    units='dimensionless',
    category='structural',
    expected_discriminating_power='high',
    reference='Lotz et al. 2004'
)
def gini_coefficient(X, mask):
    """
    Gini coefficient of flux distribution.
    Measures inequality of flux among stellar sources.
    
    Formula: standard Gini from economics applied to 
    stellar flux values (Lorentz curve area method).
    
    G = 0: all stars equal flux
    G = 1: one star has all flux
    
    Ellipticals: high Gini (central bright nucleus)
    Irregulars: low Gini (distributed HII regions)
    """
    flux = X[:, :, 0]
    G = np.zeros(X.shape[0])
    
    for i in range(X.shape[0]):
        if np.sum(mask[i]) <= 1:
            G[i] = 0.0
            continue
            
        f_i = flux[i, mask[i]]
        # Sort ascending
        f_i = np.sort(f_i)
        n = len(f_i)
        
        # Gini logic
        index = np.arange(1, n+1)
        G[i] = (np.sum((2 * index - n - 1) * f_i)) / (n * np.sum(f_i))
        
    return G


@attach_metadata(
    name='M20 Moment',
    symbol='M_{20}',
    units='dimensionless',
    category='structural',
    expected_discriminating_power='high',
    reference='Lotz et al. 2004'
)
def m20_moment(X, mask):
    """
    Second-order moment of the brightest 20% of flux.
    Sensitive to bright off-center clumps (mergers, 
    star-forming regions in spirals).
    
    M20 = log10(sum(flux_i * r_i^2) / F_total * r_total^2)
    summed over brightest 20% of sources only.
    """
    flux = X[:, :, 0] * mask
    F_tot = np.sum(flux, axis=1, keepdims=True)
    F_tot = np.where(F_tot == 0, 1e-10, F_tot)
    
    ra = X[:, :, 1]
    dec = X[:, :, 2]
    
    ra_c = np.sum(flux * ra, axis=1, keepdims=True) / F_tot
    dec_c = np.sum(flux * dec, axis=1, keepdims=True) / F_tot
    
    delta_ra = ((ra - ra_c + 180) % 360) - 180
    delta_dec = dec - dec_c
    rSq = delta_ra**2 + delta_dec**2
    
    M20 = np.zeros(X.shape[0])
    
    for i in range(X.shape[0]):
        if np.sum(mask[i]) <= 1:
            M20[i] = float('-inf')
            continue
            
        f_i = flux[i, mask[i]]
        r2_i = rSq[i, mask[i]]
        total_moment = np.sum(f_i * r2_i)
        if total_moment == 0:
            M20[i] = float('-inf')
            continue
            
        # Sort flux descending
        sort_idx = np.argsort(f_i)[::-1]
        f_i_sorted = f_i[sort_idx]
        r2_i_sorted = r2_i[sort_idx]
        
        cum_flux = np.cumsum(f_i_sorted)
        # Select brightest 20%
        limit_flux = 0.2 * F_tot[i, 0]
        
        subset_idx = np.where(cum_flux <= limit_flux)[0]
        if len(subset_idx) == 0:
            subset_idx = [0]
            
        m20_val = np.sum(f_i_sorted[subset_idx] * r2_i_sorted[subset_idx])
        if m20_val == 0:
            M20[i] = float('-inf')
        else:
            M20[i] = np.log10(m20_val / total_moment)
            
    # Handle infinite values
    M20 = np.where(np.isinf(M20), -10, M20)
    return M20


@attach_metadata(
    name='Radial Profile Slope',
    symbol='\\gamma',
    units='dimensionless',
    category='structural',
    expected_discriminating_power='low',
    reference='Custom'
)
def radial_profile_slope(X, mask, n_bins=8):
    """
    Slope of flux vs radius profile (Sersic index proxy).
    Fit a line to log(flux) vs log(radius).
    Steep slope: de Vaucouleurs (elliptical, n~4)
    Shallow slope: exponential disk (spiral, n~1)
    """
    flux = X[:, :, 0] * mask
    F_tot = np.sum(flux, axis=1, keepdims=True)
    F_tot = np.where(F_tot == 0, 1e-10, F_tot)
    
    ra = X[:, :, 1]
    dec = X[:, :, 2]
    
    ra_c = np.sum(flux * ra, axis=1, keepdims=True) / F_tot
    dec_c = np.sum(flux * dec, axis=1, keepdims=True) / F_tot
    
    delta_ra = ((ra - ra_c + 180) % 360) - 180
    delta_dec = dec - dec_c
    r = np.sqrt(delta_ra**2 + delta_dec**2)
    
    slopes = np.zeros(X.shape[0])
    for i in range(X.shape[0]):
        if np.sum(mask[i]) < 5:
            slopes[i] = 0.0
            continue
            
        r_i = r[i, mask[i]]
        f_i = flux[i, mask[i]]
        
        valid = (r_i > 0) & (f_i > 0)
        if np.sum(valid) < 3:
            slopes[i] = 0.0
            continue
            
        log_r = np.log10(r_i[valid])
        log_f = np.log10(f_i[valid])
        
        slope, _ = np.polyfit(log_r, log_f, 1)
        slopes[i] = slope
        
    return slopes


# Collect all observables
OBSERVABLES = [
    stellar_multiplicity,
    total_flux,
    half_light_radius,
    flux_dispersion,
    asymmetry_index,
    concentration_index,
    gini_coefficient,
    m20_moment,
    radial_profile_slope
]

def compute_all_observables(X, mask) -> pd.DataFrame:
    """
    Compute ALL observables for all galaxies in one call.
    Returns DataFrame with columns = observable names, 
    rows = galaxies.
    
    This is the main entry point for the library —
    equivalent to running a full jet observable suite 
    on a particle physics dataset.
    """
    results = {}
    for obs_func in OBSERVABLES:
        results[obs_func.__name__] = obs_func(X, mask)
        
    return pd.DataFrame(results)

def observable_correlation_matrix(obs_df, labels) -> plt.Figure:
    """
    Compute and plot correlation matrix of all observables,
    separately for each morphology class.
    Highlight pairs with |correlation| > 0.7.
    """
    classes = np.unique(labels)
    fig, axes = plt.subplots(1, len(classes), figsize=(5*len(classes), 5))
    
    class_names = {0: 'Elliptical', 1: 'Spiral', 2: 'Irregular'}
    
    for i, cls in enumerate(classes):
        corr = obs_df[labels == cls].corr()
        
        ax = axes[i] if len(classes) > 1 else axes
        cax = ax.matshow(corr, cmap='RdBu', vmin=-1, vmax=1)
        ax.set_title(f'{class_names.get(cls, cls)} Correlation')
        
        # Highlight high correlations
        for y in range(corr.shape[0]):
            for x in range(corr.shape[1]):
                val = corr.iloc[y, x]
                if abs(val) > 0.7 and x != y:
                    ax.text(x, y, f'{val:.1f}', va='center', ha='center', color='black', weight='bold')
                
        ax.set_xticks(range(len(corr.columns)))
        ax.set_yticks(range(len(corr.columns)))
        ax.set_xticklabels(corr.columns, rotation=90)
        ax.set_yticklabels(corr.columns)
        
    fig.colorbar(cax, ax=axes.ravel().tolist())
    return fig

def ks_discrimination_table(obs_df, labels) -> pd.DataFrame:
    """
    For each observable, compute KS-statistic between 
    every pair of morphology classes.
    Return sorted table showing most discriminating 
    observables first.
    """
    observables = obs_df.columns
    class_pairs = [(0, 1), (0, 2), (1, 2)]
    pair_names = ['Ellip vs Spiral', 'Ellip vs Irreg', 'Spiral vs Irreg']
    
    results = []
    
    for obs in observables:
        row = {'Observable': obs}
        avg_ks = 0
        for pair, p_name in zip(class_pairs, pair_names):
            d1 = obs_df[labels == pair[0]][obs]
            d2 = obs_df[labels == pair[1]][obs]
            
            # Avoid nan
            d1 = d1.dropna()
            d2 = d2.dropna()
            
            if len(d1) > 0 and len(d2) > 0:
                stat, _ = ks_2samp(d1, d2)
            else:
                stat = 0.0
                
            row[p_name] = stat
            avg_ks += stat
            
        row['Mean_KS'] = avg_ks / len(class_pairs)
        results.append(row)
        
    df_res = pd.DataFrame(results)
    df_res = df_res.sort_values('Mean_KS', ascending=False).drop(columns=['Mean_KS'])
    return df_res

# WIP: half-light radius handling edges

# Fix half light radius for 1-source

# Added flux dispersion and asymmetry
