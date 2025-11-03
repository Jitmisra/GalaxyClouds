import numpy as np
import pandas as pd

class GalaxyDataset:
    """
    Loads and manages galaxy point cloud datasets.
    Handles variable-length stellar catalogs with zero-padding.
    
    Each galaxy is stored as a zero-padded array of stellar 
    observations. Real observations have flux > 0; padded 
    entries have all features = 0. This is the standard 
    representation for variable-length point clouds in 
    computational physics.
    """
    
    def __init__(self, filepath=None, X=None, y=None):
        self.X = X
        self.y = y
        self.feature_names = ['flux', 'ra_offset', 'dec_offset', 'spectral_class', 'redshift', 'magnitude']
        if filepath is not None:
            self.load_npz(filepath)
            
    def load_npz(self, filepath) -> tuple[np.ndarray, np.ndarray]:
        """Load dataset from compressed npz file."""
        data = np.load(filepath)
        self.X = data['X']
        self.y = data['y']
        return self.X, self.y
        
    def get_mask(self) -> np.ndarray:
        """
        Boolean mask of shape (N_galaxies, max_stars) where 
        True indicates a real stellar observation vs padding.
        Uses flux > 0 as the masking criterion.
        Critical for correct computation of all observables —
        padded zeros must be excluded from all calculations.
        """
        if self.X is None:
            raise ValueError("Data not loaded.")
        # Feature 0 is flux
        return self.X[:, :, 0] > 0
        
    def get_class(self, label) -> np.ndarray:
        """Return subset of galaxies by morphology class"""
        if self.X is None or self.y is None:
            raise ValueError("Data not loaded.")
        return self.X[self.y == label]
        
    def summary_statistics(self) -> pd.DataFrame:
        """
        Return DataFrame with per-class statistics:
        mean multiplicity, std multiplicity, mean total flux,
        mean angular spread. Formatted as a clean comparison table.
        """
        mask = self.get_mask()
        multiplicity = mask.sum(axis=1)
        total_flux = (self.X[:, :, 0] * mask).sum(axis=1)
        
        # Simple angular spread approximation for summary
        ra_spread = np.std(np.where(mask, self.X[:, :, 1], np.nan), axis=1)
        dec_spread = np.std(np.where(mask, self.X[:, :, 2], np.nan), axis=1)
        angular_spread = np.sqrt(ra_spread**2 + dec_spread**2)
        
        df = pd.DataFrame({
            'class': self.y,
            'multiplicity': multiplicity,
            'total_flux': total_flux,
            'angular_spread': angular_spread
        })
        
        # Mapping for display
        class_names = {0: 'Elliptical', 1: 'Spiral', 2: 'Irregular'}
        df['class'] = df['class'].map(class_names)
        
        summary = df.groupby('class').agg({
            'multiplicity': ['mean', 'std'],
            'total_flux': ['mean'],
            'angular_spread': ['mean']
        }).round(3)
        return summary


def load_sdss_catalog(ra_center, dec_center, radius_deg):
    """Load real SDSS sources from a sky region"""
    # Placeholder for actual astronomical data query
    raise NotImplementedError("Requires active SDSS connection.")


def generate_synthetic_galaxies(n_per_class=10000, seed=42):
    """
    Generate realistic synthetic galaxy point clouds 
    for each morphology class.
    
    Elliptical: King profile flux distribution, 
                circular spatial distribution
    Spiral: exponential disk + arm perturbation
    Irregular: clumpy, asymmetric distribution
    
    Returns X, y arrays in standard zero-padded format.
    """
    np.random.seed(seed)
    max_stars = 100
    n_features = 6  # flux, ra, dec, spec, z, mag
    
    X_all = []
    y_all = []
    
    for class_idx, name in enumerate(['elliptical', 'spiral', 'irregular']):
        if name == 'elliptical':
            n_stars_mean, n_stars_std = 30, 5
        elif name == 'spiral':
            n_stars_mean, n_stars_std = 60, 10
        else:
            n_stars_mean, n_stars_std = 80, 15
            
        N_stars = np.clip(np.random.normal(n_stars_mean, n_stars_std, n_per_class).astype(int), 5, max_stars)
        
        X_class = np.zeros((n_per_class, max_stars, n_features))
        
        for i in range(n_per_class):
            n = N_stars[i]
            
            # RA, Dec distributions
            if name == 'elliptical':
                # Concentrated, symmetric
                ra = np.random.normal(0, 0.5, n)
                dec = np.random.normal(0, 0.5, n)
                flux = np.random.exponential(100, n)
                # Central bright core
                flux[0] = flux.max() * 5
            elif name == 'spiral':
                # Elliptical disk
                theta = np.random.uniform(0, 2*np.pi, n)
                r = np.random.exponential(1.5, n)
                ra = r * np.cos(theta) * 1.5
                dec = r * np.sin(theta) * 0.5
                flux = np.random.exponential(50, n)
            else:
                # Clumpy irregular
                centers = np.random.uniform(-2, 2, (3, 2))
                assignments = np.random.randint(0, 3, n)
                ra = centers[assignments, 0] + np.random.normal(0, 0.8, n)
                dec = centers[assignments, 1] + np.random.normal(0, 0.8, n)
                flux = np.random.uniform(10, 80, n)
            
            # Rotation and shift randomly to simulate sky effects
            angle = np.random.uniform(0, 2*np.pi)
            ra_rot = ra * np.cos(angle) - dec * np.sin(angle)
            dec_rot = ra * np.sin(angle) + dec * np.cos(angle)
            
            # Fill the features
            X_class[i, :n, 0] = np.clip(flux, 0.1, None)  # Ensure positive flux
            X_class[i, :n, 1] = ra_rot
            X_class[i, :n, 2] = dec_rot
            X_class[i, :n, 3] = np.random.randint(0, 7, n)  # Spectral class
            X_class[i, :n, 4] = np.random.uniform(0.01, 0.1, n)  # redshift
            X_class[i, :n, 5] = 22.5 - 2.5 * np.log10(X_class[i, :n, 0])  # simple mag approx
            
        X_all.append(X_class)
        y_all.append(np.full(n_per_class, class_idx))
        
    X_combined = np.concatenate(X_all, axis=0)
    y_combined = np.concatenate(y_all, axis=0)
    
    # Shuffle
    idx = np.random.permutation(len(y_combined))
    return X_combined[idx], y_combined[idx]

# Added King profile generators
