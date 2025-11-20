"""
GalaxyClouds - A Point Cloud Observable Library for Galaxy Morphology Analysis.
"""

from .io import GalaxyDataset, generate_synthetic_galaxies, load_sdss_catalog
from .observables import compute_all_observables, ks_discrimination_table, observable_correlation_matrix
from .transforms import CoordinateFrame, transform_to_galaxy_frame
from .classification import build_feature_matrix, train_morphology_classifier, evaluate_classifier

__all__ = [
    "GalaxyDataset", 
    "generate_synthetic_galaxies", 
    "load_sdss_catalog",
    "compute_all_observables", 
    "ks_discrimination_table", 
    "observable_correlation_matrix",
    "CoordinateFrame", 
    "transform_to_galaxy_frame",
    "build_feature_matrix", 
    "train_morphology_classifier", 
    "evaluate_classifier"
]

# Export observables
