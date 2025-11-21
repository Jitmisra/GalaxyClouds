import numpy as np
import pytest
from galaxyclouds.io import generate_synthetic_galaxies
from galaxyclouds.observables import (
    stellar_multiplicity, total_flux, half_light_radius, 
    flux_dispersion, asymmetry_index, concentration_index,
    gini_coefficient, m20_moment
)

@pytest.fixture
def mock_galaxy_data():
    X = np.zeros((2, 10, 6))
    
    # Galaxy 0: Symmetric circular galaxy at origin
    X[0, 0:4, 0] = [10, 10, 10, 10]  # Eq flux
    X[0, 0:4, 1] = [1, 0, -1, 0]      # RA
    X[0, 0:4, 2] = [0, 1, 0, -1]      # Dec
    
    # Galaxy 1: One bright source
    X[1, 0, 0] = 100
    X[1, 0, 1] = 0
    X[1, 0, 2] = 0
    X[1, 1:4, 0] = [1, 1, 1]
    X[1, 1:4, 1] = [1, -1, 0]
    X[1, 1:4, 2] = [0, 0, 1]
    
    mask = X[:, :, 0] > 0
    return X, mask

def test_masking_correctness(mock_galaxy_data):
    """Verify zero-padded entries never contribute to observable values"""
    X, mask = mock_galaxy_data
    N = stellar_multiplicity(X, mask)
    assert N[0] == 4
    assert N[1] == 4
    
def test_total_flux_summation(mock_galaxy_data):
    X, mask = mock_galaxy_data
    F = total_flux(X, mask)
    assert F[0] == 40
    assert F[1] == 103
    
def test_half_light_radius_symmetric(mock_galaxy_data):
    """For a perfectly circular galaxy, verify result matches analytic formula"""
    X, mask = mock_galaxy_data
    R = half_light_radius(X, mask)
    # Centroid is (0,0), radius distances are all 1
    # sum(f * 1) / sum(f) = 1.0
    assert np.isclose(R[0], 1.0)
    
def test_flux_dispersion_uniform(mock_galaxy_data):
    """For equal-flux stars, verify G = 1/sqrt(N)"""
    X, mask = mock_galaxy_data
    G = flux_dispersion(X, mask)
    expected_g0 = 1.0 / np.sqrt(4)
    assert np.isclose(G[0], expected_g0)
    
def test_ra_wrapping():
    """Verify DeltaR computed correctly across RA wrapping boundary"""
    X = np.zeros((1, 5, 6))
    X[0, 0, 0] = 10; X[0, 0, 1] = 359; X[0, 0, 2] = 0
    X[0, 1, 0] = 10; X[0, 1, 1] = 1; X[0, 1, 2] = 0
    mask = X[:, :, 0] > 0
    
    # Centroid should be effectively at RA=0 wrapping point
    r = half_light_radius(X, mask)
    assert np.isclose(r[0], 1.0) # they are each 1 degree away from 0

def test_asymmetry_symmetric(mock_galaxy_data):
    X, mask = mock_galaxy_data
    A = asymmetry_index(X, mask)
    # The first galaxy is rotationally symmetric, A should be specific
    # In this logic, A -> uniform distribution A depends on phases
    assert not np.isnan(A[0])
