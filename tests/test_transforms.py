import numpy as np
import pytest
from galaxyclouds.transforms import transform_to_galaxy_frame

def test_transform_verification():
    """
    After transform:
    centroid < 1e-10 and cross-term < 1e-10
    """
    # Create elliptical galaxy-like distribution
    X = np.zeros((1, 50, 6))
    theta = np.linspace(0, 2*np.pi, 50)
    r = 2.0
    # elongated along x + y = 0
    x = r * np.cos(theta) * 2.5
    y = r * np.sin(theta) * 0.5
    
    # rotate by 45 deg
    angle = np.pi/4
    ra = x * np.cos(angle) - y * np.sin(angle)
    dec = x * np.sin(angle) + y * np.cos(angle)
    
    # offset
    ra += 10.0
    dec -= 5.0
    
    X[0, :, 0] = 10.0 # uniform flux
    X[0, :, 1] = ra
    X[0, :, 2] = dec
    mask = X[:, :, 0] > 0
    
    X_trans, params = transform_to_galaxy_frame(X, mask)
    
    # Verify centroid is 0
    x_c = np.mean(X_trans[0, :, 1])
    y_c = np.mean(X_trans[0, :, 2])
    assert np.abs(x_c) < 1e-10
    assert np.abs(y_c) < 1e-10
    
    # Verify cross term is 0
    cov_xy = np.mean(X_trans[0, :, 1] * X_trans[0, :, 2])
    assert np.abs(cov_xy) < 1e-10
