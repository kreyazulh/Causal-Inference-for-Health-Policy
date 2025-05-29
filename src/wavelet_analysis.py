# src/wavelet_analysis.py - Wavelet-based change point detection

import numpy as np
import pywt
from scipy.signal import find_peaks

def multi_resolution_decomposition(timeseries, wavelet='db4', level=2):
    """Perform wavelet decomposition at multiple resolution levels"""
    # Convert to numpy array if it's not already
    ts_values = np.array(timeseries)
    
    # Check if we have enough data for the requested level
    max_level = pywt.dwt_max_level(len(ts_values), pywt.Wavelet(wavelet).dec_len)
    actual_level = min(level, max_level)
    
    if actual_level < level:
        print(f"Warning: Requested level {level} exceeds maximum possible level {max_level}. Using level {actual_level}.")
    
    # Perform wavelet decomposition
    coeffs = pywt.wavedec(ts_values, wavelet, level=actual_level)
    
    # Return approximation and details
    return coeffs[0], coeffs[1:]

def detect_potential_changepoints(details, timeseries, threshold_factor=0.7):
    """Detect potential change points across wavelet detail coefficients"""
    potential_cps = []
    
    for level, detail in enumerate(details):
        # Use peak detection on absolute coefficient values
        try:
            # More sensitive peak detection
            peaks, _ = find_peaks(np.abs(detail), 
                                height=threshold_factor*np.std(detail),
                                distance=2)  # Minimum distance between peaks
            
            # Scale peak locations back to original time scale
            scale_factor = len(timeseries) / len(detail)
            scaled_peaks = [int(p * scale_factor) for p in peaks]
            
            # Store with level information
            for peak in scaled_peaks:
                if peak < len(timeseries):  # Ensure peak is within bounds
                    potential_cps.append((peak, level + 1))
        except Exception as e:
            print(f"Error detecting peaks at level {level}: {str(e)}")
    
    return potential_cps

def analyze_all_indicators(df_scaled):
    """Perform wavelet analysis on all indicators and detect change points"""
    wavelet_cps = {}
    
    # Process wavelet analysis for each indicator
    for indicator in df_scaled.columns:
        timeseries = df_scaled[indicator]
        approx, details = multi_resolution_decomposition(timeseries)
        changepoints = detect_potential_changepoints(details, timeseries)
        wavelet_cps[indicator] = changepoints
    
    return wavelet_cps