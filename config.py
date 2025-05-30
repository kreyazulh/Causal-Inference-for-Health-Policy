# config.py - Configuration settings and constants

import os

# Create necessary directories
os.makedirs('outputs/figures', exist_ok=True)
os.makedirs('outputs/results', exist_ok=True)

# Policy timeline data
POLICY_TIMELINE = {
    '1972': 'National Health Policy in first Five Year Plan',
    '1976': 'Population Control and Family Planning Program',
    '1978': 'Adoption of Alma-Ata Declaration principles',
    '1982': 'National Drug Policy implementation',
    '1988': 'National Health Policy established',
    '1993': 'National Immunization Program expansion',
    '1998': 'Health and Population Sector Programme (HPSP)',
    '2000': 'Bangladesh Integrated Nutrition Project',
    '2003': 'Health, Nutrition and Population Sector Program',
    '2005': 'National HIV/AIDS Policy',
    '2008': 'Revitalized National Health Policy',
    '2011': 'Health Population and Nutrition Sector Development Program',
    '2016': 'Health Care Financing Strategy',
    '2021': 'Bangladesh Health Sector Strategy 2022-2031'
}

# Potential indicators to analyze
POTENTIAL_INDICATORS = [
    'Mortality rate, infant (per 1,000 live births)',
    'Life expectancy at birth, total (years)',
    'Immunization, measles (% of children ages 12-23 months)',
    'Prevalence of undernourishment (% of population)',
    'Maternal mortality ratio (modeled estimate, per 100,000 live births)',
    'Mortality rate, under-5 (per 1,000 live births)',
    'Incidence of tuberculosis (per 100,000 people)',
    'Hospital beds (per 1,000 people)'
]

# Visualization colors
COLORS = {
    'data': '#1f77b4',          # Blue for time series
    'level1': '#ff7f0e',        # Orange for level 1 changepoints
    'level2': '#2ca02c',        # Green for level 2 changepoints
    'policy': '#d62728',        # Red for policy lines
    'milestone': '#9467bd',     # Purple for milestones
    'positive': '#2ca02c',      # Green for positive impacts
    'negative': '#d62728',      # Red for negative impacts
    'neutral': '#7f7f7f',       # Gray for neutral
    'background': '#f8f8f8',    # Very light gray background
    'uncertainty': '#ff7f0e',   # Orange for uncertainty
}

# Analysis parameters
MIN_YEARS_REQUIRED = 20         # Minimum years of data required for indicators
WAVELET_TYPE = 'db4'            # Wavelet type for decomposition
WAVELET_LEVEL = 2               # Wavelet decomposition level
THRESHOLD_FACTOR = 0.7          # Threshold for change point detection
POLICY_WINDOW_SIZE = 3          # Window size for policy impact analysis