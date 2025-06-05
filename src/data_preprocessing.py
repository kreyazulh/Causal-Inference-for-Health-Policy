# src/data_preprocessing.py - Data loading and preparation functions

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import warnings

# Suppress unnecessary warnings
warnings.filterwarnings('ignore')

def load_and_preprocess_data(file_path='data/health_zwe.csv'):
    """Load and preprocess the health data for different countries"""
    print(f"Loading data from {file_path}...")
    
    # Load the dataset
    df = pd.read_csv(file_path)
    
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    
    # Convert columns to appropriate types
    df['Year'] = df['Year'].astype(str)
    df['Value'] = pd.to_numeric(df['Value'], errors='coerce')
    
    # Determine country from file path
    country = None
    if 'zwe' in file_path:
        country = 'Zimbabwe'
    elif 'phl' in file_path:
        country = 'Philippines'
    elif 'bgd' in file_path:
        country = 'Bangladesh'
    
    # Filter for specific country data if needed
    if 'Country Name' in df.columns and country:
        country_data = df[df['Country Name'] == country]
        if len(country_data) > 0:
            df = country_data
            print(f"{country} data shape: {df.shape}")
        else:
            print(f"Warning: No data for {country} found, using all data.")
    
    # Focus on post-independence period (1971 onwards)
    df_post = df[df['Year'].astype(int) >= 1971]
    print(f"Post-independence data shape: {df_post.shape}")
    
    # Show available indicators and their data coverage
    indicator_coverage = df_post.groupby('Indicator Name').size().sort_values(ascending=False)
    print("\nTop 10 indicators by data coverage:")
    print(indicator_coverage.head(10))
    
    return df_post

def select_key_indicators(df, potential_indicators, min_years_required=20):
    """Select key indicators from the dataset based on data availability"""
    selected_indicators = []
    coverage_info = {}
    
    for indicator in potential_indicators:
        indicator_data = df[df['Indicator Name'] == indicator]
        year_coverage = len(indicator_data['Year'].unique())
        
        coverage_info[indicator] = {
            'years_available': year_coverage,
            'year_range': f"{indicator_data['Year'].astype(int).min()}-{indicator_data['Year'].astype(int).max()}" if len(indicator_data) > 0 else "N/A"
        }
        
        if year_coverage >= min_years_required:
            selected_indicators.append(indicator)
    
    print(f"\nSelected {len(selected_indicators)} indicators with at least {min_years_required} years of data")
    for indicator in selected_indicators:
        print(f"- {indicator}: {coverage_info[indicator]['years_available']} years ({coverage_info[indicator]['year_range']})")
    
    return selected_indicators, coverage_info

def create_time_series_dataset(df, indicators):
    """Create a time series dataset for selected indicators"""
    # Filter for selected indicators
    df_indicators = df[df['Indicator Name'].isin(indicators)]
    
    # Pivot to create time series format (years as index, indicators as columns)
    df_timeseries = df_indicators.pivot_table(
        index='Year', 
        columns='Indicator Name', 
        values='Value'
    ).astype(float)
    
    # Handle missing values using forward and backward fill, then linear interpolation
    df_timeseries = df_timeseries.fillna(method='ffill').fillna(method='bfill').interpolate()
    
    # Normalize the data using z-scores for comparable scales
    scaler = StandardScaler()
    df_scaled = pd.DataFrame(
        scaler.fit_transform(df_timeseries), 
        index=df_timeseries.index, 
        columns=df_timeseries.columns
    )
    
    return df_timeseries, df_scaled