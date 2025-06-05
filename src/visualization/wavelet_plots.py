# src/visualization/wavelet_plots.py - Wavelet visualization

import matplotlib.pyplot as plt
import numpy as np
from config import COLORS

def create_wavelet_changepoint_grid(df_scaled, df_timeseries, wavelet_cps, policy_timeline):
    """Create publication-quality wavelet changepoint visualizations grid"""
    # Select top 4 indicators with the most changepoints for demonstration
    indicator_cp_counts = {ind: len(cps) for ind, cps in wavelet_cps.items()}
    top_indicators = sorted(indicator_cp_counts.keys(), 
                          key=lambda x: indicator_cp_counts[x], reverse=True)[:4]
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), constrained_layout=True)
    axes = axes.flatten()
    
    for i, indicator in enumerate(top_indicators):
        ax = axes[i]
        
        # Plot normalized indicator
        ax.plot(df_timeseries.index, df_scaled[indicator], color=COLORS['data'], 
               linewidth=2, label='_nolegend_')
        
        # Clean, abbreviated title
        indicator_name = indicator.split('(')[0].strip()
        ax.set_title(indicator_name, fontsize=12, fontweight='bold')
        
        # Add select changepoints (limit to clearer visualization)
        level_markers = {1: [], 2: []}
        
        for year_idx, level in wavelet_cps[indicator]:
            if year_idx < len(df_timeseries.index):
                year = df_timeseries.index[year_idx]
                level_markers[level].append((year, df_scaled[indicator][year_idx]))
                ax.axvline(x=year, color=COLORS[f'level{level}'], alpha=0.7, 
                          linestyle='--', linewidth=1)
        
        # Add just 3 key policy lines to reduce clutter
        key_years = ['1982', '1998', '2011']
        for year in key_years:
            if year in df_timeseries.index:
                ax.axvline(x=year, color=COLORS['policy'], alpha=0.4, linestyle='-.')
        
        # Improve axis appearance
        ax.tick_params(axis='x', rotation=45, labelsize=9)
        ax.tick_params(axis='y', labelsize=9)
        ax.grid(True, alpha=0.3, linestyle=':')
        
        # Only add years every 5 years to reduce clutter
        years = df_timeseries.index.astype(int).values
        keep_years = [str(y) for y in years if y % 5 == 0]
        ax.set_xticks([y for y in df_timeseries.index if y in keep_years])
    
    # Add a single legend for the entire figure
    handles = [
        plt.Line2D([0], [0], color=COLORS['data'], linewidth=2, label='Indicator'),
        plt.Line2D([0], [0], color=COLORS['level1'], linestyle='--', label='Level 1 Change'),
        plt.Line2D([0], [0], color=COLORS['level2'], linestyle='--', label='Level 2 Change'),
        plt.Line2D([0], [0], color=COLORS['policy'], linestyle='-.', alpha=0.4, label='Policy')
    ]
    fig.legend(handles=handles, loc='center', bbox_to_anchor=(0.5, 0.04), ncol=4, frameon=True)
    
    fig.suptitle('Wavelet-Based Health Milestone Detection', fontsize=14, fontweight='bold', y=0.98)
    plt.savefig('outputs/figures/wavelet_changepoints.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return fig

def create_individual_wavelet_plots(df_scaled, df_timeseries, wavelet_cps, policy_timeline):
    """Create individual wavelet plots for each indicator"""
    # Create individual indicator plots with cleaner design
    for indicator in df_scaled.columns:
        fig, ax = plt.subplots(figsize=(10, 5))
        
        # Plot data with professional styling
        ax.plot(df_timeseries.index, df_scaled[indicator], color=COLORS['data'], 
               linewidth=2.5, label='Normalized Value')
        
        # Add only meaningful changepoints (filter out noise)
        if indicator in wavelet_cps:
            added_labels = set()
            
            for year_idx, level in wavelet_cps[indicator]:
                if year_idx < len(df_timeseries.index):
                    year = df_timeseries.index[year_idx]
                    label = f'Level {level} Change' if f'Level {level}' not in added_labels else '_nolegend_'
                    added_labels.add(f'Level {level}')
                    
                    ax.axvline(x=year, color=COLORS[f'level{level}'], alpha=0.7, 
                              linestyle='--', linewidth=1.5, label=label)
        
        # Add policy lines more selectively (only those near changepoints)
        added_policy = False
        for year, policy in policy_timeline.items():
            if year in df_timeseries.index:
                label = 'Policy Implementation' if not added_policy else '_nolegend_'
                added_policy = True
                ax.axvline(x=year, color=COLORS['policy'], alpha=0.4, 
                          linestyle='-.', label=label)
        
        # Clean title and labels
        indicator_name = indicator.split('(')[0].strip()
        ax.set_title(f'{indicator_name} (1971-Present)', fontsize=14, fontweight='bold')
        ax.set_xlabel('Year', fontsize=12)
        ax.set_ylabel('Standardized Value', fontsize=12)
        
        # Improve tick appearance
        ax.tick_params(axis='x', rotation=45, labelsize=10)
        ax.tick_params(axis='y', labelsize=10)
        ax.grid(True, alpha=0.3, linestyle=':')
        
        # Selective xticks to avoid crowding
        years = df_timeseries.index.astype(int).values
        keep_years = [str(y) for y in years if y % 5 == 0]
        ax.set_xticks([y for y in df_timeseries.index if y in keep_years])
        
        # Add concise legend
        ax.legend(loc='best', frameon=True, framealpha=0.9, fontsize=10)
        
        plt.tight_layout()
        
        # Save with clean filename
        clean_name = indicator.replace('/', '_').replace(',', '').replace(' ', '_').replace('(', '').replace(')', '')[:40]
        plt.savefig(f'outputs/figures/wavelet_{clean_name}.png', dpi=300, bbox_inches='tight')
        plt.close()