# src/visualization/uncertainty_plots.py - Uncertainty visualization

import matplotlib.pyplot as plt
import numpy as np
from config import COLORS

def create_uncertainty_visualization(milestone_df, wavelet_cps, df_timeseries):
    """Create a visualization highlighting uncertainty in milestone detection"""
    # Create figure with two panels: milestone probability and uncertainty
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True, 
                                gridspec_kw={'height_ratios': [2, 1]})
    
    # Set background color
    ax1.set_facecolor(COLORS['background'])
    ax2.set_facecolor(COLORS['background'])
    
    # Get years array
    years = df_timeseries.index.astype(int).values
    
    # Panel 1: Milestone probability
    ax1.plot(milestone_df['Year'], milestone_df['Joint_Probability'], 
           color=COLORS['milestone'], linewidth=2.5, marker='o', 
           markersize=6, label='Milestone Probability')
    
    # Add uncertainty band (decreases with higher joint probability)
    uncertainty = 1 - milestone_df['Joint_Probability']
    
    # Find top milestones
    top_milestones = milestone_df.sort_values('Joint_Probability', ascending=False).head(3)
    
    # Highlight top milestones and annotate with uncertainty
    for i, row in top_milestones.iterrows():
        year = row['Year']
        prob = row['Joint_Probability']
        uncert = 1 - prob
        
        # Highlight milestone
        ax1.scatter(year, prob, s=120, color=COLORS['milestone'], zorder=5, 
                  edgecolor='white', linewidth=1.5)
        
        # Add annotation with uncertainty information
        ax1.annotate(f"Year {int(year)}\nCertainty: {prob:.2f}",
                   xy=(year, prob),
                   xytext=(0, 15),
                   textcoords="offset points",
                   ha='center',
                   bbox=dict(boxstyle="round,pad=0.3", fc='white', alpha=0.9),
                   fontsize=10)
    
    # Panel 2: Uncertainty visualization
    ax2.bar(milestone_df['Year'], milestone_df['Joint_Probability'], 
          color=COLORS['milestone'], alpha=0.7, width=1.5, label='Certainty')
    
    # Overlay uncertainty
    ax2.bar(milestone_df['Year'], uncertainty, bottom=milestone_df['Joint_Probability'],
          color=COLORS['uncertainty'], alpha=0.7, width=1.5, label='Uncertainty')
    
    # Add a line showing the certainty threshold
    certainty_threshold = 0.5
    ax2.axhline(y=certainty_threshold, color='black', linestyle='--', 
              alpha=0.7, label=f"Certainty Threshold ({certainty_threshold})")
    
    # Add text labels for high uncertainty milestones
    high_uncertainty = milestone_df[milestone_df['Joint_Probability'] < certainty_threshold].head(3)
    for i, row in high_uncertainty.iterrows():
        year = row['Year']
        prob = row['Joint_Probability']
        ax2.text(year, 0.5, f"{int(year)}\n±{(1-prob):.2f}", 
               ha='center', va='bottom', fontsize=9, 
               bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.7))
    
    # Set titles and labels
    ax1.set_title("Health Milestone Detection with Uncertainty Quantification", 
                fontsize=14, fontweight='bold')
    ax1.set_ylabel("Joint Probability", fontsize=12)
    ax2.set_xlabel("Year", fontsize=12)
    ax2.set_ylabel("Probability Breakdown", fontsize=12)
    
    # Set y-limits
    ax1.set_ylim(0, max(milestone_df['Joint_Probability']) * 1.2)
    ax2.set_ylim(0, 1)
    
    # Add grids
    ax1.grid(True, alpha=0.3, linestyle=':')
    ax2.grid(True, alpha=0.3, linestyle=':')
    
    # Set x-ticks to every 5 years
    keep_years = [y for y in range(min(years), max(years)+1, 5)]
    plt.xticks(keep_years, rotation=45)
    
    # Add legends
    ax1.legend(loc='upper right', frameon=True, framealpha=0.9, fontsize=10)
    ax2.legend(loc='upper right', frameon=True, framealpha=0.9, fontsize=10)
    
    plt.tight_layout()
    plt.savefig('outputs/figures/uncertainty_visualization.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return fig