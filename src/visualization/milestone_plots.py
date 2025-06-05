# src/visualization/milestone_plots.py - Milestone visualization

import matplotlib.pyplot as plt
import numpy as np
from config import COLORS

# Define color scheme
COLORS = {
    'milestone': '#1f77b4',  # Blue
    'milestone_top': '#ff7f0e',  # Orange for top milestones
    'policy': '#d62728',  # Red
    'policy_band1': '#f1f6ff',  # Very light blue
    'policy_band2': '#fff6e6',  # Very light orange
    'policy_line': '#d62728',   # Red
    'relationship': '#2ca02c',  # Green
    'line': '#1f77b4',         # Blue for lines
    'grid': '#cccccc',  # Light gray
    'background': '#f8f8f8'  # Very light gray
}

def create_milestone_timeline(milestone_df, policy_timeline):
    """Create a publication-quality milestone timeline visualization"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Set background color
    ax.set_facecolor(COLORS['background'])
    
    # Plot all milestones
    ax.plot(milestone_df['Year'], milestone_df['Joint_Probability'], 
           color=COLORS['milestone'], linewidth=2, marker='o', 
           markersize=5, label='Health Milestone')
    
    # Highlight top 3 milestones
    top_milestones = milestone_df.head(3)
    
    ax.scatter(top_milestones['Year'], top_milestones['Joint_Probability'], 
              s=120, color=COLORS['milestone_top'], zorder=5, 
              edgecolor='white', linewidth=1, label='Significant Milestone')
    
    # Add annotations for top milestones
    for i, row in top_milestones.iterrows():
        year = int(row['Year'])
        score = row['Joint_Probability']
        policy = row['Nearest_Policy'].split()[:3]  # First few words only
        policy_text = ' '.join(policy) + '...'
        
        # Position annotations to avoid overlap
        x_offset = 0
        y_offset = 0.03
        
        if i % 2 == 0:  # Alternate positioning
            y_offset *= -1
        
        ax.annotate(f"{year}\n({policy_text})", 
                   xy=(year, score),
                   xytext=(x_offset, y_offset * 15),
                   textcoords="offset points",
                   ha='center', va='center',
                   bbox=dict(boxstyle="round,pad=0.3", fc='white', ec=COLORS['milestone_top'], alpha=0.9),
                   fontsize=9)
    
    # Add policy implementation lines (more subtle)
    for year, policy in policy_timeline.items():
        year_int = int(year)
        ax.axvline(x=year_int, color=COLORS['policy'], alpha=0.3, linestyle='-.',
                  linewidth=1, label='Policy' if year == list(policy_timeline.keys())[0] else "")
    
    # Clean up axis and add labels
    ax.set_title("Health Milestone Detection: Bangladesh (1971-Present)", 
                fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel("Year", fontsize=12, labelpad=10)
    ax.set_ylabel("Joint Probability", fontsize=12, labelpad=10)
    
    # Improve grid and ticks
    ax.grid(True, alpha=0.3, linestyle=':')
    ax.set_axisbelow(True)  # Put grid below data points
    
    # Set limits
    ax.set_ylim(0, max(milestone_df['Joint_Probability']) * 1.2)
    
    # Set years every 5 years
    years = sorted(list(set(milestone_df['Year'].tolist() + 
                        [int(y) for y in policy_timeline.keys()])))
    keep_years = [y for y in years if y % 5 == 0]
    ax.set_xticks(keep_years)
    
    # Add legend with better positioning
    ax.legend(loc='upper right', frameon=True, framealpha=0.9, fontsize=10)
    
    plt.tight_layout()
    plt.savefig('outputs/figures/milestone_timeline.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return fig

def create_milestone_policy_relationship(milestone_df, policy_timeline, df_timeseries):
    """Create a visualization showing the relationship between milestones and policies"""
    # Create the figure
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Create year axis
    years = df_timeseries.index.astype(int).values
    min_year = min(years)
    max_year = max(years)
    
    # Plot milestone probability with improved styling
    ax.plot(milestone_df['Year'], milestone_df['Joint_Probability'], 
           color=COLORS['milestone'], linewidth=2.5, marker='o', 
           markersize=6, label='Milestone Probability')
    
    # Create colored bands for policy periods (more subtle)
    policy_years = sorted([int(year) for year in policy_timeline.keys()])
    
    # Add the start of the dataset as the first boundary
    boundaries = [min_year] + policy_years + [max_year]
    
    # Create alternating color bands for policy periods
    for i in range(len(boundaries)-1):
        start = boundaries[i]
        end = boundaries[i+1]
        
        # Alternate colors
        color = COLORS['policy_band1'] if i % 2 == 0 else COLORS['policy_band2']
        
        ax.axvspan(start, end, alpha=0.3, color=color)
    
    # Find top milestones with strongest policy relationships
    milestone_df['Policy_Relationship'] = 1 - (milestone_df['Time_Difference'] / 10)
    milestone_df['Policy_Relationship'] = milestone_df['Policy_Relationship'].clip(0, 1)
    
    top_relationship = milestone_df.sort_values('Policy_Relationship', ascending=False).head(3)
    
    # Mark these milestones and their related policies
    for i, row in top_relationship.iterrows():
        milestone_year = row['Year']
        policy_year = row['Nearest_Policy_Year']
        
        # Highlight the milestone
        ax.scatter(milestone_year, row['Joint_Probability'], 
                  s=120, color=COLORS['milestone'], zorder=5, 
                  edgecolor='white', linewidth=1.5)
        
        # Highlight the related policy with a vertical line
        ax.axvline(x=policy_year, color=COLORS['policy_line'], alpha=0.7, 
                  linestyle='-.', linewidth=1.5, zorder=4)
        
        # Draw a connection between the milestone and policy
        connection_height = row['Joint_Probability'] * 0.8
        ax.annotate('', 
                   xy=(policy_year, connection_height),
                   xytext=(milestone_year, connection_height),
                   arrowprops=dict(arrowstyle='<->', lw=1.5, 
                                  color=COLORS['relationship'], 
                                  connectionstyle='arc3,rad=0.1',
                                  alpha=0.7))
        
        # Add text label
        midpoint = (milestone_year + policy_year) / 2
        ax.text(midpoint, connection_height * 1.05, 
               f"{row['Time_Difference']} years",
               ha='center', va='bottom', fontsize=9,
               bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.7))
        
        # Add policy label with minimal text
        policy_name = row['Nearest_Policy'].split()[:3]
        policy_name = ' '.join(policy_name) + '...'
        ax.annotate(f"{int(policy_year)}: {policy_name}", 
                   xy=(policy_year, connection_height * 0.9),
                   xytext=(0, -15),
                   textcoords="offset points",
                   ha='center', rotation=90, fontsize=9,
                   bbox=dict(boxstyle="round,pad=0.2", fc='white', alpha=0.9))
    
    # Clean up axis
    ax.set_title("Health Policy and Milestone Relationship Analysis", 
                fontsize=14, fontweight='bold')
    ax.set_xlabel("Year", fontsize=12)
    ax.set_ylabel("Milestone Probability", fontsize=12)
    ax.set_ylim(0, max(milestone_df['Joint_Probability']) * 1.2)
   
    # Set years every 5 years
    keep_years = [y for y in range(min_year, max_year+1, 5)]
    ax.set_xticks(keep_years)
   
    # Add grid
    ax.grid(True, alpha=0.3, linestyle=':', which='both')
    ax.set_axisbelow(True)
   
    plt.tight_layout()
    plt.savefig('outputs/figures/policy_milestone_relationship.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return fig

def create_indicators_milestone_comparison(df_timeseries, df_scaled, milestone_df, policy_timeline):
    """Create an indicator comparison with key milestones highlighted"""
    # Select top 4 indicators to keep visualization clean
    indicators = ['Mortality rate, infant (per 1,000 live births)',
                'Life expectancy at birth, total (years)',
                'Maternal mortality ratio (modeled estimate, per 100,000 live births)',
                'Immunization, measles (% of children ages 12-23 months)']
    
    indicators = [ind for ind in indicators if ind in df_timeseries.columns][:4]
    
    # Get top 3 milestones
    top_milestones = milestone_df.head(3)
    
    fig, axes = plt.subplots(len(indicators), 1, figsize=(12, 3*len(indicators)), sharex=True)
    
    # Plot each indicator
    for i, indicator in enumerate(indicators):
        ax = axes[i]
        ax.set_facecolor(COLORS['background'])
        
        # Plot raw data
        ax.plot(df_timeseries.index, df_timeseries[indicator], color=COLORS['line'], linewidth=2.5)
        
        # Add milestone lines
        for j, row in top_milestones.iterrows():
            year = str(int(row['Year']))
            if year in df_timeseries.index:
                ax.axvline(x=year, color=COLORS['milestone'], alpha=0.7, linestyle='--',
                          linewidth=1.5)
                
                # Add annotation for first indicator only to avoid clutter
                if i == 0:
                    ax.annotate(f"Milestone {year}", 
                               xy=(year, ax.get_ylim()[1]),
                               xytext=(0, 10),
                               textcoords="offset points",
                               ha='center', fontsize=9,
                               bbox=dict(boxstyle="round,pad=0.2", fc='white', alpha=0.8))
        
        # Add key policy lines (selective)
        key_policies = ['1982', '1993', '2003', '2011']
        for year in key_policies:
            if year in policy_timeline and year in df_timeseries.index:
                ax.axvline(x=year, color=COLORS['policy'], alpha=0.3, linestyle='-.')
                
                # Add label to alternate sides
                if i % 2 == 0:
                    text_pos = 0.02
                    ha = 'left'
                else:
                    text_pos = 0.98
                    ha = 'right'
                
                ax.text(text_pos, 0.9, f"{policy_timeline[year][:20]}...",
                       transform=ax.transAxes, fontsize=8, color=COLORS['policy'],
                       ha=ha, va='top', alpha=0.8)
        
        # Clean up axis
        indicator_name = indicator.split('(')[0].strip()
        unit = indicator.split('(')[1].split(')')[0] if '(' in indicator else ""
        
        ax.set_title(f"{indicator_name}", fontsize=11, fontweight='bold')
        ax.set_ylabel(unit, fontsize=10)
        ax.grid(True, alpha=0.3, linestyle=':')
        ax.set_axisbelow(True)
    
    # Set common x-axis properties
    plt.xlabel("Year", fontsize=12)
    
    # Set years every 5 years
    years = df_timeseries.index.astype(int).values
    keep_years = [str(y) for y in years if y % 5 == 0]
    plt.xticks([y for y in df_timeseries.index if y in keep_years], rotation=45)
    
    # Add legend to the figure
    handles = [
        plt.Line2D([0], [0], color=COLORS['line'], linewidth=2, label='Health Indicator'),
        plt.Line2D([0], [0], color=COLORS['milestone'], linestyle='--', linewidth=1.5, label='Detected Milestone'),
        plt.Line2D([0], [0], color=COLORS['policy'], linestyle='-.', alpha=0.5, label='Policy Implementation')
    ]
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, 0.02),
              ncol=3, frameon=True, fontsize=10)
    
    fig.suptitle("Bangladesh Health Indicators with Detected Milestones", 
                fontsize=14, fontweight='bold')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.savefig('outputs/figures/indicators_milestone_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return fig