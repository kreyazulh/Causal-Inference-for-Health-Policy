# src/visualization/policy_plots.py - Policy impact visualization

import matplotlib.pyplot as plt
from config import COLORS

def create_policy_impact_visualization(impact_df, top_policies=5):
    """Create a publication-quality policy impact visualization"""
    if impact_df.empty:
        return
    
    # Select top policies by absolute impact
    impact_df['Abs_Impact'] = impact_df['Impact'].abs()
    policy_impact = impact_df.groupby('Policy_Name')['Abs_Impact'].mean().sort_values(ascending=False)
    top_policy_names = policy_impact.index[:top_policies].tolist()
    
    # Create a more focused visualization
    fig, axes = plt.subplots(top_policies, 1, figsize=(10, 3*top_policies), constrained_layout=True)
    
    if top_policies == 1:
        axes = [axes]  # Make sure axes is a list
    
    for i, policy in enumerate(top_policy_names):
        policy_data = impact_df[impact_df['Policy_Name'] == policy].copy()
        policy_year = policy_data.iloc[0]['Policy_Year']
        
        # Sort by impact and limit to top effects for clarity
        policy_data = policy_data.sort_values('Abs_Impact', ascending=False).head(6)
        
        # Sort for presentation (positive first, then negative)
        policy_data = policy_data.sort_values('Impact', ascending=False)
        
        ax = axes[i]
        ax.set_facecolor(COLORS['background'])
        
        # Create horizontal bar chart
        indicator_names = [name.split('(')[0].strip() for name in policy_data['Indicator']]
        indicator_names = [name[:30] + '...' if len(name) > 30 else name for name in indicator_names]
        
        bars = ax.barh(indicator_names, policy_data['Impact'], height=0.6)
        
        # Color positive and negative impacts
        for j, bar in enumerate(bars):
            bar.set_color(COLORS['positive'] if policy_data.iloc[j]['Impact'] > 0 else COLORS['negative'])
            
            # Add value labels at the end of each bar
            value = policy_data.iloc[j]['Impact']
            ax.text(value + (0.01 * ax.get_xlim()[1] if value >= 0 else -0.05 * ax.get_xlim()[1]),
                   j,
                   f"{value:.1f}%",
                   va='center',
                   ha='left' if value >= 0 else 'right',
                   fontsize=9,
                   fontweight='bold',
                   color='black')
        
        # Add reference line at zero
        ax.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
        
        # Clean up axis
        ax.set_title(f"{policy} ({policy_year})", fontsize=12, fontweight='bold')
        ax.set_xlabel("% Change Difference (Post-Pre)", fontsize=10)
        
        # Improve tick appearance
        ax.tick_params(axis='y', labelsize=10)
        ax.tick_params(axis='x', labelsize=9)
        ax.grid(True, axis='x', alpha=0.3, linestyle=':')
        ax.set_axisbelow(True)
    
    fig.suptitle("Health Policy Impact Assessment", fontsize=14, fontweight='bold')
    plt.savefig('outputs/figures/policy_impact_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return fig