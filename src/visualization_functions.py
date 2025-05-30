def visualize_all_methods(benchmark_results, output_dir):
    """Create visualizations comparing all causal inference methods"""
    import os
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Professional color palette
    colors = {
        'BayesianCausal': '#1f77b4',  # Blue
        'ITS': '#ff7f0e',            # Orange
        'DiD': '#2ca02c',            # Green
        'SCM': '#d62728',            # Red
        'ASCM': '#9467bd',           # Purple
        'CausalForest': '#8c564b',   # Brown
        'CausalImpact': '#17becf',   # Cyan
        'Granger': '#bcbd22',        # Olive
        'DoubleML': '#7f7f7f',       # Gray
        'meta_dml': '#e377c2',        # Pink
        'CausalForests': '#9467bd',   # Purple
        'BART': '#d62728',           # Red
        'PSM': '#ff7f0e'             # Orange
    }
    
    # Create figure for each indicator
    for indicator in benchmark_results.keys():
        if indicator == 'calibration':  # Skip calibration results
            continue
        for policy_year in benchmark_results[indicator].keys():
            fig, ax = plt.subplots(figsize=(12, 6))
            # Plot results for each method
            for method in benchmark_results[indicator][policy_year].keys():
                method_results = benchmark_results[indicator][policy_year][method]
                # Handle both dictionary and float results
                if isinstance(method_results, dict):
                    effect = method_results.get('relative_effect', np.nan)
                    lower_bound = method_results.get('lower_bound', np.nan)
                    upper_bound = method_results.get('upper_bound', np.nan)
                else:
                    effect = method_results
                    lower_bound = np.nan
                    upper_bound = np.nan
                # Plot effect with error bars if available
                if not np.isnan(effect):
                    if not np.isnan(lower_bound) and not np.isnan(upper_bound):
                        lower_err = max(effect - lower_bound, 0)
                        upper_err = max(upper_bound - effect, 0)
                        ax.errorbar(method, effect, 
                                  yerr=[[lower_err], [upper_err]],
                                  fmt='o', color=colors.get(method, 'gray'),
                                  capsize=5, capthick=2, elinewidth=2)
                    else:
                        ax.scatter(method, effect, color=colors.get(method, 'gray'))
            # Clean up plot
            ax.set_title(f"Effect Estimates: {indicator} (Policy Year: {policy_year})", fontsize=14, fontweight='bold')
            ax.set_xlabel("Method", fontsize=12)
            ax.set_ylabel("Relative Effect (%)", fontsize=12)
            ax.grid(True, alpha=0.3, linestyle=':')
            plt.xticks(rotation=45)
            ax.axhline(y=0, color='black', linestyle='--', alpha=0.3)
            plt.tight_layout()
            plt.savefig(f'{output_dir}/method_comparison_{indicator.replace("/", "_")}_{policy_year}.png', 
                       dpi=300, bbox_inches='tight')
            plt.close()
    
    # 2. Create method comparison by indicator
    indicators_to_plot = list(benchmark_results.keys())
    n_indicators = len(indicators_to_plot)
    
    if n_indicators > 0:
        fig, axes = plt.subplots(n_indicators, 1, figsize=(14, 4*n_indicators), sharex=True)
        if n_indicators == 1:
            axes = [axes]
        
        for i, indicator in enumerate(indicators_to_plot):
            ax = axes[i]
            ind_data = benchmark_results[indicator]
            
            # Find methods with data for this indicator
            methods_for_indicator = []
            for policy_year in ind_data.keys():
                methods_for_indicator.extend(list(ind_data[policy_year].keys()))
            methods_for_indicator = list(set(methods_for_indicator))  # Remove duplicates
            x_positions = np.arange(1, len(methods_for_indicator) + 1)
            width = 0.8
            
            # Plot individual points and means
            for j, method in enumerate(methods_for_indicator):
                effects = []
                lower_bounds = []
                upper_bounds = []
                
                # Collect data across all policy years for this method
                for policy_year in ind_data.keys():
                    if method in ind_data[policy_year]:
                        method_data = ind_data[policy_year][method]
                        if isinstance(method_data, dict):
                            effect = method_data.get('relative_effect', np.nan)
                            lower_bound = method_data.get('lower_bound', np.nan)
                            upper_bound = method_data.get('upper_bound', np.nan)
                        else:
                            effect = method_data
                            lower_bound = np.nan
                            upper_bound = np.nan
                        
                        if not np.isnan(effect):
                            effects.append(effect)
                            lower_bounds.append(lower_bound)
                            upper_bounds.append(upper_bound)
                
                if effects:  # Only plot if we have data
                    # Create positions with jitter
                    pos = np.random.normal(x_positions[j], 0.1, size=1)
                    mean_effect = np.mean(effects)
                    ax.scatter(pos, mean_effect, color=colors.get(method, 'gray'), s=80, alpha=0.7,
                              label=method if i == 0 else "")
                    
                    # Add error bars if available
                    if any(not np.isnan(lb) for lb in lower_bounds) and any(not np.isnan(ub) for ub in upper_bounds):
                        mean_lower = np.nanmean(lower_bounds)
                        mean_upper = np.nanmean(upper_bounds)
                        lower_err = max(mean_effect - mean_lower, 0)
                        upper_err = max(mean_upper - mean_effect, 0)
                        ax.errorbar([pos[0]], [mean_effect], 
                                  yerr=[[lower_err], [upper_err]],
                                  fmt='o', color=colors.get(method, 'gray'),
                                  capsize=5, capthick=2, elinewidth=2)
            
            # Add domain constraint boundaries if applicable
            if 'mortality' in indicator.lower() or 'ratio' in indicator.lower():
                ax.axhline(y=-90, color='red', linestyle='--', alpha=0.5)
                ax.axhline(y=10, color='red', linestyle='--', alpha=0.5)
                ylim = (-110, 50)
                ax.axhspan(-90, 10, alpha=0.1, color='green', label='Domain-Constrained Region' if i == 0 else "")
            elif 'life expectancy' in indicator.lower():
                ax.axhline(y=-10, color='red', linestyle='--', alpha=0.5)
                ax.axhline(y=30, color='red', linestyle='--', alpha=0.5)
                ylim = (-20, 40)
                ax.axhspan(-10, 30, alpha=0.1, color='green', label='Domain-Constrained Region' if i == 0 else "")
            elif 'immunization' in indicator.lower():
                ax.axhline(y=-20, color='red', linestyle='--', alpha=0.5)
                ax.axhline(y=100, color='red', linestyle='--', alpha=0.5)
                ylim = (-30, 120)
                ax.axhspan(-20, 100, alpha=0.1, color='green', label='Domain-Constrained Region' if i == 0 else "")
            else:
                ylim = (-100, 100)
            
            ax.set_ylim(ylim)
            ax.set_ylabel('Effect (%)')
            ax.set_title(indicator)
            ax.set_xticks(x_positions)
            ax.set_xticklabels(methods_for_indicator, rotation=45, ha='right')
        
        # Add a legend in the top plot
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.98), 
                  fancybox=True, shadow=True, ncol=len(methods_for_indicator))
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.95)
        plt.savefig(f'{output_dir}/effect_by_indicator_all_methods.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # 3. Create a table of results
    results_table = []
    
    # Get all unique methods across all indicators and policy years
    all_methods = set()
    for indicator in indicators_to_plot:
        for policy_year in benchmark_results[indicator].keys():
            all_methods.update(benchmark_results[indicator][policy_year].keys())
    
    for indicator in indicators_to_plot:
        for policy_year in benchmark_results[indicator].keys():
            policy_data = benchmark_results[indicator][policy_year]
            
            row = {
                'Indicator': indicator,
                'Policy_Year': policy_year
            }
            
            # Add data for each method
            for method in all_methods:
                if method in policy_data:  # Check if method exists in policy_data
                    if isinstance(policy_data[method], dict):
                        row[f'{method}_Effect'] = policy_data[method].get('relative_effect', np.nan)
                        row[f'{method}_Lower'] = policy_data[method].get('lower_bound', np.nan)
                        row[f'{method}_Upper'] = policy_data[method].get('upper_bound', np.nan)
                        row[f'{method}_Significant'] = policy_data[method].get('significance', False)
                    else:
                        row[f'{method}_Effect'] = policy_data[method]
                        row[f'{method}_Lower'] = np.nan
                        row[f'{method}_Upper'] = np.nan
                        row[f'{method}_Significant'] = False
                else:
                    # Handle missing method
                    row[f'{method}_Effect'] = np.nan
                    row[f'{method}_Lower'] = np.nan
                    row[f'{method}_Upper'] = np.nan
                    row[f'{method}_Significant'] = False
            
            results_table.append(row)
    
    # Convert to DataFrame and save as CSV
    results_df = pd.DataFrame(results_table)
    results_df.to_csv(f'{output_dir}/all_methods_results.csv', index=False)
    
    return {
        'results_df': results_df
    }

def visualize_calibration_comparison(calibration_results, output_dir='outputs/figures/'):
    """
    Create visualizations comparing calibration metrics across all methods.
    
    Args:
        calibration_results: Dictionary containing calibration results
        output_dir: Directory to save output figures
    """
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Method colors
    method_colors = {
        'BayesianCausal': '#0072B2',  # Blue
        'ITS': '#D55E00',             # Red-orange
        'DiD': '#CC79A7',             # Pink
        'SCM': '#009E73',             # Green
        'ASCM': '#56B4E9',            # Light blue
        'CausalImpact': '#17becf',    # Cyan
        'Granger': '#bcbd22',         # Olive
        'DoubleML': '#7f7f7f',        # Gray
        'meta_dml': '#e377c2',        # Pink
        'CausalForests': '#9467bd',   # Purple
        'BART': '#d62728',           # Red
        'PSM': '#ff7f0e'             # Orange
    }
    
    # Extract methods and metrics
    methods = list(calibration_results.keys())
    plausibility_rates = [
        calibration_results[m]['plausibility_rate'] * 100
        if 'plausibility_rate' in calibration_results[m] else np.nan
        for m in methods
    ]
    mean_abs_effects = [
        calibration_results[m]['mean_abs_effect']
        if 'mean_abs_effect' in calibration_results[m] else np.nan
        for m in methods
    ]
    effect_variances = [
        calibration_results[m]['effect_variance']
        if 'effect_variance' in calibration_results[m] else np.nan
        for m in methods
    ]
    avg_violations = [
        calibration_results[m].get('avg_violation', np.nan)
        for m in methods
    ]
    
    # Create colors list
    colors = [method_colors.get(m, 'gray') for m in methods]
    
    # 1. Create plausibility rate bar chart
    plt.figure(figsize=(10, 6))
    bars = plt.bar(methods, plausibility_rates, color=colors, alpha=0.7)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 2,
                f'{height:.1f}%', ha='center', va='bottom')
    
    plt.ylabel('Percentage of Plausible Estimates (%)')
    plt.title('Plausibility Rate by Method (higher is better)')
    plt.xticks(rotation=45, ha='right')
    plt.ylim(0, 110)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/plausibility_rate_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Create effect variance comparison (log scale)
    plt.figure(figsize=(10, 6))
    bars = plt.bar(methods, effect_variances, color=colors, alpha=0.7)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height * 1.1,
                f'{height:.1f}', ha='center', va='bottom')
    
    plt.ylabel('Effect Variance')
    plt.title('Effect Estimate Variance by Method (lower is better)')
    plt.yscale('log')  # Log scale to show large differences
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/effect_variance_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Create constraint violation comparison
    plt.figure(figsize=(10, 6))
    bars = plt.bar(methods, avg_violations, color=colors, alpha=0.7)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{height:.1f}%', ha='center', va='bottom')
    
    plt.ylabel('Average Violation Magnitude (%)')
    plt.title('Average Constraint Violation by Method (lower is better)')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/constraint_violation_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Create radar chart comparing all dimensions
    # Define categories
    categories = ['Plausibility', 'Effect Stability', 'Constraint Adherence']
    
    # Normalize metrics for radar chart (higher is better for all dimensions)
    max_variance = max(effect_variances)
    max_violation = max(avg_violations) if max(avg_violations) > 0 else 1
    
    radar_data = {}
    for i, method in enumerate(methods):
        # Convert each metric to [0, 1] range where 1 is best
        plausibility = plausibility_rates[i] / 100
        stability = 1 - (effect_variances[i] / max_variance)
        adherence = 1 - (avg_violations[i] / max_violation)
        
        radar_data[method] = [plausibility, stability, adherence]
    
    # Create radar chart
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, polar=True)
    
    # Number of variables
    N = len(categories)
    
    # What will be the angle of each axis in the plot
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Close the loop
    
    # Draw one axis per variable and add labels
    plt.xticks(angles[:-1], categories, size=12)
    
    # Draw ylabels
    ax.set_rlabel_position(0)
    plt.yticks([0.25, 0.5, 0.75], ["0.25", "0.5", "0.75"], color="grey", size=10)
    plt.ylim(0, 1)
    
    # Plot each method
    for method in methods:
        values = radar_data[method]
        values += values[:1]  # Close the loop
        
        ax.plot(angles, values, linewidth=2, linestyle='solid', 
              label=method, color=method_colors.get(method, 'gray'))
        ax.fill(angles, values, alpha=0.1, color=method_colors.get(method, 'gray'))
    
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    plt.title('Multi-dimensional Performance Comparison', size=15)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/method_radar_chart.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return radar_data