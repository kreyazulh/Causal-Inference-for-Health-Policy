import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Create output directory
import os
os.makedirs('outputs/pocme_multi_country_analysis', exist_ok=True)

# Policy timelines for each country
POLICY_TIMELINES = {
    'bangladesh': {
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
    },
    'philippines': {
        '1972': 'Implementation of Philippine Medical Care Act (Medicare)',
        '1976': 'Compulsory Basic Immunization Program (PD 996)',
        '1978': 'Revised Medicare Act with New Society policies (PD 1519)',
        '1980': 'Primary Health Care approach adoption post-Alma Ata',
        '1988': 'Generics Act implementation',
        '1991': 'Local Government Code - Health service devolution',
        '1993': 'Doctors to the Barrios Initiative launch',
        '1995': 'National Health Insurance Act - PhilHealth creation',
        '1999': 'Health Sector Reform Agenda (HSRA) 1999-2004',
        '2005': 'FOURmula One for Health strategy 2005-2010',
        '2008': 'Cheaper Medicines Act and health financing reforms',
        '2010': 'Aquino Health Agenda - Universal Health Care focus',
        '2012': 'Sin Tax Reform Law for health financing',
        '2016': 'Philippine Health Agenda 2016-2022',
        '2017': 'FOURmula One Plus for Health (F1+) 2017-2022',
        '2019': 'Universal Health Care Act - comprehensive reform',
        '2021': 'Post-pandemic health system strengthening'
    },
    'zimbabwe': {
        '1980': 'Independence and Primary Health Care adoption',
        '1982': 'Rural Health Centers expansion program',
        '1988': 'Essential Drug List implementation',
        '1990': 'Economic Structural Adjustment Program health impacts',
        '1996': 'Health Services Fund introduction with user fees',
        '1997': 'National Health Strategy 1997-2007',
        '2000': 'Land reform and donor relations crisis',
        '2003': 'National AIDS Trust Fund establishment',
        '2008': 'Health system collapse and dollarization',
        '2009': 'National Health Strategy 2009-2013 - recovery focus',
        '2013': 'Health Development Fund with international partners',
        '2016': 'National Health Strategy 2016-2020 "Leaving No One Behind"',
        '2018': 'Post-Mugabe health sector recovery initiatives',
        '2021': 'National Health Strategy 2021-2025',
        '2023': 'Health Resilience Fund launch for UHC'
    }
}

def calculate_research_based_significance_balanced(method_data):
    """Balanced research-based significance evaluation to avoid bias"""
    
    effects = method_data['Effect'].values
    abs_effects = np.abs(effects)
    
    # More conservative thresholds based on health economics literature
    trivial_threshold = 3.0      # <3% = trivial (was 2%)
    small_threshold = 7.0        # 3-7% = small (was 5%)
    moderate_threshold = 20.0    # 7-20% = moderate (was 15%)
    
    # Effect size distribution
    trivial_effects = (abs_effects < trivial_threshold).mean()
    small_effects = ((abs_effects >= trivial_threshold) & 
                    (abs_effects < small_threshold)).mean()
    moderate_effects = ((abs_effects >= small_threshold) & 
                       (abs_effects < moderate_threshold)).mean()
    large_effects = (abs_effects >= moderate_threshold).mean()
    
    # More robust statistical significance assessment
    effect_variance = method_data['Effect'].var()
    n_obs = len(method_data)
    
    # Use bootstrap for p-values instead of assuming normality
    if effect_variance > 0 and n_obs > 10:
        # Bootstrap standard errors
        bootstrap_ses = []
        for _ in range(100):
            boot_sample = np.random.choice(effects, size=n_obs, replace=True)
            bootstrap_ses.append(np.std(boot_sample) / np.sqrt(n_obs))
        
        effect_se = np.median(bootstrap_ses)  # More robust than parametric
        t_stats = effects / (effect_se + 1e-8)
        
        # Use t-distribution for small samples
        from scipy import stats
        df = max(1, n_obs - 1)
        p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), df))
    else:
        p_values = np.ones(len(effects))
    
    # More stringent significance threshold for multiple testing
    alpha_corrected = 0.05 / np.sqrt(n_obs)  # Bonferroni-style correction
    
    # Power analysis with correction for optimism
    meaningful_mask = abs_effects >= small_threshold
    if meaningful_mask.sum() > 0:
        # Use corrected alpha for power calculation
        power_proxy = (p_values[meaningful_mask] < alpha_corrected).mean()
    else:
        power_proxy = 0.0
    
    # Type I error with stricter definition
    trivial_mask = abs_effects < trivial_threshold
    if trivial_mask.sum() > 0:
        type1_proxy = (p_values[trivial_mask] < 0.05).mean()
    else:
        type1_proxy = 0.0
    
    # Enhanced FDR estimation with Storey's q-value method
    sorted_p = np.sort(p_values)
    n_tests = len(p_values)
    
    # Estimate proportion of true nulls (π₀)
    lambda_param = 0.5
    pi0 = min(1, np.mean(p_values > lambda_param) / (1 - lambda_param))
    
    # Calculate q-values
    significant_bh = 0
    for i in range(n_tests - 1, -1, -1):
        if sorted_p[i] <= (i + 1) / n_tests * 0.05 / pi0:
            significant_bh = i + 1
            break
    
    n_significant = np.sum(p_values < 0.05)
    estimated_fdr = pi0 * 0.05 * n_tests / max(1, n_significant)
    
    # Penalize methods that find too many significant effects
    significance_rate = n_significant / n_tests
    over_discovery_penalty = max(0, significance_rate - 0.5) * 2  # Penalty if >50% significant
    
    # Conservative vs Liberal assessment with penalties
    tiny_but_significant = ((abs_effects < trivial_threshold) & 
                          (p_values < 0.05)).mean()
    moderate_and_significant = ((abs_effects >= small_threshold) & 
                              (p_values < alpha_corrected)).mean()  # Use corrected alpha
    
    # Effect realism with stricter bounds
    extreme_effects = (abs_effects > 40).mean()  # Lowered from 50%
    implausible_effects = (abs_effects > 60).mean()  # Additional penalty
    near_zero_effects = (abs_effects < 0.5).mean()  # More strict
    
    # Effect consistency check
    effect_cv = np.std(effects) / (np.mean(np.abs(effects)) + 1e-8)  # Coefficient of variation
    consistency_penalty = min(1, effect_cv / 2)  # Penalty for inconsistent effects
    
    # BALANCED SCORING SYSTEM
    
    # Power score - more conservative
    if power_proxy >= 0.7:  # Lowered from 0.8
        power_score = 90  # Max 90 instead of 100
    elif power_proxy >= 0.5:
        power_score = 70
    elif power_proxy >= 0.3:
        power_score = 50
    elif power_proxy >= 0.1:
        power_score = 30
    else:
        power_score = 10
    
    # Apply over-discovery penalty to power score
    power_score *= (1 - over_discovery_penalty * 0.5)
    
    # FDR score - stricter thresholds
    if estimated_fdr <= 0.01:  # Very low FDR
        fdr_score = 100
    elif estimated_fdr <= 0.05:
        fdr_score = 85
    elif estimated_fdr <= 0.10:
        fdr_score = 70
    elif estimated_fdr <= 0.20:
        fdr_score = 50
    else:
        fdr_score = 20
    
    # Realism score - harsher penalties
    realism_score = 100
    realism_score -= extreme_effects * 150  # Increased from 200
    realism_score -= implausible_effects * 300  # Additional penalty
    realism_score -= near_zero_effects * 75  # Increased from 50
    realism_score *= (1 - consistency_penalty * 0.3)  # Consistency adjustment
    realism_score = max(0, realism_score)
    
    # Balance score - rewards conservative findings
    if tiny_but_significant <= 0.05 and moderate_and_significant >= 0.3:
        balance_score = 100
    elif tiny_but_significant <= 0.1 and moderate_and_significant >= 0.2:
        balance_score = 80
    elif tiny_but_significant <= 0.2 and moderate_and_significant >= 0.1:
        balance_score = 60
    else:
        balance_score = 40
    
    # Additional penalties for imbalanced findings
    if moderate_and_significant < 0.1:  # Too few meaningful findings
        balance_score *= 0.5
    if tiny_but_significant > 0.2:  # Too many trivial findings
        balance_score *= 0.6
    if significance_rate > 0.7:  # Too many overall significant findings
        balance_score *= 0.7
    
    # ADJUSTED WEIGHTS - Less emphasis on power, more on realism
    significance_score = (
        power_score * 0.25 +      # Reduced from 0.35
        fdr_score * 0.25 +        # Same
        realism_score * 0.30 +    # Increased from 0.20
        balance_score * 0.20      # Same
    )
    
    # Final adjustment for methods that seem too good to be true
    if power_proxy > 0.8 and estimated_fdr < 0.05 and moderate_and_significant > 0.5:
        # This combination is suspiciously good
        significance_score *= 0.85  # 15% penalty
    
    return {
        'significance_score': significance_score,
        'power_proxy': power_proxy,
        'estimated_fdr': estimated_fdr,
        'type1_proxy': type1_proxy,
        'power_score': power_score,
        'fdr_score': fdr_score,
        'realism_score': realism_score,
        'balance_score': balance_score,
        'trivial_effects_pct': trivial_effects * 100,
        'small_effects_pct': small_effects * 100,
        'moderate_effects_pct': moderate_effects * 100,
        'large_effects_pct': large_effects * 100,
        'tiny_but_significant_pct': tiny_but_significant * 100,
        'moderate_and_significant_pct': moderate_and_significant * 100,
        'mean_abs_effect': np.mean(abs_effects),
        'effect_cv': effect_cv,
        'over_discovery_rate': significance_rate,
        'consistency_penalty': consistency_penalty
    }


def calculate_pocme_v2_1_scores(data):
    """POCME v2.1 with balanced scoring to avoid Meta-DML bias"""
    
    # Domain constraints (same as before)
    domain_constraints = {
        'Mortality rate, infant (per 1,000 live births)': (-80, 10),
        'Life expectancy at birth, total (years)': (-10, 15),
        'Maternal mortality ratio (modeled estimate, per 100,000 live births)': (-80, 10),
        'Immunization, measles (% of children ages 12-23 months)': (-20, 80),
        'Prevalence of undernourishment (% of population)': (-70, 15),
        'Mortality rate, under-5 (per 1,000 live births)': (-80, 10), 
        'Incidence of tuberculosis (per 100,000 people)': (-80, 20),
        'Hospital beds (per 1,000 people)': (-50, 200)
    }
    
    # Updated computation times (Meta-DML is actually slower)
    default_times = {
        'ITS': 0.001, 'DiD': 0.001, 'SCM': 0.01, 'ASCM': 0.01,
        'CausalImpact': 0.1, 'BayesianCausal': 25.0, 'CausalForests': 0.5,
        'BART': 0.1, 'PSM': 0.1, 'DoubleML': 1.0, 
        'meta_dml': 7.0,  
        'Meta-DML': 7.0   # Both naming conventions
    }
    
    all_combinations = data[['Indicator', 'Policy_Year']].drop_duplicates()
    total_possible = len(all_combinations)
    
    method_scores = {}
    
    for method in data['Method'].unique():
        method_data = data[data['Method'] == method].copy()
        
        if len(method_data) == 0:
            continue
        
        # Domain adherence (same logic but with penalty for near-boundary values)
        plausible_count = 0
        total_count = 0
        violation_magnitudes = []
        near_boundary_count = 0
        
        for _, row in method_data.iterrows():
            if row['Indicator'] in domain_constraints:
                min_val, max_val = domain_constraints[row['Indicator']]
                total_count += 1
                
                # Check if within bounds
                if min_val <= row['Effect'] <= max_val:
                    plausible_count += 1
                    
                    # Check if suspiciously close to boundaries
                    boundary_buffer = 0.1 * (max_val - min_val)
                    if (row['Effect'] - min_val < boundary_buffer or 
                        max_val - row['Effect'] < boundary_buffer):
                        near_boundary_count += 1
                else:
                    if row['Effect'] < min_val:
                        violation_magnitudes.append(min_val - row['Effect'])
                    else:
                        violation_magnitudes.append(row['Effect'] - max_val)
        
        plausibility_rate = plausible_count / max(total_count, 1)
        boundary_rate = near_boundary_count / max(total_count, 1)
        
        # Domain score with boundary penalty
        domain_score = min(plausibility_rate * 100, 100)
        if boundary_rate > 0.3:  # Too many near-boundary values
            domain_score *= 0.9
        
        # Use balanced significance calculation
        sig_metrics = calculate_research_based_significance_balanced(method_data)
        significance_score = sig_metrics['significance_score']
        
        # Precision & reliability with outlier detection
        effect_variance = method_data['Effect'].var()
        effect_iqr = method_data['Effect'].quantile(0.75) - method_data['Effect'].quantile(0.25)
        
        # Robust variance measure
        if effect_iqr > 0:
            robust_variance = (effect_iqr / 1.349) ** 2  # IQR to variance conversion
            variance_ratio = effect_variance / (robust_variance + 1e-8)
            
            # Penalty for high variance ratio (indicates outliers)
            outlier_penalty = min(1, variance_ratio / 10)
        else:
            outlier_penalty = 0
        
        # Precision scoring with outlier adjustment
        if effect_variance == 0:
            precision_score = 90  # Not 100 - zero variance is suspicious
        elif effect_variance < 100:
            precision_score = 85
        elif effect_variance < 1000:
            precision_score = 65
        elif effect_variance < 10000:
            precision_score = 45
        else:
            precision_score = 20
        
        precision_score *= (1 - outlier_penalty * 0.2)
        
        # Completeness & coverage (similar but with consistency check)
        method_combinations = method_data[['Indicator', 'Policy_Year']].drop_duplicates()
        raw_completeness_rate = len(method_combinations) / total_possible
        missing_scenarios = total_possible - len(method_combinations)
        
        # Check for cherry-picking pattern
        indicators_per_year = method_data.groupby('Policy_Year')['Indicator'].nunique()
        years_per_indicator = method_data.groupby('Indicator')['Policy_Year'].nunique()
        
        coverage_consistency = np.std(indicators_per_year) / (np.mean(indicators_per_year) + 1e-8)
        if coverage_consistency > 1:  # High variation suggests cherry-picking
            cherry_pick_penalty = min(0.3, coverage_consistency / 5)
        else:
            cherry_pick_penalty = 0
        
        # Completeness scoring
        if raw_completeness_rate == 1.0:
            completeness_score = 95  # Not 100 - perfect completion is rare
        elif raw_completeness_rate >= 0.9:
            completeness_score = 90
        elif raw_completeness_rate >= 0.75:
            completeness_score = 75
        elif raw_completeness_rate >= 0.5:
            completeness_score = 55
        else:
            completeness_score = max(20, 40 - (0.5 - raw_completeness_rate) * 40)
        
        completeness_score *= (1 - cherry_pick_penalty)
        
        # Coverage score calculation
        effect_median = method_data['Effect'].median()
        if effect_iqr > 0:
            outlier_threshold = 3 * effect_iqr
            outliers = method_data[
                (method_data['Effect'] > effect_median + outlier_threshold) |
                (method_data['Effect'] < effect_median - outlier_threshold)
            ]
            outlier_rate = len(outliers) / len(method_data)
            positivity_score = max(0, 100 - outlier_rate * 200)
        else:
            positivity_score = 80
        
        coverage_score = (completeness_score * 0.7 + positivity_score * 0.3)
        
        # Implementation safety (stricter evaluation)
        avg_violation = np.mean(violation_magnitudes) if violation_magnitudes else 0
        max_violation = max(violation_magnitudes) if violation_magnitudes else 0
        
        if avg_violation == 0:
            safety_score = 95  # Not 100
        elif avg_violation < 25:
            safety_score = 85
        elif avg_violation < 100:
            safety_score = 70
        elif avg_violation < 500:
            safety_score = 50
        else:
            safety_score = 20
        
        # Additional penalty for extreme violations
        if max_violation > 1000:
            safety_score *= 0.7
        
        # Computational efficiency (adjusted for Meta-DML)
        computation_time = default_times.get(method, 1.0)
        
        # More realistic efficiency scoring
        if computation_time < 0.01:
            efficiency_score = 95
        elif computation_time < 0.1:
            efficiency_score = 85
        elif computation_time < 1.0:
            efficiency_score = 75
        elif computation_time < 5.0:
            efficiency_score = 65
        elif computation_time < 15.0:
            efficiency_score = 55
        else:
            efficiency_score = 40
        
        # BALANCED POCME v2.1 score with adjusted weights
        pocme_score = (
            domain_score * 0.25 +          # Reduced from 0.25
            significance_score * 0.20 +     # Increased from 0.20
            precision_score * 0.20 +        # Increased from 0.15
            coverage_score * 0.15 +         # Same
            safety_score * 0.15 +           # Reduced from 0.20
            efficiency_score * 0.05         # Same
        )
        
        # Final adjustment for suspiciously high scores
        if pocme_score > 85:
            # Apply small penalty to very high scores for fairness
            pocme_score = 85 + (pocme_score - 85) * 0.7
        
        method_scores[method] = {
            'pocme_v2_1_score': pocme_score,
            'domain_adherence': domain_score,
            'significance_quality': significance_score,
            'precision_reliability': precision_score,
            'completeness_coverage': coverage_score,
            'implementation_safety': safety_score,
            'computational_efficiency': efficiency_score,
            'power_proxy': sig_metrics['power_proxy'],
            'estimated_fdr': sig_metrics['estimated_fdr'],
            'raw_completeness_rate': raw_completeness_rate,
            'missing_scenarios': missing_scenarios,
            'n_estimates': len(method_data),
            'boundary_rate': boundary_rate,
            'outlier_penalty': outlier_penalty,
            'cherry_pick_penalty': cherry_pick_penalty,
            'over_discovery_rate': sig_metrics['over_discovery_rate']
        }
    
    return method_scores

def display_country_results(pocme_results, country_name):
    """Display POCME v2.1 results for a specific country"""
    
    print(f"\nPOCME v2.1 RESULTS FOR {country_name.upper()}:")
    print("=" * 140)
    
    # Convert to DataFrame and sort
    pocme_df = pd.DataFrame(pocme_results).T
    pocme_df = pocme_df.sort_values('pocme_v2_1_score', ascending=False)
    
    print(f"{'Rank':<4} {'Method':<15} {'POCME':<6} {'Domain':<7} {'SigQual':<8} {'Precision':<9} {'Coverage':<8} {'Safety':<7} {'Eff':<4} {'Power':<6} {'FDR':<5} {'Complete':<8}")
    print("─" * 140)
    
    for rank, (method, row) in enumerate(pocme_df.iterrows(), 1):
        emoji = "1st" if rank == 1 else "2nd" if rank == 2 else "3rd" if rank == 3 else f"{rank:2d}."
        
        score = row['pocme_v2_1_score']
        power = row['power_proxy']
        fdr = row['estimated_fdr']
        
        complete_pct = f"{row['raw_completeness_rate']*100:.0f}%"
        
        print(f"{emoji:<4} {method:<15} {row['pocme_v2_1_score']:5.1f} "
              f"{row['domain_adherence']:6.1f} {row['significance_quality']:7.1f} "
              f"{row['precision_reliability']:8.1f} {row['completeness_coverage']:7.1f} "
              f"{row['implementation_safety']:6.1f} {row['computational_efficiency']:3.1f} "
              f"{row['power_proxy']:5.2f} {row['estimated_fdr']:4.2f} {complete_pct:>7}")
    
    return pocme_df



def create_multi_country_diagram(country_results, output_dir):
    """Create enhanced stacked bar visualization with CORRECTED policy directionality"""
    
    countries = list(country_results.keys())
    fig, axes = plt.subplots(1, 4, figsize=(40, 16))

    # CRITICAL FIX: Define indicator directionality
    # For these indicators, NEGATIVE effects are GOOD (reductions are beneficial)
    NEGATIVE_GOOD_INDICATORS = {
        'Mortality rate, infant (per 1,000 live births)',
        'Mortality rate, under-5 (per 1,000 live births)', 
        'Maternal mortality ratio (modeled estimate, per 100,000 live births)',
        'Prevalence of undernourishment (% of population)',
        'Incidence of tuberculosis (per 100,000 people)'
    }
    
    # For these indicators, POSITIVE effects are GOOD (increases are beneficial)
    POSITIVE_GOOD_INDICATORS = {
        'Life expectancy at birth, total (years)',
        'Immunization, measles (% of children ages 12-23 months)',
        'Hospital beds (per 1,000 people)'
    }
    
    def calculate_beneficial_effects(effects, indicators):
        """Convert effects to beneficial direction based on indicator type"""
        beneficial_effects = []
        
        for effect, indicator in zip(effects, indicators):
            if indicator in NEGATIVE_GOOD_INDICATORS:
                # For mortality/disease indicators: negative effect = beneficial
                beneficial_effects.append(-effect)  # Flip sign
            elif indicator in POSITIVE_GOOD_INDICATORS:
                # For capacity/prevention indicators: positive effect = beneficial  
                beneficial_effects.append(effect)   # Keep as is
            else:
                # Default: treat positive as good (but warn)
                print(f"⚠️  Unknown indicator directionality: {indicator}")
                beneficial_effects.append(effect)
                
        return np.array(beneficial_effects)
    
    # Color palette for POCME components
    component_colors = {
        'domain_adherence': '#FF6B6B',      # Red (25%)
        'significance_quality': '#4ECDC4',   # Teal (20%)
        'precision_reliability': '#45B7D1',  # Blue (18%)
        'completeness_coverage': '#96CEB4',  # Green (17%)
        'implementation_safety': '#FFEAA7',  # Yellow (15%)
        'computational_efficiency': '#DDA0DD' # Purple (5%)
    }
    
    component_weights = {
        'domain_adherence': 0.25,
        'significance_quality': 0.20,
        'precision_reliability': 0.20,
        'completeness_coverage': 0.15,
        'implementation_safety': 0.15,
        'computational_efficiency': 0.05
    }
    
    component_labels = {
        'domain_adherence': 'Domain (25%)',
        'significance_quality': 'Significance (20%)',
        'precision_reliability': 'Precision (20%)',
        'completeness_coverage': 'Coverage (15%)',
        'implementation_safety': 'Safety (15%)',
        'computational_efficiency': 'Efficiency (5%)'
    }
    
    # POCME visualization (first 3 panels) - keep as before
    for i, (country, pocme_df) in enumerate(country_results.items()):
        if i >= 3:
            break
        ax = axes[i]
        
        # Sort by POCME score (bottom-up: highest at bottom)
        pocme_df = pocme_df.sort_values('pocme_v2_1_score', ascending=True)
        
        methods = pocme_df.index
        n_methods = len(methods)
        y_positions = np.arange(n_methods) * 3.0
        
        # Create stacked bars
        bottom = np.zeros(n_methods)
        
        for component in component_colors.keys():
            weighted_values = pocme_df[component] * component_weights[component]
            bars = ax.barh(y_positions, weighted_values, left=bottom, height=2.0,
                          color=component_colors[component], 
                          alpha=0.8, 
                          label=component_labels[component] if i == 0 else "")
            bottom += weighted_values
        
        # Add POCME scores
        for j, score in enumerate(pocme_df['pocme_v2_1_score']):
            ax.text(score + 4, y_positions[j], f'{score:.1f}', va='center', fontweight='bold', 
                   fontsize=13, color='#2C3E50')
        
        # Method labels with Meta-DML highlighting
        method_labels = []
        method_colors = []
        method_weights = []
        
        for method in methods:
            if method == 'meta_dml' or method == 'Meta-DML':
                method_labels.append('META-DML')
                method_colors.append('#E74C3C')
                method_weights.append('bold')
            else:
                method_labels.append(method)
                method_colors.append('#2C3E50')
                method_weights.append('normal')
        
        ax.set_yticks(y_positions)
        ax.set_yticklabels(method_labels, fontsize=13)
        
        # Apply formatting
        for j, (label, color, weight) in enumerate(zip(method_labels, method_colors, method_weights)):
            tick_label = ax.get_yticklabels()[j]
            tick_label.set_color(color)
            tick_label.set_fontweight(weight)
            if label == 'META-DML':
                tick_label.set_fontsize(16)
                tick_label.set_bbox(dict(boxstyle="round,pad=0.3", facecolor='yellow', alpha=0.3, edgecolor='red'))
        
        # Meta-DML border
        if 'meta_dml' in methods or 'Meta-DML' in methods:
            meta_idx = None
            for idx, method in enumerate(methods):
                if method == 'meta_dml' or method == 'Meta-DML':
                    meta_idx = idx
                    break
            
            if meta_idx is not None:
                meta_y = y_positions[meta_idx]
                total_width = pocme_df.iloc[meta_idx]['pocme_v2_1_score']
                ax.add_patch(plt.Rectangle((0, meta_y - 1), total_width, 2, 
                                         fill=False, edgecolor='red', linewidth=4, alpha=0.8))
        
        ax.set_title(f'{country.title()}', fontsize=18, fontweight='bold', pad=40)
        ax.set_xlim(0, 115)
        ax.set_ylim(-2, y_positions[-1] + 2)
        ax.grid(True, alpha=0.3, axis='x')
        
        # Reference lines
        ax.axvline(x=60, color='#FF8C00', linestyle='--', alpha=0.9, linewidth=3, 
                  label='Threshold (60)' if i == 0 else "")
        ax.axvline(x=75, color='#228B22', linestyle='--', alpha=0.9, linewidth=3, 
                  label='Good (75)' if i == 0 else "")
    
    # Fourth panel: CORRECTED Policy Insights
    policy_ax = axes[3]
    policy_ax.set_xlim(0, 1)
    policy_ax.set_ylim(0, 1)
    policy_ax.axis('off')
    
    # Get CORRECTED successful policy data
    successful_policies = {}
    country_colors = {
        'bangladesh': '#2E4057',
        'philippines': '#048A81', 
        'zimbabwe': '#C4302B'
    }
    
    for country in ['bangladesh', 'philippines', 'zimbabwe']:
        try:
            data_path = f'outputs/enhanced_benchmark_{country}/all_methods_results.csv'
            results = pd.read_csv(data_path)
            meta_dml_data = results[(results['Method'] == 'Meta-DML') | (results['Method'] == 'meta_dml')].copy()
            
            if len(meta_dml_data) == 0:
                continue
            
            policy_performance = {}
            for policy_year in meta_dml_data['Policy_Year'].unique():
                year_data = meta_dml_data[meta_dml_data['Policy_Year'] == policy_year]
                
                # CORRECTED: Apply directional transformation
                raw_effects = year_data['Effect'].values
                indicators = year_data['Indicator'].values
                
                # Transform effects to beneficial direction
                beneficial_effects = calculate_beneficial_effects(raw_effects, indicators)
                
                # Now calculate metrics on beneficial effects
                avg_beneficial_effect = np.mean(beneficial_effects)
                beneficial_magnitude = np.mean(np.abs(beneficial_effects))
                n_beneficial_effects = np.sum(beneficial_effects > 0)  # Now correctly counts beneficial effects
                consistency_score = (n_beneficial_effects / len(beneficial_effects)) * 100
                
                # Policy impact score using corrected metrics
                policy_score = (
                    beneficial_magnitude * 0.4 +  # Magnitude of beneficial effects
                    consistency_score * 0.3 +     # % of truly beneficial effects
                    (len(beneficial_effects) / 10) * 0.3  # Coverage
                )
                
                policy_performance[policy_year] = {
                    'policy_score': policy_score,
                    'avg_beneficial_effect': avg_beneficial_effect,
                    'beneficial_magnitude': beneficial_magnitude,
                    'consistency': consistency_score,
                    'n_indicators': len(year_data)
                }
            
            # Sort by corrected policy score
            top_policies = sorted(policy_performance.items(), 
                                key=lambda x: x[1]['policy_score'], reverse=True)[:3]
            successful_policies[country] = top_policies
            
        except Exception as e:
            print(f"Error processing {country}: {e}")
            continue
    
    # Display corrected policy insights (rest of visualization code stays the same)
    policy_ax.text(0.5, 1.05, 'HIGH-IMPACT POLICIES', 
                   transform=policy_ax.transAxes, ha='center', va='center',
                   fontsize=18, fontweight='bold', color='#1a1a1a')
    
    policy_ax.text(0.5, 0.98, 'Meta-DML (Ours) Evidence-Based Rankings', 
                   transform=policy_ax.transAxes, ha='center', va='top',
                   fontsize=14, style='italic', color='#333333')
    
    # Display corrected policies
    y_start = 0.92
    country_names = {'bangladesh': 'BANGLADESH', 'philippines': 'PHILIPPINES', 'zimbabwe': 'ZIMBABWE'}
    
    for country_idx, (country, policies) in enumerate(successful_policies.items()):
        y_country = y_start - (country_idx * 0.31)
        
        # Country header
        policy_ax.add_patch(plt.Rectangle((0.05, y_country - 0.03), 0.9, 0.05,
                                        transform=policy_ax.transAxes,
                                        facecolor=country_colors[country], alpha=0.9))
        
        policy_ax.text(0.5, y_country - 0.005, country_names[country],
                      transform=policy_ax.transAxes, ha='center', va='center',
                      fontsize=12, fontweight='bold', color='white')
        
        # Top 3 corrected policies
        for rank, (year, metrics) in enumerate(policies):
            y_policy = y_country - 0.08 - (rank * 0.075)
            
            # Rank circle
            circle = plt.Circle((0.08, y_policy), 0.015,
                              transform=policy_ax.transAxes,
                              facecolor=country_colors[country], alpha=0.8)
            policy_ax.add_patch(circle)
            
            policy_ax.text(0.08, y_policy, str(rank + 1),
                          transform=policy_ax.transAxes, ha='center', va='center',
                          fontsize=9, fontweight='bold', color='white')
            
            # Policy info
            policy_name = POLICY_TIMELINES.get(country, {}).get(str(year), f'Policy {year}')
            
            policy_ax.text(0.12, y_policy + 0.015, f'{year}:',
                          transform=policy_ax.transAxes, ha='left', va='center',
                          fontsize=10, fontweight='bold', color=country_colors[country])
            
            policy_ax.text(0.12, y_policy - 0.005, policy_name,
                          transform=policy_ax.transAxes, ha='left', va='center',
                          fontsize=9, color='#2C3E50')
            
            # CORRECTED metrics display
            policy_ax.text(0.12, y_policy - 0.025, 
                          f'Impact: {metrics["policy_score"]:.1f} • Beneficial Effect: {metrics["avg_beneficial_effect"]:+.1f}% • Consistency: {metrics["consistency"]:.0f}%',
                          transform=policy_ax.transAxes, ha='left', va='center',
                          fontsize=8, color='#666666')
    
    # Add corrected methodology note
    policy_ax.text(0.5, 0.005, 
                  'Accounts for indicator directionality\n(mortality reductions = beneficial, capacity increases = beneficial)',
                  transform=policy_ax.transAxes, ha='center', va='bottom',
                  fontsize=9, style='italic', color='#666666')
    
    # Legend (same as before)
    legend_elements = []
    for component, color in component_colors.items():
        legend_elements.append(plt.Rectangle((0,0),1,1, facecolor=color, alpha=0.8, 
                                           label=component_labels[component]))
    
    legend_elements.append(plt.Line2D([0], [0], color='#FF8C00', linestyle='--', linewidth=3, 
                                    label='Passing Threshold (60)'))
    legend_elements.append(plt.Line2D([0], [0], color='#228B22', linestyle='--', linewidth=3, 
                                    label='Acceptable Threshold (75)'))
    legend_elements.append(plt.Line2D([0], [0], color='#E74C3C', linewidth=0, 
                                    marker='s', markersize=10, label='Meta-DML Highlight'))
    
    fig.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, 0.06), 
              ncol=4, fontsize=13, frameon=True, fancybox=True, shadow=True)
    
    
    plt.subplots_adjust(left=0.01, right=0.99, bottom=0.14, top=0.74, wspace=0.15)
    plt.savefig(f'{output_dir}/corrected_multi_country_pocme_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return fig

def analyze_best_policies_by_meta_dml(country_results):
    """Analyze the best policies according to Meta-DML results for strategic insights"""
    
    print("\n" + "="*80)
    print("BEST POLICIES ACCORDING TO META-DML ANALYSIS")
    print("Strategic Insights for Policy Replication")
    print("="*80)
    
    countries = ['bangladesh', 'philippines', 'zimbabwe']
    all_policy_insights = {}
    
    for country in countries:
        try:
            # Read country-specific data
            data_path = f'outputs/enhanced_benchmark_{country}/all_methods_results.csv'
            results = pd.read_csv(data_path)
            
            # Filter for Meta-DML results only
            meta_dml_data = results[(results['Method'] == 'Meta-DML') | (results['Method'] == 'meta_dml')].copy()
            
            if len(meta_dml_data) == 0:
                print(f"\n{country.upper()}: No Meta-DML results found")
                continue
            
            print(f"\n{country.upper()} - META-DML POLICY ANALYSIS")
            print("-" * 60)
            
            # Analyze policies by year and indicator
            policy_performance = {}
            
            for policy_year in meta_dml_data['Policy_Year'].unique():
                year_data = meta_dml_data[meta_dml_data['Policy_Year'] == policy_year]
                
                # Calculate policy effectiveness metrics
                effects = year_data['Effect'].values
                significant_effects = year_data[year_data['Significant'] == True]['Effect'].values if 'Significant' in year_data.columns else effects
                
                avg_effect = np.mean(effects)
                significant_avg = np.mean(significant_effects) if len(significant_effects) > 0 else 0
                effect_magnitude = np.mean(np.abs(effects))
                n_positive_effects = np.sum(effects > 0)
                n_significant = len(significant_effects)
                
                # Policy impact score (combination of effect size, significance, and consistency)
                consistency_score = (n_positive_effects / len(effects)) * 100
                significance_rate = (n_significant / len(effects)) * 100 if len(effects) > 0 else 0
                
                policy_score = (
                    effect_magnitude * 0.4 +  # Effect magnitude
                    consistency_score * 0.3 +  # Consistency (% positive)
                    significance_rate * 0.3     # Significance rate
                )
                
                policy_performance[policy_year] = {
                    'policy_score': policy_score,
                    'avg_effect': avg_effect,
                    'significant_avg': significant_avg,
                    'effect_magnitude': effect_magnitude,
                    'consistency': consistency_score,
                    'significance_rate': significance_rate,
                    'n_indicators': len(year_data),
                    'best_indicators': year_data.nlargest(3, 'Effect')[['Indicator', 'Effect']].to_dict('records'),
                    'worst_indicators': year_data.nsmallest(2, 'Effect')[['Indicator', 'Effect']].to_dict('records')
                }
            
            # Sort policies by effectiveness
            top_policies = sorted(policy_performance.items(), key=lambda x: x[1]['policy_score'], reverse=True)
            
            print(f"Policy Effectiveness Ranking (Meta-DML Analysis):")
            print(f"{'Rank':<4} {'Year':<6} {'Score':<8} {'Avg Effect':<12} {'Magnitude':<10} {'Consistency':<12} {'Sig Rate':<10} {'Indicators':<10}")
            print("-" * 90)
            
            for rank, (year, metrics) in enumerate(top_policies[:5], 1):
                rank_str = "1st" if rank == 1 else "2nd" if rank == 2 else "3rd" if rank == 3 else f"{rank}."
                print(f"{rank_str:<4} {year:<6} {metrics['policy_score']:7.1f} "
                      f"{metrics['avg_effect']:+11.1f}% {metrics['effect_magnitude']:9.1f}% "
                      f"{metrics['consistency']:11.1f}% {metrics['significance_rate']:9.1f}% "
                      f"{metrics['n_indicators']:9d}")
            
            # Store for cross-country analysis
            all_policy_insights[country] = {
                'top_policies': top_policies[:3],
                'policy_performance': policy_performance
            }
            
        except FileNotFoundError:
            print(f"\n{country.upper()}: Data file not found")
            continue
        except Exception as e:
            print(f"\n{country.upper()}: Error in analysis - {str(e)}")
            continue
    
    return all_policy_insights

def create_meta_dml_pipeline_diagram(output_dir):
    """Create a concise pipeline diagram of Meta-DML architecture"""
    
    fig, ax = plt.subplots(figsize=(22, 16))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 14)
    ax.axis('off')
    
    # Color scheme
    colors = {
        'input': '#E8F4FD',      # Light blue
        'base': '#FFE6CC',       # Light orange  
        'container': '#FFF2CC',  # Light yellow for containers
        'meta': '#D4EDDA',       # Light green
        'output': '#F8D7DA',     # Light red
        'arrow': '#495057',      # Dark gray
        'feedback': '#DC3545',   # Red for feedback
        'text': '#212529'        # Very dark gray
    }
    
    # Title
    fig.suptitle('Meta-DML: Meta-Learning Enhanced Double Machine Learning Pipeline', 
                 fontsize=22, fontweight='bold', color=colors['text'], y=0.95)
    
    # Step labels
    step_y = 12.5
    ax.text(2, step_y, 'STEP 1', ha='center', va='center', fontsize=14, fontweight='bold', 
           bbox=dict(boxstyle="round,pad=0.3", facecolor='#007BFF', alpha=0.8), color='white')
    ax.text(6, step_y, 'STEP 2', ha='center', va='center', fontsize=14, fontweight='bold',
           bbox=dict(boxstyle="round,pad=0.3", facecolor='#007BFF', alpha=0.8), color='white')
    ax.text(10, step_y, 'STEP 3', ha='center', va='center', fontsize=14, fontweight='bold',
           bbox=dict(boxstyle="round,pad=0.3", facecolor='#007BFF', alpha=0.8), color='white')
    
    # STEP 1: Input Data
    input_box = plt.Rectangle((0.5, 10), 3, 1.8, facecolor=colors['input'], 
                             edgecolor='black', linewidth=2)
    ax.add_patch(input_box)
    ax.text(2, 10.9, 'INPUT DATA', ha='center', va='center', 
           fontsize=14, fontweight='bold', color=colors['text'])
    ax.text(2, 10.5, '• Features: X', ha='center', va='center', 
           fontsize=11, color=colors['text'])
    ax.text(2, 10.2, '• Outcome: Y', ha='center', va='center', 
           fontsize=11, color=colors['text'])
    ax.text(2, 9.9, '• Treatment: T', ha='center', va='center', 
           fontsize=11, color=colors['text'])
    
    # STEP 2: Base Learners Container
    container_box = plt.Rectangle((4.5, 6.5), 3, 5, facecolor=colors['container'], 
                                 edgecolor='#856404', linewidth=3)
    ax.add_patch(container_box)
    ax.text(6, 11.2, 'BASE LEARNERS ENSEMBLE', ha='center', va='center', 
           fontsize=13, fontweight='bold', color='#856404')
    ax.text(6, 10.8, '(Cross-Validation Training)', ha='center', va='center', 
           fontsize=10, style='italic', color='#856404')
    
    # Individual Base Learners inside container
    base_learners = ['Random Forest', 'Gradient Boost', 'Neural Net', 'ElasticNet', 'Ridge', 'Lasso']
    base_y_positions = [10.2, 9.6, 9.0, 8.4, 7.8, 7.2]
    
    for i, learner in enumerate(base_learners):
        box = plt.Rectangle((4.7, base_y_positions[i] - 0.2), 2.6, 0.4, 
                           facecolor=colors['base'], edgecolor='black', linewidth=1)
        ax.add_patch(box)
        ax.text(6, base_y_positions[i], learner, ha='center', va='center', 
               fontsize=10, fontweight='bold', color=colors['text'])
    
    # STEP 2 Outputs: Performance & Residuals
    # Performance Metrics Box
    perf_box = plt.Rectangle((8.5, 8.5), 2.5, 2.5, facecolor='#FFF3E0', 
                            edgecolor='black', linewidth=2)
    ax.add_patch(perf_box)
    ax.text(9.75, 10.2, 'PERFORMANCE', ha='center', va='center', 
           fontsize=12, fontweight='bold', color=colors['text'])
    ax.text(9.75, 9.9, 'METRICS', ha='center', va='center', 
           fontsize=12, fontweight='bold', color=colors['text'])
    ax.text(9.75, 9.5, 'MSE(Y), MSE(T)', ha='center', va='center', 
           fontsize=10, color=colors['text'])
    ax.text(9.75, 9.2, 'for each learner', ha='center', va='center', 
           fontsize=9, style='italic', color=colors['text'])
    ax.text(9.75, 8.8, '→ α weights', ha='center', va='center', 
           fontsize=9, fontweight='bold', color=colors['feedback'])
    
    # DML Residuals Box  
    residual_box = plt.Rectangle((8.5, 5.5), 2.5, 2.5, facecolor=colors['base'], 
                                edgecolor='black', linewidth=2)
    ax.add_patch(residual_box)
    ax.text(9.75, 7.2, 'DML RESIDUALS', ha='center', va='center', 
           fontsize=12, fontweight='bold', color=colors['text'])
    ax.text(9.75, 6.8, 'Y - Ŷ(X)', ha='center', va='center', 
           fontsize=11, color=colors['text'])
    ax.text(9.75, 6.4, 'T - T̂(X)', ha='center', va='center', 
           fontsize=11, color=colors['text'])
    ax.text(9.75, 6.0, 'for each learner', ha='center', va='center', 
           fontsize=9, style='italic', color=colors['text'])
    
    # STEP 3: Meta-Learning Stage
    meta_container = plt.Rectangle((4.5, 2), 7, 3, facecolor='#E8F5E8', 
                                  edgecolor='#28A745', linewidth=3)
    ax.add_patch(meta_container)
    ax.text(8, 4.6, 'META-LEARNING STAGE', ha='center', va='center', 
           fontsize=14, fontweight='bold', color='#28A745')
    
    # Meta Features Box (inside meta container)
    meta_features_box = plt.Rectangle((1, 2.5), 2.8, 2, facecolor=colors['meta'], 
                                     edgecolor='black', linewidth=1.5)
    ax.add_patch(meta_features_box)
    ax.text(2.4, 3.8, 'META-FEATURES', ha='center', va='center', 
           fontsize=11, fontweight='bold', color=colors['text'])
    ax.text(2.4, 3.4, '• Sample size', ha='center', va='center', 
           fontsize=9, color=colors['text'])
    ax.text(2.4, 3.1, '• Feature dimensions', ha='center', va='center', 
           fontsize=9, color=colors['text'])
    ax.text(2.4, 2.8, '• Outcome variance', ha='center', va='center', 
           fontsize=9, color=colors['text'])
    
    # Meta-Learner Box (inside meta container)
    meta_box = plt.Rectangle((5, 2.5), 3, 2, facecolor=colors['meta'], 
                            edgecolor='black', linewidth=2)
    ax.add_patch(meta_box)
    ax.text(6.5, 3.8, 'META-LEARNER', ha='center', va='center', 
           fontsize=12, fontweight='bold', color=colors['text'])
    ax.text(6.5, 3.4, 'Optimal Weights:', ha='center', va='center', 
           fontsize=10, color=colors['text'])
    ax.text(6.5, 3.1, 'α = [α₁, α₂, ..., α₆]', ha='center', va='center', 
           fontsize=10, fontweight='bold', color=colors['text'])
    ax.text(6.5, 2.8, 'Neural Net or Weighted', ha='center', va='center', 
           fontsize=9, style='italic', color=colors['text'])
    
    # Final Output Box
    output_box = plt.Rectangle((9, 2.5), 2.5, 2, facecolor=colors['output'], 
                              edgecolor='black', linewidth=2)
    ax.add_patch(output_box)
    ax.text(10.25, 3.8, 'TREATMENT', ha='center', va='center', 
           fontsize=11, fontweight='bold', color=colors['text'])
    ax.text(10.25, 3.5, 'EFFECT', ha='center', va='center', 
           fontsize=11, fontweight='bold', color=colors['text'])
    ax.text(10.25, 3.1, 'θ̂ = Σ αᵢ · θ̂ᵢ', ha='center', va='center', 
           fontsize=10, color=colors['text'])
    ax.text(10.25, 2.8, '± Bootstrap CI', ha='center', va='center', 
           fontsize=9, style='italic', color=colors['text'])
    
    # ARROWS - Clear flow between containers
    arrow_props = dict(arrowstyle='->', lw=3, color=colors['arrow'])
    thick_arrow_props = dict(arrowstyle='->', lw=4, color=colors['arrow'])
    
    # Step 1 to Step 2: Input to Base Learners Container
    ax.annotate('', xy=(4.4, 9), xytext=(3.6, 10.5), arrowprops=thick_arrow_props)
    ax.text(3.8, 9.7, 'Train All\nLearners', ha='center', va='center', 
           fontsize=9, fontweight='bold', color=colors['arrow'])
    
    # Step 2 to Performance Metrics
    ax.annotate('', xy=(8.4, 9.5), xytext=(7.6, 9.5), arrowprops=arrow_props)
    ax.text(8, 10, 'Evaluate', ha='center', va='center', 
           fontsize=9, fontweight='bold', color=colors['arrow'])
    
    # Step 2 to Residuals
    ax.annotate('', xy=(8.4, 6.5), xytext=(7.6, 7.5), arrowprops=arrow_props)
    ax.text(8, 7, 'Residualize', ha='center', va='center', 
           fontsize=9, fontweight='bold', color=colors['arrow'])
    
    # Performance to Meta-Learner (RED feedback arrow)
    feedback_props = dict(arrowstyle='->', lw=4, color=colors['feedback'])
    ax.annotate('', xy=(7.5, 3.5), xytext=(9, 8.5), arrowprops=feedback_props)
    ax.text(8.5, 6, 'PERFORMANCE\nFEEDBACK', ha='center', va='center', 
           fontsize=10, fontweight='bold', color=colors['feedback'],
           bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.9))
    
    # Residuals to Meta-Learner
    ax.annotate('', xy=(7.5, 3.2), xytext=(8.5, 6), arrowprops=arrow_props)
    ax.text(7.8, 4.5, 'Residuals', ha='center', va='center', 
           fontsize=9, fontweight='bold', color=colors['arrow'])
    
    # Meta-Features to Meta-Learner
    ax.annotate('', xy=(4.9, 3.5), xytext=(3.9, 3.5), arrowprops=arrow_props)
    
    # Meta-Learner to Output
    ax.annotate('', xy=(8.9, 3.5), xytext=(8.1, 3.5), arrowprops=thick_arrow_props)
    ax.text(8.5, 3.8, 'Final\nEffect', ha='center', va='center', 
           fontsize=9, fontweight='bold', color=colors['arrow'])
    
    # How it works explanation box
    explanation_box = plt.Rectangle((0.5, 13), 13, 0.8, facecolor='#E9ECEF', 
                                   edgecolor='#6C757D', linewidth=2)
    ax.add_patch(explanation_box)
    ax.text(7, 13.6, 'WORKFLOW: Input → All Base Learners (Container) → Performance Metrics + Residuals → Meta-Learner → Final Treatment Effect', 
           ha='center', va='center', fontsize=12, fontweight='bold', color=colors['text'])
    ax.text(7, 13.2, 'Each base learner contributes to BOTH performance evaluation AND residual computation, then meta-learner optimally combines them', 
           ha='center', va='center', fontsize=11, color=colors['text'])
    
    # Key Innovation Box
    innovation_box = plt.Rectangle((0.5, 0.2), 13, 0.6, facecolor='#FFF3CD', 
                                  edgecolor='#856404', linewidth=2)
    ax.add_patch(innovation_box)
    ax.text(7, 0.6, 'KEY INNOVATION: Containers show collective ensemble behavior - ALL base learners contribute to performance metrics', 
           ha='center', va='center', fontsize=12, fontweight='bold', color='#856404')
    ax.text(7, 0.3, 'Meta-learner prevents poor models from dominating by learning optimal weights α based on cross-validation performance', 
           ha='center', va='center', fontsize=11, color='#856404')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/meta_dml_pipeline_diagram.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return fig

def main():
    """Main function to analyze all three countries"""
    
    countries = ['bangladesh', 'philippines', 'zimbabwe']
    country_results = {}
    
    print("Loading data from all three countries...")
    print("Reading from enhanced_benchmark directories for each country...")
    
    for country in countries:
        try:
            # Read country-specific data from enhanced_benchmark_{country} directory
            data_path = f'outputs/enhanced_benchmark_{country}/all_methods_results.csv'
            print(f"\nReading data for {country.upper()}: {data_path}")
            results = pd.read_csv(data_path)
            
            # Clean data
            clean_data = results.dropna(subset=['Effect']).copy()
            clean_data = clean_data[clean_data['Method'] != 'calibration']
            
            print(f"{country.upper()}: Analyzing {len(clean_data)} estimates from {clean_data['Method'].nunique()} methods")
            print(f"Covering {clean_data['Indicator'].nunique()} indicators and {clean_data['Policy_Year'].nunique()} policy years")
            
            # Calculate POCME v2.1 scores (run analysis)
            pocme_results = calculate_pocme_v2_1_scores(clean_data)
            
            # Display results for this country
            pocme_df = display_country_results(pocme_results, country)
            
            # Store results
            country_results[country] = pocme_df
            
            # Save country-specific results
            os.makedirs(f'outputs/pocme_multi_country_analysis/{country}', exist_ok=True)
            pocme_df.to_csv(f'outputs/pocme_multi_country_analysis/{country}/pocme_scores.csv')
            
        except FileNotFoundError:
            print(f"Error: Could not find data for {country}")
            print(f"   Expected file: outputs/enhanced_benchmark_{country}/all_methods_results.csv")
            print("   Please ensure enhanced_benchmark files exist for each country")
            continue
    
    if country_results:
        # Create enhanced stacked bar visualization
        print(f"\nCreating enhanced multi-country visualization...")
        create_multi_country_diagram(country_results, 'outputs/pocme_multi_country_analysis')
        
        # Create Meta-DML pipeline diagram
        print(f"\nCreating Meta-DML pipeline diagram...")
        create_meta_dml_pipeline_diagram('outputs/pocme_multi_country_analysis')
        
        # Analyze best policies according to Meta-DML
        policy_insights = analyze_best_policies_by_meta_dml(country_results)
        
        print(f"\nMULTI-COUNTRY POCME v2.1 ANALYSIS COMPLETE!")
        print(f"Results saved to outputs/pocme_multi_country_analysis/")
        print(f"   • enhanced_multi_country_pocme_analysis.png (with integrated policy insights)")
        for country in country_results.keys():
            print(f"   • {country}/pocme_scores.csv")
        
        # Summary comparison
        print(f"\nCROSS-COUNTRY SUMMARY:")
        print("=" * 60)
        for country, pocme_df in country_results.items():
            top_method = pocme_df.index[0]
            top_score = pocme_df.iloc[0]['pocme_v2_1_score']
            print(f"{country.title():>12}: {top_method:<12} ({top_score:.1f} points)")
            
            # Check if meta_dml is in top 3
            if 'meta_dml' in pocme_df.head(3).index:
                meta_rank = list(pocme_df.index).index('meta_dml') + 1
                meta_score = pocme_df.loc['meta_dml', 'pocme_v2_1_score']
                print(f"             Meta-DML Rank: #{meta_rank} ({meta_score:.1f} points)")
            else:
                print(f"             Meta-DML: Outside top 3")
        
        print(f"\nENHANCED VISUALIZATION FEATURES:")
        print(f"   • Stacked bars showing individual component contributions")
        print(f"   • Bottom-up ranking (best performers at bottom)")
        print(f"   • Meta-DML highlighted with red border")
        print(f"   • Reference lines for thresholds")
        print(f"   • Integrated high-impact policy insights (4th column)")
        print(f"   • Evidence-based policy rankings with impact metrics")
        
    else:
        print("No country data found. Please ensure the enhanced_benchmark files exist.")
        print("   Required files:")
        for country in countries:
            print(f"   • outputs/enhanced_benchmark_{country}/all_methods_results.csv")

if __name__ == "__main__":
    main()