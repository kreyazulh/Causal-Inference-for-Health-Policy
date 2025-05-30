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
os.makedirs('outputs/pocme_v2_1_analysis', exist_ok=True)

print("🚀 POCME v2.1: RESEARCH-BASED CAUSAL METHOD EVALUATION")
print("🔬 Integrated Statistical Significance Quality Assessment")
print("=" * 70)

def calculate_research_based_significance(method_data):
    """
    Research-based significance evaluation replacing flawed "clinical significance"
    
    Based on:
    - Benjamini & Hochberg (1995) False Discovery Rate
    - Power analysis for causal discovery (2023)
    - Sensitivity analysis frameworks for causal inference
    """
    
    effects = method_data['Effect'].values
    abs_effects = np.abs(effects)
    
    # Define policy-relevant significance thresholds
    trivial_threshold = 2.0    # <2% = trivial for policy
    small_threshold = 5.0      # 2-5% = small but meaningful
    moderate_threshold = 15.0  # 5-15% = moderate impact
    # >15% = large impact
    
    # Effect size distribution
    trivial_effects = (abs_effects < trivial_threshold).mean()
    small_effects = ((abs_effects >= trivial_threshold) & 
                    (abs_effects < small_threshold)).mean()
    moderate_effects = ((abs_effects >= small_threshold) & 
                       (abs_effects < moderate_threshold)).mean()
    large_effects = (abs_effects >= moderate_threshold).mean()
    
    # Statistical significance assessment
    effect_variance = method_data['Effect'].var()
    n_obs = len(method_data)
    
    if effect_variance > 0 and n_obs > 1:
        effect_se = np.sqrt(effect_variance / n_obs)
        t_stats = effects / effect_se
        p_values = 2 * (1 - stats.norm.cdf(np.abs(t_stats)))
    else:
        p_values = np.ones(len(effects))
    
    # Power analysis: detecting meaningful effects
    meaningful_mask = abs_effects >= small_threshold
    if meaningful_mask.sum() > 0:
        power_proxy = (p_values[meaningful_mask] < 0.05).mean()
    else:
        power_proxy = 0.0
    
    # Type I error: falsely detecting trivial effects
    trivial_mask = abs_effects < trivial_threshold
    if trivial_mask.sum() > 0:
        type1_proxy = (p_values[trivial_mask] < 0.05).mean()
    else:
        type1_proxy = 0.0
    
    # False Discovery Rate estimation (Benjamini-Hochberg)
    sorted_p = np.sort(p_values)
    n_tests = len(p_values)
    bh_threshold = 0.05
    
    significant_bh = 0
    for i in range(n_tests - 1, -1, -1):
        if sorted_p[i] <= (i + 1) / n_tests * bh_threshold:
            significant_bh = i + 1
            break
    
    n_significant = np.sum(p_values < 0.05)
    estimated_fdr = 1 - (significant_bh / max(1, n_significant))
    
    # Conservative vs Liberal assessment
    tiny_but_significant = ((abs_effects < trivial_threshold) & 
                          (p_values < 0.05)).mean()
    moderate_and_significant = ((abs_effects >= small_threshold) & 
                              (p_values < 0.05)).mean()
    
    # Effect realism (avoid extreme outliers)
    extreme_effects = (abs_effects > 50).mean()  # >50% is usually implausible
    near_zero_effects = (abs_effects < 1).mean()  # <1% might be just noise
    
    # Scoring based on research criteria
    
    # 1. Power Score (0-100)
    if power_proxy >= 0.8:
        power_score = 100
    elif power_proxy >= 0.6:
        power_score = 80
    elif power_proxy >= 0.4:
        power_score = 60
    elif power_proxy >= 0.2:
        power_score = 40
    else:
        power_score = 20
    
    # 2. FDR Control Score (0-100)
    if estimated_fdr <= 0.05:
        fdr_score = 100
    elif estimated_fdr <= 0.1:
        fdr_score = 80
    elif estimated_fdr <= 0.2:
        fdr_score = 60
    elif estimated_fdr <= 0.3:
        fdr_score = 40
    else:
        fdr_score = 20
    
    # 3. Effect Realism Score (0-100)
    realism_score = 100
    realism_score -= extreme_effects * 200  # Heavy penalty for extreme effects
    realism_score -= near_zero_effects * 50  # Penalty for too much noise
    realism_score = max(0, realism_score)
    
    # 4. Balance Score (0-100)
    if tiny_but_significant <= 0.1 and moderate_and_significant >= 0.7:
        balance_score = 100
    elif tiny_but_significant <= 0.2 and moderate_and_significant >= 0.5:
        balance_score = 80
    elif tiny_but_significant <= 0.3 and moderate_and_significant >= 0.3:
        balance_score = 60
    else:
        balance_score = 40
    
    # Adjust for being too conservative or liberal
    if moderate_and_significant < 0.3:  # Too conservative
        balance_score *= 0.7
    if tiny_but_significant > 0.3:      # Too liberal
        balance_score *= 0.7
    
    # Overall significance quality score
    significance_score = (
        power_score * 0.35 +      # Power is most important
        fdr_score * 0.25 +        # FDR control is critical
        realism_score * 0.20 +    # Effect sizes should be plausible
        balance_score * 0.20      # Balance between conservative/liberal
    )
    
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
        'mean_abs_effect': np.mean(abs_effects)
    }

def calculate_pocme_v2_1_scores(data):
    """
    POCME v2.1: Integrated research-based significance evaluation
    
    KEY IMPROVEMENTS:
    1. Replaced flawed "clinical significance" with research-based significance quality
    2. Proper completeness calculation with failure diagnostics
    3. Balanced implementation safety weighting
    4. Comprehensive policy-oriented evaluation
    """
    
    # Domain constraints (evidence-based from literature)
    domain_constraints = {
        'Mortality rate, infant (per 1,000 live births)': (-90, 10),
        'Life expectancy at birth, total (years)': (-10, 15),
        'Maternal mortality ratio (modeled estimate, per 100,000 live births)': (-80, 10),
        'Immunization, measles (% of children ages 12-23 months)': (-20, 100),
        'Prevalence of undernourishment (% of population)': (-70, 15),
        'Mortality rate, under-5 (per 1,000 live births)': (-85, 10), 
        'Incidence of tuberculosis (per 100,000 people)': (-80, 20),
        'Hospital beds (per 1,000 people)': (-50, 200)
    }
    
    # Default computation times (seconds)
    default_times = {
        'ITS': 0.001, 'DiD': 0.001, 'SCM': 0.01, 'ASCM': 0.01,
        'CausalImpact': 0.1, 'BayesianCausal': 25.0, 'CausalForests': 0.5,
        'BART': 0.1, 'PSM': 0.1, 'DoubleML': 1.0, 'meta_dml': 1.5
    }
    
    # Get all possible indicator/policy year combinations
    all_combinations = data[['Indicator', 'Policy_Year']].drop_duplicates()
    total_possible = len(all_combinations)
    
    print(f"📊 Total possible scenarios: {total_possible}")
    print(f"📈 Unique indicators: {data['Indicator'].nunique()}")
    print(f"📅 Unique policy years: {data['Policy_Year'].nunique()}")
    
    method_scores = {}
    
    for method in data['Method'].unique():
        method_data = data[data['Method'] == method].copy()
        
        if len(method_data) == 0:
            continue
        
        # ==== METRIC 1: DOMAIN ADHERENCE (25% weight) ====
        plausible_count = 0
        total_count = 0
        violation_magnitudes = []
        
        for _, row in method_data.iterrows():
            if row['Indicator'] in domain_constraints:
                min_val, max_val = domain_constraints[row['Indicator']]
                total_count += 1
                if min_val <= row['Effect'] <= max_val:
                    plausible_count += 1
                else:
                    if row['Effect'] < min_val:
                        violation_magnitudes.append(min_val - row['Effect'])
                    else:
                        violation_magnitudes.append(row['Effect'] - max_val)
        
        plausibility_rate = plausible_count / max(total_count, 1)
        domain_score = min(plausibility_rate * 100, 100)
        
        if plausibility_rate > 0.95:
            domain_score = min(domain_score + 10, 100)
        
        # ==== METRIC 2: SIGNIFICANCE QUALITY (20% weight) ====
        # NEW: Research-based significance evaluation
        sig_metrics = calculate_research_based_significance(method_data)
        significance_score = sig_metrics['significance_score']
        
        # ==== METRIC 3: PRECISION & RELIABILITY (18% weight) ====
        effect_variance = method_data['Effect'].var()
        
        if effect_variance == 0:
            precision_score = 100
        elif effect_variance < 1000:
            precision_score = 90
        elif effect_variance < 10000:
            precision_score = 70
        elif effect_variance < 100000:
            precision_score = 50
        else:
            precision_score = 20
        
        # ==== METRIC 4: COMPLETENESS & COVERAGE (17% weight) ====
        # Proper completeness calculation with diagnostic info
        method_combinations = method_data[['Indicator', 'Policy_Year']].drop_duplicates()
        raw_completeness_rate = len(method_combinations) / total_possible
        missing_scenarios = total_possible - len(method_combinations)
        
        # Completeness scoring with graduated penalties
        if raw_completeness_rate == 1.0:
            completeness_score = 100
        elif raw_completeness_rate >= 0.9:
            completeness_score = 95 - (0.1 - (1.0 - raw_completeness_rate)) * 50
        elif raw_completeness_rate >= 0.75:
            completeness_score = 80 - (0.15 - (0.9 - raw_completeness_rate)) * 100
        elif raw_completeness_rate >= 0.5:
            completeness_score = 60 - (0.25 - (0.75 - raw_completeness_rate)) * 80
        else:
            completeness_score = max(20, 40 - (0.5 - raw_completeness_rate) * 40)
        
        # Positivity assessment (outlier detection)
        effect_iqr = method_data['Effect'].quantile(0.75) - method_data['Effect'].quantile(0.25)
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
        
        # ==== METRIC 5: IMPLEMENTATION SAFETY (15% weight) ====
        avg_violation = np.mean(violation_magnitudes) if violation_magnitudes else 0
        
        if avg_violation == 0:
            safety_score = 100
        elif avg_violation < 25:
            safety_score = 90
        elif avg_violation < 50:
            safety_score = 70
        elif avg_violation < 100:
            safety_score = 50
        else:
            safety_score = 20
        
        # Adjust for completeness reliability
        if raw_completeness_rate < 0.5:
            safety_score *= 0.8
        
        # ==== METRIC 6: COMPUTATIONAL EFFICIENCY (5% weight) ====
        computation_time = default_times.get(method, 1.0)
        
        if computation_time < 0.1:
            efficiency_score = 100
        elif computation_time < 1.0:
            efficiency_score = 90
        elif computation_time < 10.0:
            efficiency_score = 70
        elif computation_time < 60.0:
            efficiency_score = 50
        else:
            efficiency_score = 30
        
        # ==== CALCULATE POCME v2.1 SCORE ====
        pocme_score = (
            domain_score * 0.25 +           # Domain adherence (unchanged)
            significance_score * 0.20 +     # Significance quality (NEW)
            precision_score * 0.18 +        # Precision & reliability
            coverage_score * 0.17 +         # Completeness & coverage
            safety_score * 0.15 +           # Implementation safety
            efficiency_score * 0.05         # Computational efficiency
        )
        
        # Store comprehensive results
        method_scores[method] = {
            'pocme_v2_1_score': pocme_score,
            'domain_adherence': domain_score,
            'significance_quality': significance_score,
            'precision_reliability': precision_score,
            'completeness_coverage': coverage_score,
            'implementation_safety': safety_score,
            'computational_efficiency': efficiency_score,
            
            # Significance quality breakdown
            'power_proxy': sig_metrics['power_proxy'],
            'estimated_fdr': sig_metrics['estimated_fdr'],
            'type1_proxy': sig_metrics['type1_proxy'],
            'power_score': sig_metrics['power_score'],
            'fdr_score': sig_metrics['fdr_score'],
            'realism_score': sig_metrics['realism_score'],
            'balance_score': sig_metrics['balance_score'],
            
            # Diagnostic metrics
            'plausibility_rate': plausibility_rate,
            'effect_variance': effect_variance,
            'raw_completeness_rate': raw_completeness_rate,
            'missing_scenarios': missing_scenarios,
            'positivity_score': positivity_score,
            'avg_violation': avg_violation,
            'computation_time': computation_time,
            'n_estimates': len(method_data),
            'n_covered_scenarios': len(method_combinations),
            
            # Effect distribution
            'trivial_effects_pct': sig_metrics['trivial_effects_pct'],
            'small_effects_pct': sig_metrics['small_effects_pct'],
            'moderate_effects_pct': sig_metrics['moderate_effects_pct'],
            'large_effects_pct': sig_metrics['large_effects_pct'],
            'tiny_but_significant_pct': sig_metrics['tiny_but_significant_pct'],
            'moderate_and_significant_pct': sig_metrics['moderate_and_significant_pct'],
            'mean_abs_effect': sig_metrics['mean_abs_effect']
        }
    
    return method_scores

def display_pocme_v2_1_results(pocme_results):
    """Display comprehensive POCME v2.1 results with significance quality"""
    
    print("\n🎯 POCME v2.1 FRAMEWORK WITH RESEARCH-BASED SIGNIFICANCE:")
    print("=" * 70)
    print("✅ REPLACED: Flawed 'clinical significance' with research-based significance quality")
    print("✅ INTEGRATED: False Discovery Rate, Power Analysis, Effect Calibration")
    print("✅ FIXED: Proper completeness calculation with failure diagnostics")
    print("✅ BALANCED: Implementation safety weighting for policy contexts")
    
    print("\n📊 Updated Weight Distribution:")
    weights = {
        'Domain Adherence': (25, 'Evidence-based plausibility constraints'),
        'Significance Quality': (20, 'Power + FDR + Effect calibration'),
        'Precision & Reliability': (18, 'Implementation confidence'),
        'Completeness & Coverage': (17, 'Scenario coverage + positivity'),
        'Implementation Safety': (15, 'Policy risk assessment'),
        'Computational Efficiency': (5, 'Real-time decision needs')
    }
    
    for metric, (weight, rationale) in weights.items():
        print(f"   {metric:>22}: {weight:>2}% - {rationale}")
    
    # Convert to DataFrame and sort
    pocme_df = pd.DataFrame(pocme_results).T
    pocme_df = pocme_df.sort_values('pocme_v2_1_score', ascending=False)
    
    print(f"\n📊 POCME v2.1 COMPREHENSIVE RESULTS:")
    print("=" * 140)
    print(f"{'Rank':<4} {'Method':<15} {'POCME':<6} {'Domain':<7} {'SigQual':<8} {'Precision':<9} {'Coverage':<8} {'Safety':<7} {'Eff':<4} {'Power':<6} {'FDR':<5} {'Complete':<8}")
    print("─" * 140)
    
    for rank, (method, row) in enumerate(pocme_df.iterrows(), 1):
        emoji = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else f"{rank:2d}."
        
        # Enhanced status considering significance quality
        score = row['pocme_v2_1_score']
        power = row['power_proxy']
        fdr = row['estimated_fdr']
        
        if score >= 85 and power >= 0.7 and fdr <= 0.1:
            status = "🟢"  # Excellent overall
        elif score >= 70 and power >= 0.5 and fdr <= 0.2:
            status = "🟡"  # Good overall
        elif score >= 55:
            status = "🟠"  # Acceptable
        else:
            status = "🔴"  # Poor
        
        # Special warnings for significance issues
        if fdr > 0.2:
            status = "⚠️"   # High FDR warning
        elif power < 0.3:
            status = "⚡"   # Low power warning
        
        complete_pct = f"{row['raw_completeness_rate']*100:.0f}%"
        
        print(f"{emoji:<4} {method:<15} {row['pocme_v2_1_score']:5.1f}{status} "
              f"{row['domain_adherence']:6.1f} {row['significance_quality']:7.1f} "
              f"{row['precision_reliability']:8.1f} {row['completeness_coverage']:7.1f} "
              f"{row['implementation_safety']:6.1f} {row['computational_efficiency']:3.1f} "
              f"{row['power_proxy']:5.2f} {row['estimated_fdr']:4.2f} {complete_pct:>7}")
    
    print(f"\n🔬 SIGNIFICANCE QUALITY LEGEND:")
    print(f"🟢 = Excellent (high power + low FDR)  🟡 = Good (moderate power + FDR)")
    print(f"🟠 = Acceptable overall score          🔴 = Poor overall performance")
    print(f"⚠️ = HIGH FDR: Many false discoveries  ⚡ = LOW POWER: Missing true effects")
    
    # Significance quality analysis
    print(f"\n🔍 SIGNIFICANCE QUALITY ANALYSIS:")
    print("=" * 60)
    
    # High FDR methods
    high_fdr_methods = pocme_df[pocme_df['estimated_fdr'] > 0.2]
    if len(high_fdr_methods) > 0:
        print("🚨 HIGH FALSE DISCOVERY RATE METHODS:")
        for method, row in high_fdr_methods.iterrows():
            print(f"   ⚠️ {method}: FDR = {row['estimated_fdr']:.2f} ({row['estimated_fdr']*100:.0f}% false discoveries)")
    
    # Low power methods
    low_power_methods = pocme_df[pocme_df['power_proxy'] < 0.4]
    if len(low_power_methods) > 0:
        print("⚡ LOW STATISTICAL POWER METHODS:")
        for method, row in low_power_methods.iterrows():
            print(f"   ⚡ {method}: Power = {row['power_proxy']:.2f} (missing {(1-row['power_proxy'])*100:.0f}% of true effects)")
    
    # Liberal methods
    liberal_methods = pocme_df[pocme_df['tiny_but_significant_pct'] > 30]
    if len(liberal_methods) > 0:
        print("🔥 OVERLY LIBERAL METHODS:")
        for method, row in liberal_methods.iterrows():
            print(f"   🔥 {method}: {row['tiny_but_significant_pct']:.0f}% of trivial effects declared significant")
    
    # Conservative methods
    conservative_methods = pocme_df[pocme_df['moderate_and_significant_pct'] < 30]
    if len(conservative_methods) > 0:
        print("🧊 OVERLY CONSERVATIVE METHODS:")
        for method, row in conservative_methods.iterrows():
            print(f"   🧊 {method}: Only {row['moderate_and_significant_pct']:.0f}% of moderate+ effects detected")
    
    # Top performers with significance analysis
    print(f"\n🏆 TOP PERFORMERS WITH SIGNIFICANCE ANALYSIS:")
    top_methods = pocme_df.head(3)
    for i, (method, row) in enumerate(top_methods.iterrows(), 1):
        print(f"   {i}. {method}: {row['pocme_v2_1_score']:.1f} points")
        sig_strengths = []
        if row['power_proxy'] >= 0.7:
            sig_strengths.append("high power")
        if row['estimated_fdr'] <= 0.1:
            sig_strengths.append("low FDR")
        if row['balance_score'] >= 80:
            sig_strengths.append("well-balanced")
        if row['realism_score'] >= 80:
            sig_strengths.append("realistic effects")
        
        if sig_strengths:
            print(f"      Significance strengths: {', '.join(sig_strengths)}")
        
        sig_concerns = []
        if row['power_proxy'] < 0.5:
            sig_concerns.append("low power")
        if row['estimated_fdr'] > 0.15:
            sig_concerns.append("high FDR")
        if row['tiny_but_significant_pct'] > 20:
            sig_concerns.append("too liberal")
        if row['moderate_and_significant_pct'] < 40:
            sig_concerns.append("too conservative")
        
        if sig_concerns:
            print(f"      Significance concerns: {', '.join(sig_concerns)}")
    
    return pocme_df

def create_integrated_analysis_plots(pocme_df, output_dir):
    """Create comprehensive visualizations for POCME v2.1 with significance analysis"""
    
    fig, axes = plt.subplots(3, 3, figsize=(20, 15))
    fig.suptitle('POCME v2.1: Integrated Research-Based Method Evaluation', fontsize=16, fontweight='bold')
    
    # 1. Main POCME v2.1 scores
    ax1 = axes[0, 0]
    methods = pocme_df.index
    scores = pocme_df['pocme_v2_1_score']
    colors = plt.cm.viridis(np.linspace(0, 1, len(methods)))
    
    bars = ax1.barh(methods, scores, color=colors)
    for i, (method, score) in enumerate(zip(methods, scores)):
        ax1.text(score + 1, i, f'{score:.1f}', va='center', fontweight='bold')
    
    ax1.set_xlabel('POCME v2.1 Score')
    ax1.set_title('Overall Method Rankings')
    ax1.set_xlim(0, 110)
    ax1.axvline(x=70, color='orange', linestyle='--', alpha=0.7, label='Good')
    ax1.axvline(x=85, color='green', linestyle='--', alpha=0.7, label='Excellent')
    ax1.legend()
    
    # 2. Power vs FDR Analysis
    ax2 = axes[0, 1]
    scatter = ax2.scatter(pocme_df['power_proxy'], pocme_df['estimated_fdr'],
                         s=pocme_df['pocme_v2_1_score']*2, alpha=0.7, 
                         c=pocme_df['significance_quality'], cmap='viridis')
    
    for method, row in pocme_df.iterrows():
        ax2.annotate(method, (row['power_proxy'], row['estimated_fdr']),
                    xytext=(3, 3), textcoords='offset points', fontsize=8)
    
    ax2.set_xlabel('Statistical Power')
    ax2.set_ylabel('False Discovery Rate')
    ax2.set_title('Power vs FDR Trade-off\n(Color=Sig Quality, Size=POCME Score)')
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0.05, color='green', linestyle='--', alpha=0.7, label='Good FDR')
    ax2.axhline(y=0.1, color='orange', linestyle='--', alpha=0.7, label='OK FDR')
    ax2.axvline(x=0.8, color='green', linestyle='--', alpha=0.7, label='Good Power')
    ax2.axvline(x=0.6, color='orange', linestyle='--', alpha=0.7, label='OK Power')
    ax2.legend()
    plt.colorbar(scatter, ax=ax2, label='Significance Quality')
    
    # 3. Completeness vs Significance Quality
    ax3 = axes[0, 2]
    ax3.scatter(pocme_df['raw_completeness_rate']*100, pocme_df['significance_quality'],
               s=100, alpha=0.7, c=pocme_df['pocme_v2_1_score'], cmap='viridis')
    
    for method, row in pocme_df.iterrows():
        ax3.annotate(method, (row['raw_completeness_rate']*100, row['significance_quality']),
                    xytext=(3, 3), textcoords='offset points', fontsize=8)
    
    ax3.set_xlabel('Completeness Rate (%)')
    ax3.set_ylabel('Significance Quality Score')
    ax3.set_title('Completeness vs Significance\n(Color = POCME Score)')
    ax3.grid(True, alpha=0.3)
    ax3.axvline(x=90, color='green', linestyle='--', alpha=0.7, label='Good Coverage')
    ax3.axhline(y=70, color='orange', linestyle='--', alpha=0.7, label='Good Significance')
    ax3.legend()
    
    # 4. Component breakdown
    ax4 = axes[1, 0]
    components = ['domain_adherence', 'significance_quality', 'precision_reliability', 
                 'completeness_coverage', 'implementation_safety', 'computational_efficiency']
    component_labels = ['Domain', 'Significance', 'Precision', 'Coverage', 'Safety', 'Efficiency']
    weights = [0.25, 0.20, 0.18, 0.17, 0.15, 0.05]
    
    bottom = np.zeros(len(methods))
    for i, (comp, label, weight) in enumerate(zip(components, component_labels, weights)):
        values = pocme_df[comp] * weight
        ax4.barh(methods, values, left=bottom, label=f'{label} ({weight*100:.0f}%)', alpha=0.8)
        bottom += values
    
    ax4.set_xlabel('Weighted Contribution to Score')
    ax4.set_title('POCME v2.1 Component Breakdown')
    ax4.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # 5. Effect Size Distribution by Method
    ax5 = axes[1, 1]
    effect_categories = ['trivial_effects_pct', 'small_effects_pct', 'moderate_effects_pct', 'large_effects_pct']
    category_labels = ['Trivial (<2%)', 'Small (2-5%)', 'Moderate (5-15%)', 'Large (>15%)']
    colors_cat = ['lightcoral', 'gold', 'lightblue', 'darkgreen']
    
    bottom = np.zeros(len(pocme_df))
    for cat, label, color in zip(effect_categories, category_labels, colors_cat):
        values = pocme_df[cat]
        ax5.barh(pocme_df.index, values, left=bottom, label=label, color=color, alpha=0.8)
        bottom += values
    
    ax5.set_xlabel('Percentage of Effects')
    ax5.set_title('Effect Size Distribution')
    ax5.legend()
    
    # 6. Conservative vs Liberal Assessment
    ax6 = axes[1, 2]
    ax6.scatter(pocme_df['tiny_but_significant_pct'], pocme_df['moderate_and_significant_pct'],
               s=pocme_df['balance_score']*2, alpha=0.7, c=pocme_df['significance_quality'], cmap='viridis')
    
    for method, row in pocme_df.iterrows():
        ax6.annotate(method, (row['tiny_but_significant_pct'], row['moderate_and_significant_pct']),
                    xytext=(3, 3), textcoords='offset points', fontsize=8)
    
    ax6.set_xlabel('% Trivial Effects Declared Significant')
    ax6.set_ylabel('% Moderate+ Effects Detected')
    ax6.set_title('Conservative vs Liberal Balance\n(Size=Balance Score, Color=Sig Quality)')
    ax6.grid(True, alpha=0.3)
    ax6.axhline(y=70, color='green', linestyle='--', alpha=0.7, label='Good Power')
    ax6.axvline(x=10, color='green', linestyle='--', alpha=0.7, label='Good Specificity')
    ax6.legend()
    
    # 7. Missing scenarios analysis
    ax7 = axes[2, 0]
    missing_data = pocme_df['missing_scenarios'].sort_values(ascending=True)
    colors_missing = ['green' if x == 0 else 'orange' if x <= 5 else 'red' for x in missing_data.values]
    
    bars = ax7.barh(missing_data.index, missing_data.values, color=colors_missing, alpha=0.7)
    ax7.set_xlabel('Number of Missing Scenarios')
    ax7.set_title('Method Reliability\n(Missing Scenario Count)')
    ax7.axvline(x=5, color='orange', linestyle='--', alpha=0.5, label='Concern Threshold')
    ax7.axvline(x=10, color='red', linestyle='--', alpha=0.5, label='High Concern')
    ax7.legend()
    
    # 8. Significance Quality Components
    ax8 = axes[2, 1]
    sig_components = ['power_score', 'fdr_score', 'realism_score', 'balance_score']
    sig_labels = ['Power', 'FDR Control', 'Realism', 'Balance']
    sig_weights = [0.35, 0.25, 0.20, 0.20]
    
    bottom = np.zeros(len(pocme_df))
    for comp, label, weight in zip(sig_components, sig_labels, sig_weights):
        values = pocme_df[comp] * weight
        ax8.barh(pocme_df.index, values, left=bottom, label=f'{label} ({weight*100:.0f}%)', alpha=0.8)
        bottom += values
    
    ax8.set_xlabel('Weighted Contribution')
    ax8.set_title('Significance Quality Breakdown')
    ax8.legend()
    
    # 9. Implementation Guidance
    ax9 = axes[2, 2]
    
    # Categorize methods for policy implementation
    excellent = pocme_df[(pocme_df['pocme_v2_1_score'] >= 75) & 
                        (pocme_df['power_proxy'] >= 0.6) & 
                        (pocme_df['estimated_fdr'] <= 0.15)]
    good = pocme_df[(pocme_df['pocme_v2_1_score'] >= 65) & 
                   (pocme_df['power_proxy'] >= 0.4) & 
                   (pocme_df['estimated_fdr'] <= 0.25)]
    caution = pocme_df[(pocme_df['pocme_v2_1_score'] >= 55) | 
                      (pocme_df['implementation_safety'] >= 60)]
    avoid = pocme_df[~pocme_df.index.isin(caution.index)]
    
    categories = ['RECOMMENDED\n(High Score + Good Sig)', 'ACCEPTABLE\n(Good Score + OK Sig)', 
                 'USE WITH CAUTION\n(Limitations Present)', 'AVOID\n(Poor Performance)']
    counts = [len(excellent), len(good) - len(excellent), 
              len(caution) - len(good), len(avoid)]
    colors_rec = ['darkgreen', 'gold', 'orange', 'red']
    
    bars = ax9.bar(range(len(categories)), counts, color=colors_rec, alpha=0.7)
    ax9.set_xticks(range(len(categories)))
    ax9.set_xticklabels(categories, rotation=45, ha='right', fontsize=9)
    ax9.set_ylabel('Number of Methods')
    ax9.set_title('Policy Implementation\nRecommendations')
    
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        ax9.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{count}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/pocme_v2_1_integrated_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return excellent.index.tolist(), good.index.tolist(), caution.index.tolist(), avoid.index.tolist()

def analyze_method_significance_failures(data, pocme_results):
    """Analyze significance detection failures for specific methods"""
    
    print(f"\n🔍 METHOD SIGNIFICANCE FAILURE ANALYSIS:")
    print("=" * 60)
    
    for method in ['meta_dml', 'DoubleML', 'BART']:  # Focus on ML methods
        if method in pocme_results:
            method_data = data[data['Method'] == method]
            result = pocme_results[method]
            
            print(f"\n📊 {method} Significance Analysis:")
            print(f"   • Power (detecting true effects): {result['power_proxy']:.2f}")
            print(f"   • FDR (false discovery rate): {result['estimated_fdr']:.2f}")
            print(f"   • Type I error proxy: {result['type1_proxy']:.2f}")
            print(f"   • Effect distribution: {result['trivial_effects_pct']:.0f}% trivial, "
                  f"{result['moderate_effects_pct']:.0f}% moderate, {result['large_effects_pct']:.0f}% large")
            print(f"   • Conservative-liberal balance: {result['tiny_but_significant_pct']:.0f}% trivial significant, "
                  f"{result['moderate_and_significant_pct']:.0f}% moderate significant")
            
            # Diagnosis
            issues = []
            if result['power_proxy'] < 0.5:
                issues.append("LOW POWER: Missing many true effects")
            if result['estimated_fdr'] > 0.2:
                issues.append("HIGH FDR: Many false discoveries")
            if result['tiny_but_significant_pct'] > 30:
                issues.append("TOO LIBERAL: Declaring trivial effects significant")
            if result['moderate_and_significant_pct'] < 30:
                issues.append("TOO CONSERVATIVE: Missing moderate effects")
            
            if issues:
                print(f"   ⚠️ Issues: {'; '.join(issues)}")
            else:
                print(f"   ✅ Good significance detection performance")

# Main execution
if __name__ == "__main__":
    try:
        results = pd.read_csv('outputs/enhanced_benchmark/all_methods_results.csv')
        
        # Clean data
        clean_data = results.dropna(subset=['Effect']).copy()
        clean_data = clean_data[clean_data['Method'] != 'calibration']
        
        print(f"📊 Analyzing {len(clean_data)} estimates from {clean_data['Method'].nunique()} methods")
        print(f"📈 Covering {clean_data['Indicator'].nunique()} indicators and {clean_data['Policy_Year'].nunique()} policy years")
        
        # Calculate POCME v2.1 scores with integrated significance evaluation
        pocme_results = calculate_pocme_v2_1_scores(clean_data)
        
        # Display comprehensive results
        pocme_df = display_pocme_v2_1_results(pocme_results)
        
        # Create integrated visualizations
        excellent, good, caution, avoid = create_integrated_analysis_plots(pocme_df, 'outputs/pocme_v2_1_analysis')
        
        # Analyze significance failures for specific methods
        analyze_method_significance_failures(clean_data, pocme_results)
        
        # Save results
        pocme_df.to_csv('outputs/pocme_v2_1_analysis/pocme_v2_1_integrated_scores.csv')
        
        print(f"\n✅ POCME v2.1 INTEGRATED ANALYSIS COMPLETE!")
        print(f"📁 Results saved to outputs/pocme_v2_1_analysis/")
        print(f"   • pocme_v2_1_integrated_scores.csv")
        print(f"   • pocme_v2_1_integrated_analysis.png")
        
        print(f"\n🎯 KEY IMPROVEMENTS IN v2.1:")
        print(f"   ✅ Research-based significance evaluation (Power + FDR + Calibration)")
        print(f"   ✅ Proper completeness calculation with failure diagnostics")
        print(f"   ✅ Balanced risk assessment without over-emphasis")
        print(f"   ✅ Comprehensive policy-oriented method evaluation")
        
    except FileNotFoundError:
        print("❌ Error: Could not find 'outputs/enhanced_benchmark/all_methods_results.csv'")
        print("Please run the benchmark analysis first to generate the required data.")
        print("\n📝 For testing purposes, here's how the integrated framework works:")
        
        # Create sample data for demonstration
        np.random.seed(42)
        sample_data = pd.DataFrame({
            'Method': ['ITS', 'DiD', 'meta_dml', 'SCM', 'BART'] * 100,
            'Indicator': ['Mortality rate, infant (per 1,000 live births)', 
                         'Life expectancy at birth, total (years)'] * 250,
            'Policy_Year': [2010, 2015] * 250,
            'Effect': np.concatenate([
                np.random.normal(0, 5, 100),      # ITS: small effects
                np.random.normal(8, 10, 100),     # DiD: moderate effects with noise
                np.random.normal(0, 25, 100),     # meta_dml: high variance
                np.random.normal(15, 8, 100),     # SCM: larger, consistent effects
                np.random.normal(3, 3, 100)       # BART: small, precise effects
            ])
        })
        
        print(f"\n🧪 DEMO WITH SAMPLE DATA:")
        pocme_results = calculate_pocme_v2_1_scores(sample_data)
        pocme_df = display_pocme_v2_1_results(pocme_results)
        analyze_method_significance_failures(sample_data, pocme_results)