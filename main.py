# Example main script to run the enhanced analysis with new methods

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from config import POLICY_TIMELINE, POTENTIAL_INDICATORS, MIN_YEARS_REQUIRED

# Import modules
from src.data_preprocessing import (
    load_and_preprocess_data, 
    select_key_indicators, 
    create_time_series_dataset
)
from src.causal_inference import BayesianCausalImpactModel
from src.causal_wavelet import CausalWaveletAnalysis
from src.benchmarks import (
    ChangePointDetectionEvaluator, 
    CausalInferenceEvaluator
)

# Global variable to store benchmark results
benchmark_results = None
all_country_results = {}

def run_enhanced_analysis(data_path=None):
    """Run the enhanced analysis with additional causal methods."""
    global benchmark_results
    print("Starting Enhanced Health Policy Analysis with Multiple Causal Methods...")
    
    try:
        # Step 1: Load and preprocess data
        print("\n--- Step 1: Loading and preprocessing data ---")
        df = load_and_preprocess_data(data_path)
        
        key_indicators, coverage_info = select_key_indicators(
            df, 
            POTENTIAL_INDICATORS, 
            min_years_required=MIN_YEARS_REQUIRED
        )
        
        # Step 2: Create time series dataset
        print("\n--- Step 2: Creating time series dataset ---")
        df_timeseries, df_scaled = create_time_series_dataset(df, key_indicators)
        
        # Define key analysis indicators and policies
        key_analysis_indicators = [
            'Mortality rate, infant (per 1,000 live births)',
            'Life expectancy at birth, total (years)',
            'Maternal mortality ratio (modeled estimate, per 100,000 live births)',
            'Immunization, measles (% of children ages 12-23 months)',
            'Prevalence of undernourishment (% of population)',
            'Mortality rate, under-5 (per 1,000 live births)',
            'Incidence of tuberculosis (per 100,000 people)',
            'Hospital beds (per 1,000 people)',
        ]
        key_analysis_indicators = [ind for ind in key_analysis_indicators if ind in df_timeseries.columns]

        # Determine country and policy years based on data path
        if 'zwe' in data_path:
            key_policies = ['1980', '1982', '1988', '1990', '1996', '1997', '2000', '2003', '2008', '2009', '2013', '2016', '2018', '2021', '2023']
 
        elif 'phl' in data_path:
            key_policies = ['1972', '1976', '1978', '1980', '1988', '1991', '1993', '1995', '1999', '2005', '2008', '2010', '2012', '2016', '2017', '2019', '2021']

        else:  # Bangladesh
            key_policies = ['1972', '1976', '1982', '1988', '1993', '1998', '2000', '2003', '2005', '2008', '2011', '2016', '2021']

            
        policy_years_int = [int(year) for year in key_policies]
        
        print("\n--- Step 3: Running comprehensive benchmark with all causal methods (including Meta-DML) ---")
        
        # Create output directory
        output_dir = 'outputs/enhanced_benchmark/'
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize evaluator
        ci_evaluator = CausalInferenceEvaluator(df_timeseries, POLICY_TIMELINE)
        
        # Run comparative analysis with all methods (now includes Meta-DML automatically)
        benchmark_results = ci_evaluator.run_comparative_analysis(
            key_analysis_indicators, 
            policy_years_int
        )
        
        # Evaluate calibration (now includes Meta-DML)
        calibration_results = ci_evaluator.evaluate_calibration(benchmark_results)
        
        # Add calibration results to benchmark_results
        benchmark_results['calibration'] = calibration_results
        
        # Save results as CSV for easy reference
        methods_df = []
        
        for indicator in key_analysis_indicators:
            for policy_year in policy_years_int:
                for method, results in benchmark_results[indicator][policy_year].items():
                    # Handle both dictionary and float results
                    if isinstance(results, dict):
                        effect = results.get('relative_effect', np.nan)
                        lower_bound = results.get('lower_bound', np.nan)
                        upper_bound = results.get('upper_bound', np.nan)
                        significant = results.get('significance', False)
                        
                        # Special handling for Meta-DML insights
                        if method == 'Meta-DML':
                            meta_insights = results.get('meta_insights', {})
                            dominant_learner = meta_insights.get('dominant_learner', 'unknown')
                            weight_entropy = meta_insights.get('weight_entropy', np.nan)
                            n_effective_learners = meta_insights.get('n_effective_learners', np.nan)
                        else:
                            dominant_learner = ''
                            weight_entropy = np.nan
                            n_effective_learners = np.nan
                            
                    else:
                        effect = results
                        lower_bound = np.nan
                        upper_bound = np.nan
                        significant = False
                        dominant_learner = ''
                        weight_entropy = np.nan
                        n_effective_learners = np.nan
                        
                    methods_df.append({
                        'Indicator': indicator,
                        'Policy_Year': policy_year,
                        'Method': method,
                        'Effect': effect,
                        'Lower_Bound': lower_bound,
                        'Upper_Bound': upper_bound,
                        'Significant': significant,
                        'Dominant_Learner': dominant_learner,
                        'Weight_Entropy': weight_entropy,
                        'N_Effective_Learners': n_effective_learners
                    })
        
        pd.DataFrame(methods_df).to_csv(f'{output_dir}/all_methods_results.csv', index=False)
        
        # Create summary table with all methods
        method_summary = []
        
        all_methods = ['ASCM', 'SCM', 'ITS', 'DiD', 
                      'CausalImpact', 'CausalForests', 'BART', 'PSM', 'DoubleML', 'Meta-DML']
        
        for method in all_methods:
            if method in calibration_results:
                cal = calibration_results[method]
                method_summary.append({
                    'Method': method,
                    'Plausibility_Rate': cal.get('plausibility_rate', np.nan) * 100,
                    'Mean_Abs_Effect': cal.get('mean_abs_effect', np.nan),
                    'Effect_Variance': cal.get('effect_variance', np.nan),
                    'Avg_Violation': cal.get('avg_violation', 0),
                    'Computation_Time': ci_evaluator.method_timings.get(method, np.nan)
                })
        
        pd.DataFrame(method_summary).to_csv(f'{output_dir}/method_comparison.csv', index=False)
        
        return benchmark_results, method_summary, methods_df
        
    except Exception as e:
        print(f"Error in enhanced analysis: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None, None

def print_comparative_results(all_country_results):
    """Print comparative results across all countries"""
    print("\n" + "="*80)
    print("COMPARATIVE META-DML ANALYSIS ACROSS COUNTRIES")
    print("="*80)
    
    for country, (benchmark_results, method_summary, methods_df) in all_country_results.items():
        print(f"\n{'-'*40}")
        print(f"Results for {country}")
        print(f"{'-'*40}")
        
        if method_summary:
            # Find Meta-DML results
            meta_dml_results = next((m for m in method_summary if m['Method'] == 'Meta-DML'), None)
            if meta_dml_results:
                print("\nMeta-DML Performance:")
                print(f"  Plausibility Rate: {meta_dml_results['Plausibility_Rate']:.1f}%")
                print(f"  Mean Absolute Effect: {meta_dml_results['Mean_Abs_Effect']:.1f}%")
                print(f"  Effect Variance: {meta_dml_results['Effect_Variance']:.1f}")
                print(f"  Computation Time: {meta_dml_results['Computation_Time']:.1f}s")
            
            # Compare with DoubleML
            dml_results = next((m for m in method_summary if m['Method'] == 'DoubleML'), None)
            if meta_dml_results and dml_results:
                improvement = meta_dml_results['Plausibility_Rate'] - dml_results['Plausibility_Rate']
                print(f"\nImprovement over DoubleML: {improvement:+.1f} percentage points")
                
                if improvement > 0:
                    print("✅ Meta-DML outperforms standard DoubleML!")
                else:
                    print("⚠️  Meta-DML needs further tuning")
        
        # Print top 3 methods by plausibility
        if method_summary:
            top_methods = sorted(method_summary, key=lambda x: x['Plausibility_Rate'], reverse=True)[:3]
            print("\nTop 3 Methods by Plausibility:")
            for i, method in enumerate(top_methods):
                rank_emoji = "🥇" if i == 0 else "🥈" if i == 1 else "🥉"
                print(f"  {rank_emoji} {method['Method']}: {method['Plausibility_Rate']:.1f}%")

if __name__ == "__main__":
    datasets = {
        'Zimbabwe': 'data/health_zwe.csv',
        'Philippines': 'data/health_phl.csv',
        'Bangladesh': 'data/health_bgd.csv'
    }
    
    for country, data_path in datasets.items():
        print(f"\n{'='*50}")
        print(f"Running Meta-DML Analysis for {country}")
        print(f"{'='*50}")
        
        try:
            # Run the enhanced analysis with the current data path
            benchmark_results, method_summary, methods_df = run_enhanced_analysis(data_path)
            
            # Store results for this country
            if benchmark_results is not None:
                all_country_results[country] = (benchmark_results, method_summary, methods_df)
                
                # Save results in country-specific directory
                output_dir = f'outputs/enhanced_benchmark_{country.lower()}/'
                os.makedirs(output_dir, exist_ok=True)
                
                # Save detailed results
                if method_summary:
                    pd.DataFrame(method_summary).to_csv(f'{output_dir}/method_comparison.csv', index=False)
                
                # Save the detailed all_methods_results.csv for this specific country
                if methods_df:
                    pd.DataFrame(methods_df).to_csv(f'{output_dir}/all_methods_results.csv', index=False)
                    print(f"Saved detailed results to {output_dir}/all_methods_results.csv")
            else:
                print(f"No benchmark results available for {country} Meta-DML analysis")
                
        except Exception as e:
            print(f"Error processing {country} dataset: {str(e)}")
            import traceback
            traceback.print_exc()
            continue
    
    # Print comparative results across all countries
    print_comparative_results(all_country_results)