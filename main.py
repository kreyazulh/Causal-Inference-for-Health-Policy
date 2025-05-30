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

def run_enhanced_analysis():
    """Run the enhanced analysis with additional causal methods."""
    global benchmark_results
    print("Starting Enhanced Bangladesh Health Policy Analysis with Multiple Causal Methods...")
    
    try:
        # Step 1: Load and preprocess data
        print("\n--- Step 1: Loading and preprocessing data ---")
        df = load_and_preprocess_data('data/health_bgd.csv')
        
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
        # key_analysis_indicators = [
        #     'Mortality rate, infant (per 1,000 live births)',
        #     'Life expectancy at birth, total (years)',
        #     'Maternal mortality ratio (modeled estimate, per 100,000 live births)',
        #     'Mortality rate, neonatal (per 1,000 live births)',
        # ]
        key_analysis_indicators = [ind for ind in key_analysis_indicators if ind in df_timeseries.columns]
        
        key_policies = ['1972', '1976', '1982', '1988', '1993', '1998', '2000', '2003', '2005', '2008', '2011', '2016', '2021']
        #key_policies = ['1982', '1998', '2011']
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
        
        # DEBUG: Print benchmark_results structure for inspection
        import pprint
        print("\n--- DEBUG: benchmark_results structure (now with Meta-DML) ---")
        pprint.pprint({k: list(v.keys()) if isinstance(v, dict) else v for k, v in benchmark_results.items() if k != 'calibration'})
        
        # Save results as CSV for easy reference (now includes Meta-DML)
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
                        'Dominant_Learner': dominant_learner,  # Meta-DML specific
                        'Weight_Entropy': weight_entropy,      # Meta-DML specific
                        'N_Effective_Learners': n_effective_learners  # Meta-DML specific
                    })
        
        pd.DataFrame(methods_df).to_csv(f'{output_dir}/all_methods_results.csv', index=False)
        
        # ... existing visualization code remains the same ...
        
        # Step 7: Updated final comparison table (now includes Meta-DML)
        print("\n--- Step 7: Generating final comparison tables (with Meta-DML) ---")
        
        # Create summary table with all methods INCLUDING Meta-DML
        method_summary = []
        
        # UPDATED - Include Meta-DML in the method list
        all_methods = ['ASCM', 'SCM', 'ITS', 'DiD', 
                      'CausalImpact', 'CausalForests', 'BART', 'PSM', 'DoubleML', 'Meta-DML']
        # all_methods = ['BayesianCausal', 'ASCM', 'SCM', 'ITS', 'DiD', 
        #               'CausalImpact', 'CausalForests', 'BART', 'PSM', 'DoubleML', 'Meta-DML']
        
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
        
        # Print key findings INCLUDING Meta-DML performance
        print("\n=== KEY FINDINGS (Including Meta-DML) ===")
        
        if 'Meta-DML' in calibration_results:
            meta_dml_cal = calibration_results['Meta-DML']
            print(f"🚀 Meta-DML Performance:")
            print(f"   Plausibility Rate: {meta_dml_cal.get('plausibility_rate', 0)*100:.1f}%")
            print(f"   Mean Absolute Effect: {meta_dml_cal.get('mean_abs_effect', 0):.1f}%")
            print(f"   Effect Variance: {meta_dml_cal.get('effect_variance', 0):.1f}")
            
            # Compare with DoubleML baseline
            if 'DoubleML' in calibration_results:
                dml_cal = calibration_results['DoubleML']
                dml_plaus = dml_cal.get('plausibility_rate', 0) * 100
                meta_plaus = meta_dml_cal.get('plausibility_rate', 0) * 100
                improvement = meta_plaus - dml_plaus
                print(f"   Improvement over DoubleML: {improvement:+.1f} percentage points")
                
                if improvement > 0:
                    print("   ✅ Meta-DML outperforms standard DoubleML!")
                else:
                    print("   ⚠️  Meta-DML needs further tuning")
        
        # Rank all methods by plausibility
        method_rankings = sorted(
            [(name, cal.get('plausibility_rate', 0)*100) 
             for name, cal in calibration_results.items()],
            key=lambda x: x[1], reverse=True
        )
        
        print(f"\n📊 Method Rankings by Plausibility:")
        for i, (method, plaus_rate) in enumerate(method_rankings[:5]):
            rank_emoji = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else f"{i+1}."
            highlight = " 🚀" if method == "Meta-DML" else ""
            print(f"   {rank_emoji} {method}: {plaus_rate:.1f}%{highlight}")
        
        print("\nEnhanced analysis complete with Meta-DML!")
        print(f"Results saved to {output_dir}/")
        
    except Exception as e:
        print(f"Error in enhanced analysis: {str(e)}")
        import traceback
        traceback.print_exc()
        return

# ADD this helper function for Meta-DML specific analysis
def analyze_meta_dml_insights(benchmark_results, output_dir):
    """Analyze Meta-DML specific insights like learner weights and adaptability."""
    print("\n--- Analyzing Meta-DML Specific Insights ---")
    
    meta_dml_insights = []
    
    for indicator, indicator_results in benchmark_results.items():
        if indicator == 'calibration':  # Skip calibration results
            continue
            
        for policy_year, policy_results in indicator_results.items():
            if 'Meta-DML' in policy_results:
                meta_results = policy_results['Meta-DML']
                
                if isinstance(meta_results, dict) and 'meta_insights' in meta_results:
                    insights = meta_results['meta_insights']
                    weights = meta_results.get('meta_weights', {})
                    
                    meta_dml_insights.append({
                        'Indicator': indicator,
                        'Policy_Year': policy_year,
                        'Dominant_Learner': insights.get('dominant_learner', 'unknown'),
                        'Weight_Entropy': insights.get('weight_entropy', np.nan),
                        'N_Effective_Learners': insights.get('n_effective_learners', 0),
                        'RF_Weight': weights.get('rf', 0),
                        'GBM_Weight': weights.get('gbm', 0),
                        'MLP_Weight': weights.get('mlp', 0),
                        'Elastic_Weight': weights.get('elastic', 0),
                        'Ridge_Weight': weights.get('ridge', 0),
                        'Effect': meta_results.get('relative_effect', np.nan),
                        'Significant': meta_results.get('significance', False)
                    })
    
    if meta_dml_insights:
        insights_df = pd.DataFrame(meta_dml_insights)
        insights_df.to_csv(f'{output_dir}/meta_dml_insights.csv', index=False)
        
        # Print key insights
        print("🔍 Meta-DML Insights:")
        
        # Most frequently dominant learner
        dominant_counts = insights_df['Dominant_Learner'].value_counts()
        if not dominant_counts.empty:
            print(f"   Most dominant learner: {dominant_counts.index[0]} ({dominant_counts.iloc[0]} cases)")
        
        # Average weight entropy (higher = more diverse combination)
        avg_entropy = insights_df['Weight_Entropy'].mean()
        if not np.isnan(avg_entropy):
            print(f"   Average weight entropy: {avg_entropy:.2f} (higher = more diverse)")
        
        # Adaptability across indicators
        adaptability = insights_df.groupby('Indicator')['Dominant_Learner'].nunique().mean()
        print(f"   Adaptability score: {adaptability:.1f} (learner diversity across contexts)")
        
        return insights_df
    else:
        print("   No Meta-DML insights found")
        return None



def run_synthetic_test():
    """Run synthetic benchmark test to validate method performance against ground truth."""
    print("=== SYNTHETIC BENCHMARK TEST ===")
    print("Validating causal inference methods against known ground truth effects...")
    
    from src.benchmarks import run_synthetic_benchmark_test_v2
    
    try:
        # Run the comprehensive synthetic benchmark
        results = run_synthetic_benchmark_test_v2(output_dir='outputs/synthetic_benchmark/')
        
        print("\n=== SYNTHETIC TEST COMPLETED ===")
        print("Key outputs generated:")
        print("1. synthetic_benchmark_results.csv - All method estimates")
        print("2. ground_truth_effects.csv - Known true effects")
        print("3. accuracy_metrics.csv - Method accuracy comparison")
        
        # Print summary findings
        accuracy_df = results['accuracy_metrics']
        
        if not accuracy_df.empty:
            print("\n=== TOP PERFORMING METHODS ===")
            
            # Sort by MAE and show top 3
            top_methods = accuracy_df.sort_values('MAE').head(3)
            
            for i, (_, row) in enumerate(top_methods.iterrows()):
                rank = i + 1
                method = row['Method']
                mae = row['MAE']
                success_rate = row['Success_Rate']
                
                if not np.isnan(mae):
                    print(f"{rank}. {method}: MAE = {mae:.1f}pp, Success = {success_rate:.1f}%")
                else:
                    print(f"{rank}. {method}: No valid estimates")
            
            # Special analysis for BWSC
            bwsc_row = accuracy_df[accuracy_df['Method'] == 'BWSC']
            if not bwsc_row.empty:
                bwsc_mae = bwsc_row.iloc[0]['MAE']
                bwsc_success = bwsc_row.iloc[0]['Success_Rate']
                bwsc_rank = (accuracy_df.sort_values('MAE')['Method'] == 'BWSC').idxmax() + 1
                
                print(f"\n🎯 BWSC Performance:")
                if not np.isnan(bwsc_mae):
                    print(f"   Ranking: #{bwsc_rank} out of {len(accuracy_df)} methods")
                    print(f"   Mean Absolute Error: {bwsc_mae:.1f} percentage points")
                    print(f"   Success Rate: {bwsc_success:.1f}%")
                    
                    if bwsc_rank <= 3:
                        print("   ✅ BWSC ranks in TOP 3 for ground truth recovery!")
                    elif bwsc_rank <= len(accuracy_df) // 2:
                        print("   ✅ BWSC ranks in TOP HALF for ground truth recovery")
                    else:
                        print("   ⚠️  BWSC has room for improvement in accuracy")
                else:
                    print("   ⚠️  BWSC failed to generate valid estimates")
        
        print("\n📊 Use these results in your CIKM paper to demonstrate:")
        print("1. Ground truth validation of method accuracy")
        print("2. Quantitative comparison across all 9 methods")
        print("3. Evidence for BWSC's specialized performance characteristics")
        
        return results
        
    except Exception as e:
        print(f"Error in synthetic test: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

# ADD THIS TO THE main.py if __name__ == "__main__" block:

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "synthetic":
        # Run synthetic test
        run_synthetic_test()
    elif len(sys.argv) > 1 and sys.argv[1] == "real":
        # Run with real data
        run_enhanced_analysis()
    elif len(sys.argv) > 1 and sys.argv[1] == "meta-analysis":
        # NEW: Special analysis focusing on Meta-DML
        print("Running Meta-DML focused analysis...")
        run_enhanced_analysis()
        
        # Additional Meta-DML specific analysis
        try:
            from src.benchmarks import CausalInferenceEvaluator
            # Load results and analyze Meta-DML insights
            if benchmark_results is not None:
                analyze_meta_dml_insights(benchmark_results, 'outputs/enhanced_benchmark/')
            else:
                print("No benchmark results available for Meta-DML analysis")
        except Exception as e:
            print(f"Could not run Meta-DML specific analysis: {str(e)}")
    else:
        # Default: ask user
        choice = input("Run (1) Real data analysis, (2) Synthetic test, or (3) Meta-DML focused analysis? Enter 1, 2, or 3: ")
        if choice == "2":
            run_synthetic_test()
        elif choice == "3":
            print("Running Meta-DML focused analysis...")
            run_enhanced_analysis()
            analyze_meta_dml_insights(benchmark_results, 'outputs/enhanced_benchmark/')
        else:
            run_enhanced_analysis()