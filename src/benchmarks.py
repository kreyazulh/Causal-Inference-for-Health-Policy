"""
Change Point Detection and Causal Inference Comparative Analysis Framework

This module implements a focused comparative framework for evaluating the 
causally-informed wavelet analysis method against established benchmark methods.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ruptures as rpt
import warnings
import time
from typing import Dict, List, Tuple
import os

# Import the causally-informed wavelet method
from src.causal_wavelet import CausalWaveletAnalysis
from src.causal_inference import BayesianCausalImpactModel
from src.meta_dml import create_meta_dml_estimator
class ChangePointDetectionEvaluator:
    """
    Evaluator for change point detection methods.
    Compares methods and calculates performance metrics.
    """
    
    def __init__(self, time_series_data: pd.DataFrame, policy_timeline: Dict[str, str], 
                 tolerance: int = 2):
        """Initialize the evaluator."""
        self.data = time_series_data
        self.policy_timeline = policy_timeline
        self.tolerance = tolerance
        self.policy_years = np.array([int(year) for year in policy_timeline.keys()])
        self.years = np.array([int(year) for year in self.data.index])
        self.method_timings = {}
        
    def detect_benchmark_methods(self, indicator: str, method_name: str = 'all') -> Dict:
        """Run change point detection with benchmark methods."""
        # Get time series data
        ts = self.data[indicator].values
        years = self.years
        
        # Initialize results
        method_results = {}
        
        # Method 1: PELT (Pruned Exact Linear Time)
        if method_name in ['all', 'PELT']:
            try:
                start_time = time.time()
                algo = rpt.Pelt(model="rbf").fit(ts)
                result = algo.predict(pen=10)
                cps_years = [years[idx-1] for idx in result[:-1] if idx < len(years)]
                method_results['PELT'] = cps_years
                self.method_timings['PELT'] = time.time() - start_time
            except Exception as e:
                warnings.warn(f"Error with PELT method: {str(e)}")
        
        # Method 4: Binary Segmentation (best performer in benchmarks)
        if method_name in ['all', 'BinSeg']:
            try:
                start_time = time.time()
                algo = rpt.Binseg(model="l2").fit(ts)
                result = algo.predict(n_bkps=5)
                cps_years = [years[idx-1] for idx in result[:-1] if idx < len(years)]
                method_results['BinSeg'] = cps_years
                self.method_timings['BinSeg'] = time.time() - start_time
            except Exception as e:
                warnings.warn(f"Error with BinSeg method: {str(e)}")
        
        return method_results
    
    def detect_with_causal_wavelet(self, indicator: str) -> Tuple[List[int], List[int]]:
        """Run change point detection with the causally-informed wavelet method."""
        start_time = time.time()
        
        # Initialize a CausalWaveletAnalysis instance
        causal_wavelet = CausalWaveletAnalysis(self.data, self.policy_timeline)
        
        # Detect standard change points
        standard_cps = causal_wavelet.detect_standard_change_points(indicator)
        
        # Extract years
        standard_years = []
        for cp in standard_cps:
            year_idx = cp[0]
            if year_idx < len(self.years):
                standard_years.append(self.years[year_idx])
        
        # Detect causal change points
        causal_cps = causal_wavelet.detect_causal_change_points(indicator)
        
        # Extract years
        causal_years = []
        for cp in causal_cps:
            year_idx = cp[0]
            if year_idx < len(self.years):
                causal_years.append(self.years[year_idx])
        
        self.method_timings['StandardWavelet'] = time.time() - start_time
        self.method_timings['CausalWavelet'] = time.time() - start_time
        
        return standard_years, causal_years
    
    def measure_policy_alignment(self, standard_cps, causal_cps, policy_years):
        """Quantify the alignment of change points with known policy implementations."""
        policy_years = np.array(policy_years)
        
        # Filter out empty results
        if not standard_cps and not causal_cps:
            return {
                'mean_distance_standard': None,
                'mean_distance_causal': None,
                'improvement_percentage': None
            }
        
        # Calculate minimum distance to policy for each detected change point
        std_distances = np.array([min(abs(cp - policy_years)) for cp in standard_cps]) if standard_cps else np.array([])
        causal_distances = np.array([min(abs(cp - policy_years)) for cp in causal_cps]) if causal_cps else np.array([])
        
        # Calculate mean distances
        mean_std = np.mean(std_distances) if len(std_distances) > 0 else float('inf')
        mean_causal = np.mean(causal_distances) if len(causal_distances) > 0 else float('inf')
        
        # Calculate improvement percentage
        if mean_std > 0 and mean_std != float('inf') and mean_causal != float('inf'):
            improvement = ((mean_std - mean_causal) / mean_std) * 100
        else:
            improvement = None
        
        alignment = {
            'mean_distance_standard': mean_std,
            'mean_distance_causal': mean_causal,
            'improvement_percentage': improvement,
            'num_standard_cps': len(standard_cps),
            'num_causal_cps': len(causal_cps)
        }
        
        return alignment


class CausalInferenceEvaluator:
    """
    Evaluator for causal inference methods.
    Compares methods and calculates performance metrics.
    """
    
    def __init__(self, time_series_data: pd.DataFrame, policy_timeline: Dict[str, str]):
        """Initialize the evaluator."""
        self.data = time_series_data
        self.policy_timeline = policy_timeline
        self.policy_years = [int(year) for year in policy_timeline.keys()]
        self.method_timings = {}
    
    def estimate_with_interrupted_time_series(self, indicator: str, policy_year: int) -> Dict:
        """Estimate causal impact using interrupted time series analysis."""
        try:
            start_time = time.time()
            
            # Get indicator data
            ts_data = self.data[indicator].copy()
            years = np.array([int(year) for year in ts_data.index])
            
            # Create design matrix
            X = np.column_stack([
                np.ones(len(years)),  # Intercept
                years - years[0],     # Time trend (centered at first year)
                (years >= policy_year).astype(int),  # Level change
                (years >= policy_year).astype(int) * (years - policy_year)  # Slope change
            ])
            
            # Fit linear regression
            from sklearn.linear_model import LinearRegression
            model = LinearRegression().fit(X, ts_data.values)
            
            # Calculate predictions
            predictions = model.predict(X)
            
            # Create counterfactual predictions (no intervention)
            X_cf = X.copy()
            X_cf[:, 2] = 0  # No level change
            X_cf[:, 3] = 0  # No slope change
            counterfactual = model.predict(X_cf)
            
            # Calculate effect
            post_idx = np.where(years >= policy_year)[0]
            effect = ts_data.values[post_idx] - counterfactual[post_idx]
            
            # Calculate average effect and relative effect
            avg_effect = np.mean(effect)
            avg_counterfactual = np.mean(counterfactual[post_idx])
            relative_effect = (avg_effect / avg_counterfactual) * 100  # Convert to percentage
            
            # Calculate standard error
            residuals = ts_data.values - predictions
            std_error = np.std(residuals) / np.sqrt(len(post_idx))
            
            # Calculate confidence interval
            lower_bound = avg_effect - 1.96 * std_error
            upper_bound = avg_effect + 1.96 * std_error
            
            # Test significance
            t_stat = avg_effect / std_error
            significance = abs(t_stat) > 1.96  # 95% confidence
            
            results = {
                'point_effect': avg_effect,
                'relative_effect': relative_effect,
                'lower_bound': lower_bound,
                'upper_bound': upper_bound,
                'significance': significance
            }
            
            self.method_timings['ITS'] = time.time() - start_time
            
            return results
        
        except Exception as e:
            warnings.warn(f"Error with Interrupted Time Series method: {str(e)}")
            return {
                'point_effect': np.nan,
                'relative_effect': np.nan,
                'lower_bound': np.nan,
                'upper_bound': np.nan,
                'significance': False
            }
    
    def estimate_with_diff_in_diff(self, indicator: str, policy_year: int, 
                                  control_indicators: List[str]) -> Dict:
        """Estimate causal impact using difference-in-differences."""
        try:
            start_time = time.time()
            
            # Get treatment indicator data
            treatment_data = self.data[indicator].copy()
            years = np.array([int(year) for year in treatment_data.index])
            
            # Get control indicator data
            control_data = self.data[control_indicators].mean(axis=1).copy()
            
            # Create panel data
            panel_data = pd.DataFrame({
                'Year': np.concatenate([years, years]),
                'Indicator': np.concatenate([np.repeat(indicator, len(years)), 
                                           np.repeat('Control', len(years))]),
                'Value': np.concatenate([treatment_data.values, control_data.values]),
                'Treatment': np.concatenate([np.ones(len(years)), np.zeros(len(years))]),
                'Post': np.concatenate([years >= policy_year, years >= policy_year])
            })
            
            # Create interaction term
            panel_data['Treatment_Post'] = panel_data['Treatment'] * panel_data['Post']
            
            # Fit linear regression
            from sklearn.linear_model import LinearRegression
            X = panel_data[['Treatment', 'Post', 'Treatment_Post']]
            y = panel_data['Value']
            model = LinearRegression().fit(X, y)
            
            # Extract DiD coefficient (interaction term)
            did_effect = model.coef_[2]
            
            # Calculate relative effect
            pre_treatment_avg = panel_data.loc[(panel_data['Treatment'] == 1) & 
                                              (panel_data['Post'] == 0), 'Value'].mean()
            relative_effect = (did_effect / pre_treatment_avg) * 100  # Convert to percentage
            
            # Calculate standard error
            predictions = model.predict(X)
            residuals = y - predictions
            n = len(panel_data)
            p = 3  # Number of predictors
            std_error = np.sqrt(np.sum(residuals**2) / (n - p)) / np.sqrt(n)
            
            # Calculate confidence interval
            lower_bound = did_effect - 1.96 * std_error
            upper_bound = did_effect + 1.96 * std_error
            
            # Test significance
            t_stat = did_effect / std_error
            significance = abs(t_stat) > 1.96  # 95% confidence
            
            results = {
                'point_effect': did_effect,
                'relative_effect': relative_effect,
                'lower_bound': lower_bound,
                'upper_bound': upper_bound,
                'significance': significance
            }
            
            self.method_timings['DiD'] = time.time() - start_time
            
            return results
        
        except Exception as e:
            warnings.warn(f"Error with Diff-in-Diff method: {str(e)}")
            return {
                'point_effect': np.nan,
                'relative_effect': np.nan,
                'lower_bound': np.nan,
                'upper_bound': np.nan,
                'significance': False
            }
    
    def estimate_with_bayesian_causal_model(self, indicator: str, policy_year: int) -> Dict:
        """Estimate causal impact using our Bayesian causal impact model."""
        try:
            start_time = time.time()
            print(f"Running BayesianCausal for {indicator} at {policy_year}...")
            
            # Initialize model
            model = BayesianCausalImpactModel(self.data, self.policy_timeline)
            print("Model initialized successfully")
            
            # Fit model and estimate causal effect
            print("Fitting state space model...")
            model.fit_state_space_model(indicator, str(policy_year))
            print("State space model fitted")
            
            print("Estimating counterfactual...")
            model.estimate_counterfactual(indicator, str(policy_year))
            print("Counterfactual estimated")
            
            print("Computing causal effect...")
            effect = model.compute_causal_effect(indicator, str(policy_year))
            print("Causal effect computed")
            
            print("Testing significance...")
            test = model.test_significance(indicator, str(policy_year))
            print("Significance tested")
            
            # Extract results
            results = {
                'point_effect': effect['mean_abs_effect'],
                'relative_effect': effect['mean_rel_effect'],
                'lower_bound': effect['rel_effect_ci'][0],
                'upper_bound': effect['rel_effect_ci'][1],
                'significance': test['bayes_factor_10'] > 3,  # Moderate evidence
                'bayes_factor': test['bayes_factor_10']
            }
            
            self.method_timings['BayesianCausal'] = time.time() - start_time
            print(f"BayesianCausal completed in {self.method_timings['BayesianCausal']:.2f} seconds")
            
            return results
        
        except Exception as e:
            print(f"Error in BayesianCausal for {indicator} at {policy_year}: {str(e)}")
            import traceback
            traceback.print_exc()
            warnings.warn(f"Error with Bayesian Causal Model: {str(e)}")
            return {
                'point_effect': np.nan,
                'relative_effect': np.nan,
                'lower_bound': np.nan,
                'upper_bound': np.nan,
                'significance': False,
                'bayes_factor': np.nan
            }
        
    def estimate_with_synthetic_control(self, indicator: str, policy_year: int) -> Dict:
        """Estimate causal impact using Synthetic Control Method (SCM)."""
        try:
            start_time = time.time()
            
            # Get indicator data
            target_series = self.data[indicator].copy()
            years = np.array([int(year) for year in target_series.index])
            
            # Get donor pool (other indicators)
            donor_pool = self.data.drop(columns=[indicator])
            
            # Convert to numpy arrays
            pre_period = np.where(years < policy_year)[0]
            post_period = np.where(years >= policy_year)[0]
            
            if len(pre_period) < 3 or len(post_period) < 1:
                warnings.warn(f"Not enough pre/post periods for SCM on {indicator}")
                return self._empty_result()
            
            # Fit synthetic control
            from scipy import optimize
            
            # Function to minimize: sum of squared differences in pre-treatment period
            def objective_fn(weights):
                # Ensure weights sum to 1
                weights = weights / np.sum(weights)
                
                # Compute synthetic control
                synthetic_control = donor_pool.values @ weights
                
                # Compute pre-treatment fit
                pre_treatment_mse = np.mean((target_series.values[pre_period] - synthetic_control[pre_period])**2)
                
                return pre_treatment_mse
            
            # Initial weights
            initial_weights = np.ones(donor_pool.shape[1]) / donor_pool.shape[1]
            
            # Constraints: weights must be non-negative and sum to 1
            constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
            bounds = [(0, 1) for _ in range(donor_pool.shape[1])]
            
            # Optimize weights
            result = optimize.minimize(
                objective_fn, 
                initial_weights, 
                method='SLSQP', 
                bounds=bounds, 
                constraints=constraints
            )
            
            # Get optimal weights and normalize
            weights = result.x / np.sum(result.x)
            
            # Compute synthetic control with optimal weights
            synthetic_control = donor_pool.values @ weights
            
            # Compute effect
            effect = target_series.values[post_period] - synthetic_control[post_period]
            
            # Calculate average effect and relative effect
            avg_effect = np.mean(effect)
            avg_counterfactual = np.mean(synthetic_control[post_period])
            relative_effect = (avg_effect / avg_counterfactual) * 100
            
            # Calculate RMSPE for pre-period and post-period
            pre_rmspe = np.sqrt(np.mean((target_series.values[pre_period] - synthetic_control[pre_period])**2))
            post_rmspe = np.sqrt(np.mean((target_series.values[post_period] - synthetic_control[post_period])**2))
            
            # Ratio of post/pre RMSPE (for inference)
            rmspe_ratio = post_rmspe / pre_rmspe if pre_rmspe > 0 else np.inf
            
            # Simple significance test based on ratio
            significance = rmspe_ratio > 2  # Rule of thumb: ratio > 2 suggests significant effect
            
            # Calculate confidence interval (simplified)
            std_error = np.std(effect) / np.sqrt(len(post_period))
            lower_bound = avg_effect - 1.96 * std_error
            upper_bound = avg_effect + 1.96 * std_error
            
            results = {
                'point_effect': avg_effect,
                'relative_effect': relative_effect,
                'lower_bound': lower_bound,
                'upper_bound': upper_bound,
                'significance': significance,
                'pre_rmspe': pre_rmspe,
                'post_rmspe': post_rmspe,
                'rmspe_ratio': rmspe_ratio,
                'weights': dict(zip(donor_pool.columns, weights)),
                'synthetic_values': synthetic_control
            }
            
            self.method_timings['SCM'] = time.time() - start_time
            
            return results
        
        except Exception as e:
            warnings.warn(f"Error with Synthetic Control Method: {str(e)}")
            return self._empty_result()

    def estimate_with_augmented_scm(self, indicator: str, policy_year: int) -> Dict:
        """Estimate causal impact using Augmented Synthetic Control Method."""
        try:
            start_time = time.time()
            
            # Step 1: Run standard SCM
            scm_results = self.estimate_with_synthetic_control(indicator, policy_year)
            
            # Get indicator data
            target_series = self.data[indicator].copy()
            years = np.array([int(year) for year in target_series.index])
            
            # Define pre and post periods
            pre_period = np.where(years < policy_year)[0]
            post_period = np.where(years >= policy_year)[0]
            
            # Get SCM synthetic control
            scm_synthetic = scm_results.get('synthetic_values')
            
            if scm_synthetic is None:
                return scm_results  # Return regular SCM if it failed
            
            # Step 2: Calculate residuals in pre-treatment period
            scm_residuals_pre = target_series.values[pre_period] - scm_synthetic[pre_period]
            
            # Step 3: Fit a ridge regression model on residuals
            from sklearn.linear_model import Ridge
            
            # Create features matrix (time trend, squared time trend)
            time_features = np.column_stack([
                years[pre_period] - years[pre_period].min(),
                (years[pre_period] - years[pre_period].min())**2
            ])
            
            # Fit ridge regression model on pre-treatment residuals
            ridge_model = Ridge(alpha=1.0).fit(time_features, scm_residuals_pre)
            
            # Step 4: Predict residuals for post-treatment period
            post_time_features = np.column_stack([
                years[post_period] - years[pre_period].min(),
                (years[post_period] - years[pre_period].min())**2
            ])
            predicted_residuals_post = ridge_model.predict(post_time_features)
            
            # Step 5: Augment the synthetic control
            augmented_synthetic = np.copy(scm_synthetic)
            augmented_synthetic[post_period] += predicted_residuals_post
            
            # Compute effect for augmented SCM
            effect = target_series.values[post_period] - augmented_synthetic[post_period]
            
            # Calculate average effect and relative effect
            avg_effect = np.mean(effect)
            avg_counterfactual = np.mean(augmented_synthetic[post_period])
            relative_effect = (avg_effect / avg_counterfactual) * 100
            
            # Calculate RMSPE for pre-period and post-period with augmented synthetic
            pre_rmspe = np.sqrt(np.mean((target_series.values[pre_period] - augmented_synthetic[pre_period])**2))
            post_rmspe = np.sqrt(np.mean((target_series.values[post_period] - augmented_synthetic[post_period])**2))
            
            # Ratio of post/pre RMSPE (for inference)
            rmspe_ratio = post_rmspe / pre_rmspe if pre_rmspe > 0 else np.inf
            
            # Simple significance test
            significance = rmspe_ratio > 2
            
            # Calculate confidence interval (simplified)
            std_error = np.std(effect) / np.sqrt(len(post_period))
            lower_bound = avg_effect - 1.96 * std_error
            upper_bound = avg_effect + 1.96 * std_error
            
            results = {
                'point_effect': avg_effect,
                'relative_effect': relative_effect,
                'lower_bound': lower_bound,
                'upper_bound': upper_bound,
                'significance': significance,
                'pre_rmspe': pre_rmspe,
                'post_rmspe': post_rmspe,
                'rmspe_ratio': rmspe_ratio,
                'scm_weights': scm_results['weights'],
                'synthetic_values': augmented_synthetic
            }
            
            self.method_timings['ASCM'] = time.time() - start_time
            
            return results
        
        except Exception as e:
            warnings.warn(f"Error with Augmented Synthetic Control Method: {str(e)}")
            return self._empty_result()

    def estimate_with_causal_impact(self, indicator: str, policy_year: int) -> Dict:
        """Fixed CausalImpact using Bayesian structural time series."""
        try:
            start_time = time.time()
            
            # Get the data
            data = self.data[indicator].copy()
            years = np.array([int(year) for year in data.index])
            
            # Find policy intervention point
            policy_idx = np.where(years == policy_year)[0]
            if len(policy_idx) == 0:
                return self._empty_result()
            policy_idx = policy_idx[0]
            
            # Split into pre and post periods
            pre_data = data.iloc[:policy_idx]
            post_data = data.iloc[policy_idx:]
            
            if len(pre_data) < 5 or len(post_data) < 1:
                return self._empty_result()
            
            # Simple Bayesian structural time series using local level model
            from statsmodels.tsa.statespace.structural import UnobservedComponents
            
            # Fit model on pre-intervention data (no seasonality for yearly data)
            model = UnobservedComponents(
                pre_data.values,
                level='local level',  # Local level only
                irregular=True        # Allow for noise
            )
            
            fitted = model.fit(disp=False, method='powell')
            
            # Forecast post-intervention period
            n_periods = len(post_data)
            forecast = fitted.get_forecast(steps=n_periods)
            
            # Get predictions and confidence intervals
            predicted_mean = forecast.predicted_mean
            conf_int = forecast.conf_int()
            
            # Calculate effects
            actual_values = post_data.values
            predicted_values = predicted_mean.values if hasattr(predicted_mean, 'values') else predicted_mean
            
            # Ensure same length
            min_len = min(len(actual_values), len(predicted_values))
            actual_values = actual_values[:min_len]
            predicted_values = predicted_values[:min_len]
            
            # Calculate point effect
            point_effect = np.mean(actual_values - predicted_values)
            
            # Calculate relative effect
            baseline_mean = np.mean(pre_data.values)
            relative_effect = (point_effect / baseline_mean) * 100 if baseline_mean != 0 else 0
            
            # Calculate confidence intervals for relative effect
            if hasattr(conf_int, 'iloc'):
                ci_lower_abs = np.mean(conf_int.iloc[:min_len, 0])
                ci_upper_abs = np.mean(conf_int.iloc[:min_len, 1])
            else:
                ci_lower_abs = np.mean(predicted_values) - 1.96 * np.std(predicted_values)
                ci_upper_abs = np.mean(predicted_values) + 1.96 * np.std(predicted_values)
            
            # Convert to relative confidence intervals
            effect_lower = np.mean(actual_values) - ci_upper_abs
            effect_upper = np.mean(actual_values) - ci_lower_abs
            
            rel_lower = (effect_lower / baseline_mean) * 100 if baseline_mean != 0 else 0
            rel_upper = (effect_upper / baseline_mean) * 100 if baseline_mean != 0 else 0
            
            # Test significance (CI doesn't include zero)
            significance = not (rel_lower <= 0 <= rel_upper)
            
            self.method_timings['CausalImpact'] = time.time() - start_time
            
            return {
                'point_effect': point_effect,
                'relative_effect': relative_effect,
                'lower_bound': rel_lower,
                'upper_bound': rel_upper,
                'significance': significance
            }
            
        except Exception as e:
            print(f"Error in CausalImpact: {str(e)}")
            import traceback
            traceback.print_exc()
            return self._empty_result()

    def estimate_with_granger_causality(self, indicator: str, policy_year: int) -> Dict:
        """Classic Granger causality test with policy intervention."""
        try:
            from statsmodels.tsa.stattools import grangercausalitytests
            from statsmodels.tsa.arima.model import ARIMA
            
            # Create intervention dummy variable
            years = np.array([int(y) for y in self.data.index])
            intervention = (years >= policy_year).astype(int)
            
            # Test causality from intervention to indicator
            test_data = np.column_stack([self.data[indicator].values, intervention])
            
            # Granger test (intervention -> indicator)
            result = grangercausalitytests(test_data, maxlag=4, verbose=False)
            
            # Extract p-value from lag 1
            p_value = result[1][0]['ssr_ftest'][1]
            
            # Estimate effect size using ARIMA with intervention
            arima_model = ARIMA(self.data[indicator], exog=intervention, order=(1,1,1))
            fitted = arima_model.fit()
            
            intervention_coeff = fitted.params[-1]  # Last parameter is intervention effect
            baseline_mean = self.data[indicator].loc[:str(policy_year-1)].mean()
            relative_effect = (intervention_coeff / baseline_mean) * 100
            
            return {
                'point_effect': intervention_coeff,
                'relative_effect': relative_effect,
                'lower_bound': relative_effect - 10,  # Simplified CI
                'upper_bound': relative_effect + 10,
                'significance': p_value < 0.05,
                'p_value': p_value
            }
        except Exception as e:
            return self._empty_result()
        
    def estimate_with_causal_forests(self, indicator: str, policy_year: int) -> Dict:
        """Estimate causal impact using Causal Forests."""
        try:
            start_time = time.time()
            
            # Prepare data
            years = np.array([int(y) for y in self.data.index])
            treatment = (years >= policy_year).astype(int)
            outcome = self.data[indicator].values
            
            # Create features (time trends, lagged values)
            features = []
            base_features = [
                years - years.min(),  # Time trend
                (years - years.min())**2,  # Quadratic trend
            ]
            
            # Add lagged outcome if enough data
            if len(outcome) > 2:
                lag1 = np.concatenate([[outcome[0]], outcome[:-1]])
                base_features.append(lag1)
                
            # Add control variables from other indicators
            other_indicators = [col for col in self.data.columns if col != indicator][:3]
            if other_indicators:
                for other_ind in other_indicators:
                    base_features.append(self.data[other_ind].values)
            
            X = np.column_stack(base_features)
            
            # Causal Forest implementation using Random Forest
            from sklearn.ensemble import RandomForestRegressor
            from sklearn.model_selection import train_test_split
            
            # Split data for honest estimation
            if len(X) < 20:
                # Too small for splitting, use simple approach
                forest = RandomForestRegressor(n_estimators=100, random_state=42)
                
                # Fit separate models for treated and control
                treated_mask = treatment == 1
                control_mask = treatment == 0
                
                if np.sum(treated_mask) > 3 and np.sum(control_mask) > 3:
                    forest_treated = RandomForestRegressor(n_estimators=50, random_state=42)
                    forest_control = RandomForestRegressor(n_estimators=50, random_state=42)
                    
                    forest_treated.fit(X[treated_mask], outcome[treated_mask])
                    forest_control.fit(X[control_mask], outcome[control_mask])
                    
                    # Predict treatment effects
                    mu1 = forest_treated.predict(X)
                    mu0 = forest_control.predict(X)
                    treatment_effects = mu1 - mu0
                    
                else:
                    # Fallback to simple difference
                    if np.sum(treated_mask) > 0 and np.sum(control_mask) > 0:
                        treatment_effects = np.full(len(X), 
                                                  np.mean(outcome[treated_mask]) - np.mean(outcome[control_mask]))
                    else:
                        return self._empty_result()
            else:
                # Honest splitting approach
                X_train, X_test, y_train, y_test, t_train, t_test = train_test_split(
                    X, outcome, treatment, test_size=0.5, random_state=42
                )
                
                # Fit separate forests on training data
                treated_train = t_train == 1
                control_train = t_train == 0
                
                if np.sum(treated_train) > 2 and np.sum(control_train) > 2:
                    forest_treated = RandomForestRegressor(n_estimators=100, random_state=42)
                    forest_control = RandomForestRegressor(n_estimators=100, random_state=42)
                    
                    forest_treated.fit(X_train[treated_train], y_train[treated_train])
                    forest_control.fit(X_train[control_train], y_train[control_train])
                    
                    # Predict on test set
                    mu1_test = forest_treated.predict(X_test)
                    mu0_test = forest_control.predict(X_test)
                    treatment_effects = mu1_test - mu0_test
                else:
                    return self._empty_result()
            
            # Calculate average treatment effect
            ate = np.mean(treatment_effects)
            
            # Calculate relative effect
            baseline = np.mean(outcome[treatment == 0]) if np.sum(treatment == 0) > 0 else np.mean(outcome)
            relative_effect = (ate / baseline) * 100 if baseline != 0 else 0
            
            # Bootstrap confidence intervals
            n_bootstrap = 100
            bootstrap_effects = []
            
            for _ in range(n_bootstrap):
                # Bootstrap sample
                idx = np.random.choice(len(treatment_effects), len(treatment_effects), replace=True)
                boot_ate = np.mean(treatment_effects[idx])
                boot_rel = (boot_ate / baseline) * 100 if baseline != 0 else 0
                bootstrap_effects.append(boot_rel)
            
            ci_lower, ci_upper = np.percentile(bootstrap_effects, [2.5, 97.5])
            significance = not (ci_lower <= 0 <= ci_upper)
            
            self.method_timings['CausalForests'] = time.time() - start_time
            
            return {
                'point_effect': ate,
                'relative_effect': relative_effect,
                'lower_bound': ci_lower,
                'upper_bound': ci_upper,
                'significance': significance
            }
            
        except Exception as e:
            warnings.warn(f"Error with Causal Forests: {str(e)}")
            return self._empty_result()

    def estimate_with_bart(self, indicator: str, policy_year: int) -> Dict:
        """Estimate causal impact using BART (Bayesian Additive Regression Trees)."""
        try:
            start_time = time.time()
            
            # Prepare data
            years = np.array([int(y) for y in self.data.index])
            treatment = (years >= policy_year).astype(int)
            outcome = self.data[indicator].values
            
            # Create features
            features = []
            base_features = [
                years - years.min(),  # Time trend
                (years - years.min())**2,  # Quadratic trend
            ]
            
            # Add lagged outcome
            if len(outcome) > 2:
                lag1 = np.concatenate([[outcome[0]], outcome[:-1]])
                base_features.append(lag1)
            
            # Add other indicators as confounders
            other_indicators = [col for col in self.data.columns if col != indicator][:2]
            if other_indicators:
                for other_ind in other_indicators:
                    base_features.append(self.data[other_ind].values)
            
            X = np.column_stack(base_features)
            
            # Simplified BART implementation using ensemble of decision trees with Bayesian updates
            from sklearn.tree import DecisionTreeRegressor
            from sklearn.ensemble import BaggingRegressor
            
            # Bayesian ensemble approach (simplified BART)
            n_trees = 50
            trees = []
            predictions = np.zeros((n_trees, len(outcome)))
            
            # Bootstrap aggregating with different random states (approximates BART sampling)
            for i in range(n_trees):
                # Create bootstrap sample
                idx = np.random.choice(len(X), len(X), replace=True)
                X_boot = X[idx]
                y_boot = outcome[idx]
                t_boot = treatment[idx]
                
                # Fit tree with regularization (mimics BART priors)
                tree = DecisionTreeRegressor(
                    max_depth=3,  # Shallow trees like BART
                    min_samples_leaf=5,  # Regularization
                    random_state=i
                )
                
                # Include treatment in features
                X_with_treatment = np.column_stack([X_boot, t_boot])
                tree.fit(X_with_treatment, y_boot)
                trees.append(tree)
                
                # Predict for all observations
                X_all_with_treatment = np.column_stack([X, treatment])
                predictions[i] = tree.predict(X_all_with_treatment)
            
            # BART posterior samples (average across trees)
            posterior_mean = np.mean(predictions, axis=0)
            
            # Estimate treatment effects
            # Predict with treatment = 1 for all
            X_treated = np.column_stack([X, np.ones(len(X))])
            # Predict with treatment = 0 for all  
            X_control = np.column_stack([X, np.zeros(len(X))])
            
            treated_predictions = np.zeros((n_trees, len(X)))
            control_predictions = np.zeros((n_trees, len(X)))
            
            for i, tree in enumerate(trees):
                treated_predictions[i] = tree.predict(X_treated)
                control_predictions[i] = tree.predict(X_control)
            
            # Treatment effects from posterior
            treatment_effects = np.mean(treated_predictions - control_predictions, axis=0)
            ate = np.mean(treatment_effects)
            
            # Calculate relative effect
            baseline = np.mean(outcome[treatment == 0]) if np.sum(treatment == 0) > 0 else np.mean(outcome)
            relative_effect = (ate / baseline) * 100 if baseline != 0 else 0
            
            # Posterior credible intervals
            posterior_ates = np.mean(treated_predictions - control_predictions, axis=1)
            posterior_rel_effects = (posterior_ates / baseline) * 100 if baseline != 0 else posterior_ates
            
            ci_lower, ci_upper = np.percentile(posterior_rel_effects, [2.5, 97.5])
            significance = not (ci_lower <= 0 <= ci_upper)
            
            self.method_timings['BART'] = time.time() - start_time
            
            return {
                'point_effect': ate,
                'relative_effect': relative_effect,
                'lower_bound': ci_lower,
                'upper_bound': ci_upper,
                'significance': significance
            }
            
        except Exception as e:
            warnings.warn(f"Error with BART: {str(e)}")
            return self._empty_result()

    def estimate_with_psm(self, indicator: str, policy_year: int) -> Dict:
        """Estimate causal impact using Propensity Score Matching."""
        try:
            start_time = time.time()
            
            # Prepare data
            years = np.array([int(y) for y in self.data.index])
            treatment = (years >= policy_year).astype(int)
            outcome = self.data[indicator].values
            
            # Check if we have both treated and control units
            n_treated = np.sum(treatment)
            n_control = np.sum(1 - treatment)
            
            if n_treated < 2 or n_control < 2:
                print(f"PSM: Insufficient variation - {n_treated} treated, {n_control} control")
                return self._empty_result()
            
            # Create covariates for propensity score (simplified for time series)
            base_covariates = [
                years - years.min(),  # Time trend
                np.log(years - years.min() + 1),  # Log time trend
            ]
            
            # Add lagged outcome if available
            if len(outcome) > 2:
                lag1 = np.concatenate([[outcome[0]], outcome[:-1]])
                base_covariates.append(lag1)
            
            # Add one other indicator as confounder
            other_indicators = [col for col in self.data.columns if col != indicator]
            if other_indicators:
                base_covariates.append(self.data[other_indicators[0]].values)
            
            X = np.column_stack(base_covariates)
            
            # Standardize features for better propensity score estimation
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Step 1: Estimate propensity scores with regularization
            from sklearn.linear_model import LogisticRegression
            
            ps_model = LogisticRegression(
                random_state=42, 
                max_iter=1000,
                penalty='l2',
                C=1.0  # Regularization
            )
            ps_model.fit(X_scaled, treatment)
            propensity_scores = ps_model.predict_proba(X_scaled)[:, 1]
            
            # Check propensity score overlap
            treated_ps = propensity_scores[treatment == 1]
            control_ps = propensity_scores[treatment == 0]
            
            # Adaptive caliper based on data
            ps_std = np.std(propensity_scores)
            caliper = min(0.5, 0.25 * ps_std)  # More flexible caliper
            
            print(f"PSM: PS range treated [{treated_ps.min():.3f}, {treated_ps.max():.3f}], "
                  f"control [{control_ps.min():.3f}, {control_ps.max():.3f}], caliper={caliper:.3f}")
            
            # Step 2: Greedy nearest neighbor matching with replacement
            treated_indices = np.where(treatment == 1)[0]
            control_indices = np.where(treatment == 0)[0]
            
            matched_pairs = []
            used_controls = set()  # Track without replacement initially
            
            # Sort treated units by propensity score for better matching
            treated_ps_sorted = sorted(zip(treated_indices, treated_ps), key=lambda x: x[1])
            
            for treated_idx, treated_ps_val in treated_ps_sorted:
                # Find closest available control unit
                available_controls = [idx for idx in control_indices if idx not in used_controls]
                
                if not available_controls:
                    # Allow replacement if necessary
                    available_controls = control_indices
                
                distances = np.abs(propensity_scores[available_controls] - treated_ps_val)
                best_match_pos = np.argmin(distances)
                closest_control_idx = available_controls[best_match_pos]
                
                # Accept match if within caliper OR if it's the best available
                if distances[best_match_pos] < caliper or len(matched_pairs) < max(2, n_treated // 3):
                    matched_pairs.append((treated_idx, closest_control_idx))
                    used_controls.add(closest_control_idx)
            
            print(f"PSM: Found {len(matched_pairs)} matches out of {n_treated} treated units")
            
            if len(matched_pairs) < 2:
                return self._empty_result()
            
            # Step 3: Calculate treatment effect on matched sample
            treated_outcomes = outcome[[pair[0] for pair in matched_pairs]]
            control_outcomes = outcome[[pair[1] for pair in matched_pairs]]
            
            # Average treatment effect on treated (ATT)
            att = np.mean(treated_outcomes - control_outcomes)
            
            # Calculate relative effect
            baseline = np.mean(control_outcomes)
            relative_effect = (att / baseline) * 100 if baseline != 0 else 0
            
            # Bootstrap confidence intervals
            n_bootstrap = 50  # Reduced for speed
            bootstrap_effects = []
            
            for _ in range(n_bootstrap):
                # Bootstrap matched pairs
                boot_idx = np.random.choice(len(matched_pairs), len(matched_pairs), replace=True)
                boot_treated = treated_outcomes[boot_idx]
                boot_control = control_outcomes[boot_idx]
                
                boot_att = np.mean(boot_treated - boot_control)
                boot_baseline = np.mean(boot_control)
                boot_rel = (boot_att / boot_baseline) * 100 if boot_baseline != 0 else 0
                bootstrap_effects.append(boot_rel)
            
            ci_lower, ci_upper = np.percentile(bootstrap_effects, [2.5, 97.5])
            significance = not (ci_lower <= 0 <= ci_upper)
            
            # Calculate match quality
            match_quality = np.mean([np.abs(propensity_scores[t] - propensity_scores[c]) 
                                   for t, c in matched_pairs])
            
            self.method_timings['PSM'] = time.time() - start_time
            
            return {
                'point_effect': att,
                'relative_effect': relative_effect,
                'lower_bound': ci_lower,
                'upper_bound': ci_upper,
                'significance': significance,
                'n_matches': len(matched_pairs),
                'match_quality': match_quality
            }
            
        except Exception as e:
            print(f"PSM Error for {indicator} at {policy_year}: {str(e)}")
            warnings.warn(f"Error with PSM: {str(e)}")
            return self._empty_result()

    def estimate_with_double_ml(self, indicator: str, policy_year: int) -> Dict:
        """Double Machine Learning for causal inference."""
        try:
            start_time = time.time()
            from sklearn.ensemble import RandomForestRegressor
            from sklearn.model_selection import cross_val_predict
            
            # Prepare features and treatment
            years = np.array([int(y) for y in self.data.index])
            treatment = (years >= policy_year).astype(int)
            
            # Create feature matrix (lagged values, trends)
            features = np.column_stack([
                years - years.min(),  # Time trend
                np.roll(self.data[indicator].values, 1),  # Lagged outcome
            ])[1:]  # Remove first row due to lag
            
            outcome = self.data[indicator].values[1:]
            treatment = treatment[1:]
            
            # Step 1: Predict outcome using features (residualize)
            outcome_model = RandomForestRegressor(n_estimators=100, random_state=42)
            outcome_pred = cross_val_predict(outcome_model, features, outcome, cv=5)
            outcome_residual = outcome - outcome_pred
            
            # Step 2: Predict treatment using features (residualize)  
            treatment_model = RandomForestRegressor(n_estimators=100, random_state=42)
            treatment_pred = cross_val_predict(treatment_model, features, treatment, cv=5)
            treatment_residual = treatment - treatment_pred
            
            # Step 3: Regress residualized outcome on residualized treatment
            treatment_effect = np.cov(outcome_residual, treatment_residual)[0,1] / np.var(treatment_residual)
            
            baseline_mean = np.mean(outcome[treatment == 0])
            relative_effect = (treatment_effect / baseline_mean) * 100

            self.method_timings['DoubleML'] = time.time() - start_time
            
            # Simple bootstrap CI
            n_bootstrap = 100
            bootstrap_effects = []
            for _ in range(n_bootstrap):
                idx = np.random.choice(len(outcome_residual), len(outcome_residual), replace=True)
                boot_effect = np.cov(outcome_residual[idx], treatment_residual[idx])[0,1] / np.var(treatment_residual[idx])
                bootstrap_effects.append((boot_effect / baseline_mean) * 100)
            
            ci_lower, ci_upper = np.percentile(bootstrap_effects, [2.5, 97.5])
            
            return {
                'point_effect': treatment_effect,
                'relative_effect': relative_effect,
                'lower_bound': ci_lower,
                'upper_bound': ci_upper,
                'significance': not (ci_lower <= 0 <= ci_upper)
            }
        except Exception as e:
            return self._empty_result()
        
        # ADD this new method for Meta-DML
    def estimate_with_meta_dml(self, indicator: str, policy_year: int) -> Dict:
        """Estimate causal impact using Meta-Learning Double ML."""
        try:
            start_time = time.time()
            
            # Prepare features
            years = np.array([int(y) for y in self.data.index])
            treatment = (years >= policy_year).astype(int)
            
            # Enhanced feature matrix with more informative features
            outcome_values = self.data[indicator].values
            
            # Create feature matrix with lags and trends
            features = []
            base_features = [
                years - years.min(),  # Time trend
                (years - years.min())**2,  # Quadratic trend
            ]
            
            # Add lagged outcomes (if we have enough data)
            if len(outcome_values) > 2:
                lag1 = np.concatenate([[outcome_values[0]], outcome_values[:-1]])  # Lag 1
                base_features.append(lag1)
                
            if len(outcome_values) > 4:
                lag2 = np.concatenate([[outcome_values[0], outcome_values[1]], outcome_values[:-2]])  # Lag 2
                base_features.append(lag2) 
                
            # Add moving averages (if we have enough data)
            if len(outcome_values) > 5:
                ma3 = pd.Series(outcome_values).rolling(window=3, min_periods=1).mean().values
                base_features.append(ma3)
            
            # Stack features
            features_matrix = np.column_stack(base_features)
            
            # Remove any rows with NaN (though our construction should avoid this)
            valid_mask = ~np.isnan(features_matrix).any(axis=1)
            features_clean = features_matrix[valid_mask]
            outcome_clean = outcome_values[valid_mask]
            treatment_clean = treatment[valid_mask]
            
            # Ensure we have enough data
            if len(features_clean) < 10:
                warnings.warn(f"Insufficient data for Meta-DML on {indicator}")
                return self._empty_result()
            
            # Create and configure Meta-DML estimator
            meta_dml = create_meta_dml_estimator(
                enhanced_features=True,  # Use enhanced base learners
                use_neural_meta=True     # Try neural meta-learning first
            )
            
            # Estimate treatment effect
            results = meta_dml.estimate_treatment_effect(
                features_clean, 
                outcome_clean, 
                treatment_clean
            )
            
            # Record timing
            self.method_timings['Meta-DML'] = time.time() - start_time
            
            # Add some Meta-DML specific insights to results
            results['meta_insights'] = {
                'dominant_learner': max(results['meta_weights'].items(), key=lambda x: x[1])[0],
                'weight_entropy': -sum(w * np.log(w + 1e-8) for w in results['meta_weights'].values()),
                'n_effective_learners': sum(1 for w in results['meta_weights'].values() if w > 0.1)
            }
            
            return results
            
        except Exception as e:
            warnings.warn(f"Error with Meta-DML method: {str(e)}")
            print(f"Meta-DML error details: {str(e)}")  # For debugging
            return self._empty_result()


    def estimate_with_bayesian_wavelet_synthetic(self, indicator: str, policy_year: int) -> Dict:
        """Estimate causal impact using the improved BWSC method."""
        try:
            start_time = time.time()
            
            from src.causal_inference import BayesianWaveletSyntheticControl
            model = BayesianWaveletSyntheticControl(self.data, self.policy_timeline)
            
            model.fit_hybrid_model(indicator, str(policy_year))
            effect = model.compute_causal_effect(indicator, str(policy_year))
            
            results = {
                'point_effect': effect['mean_abs_effect'],
                'relative_effect': effect['mean_rel_effect'],
                'lower_bound': effect['rel_effect_ci'][0],
                'upper_bound': effect['rel_effect_ci'][1],
                'significance': effect['significance'],  # This should now work!
                'energy_ratio': effect['energy_ratio'],
                'bayesian_weight': effect['bayesian_weight'],
                'synthetic_weight': effect['synthetic_weight'],
                'significance_score': effect.get('significance_score', 0),
                'standardized_effect': effect.get('standardized_effect', 0)
            }
            
            self.method_timings['BWSC'] = time.time() - start_time
            return results
            
        except Exception as e:
            warnings.warn(f"Error with BWSC method: {str(e)}")
            return self._empty_result()

    def _empty_result(self):
        """Helper method to return empty results when methods fail."""
        return {
            'point_effect': np.nan,
            'relative_effect': np.nan,
            'lower_bound': np.nan,
            'upper_bound': np.nan,
            'significance': False
        }

    def run_comparative_analysis(self, indicators: List[str] = None, 
                            policy_years: List[int] = None) -> Dict:
        """Run a focused comparative analysis across indicators and policies."""
        if indicators is None:
            indicators = self.data.columns
            
        if policy_years is None:
            policy_years = self.policy_years
        
        # Initialize results container
        comparative_results = {}
        
        for indicator in indicators:
            print(f"Running causal comparative analysis for {indicator}...")
            
            # Results for this indicator
            indicator_results = {}
            
            for policy_year in policy_years:
                print(f"  Analyzing policy year {policy_year}...")
                
                # Results for this policy
                policy_results = {}
                
                # Method 1: Interrupted Time Series
                its_results = self.estimate_with_interrupted_time_series(indicator, policy_year)
                policy_results['ITS'] = its_results
                
                # Method 2: Difference-in-Differences
                control_indicators = [ind for ind in self.data.columns if ind != indicator][:3]
                
                if len(control_indicators) > 0:
                    did_results = self.estimate_with_diff_in_diff(indicator, policy_year, control_indicators)
                    policy_results['DiD'] = did_results
                
                # Method 3: Bayesian Causal Model (final run er time e comment out)
                # bayesian_results = self.estimate_with_bayesian_causal_model(indicator, policy_year)
                # policy_results['BayesianCausal'] = bayesian_results
                
                # Method 4: Synthetic Control
                scm_results = self.estimate_with_synthetic_control(indicator, policy_year)
                policy_results['SCM'] = scm_results
                
                # Method 5: Augmented Synthetic Control
                ascm_results = self.estimate_with_augmented_scm(indicator, policy_year)
                policy_results['ASCM'] = ascm_results
                

                # Method 7: CausalImpact
                causal_impact_results = self.estimate_with_causal_impact(indicator, policy_year)
                policy_results['CausalImpact'] = causal_impact_results


                # Method 9: Causal Forests
                causal_forests_results = self.estimate_with_causal_forests(indicator, policy_year)
                policy_results['CausalForests'] = causal_forests_results

                # Method 10: BART
                bart_results = self.estimate_with_bart(indicator, policy_year)
                policy_results['BART'] = bart_results

                # Method 11: PSM
                psm_results = self.estimate_with_psm(indicator, policy_year)
                policy_results['PSM'] = psm_results

                # Method 9: Double ML
                dml_results = self.estimate_with_double_ml(indicator, policy_year)
                policy_results['DoubleML'] = dml_results

                               # Method 10: NEW - Meta-Learning Double ML
                print(f"    Running Meta-DML...")
                meta_dml_results = self.estimate_with_meta_dml(indicator, policy_year)
                policy_results['Meta-DML'] = meta_dml_results
                print(f"    Meta-DML completed")
                
                # Store results for this policy
                indicator_results[policy_year] = policy_results
            
            # Store results for this indicator
            comparative_results[indicator] = indicator_results
        
        return comparative_results
    
    
    def evaluate_calibration(self, results: Dict) -> Dict:
        """Evaluate calibration of causal effect estimates against domain constraints."""
        # Domain knowledge constraints (realistic bounds based on public health literature)
        # Domain knowledge constraints (realistic bounds based on systematic literature review)
        domain_constraints = {
            # CORRECTED ORIGINAL CONSTRAINTS
            'Mortality rate, infant (per 1,000 live births)': (-90, 10),  # Supported: 90% tetanus reduction
            'Life expectancy at birth, total (years)': (-10, 15),         # CORRECTED: Literature shows max 9.5% improvement
            'Maternal mortality ratio (modeled estimate, per 100,000 live births)': (-80, 10),  # CORRECTED: Max 75% reduction documented
            'Immunization, measles (% of children ages 12-23 months)': (-20, 100),  # Supported: 74% increase documented
            
            # NEW EVIDENCE-BASED CONSTRAINTS
            'Prevalence of undernourishment (% of population)': (-70, 15),
            'Mortality rate, under-5 (per 1,000 live births)': (-85, 10), 
            'Incidence of tuberculosis (per 100,000 people)': (-80, 20),
            'Hospital beds (per 1,000 people)': (-50, 200)
        }
        
        # Initialize tracking containers
        calibration_scores = {}
        implausible_counts = {}
        effect_sizes = {}
        magnitude_errors = {}  # Track how far outside constraints
        
        # Extract all effects for all methods
        for indicator, indicator_results in results.items():
            for policy_year, policy_results in indicator_results.items():
                for method, method_results in policy_results.items():
                    # Initialize method trackers if needed
                    if method not in implausible_counts:
                        implausible_counts[method] = 0
                        effect_sizes[method] = []
                        magnitude_errors[method] = []
                    
                    effect = method_results['relative_effect']
                    
                    # Skip NaN effects
                    if np.isnan(effect):
                        continue
                        
                    effect_sizes[method].append(effect)
                    
                    # Check if effect is within plausible range
                    if indicator in domain_constraints:
                        min_val, max_val = domain_constraints[indicator]
                        if effect < min_val or effect > max_val:
                            implausible_counts[method] += 1
                            
                            # Calculate magnitude of error
                            if effect < min_val:
                                error = min_val - effect
                            else:
                                error = effect - max_val
                            magnitude_errors[method].append(error)
        
        # Calculate calibration metrics for each method
        for method in effect_sizes:
            total_estimates = len(effect_sizes[method])
            if total_estimates > 0:
                # Calculate plausibility rate
                plausibility_rate = 1 - (implausible_counts[method] / total_estimates)
                
                # Calculate effect statistics
                mean_abs_effect = np.mean(np.abs(effect_sizes[method]))
                effect_variance = np.var(effect_sizes[method])
                
                # Calculate average magnitude of constraint violation
                avg_violation = np.mean(magnitude_errors[method]) if implausible_counts[method] > 0 else 0
                
                # Store all metrics
                calibration_scores[method] = {
                    'plausibility_rate': plausibility_rate,
                    'mean_abs_effect': mean_abs_effect,
                    'effect_variance': effect_variance,
                    'avg_violation': avg_violation,
                    'max_effect': np.max(np.abs(effect_sizes[method])) if total_estimates > 0 else 0,
                    'min_effect': np.min(np.abs(effect_sizes[method])) if total_estimates > 0 else 0,
                    'total_estimates': total_estimates
                }
        
        return calibration_scores


