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

    def estimate_with_double_ml(self, indicator: str, policy_year: int) -> Dict:
        """Double Machine Learning for causal inference."""
        try:
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
            self.method_timings['meta_dml'] = time.time() - start_time
            
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
                
                # Method 3: Bayesian Causal Model
                bayesian_results = self.estimate_with_bayesian_causal_model(indicator, policy_year)
                policy_results['BayesianCausal'] = bayesian_results
                
                # Method 4: Synthetic Control
                scm_results = self.estimate_with_synthetic_control(indicator, policy_year)
                policy_results['SCM'] = scm_results
                
                # Method 5: Augmented Synthetic Control
                ascm_results = self.estimate_with_augmented_scm(indicator, policy_year)
                policy_results['ASCM'] = ascm_results
                
                # Method 6: Novel BayesianWaveletSyntheticControl
                bwsc_results = self.estimate_with_bayesian_wavelet_synthetic(indicator, policy_year)
                policy_results['BWSC'] = bwsc_results

                # Method 7: CausalImpact
                causal_impact_results = self.estimate_with_causal_impact(indicator, policy_year)
                policy_results['CausalImpact'] = causal_impact_results

                # Method 8: Granger Causality  
                granger_results = self.estimate_with_granger_causality(indicator, policy_year)
                policy_results['Granger'] = granger_results

                # Method 9: Double ML
                dml_results = self.estimate_with_double_ml(indicator, policy_year)
                policy_results['DoubleML'] = dml_results

                               # Method 10: NEW - Meta-Learning Double ML
                print(f"    Running Meta-DML...")
                meta_dml_results = self.estimate_with_meta_dml(indicator, policy_year)
                policy_results['meta_dml'] = meta_dml_results
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


# Standalone functions




def run_focused_benchmark(df_timeseries, policy_timeline, indicators=None, policy_years=None, output_dir='outputs/benchmark/'):
    """Run focused benchmark comparing methods for CIKM paper."""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Select indicators if not provided
    if indicators is None:
        indicators = [
            'Mortality rate, infant (per 1,000 live births)',
            'Life expectancy at birth, total (years)',
            'Maternal mortality ratio (modeled estimate, per 100,000 live births)',
            'Immunization, measles (% of children ages 12-23 months)'
        ]
        indicators = [ind for ind in indicators if ind in df_timeseries.columns]
    
    # Select policy years if not provided
    if policy_years is None:
        policy_years = [1982, 1998, 2011]
    
    # 1. Focused change point detection benchmark
    print("Running focused change point detection benchmark...")
    cp_evaluator = ChangePointDetectionEvaluator(df_timeseries, policy_timeline)
    
    cp_alignment_results = {}
    for indicator in indicators:
        # Get BinSeg results (best baseline)
        binseg_cps = cp_evaluator.detect_benchmark_methods(indicator, method_name='BinSeg')['BinSeg']
        
        # Get wavelet results
        standard_cps, causal_cps = cp_evaluator.detect_with_causal_wavelet(indicator)
        
        # Measure policy alignment
        alignment = cp_evaluator.measure_policy_alignment(standard_cps, causal_cps, policy_years)
        
        # Measure BinSeg vs CausalWavelet
        binseg_vs_causal = cp_evaluator.measure_policy_alignment(binseg_cps, causal_cps, policy_years)
        
        cp_alignment_results[indicator] = {
            'standard_vs_causal': alignment,
            'binseg_vs_causal': binseg_vs_causal
        }
    
    # 2. Focused causal inference benchmark
    print("Running focused causal inference benchmark...")
    ci_evaluator = CausalInferenceEvaluator(df_timeseries, policy_timeline)
    
    # Run comparative analysis
    ci_results = ci_evaluator.run_comparative_analysis(indicators, policy_years)
    
    # Calculate calibration metrics
    calibration_results = ci_evaluator.evaluate_calibration(ci_results)
    
    # Save calibration results
    calibration_df = pd.DataFrame([
        {
            'Method': method,
            'Plausibility_Rate': results['plausibility_rate'],
            'Mean_Abs_Effect': results['mean_abs_effect'],
            'Effect_Variance': results['effect_variance'],
            'Max_Effect': results['max_effect'],
            'Avg_Violation': results.get('avg_violation', 0)
        }
        for method, results in calibration_results.items()
    ])
    
    calibration_df.to_csv(f'{output_dir}/calibration_results.csv', index=False)
    
    # Create focused results summary
    focused_results = {
        'cp_alignment': cp_alignment_results,
        'calibration': calibration_results,
        'ci_results': ci_results
    }
    
    # Output key findings
    print("\n--- Key Findings ---")
    print(f"1. BayesianCausal has {calibration_results['BayesianCausal']['plausibility_rate']*100:.1f}% plausible estimates")
    print(f"   vs. {calibration_results['ITS']['plausibility_rate']*100:.1f}% for ITS")
    
    # Calculate average alignment improvement
    improvements = [
        result['standard_vs_causal'].get('improvement_percentage', 0) 
        for result in cp_alignment_results.values()
        if result['standard_vs_causal'].get('improvement_percentage') is not None
    ]
    if improvements:
        avg_improvement = np.mean(improvements)
        print(f"2. CausalWavelet improves policy alignment by {avg_improvement:.1f}% on average")
    
    return focused_results


# # REPLACE the generate_realistic_synthetic_data function with this improved version:

# def generate_realistic_synthetic_data_v2(n_years=50, n_indicators=4, 
#                                         policy_years=[10, 25, 35], 
#                                         true_effects=[-15, 20, -10],
#                                         random_seed=42) -> Tuple[pd.DataFrame, Dict, Dict]:
#     """
#     Generate MORE realistic synthetic health policy data that better matches real-world characteristics.
    
#     Key improvements:
#     1. Complex autocorrelation patterns
#     2. Multiple overlapping trends 
#     3. Realistic noise structures
#     4. Policy implementation delays
#     5. Confounding seasonal effects
#     6. Missing data patterns
#     """
#     np.random.seed(random_seed)
#     years = np.arange(1970, 1970 + n_years)
    
#     indicators = [
#         'Mortality rate, infant (per 1,000 live births)',
#         'Life expectancy at birth, total (years)',
#         'Maternal mortality ratio (modeled estimate, per 100,000 live births)',
#         'Immunization, measles (% of children ages 12-23 months)'
#     ]
    
#     # Create policy timeline
#     policy_timeline = {}
#     for i, policy_idx in enumerate(policy_years):
#         if policy_idx < len(years):
#             policy_timeline[str(years[policy_idx])] = f"Health_Policy_{i+1}"
    
#     synthetic_data = {}
#     ground_truth = {}
    
#     # Add confounding donor pool indicators (like real data)
#     donor_indicators = [
#         'GDP per capita growth (annual %)',
#         'Urban population (% of total population)', 
#         'Education expenditure (% of GDP)',
#         'Physicians (per 1,000 people)',
#         'Access to electricity (% of population)',
#         'Prevalence of wasting (% of children under 5)'
#     ]
    
#     all_indicators = indicators[:n_indicators] + donor_indicators
    
#     for i, indicator in enumerate(all_indicators):
#         is_mortality = 'mortality' in indicator.lower() or 'wasting' in indicator.lower()
#         is_main_indicator = indicator in indicators[:n_indicators]
        
#         # 1. COMPLEX BASE TRENDS (multiple components)
#         if is_mortality:
#             # Mortality: exponential decay + linear trend + cyclical
#             base_level = 150
#             exponential_decay = base_level * np.exp(-0.03 * (years - 1970))
#             linear_trend = -0.8 * (years - 1970) 
#             cyclical = 5 * np.sin(0.2 * (years - 1970))  # 10-year cycles
#             base_trend = exponential_decay + linear_trend + cyclical
            
#         elif 'life expectancy' in indicator.lower():
#             # Life expectancy: logistic growth with saturation
#             base_level = 45
#             max_level = 80
#             growth_rate = 0.08
#             logistic_growth = max_level / (1 + np.exp(-growth_rate * (years - 1990)))
#             base_trend = base_level + (logistic_growth - max_level/2)
            
#         elif 'gdp' in indicator.lower():
#             # GDP: volatile with business cycles
#             base_trend = 2 + 3 * np.sin(0.15 * (years - 1970)) + np.random.normal(0, 2, len(years))
            
#         elif 'urban' in indicator.lower():
#             # Urban population: steady growth with saturation
#             base_trend = 20 + 60 * (1 - np.exp(-0.04 * (years - 1970)))
            
#         else:
#             # Other indicators: steady improvement with noise
#             base_trend = 30 + 1.2 * (years - 1970) + 3 * np.sin(0.1 * (years - 1970))
        
#         # 2. COMPLEX AUTOCORRELATION STRUCTURE (AR(2) + MA(1))
#         # This is crucial - real health data has strong autocorrelation
#         noise = np.random.normal(0, 1, len(years))
#         autocorr_noise = np.zeros(len(years))
        
#         for t in range(2, len(years)):
#             # AR(2) + MA(1) process
#             autocorr_noise[t] = (0.6 * autocorr_noise[t-1] + 
#                                0.2 * autocorr_noise[t-2] + 
#                                noise[t] + 0.3 * noise[t-1])
        
#         # Scale noise based on indicator type
#         if is_mortality:
#             noise_scale = 8  # Higher noise for mortality data
#         else:
#             noise_scale = 4
            
#         series = base_trend + noise_scale * autocorr_noise
        
#         # 3. REALISTIC POLICY EFFECTS (only for main indicators)
#         if is_main_indicator:
#             for j, policy_idx in enumerate(policy_years):
#                 if j < len(true_effects) and policy_idx < len(years):
#                     effect_size = true_effects[j]
                    
#                     # Adjust effect direction
#                     if is_mortality:
#                         effect_size = -abs(effect_size)
#                     else:
#                         effect_size = abs(effect_size)
                    
#                     # Store ground truth
#                     policy_year = years[policy_idx]
#                     ground_truth[f"{indicator}_{policy_year}"] = effect_size
                    
#                     # 4. REALISTIC POLICY IMPLEMENTATION (gradual + delay + diminishing returns)
#                     implementation_delay = np.random.randint(0, 2)  # 0-1 year delay
#                     implementation_years = 4  # Takes 4 years to fully implement
                    
#                     for k in range(implementation_years):
#                         year_idx = policy_idx + implementation_delay + k
#                         if year_idx < len(years):
#                             # Diminishing returns: 50%, 30%, 15%, 5% of effect each year
#                             yearly_weights = [0.5, 0.3, 0.15, 0.05]
#                             yearly_effect = effect_size * yearly_weights[k]
                            
#                             # Apply to all subsequent years
#                             series[year_idx:] += yearly_effect
                            
#                             # Add implementation noise
#                             if k == 0:  # First year has most uncertainty
#                                 impl_noise = np.random.normal(0, abs(yearly_effect) * 0.3)
#                                 series[year_idx:] += impl_noise
        
#         # 4. ADD MISSING DATA PATTERNS (like real-world data)
#         if np.random.random() < 0.1:  # 10% chance of missing data
#             missing_years = np.random.choice(len(years), size=2, replace=False)
#             for missing_idx in missing_years:
#                 # Interpolate missing values (realistic data collection)
#                 if missing_idx > 0 and missing_idx < len(years) - 1:
#                     series[missing_idx] = (series[missing_idx-1] + series[missing_idx+1]) / 2
        
#         # 5. ENSURE REALISTIC BOUNDS
#         if is_mortality:
#             series = np.clip(series, 5, 200)  # Realistic mortality bounds
#         elif 'life expectancy' in indicator.lower():
#             series = np.clip(series, 35, 85)  # Realistic life expectancy bounds
#         elif 'immunization' in indicator.lower():
#             series = np.clip(series, 0, 100)  # Percentage bounds
        
#         synthetic_data[indicator] = series
    
#     # Convert to DataFrame
#     df_synthetic = pd.DataFrame(synthetic_data, index=[str(year) for year in years])
    
#     # 6. ADD REALISTIC CORRELATIONS between indicators (crucial for donor pool methods)
#     # Mortality should be negatively correlated with GDP, urban development
#     correlation_matrix = np.corrcoef(df_synthetic.values.T)
    
#     # Adjust to make correlations more realistic
#     for i, ind1 in enumerate(df_synthetic.columns):
#         for j, ind2 in enumerate(df_synthetic.columns):
#             if i != j:
#                 # Mortality should correlate negatively with development indicators
#                 if ('mortality' in ind1.lower() and 
#                     ('gdp' in ind2.lower() or 'urban' in ind2.lower() or 'education' in ind2.lower())):
#                     # Add negative correlation
#                     df_synthetic.iloc[:, i] -= 0.3 * df_synthetic.iloc[:, j]
    
#     return df_synthetic, ground_truth, policy_timeline

# # ALSO ADD this function to fix the evaluation metrics mismatch:

# def evaluate_synthetic_with_real_metrics(results: Dict, synthetic_data: pd.DataFrame) -> Dict:
#     """
#     Evaluate synthetic results using the SAME metrics as real data analysis.
#     This ensures consistent comparison.
#     """
#     # Use the same domain constraints as real data
#     domain_constraints = {
#         'Mortality rate, infant (per 1,000 live births)': (-90, 10),
#         'Life expectancy at birth, total (years)': (-10, 30),
#         'Maternal mortality ratio (modeled estimate, per 100,000 live births)': (-90, 10),
#         'Immunization, measles (% of children ages 12-23 months)': (-20, 100)
#     }
    
#     # Calculate plausibility rates (same as real data evaluation)
#     method_plausibility = {}
#     method_significance = {}
    
#     for indicator, indicator_results in results.items():
#         if indicator in domain_constraints:
#             min_val, max_val = domain_constraints[indicator]
            
#             for policy_year, policy_results in indicator_results.items():
#                 for method, method_results in policy_results.items():
#                     if method not in method_plausibility:
#                         method_plausibility[method] = []
#                         method_significance[method] = []
                    
#                     if isinstance(method_results, dict):
#                         effect = method_results.get('relative_effect', np.nan)
#                         significant = method_results.get('significance', False)
                        
#                         if not np.isnan(effect):
#                             # Check plausibility
#                             is_plausible = min_val <= effect <= max_val
#                             method_plausibility[method].append(is_plausible)
#                             method_significance[method].append(significant)
    
#     # Calculate final metrics (same as real data)
#     consistency_results = {}
    
#     for method in method_plausibility:
#         plausible_estimates = method_plausibility[method]
#         significant_estimates = method_significance[method]
        
#         if len(plausible_estimates) > 0:
#             plausibility_rate = np.mean(plausible_estimates)
#             significance_rate = np.mean(significant_estimates)
            
#             consistency_results[method] = {
#                 'plausibility_rate': plausibility_rate,
#                 'significance_rate': significance_rate,
#                 'total_estimates': len(plausible_estimates)
#             }
    
#     return consistency_results

# # UPDATE the run_synthetic_benchmark_test function to use both evaluations:

# def run_synthetic_benchmark_test_v2(output_dir='outputs/synthetic_benchmark/'):
#     """
#     Run improved synthetic benchmark with consistent evaluation metrics.
#     """
#     print("=== RUNNING IMPROVED SYNTHETIC BENCHMARK TEST ===")
    
#     os.makedirs(output_dir, exist_ok=True)
    
#     # Use the improved synthetic data generation
#     print("Generating realistic synthetic data...")
#     df_synthetic, ground_truth, policy_timeline = generate_realistic_synthetic_data_v2(
#         n_years=50, 
#         n_indicators=4,
#         policy_years=[2, 6, 12],  # Years 1972, 1976, 1982 
#         true_effects=[-15, -20, -10],
#         random_seed=42
#     )
    
#     print("Ground Truth Effects:")
#     for key, effect in ground_truth.items():
#         print(f"  {key}: {effect:.1f}%")
    
#     # Run methods
#     policy_years_int = [int(year) for year in policy_timeline.keys()]
#     ci_evaluator = CausalInferenceEvaluator(df_synthetic, policy_timeline)
    
#     # Focus on main health indicators (not donor pool)
#     main_indicators = [
#         'Mortality rate, infant (per 1,000 live births)',
#         'Life expectancy at birth, total (years)',
#         'Maternal mortality ratio (modeled estimate, per 100,000 live births)',
#         'Immunization, measles (% of children ages 12-23 months)'
#     ]
    
#     benchmark_results = ci_evaluator.run_comparative_analysis(main_indicators, policy_years_int)
    
#     # DUAL EVALUATION: Both ground truth accuracy AND plausibility (like real data)
    
#     # 1. Ground truth accuracy (MAE)
#     ground_truth_metrics = calculate_ground_truth_accuracy(benchmark_results, ground_truth)
    
#     # 2. Plausibility metrics (same as real data)
#     plausibility_metrics = evaluate_synthetic_with_real_metrics(benchmark_results, df_synthetic)
    
#     # Print both sets of results
#     print("\n=== GROUND TRUTH ACCURACY ===")
#     print("Method          MAE     Success%")
#     print("-" * 35)
    
#     for method, metrics in sorted(ground_truth_metrics.items(), key=lambda x: x[1]['mae']):
#         mae = metrics['mae']
#         success = metrics['success_rate']
#         print(f"{method:<15} {mae:>6.1f}   {success:>6.1f}%")
    
#     print("\n=== PLAUSIBILITY ANALYSIS (Same as Real Data) ===")
#     print("Method          Plausible%  Significant%")
#     print("-" * 40)
    
#     for method, metrics in sorted(plausibility_metrics.items(), 
#                                  key=lambda x: x[1]['plausibility_rate'], reverse=True):
#         plaus = metrics['plausibility_rate'] * 100
#         signif = metrics['significance_rate'] * 100
#         print(f"{method:<15} {plaus:>9.1f}%   {signif:>10.1f}%")
    
#     # Save comprehensive results
#     combined_results = {}
#     for method in set(list(ground_truth_metrics.keys()) + list(plausibility_metrics.keys())):
#         combined_results[method] = {
#             'ground_truth_mae': ground_truth_metrics.get(method, {}).get('mae', np.nan),
#             'ground_truth_success': ground_truth_metrics.get(method, {}).get('success_rate', 0),
#             'plausibility_rate': plausibility_metrics.get(method, {}).get('plausibility_rate', np.nan),
#             'significance_rate': plausibility_metrics.get(method, {}).get('significance_rate', np.nan)
#         }
    
#     # Save to CSV
#     results_df = pd.DataFrame(combined_results).T
#     results_df.to_csv(os.path.join(output_dir, 'comprehensive_synthetic_results.csv'))
    
#     # Convert ground truth metrics to DataFrame with consistent column names
#     accuracy_df = pd.DataFrame.from_dict(ground_truth_metrics, orient='index')
#     accuracy_df = accuracy_df.rename(columns={
#         'mae': 'MAE',
#         'success_rate': 'Success_Rate',
#         'num_estimates': 'Num_Estimates'
#     })
#     # Reset index to make method names a regular column
#     accuracy_df = accuracy_df.reset_index().rename(columns={'index': 'Method'})
    
#     return {
#         'ground_truth_metrics': ground_truth_metrics,
#         'plausibility_metrics': plausibility_metrics,
#         'synthetic_data': df_synthetic,
#         'benchmark_results': benchmark_results,
#         'accuracy_metrics': accuracy_df
#     }

# def calculate_ground_truth_accuracy(benchmark_results, ground_truth):
#     """Calculate MAE against ground truth for each method."""
#     method_estimates = {}
    
#     # Extract estimates
#     for indicator, indicator_results in benchmark_results.items():
#         for policy_year, policy_results in indicator_results.items():
#             for method, method_results in policy_results.items():
#                 if method not in method_estimates:
#                     method_estimates[method] = {'estimates': [], 'true_values': []}
                
#                 key = f"{indicator}_{policy_year}"
#                 if key in ground_truth:
#                     if isinstance(method_results, dict):
#                         effect = method_results.get('relative_effect', np.nan)
#                         if not np.isnan(effect):
#                             method_estimates[method]['estimates'].append(effect)
#                             method_estimates[method]['true_values'].append(ground_truth[key])
    
#     # Calculate accuracy metrics
#     accuracy_results = {}
    
#     for method, data in method_estimates.items():
#         estimates = np.array(data['estimates'])
#         true_vals = np.array(data['true_values'])
        
#         if len(estimates) > 0:
#             errors = np.abs(estimates - true_vals)
#             mae = np.mean(errors)
#             success_count = np.sum(errors < 50)  # Within 50pp
#             success_rate = success_count / len(errors) * 100
            
#             accuracy_results[method] = {
#                 'mae': mae,
#                 'success_rate': success_rate,
#                 'num_estimates': len(estimates)
#             }
    
#     return accuracy_results