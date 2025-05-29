# src/causal_inference.py - Bayesian Causal Impact Analysis Framework

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pymc as pm
import arviz as az
from scipy import stats
from typing import Dict, List, Tuple, Optional

class BayesianCausalImpactModel:
    """
    A Bayesian framework for analyzing the causal impact of health policies
    on various health indicators.
    
    This model uses Bayesian structural time series to construct counterfactuals
    and estimate the causal effect of policy interventions.
    """
    
    def __init__(self, df_timeseries: pd.DataFrame, policy_timeline: Dict[str, str]):
        """
        Initialize the causal impact model.
        
        Parameters:
        -----------
        df_timeseries : pandas.DataFrame
            Time series data with years as index and indicators as columns
        policy_timeline : dict
            Dictionary mapping policy years to policy names
        """
        self.df_timeseries = df_timeseries
        self.policy_timeline = policy_timeline
        self.models = {}
        self.traces = {}
        self.counterfactuals = {}
        self.effects = {}
        self.tests = {}
        
    def fit_state_space_model(self, indicator: str, policy_year: str) -> None:
        """Fit a Bayesian structural time series model for causal impact analysis."""
        # Get the time series data
        data = self.df_timeseries[indicator].values
        years = self.df_timeseries.index.astype(int).values
        policy_idx = np.where(years == int(policy_year))[0][0]
        
        # Define the model
        with pm.Model() as model:
            # Prior for the local level
            level = pm.GaussianRandomWalk('level', sigma=0.1, shape=len(data))
            
            # Prior for the observation noise
            sigma = pm.HalfNormal('sigma', sigma=1.0)
            
            # Likelihood
            y = pm.Normal('y', mu=level, sigma=sigma, observed=data)
            
            # Sample from the posterior with optimized parameters
            trace = pm.sample(
                draws=200,  # Reduced from 2000
                tune=100,    # Reduced from 1000
                cores=4,     # Single core to avoid overhead
                return_inferencedata=True,
                target_accept=0.8,  # Slightly lower acceptance rate for faster sampling
            )
        
        self.models[(indicator, policy_year)] = model
        self.traces[(indicator, policy_year)] = trace
        
    def estimate_counterfactual(self, indicator: str, policy_year: str) -> None:
        """Estimate the counterfactual time series using the fitted model."""
        if (indicator, policy_year) not in self.traces:
            raise ValueError("Model must be fitted before estimating counterfactual")
            
        trace = self.traces[(indicator, policy_year)]
        years = self.df_timeseries.index.astype(int).values
        policy_idx = np.where(years == int(policy_year))[0][0]
        
        # Get posterior samples of the level
        level_samples = trace.posterior['level'].values
        
        # Calculate counterfactual mean and credible intervals
        counterfactual_mean = np.mean(level_samples, axis=(0, 1))
        counterfactual_lower = np.percentile(level_samples, 2.5, axis=(0, 1))
        counterfactual_upper = np.percentile(level_samples, 97.5, axis=(0, 1))
        
        self.counterfactuals[(indicator, policy_year)] = {
            'mean': counterfactual_mean,
            'lower': counterfactual_lower,
            'upper': counterfactual_upper
        }
        
    def compute_causal_effect(self, indicator: str, policy_year: str) -> Dict:
        """Compute the causal effect of the policy intervention."""
        if (indicator, policy_year) not in self.counterfactuals:
            raise ValueError("Counterfactual must be estimated before computing effect")
            
        data = self.df_timeseries[indicator].values
        years = self.df_timeseries.index.astype(int).values
        policy_idx = np.where(years == int(policy_year))[0][0]
        
        counterfactual = self.counterfactuals[(indicator, policy_year)]
        
        # Calculate effect sizes
        post_data = data[policy_idx:]
        post_counterfactual = counterfactual['mean'][policy_idx:]
        
        absolute_effect = post_data - post_counterfactual
        relative_effect = (absolute_effect / post_counterfactual) * 100
        
        effect = {
            'mean_abs_effect': np.mean(absolute_effect),
            'mean_rel_effect': np.mean(relative_effect),
            'abs_effect_ci': np.percentile(absolute_effect, [2.5, 97.5]),
            'rel_effect_ci': np.percentile(relative_effect, [2.5, 97.5])
        }
        
        self.effects[(indicator, policy_year)] = effect
        return effect
        
    def test_significance(self, indicator: str, policy_year: str) -> Dict:
        """Test the significance of the causal effect using Bayes factors."""
        if (indicator, policy_year) not in self.effects:
            raise ValueError("Effect must be computed before testing significance")
            
        effect = self.effects[(indicator, policy_year)]
        
        # Calculate Bayes factor for the effect
        bf_10 = np.exp(effect['mean_abs_effect'] / effect['abs_effect_ci'][1])
        
        # Interpret the Bayes factor
        if bf_10 > 10:
            interpretation = "Strong evidence for effect"
        elif bf_10 > 3:
            interpretation = "Moderate evidence for effect"
        elif bf_10 > 1:
            interpretation = "Weak evidence for effect"
        else:
            interpretation = "No evidence for effect"
            
        test = {
            'bayes_factor_10': bf_10,
            'interpretation': interpretation
        }
        
        self.tests[(indicator, policy_year)] = test
        return test
        
    def generate_counterfactual_visualization(self, indicator: str, policy_year: str) -> plt.Figure:
        """Generate a visualization of the counterfactual analysis."""
        if (indicator, policy_year) not in self.counterfactuals:
            raise ValueError("Counterfactual must be estimated before visualization")
            
        data = self.df_timeseries[indicator].values
        years = self.df_timeseries.index.astype(int).values
        policy_idx = np.where(years == int(policy_year))[0][0]
        
        counterfactual = self.counterfactuals[(indicator, policy_year)]
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot actual data
        ax.plot(years, data, 'b-', label='Actual', linewidth=2)
        
        # Plot counterfactual
        ax.plot(years, counterfactual['mean'], 'r--', label='Counterfactual', linewidth=2)
        ax.fill_between(years, counterfactual['lower'], counterfactual['upper'],
                       color='r', alpha=0.2, label='95% Credible Interval')
        
        # Add policy line
        ax.axvline(x=int(policy_year), color='k', linestyle='--', alpha=0.5,
                  label=f'Policy: {self.policy_timeline[policy_year]}')
        
        # Add effect annotation
        effect = self.effects[(indicator, policy_year)]
        ax.annotate(f"Effect: {effect['mean_rel_effect']:.1f}%",
                   xy=(years[policy_idx], data[policy_idx]),
                   xytext=(10, 10), textcoords='offset points',
                   bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5))
        
        ax.set_title(f'Causal Impact Analysis: {indicator}\nPolicy: {self.policy_timeline[policy_year]}')
        ax.set_xlabel('Year')
        ax.set_ylabel('Value')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        return fig
        
    def export_summary(self, output_file: str) -> None:
        """Export a summary of all causal impact analyses."""
        results = []
        
        for (indicator, policy_year) in self.effects.keys():
            effect = self.effects[(indicator, policy_year)]
            test = self.tests[(indicator, policy_year)]
            
            results.append({
                'Indicator': indicator,
                'Policy_Year': policy_year,
                'Policy_Name': self.policy_timeline[policy_year],
                'Mean_Rel_Effect': effect['mean_rel_effect'],
                'Rel_Effect_CI_Lower': effect['rel_effect_ci'][0],
                'Rel_Effect_CI_Upper': effect['rel_effect_ci'][1],
                'Bayes_Factor': test['bayes_factor_10'],
                'Evidence': test['interpretation']
            })
            
        summary_df = pd.DataFrame(results)
        summary_df.to_csv(output_file, index=False)
        
    def summarize_results(self) -> pd.DataFrame:
        """Return a summary of all causal impact analyses as a DataFrame."""
        results = []
        
        for (indicator, policy_year) in self.effects.keys():
            effect = self.effects[(indicator, policy_year)]
            test = self.tests[(indicator, policy_year)]
            
            results.append({
                'Indicator': indicator,
                'Policy_Year': policy_year,
                'Policy_Name': self.policy_timeline[policy_year],
                'Mean_Rel_Effect': effect['mean_rel_effect'],
                'Rel_Effect_CI_Lower': effect['rel_effect_ci'][0],
                'Rel_Effect_CI_Upper': effect['rel_effect_ci'][1],
                'Bayes_Factor': test['bayes_factor_10'],
                'Evidence': test['interpretation']
            })
            
        return pd.DataFrame(results)

class BayesianWaveletSyntheticControl:
    """Robust BWSC with better error handling and convergence."""
    
    def __init__(self, df_timeseries: pd.DataFrame, policy_timeline: Dict[str, str]):
        self.df_timeseries = df_timeseries
        self.policy_timeline = policy_timeline
        self.models = {}
        self.traces = {}
        self.wavelet_coeffs = {}
        self.synthetic_controls = {}
        self.effects = {}
        
    def _safe_wavelet_decomposition(self, data: np.ndarray) -> Tuple[List, str, int]:
        """Safe wavelet decomposition with adaptive level selection and robust handling."""
        import pywt
        
        # Try different wavelets with adaptive level selection
        wavelets = ['db8', 'db4', 'sym8', 'coif3', 'haar']
        best_coeffs = None
        best_wavelet = 'db4'
        best_level = 2
        best_energy = 0
        
        for wavelet in wavelets:
            try:
                # Adaptive level selection based on data length
                max_level = pywt.dwt_max_level(len(data), pywt.Wavelet(wavelet).dec_len)
                # Use at least 2 levels but not more than max_level
                level = min(max(2, max_level - 1), max_level)
                
                coeffs = pywt.wavedec(data, wavelet, level=level)
                
                # Validate coefficients and compute energy
                if all(len(c) > 0 and not np.isnan(c).any() for c in coeffs):
                    # Compute energy in detail coefficients
                    energy = sum(np.sum(c**2) for c in coeffs[1:])
                    
                    if energy > best_energy:
                        best_coeffs = coeffs
                        best_wavelet = wavelet
                        best_level = level
                        best_energy = energy
                        
            except Exception:
                continue
        
        if best_coeffs is None:
            # Ultimate fallback - simple decomposition
            try:
                best_coeffs = pywt.wavedec(data, 'db4', level=2)
                best_wavelet = 'db4'
                best_level = 2
            except Exception:
                # Create dummy coefficients if wavelet completely fails
                best_coeffs = [data.copy(), data[:len(data)//2]]
                best_wavelet = 'none'
                best_level = 1
        
        return best_coeffs, best_wavelet, best_level
    
    def _robust_significance_testing(self, absolute_effect: np.ndarray, 
                                   energy_ratio: float, data_length: int) -> Dict:
        """Robust significance testing with multiple statistical criteria."""
        
        if len(absolute_effect) == 0 or np.isnan(absolute_effect).all():
            return self._create_empty_significance_result()
        
        # Clean the effect data
        clean_effects = absolute_effect[~np.isnan(absolute_effect)]
        if len(clean_effects) == 0:
            return self._create_empty_significance_result()
        
        effect_mean = np.mean(clean_effects)
        effect_std = np.std(clean_effects)
        
        # 1. Directional test with confidence
        prob_positive = np.mean(clean_effects > 0)
        directional_sig = prob_positive > 0.65 or prob_positive < 0.35
        
        # 2. Effect size test with Cohen's d
        if effect_std > 0:
            cohens_d = abs(effect_mean) / effect_std
            effect_size_sig = cohens_d > 0.2
        else:
            cohens_d = 0.0
            effect_size_sig = abs(effect_mean) > 1.0
            
        # 3. Energy test with confidence bands
        energy_sig = False
        if not np.isnan(energy_ratio) and energy_ratio > 0:
            # Use confidence bands for energy ratio
            energy_ci = np.percentile(clean_effects**2, [10, 90])
            energy_sig = energy_ratio > 1.1 and energy_ci[0] > 0
            
        # 4. Credible interval test with multiple thresholds
        try:
            ci_80 = np.percentile(clean_effects, [10, 90])
            ci_95 = np.percentile(clean_effects, [2.5, 97.5])
            ci_excludes_zero_80 = not (ci_80[0] <= 0 <= ci_80[1])
            ci_excludes_zero_95 = not (ci_95[0] <= 0 <= ci_95[1])
        except:
            ci_excludes_zero_80 = False
            ci_excludes_zero_95 = False
            
        # 5. Magnitude test with relative scale
        magnitude_sig = abs(effect_mean) > 0.5 * effect_std if effect_std > 0 else abs(effect_mean) > 0.5
        
        # 6. Trend consistency test
        if len(clean_effects) > 2:
            trend = np.polyfit(np.arange(len(clean_effects)), clean_effects, 1)[0]
            trend_consistent = (trend > 0 and effect_mean > 0) or (trend < 0 and effect_mean < 0)
        else:
            trend_consistent = True
        
        # Combined significance with weighted scoring
        significance_score = (
            directional_sig * 0.25 +           # Direction
            effect_size_sig * 0.20 +          # Effect size
            energy_sig * 0.15 +               # Energy
            ci_excludes_zero_80 * 0.15 +      # 80% CI
            ci_excludes_zero_95 * 0.10 +      # 95% CI
            magnitude_sig * 0.10 +            # Magnitude
            trend_consistent * 0.05           # Trend consistency
        )
        
        # Adaptive threshold based on data length
        threshold = max(0.25, 0.4 - 0.01 * min(data_length, 20))
        is_significant = significance_score > threshold
        
        return {
            'is_significant': is_significant,
            'significance_score': significance_score,
            'prob_positive': prob_positive,
            'standardized_effect': cohens_d,
            'energy_ratio': energy_ratio,
            'ci_excludes_zero_80': ci_excludes_zero_80,
            'ci_excludes_zero_95': ci_excludes_zero_95,
            'magnitude_significant': magnitude_sig,
            'trend_consistent': trend_consistent,
            'effect_mean': effect_mean,
            'effect_std': effect_std
        }
        
    def _create_empty_significance_result(self) -> Dict:
        """Create empty significance result for error cases."""
        return {
            'is_significant': False,
            'significance_score': 0.0,
            'prob_positive': 0.5,
            'standardized_effect': 0.0,
            'energy_ratio': 1.0,
            'ci_excludes_zero_80': False,
            'ci_excludes_zero_95': False,
            'magnitude_significant': False,
            'trend_consistent': False,
            'effect_mean': 0.0,
            'effect_std': 0.0
        }
        
    def fit_hybrid_model(self, indicator: str, policy_year: str) -> None:
        """Robust hybrid model fitting with adaptive convergence."""
        data = self.df_timeseries[indicator].values
        years = self.df_timeseries.index.astype(int).values
        policy_idx = np.where(years == int(policy_year))[0][0]
        
        # Step 1: Safe wavelet decomposition
        coeffs, wavelet_type, level = self._safe_wavelet_decomposition(data)
        self.wavelet_coeffs[(indicator, policy_year)] = {
            'coeffs': coeffs,
            'wavelet': wavelet_type,
            'level': level
        }
        
        # Step 2: Enhanced Bayesian model with adaptive parameters
        with pm.Model() as model:
            # Adaptive priors based on data characteristics
            data_std = np.std(data)
            data_range = np.ptp(data)
            
            # Level component with adaptive variance
            level_sigma = pm.HalfNormal('level_sigma', sigma=max(0.1, 0.1 * data_std))
            level = pm.GaussianRandomWalk('level', sigma=level_sigma, shape=len(data))
            
            # Seasonal component if enough data points
            if len(data) > 12:
                seasonal_period = 12  # Assuming yearly data
                seasonal_sigma = pm.HalfNormal('seasonal_sigma', sigma=0.1 * data_std)
                seasonal = pm.GaussianRandomWalk('seasonal', sigma=seasonal_sigma, shape=len(data))
                seasonal = pm.Deterministic('seasonal_effect', 
                    pm.math.sin(2 * np.pi * np.arange(len(data)) / seasonal_period) * seasonal)
            else:
                seasonal = pm.Constant('seasonal', 0)
            
            # Observation model with adaptive noise
            sigma = pm.HalfNormal('sigma', sigma=max(0.1, 0.1 * data_std))
            y = pm.Normal('y', mu=level + seasonal, sigma=sigma, observed=data)
            
            # More robust sampling with adaptive parameters
            try:
                # First try with NUTS
                trace = pm.sample(
                    draws=200,
                    tune=100,
                    cores=1,
                    chains=2,
                    return_inferencedata=True,
                    target_accept=0.8,
                    progressbar=False
                )
            except Exception as e:
                print(f"Warning: NUTS sampling failed ({e}), trying with Metropolis")
                try:
                    # Fallback to Metropolis
                    trace = pm.sample(
                        draws=200,
                        tune=100,
                        cores=1,
                        chains=2,
                        return_inferencedata=True,
                        progressbar=False,
                        step=pm.Metropolis()
                    )
                except Exception as e:
                    print(f"Warning: Metropolis sampling failed ({e}), using simpler model")
                    # Ultimate fallback - very simple model
                    with pm.Model() as simple_model:
                        level_sigma = pm.HalfNormal('level_sigma', sigma=0.05)
                        level = pm.Normal('level', mu=np.mean(data), sigma=level_sigma, shape=len(data))
                        sigma = pm.HalfNormal('sigma', sigma=0.5)
                        y = pm.Normal('y', mu=level, sigma=sigma, observed=data)
                        
                        trace = pm.sample(
                            draws=100,
                            tune=50,
                            cores=1,
                            chains=1,
                            return_inferencedata=True,
                            progressbar=False
                        )
        
        self.models[(indicator, policy_year)] = model
        self.traces[(indicator, policy_year)] = trace
        
        # Step 3: Enhanced synthetic control with adaptive weighting
        donor_pool = self.df_timeseries.drop(columns=[indicator])
        pre_period = np.where(years < int(policy_year))[0]
        
        if len(pre_period) < 3:
            # Not enough pre-period data, use simple weighting
            donor_weights = np.ones(donor_pool.shape[1]) / donor_pool.shape[1]
        else:
            # Enhanced donor selection with multiple metrics
            donor_weights = np.zeros(donor_pool.shape[1])
            for i, col in enumerate(donor_pool.columns):
                try:
                    donor_data = donor_pool[col].values
                    
                    # Correlation-based weight
                    corr = np.corrcoef(data[pre_period], donor_data[pre_period])[0,1]
                    if np.isnan(corr):
                        corr = 0.0
                    
                    # Trend similarity weight
                    if len(pre_period) > 2:
                        data_trend = np.polyfit(pre_period, data[pre_period], 1)[0]
                        donor_trend = np.polyfit(pre_period, donor_data[pre_period], 1)[0]
                        trend_sim = 1.0 / (1.0 + abs(data_trend - donor_trend))
                    else:
                        trend_sim = 0.5
                    
                    # Combine weights
                    donor_weights[i] = max(0.01, 0.7 * abs(corr) + 0.3 * trend_sim)
                    
                except Exception:
                    donor_weights[i] = 0.01  # Fallback weight
            
            # Normalize weights
            donor_weights = donor_weights / np.sum(donor_weights)
        
        synthetic_control = donor_pool.values @ donor_weights
        
        self.synthetic_controls[(indicator, policy_year)] = {
            'values': synthetic_control,
            'weights': dict(zip(donor_pool.columns, donor_weights))
        }
    
    def estimate_counterfactual(self, indicator: str, policy_year: str) -> Dict:
        """Robust counterfactual estimation with NaN handling."""
        if (indicator, policy_year) not in self.traces:
            raise ValueError("Model must be fitted before estimating counterfactual")
            
        trace = self.traces[(indicator, policy_year)]
        years = self.df_timeseries.index.astype(int).values
        policy_idx = np.where(years == int(policy_year))[0][0]
        
        try:
            # Get bayesian predictions with error handling
            level_samples = trace.posterior['level'].values
            bayesian_counterfactual = np.mean(level_samples, axis=(0, 1))
            
            # Check for NaN values
            if np.isnan(bayesian_counterfactual).any():
                print("Warning: NaN in Bayesian counterfactual, using data mean")
                bayesian_counterfactual = np.full(len(years), np.mean(self.df_timeseries[indicator].values))
                
        except Exception as e:
            print(f"Warning: Bayesian counterfactual failed ({e}), using data mean")
            bayesian_counterfactual = np.full(len(years), np.mean(self.df_timeseries[indicator].values))
        
        # Get synthetic control
        synthetic_control = self.synthetic_controls[(indicator, policy_year)]['values']
        
        # Simple 50-50 weighting (can be made more sophisticated later)
        bayesian_weight = 0.5
        synthetic_weight = 0.5
        
        # Combine counterfactuals
        hybrid_counterfactual = (bayesian_weight * bayesian_counterfactual + 
                               synthetic_weight * synthetic_control)
        
        # Simple uncertainty bounds
        data_std = np.std(self.df_timeseries[indicator].values)
        lower = hybrid_counterfactual - 1.96 * data_std
        upper = hybrid_counterfactual + 1.96 * data_std
        
        return {
            'mean': hybrid_counterfactual,
            'lower': lower,
            'upper': upper,
            'bayesian_weight': bayesian_weight,
            'synthetic_weight': synthetic_weight
        }
        
    def compute_causal_effect(self, indicator: str, policy_year: str) -> Dict:
        """Robust causal effect computation with extensive error handling."""
        if (indicator, policy_year) not in self.traces:
            raise ValueError("Model must be fitted before computing effect")
            
        data = self.df_timeseries[indicator].values
        years = self.df_timeseries.index.astype(int).values
        policy_idx = np.where(years == int(policy_year))[0][0]
        
        try:
            # Get counterfactual
            counterfactual = self.estimate_counterfactual(indicator, policy_year)
            
            # Calculate effects with robust handling
            post_data = data[policy_idx:]
            post_counterfactual = counterfactual['mean'][policy_idx:]
            
            # Handle edge cases
            if len(post_data) == 0 or len(post_counterfactual) == 0:
                print("Warning: No post-treatment data available")
                return self._create_empty_effect_result()
            
            absolute_effect = post_data - post_counterfactual
            
            # Robust relative effect calculation
            relative_effect = np.zeros_like(absolute_effect)
            for i, (abs_eff, counterfactual_val) in enumerate(zip(absolute_effect, post_counterfactual)):
                if abs(counterfactual_val) > 1e-8:  # Avoid division by zero
                    relative_effect[i] = (abs_eff / abs(counterfactual_val)) * 100
                else:
                    relative_effect[i] = 0.0
            
            # Calculate energy ratio safely
            wavelet_info = self.wavelet_coeffs[(indicator, policy_year)]
            coeffs = wavelet_info['coeffs']
            
            try:
                # Safe energy calculation
                pre_energy = 0.0
                post_energy = 0.0
                
                for c in coeffs[1:]:  # Skip approximation coefficients
                    if len(c) > policy_idx:
                        pre_energy += np.sum(c[:policy_idx]**2)
                        post_energy += np.sum(c[policy_idx:]**2)
                
                energy_ratio = post_energy / max(pre_energy, 1e-8)
                
            except Exception as e:
                print(f"Warning: Energy calculation failed ({e})")
                energy_ratio = 1.0  # Neutral value
            
            # Robust significance testing
            significance_results = self._robust_significance_testing(
                absolute_effect, energy_ratio, len(data)
            )
            
            # Create effect dictionary with safe values
            effect = {
                'mean_abs_effect': np.mean(absolute_effect) if len(absolute_effect) > 0 else 0.0,
                'mean_rel_effect': np.mean(relative_effect) if len(relative_effect) > 0 else 0.0,
                'abs_effect_ci': np.percentile(absolute_effect, [2.5, 97.5]) if len(absolute_effect) > 0 else [0.0, 0.0],
                'rel_effect_ci': np.percentile(relative_effect, [2.5, 97.5]) if len(relative_effect) > 0 else [0.0, 0.0],
                'energy_ratio': energy_ratio,
                'significance': significance_results['is_significant'],
                'significance_score': significance_results['significance_score'],
                'bayesian_weight': counterfactual['bayesian_weight'],
                'synthetic_weight': counterfactual['synthetic_weight'],
                'prob_positive_effect': significance_results['prob_positive'],
                'standardized_effect': significance_results['standardized_effect']
            }
            
            self.effects[(indicator, policy_year)] = effect
            return effect
            
        except Exception as e:
            print(f"Error in causal effect computation: {e}")
            return self._create_empty_effect_result()
    
    def _create_empty_effect_result(self) -> Dict:
        """Create empty effect result for error cases."""
        return {
            'mean_abs_effect': 0.0,
            'mean_rel_effect': 0.0,
            'abs_effect_ci': [0.0, 0.0],
            'rel_effect_ci': [0.0, 0.0],
            'energy_ratio': 1.0,
            'significance': False,
            'significance_score': 0.0,
            'bayesian_weight': 0.5,
            'synthetic_weight': 0.5,
            'prob_positive_effect': 0.5,
            'standardized_effect': 0.0
        }