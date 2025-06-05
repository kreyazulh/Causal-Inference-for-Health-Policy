"""
Meta-Learning Enhanced Double Machine Learning for Causal Policy Analysis

This module implements Meta-DML with two key novel contributions:
1. Adaptive Orthogonalization: Iterative refinement of nuisance estimation
   with conservative updates to maintain stability
2. Uncertainty-Aware Meta-Learning: Weights base learners by their prediction 
   uncertainty with regularization to prevent extreme weights

Author: CIKM 2025 Submission
Date: 2025
"""

import numpy as np
import pandas as pd
import warnings
from typing import Dict, List, Tuple, Optional
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import ElasticNet, Ridge, Lasso
from sklearn.model_selection import cross_val_predict, KFold
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

# Try to import torch, but provide fallback if not available
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    warnings.warn("PyTorch not available. Meta-DML will use simplified meta-learning.")


class UncertaintyAwareMetaLearner:
    """
    Novel contribution #1: Meta-learner that incorporates prediction uncertainty
    to weight base learners more robustly.
    """
    
    def __init__(self, n_learners: int):
        self.n_learners = n_learners
        self.weights = np.ones(n_learners) / n_learners
        
    def learn_weights(self, performances: Dict[str, Dict], residuals: Dict[str, Dict]) -> np.ndarray:
        """
        Learn optimal weights considering both performance AND prediction uncertainty.
        
        Key innovation: We estimate uncertainty from residual variance across CV folds
        and use it to down-weight unstable learners.
        """
        
        # Extract performance and compute uncertainty
        weights = []
        uncertainties = []
        
        for name in performances.keys():
            # Performance-based weight (inverse MSE)
            combined_mse = performances[name]['combined_mse']
            perf_weight = 1.0 / (combined_mse + 1e-8)
            
            # Uncertainty estimation from residual stability
            y_res = residuals[name]['y_residual']
            t_res = residuals[name]['t_residual']
            
            # Novel: Compute residual variance in local neighborhoods
            # Split data into chunks to estimate local variance
            n_chunks = min(10, len(y_res) // 20)
            if n_chunks > 1:
                chunk_vars = []
                for i in range(n_chunks):
                    start_idx = i * len(y_res) // n_chunks
                    end_idx = (i + 1) * len(y_res) // n_chunks
                    chunk_var = np.var(y_res[start_idx:end_idx]) + np.var(t_res[start_idx:end_idx])
                    chunk_vars.append(chunk_var)
                
                # Uncertainty is the variance of variances (stability measure)
                uncertainty = np.var(chunk_vars) + 1e-8
            else:
                uncertainty = np.var(y_res) + np.var(t_res) + 1e-8
            
            weights.append(perf_weight)
            uncertainties.append(1.0 / uncertainty)  # Inverse uncertainty
        
        # Combine performance and uncertainty scores
        weights = np.array(weights)
        uncertainties = np.array(uncertainties)
        
        # Novel weighting scheme: geometric mean of performance and certainty
        final_weights = np.sqrt(weights * uncertainties)
        
        # Add conservative regularization to prevent extreme weights
        # This ensures no single learner dominates completely
        min_weight = 0.05  # Minimum 5% weight for each learner
        final_weights = (1 - min_weight * self.n_learners) * (final_weights / np.sum(final_weights)) + min_weight
        
        self.weights = final_weights
        return final_weights


class AdaptiveOrthogonalizer:
    """
    Novel contribution #2: Iterative refinement of DML orthogonalization
    that improves nuisance function estimation.
    """
    
    def __init__(self, max_iterations: int = 2, tolerance: float = 1e-3):
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        
    def refine_residuals(self, learner, X: np.ndarray, y: np.ndarray, 
                        treatment: np.ndarray, cv_folds: int = 3) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Key innovation: Iteratively refine residuals by incorporating the estimated
        treatment effect back into nuisance estimation, reducing bias.
        """
        
        cv = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
        
        # Initial standard DML residualization
        y_pred = cross_val_predict(learner, X, y, cv=cv)
        t_pred = cross_val_predict(learner, X, treatment, cv=cv)
        
        y_residual = y - y_pred
        t_residual = treatment - t_pred
        
        # Initial effect estimate
        if np.var(t_residual) > 1e-8:
            current_effect = np.cov(y_residual, t_residual)[0,1] / np.var(t_residual)
        else:
            return y_residual, t_residual, 0.0
        
        # Store initial values for stability
        initial_effect = current_effect
        
        # Adaptive refinement loop with regularization
        for iteration in range(self.max_iterations):
            # Novel step: Adjust outcome by current effect estimate
            # Add damping factor to prevent overshooting
            damping_factor = 0.7  # Conservative update
            adjusted_outcome = y - (damping_factor * current_effect) * treatment
            
            # Re-estimate outcome model with adjusted target
            y_pred_adjusted = cross_val_predict(learner, X, adjusted_outcome, cv=cv)
            
            # Compute refined residuals
            y_residual_new = y - y_pred_adjusted - current_effect * (treatment - t_pred)
            
            # Update effect estimate with refined residuals
            if np.var(t_residual) > 1e-8:
                new_effect = np.cov(y_residual_new, t_residual)[0,1] / np.var(t_residual)
                
                # Add regularization: limit maximum change per iteration
                max_change = 0.5 * abs(initial_effect) + 0.1
                change = new_effect - current_effect
                if abs(change) > max_change:
                    new_effect = current_effect + np.sign(change) * max_change
                
                # Check convergence
                if abs(new_effect - current_effect) < self.tolerance:
                    break
                    
                current_effect = new_effect
                y_residual = y_residual_new
        
        return y_residual, t_residual, current_effect


class MetaLearnerDML:
    """
    Meta-Learning Enhanced Double Machine Learning with two key innovations:
    1. Uncertainty-aware meta-weighting of base learners
    2. Adaptive orthogonalization for improved nuisance estimation
    """
    
    def __init__(self, base_learners: Optional[Dict] = None, cv_folds: int = 3,
                 use_adaptive_orthogonalization: bool = True):
        """
        Initialize Meta-DML with focused novel contributions.
        
        Args:
            base_learners: Dictionary of base ML models
            cv_folds: Number of cross-validation folds
            use_adaptive_orthogonalization: Whether to use our novel orthogonalization
        """
        self.cv_folds = cv_folds
        self.use_adaptive_orthogonalization = use_adaptive_orthogonalization
        
        # Standard base learners
        if base_learners is None:
            self.base_learners = {
                'rf': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
                'gbm': GradientBoostingRegressor(n_estimators=100, random_state=42),
                'mlp': MLPRegressor(hidden_layer_sizes=(50,), random_state=42, max_iter=500),
                'elastic': ElasticNet(random_state=42, max_iter=1000),
                'ridge': Ridge(random_state=42),
            }
        else:
            self.base_learners = base_learners
            
        # Our novel components
        self.meta_learner = UncertaintyAwareMetaLearner(len(self.base_learners))
        self.orthogonalizer = AdaptiveOrthogonalizer() if use_adaptive_orthogonalization else None
        
    def _extract_meta_features(self, X: np.ndarray, y: np.ndarray, treatment: np.ndarray) -> np.ndarray:
        """Extract basic meta-features for the problem."""
        meta_features = []
        
        # Basic statistics
        n_samples, n_features = X.shape
        meta_features.extend([
            n_samples,
            n_features,
            np.std(y),
            np.mean(treatment),
            np.var(treatment)
        ])
        
        return np.array(meta_features).reshape(1, -1)
    
    def _compute_residuals_for_learner(self, learner, X: np.ndarray, y: np.ndarray, 
                                      treatment: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        """Compute residuals using either standard or adaptive orthogonalization."""
        
        if self.use_adaptive_orthogonalization and self.orthogonalizer is not None:
            # Use our novel adaptive orthogonalization
            y_residual, t_residual, effect_est = self.orthogonalizer.refine_residuals(
                learner, X, y, treatment, self.cv_folds
            )
            
            # Safety check: if adaptive method produces extreme effects, fall back to standard
            baseline_mean = np.mean(y[treatment == 0]) if np.sum(treatment == 0) > 0 else np.mean(y)
            if baseline_mean != 0:
                relative_effect = abs(effect_est / baseline_mean)
                # If effect is unreasonably large (>200%), fall back to standard DML
                if relative_effect > 2.0:
                    warnings.warn(f"Adaptive orthogonalization produced extreme effect ({relative_effect:.1%}), falling back to standard DML")
                    # Fall back to standard method below
                else:
                    return y_residual, t_residual, effect_est
        
        # Standard DML residualization (also used as fallback)
        cv = KFold(n_splits=self.cv_folds, shuffle=True, random_state=42)
        y_pred = cross_val_predict(learner, X, y, cv=cv)
        t_pred = cross_val_predict(learner, X, treatment, cv=cv)
        
        y_residual = y - y_pred
        t_residual = treatment - t_pred
        
        if np.var(t_residual) > 1e-8:
            effect_est = np.cov(y_residual, t_residual)[0,1] / np.var(t_residual)
        else:
            effect_est = 0.0
            
        return y_residual, t_residual, effect_est
    
    def estimate_treatment_effect(self, X: np.ndarray, y: np.ndarray, treatment: np.ndarray) -> Dict:
        """
        Estimate treatment effect using Meta-DML with our novel contributions.
        """
        
        # Input validation
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        treatment = np.asarray(treatment, dtype=np.float64)
        
        # Store residuals and performance for each learner
        residuals = {}
        performances = {}
        initial_effects = {}
        
        # Compute residuals for each base learner
        for name, learner in self.base_learners.items():
            try:
                # Get residuals (using adaptive orthogonalization if enabled)
                y_res, t_res, init_effect = self._compute_residuals_for_learner(
                    learner, X, y, treatment
                )
                
                residuals[name] = {
                    'y_residual': y_res,
                    't_residual': t_res
                }
                
                initial_effects[name] = init_effect
                
                # Compute performance metrics
                # Novel: Include uncertainty in performance assessment
                y_cv_pred = y - y_res  # Reconstructed predictions
                t_cv_pred = treatment - t_res
                
                y_mse = mean_squared_error(y, y_cv_pred)
                t_mse = mean_squared_error(treatment, t_cv_pred)
                
                performances[name] = {
                    'y_mse': y_mse,
                    't_mse': t_mse,
                    'combined_mse': y_mse + t_mse
                }
                
            except Exception as e:
                warnings.warn(f"Error with learner {name}: {str(e)}")
                # Fallback values
                residuals[name] = {
                    'y_residual': np.zeros_like(y),
                    't_residual': np.zeros_like(treatment)
                }
                performances[name] = {'y_mse': np.inf, 't_mse': np.inf, 'combined_mse': np.inf}
                initial_effects[name] = 0.0
        
        # Learn optimal weights using our uncertainty-aware meta-learner
        optimal_weights = self.meta_learner.learn_weights(performances, residuals)
        
        # Compute final weighted residuals
        weighted_y_residual = np.zeros_like(y)
        weighted_t_residual = np.zeros_like(treatment)
        
        for i, name in enumerate(residuals.keys()):
            weighted_y_residual += optimal_weights[i] * residuals[name]['y_residual']
            weighted_t_residual += optimal_weights[i] * residuals[name]['t_residual']
        
        # Final treatment effect
        if np.var(weighted_t_residual) > 1e-8:
            treatment_effect = np.cov(weighted_y_residual, weighted_t_residual)[0,1] / np.var(weighted_t_residual)
        else:
            treatment_effect = 0.0
            warnings.warn("No variation in weighted treatment residuals")
        
        # Bootstrap confidence intervals
        n_bootstrap = 200
        bootstrap_effects = []
        
        for _ in range(n_bootstrap):
            idx = np.random.choice(len(y), len(y), replace=True)
            boot_y = weighted_y_residual[idx]
            boot_t = weighted_t_residual[idx]
            
            if np.var(boot_t) > 1e-8:
                boot_effect = np.cov(boot_y, boot_t)[0,1] / np.var(boot_t)
                bootstrap_effects.append(boot_effect)
        
        # Confidence intervals
        if len(bootstrap_effects) > 10:
            ci_lower, ci_upper = np.percentile(bootstrap_effects, [2.5, 97.5])
        else:
            se = np.std(weighted_y_residual) / np.sqrt(len(y))
            ci_lower = treatment_effect - 1.96 * se
            ci_upper = treatment_effect + 1.96 * se
        
        # Convert to relative effect
        baseline_mean = np.mean(y[treatment == 0]) if np.sum(treatment == 0) > 0 else np.mean(y)
        relative_effect = (treatment_effect / baseline_mean) * 100 if baseline_mean != 0 else 0
        
        # Prepare meta insights for the main script
        meta_insights = {
            'dominant_learner': max(self.base_learners.keys(), 
                                  key=lambda k: optimal_weights[list(self.base_learners.keys()).index(k)]),
            'weight_entropy': -np.sum(optimal_weights * np.log(optimal_weights + 1e-10)),
            'n_effective_learners': 1.0 / np.sum(optimal_weights**2)
        }
        
        return {
            'treatment_effect': float(treatment_effect),
            'relative_effect': float(relative_effect),
            'lower_bound': float(ci_lower),  # Changed from ci_lower
            'upper_bound': float(ci_upper),  # Changed from ci_upper
            'significance': not (ci_lower <= 0 <= ci_upper),  # Changed from significant
            'meta_weights': {k: float(v) for k, v in zip(self.base_learners.keys(), optimal_weights)},
            'used_adaptive_orthogonalization': self.use_adaptive_orthogonalization,
            'initial_effects': {k: float(v) for k, v in initial_effects.items()},
            'method': 'Meta-DML',
            'meta_insights': meta_insights  # Added for compatibility
        }


def create_meta_dml_estimator(enhanced_features: bool = True, use_neural_meta: bool = True) -> MetaLearnerDML:
    """
    Factory function to create a Meta-DML estimator with optional enhancements.
    
    Args:
        enhanced_features: Whether to use enhanced base learners
        use_neural_meta: Whether to use neural meta-learning (currently ignored as we use uncertainty-aware meta-learning)
    
    Returns:
        MetaLearnerDML: Configured Meta-DML estimator
    """
    
    if enhanced_features:
        # Enhanced base learners with more diverse models
        base_learners = {
            'rf': RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1),
            'gbm': GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, max_depth=6, random_state=42),
            'mlp': MLPRegressor(hidden_layer_sizes=(100, 50), random_state=42, max_iter=1000, alpha=0.01),
            'elastic': ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42, max_iter=2000),
            'ridge': Ridge(alpha=1.0, random_state=42),
            'lasso': Lasso(alpha=0.1, random_state=42, max_iter=2000),
        }
    else:
        # Standard base learners
        base_learners = None
    
    # Create Meta-DML estimator with adaptive orthogonalization enabled
    meta_dml = MetaLearnerDML(
        base_learners=base_learners,
        cv_folds=5 if enhanced_features else 3,
        use_adaptive_orthogonalization=True  # Always use our novel orthogonalization
    )
    
    return meta_dml