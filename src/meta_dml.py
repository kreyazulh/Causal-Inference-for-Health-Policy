"""
Meta-Learning Enhanced Double Machine Learning for Causal Policy Analysis

This module implements Meta-DML, a novel approach that uses neural meta-learning 
to optimally combine multiple base learners in the Double Machine Learning framework
for improved causal effect estimation in policy analysis.

Author: Your Name
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
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    warnings.warn("PyTorch not available. Meta-DML will use simplified meta-learning.")


class SimplifiedMetaLearner:
    """
    Simplified meta-learner using weighted combinations when PyTorch is not available.
    """
    
    def __init__(self, n_learners: int):
        self.n_learners = n_learners
        self.weights = np.ones(n_learners) / n_learners  # Equal weights initially
        
    def learn_weights(self, performances: List[float], meta_features: np.ndarray) -> np.ndarray:
        """Learn optimal weights based on base learner performances."""
        performances = np.array(performances)
        
        # Inverse performance weighting (lower error = higher weight)
        inv_perf = 1.0 / (performances + 1e-8)
        weights = inv_perf / np.sum(inv_perf)
        
        # Add some meta-feature based adjustment
        if meta_features.size > 0:
            # Simple rule: if high variance in outcome, prefer ensemble methods
            variance_factor = meta_features[0, 2] if meta_features.shape[1] > 2 else 1.0
            if variance_factor > np.median([1.0, 5.0, 10.0]):  # High variance
                # Boost tree-based methods (assume first two are RF and GBM)
                weights[:2] *= 1.2
                weights = weights / np.sum(weights)
        
        self.weights = weights
        return weights


class MetaNetwork(nn.Module):
    """Neural network for meta-learning optimal base learner combinations."""
    
    def __init__(self, input_dim: int, hidden_dim: int, n_learners: int):
        super().__init__()
        self.fc1 = nn.Linear(input_dim + n_learners, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, n_learners)
        self.softmax = nn.Softmax(dim=1)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, features: torch.Tensor, base_predictions: torch.Tensor) -> torch.Tensor:
        # Concatenate data features with base learner predictions
        x = torch.cat([features, base_predictions], dim=1)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        weights = self.softmax(self.fc3(x))  # Learner weights
        return weights


class MetaLearnerDML:
    """
    Meta-Learning Enhanced Double Machine Learning
    
    Novel contribution: Uses meta-learning to optimally combine multiple base learners
    in the DoubleML framework, automatically adapting to different policy contexts.
    """
    
    def __init__(self, base_learners: Optional[Dict] = None, meta_hidden_dim: int = 32, 
                 cv_folds: int = 3, use_neural_meta: bool = True):
        """
        Initialize Meta-DML with configurable base learners and meta-learning approach.
        
        Args:
            base_learners: Dictionary of base ML models
            meta_hidden_dim: Hidden dimension for neural meta-learner
            cv_folds: Number of cross-validation folds
            use_neural_meta: Whether to use neural meta-learner (requires PyTorch)
        """
        self.cv_folds = cv_folds
        self.meta_hidden_dim = meta_hidden_dim
        self.use_neural_meta = use_neural_meta and TORCH_AVAILABLE
        
        # Define diverse base learners with different inductive biases
        if base_learners is None:
            self.base_learners = {
                'rf': RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1),
                'gbm': GradientBoostingRegressor(n_estimators=50, random_state=42),
                'mlp': MLPRegressor(hidden_layer_sizes=(32,), random_state=42, 
                                  max_iter=300, early_stopping=True, validation_fraction=0.2),
                'elastic': ElasticNet(random_state=42, max_iter=1000),
                'ridge': Ridge(random_state=42),
            }
        else:
            self.base_learners = base_learners
            
        # Meta-learner components
        self.meta_network = None
        self.simple_meta = SimplifiedMetaLearner(len(self.base_learners))
        self.scaler = StandardScaler()
        self.is_fitted = False
        
    def _extract_meta_features(self, X: np.ndarray, y: np.ndarray, treatment: np.ndarray) -> np.ndarray:
        """Extract meta-features to help the meta-learner decide on base learner weights."""
        meta_features = []
        
        # Data characteristics
        n_samples, n_features = X.shape
        meta_features.extend([
            n_samples,  # Sample size
            n_features,  # Feature dimensionality  
            np.std(y),   # Outcome variance
            np.mean(treatment),  # Treatment prevalence
        ])
        
        # Treatment-outcome relationship
        if len(np.unique(treatment)) > 1:
            try:
                corr = np.corrcoef(y, treatment)[0, 1]
                meta_features.append(corr if not np.isnan(corr) else 0.0)
            except:
                meta_features.append(0.0)
        else:
            meta_features.append(0.0)
        
        # Time series characteristics (assuming first column is time-based) 
        if n_features > 0:
            time_col = X[:, 0]
            meta_features.extend([
                np.std(np.diff(y)) if len(y) > 1 else 0.0,  # Temporal stability
            ])
            
            # Time trend correlation
            if np.std(time_col) > 1e-8:
                try:
                    time_corr = np.corrcoef(time_col, y)[0, 1]
                    meta_features.append(time_corr if not np.isnan(time_corr) else 0.0)
                except:
                    meta_features.append(0.0)
            else:
                meta_features.append(0.0)
        
        return np.array(meta_features).reshape(1, -1)
    
    def _train_base_learners(self, X: np.ndarray, y: np.ndarray, treatment: np.ndarray) -> Tuple[Dict, Dict]:
        """Train all base learners using cross-validation."""
        base_predictions = {}
        base_performance = {}
        
        cv = KFold(n_splits=self.cv_folds, shuffle=True, random_state=42)
        
        for name, learner in self.base_learners.items():
            try:
                # Ensure consistent data types for cross-validation
                X_cv = np.asarray(X, dtype=np.float64)
                y_cv = np.asarray(y, dtype=np.float64)
                treatment_cv = np.asarray(treatment, dtype=np.float64)
                
                # Outcome model predictions
                y_pred = cross_val_predict(learner, X_cv, y_cv, cv=cv)
                
                # Treatment model predictions  
                t_pred = cross_val_predict(learner, X_cv, treatment_cv, cv=cv)
                
                # Ensure predictions are float64
                y_pred = np.asarray(y_pred, dtype=np.float64)
                t_pred = np.asarray(t_pred, dtype=np.float64)
                
                base_predictions[name] = {
                    'outcome_pred': y_pred,
                    'treatment_pred': t_pred
                }
                
                # Calculate performance metrics for meta-learning
                y_mse = mean_squared_error(y_cv, y_pred)
                t_mse = mean_squared_error(treatment_cv, t_pred)
                base_performance[name] = {
                    'y_mse': y_mse, 
                    't_mse': t_mse,
                    'combined_mse': y_mse + t_mse  # Combined performance metric
                }
                
            except Exception as e:
                warnings.warn(f"Error training base learner {name}: {str(e)}")
                # Provide fallback predictions
                base_predictions[name] = {
                    'outcome_pred': np.full_like(y, np.mean(y), dtype=np.float64),
                    'treatment_pred': np.full_like(treatment, np.mean(treatment), dtype=np.float64)
                }
                base_performance[name] = {'y_mse': np.inf, 't_mse': np.inf, 'combined_mse': np.inf}
                
        return base_predictions, base_performance
    
    def _compute_dml_residuals(self, base_predictions: Dict, y: np.ndarray, treatment: np.ndarray) -> Dict:
        """Compute residualized outcomes and treatments for each base learner."""
        residuals = {}
        
        for name, preds in base_predictions.items():
            # Residualize outcome and treatment - ensure float64 types
            y_residual = np.asarray(y, dtype=np.float64) - np.asarray(preds['outcome_pred'], dtype=np.float64)
            t_residual = np.asarray(treatment, dtype=np.float64) - np.asarray(preds['treatment_pred'], dtype=np.float64)
            
            residuals[name] = {
                'y_residual': y_residual,
                't_residual': t_residual
            }
            
        return residuals
    
    def _train_neural_meta_learner(self, X: np.ndarray, y: np.ndarray, treatment: np.ndarray,
                                  base_predictions: Dict, residuals: Dict) -> np.ndarray:
        """Train the neural meta-learner to predict optimal base learner weights."""
        
        # Prepare training data for meta-learner
        meta_features = self._extract_meta_features(X, y, treatment)
        meta_features_scaled = self.scaler.fit_transform(meta_features)
        meta_features_tensor = torch.FloatTensor(meta_features_scaled)
        
        # Stack base learner predictions
        base_preds_array = np.column_stack([
            preds['outcome_pred'] for preds in base_predictions.values()
        ])
        base_preds_tensor = torch.FloatTensor(base_preds_array)
        
        # Build meta-network
        if self.meta_network is None:
            self.meta_network = MetaNetwork(
                meta_features_scaled.shape[1], 
                self.meta_hidden_dim, 
                len(self.base_learners)
            )
        
        optimizer = optim.Adam(self.meta_network.parameters(), lr=0.001)
        
        # Training loop - optimize for stable causal effect estimation
        n_epochs = 50  # Reduced for efficiency
        best_loss = float('inf')
        patience = 10
        no_improve = 0
        
        for epoch in range(n_epochs):
            optimizer.zero_grad()
            
            # Get learner weights from meta-network
            weights = self.meta_network(meta_features_tensor, base_preds_tensor)
            weights_np = weights.detach().numpy().flatten()
            
            # Compute weighted residuals
            weighted_y_residual = np.zeros_like(y, dtype=np.float64)
            weighted_t_residual = np.zeros_like(treatment, dtype=np.float64)
            
            for i, name in enumerate(residuals.keys()):
                weighted_y_residual += weights_np[i] * residuals[name]['y_residual']
                weighted_t_residual += weights_np[i] * residuals[name]['t_residual']
            
            # Compute treatment effect and stability
            if np.var(weighted_t_residual) > 1e-8:
                treatment_effect = np.cov(weighted_y_residual, weighted_t_residual)[0,1] / np.var(weighted_t_residual)
                effect_residuals = weighted_y_residual - treatment_effect * weighted_t_residual
                effect_stability = 1.0 / (1.0 + np.var(effect_residuals))
            else:
                effect_stability = 0.1  # Low stability if no treatment variation
            
            # Loss: Encourage stable, well-identified treatment effects
            loss = -torch.log(torch.tensor(effect_stability + 1e-8, dtype=torch.float32))
            
            loss.backward()
            optimizer.step()
            
            # Early stopping
            current_loss = loss.item()
            if current_loss < best_loss:
                best_loss = current_loss
                no_improve = 0
            else:
                no_improve += 1
                
            if no_improve >= patience:
                break
                
        return weights.detach().numpy().flatten()
    
    def _train_simple_meta_learner(self, base_performance: Dict, meta_features: np.ndarray) -> np.ndarray:
        """Train simplified meta-learner when neural approach is not available."""
        
        # Extract performance scores
        performance_scores = [
            perf['combined_mse'] for perf in base_performance.values()
        ]
        
        # Learn optimal weights
        optimal_weights = self.simple_meta.learn_weights(performance_scores, meta_features)
        
        return optimal_weights
    
    def estimate_treatment_effect(self, X: np.ndarray, y: np.ndarray, treatment: np.ndarray) -> Dict:
        """
        Estimate treatment effect using Meta-Learning Enhanced Double ML.
        
        Args:
            X: Feature matrix
            y: Outcome variable
            treatment: Treatment indicator
            
        Returns:
            Dict with treatment effect, confidence intervals, and meta-learner insights
        """
        
        # Input validation and type conversion
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        treatment = np.asarray(treatment, dtype=np.float64)
        
        if len(X) != len(y) or len(X) != len(treatment):
            raise ValueError("X, y, and treatment must have the same length")
            
        if len(np.unique(treatment)) < 2:
            warnings.warn("Treatment has less than 2 unique values")
            
        # Step 1: Train base learners
        base_predictions, base_performance = self._train_base_learners(X, y, treatment)
        
        # Step 2: Compute DML residuals for each base learner
        residuals = self._compute_dml_residuals(base_predictions, y, treatment)
        
        # Step 3: Train meta-learner to find optimal combination
        meta_features = self._extract_meta_features(X, y, treatment)
        
        if self.use_neural_meta:
            try:
                optimal_weights = self._train_neural_meta_learner(
                    X, y, treatment, base_predictions, residuals
                )
            except Exception as e:
                warnings.warn(f"Neural meta-learner failed: {str(e)}. Using simple meta-learner.")
                optimal_weights = self._train_simple_meta_learner(base_performance, meta_features)
        else:
            optimal_weights = self._train_simple_meta_learner(base_performance, meta_features)
        
        # Step 4: Compute final treatment effect using optimal weights
        weighted_y_residual = np.zeros_like(y, dtype=np.float64)
        weighted_t_residual = np.zeros_like(treatment, dtype=np.float64)
        
        for i, name in enumerate(residuals.keys()):
            weighted_y_residual += optimal_weights[i] * residuals[name]['y_residual']
            weighted_t_residual += optimal_weights[i] * residuals[name]['t_residual']
        
        # Final treatment effect
        if np.var(weighted_t_residual) > 1e-8:
            treatment_effect = float(np.cov(weighted_y_residual, weighted_t_residual)[0,1] / np.var(weighted_t_residual))
        else:
            treatment_effect = 0.0
            warnings.warn("No variation in treatment residuals - cannot estimate effect")
        
        # Bootstrap confidence intervals
        n_bootstrap = 100  # Reduced for efficiency
        bootstrap_effects = []
        
        for _ in range(n_bootstrap):
            try:
                idx = np.random.choice(len(y), len(y), replace=True)
                boot_y_res = weighted_y_residual[idx]
                boot_t_res = weighted_t_residual[idx]
                
                if np.var(boot_t_res) > 1e-8:
                    boot_effect = np.cov(boot_y_res, boot_t_res)[0,1] / np.var(boot_t_res)
                    if not np.isnan(boot_effect) and np.isfinite(boot_effect):
                        bootstrap_effects.append(float(boot_effect))  # Ensure float conversion
            except:
                continue  # Skip failed bootstrap samples
        
        # Calculate confidence intervals
        if len(bootstrap_effects) >= 10:  # Need minimum samples for CI
            ci_lower, ci_upper = np.percentile(bootstrap_effects, [2.5, 97.5])
        else:
            # Fallback to simple standard error
            se = np.std(weighted_y_residual) / np.sqrt(len(y))
            ci_lower = float(treatment_effect - 1.96 * se)
            ci_upper = float(treatment_effect + 1.96 * se)
        
        # Calculate baseline for relative effect
        baseline_mean = np.mean(y[treatment == 0]) if np.sum(treatment == 0) > 0 else np.mean(y)
        if baseline_mean != 0:
            relative_effect = float((treatment_effect / baseline_mean) * 100)
            rel_ci_lower = float((ci_lower / baseline_mean) * 100) 
            rel_ci_upper = float((ci_upper / baseline_mean) * 100)
        else:
            relative_effect = 0.0
            rel_ci_lower = 0.0
            rel_ci_upper = 0.0
        
        # Test significance
        significance = not (ci_lower <= 0 <= ci_upper)
        
        self.is_fitted = True
        
        # Ensure all values are serializable Python types
        return {
            'point_effect': float(treatment_effect),
            'relative_effect': float(relative_effect),
            'lower_bound': float(rel_ci_lower),
            'upper_bound': float(rel_ci_upper),
            'significance': bool(significance),
            'meta_weights': {k: float(v) for k, v in zip(self.base_learners.keys(), optimal_weights)},
            'base_performance': {k: {kk: float(vv) for kk, vv in v.items()} for k, v in base_performance.items()},
            'method_name': 'Meta-DML',
            'meta_features': [float(x) for x in meta_features.flatten().tolist()],
            'n_bootstrap_samples': int(len(bootstrap_effects)),
            'uses_neural_meta': bool(self.use_neural_meta)
        }


def create_meta_dml_estimator(enhanced_features: bool = True, 
                             use_neural_meta: bool = True) -> MetaLearnerDML:
    """
    Factory function to create a Meta-DML estimator with optimal configuration.
    
    Args:
        enhanced_features: Whether to use enhanced base learners
        use_neural_meta: Whether to use neural meta-learning
        
    Returns:
        Configured MetaLearnerDML instance
    """
    
    if enhanced_features:
        # Enhanced base learners with better hyperparameters
        base_learners = {
            'rf': RandomForestRegressor(
                n_estimators=100, 
                max_depth=10,
                min_samples_split=5,
                random_state=42, 
                n_jobs=-1
            ),
            'gbm': GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                random_state=42
            ),
            'mlp': MLPRegressor(
                hidden_layer_sizes=(64, 32), 
                random_state=42,
                max_iter=500,
                early_stopping=True,
                validation_fraction=0.2,
                alpha=0.01
            ),
            'elastic': ElasticNet(
                alpha=0.1,
                l1_ratio=0.5,
                random_state=42,
                max_iter=2000
            ),
            'ridge': Ridge(alpha=1.0, random_state=42),
            'lasso': Lasso(alpha=0.1, random_state=42, max_iter=2000)
        }
    else:
        base_learners = None  # Use default simple configuration
    
    return MetaLearnerDML(
        base_learners=base_learners,
        meta_hidden_dim=64 if use_neural_meta else 32,
        cv_folds=3,  # Balance between accuracy and efficiency
        use_neural_meta=use_neural_meta
    )