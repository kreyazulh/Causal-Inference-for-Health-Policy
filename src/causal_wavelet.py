# src/causal_wavelet.py - Causally-Informed Wavelet Analysis

import numpy as np
import pandas as pd
import pywt
import os
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde, norm
from sklearn.preprocessing import StandardScaler
import warnings

class CausalWaveletAnalysis:
    """
    A novel approach that integrates wavelet decomposition with causal inference
    to detect and attribute health milestones to policy interventions.
    
    This class performs wavelet decomposition while accounting for known intervention
    points in the causal structure, creating intervention-aware wavelet representations.
    """
    
    def __init__(self, time_series_data, policy_timeline, wavelet_type='db4', level=3, threshold_factor=0.7):
        """
        Initialize the causal wavelet analysis.
        
        Parameters:
        -----------
        time_series_data : pandas.DataFrame
            Time series data with years as index and indicators as columns
        policy_timeline : dict
            Dictionary mapping policy years to policy names
        wavelet_type : str, optional
            Wavelet type to use for decomposition
        level : int, optional
            Maximum level of wavelet decomposition
        threshold_factor : float, optional
            Threshold factor for change point detection
        """
        self.data = time_series_data
        self.policy_timeline = policy_timeline
        self.wavelet_type = wavelet_type
        self.max_level = level
        self.threshold_factor = threshold_factor
        
        # Initialize result containers
        self.wavelet_coeffs = {}
        self.causal_coeffs = {}
        self.change_points = {}
        self.causal_change_points = {}
        self.causal_attribution = {}
        
        # Convert policy years to integers
        self.policy_years = [int(year) for year in policy_timeline.keys()]
        
    def standard_wavelet_decomposition(self, indicator):
        """
        Perform standard wavelet decomposition on a time series.
        
        Parameters:
        -----------
        indicator : str
            Name of the health indicator
            
        Returns:
        --------
        tuple
            Approximation and details coefficients
        """
        # Extract time series data
        time_series = self.data[indicator].values
        
        # Check if we have enough data for the requested level
        max_level = pywt.dwt_max_level(len(time_series), pywt.Wavelet(self.wavelet_type).dec_len)
        actual_level = min(self.max_level, max_level)
        
        if actual_level < self.max_level:
            warnings.warn(f"Requested level {self.max_level} exceeds maximum possible level {max_level}. "
                         f"Using level {actual_level} for indicator {indicator}.")
        
        # Perform wavelet decomposition
        coeffs = pywt.wavedec(time_series, self.wavelet_type, level=actual_level)
        
        # Store coefficients
        self.wavelet_coeffs[indicator] = coeffs
        
        # Return approximation and details
        return coeffs[0], coeffs[1:]
    
    def align_coefficients_to_interventions(self, coeffs, level):
        """
        Adjust wavelet coefficients based on intervention timing and scale.
        
        Parameters:
        -----------
        coeffs : numpy.ndarray
            Wavelet coefficients at a specific level
        level : int
            Decomposition level
            
        Returns:
        --------
        numpy.ndarray
            Intervention-aligned coefficients
        """
        # Get years from data
        years = np.array([int(year) for year in self.data.index])
        
        # Calculate scale factor for this level
        scale_factor = 2**level
        
        # Length of the coefficients array
        coeff_len = len(coeffs)
        
        # Create array to store aligned coefficients
        aligned_coeffs = coeffs.copy()
        
        # Calculate years corresponding to each coefficient
        # In wavelet decomposition, each coefficient at level j corresponds to 2^j original points
        coeff_years = []
        for i in range(coeff_len):
            # Calculate the range of years this coefficient influences
            start_idx = i * scale_factor
            end_idx = min((i + 1) * scale_factor, len(years))
            
            # Get the middle year in this range
            if start_idx < len(years) and end_idx > start_idx:
                mid_year = years[start_idx:end_idx].mean()
                coeff_years.append(mid_year)
            else:
                # Handle edge case
                coeff_years.append(None)
        
        # Adjust coefficients based on proximity to policy interventions
        for i, year in enumerate(coeff_years):
            if year is not None:
                # Find closest policy year
                closest_policy = min(self.policy_years, key=lambda y: abs(y - year))
                
                # Calculate temporal proximity (exponential decay with distance)
                temporal_proximity = np.exp(-0.5 * (abs(year - closest_policy) / scale_factor)**2)
                
                # Amplify coefficient if it's close to a policy intervention
                # This makes the wavelet decomposition more sensitive to changes near policies
                aligned_coeffs[i] *= (1 + temporal_proximity)
        
        return aligned_coeffs
    
    def causal_wavelet_decomposition(self, indicator):
        """
        Perform wavelet decomposition while accounting for known intervention points.
        
        Parameters:
        -----------
        indicator : str
            Name of the health indicator
            
        Returns:
        --------
        tuple
            Standard coefficients, causally-aligned coefficients, and causal attribution
        """
        # First perform standard wavelet decomposition
        approx, details = self.standard_wavelet_decomposition(indicator)
        
        # Store standard coefficients
        std_coeffs = [approx] + details
        
        # Calculate intervention-aligned wavelet coefficients
        intervention_aligned_coeffs = []
        intervention_aligned_coeffs.append(approx)  # Keep approximation as is
        
        for i, detail in enumerate(details):
            # Adjust coefficients based on intervention timing
            aligned_coeff = self.align_coefficients_to_interventions(detail, i + 1)
            intervention_aligned_coeffs.append(aligned_coeff)
        
        # Store causally-aligned coefficients
        self.causal_coeffs[indicator] = intervention_aligned_coeffs
        
        # Calculate causal attribution scores
        causal_attribution = self.compute_causal_attribution(indicator, std_coeffs, intervention_aligned_coeffs)
        
        return std_coeffs, intervention_aligned_coeffs, causal_attribution
    
    def compute_causal_attribution(self, indicator, std_coeffs, causal_coeffs):
        """
        Calculate causal attribution scores for each policy intervention.
        
        Parameters:
        -----------
        indicator : str
            Name of the health indicator
        std_coeffs : list
            Standard wavelet coefficients
        causal_coeffs : list
            Causally-aligned wavelet coefficients
            
        Returns:
        --------
        dict
            Dictionary with causal attribution scores for each policy
        """
        # Get years from data
        years = np.array([int(year) for year in self.data.index])
        
        # Initialize attribution scores
        attribution_scores = {}
        for policy_year in self.policy_years:
            attribution_scores[policy_year] = 0.0
        
        # Calculate energy in standard and causal coefficients
        std_energy = 0
        causal_energy = 0
        
        for level in range(1, len(std_coeffs)):
            std_energy += np.sum(std_coeffs[level]**2)
            causal_energy += np.sum(causal_coeffs[level]**2)
            
            # Calculate energy increase at this level
            energy_increase = causal_energy - std_energy
            
            # Calculate coefficient years for this level
            scale_factor = 2**level
            coeff_len = len(std_coeffs[level])
            
            for policy_year in self.policy_years:
                # Find coefficient index closest to policy year
                year_indices = np.where((years >= policy_year - scale_factor) & 
                                    (years <= policy_year + scale_factor))[0]
                
                if len(year_indices) > 0:
                    # Calculate mean coefficient value around policy year
                    # FIX: Properly handle the boolean condition
                    coeff_indices = []
                    for i in range(coeff_len):
                        if i * scale_factor < len(years):
                            # Check if any year is within scale_factor distance of policy_year
                            if i * scale_factor < len(years):
                                year_at_index = years[i * scale_factor]
                                if abs(year_at_index - policy_year) <= scale_factor:
                                    coeff_indices.append(i)
                    
                    if coeff_indices:  # If we found valid indices
                        # Calculate energy in standard and causal coefficients near policy
                        std_policy_energy = np.sum([std_coeffs[level][i]**2 for i in coeff_indices])
                        causal_policy_energy = np.sum([causal_coeffs[level][i]**2 for i in coeff_indices])
                        
                        # Attribution is proportional to energy increase near policy
                        if energy_increase > 0:
                            attribution = (causal_policy_energy - std_policy_energy) / energy_increase
                            attribution_scores[policy_year] += max(0, attribution)
                            
        # Normalize attribution scores
        total_attribution = sum(attribution_scores.values())
        if total_attribution > 0:
            for policy_year in attribution_scores:
                attribution_scores[policy_year] /= total_attribution
        
        # Store causal attribution
        self.causal_attribution[indicator] = attribution_scores
        
        return attribution_scores
    
    def detect_standard_change_points(self, indicator):
        """
        Detect potential change points using standard wavelet coefficients.
        
        Parameters:
        -----------
        indicator : str
            Name of the health indicator
            
        Returns:
        --------
        list
            List of tuples (year_idx, level, strength) for detected change points
        """
        # Check if wavelet decomposition has been performed
        if indicator not in self.wavelet_coeffs:
            self.standard_wavelet_decomposition(indicator)
            
        # Get approximation and details
        coeffs = self.wavelet_coeffs[indicator]
        approx, details = coeffs[0], coeffs[1:]
        
        # Detect change points
        change_points = []
        
        for level, detail in enumerate(details):
            try:
                # Use peak detection on absolute coefficient values
                peaks, properties = find_peaks(np.abs(detail), 
                                             height=self.threshold_factor * np.std(detail),
                                             distance=2)  # Minimum distance between peaks
                
                # Scale peak locations back to original time scale
                scale_factor = len(self.data[indicator]) / len(detail)
                
                for i, peak in enumerate(peaks):
                    scaled_peak = int(peak * scale_factor)
                    if scaled_peak < len(self.data[indicator]):
                        # Calculate peak strength based on height
                        strength = properties['peak_heights'][i] / np.std(detail)
                        change_points.append((scaled_peak, level + 1, strength))
                        
            except Exception as e:
                warnings.warn(f"Error detecting peaks at level {level} for {indicator}: {str(e)}")
                
        # Sort change points by strength
        change_points.sort(key=lambda x: x[2], reverse=True)
        
        # Store change points
        self.change_points[indicator] = change_points
        
        return change_points
    
    def detect_causal_change_points(self, indicator):
        """
        Detect potential change points using causally-aligned wavelet coefficients.
        
        Parameters:
        -----------
        indicator : str
            Name of the health indicator
            
        Returns:
        --------
        list
            List of tuples (year_idx, level, strength, policy_attribution) for detected change points
        """
        # Check if causal wavelet decomposition has been performed
        if indicator not in self.causal_coeffs:
            self.causal_wavelet_decomposition(indicator)
            
        # Get causally-aligned coefficients
        coeffs = self.causal_coeffs[indicator]
        approx, details = coeffs[0], coeffs[1:]
        
        # Get years
        years = np.array([int(year) for year in self.data.index])
        
        # Detect change points
        change_points = []
        
        for level, detail in enumerate(details):
            try:
                # Use peak detection on absolute coefficient values
                peaks, properties = find_peaks(np.abs(detail), 
                                             height=self.threshold_factor * np.std(detail),
                                             distance=2)  # Minimum distance between peaks
                
                # Scale peak locations back to original time scale
                scale_factor = len(self.data[indicator]) / len(detail)
                
                for i, peak in enumerate(peaks):
                    scaled_peak = int(peak * scale_factor)
                    if scaled_peak < len(self.data[indicator]):
                        # Calculate peak strength based on height
                        strength = properties['peak_heights'][i] / np.std(detail)
                        
                        # Find the year corresponding to this peak
                        if scaled_peak < len(years):
                            peak_year = years[scaled_peak]
                            
                            # Find closest policy and calculate attribution
                            closest_policy = min(self.policy_years, key=lambda y: abs(y - peak_year))
                            temporal_distance = abs(peak_year - closest_policy)
                            
                            # Calculate policy attribution based on temporal proximity
                            # and the causal attribution scores
                            if indicator in self.causal_attribution:
                                policy_attribution = self.causal_attribution[indicator].get(closest_policy, 0)
                                
                                # Adjust attribution based on temporal distance
                                temporal_factor = np.exp(-0.5 * (temporal_distance / (2**level))**2)
                                policy_attribution *= temporal_factor
                                
                                change_points.append((scaled_peak, level + 1, strength, policy_attribution))
                            else:
                                change_points.append((scaled_peak, level + 1, strength, 0.0))
                        
            except Exception as e:
                warnings.warn(f"Error detecting causal peaks at level {level} for {indicator}: {str(e)}")
                
        # Sort change points by combined strength and policy attribution
        change_points.sort(key=lambda x: x[2] * (1 + x[3]), reverse=True)
        
        # Store causal change points
        self.causal_change_points[indicator] = change_points
        
        return change_points
    
    def calculate_milestone_probability(self, indicators=None, use_causal=True):
        """
        Calculate milestone probability across multiple indicators.
        
        Parameters:
        -----------
        indicators : list, optional
            List of indicators to include. If None, use all indicators.
        use_causal : bool, optional
            Whether to use causally-aligned change points.
            
        Returns:
        --------
        dict
            Dictionary with milestone probabilities for each year
        """
        if indicators is None:
            indicators = self.data.columns
            
        # Get years
        years = np.array([int(year) for year in self.data.index])
        
        # Initialize milestone probabilities
        milestone_probs = {year: 0.0 for year in years}
        
        # Collect change points across indicators
        for indicator in indicators:
            # Detect change points if not already done
            if use_causal:
                if indicator not in self.causal_change_points:
                    self.detect_causal_change_points(indicator)
                change_points = self.causal_change_points[indicator]
            else:
                if indicator not in self.change_points:
                    self.detect_standard_change_points(indicator)
                change_points = self.change_points[indicator]
            
            # Add contribution from each change point
            for cp in change_points:
                year_idx = cp[0]
                if year_idx < len(years):
                    year = years[year_idx]
                    
                    # For standard change points, use only strength
                    if use_causal and len(cp) > 3:
                        # For causal change points, use strength and policy attribution
                        milestone_probs[year] += cp[2] * (1 + cp[3])
                    else:
                        milestone_probs[year] += cp[2]
        
        # Normalize probabilities
        max_prob = max(milestone_probs.values()) if milestone_probs else 1.0
        if max_prob > 0:
            for year in milestone_probs:
                milestone_probs[year] /= max_prob
                
        return milestone_probs
    
    def identify_top_milestones(self, indicators=None, use_causal=True, top_n=5):
        """
        Identify top health milestones based on joint probability.
        
        Parameters:
        -----------
        indicators : list, optional
            List of indicators to include. If None, use all indicators.
        use_causal : bool, optional
            Whether to use causally-aligned change points.
        top_n : int, optional
            Number of top milestones to return.
            
        Returns:
        --------
        list
            List of dictionaries with milestone information
        """
        # Calculate milestone probabilities
        milestone_probs = self.calculate_milestone_probability(indicators, use_causal)
        
        # Get years
        years = np.array([int(year) for year in self.data.index])
        
        # Create milestone data
        milestones = []
        for year, prob in milestone_probs.items():
            # Find nearest policy
            nearest_policy_year = min(self.policy_years, key=lambda y: abs(y - year))
            time_diff = abs(nearest_policy_year - year)
            
            # Calculate causal attribution based on temporal proximity
            if time_diff == 0:
                causal_score = 1.0  # Perfect match
            elif time_diff <= 1:
                causal_score = 0.9  # Strong causal link
            elif time_diff <= 2:
                causal_score = 0.7  # Moderate causal link
            elif time_diff <= 4:
                causal_score = 0.5  # Possible causal link
            else:
                causal_score = 0.3  # Weak causal link
                
            # Add additional causal attribution from wavelet analysis if using causal change points
            if use_causal:
                # Average causal attribution across indicators
                wavelet_attribution = 0.0
                count = 0
                
                for indicator in indicators:
                    if indicator in self.causal_attribution:
                        attribution = self.causal_attribution[indicator].get(nearest_policy_year, 0.0)
                        wavelet_attribution += attribution
                        count += 1
                        
                if count > 0:
                    wavelet_attribution /= count
                    
                    # Combine temporal causal score with wavelet attribution
                    causal_score = 0.5 * causal_score + 0.5 * wavelet_attribution
            
            # Calculate composite score
            composite_score = (0.6 * prob) + (0.4 * causal_score)
            
            milestones.append({
                'Year': year,
                'Joint_Probability': prob,
                'Nearest_Policy_Year': nearest_policy_year,
                'Nearest_Policy': self.policy_timeline.get(str(nearest_policy_year), f"Policy {nearest_policy_year}"),
                'Time_Difference': time_diff,
                'Causal_Attribution': causal_score,
                'Composite_Score': composite_score
            })
        
        # Sort by composite score and select top N
        milestones.sort(key=lambda x: x['Composite_Score'], reverse=True)
        top_milestones = milestones[:top_n]
        
        return top_milestones
    
    def visualize_causal_wavelet_decomposition(self, indicator, figsize=(14, 10)):
        """
        Create visualization of causally-informed wavelet decomposition.
        
        Parameters:
        -----------
        indicator : str
            Name of the health indicator
        figsize : tuple, optional
            Figure size
            
        Returns:
        --------
        matplotlib.figure.Figure
            Figure object
        """
        # Check if causal wavelet decomposition has been performed
        if indicator not in self.causal_coeffs:
            self.causal_wavelet_decomposition(indicator)
            
        # Get standard and causal coefficients
        std_coeffs = self.wavelet_coeffs[indicator]
        causal_coeffs = self.causal_coeffs[indicator]
        
        # Get years
        years = np.array([int(year) for year in self.data.index])
        
        # Create figure
        fig, axes = plt.subplots(len(std_coeffs), 2, figsize=figsize)
        
        # Plot original time series
        axes[0, 0].plot(years, self.data[indicator], 'b-')
        axes[0, 0].set_title(f"Original Time Series: {indicator}")
        axes[0, 0].set_ylabel("Value")
        
        # Add policy lines
        for policy_year in self.policy_years:
            axes[0, 0].axvline(x=policy_year, color='r', linestyle='--', alpha=0.5)
            
        # Plot approximation coefficients
        axes[0, 1].plot(std_coeffs[0], 'b-', label='Standard')
        axes[0, 1].plot(causal_coeffs[0], 'g-', label='Causal')
        axes[0, 1].set_title("Approximation Coefficients")
        axes[0, 1].legend()
        
        # Plot detail coefficients for each level
        for i in range(1, len(std_coeffs)):
            # Standard coefficients
            axes[i, 0].plot(std_coeffs[i], 'b-')
            axes[i, 0].set_title(f"Standard Detail Coefficients (Level {i})")
            axes[i, 0].set_ylabel("Coefficient Value")
            
            # Causal coefficients
            axes[i, 1].plot(causal_coeffs[i], 'g-')
            axes[i, 1].set_title(f"Causal Detail Coefficients (Level {i})")
            
            # Add marker for policy-influenced coefficients
            scale_factor = 2**i
            for policy_year in self.policy_years:
                # Find coefficient index closest to policy year
                policy_idx = np.argmin(abs(years - policy_year))
                coeff_idx = policy_idx // scale_factor
                
                if coeff_idx < len(std_coeffs[i]):
                    axes[i, 0].plot(coeff_idx, std_coeffs[i][coeff_idx], 'ro', markersize=8)
                    axes[i, 1].plot(coeff_idx, causal_coeffs[i][coeff_idx], 'ro', markersize=8)
        
        # Adjust layout
        plt.tight_layout()
        
        return fig
    
    def visualize_change_points(self, indicator, figsize=(12, 6), use_causal=True):
        """
        Visualize detected change points for an indicator.
        
        Parameters:
        -----------
        indicator : str
            Name of the health indicator
        figsize : tuple, optional
            Figure size
        use_causal : bool, optional
            Whether to use causally-aligned change points
            
        Returns:
        --------
        matplotlib.figure.Figure
            Figure object
        """
        # Detect change points if not already done
        if use_causal:
            if indicator not in self.causal_change_points:
                self.detect_causal_change_points(indicator)
            change_points = self.causal_change_points[indicator]
            title = f"Causally-Informed Change Points: {indicator}"
        else:
            if indicator not in self.change_points:
                self.detect_standard_change_points(indicator)
            change_points = self.change_points[indicator]
            title = f"Standard Change Points: {indicator}"
        
        # Get years and indicator data
        years = np.array([int(year) for year in self.data.index])
        indicator_data = self.data[indicator].values
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot indicator data
        ax.plot(years, indicator_data, 'b-', linewidth=2, label=indicator)
        
        # Plot change points
        for cp in change_points:
            year_idx = cp[0]
            level = cp[1]
            strength = cp[2]
            
            if year_idx < len(years):
                year = years[year_idx]
                
                # Use different colors for different levels
                if level == 1:
                    color = 'g'
                    label = "Level 1 Change" if level == 1 and year_idx == change_points[0][0] else None
                else:
                    color = 'orange'
                    label = "Level 2+ Change" if level > 1 and year_idx == change_points[0][0] else None
                
                # Plot vertical line for change point
                ax.axvline(x=year, color=color, linestyle='--', alpha=0.7, label=label)
                
                # Add annotation for strong change points
                if strength > 2.0:
                    ax.annotate(f"Year {year}", xy=(year, min(indicator_data) + 0.1 * (max(indicator_data) - min(indicator_data))),
                               xytext=(0, -20), textcoords='offset points', ha='center',
                               arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
        
        # Add policy lines
        for policy_year in self.policy_years:
            policy_name = self.policy_timeline.get(str(policy_year), f"Policy {policy_year}")
            label = "Policy Implementation" if policy_year == self.policy_years[0] else None
            
            ax.axvline(x=policy_year, color='r', linestyle='-.', alpha=0.5, label=label)
            
            # Add policy annotation
            short_name = policy_name.split()[0]
            ax.annotate(f"{short_name} ({policy_year})", xy=(policy_year, max(indicator_data) - 0.1 * (max(indicator_data) - min(indicator_data))),
                       xytext=(0, 20), textcoords='offset points', ha='center', rotation=90,
                       bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.7))
        
        # Set title and labels
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel("Year", fontsize=12)
        ax.set_ylabel(indicator, fontsize=12)
        
        # Add legend
        ax.legend(loc='best')
        
        # Add grid
        ax.grid(True, alpha=0.3, linestyle=':')
        
        # Adjust layout
        plt.tight_layout()
        
        return fig
    
    def visualize_milestone_probability(self, indicators=None, use_causal=True, figsize=(12, 6)):
        """
        Visualize milestone probability across years.
        
        Parameters:
        -----------
        indicators : list, optional
            List of indicators to include. If None, use all indicators.
        use_causal : bool, optional
            Whether to use causally-aligned change points.
        figsize : tuple, optional
            Figure size
            
        Returns:
        --------
        matplotlib.figure.Figure
            Figure object
        """
        # Calculate milestone probabilities
        milestone_probs = self.calculate_milestone_probability(indicators, use_causal)
        
        # Get years
        years = list(milestone_probs.keys())
        probs = list(milestone_probs.values())
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot milestone probabilities
        ax.plot(years, probs, 'b-', linewidth=2, marker='o', markersize=5, label='Milestone Probability')
        
        # Add policy lines
        for policy_year in self.policy_years:
            policy_name = self.policy_timeline.get(str(policy_year), f"Policy {policy_year}")
            label = "Policy Implementation" if policy_year == self.policy_years[0] else None
            
            ax.axvline(x=policy_year, color='r', linestyle='-.', alpha=0.5, label=label)
            
            # Add policy annotation for important policies
            if policy_year in [1982, 1998, 2011]:  # Example key years
                short_name = " ".join(policy_name.split()[:2])
                ax.annotate(f"{short_name}...", xy=(policy_year, 0.05),
                           xytext=(0, 20), textcoords='offset points', ha='center', rotation=90,
                           bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.7))
        
        # Highlight top 3 milestones
        top_milestones = self.identify_top_milestones(indicators, use_causal, top_n=3)
        
        for milestone in top_milestones:
            year = milestone['Year']
            prob = milestone['Joint_Probability']
            
            # Highlight milestone
            ax.scatter(year, prob, s=120, color='orange', zorder=5, edgecolor='white', linewidth=1.5, label='Significant Milestone' if milestone == top_milestones[0] else None)
            
            # Add annotation
            policy_text = milestone['Nearest_Policy'].split()[:2]
            policy_text = " ".join(policy_text) + "..."
            
            ax.annotate(f"{year}\n({policy_text})", xy=(year, prob),
                       xytext=(0, 15), textcoords='offset points', ha='center',
                       bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='orange', alpha=0.9),
                       fontsize=9)
        
        # Set title and labels
        method_text = "Causally-Informed" if use_causal else "Standard"
        ax.set_title(f"{method_text} Health Milestone Detection", fontsize=14, fontweight='bold')
        ax.set_xlabel("Year", fontsize=12)
        ax.set_ylabel("Joint Probability", fontsize=12)
        
        # Add legend
        ax.legend(loc='upper right', frameon=True, framealpha=0.9)
        
        # Add grid
        ax.grid(True, alpha=0.3, linestyle=':')
        
        # Adjust y-limits
        ax.set_ylim(0, 1.1)
        
        # Adjust layout
        plt.tight_layout()
        
        return fig
    
    def visualize_policy_attribution(self, indicator, figsize=(10, 6)):
        """
        Visualize policy attribution for an indicator.
        
        Parameters:
        -----------
        indicator : str
            Name of the health indicator
        figsize : tuple, optional
            Figure size
            
        Returns:
        --------
        matplotlib.figure.Figure
            Figure object
        """
        # Check if causal wavelet decomposition has been performed
        if indicator not in self.causal_attribution:
            self.causal_wavelet_decomposition(indicator)
            
        # Get causal attribution
        attribution = self.causal_attribution[indicator]
        
        # Sort policies by year
        policy_years = sorted(attribution.keys())
        attribution_values = [attribution[year] for year in policy_years]
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create policy labels
        policy_labels = []
        for year in policy_years:
            policy_name = self.policy_timeline.get(str(year), f"Policy {year}")
            short_name = " ".join(policy_name.split()[:2])
            policy_labels.append(f"{short_name}... ({year})")
        
        # Plot attribution as bar chart
        bars = ax.barh(policy_labels, attribution_values, height=0.6)
        
        # Color bars based on attribution value
        for i, bar in enumerate(bars):
            if attribution_values[i] > 0.3:
                bar.set_color('g')
            elif attribution_values[i] > 0.1:
                bar.set_color('orange')
            else:
                bar.set_color('r')
                
        # Set title and labels
        ax.set_title(f"Policy Attribution for {indicator}", fontsize=14, fontweight='bold')
        ax.set_xlabel("Causal Attribution Score", fontsize=12)
        ax.set_ylabel("Policy", fontsize=12)
        
        # Add grid
        ax.grid(True, alpha=0.3, linestyle=':', axis='x')
        
        # Set x-limits
        ax.set_xlim(0, max(attribution_values) * 1.1)
        
        # Adjust layout
        plt.tight_layout()
        
        return fig
    
    def analyze_all_indicators(self, indicators=None, use_causal=True, output_dir='outputs/'):
        """
        Perform full analysis on all indicators.
        
        Parameters:
        -----------
        indicators : list, optional
            List of indicators to analyze. If None, analyze all indicators.
        use_causal : bool, optional
            Whether to use causally-aligned change points.
        output_dir : str, optional
            Directory to save output files
            
        Returns:
        --------
        dict
            Dictionary with analysis results
        """
        if indicators is None:
            indicators = self.data.columns
            
        # Ensure output directory exists
        os.makedirs(f"{output_dir}/figures", exist_ok=True)
        os.makedirs(f"{output_dir}/results", exist_ok=True)
        
        # Results container
        results = {}
        
        # Analyze each indicator
        for indicator in indicators:
            print(f"Analyzing {indicator}...")
            
            try:
                # Perform causal wavelet decomposition
                if use_causal:
                    self.causal_wavelet_decomposition(indicator)
                    change_points = self.detect_causal_change_points(indicator)
                else:
                    self.standard_wavelet_decomposition(indicator)
                    change_points = self.detect_standard_change_points(indicator)
                
                # Generate visualizations
                fig1 = self.visualize_change_points(indicator, use_causal=use_causal)
                fig1.savefig(f"{output_dir}/figures/changepoints_{indicator.replace(' ', '_')}.png", dpi=300, bbox_inches='tight')
                plt.close(fig1)
                
                if use_causal:
                    fig2 = self.visualize_causal_wavelet_decomposition(indicator)
                    fig2.savefig(f"{output_dir}/figures/causal_wavelet_{indicator.replace(' ', '_')}.png", dpi=300, bbox_inches='tight')
                    plt.close(fig2)
                    
                    fig3 = self.visualize_policy_attribution(indicator)
                    fig3.savefig(f"{output_dir}/figures/policy_attribution_{indicator.replace(' ', '_')}.png", dpi=300, bbox_inches='tight')
                    plt.close(fig3)
                
                # Store results
                results[indicator] = {
                    'change_points': change_points,
                    'causal_attribution': self.causal_attribution.get(indicator, {}) if use_causal else None
                }
                
            except Exception as e:
                print(f"Error analyzing {indicator}: {str(e)}")
        
        # Calculate and visualize milestone probabilities
        milestone_probs = self.calculate_milestone_probability(indicators, use_causal)
        top_milestones = self.identify_top_milestones(indicators, use_causal)
        
        fig = self.visualize_milestone_probability(indicators, use_causal)
        fig.savefig(f"{output_dir}/figures/milestone_probability.png", dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        # Save results to CSV
        milestone_df = pd.DataFrame(top_milestones)
        milestone_df.to_csv(f"{output_dir}/results/milestone_summary.csv", index=False)
        
        # Create summary dataframe
        attribution_data = []
        for indicator in indicators:
            if indicator in self.causal_attribution:
                for policy_year, attribution in self.causal_attribution[indicator].items():
                    attribution_data.append({
                        'Indicator': indicator,
                        'Policy_Year': policy_year,
                        'Policy_Name': self.policy_timeline.get(str(policy_year), f"Policy {policy_year}"),
                        'Attribution_Score': attribution
                    })
        
        if attribution_data:
            attribution_df = pd.DataFrame(attribution_data)
            attribution_df.to_csv(f"{output_dir}/results/policy_attribution.csv", index=False)
        
        # Return results
        results['milestone_probabilities'] = milestone_probs
        results['top_milestones'] = top_milestones
        
        return results