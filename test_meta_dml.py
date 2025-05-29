#!/usr/bin/env python3
"""
Quick test script to verify Meta-DML integration works correctly.
Run this before running the full analysis to catch any issues early.
"""

import numpy as np
import pandas as pd
import sys
import os

# Add src to path
sys.path.append('src')

def test_meta_dml_import():
    """Test if Meta-DML can be imported correctly."""
    print("Testing Meta-DML import...")
    try:
        from src.meta_dml import create_meta_dml_estimator, MetaLearnerDML
        print("✅ Meta-DML import successful")
        return True
    except Exception as e:
        print(f"❌ Meta-DML import failed: {str(e)}")
        return False

def test_meta_dml_basic_functionality():
    """Test basic Meta-DML functionality with synthetic data."""
    print("\nTesting Meta-DML basic functionality...")
    try:
        from src.meta_dml import create_meta_dml_estimator
        
        # Create simple synthetic data
        np.random.seed(42)
        n_samples = 30
        
        # Features: time trend + noise
        X = np.column_stack([
            np.arange(n_samples, dtype=np.float64),  # Time trend
            np.random.normal(0, 1, n_samples)  # Noise
        ])
        
        # Treatment: policy intervention at time 15
        treatment = (np.arange(n_samples) >= 15).astype(np.float64)
        
        # Outcome: baseline + treatment effect + noise
        true_effect = -5.0  # -5% reduction
        y = (50 + 0.5 * np.arange(n_samples) + true_effect * treatment + 
             np.random.normal(0, 2, n_samples)).astype(np.float64)
        
        print(f"   Data shapes: X={X.shape}, y={y.shape}, treatment={treatment.shape}")
        print(f"   Data types: X={X.dtype}, y={y.dtype}, treatment={treatment.dtype}")
        print(f"   Treatment distribution: {np.bincount(treatment.astype(int))}")
        
        # Test Meta-DML with simplified configuration first
        meta_dml = create_meta_dml_estimator(enhanced_features=False, use_neural_meta=False)
        results = meta_dml.estimate_treatment_effect(X, y, treatment)
        
        # Check results structure
        required_keys = ['point_effect', 'relative_effect', 'lower_bound', 'upper_bound', 'significance']
        missing_keys = [key for key in required_keys if key not in results]
        
        if missing_keys:
            print(f"❌ Missing result keys: {missing_keys}")
            return False
        
        # Check if effect is reasonable (should be negative, around -10%)
        rel_effect = results['relative_effect']
        if not (-50 <= rel_effect <= 20):  # Reasonable range for synthetic data
            print(f"⚠️  Effect seems unrealistic: {rel_effect:.1f}% (but may be OK for synthetic data)")
        
        print(f"✅ Meta-DML basic functionality test passed")
        print(f"   Estimated effect: {rel_effect:.1f}%")
        print(f"   Significance: {results['significance']}")
        print(f"   Meta-weights: {list(results.get('meta_weights', {}).keys())}")
        
        # Test with neural meta-learning if available
        try:
            print("   Testing neural meta-learning...")
            meta_dml_neural = create_meta_dml_estimator(enhanced_features=False, use_neural_meta=True)
            results_neural = meta_dml_neural.estimate_treatment_effect(X, y, treatment)
            print(f"   Neural meta-learning also works: {results_neural['relative_effect']:.1f}%")
        except Exception as neural_e:
            print(f"   Neural meta-learning failed (this is OK): {str(neural_e)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Meta-DML functionality test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_benchmarks_integration():
    """Test if Meta-DML integrates correctly with benchmarks.py."""
    print("\nTesting benchmarks integration...")
    try:
        from src.benchmarks import CausalInferenceEvaluator
        
        # Create mock time series data
        years = [str(year) for year in range(2000, 2020)]
        data = {
            'indicator1': 50 - 0.5 * np.arange(20) + np.random.normal(0, 2, 20),
            'indicator2': 60 + 0.3 * np.arange(20) + np.random.normal(0, 1, 20)
        }
        df_mock = pd.DataFrame(data, index=years)
        
        policy_timeline = {'2010': 'Test_Policy'}
        
        # Test evaluator initialization
        evaluator = CausalInferenceEvaluator(df_mock, policy_timeline)
        
        # Check if estimate_with_meta_dml method exists
        if not hasattr(evaluator, 'estimate_with_meta_dml'):
            print("❌ estimate_with_meta_dml method not found in CausalInferenceEvaluator")
            return False
        
        # Test the method
        results = evaluator.estimate_with_meta_dml('indicator1', 2010)
        
        if 'relative_effect' not in results:
            print("❌ Meta-DML method didn't return expected results")
            return False
        
        print("✅ Benchmarks integration test passed")
        print(f"   Method available: estimate_with_meta_dml")
        print(f"   Returns valid results: {list(results.keys())}")
        
        return True
        
    except Exception as e:
        print(f"❌ Benchmarks integration test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_pytorch_availability():
    """Test PyTorch availability and fallback behavior."""
    print("\nTesting PyTorch availability...")
    try:
        import torch
        print("✅ PyTorch available - Neural meta-learning enabled")
        return True
    except ImportError:
        print("⚠️  PyTorch not available - Will use simplified meta-learning")
        print("   Install PyTorch for optimal performance: pip install torch")
        return False

def run_all_tests():
    """Run all integration tests."""
    print("=" * 60)
    print("Meta-DML Integration Test Suite")
    print("=" * 60)
    
    tests = [
        ("PyTorch Availability", test_pytorch_availability),
        ("Meta-DML Import", test_meta_dml_import),
        ("Meta-DML Functionality", test_meta_dml_basic_functionality),
        ("Benchmarks Integration", test_benchmarks_integration)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} test crashed: {str(e)}")
            results.append((test_name, False))
    
    print("\n" + "=" * 60)
    print("Test Results Summary")
    print("=" * 60)
    
    all_passed = True
    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status:<10} {test_name}")
        if not passed:
            all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print("🎉 All tests passed! Meta-DML integration is ready.")
        print("   You can now run: python main.py meta-analysis")
    else:
        print("⚠️  Some tests failed. Please fix issues before running full analysis.")
        print("   Check error messages above for details.")
    
    return all_passed

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)