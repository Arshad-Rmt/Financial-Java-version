"""
validate_onnx.py

Comprehensive validation script comparing predictions and probabilities between
the original Python hybrid model (model.pkl) and the converted ONNX models:
- random_forest.onnx
- xgboost.onnx
- lightgbm.onnx

Tests:
1. Manual test sample from 4_hybrid_model.py:
   [-0.25, 0.94, 0, 0.45, 0.06, -1.0, 0.18, -0.11, -3.2]
2. All 1,364 test samples from data.csv (stratified 20% test split, random_state=42).
"""

import os
import joblib
import numpy as np
import pandas as pd
import onnxruntime as rt
from sklearn.model_selection import train_test_split


def run_onnx_inference(sess_rf, sess_xgb, sess_lgbm, X):
    """
    Simulates the Java Spring Boot inference and weighted soft voting logic.
    Weights: RF=1, XGB=2, LGBM=2. Total weight = 5.
    Threshold: 0.55
    """
    X_f32 = np.ascontiguousarray(X, dtype=np.float32)
    
    # 1. Random Forest inference
    # Output: index 1 is probabilities tensor [N, 2]
    rf_out = sess_rf.run(None, {'float_input': X_f32})
    rf_prob_class1 = rf_out[1][:, 1]

    # 2. XGBoost inference
    xgb_out = sess_xgb.run(None, {'float_input': X_f32})
    xgb_prob_class1 = xgb_out[1][:, 1]

    # 3. LightGBM inference
    lgbm_out = sess_lgbm.run(None, {'float_input': X_f32})
    lgbm_prob_class1 = lgbm_out[1][:, 1]

    # 4. Weighted soft-voting: [1, 2, 2]
    ensemble_prob_class1 = (1.0 * rf_prob_class1 + 2.0 * xgb_prob_class1 + 2.0 * lgbm_prob_class1) / 5.0
    
    # 5. Threshold 0.55
    ensemble_pred = (ensemble_prob_class1 >= 0.55).astype(int)

    return {
        'rf_prob': rf_prob_class1,
        'xgb_prob': xgb_prob_class1,
        'lgbm_prob': lgbm_prob_class1,
        'ensemble_prob': ensemble_prob_class1,
        'ensemble_pred': ensemble_pred
    }


def validate():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    model_path = os.path.join(current_dir, "model.pkl")
    onnx_dir = os.path.join(project_root, "onnx-models")

    rf_onnx_path = os.path.join(onnx_dir, "random_forest.onnx")
    xgb_onnx_path = os.path.join(onnx_dir, "xgboost.onnx")
    lgbm_onnx_path = os.path.join(onnx_dir, "lightgbm.onnx")

    for p in [model_path, rf_onnx_path, xgb_onnx_path, lgbm_onnx_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing required file: {p}")

    print("Loading original model.pkl...")
    hybrid = joblib.load(model_path)
    rf_py = hybrid.named_estimators_['rf']
    xgb_py = hybrid.named_estimators_['xgb']
    lgbm_py = hybrid.named_estimators_['lgbm']

    print("Loading ONNX sessions...")
    sess_opts = rt.SessionOptions()
    sess_opts.inter_op_num_threads = 1
    sess_opts.intra_op_num_threads = 1
    sess_rf = rt.InferenceSession(rf_onnx_path, sess_opts, providers=['CPUExecutionProvider'])
    sess_xgb = rt.InferenceSession(xgb_onnx_path, sess_opts, providers=['CPUExecutionProvider'])
    sess_lgbm = rt.InferenceSession(lgbm_onnx_path, sess_opts, providers=['CPUExecutionProvider'])

    # =========================================================================
    # TEST 1: Manual Test Sample from 4_hybrid_model.py
    # =========================================================================
    print("\n" + "="*70)
    print("TEST 1: MANUAL TEST SAMPLE VALIDATION")
    print("="*70)
    manual_input = np.array([[-0.25, 0.94, 0, 0.45, 0.06, -1.0, 0.18, -0.11, -3.2]], dtype=np.float32)

    # Python Predictions
    py_rf_prob = rf_py.predict_proba(manual_input)[0][1]
    py_xgb_prob = xgb_py.predict_proba(manual_input)[0][1]
    py_lgbm_prob = lgbm_py.predict_proba(manual_input)[0][1]
    py_ensemble_prob = hybrid.predict_proba(manual_input)[0][1]
    py_ensemble_pred = int(py_ensemble_prob >= 0.55)

    # ONNX Predictions
    onnx_res = run_onnx_inference(sess_rf, sess_xgb, sess_lgbm, manual_input)
    onnx_rf_prob = onnx_res['rf_prob'][0]
    onnx_xgb_prob = onnx_res['xgb_prob'][0]
    onnx_lgbm_prob = onnx_res['lgbm_prob'][0]
    onnx_ensemble_prob = onnx_res['ensemble_prob'][0]
    onnx_ensemble_pred = onnx_res['ensemble_pred'][0]

    print(f"{'Component':<20} | {'Python':<12} | {'ONNX':<12} | {'Difference':<12}")
    print("-" * 65)
    print(f"{'Random Forest Prob':<20} | {py_rf_prob:<12.6f} | {onnx_rf_prob:<12.6f} | {abs(py_rf_prob - onnx_rf_prob):<12.2e}")
    print(f"{'XGBoost Prob':<20} | {py_xgb_prob:<12.6f} | {onnx_xgb_prob:<12.6f} | {abs(py_xgb_prob - onnx_xgb_prob):<12.2e}")
    print(f"{'LightGBM Prob':<20} | {py_lgbm_prob:<12.6f} | {onnx_lgbm_prob:<12.6f} | {abs(py_lgbm_prob - onnx_lgbm_prob):<12.2e}")
    print(f"{'Weighted Ensemble':<20} | {py_ensemble_prob:<12.6f} | {onnx_ensemble_prob:<12.6f} | {abs(py_ensemble_prob - onnx_ensemble_prob):<12.2e}")
    print(f"{'Class Prediction':<20} | {py_ensemble_pred:<12} | {onnx_ensemble_pred:<12} | {'MATCH' if py_ensemble_pred == onnx_ensemble_pred else 'MISMATCH'}")

    # =========================================================================
    # TEST 2: Validation on Test Dataset (1,364 samples)
    # =========================================================================
    print("\n" + "="*70)
    print("TEST 2: FULL DATASET VALIDATION (20% Stratified Test Split)")
    print("="*70)

    data_path = os.path.join(project_root, "data.csv", "data.csv")
    if not os.path.exists(data_path):
        print(f"Dataset not found at {data_path}, skipping dataset test.")
        return

    data = pd.read_csv(data_path)
    data.columns = data.columns.str.strip()

    selected_features = [
        'ROA(C) before interest and depreciation before interest',
        'Debt ratio %',
        'Net Income Flag',
        'Current Ratio',
        'Operating Gross Margin',
        'Interest Coverage Ratio (Interest expense to EBIT)',
        'Equity to Liability',
        'Net Income to Total Assets',
        'Cash Flow Per Share'
    ]

    _, df_test = train_test_split(data, test_size=0.2, random_state=42, stratify=data['Bankrupt?'])
    X_test = df_test[selected_features].values
    y_test = df_test['Bankrupt?'].values

    print(f"Loaded {len(X_test)} test samples from data.csv.")

    # Run Python Ensemble
    py_all_probs = hybrid.predict_proba(X_test)[:, 1]
    py_all_preds = (py_all_probs >= 0.55).astype(int)

    # Run ONNX Ensemble
    onnx_all_res = run_onnx_inference(sess_rf, sess_xgb, sess_lgbm, X_test)
    onnx_all_probs = onnx_all_res['ensemble_prob']
    onnx_all_preds = onnx_all_res['ensemble_pred']

    prob_diff = np.abs(py_all_probs - onnx_all_probs)
    max_diff = np.max(prob_diff)
    mean_diff = np.mean(prob_diff)
    matching_preds = np.sum(py_all_preds == onnx_all_preds)
    total_samples = len(y_test)
    agreement_rate = (matching_preds / total_samples) * 100.0

    print(f"Total Test Samples:            {total_samples}")
    print(f"Exact Prediction Matches:      {matching_preds} / {total_samples} ({agreement_rate:.2f}%)")
    print(f"Mean Absolute Error (Prob):    {mean_diff:.2e}")
    print(f"Max Absolute Error (Prob):     {max_diff:.2e}")

    # Inspect the first 10 samples
    print("\nFirst 10 test samples comparison:")
    print(f"{'Sample':<6} | {'Py Prob':<10} | {'ONNX Prob':<10} | {'Diff':<10} | {'Py Pred':<8} | {'ONNX Pred':<9} | {'True Label':<10}")
    print("-" * 75)
    for i in range(10):
        print(f"{i:<6} | {py_all_probs[i]:<10.4f} | {onnx_all_probs[i]:<10.4f} | {prob_diff[i]:<10.2e} | {py_all_preds[i]:<8} | {onnx_all_preds[i]:<9} | {y_test[i]:<10}")

    print("\n" + "="*70)
    if agreement_rate == 100.0 and max_diff < 1e-4:
        print(">>> VERIFICATION RESULT: PERFECT MATCH (100% IDENTICAL BEHAVIOR) <<<")
    else:
        print(">>> VERIFICATION RESULT: CHECK DISCREPANCIES <<<")
    print("="*70)


if __name__ == "__main__":
    validate()

