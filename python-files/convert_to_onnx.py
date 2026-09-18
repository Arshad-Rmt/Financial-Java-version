"""
convert_to_onnx.py

Extracts individual trained estimators (Random Forest, XGBoost, LightGBM) from model.pkl
and converts each model to standard ONNX format for high-performance Java Spring Boot execution.
Preserves exact 9-feature input ordering and standardizes outputs to 2D probability tensors.
"""

import os
import sys
import copy
import json
import joblib
import numpy as np

import onnx
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType as SklFloatTensorType
import onnxmltools
from onnxmltools.convert.common.data_types import FloatTensorType as OnnxMLFloatTensorType


def convert_models():
    # Resolve directory paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    model_path = os.path.join(current_dir, "model.pkl")
    
    if not os.path.exists(model_path):
        # Fallback if run from root
        model_path = os.path.join(project_root, "model.pkl")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Cannot find model.pkl at {model_path}")

    output_dir = os.path.join(project_root, "onnx-models")
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading original model from: {model_path}")
    hybrid_model = joblib.load(model_path)

    # Verify model structure
    if not hasattr(hybrid_model, "named_estimators_"):
        raise ValueError("Loaded object does not have named_estimators_. Expected VotingClassifier.")

    estimators = hybrid_model.named_estimators_
    print(f"Extracted estimators: {list(estimators.keys())}")
    
    rf = estimators['rf']
    xgb = estimators['xgb']
    lgbm = estimators['lgbm']

    feature_names = [
        "ROA(C) before interest and depreciation before interest",
        "Debt ratio %",
        "Net Income Flag",
        "Current Ratio",
        "Operating Gross Margin",
        "Interest Coverage Ratio (Interest expense to EBIT)",
        "Equity to Liability",
        "Net Income to Total Assets",
        "Cash Flow Per Share"
    ]
    num_features = len(feature_names)

    target_opset = 15

    # -------------------------------------------------------------
    # 1. Convert Random Forest
    # -------------------------------------------------------------
    print("\n[1/3] Converting Random Forest to ONNX...")
    rf_initial_type = [('float_input', SklFloatTensorType([None, num_features]))]
    rf_onnx = convert_sklearn(
        rf,
        initial_types=rf_initial_type,
        target_opset=target_opset,
        options={'zipmap': False}  # Output 2D probability tensor [None, 2]
    )
    rf_path = os.path.join(output_dir, "random_forest.onnx")
    with open(rf_path, "wb") as f:
        f.write(rf_onnx.SerializeToString())
    print(f"  [OK] Saved Random Forest ONNX: {rf_path} ({os.path.getsize(rf_path)} bytes)")

    # -------------------------------------------------------------
    # 2. Convert XGBoost
    # -------------------------------------------------------------
    print("\n[2/3] Converting XGBoost to ONNX...")
    # XGBoost booster stores string feature names from training dataframe ('Debt ratio %').
    # onnxmltools parser requires feature names to be indexed ('f0'..'f8') or None.
    # We copy the booster and reset feature names during export to avoid touching original model.
    xgb_copy = copy.deepcopy(xgb)
    xgb_copy.get_booster().feature_names = None
    
    xgb_initial_type = [('float_input', OnnxMLFloatTensorType([None, num_features]))]
    xgb_onnx = onnxmltools.convert_xgboost(
        xgb_copy,
        initial_types=xgb_initial_type,
        target_opset=target_opset
    )
    xgb_path = os.path.join(output_dir, "xgboost.onnx")
    with open(xgb_path, "wb") as f:
        f.write(xgb_onnx.SerializeToString())
    print(f"  [OK] Saved XGBoost ONNX: {xgb_path} ({os.path.getsize(xgb_path)} bytes)")

    # -------------------------------------------------------------
    # 3. Convert LightGBM
    # -------------------------------------------------------------
    print("\n[3/3] Converting LightGBM to ONNX...")
    lgbm_initial_type = [('float_input', OnnxMLFloatTensorType([None, num_features]))]
    lgbm_onnx = onnxmltools.convert_lightgbm(
        lgbm,
        initial_types=lgbm_initial_type,
        target_opset=target_opset,
        zipmap=False  # Output 2D probability tensor [None, 2]
    )
    # Ensure dynamic batch dimension on label output
    if len(lgbm_onnx.graph.output) > 0 and len(lgbm_onnx.graph.output[0].type.tensor_type.shape.dim) > 0:
        lgbm_onnx.graph.output[0].type.tensor_type.shape.dim[0].ClearField('dim_value')
        lgbm_onnx.graph.output[0].type.tensor_type.shape.dim[0].dim_param = 'None'

    lgbm_path = os.path.join(output_dir, "lightgbm.onnx")
    with open(lgbm_path, "wb") as f:
        f.write(lgbm_onnx.SerializeToString())
    print(f"  [OK] Saved LightGBM ONNX: {lgbm_path} ({os.path.getsize(lgbm_path)} bytes)")

    # -------------------------------------------------------------
    # 4. Save metadata JSON for Java Spring Boot integration
    # -------------------------------------------------------------
    metadata = {
        "model_architecture": "Hybrid Weighted Soft Voting Ensemble",
        "features": [
            {"index": i, "name": name} for i, name in enumerate(feature_names)
        ],
        "input_tensor_name": "float_input",
        "input_shape": [1, num_features],
        "input_dtype": "float32",
        "output_label_name": "label",
        "output_probability_name": "probabilities",
        "output_probability_shape": [1, 2],
        "ensemble_weights": {
            "random_forest": 1,
            "xgboost": 2,
            "lightgbm": 2
        },
        "total_weight": 5,
        "classification_threshold": 0.55,
        "models": {
            "random_forest": "random_forest.onnx",
            "xgboost": "xgboost.onnx",
            "lightgbm": "lightgbm.onnx"
        }
    }
    meta_path = os.path.join(output_dir, "model_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"\n  [OK] Saved Model Metadata: {meta_path}")

    print("\n" + "="*60)
    print("ALL 3 MODELS SUCCESSFULLY CONVERTED TO ONNX!")
    print("="*60)


if __name__ == "__main__":
    convert_models()

