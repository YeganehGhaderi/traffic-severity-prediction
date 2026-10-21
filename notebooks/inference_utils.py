
import json
import joblib
import pandas as pd
from pathlib import Path

from custom_transformers import (
    PreprocessingPipelineWithDedup,
    DatetimeFeatureExtractor,
    BoolToInt,
    TargetEncoder,
    DayNightBinaryEncoder,
    IQRCapper,
    TextKeywordExtractor
)

def preprocess_and_select_25(raw_df, project_dir=None, y=None):
    """
    اگر y داده شود: روی حالت Train اجرا می‌شود (شامل Deduplication و بازگرداندن y_clean).
    اگر y داده نشود: روی حالت Val/Test اجرا می‌شود (فقط Transform بدون حذف سطر).
    """
    if project_dir is None:
        project_dir = Path(__file__).resolve().parents[1]
    else:
        project_dir = Path(project_dir)

    models_dir = project_dir / "models"

    pipeline_path = models_dir / "main_preprocess_pipeline.pkl"
    selected_features_path = models_dir / "selected_25_features.json"

    preprocess_pipeline = joblib.load(pipeline_path)

    X_raw = raw_df.drop(columns=["y"], errors="ignore")

    if y is not None:
        X_processed = preprocess_pipeline.fit_transform(X_raw, y)

        y_clean = preprocess_pipeline.y_clean_.reset_index(drop=True)
    else:
        X_processed = preprocess_pipeline.transform(X_raw)
        y_clean = None

    feature_names = preprocess_pipeline.named_steps["col_transformer"].get_feature_names_out()
    X_processed_df = pd.DataFrame(X_processed, columns=feature_names)

    with open(selected_features_path, "r", encoding="utf-8") as f:
        selected_features = json.load(f)

    X_selected = X_processed_df[selected_features].copy()

    if y_clean is not None:
        return X_selected, y_clean

    return X_selected
