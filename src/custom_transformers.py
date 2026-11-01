
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.pipeline import Pipeline

# ==========================================================
# Pipeline: Custom Pipeline with Deduplication
# ==========================================================
class PreprocessingPipelineWithDedup(Pipeline):
    def fit(self, X, y=None, **fit_params):
        if y is not None:
            X_clean, y_clean = self._apply_deduplication(X, y)
            self.y_clean_ = y_clean
            return super().fit(X_clean, y_clean, **fit_params)
        return super().fit(X, y, **fit_params)

    def fit_transform(self, X, y=None, **fit_params):
        if y is not None:
            X_clean, y_clean = self._apply_deduplication(X, y)
            self.y_clean_ = y_clean
            X_transformed = super().fit_transform(X_clean, y_clean, **fit_params)
            return X_transformed
        return super().fit_transform(X, y, **fit_params)

    def transform(self, X, y=None):
        return super().transform(X)

    def _apply_deduplication(self, X, y):
        print(f"[Deduplication] X_train before: {X.shape[0]:,} rows")

        train_combined = X.copy()
        train_combined['__target__'] = y.values
        train_deduped = train_combined.drop_duplicates(subset=X.columns.tolist(), keep='first')

        X_clean = train_deduped.drop(columns=['__target__'])
        y_clean = train_deduped['__target__']

        removed = len(X) - len(X_clean)
        print(f"[Deduplication] X_train after:  {X_clean.shape[0]:,} rows")
        print(f"[Deduplication] Removed: {removed} duplicate rows")
        return X_clean, y_clean


# ========================================================== 
# Transformer 1: Datetime Feature Extractor 
# ==========================================================
class DatetimeFeatureExtractor(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X).copy()
        f2_dt = pd.to_datetime(X['f2'], errors='coerce', format='mixed')
        f3_dt = pd.to_datetime(X['f3'], errors='coerce', format='mixed')

        X['duration_minutes'] = (f3_dt - f2_dt).dt.total_seconds() / 60
        X['start_hour']       = f2_dt.dt.hour.astype('float64')
        X['start_month']      = f2_dt.dt.month.astype('float64')
        X['start_dayofweek']  = f2_dt.dt.dayofweek.astype('float64')
        #return X
        return X.drop(columns=['f2', 'f3', 'f13'], errors='ignore')

    def get_feature_names_out(self, input_features=None):
        if input_features is not None:
            return np.array(input_features)
        return None

# ========================================================== 
# Transformer 2: Bool To Int 
# ==========================================================
class BoolToInt(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy() if not isinstance(X, pd.DataFrame) else X.copy()
        for col in X.select_dtypes(include='bool').columns:
            X[col] = X[col].astype(int)
        return X

    def get_feature_names_out(self, input_features=None):
        if input_features is not None:
            return np.array(input_features)
        return None

# ========================================================== 
# Transformer 3: Target Encoder
# ==========================================================
class TargetEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, cols, smoothing=10):
        self.cols = cols
        self.smoothing = smoothing

    def fit(self, X, y):
        X = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X).copy()
        self.global_mean_ = float(np.mean(y))
        self.encoding_maps_ = {}

        for col in self.cols:
            if col not in X.columns:
                continue
            tmp = pd.DataFrame({'cat': X[col].astype(str).values, 'target': np.array(y)})
            stats = tmp.groupby('cat')['target'].agg(['mean', 'count'])
            smoother = stats['count'] / (stats['count'] + self.smoothing)
            smoothed = smoother * stats['mean'] + (1 - smoother) * self.global_mean_
            self.encoding_maps_[col] = smoothed.to_dict()
        return self

    def transform(self, X):
        X = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X).copy()
        for col in self.cols:
            if col not in X.columns:
                continue
            X[col] = X[col].astype(str).map(self.encoding_maps_[col]).fillna(self.global_mean_)
        return X

    def get_feature_names_out(self, input_features=None):
        return np.array(input_features) if input_features is not None else None

# ========================================================== 
# Transformer 4: Day / Night Binary Encoder 
# ==========================================================
class DayNightBinaryEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, cols):
        self.cols = cols

    def fit(self, X, y=None):
        X = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X).copy()
        self.fill_values_ = {}
        for col in self.cols:
            if col in X.columns:
                self.fill_values_[col] = X[col].mode()[0]
        return self

    def transform(self, X):
        X = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X).copy()
        for col in self.cols:
            if col not in X.columns:
                continue
            X[col] = X[col].fillna(self.fill_values_[col])
            X[col] = (X[col].str.strip().str.lower() == 'day').astype(int)
        return X

    def get_feature_names_out(self, input_features=None):
        return np.array(input_features) if input_features is not None else None

# ========================================================== 
# Transformer 5: IQR Capper 
# ==========================================================
class IQRCapper(BaseEstimator, TransformerMixin):
    def __init__(self, factor=1.5):
        self.factor = factor  # ضریب IQR (استاندارد = 1.5)

    def fit(self, X, y=None):
        X_arr = np.array(X, dtype=float)
        self.n_features_ = X_arr.shape[1]
        self.bounds_ = {}
        for i in range(self.n_features_):
            col_data = X_arr[:, i]
            col_data = col_data[~np.isnan(col_data)]  # NaN را نادیده بگیر

            if len(col_data) == 0:
                self.bounds_[i] = {
                    'lower': -np.inf,
                    'upper': np.inf,
                    'q1': np.nan,
                    'q3': np.nan,
                    'iqr': np.nan
                }
                continue

            q1 = np.percentile(col_data, 25)
            q3 = np.percentile(col_data, 75)
            iqr = q3 - q1
            if iqr == 0:
                lower = np.percentile(col_data, 1)
                upper = np.percentile(col_data, 99)

                if lower == upper:
                    self.bounds_[i] = None
                    continue

                self.bounds_[i] = {
                    'lower': lower,
                    'upper': upper,
                    'q1': q1, 'q3': q3, 'iqr': iqr
                }
                #print(f"Feature {i}: IQR=0")
            else:
                self.bounds_[i] = {
                    'lower': q1 - self.factor * iqr,
                    'upper': q3 + self.factor * iqr,
                    'q1': q1, 'q3': q3, 'iqr': iqr
                }
        return self

    def transform(self, X):
        X_arr = np.array(X, dtype=float).copy()
        for i in range(self.n_features_):
            if self.bounds_[i] is None:
                continue
            lower = self.bounds_[i]['lower']
            upper = self.bounds_[i]['upper']
            X_arr[:, i] = np.clip(X_arr[:, i], lower, upper)
        return X_arr

    def get_feature_names_out(self, input_features=None):
        if input_features is not None:
            return np.array(input_features)
        return np.array([f'x{i}' for i in range(self.n_features_)])

# ==========================================================
# Transformer 6: Text Keyword Extractor (Replaces Target Encoding for f8, f9)
# ==========================================================
class TextKeywordExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, text_cols):
        self.text_cols = text_cols

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X).copy()

        for col in self.text_cols:
            if col not in X.columns:
                continue

            text_data = X[col].astype(str).fillna("missing")

            if col == 'f8':
                X['f8_is_highway'] = text_data.str.contains(r'\b(?:I-|US-|[A-Z]{2}-\d+)\b', case=False, na=False).astype(int)

                X['f8_is_local_road'] = text_data.str.contains(r'\b(?:St|Ave|Rd|Blvd|Dr|Ln|Way)\b', case=False, na=False).astype(int)

                X['f8_is_severe_crash'] = text_data.str.contains(r'\b(?:Crash|Accident|Overturned|Blocked|Stall)\b', case=False, na=False).astype(int)

                X['f8_is_slow_traffic'] = text_data.str.contains(r'\b(?:Slow traffic|Congestion)\b', case=False, na=False).astype(int)

            elif col == 'f9':
                X['f9_is_highway_route'] = text_data.str.contains(r'\b(?:I-|US-|[A-Z]{2}-\d+|Fwy|Hwy|Expy)\b', case=False, na=False).astype(int)

            X = X.drop(columns=[col], errors='ignore')

        return X

    def get_feature_names_out(self, input_features=None):
        return np.array([
            'f8_is_highway', 
            'f8_is_local_road', 
            'f8_is_severe_crash', 
            'f8_is_slow_traffic', 
            'f9_is_highway_route'
        ])
