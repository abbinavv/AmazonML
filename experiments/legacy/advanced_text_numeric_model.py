"""
Advanced Text+Numeric Regression for Product Price Prediction
Goal: Drive SMAPE as low as possible (<44% target) using a blended model.
"""

import re
import math
import numpy as np
import pandas as pd
from typing import List, Dict, Any

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge, SGDRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.decomposition import TruncatedSVD

RANDOM_STATE = 42


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    denom[denom == 0] = 1.0
    return float(np.mean(np.abs(y_true - y_pred) / denom) * 100.0)


class FeatureExtractor:
    def __init__(self, token_price_map: Dict[str, float] | None = None):
        self._keyword_groups: Dict[str, List[str]] = {
            'qual_budget': ['budget', 'affordable', 'economy', 'basic', 'value'],
            'qual_premium': ['premium', 'luxury', 'deluxe', 'professional', 'authentic', 'genuine'],
            'cat_automotive': ['car', 'auto', 'vehicle', 'tire', 'engine', 'oil', 'brake'],
            'cat_jewelry': ['jewelry', 'ring', 'necklace', 'watch', 'gold', 'silver', 'diamond'],
            'cat_health': ['health', 'vitamin', 'supplement', 'organic'],
            'cat_electronics': ['electronic', 'device', 'smart', 'digital', 'tech'],
            'cat_tools': ['tool', 'drill', 'hammer', 'equipment', 'hardware'],
            'cat_home': ['home', 'kitchen', 'furniture', 'house'],
            'cat_food': ['food', 'snack', 'gourmet', 'cooking'],
            'brand_indicators': ['brand', 'official', 'licensed', 'certified']
        }
        self._token_price_map = token_price_map or {}

    def _extract_quantity(self, text_lower: str) -> float:
        patterns = [
            r'pack of (\d+)', r'(\d+)\s*pack', r'(\d+)\s*count', r'set of (\d+)',
            r'value:\s*(\d+\.?\d*)', r'(\d+)\s*piece', r'(\d+)\s*items?'
        ]
        for pattern in patterns:
            m = re.search(pattern, text_lower)
            if m:
                try:
                    return float(np.clip(float(m.group(1)), 1, 50))
                except Exception:
                    continue
        return 1.0

    def _extract_numbers(self, text: str) -> List[float]:
        nums = re.findall(r'\d+\.?\d*', text or '')
        out = []
        for n in nums:
            try:
                v = float(n)
                if 0.1 <= v <= 50000:
                    out.append(v)
            except Exception:
                pass
        return out

    def _extract_units(self, text_lower: str) -> Dict[str, float]:
        # common size/weight/volume units
        feats: Dict[str, float] = {}
        # weight
        m = re.search(r'(\d+\.?\d*)\s*(oz|ounce|ounces)', text_lower)
        feats['oz'] = float(m.group(1)) if m else 0.0
        m = re.search(r'(\d+\.?\d*)\s*(lb|pound|pounds)', text_lower)
        feats['lb'] = float(m.group(1)) if m else 0.0
        m = re.search(r'(\d+\.?\d*)\s*(g|gram|grams)', text_lower)
        feats['g'] = float(m.group(1)) if m else 0.0
        m = re.search(r'(\d+\.?\d*)\s*(kg|kilogram|kilograms)', text_lower)
        feats['kg'] = float(m.group(1)) if m else 0.0
        # volume
        m = re.search(r'(\d+\.?\d*)\s*(ml|milliliter|milliliters)', text_lower)
        feats['ml'] = float(m.group(1)) if m else 0.0
        m = re.search(r'(\d+\.?\d*)\s*(l|liter|liters)', text_lower)
        feats['l'] = float(m.group(1)) if m else 0.0
        # length
        m = re.search(r'(\d+\.?\d*)\s*(in|inch|inches|\")', text_lower)
        feats['inch'] = float(m.group(1)) if m else 0.0
        m = re.search(r'(\d+\.?\d*)\s*(ft|feet|foot)', text_lower)
        feats['ft'] = float(m.group(1)) if m else 0.0
        # count terms
        m = re.search(r'(\d+)\s*(pairs|pair)', text_lower)
        feats['pairs'] = float(m.group(1)) if m else 0.0
        m = re.search(r'(\d+)\s*(dozen|dz)', text_lower)
        feats['dozen'] = float(m.group(1)) if m else 0.0
        return feats

    def _count_keywords(self, text_lower: str, words: List[str]) -> int:
        return sum(1 for w in words if w in text_lower)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        rows: List[Dict[str, Any]] = []
        for _, row in df.iterrows():
            text = str(row.get('catalog_content', '') or '')
            tl = text.lower()
            features: Dict[str, Any] = {}

            qty = self._extract_quantity(tl)
            features['quantity'] = qty
            features['pack_size_log'] = float(np.log1p(qty))
            features['is_bulk'] = 1 if qty > 1 else 0

            features['text_length'] = len(text)
            wc = len(text.split())
            features['word_count'] = wc
            features['char_per_word'] = (features['text_length'] / max(wc, 1)) if wc else 0.0
            features['sentence_count'] = len(re.split(r'[.!?]+', text))

            nums = self._extract_numbers(text)
            features['number_count'] = len(nums)
            features['max_number'] = max(nums) if nums else 0.0
            features['avg_number'] = float(np.mean(nums)) if nums else 0.0

            for key, words in self._keyword_groups.items():
                features[key] = self._count_keywords(tl, words)

            features['category_total'] = sum(features[k] for k in self._keyword_groups.keys() if k.startswith('cat_'))

            # unit/size features
            unit_feats = self._extract_units(tl)
            features.update({f'u_{k}': v for k, v in unit_feats.items()})

            # Token-level prior price statistics
            tokens = re.findall(r"[a-zA-Z0-9$%\.\-]+", tl)
            priors = [self._token_price_map[t] for t in tokens if t in self._token_price_map]
            features['token_prior_mean'] = float(np.mean(priors)) if priors else 0.0
            features['token_prior_max'] = float(np.max(priors)) if priors else 0.0
            features['token_prior_count'] = len(priors)

            rows.append(features)
        out_df = pd.DataFrame(rows).fillna(0)
        return out_df


def train_and_validate(train_csv: str, sample_size: int = 30000, val_size: int = 6000):
    print(f"Loading {sample_size + val_size} rows from {train_csv}...")
    df = pd.read_csv(train_csv, nrows=sample_size + val_size)

    # Train/val split preserving order to simulate time-based split
    train_df = df.iloc[:sample_size].copy()
    val_df = df.iloc[sample_size:sample_size+val_size].copy()

    print("Extracting numeric features...")
    # Build token-level price priors from training set
    token_price_map: Dict[str, float] = {}
    for _, r in train_df.iterrows():
        text = str(r.get('catalog_content', '') or '').lower()
        price = float(r['price'])
        for tok in set(re.findall(r"[a-zA-Z0-9$%\.\-]+", text)):
            if 0.1 <= price <= 1000:
                token_price_map.setdefault(tok, []).append(price)
    token_price_map = {k: float(np.median(v)) for k, v in token_price_map.items() if len(v) >= 3}

    fx = FeatureExtractor(token_price_map=token_price_map)
    X_num_train = fx.transform(train_df)
    X_num_val = fx.transform(val_df)

    # Targets (log-transform for stability)
    y_train = np.log1p(train_df['price'].values)
    y_val_true = val_df['price'].values
    # Price clipping bounds from training distribution
    q_low, q_high = np.percentile(train_df['price'].values, [1, 99])

    print("Vectorizing text with TF-IDF...")
    vectorizer_word = TfidfVectorizer(
        analyzer='word',
        lowercase=True,
        ngram_range=(1, 2),
        min_df=3,
        max_df=0.85,
        max_features=250000,
        strip_accents='unicode'
    )
    vectorizer_char = TfidfVectorizer(
        analyzer='char',
        ngram_range=(3, 5),
        min_df=3,
        max_df=0.9,
        max_features=150000
    )
    X_txt_train_word = vectorizer_word.fit_transform(train_df['catalog_content'].fillna('').astype(str))
    X_txt_val_word = vectorizer_word.transform(val_df['catalog_content'].fillna('').astype(str))
    X_txt_train_char = vectorizer_char.fit_transform(train_df['catalog_content'].fillna('').astype(str))
    X_txt_val_char = vectorizer_char.transform(val_df['catalog_content'].fillna('').astype(str))

    # Concatenate sparse matrices
    from scipy.sparse import hstack
    X_txt_train = hstack([X_txt_train_word, X_txt_train_char])
    X_txt_val = hstack([X_txt_val_word, X_txt_val_char])

    print("Training text Ridge model...")
    text_model = Ridge(alpha=2.0, random_state=RANDOM_STATE)
    text_model.fit(X_txt_train, y_train)

    robust_text_model = SGDRegressor(loss='huber', alpha=1e-4, max_iter=2000, random_state=RANDOM_STATE)
    robust_text_model.fit(X_txt_train, y_train)

    print("Training numeric HistGBR model...")
    num_model = HistGradientBoostingRegressor(
        loss='absolute_error',
        max_depth=6,
        max_iter=400,
        learning_rate=0.06,
        l2_regularization=0.05,
        random_state=RANDOM_STATE
    )
    num_model.fit(X_num_train, y_train)

    # Dimensionality reduction on text for meta features
    print("Fitting SVD on text features...")
    svd = TruncatedSVD(n_components=256, random_state=RANDOM_STATE)
    X_txt_train_svd = svd.fit_transform(X_txt_train)
    X_txt_val_svd = svd.transform(X_txt_val)

    # Meta model combining numeric + svd text features
    print("Training meta HistGBR model on dense features...")
    meta_X_train = np.hstack([X_num_train.values, X_txt_train_svd])
    meta_X_val = np.hstack([X_num_val.values, X_txt_val_svd])
    meta_model = HistGradientBoostingRegressor(
        loss='absolute_error',
        max_depth=6,
        max_iter=500,
        learning_rate=0.05,
        l2_regularization=0.05,
        random_state=RANDOM_STATE
    )
    meta_model.fit(meta_X_train, y_train)

    print("Validating and optimizing blend...")
    pred_log_text = text_model.predict(X_txt_val)
    pred_log_text_robust = robust_text_model.predict(X_txt_val)
    pred_log_num = num_model.predict(X_num_val)
    pred_log_meta = meta_model.predict(meta_X_val)

    # Grid search blend weight among four models (text ridge, text robust, numeric, meta)
    best_weights = (0.25, 0.25, 0.25, 0.25)
    best_smape = float('inf')
    best_pred = None
    ws = np.linspace(0.0, 1.0, 9)
    for w1 in ws:
        for w2 in ws:
            for w3 in ws:
                w4 = 1.0 - w1 - w2 - w3
                if w4 < 0:
                    continue
                pred_log_blend = (
                    w1 * pred_log_text +
                    w2 * pred_log_text_robust +
                    w3 * pred_log_num +
                    w4 * pred_log_meta
                )
                pred_price_try = np.expm1(pred_log_blend)
                pred_price_try = np.clip(pred_price_try, q_low * 0.9, q_high * 1.05)
                s_try = smape(y_val_true, pred_price_try)
                if s_try < best_smape:
                    best_smape = s_try
                    best_weights = (w1, w2, w3, w4)
                    best_pred = pred_price_try

    s = best_smape
    pred_price = best_pred
    corr = np.corrcoef(y_val_true, pred_price)[0, 1]

    # Optional isotonic calibration on validation (mapping raw->calibrated)
    try:
        from sklearn.isotonic import IsotonicRegression
        iso = IsotonicRegression(out_of_bounds='clip')
        iso.fit(pred_price, y_val_true)
    except Exception:
        iso = None

    print("=== Validation (optimized blend) ===")
    print(f"Best weights -> text_ridge: {best_weights[0]:.3f}, text_robust: {best_weights[1]:.3f}, numeric: {best_weights[2]:.3f}, meta: {best_weights[3]:.3f}")
    print(f"SMAPE: {s:.2f}%")
    print(f"Correlation: {corr:.4f}")
    print(f"Pred mean: ${pred_price.mean():.2f}")
    print(f"Actual mean: ${y_val_true.mean():.2f}")

    artifacts = {
        'fx': fx,
        'vectorizer_word': vectorizer_word,
        'vectorizer_char': vectorizer_char,
        'svd': svd,
        'text_model': text_model,
        'robust_text_model': robust_text_model,
        'num_model': num_model,
        'meta_model': meta_model,
        'weights': tuple(float(x) for x in best_weights),
        'clip_low': float(q_low),
        'clip_high': float(q_high),
        'iso': iso
    }
    return s, artifacts


def predict_test(test_csv: str, artifacts: dict, out_path: str):
    print("Predicting test set...")
    test_df = pd.read_csv(test_csv)

    fx = artifacts['fx']
    vectorizer_word = artifacts['vectorizer_word']
    vectorizer_char = artifacts['vectorizer_char']
    svd = artifacts['svd']
    text_model = artifacts['text_model']
    robust_text_model = artifacts['robust_text_model']
    num_model = artifacts['num_model']
    meta_model = artifacts['meta_model']

    X_num_test = fx.transform(test_df)
    from scipy.sparse import hstack
    X_txt_test_word = vectorizer_word.transform(test_df['catalog_content'].fillna('').astype(str))
    X_txt_test_char = vectorizer_char.transform(test_df['catalog_content'].fillna('').astype(str))
    X_txt_test = hstack([X_txt_test_word, X_txt_test_char])

    pred_log_text = text_model.predict(X_txt_test)
    pred_log_text_robust = robust_text_model.predict(X_txt_test)
    pred_log_num = num_model.predict(X_num_test)
    X_txt_test_svd = svd.transform(X_txt_test)
    meta_X_test = np.hstack([X_num_test.values, X_txt_test_svd])
    pred_log_meta = meta_model.predict(meta_X_test)

    w1, w2, w3, w4 = artifacts['weights']
    pred_log_blend = w1 * pred_log_text + w2 * pred_log_text_robust + w3 * pred_log_num + w4 * pred_log_meta
    pred_price = np.expm1(pred_log_blend)
    clip_low = artifacts.get('clip_low', 0.5)
    clip_high = artifacts.get('clip_high', 400.0)
    pred_price = np.clip(pred_price, clip_low, clip_high)

    iso = artifacts.get('iso')
    if iso is not None:
        try:
            pred_price = iso.predict(pred_price)
            pred_price = np.clip(pred_price, clip_low, clip_high)
        except Exception:
            pass

    out = pd.DataFrame({
        'sample_id': test_df['sample_id'],
        'price': np.round(pred_price, 2)
    })
    out.to_csv(out_path, index=False)
    print(f"Saved predictions to {out_path}")


if __name__ == "__main__":
    smape_val, art = train_and_validate('dataset/train.csv', sample_size=30000, val_size=6000)
    if smape_val < 60.0:
        predict_test('dataset/test.csv', art, 'dataset/test_out_ml_advanced.csv')
    print(f"Final SMAPE (validation): {smape_val:.2f}%")
