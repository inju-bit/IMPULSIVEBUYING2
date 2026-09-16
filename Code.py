# ============================================================
# IMPULSIVE BUYING BEHAVIOUR ANALYSIS
# Improved Streamlit ML App - no PLS-SEM
# ============================================================

import warnings
from functools import partial

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import chi2_contingency

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
    StackingClassifier,
    VotingClassifier,
)
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import (
    RepeatedStratifiedKFold,
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.pipeline import Pipeline as SklearnPipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.svm import SVC

from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import Pipeline

# Optional libraries ---------------------------------------------------------
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except Exception:
    XGBOOST_AVAILABLE = False

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except Exception:
    LIGHTGBM_AVAILABLE = False

try:
    from catboost import CatBoostClassifier
    CATBOOST_AVAILABLE = True
except Exception:
    CATBOOST_AVAILABLE = False

try:
    import optuna
    OPTUNA_AVAILABLE = True
    optuna.logging.set_verbosity(optuna.logging.WARNING)
except Exception:
    OPTUNA_AVAILABLE = False

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
TARGET_COLUMNS = ["IB1", "IB2", "IB3"]

CONSTRUCT_BLOCKS = {
    "PE": ["PE1", "PE2", "PE3", "PE4"],
    "SE": ["SE1", "SE2", "SE3"],
    "TP": ["TP1", "TP2", "TP3"],
    "HB": ["HB1", "HB2", "HB3"],
    "UB": ["UB1", "UB2", "UB3", "UB4", "UB5"],
}

# Streamlit page -------------------------------------------------------------
st.set_page_config(
    page_title="Impulsive Buying ML Analysis",
    page_icon="🛒",
    layout="wide",
)

st.markdown(
    """
    <style>
        .main-title {
            font-size: 2.5rem;
            font-weight: 700;
            color: #2C3E50;
            text-align: center;
            margin-bottom: 0.25rem;
        }
        .subtitle {
            font-size: 1.05rem;
            color: #7F8C8D;
            text-align: center;
            margin-bottom: 1.5rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    '<p class="main-title">🛒 Impulsive Buying Behaviour Analysis</p>',
    unsafe_allow_html=True,
)
st.markdown(
    '<p class="subtitle">Improved Machine Learning Analysis | MRes Artificial Intelligence</p>',
    unsafe_allow_html=True,
)

# Session state --------------------------------------------------------------
for key, default in {
    "df": None,
    "ml_results": None,
    "trained_models": {},
    "target_name": None,
    "target_classes": None,
    "feature_notes": None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default


# Helpers -------------------------------------------------------------------
def cramers_v(x, y):
    data = pd.DataFrame({"x": x, "y": y}).dropna()
    if data.empty:
        return np.nan

    table = pd.crosstab(data["x"], data["y"])
    if table.empty:
        return np.nan

    chi2 = chi2_contingency(table, correction=False)[0]
    n = table.to_numpy().sum()
    if n <= 1:
        return 0.0

    phi2 = chi2 / n
    r, k = table.shape
    phi2corr = max(0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
    rcorr = r - ((r - 1) ** 2) / (n - 1)
    kcorr = k - ((k - 1) ** 2) / (n - 1)
    denom = min(kcorr - 1, rcorr - 1)
    return float(np.sqrt(phi2corr / denom)) if denom > 0 else 0.0


def make_one_hot_encoder():
    """Works with both newer and older scikit-learn releases."""
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def engineer_features(data, add_means=True, add_interactions=True):
    """Create construct means and selected theory-informed interaction features."""
    out = data.copy()
    created = []

    if add_means:
        for construct, items in CONSTRUCT_BLOCKS.items():
            present = [c for c in items if c in out.columns]
            if len(present) >= 2:
                name = f"{construct}_mean"
                out[name] = out[present].apply(pd.to_numeric, errors="coerce").mean(axis=1)
                created.append(name)

    if add_interactions:
        interaction_pairs = [
            ("HB_mean", "SE_mean", "HB_x_SE"),
            ("HB_mean", "TP_mean", "HB_x_TP"),
            ("PE_mean", "SE_mean", "PE_x_SE"),
            ("UB_mean", "HB_mean", "UB_x_HB"),
        ]
        for left, right, name in interaction_pairs:
            if left in out.columns and right in out.columns:
                out[name] = out[left] * out[right]
                created.append(name)

    return out, created


def create_preprocessor(X):
    numeric_features = X.select_dtypes(include=np.number).columns.tolist()
    categorical_features = X.select_dtypes(exclude=np.number).columns.tolist()

    numeric_pipeline = SklearnPipeline(
        steps=[("imputer", SimpleImputer(strategy="median"))]
    )

    categorical_pipeline = SklearnPipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", make_one_hot_encoder()),
        ]
    )

    transformers = []
    if numeric_features:
        transformers.append(("numeric", numeric_pipeline, numeric_features))
    if categorical_features:
        transformers.append(("categorical", categorical_pipeline, categorical_features))

    return ColumnTransformer(transformers=transformers, remainder="drop")


def mi_score(X, y):
    return mutual_info_classif(X, y, random_state=RANDOM_STATE)


def class_weight_value(balance_mode):
    return "balanced" if balance_mode == "Class Weight" else None


def create_model(model_name, params=None, balance_mode="None"):
    params = params or {}
    cw = class_weight_value(balance_mode)

    if model_name == "Random Forest":
        defaults = dict(
            n_estimators=250,
            max_depth=10,
            min_samples_split=2,
            min_samples_leaf=1,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            class_weight=cw,
        )
        defaults.update(params)
        return RandomForestClassifier(**defaults)

    if model_name == "Extra Trees":
        defaults = dict(
            n_estimators=250,
            max_depth=10,
            min_samples_split=2,
            min_samples_leaf=1,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            class_weight=cw,
        )
        defaults.update(params)
        return ExtraTreesClassifier(**defaults)

    if model_name == "Gradient Boosting":
        defaults = dict(
            n_estimators=150,
            learning_rate=0.05,
            max_depth=3,
            subsample=0.9,
            random_state=RANDOM_STATE,
        )
        defaults.update(params)
        return GradientBoostingClassifier(**defaults)

    if model_name == "Hist Gradient Boosting":
        defaults = dict(
            learning_rate=0.08,
            max_iter=180,
            max_leaf_nodes=31,
            random_state=RANDOM_STATE,
        )
        # class_weight exists in current sklearn; only pass it when requested.
        if cw is not None:
            defaults["class_weight"] = cw
        defaults.update(params)
        return HistGradientBoostingClassifier(**defaults)

    if model_name == "SVM (RBF)":
        defaults = dict(
            C=5.0,
            gamma="scale",
            kernel="rbf",
            probability=True,
            class_weight=cw,
            random_state=RANDOM_STATE,
        )
        defaults.update(params)
        return SVC(**defaults)

    if model_name == "Logistic Regression":
        defaults = dict(
            C=1.0,
            max_iter=3000,
            class_weight=cw,
            random_state=RANDOM_STATE,
        )
        defaults.update(params)
        return LogisticRegression(**defaults)

    if model_name == "XGBoost":
        if not XGBOOST_AVAILABLE:
            raise ImportError("XGBoost is not installed.")
        defaults = dict(
            n_estimators=220,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="mlogloss",
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbosity=0,
        )
        defaults.update(params)
        return xgb.XGBClassifier(**defaults)

    if model_name == "LightGBM":
        if not LIGHTGBM_AVAILABLE:
            raise ImportError("LightGBM is not installed.")
        defaults = dict(
            n_estimators=220,
            max_depth=8,
            learning_rate=0.05,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbosity=-1,
        )
        if cw is not None:
            defaults["class_weight"] = cw
        defaults.update(params)
        return lgb.LGBMClassifier(**defaults)

    if model_name == "CatBoost":
        if not CATBOOST_AVAILABLE:
            raise ImportError("CatBoost is not installed.")
        defaults = dict(
            iterations=220,
            depth=6,
            learning_rate=0.05,
            random_state=RANDOM_STATE,
            verbose=False,
            allow_writing_files=False,
        )
        if cw is not None:
            defaults["auto_class_weights"] = "Balanced"
        defaults.update(params)
        return CatBoostClassifier(**defaults)

    if model_name == "Voting Ensemble":
        estimators = [
            (
                "rf",
                RandomForestClassifier(
                    n_estimators=180,
                    max_depth=10,
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                    class_weight=cw,
                ),
            ),
            (
                "et",
                ExtraTreesClassifier(
                    n_estimators=180,
                    max_depth=10,
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                    class_weight=cw,
                ),
            ),
            (
                "gb",
                GradientBoostingClassifier(
                    n_estimators=120,
                    learning_rate=0.05,
                    max_depth=3,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
        return VotingClassifier(estimators=estimators, voting="soft", n_jobs=1)

    if model_name == "Stacking Ensemble":
        estimators = [
            (
                "rf",
                RandomForestClassifier(
                    n_estimators=150,
                    max_depth=10,
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                    class_weight=cw,
                ),
            ),
            (
                "et",
                ExtraTreesClassifier(
                    n_estimators=150,
                    max_depth=10,
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                    class_weight=cw,
                ),
            ),
            (
                "gb",
                GradientBoostingClassifier(
                    n_estimators=100,
                    learning_rate=0.05,
                    max_depth=3,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
        final_estimator = LogisticRegression(max_iter=2500, class_weight=cw)
        return StackingClassifier(
            estimators=estimators,
            final_estimator=final_estimator,
            cv=3,
            n_jobs=1,
            passthrough=False,
        )

    raise ValueError(f"Unknown model: {model_name}")


def get_optuna_params(trial, model_name):
    if model_name == "Random Forest":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 450),
            "max_depth": trial.suggest_int("max_depth", 3, 20),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 5),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
        }

    if model_name == "Extra Trees":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 450),
            "max_depth": trial.suggest_int("max_depth", 3, 20),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 5),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
        }

    if model_name == "Gradient Boosting":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 70, 350),
            "max_depth": trial.suggest_int("max_depth", 2, 6),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.20, log=True),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 5),
            "subsample": trial.suggest_float("subsample", 0.65, 1.0),
        }

    if model_name == "Hist Gradient Boosting":
        return {
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.20, log=True),
            "max_iter": trial.suggest_int("max_iter", 80, 300),
            "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 10, 50),
            "l2_regularization": trial.suggest_float("l2_regularization", 0.0, 5.0),
        }

    if model_name == "SVM (RBF)":
        return {
            "C": trial.suggest_float("C", 0.1, 50.0, log=True),
            "gamma": trial.suggest_float("gamma", 1e-4, 1.0, log=True),
        }

    if model_name == "Logistic Regression":
        return {
            "C": trial.suggest_float("C", 0.01, 30.0, log=True),
        }

    if model_name == "XGBoost":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 450),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.20, log=True),
            "subsample": trial.suggest_float("subsample", 0.65, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.65, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 8),
        }

    if model_name == "LightGBM":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 450),
            "max_depth": trial.suggest_int("max_depth", 3, 15),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.20, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 10, 70),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 40),
        }

    if model_name == "CatBoost":
        return {
            "iterations": trial.suggest_int("iterations", 100, 450),
            "depth": trial.suggest_int("depth", 4, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.20, log=True),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0),
        }

    return {}


def create_ml_pipeline(
    X,
    model,
    balance_mode="None",
    use_feature_selection=False,
    feature_k=10,
    scale_features=False,
):
    steps = [("preprocessing", create_preprocessor(X))]

    if use_feature_selection:
        k = min(int(feature_k), max(1, X.shape[1]))
        steps.append(("feature_selection", SelectKBest(score_func=mi_score, k=k)))

    if scale_features:
        steps.append(("scaler", StandardScaler()))

    if balance_mode == "Random Oversampling":
        steps.append(("oversampling", RandomOverSampler(random_state=RANDOM_STATE)))

    steps.append(("model", model))
    return Pipeline(steps=steps)


def model_needs_scaling(model_name):
    return model_name in {"SVM (RBF)", "Logistic Regression"}


def make_cv(cv_folds, repeated=False, repeats=2):
    if repeated:
        return RepeatedStratifiedKFold(
            n_splits=cv_folds,
            n_repeats=repeats,
            random_state=RANDOM_STATE,
        )
    return StratifiedKFold(
        n_splits=cv_folds,
        shuffle=True,
        random_state=RANDOM_STATE,
    )


def get_available_models():
    models = [
        "Random Forest",
        "Extra Trees",
        "Gradient Boosting",
        "Hist Gradient Boosting",
        "SVM (RBF)",
        "Logistic Regression",
    ]
    if XGBOOST_AVAILABLE:
        models.append("XGBoost")
    if LIGHTGBM_AVAILABLE:
        models.append("LightGBM")
    if CATBOOST_AVAILABLE:
        models.append("CatBoost")
    models += ["Voting Ensemble", "Stacking Ensemble"]
    return models


# Sidebar -------------------------------------------------------------------
with st.sidebar:
    st.title("🎯 Prediction")
    st.markdown("---")

    uploaded_file = st.file_uploader("📂 Upload CSV dataset", type=["csv"])

    if uploaded_file is not None:
        try:
            loaded_df = pd.read_csv(uploaded_file)
            st.session_state.df = loaded_df
            st.success(f"✅ Loaded {len(loaded_df):,} rows")
        except Exception as error:
            st.error(f"Could not read CSV: {error}")

    st.markdown("---")
    st.markdown("### Accuracy-improvement tools")
    st.caption(
        "Feature engineering, mutual-information feature selection, class balancing, "
        "SVM, tree boosting, voting/stacking ensembles, Optuna, and repeated CV."
    )

    st.markdown("---")
    st.warning(
        "Streamlit Cloud has limited CPU. Start with 1–3 models, 3-fold CV, "
        "and Optuna OFF. Run heavier experiments locally."
    )


# Main app ------------------------------------------------------------------
if st.session_state.df is not None:
    raw_df = st.session_state.df.copy()

    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        [
            "📊 Data Overview",
            "🔍 EDA",
            "🔗 Correlations",
            "🤖 ML Lab",
            "📈 Results",
        ]
    )

    # TAB 1 -----------------------------------------------------------------
    with tab1:
        st.header("📊 Data Overview")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Rows", f"{raw_df.shape[0]:,}")
        c2.metric("Columns", raw_df.shape[1])
        c3.metric("Missing", f"{raw_df.isna().sum().sum():,}")
        c4.metric("Duplicate rows", f"{raw_df.duplicated().sum():,}")

        st.subheader("Data Preview")
        st.dataframe(raw_df.head(10), use_container_width=True)

        st.subheader("Column Summary")
        info = pd.DataFrame(
            {
                "Column": raw_df.columns,
                "Type": raw_df.dtypes.astype(str).values,
                "Unique": raw_df.nunique(dropna=True).values,
                "Missing": raw_df.isna().sum().values,
                "Missing %": (raw_df.isna().mean() * 100).round(2).values,
            }
        )
        st.dataframe(info, use_container_width=True)

        available_targets = [c for c in TARGET_COLUMNS if c in raw_df.columns]
        if available_targets:
            st.subheader("Target Class Balance")
            for target in available_targets:
                counts = raw_df[target].value_counts(dropna=False).sort_index()
                pct = (counts / counts.sum() * 100).round(2)
                summary = pd.DataFrame({"Count": counts, "Percent": pct})
                st.markdown(f"**{target}**")
                st.dataframe(summary, use_container_width=True)

    # TAB 2 -----------------------------------------------------------------
    with tab2:
        st.header("🔍 Exploratory Data Analysis")

        available_targets = [c for c in TARGET_COLUMNS if c in raw_df.columns]
        if available_targets:
            st.subheader("Target Distributions")
            fig, axes = plt.subplots(1, len(available_targets), figsize=(5 * len(available_targets), 4))
            if len(available_targets) == 1:
                axes = [axes]
            for i, target in enumerate(available_targets):
                counts = raw_df[target].value_counts().sort_index()
                axes[i].bar(counts.index.astype(str), counts.values)
                axes[i].set_title(f"{target} Distribution")
                axes[i].set_xlabel("Response")
                axes[i].set_ylabel("Count")
                for j, value in enumerate(counts.values):
                    axes[i].text(j, value, str(value), ha="center", va="bottom")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

        st.subheader("Likert Data Quality Check")
        research_items = [
            c
            for items in CONSTRUCT_BLOCKS.values()
            for c in items
            if c in raw_df.columns
        ] + [c for c in TARGET_COLUMNS if c in raw_df.columns]

        if research_items:
            problems = []
            for col in research_items:
                numeric = pd.to_numeric(raw_df[col], errors="coerce")
                invalid = numeric.notna() & ~numeric.isin([1, 2, 3, 4, 5])
                non_numeric = raw_df[col].notna() & numeric.isna()
                if invalid.any() or non_numeric.any():
                    problems.append(
                        {
                            "Column": col,
                            "Outside 1–5": int(invalid.sum()),
                            "Non-numeric": int(non_numeric.sum()),
                        }
                    )
            if problems:
                st.warning("Some research-item values need checking.")
                st.dataframe(pd.DataFrame(problems), use_container_width=True)
            else:
                st.success("No obvious non-numeric or outside-1–5 values were found in the detected research items.")

        st.subheader("Feature Distributions")
        numeric_cols = raw_df.select_dtypes(include=np.number).columns.tolist()
        selected = st.multiselect("Choose numeric variables", numeric_cols, default=numeric_cols[:6])
        if selected:
            ncols = min(3, len(selected))
            nrows = (len(selected) + ncols - 1) // ncols
            fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
            axes = np.array(axes).reshape(-1)
            for i, col in enumerate(selected):
                axes[i].hist(raw_df[col].dropna(), bins=15, edgecolor="white")
                axes[i].set_title(col)
            for i in range(len(selected), len(axes)):
                axes[i].set_visible(False)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

    # TAB 3 -----------------------------------------------------------------
    with tab3:
        st.header("🔗 Cramér's V Correlation")
        cat_cols = [
            c
            for c in raw_df.columns
            if raw_df[c].dtype == "object" or raw_df[c].nunique(dropna=True) <= 10
        ]
        if len(cat_cols) > 1:
            matrix = pd.DataFrame(np.zeros((len(cat_cols), len(cat_cols))), index=cat_cols, columns=cat_cols)
            with st.spinner("Calculating Cramér's V matrix..."):
                for i in range(len(cat_cols)):
                    for j in range(i, len(cat_cols)):
                        value = cramers_v(raw_df[cat_cols[i]], raw_df[cat_cols[j]])
                        matrix.iloc[i, j] = value
                        matrix.iloc[j, i] = value
            fig, ax = plt.subplots(figsize=(13, 10))
            sns.heatmap(matrix, annot=True, fmt=".2f", cmap="YlOrRd", ax=ax)
            ax.set_title("Cramér's V Association Matrix")
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
        else:
            st.info("Not enough categorical/ordinal variables were detected.")

    # TAB 4 -----------------------------------------------------------------
    with tab4:
        st.header("🤖 Machine Learning Lab")

        available_targets = [c for c in TARGET_COLUMNS if c in raw_df.columns]
        if not available_targets:
            st.error("No target columns IB1, IB2, or IB3 were found.")
            st.stop()

        target = st.selectbox("Target variable", available_targets)

        st.subheader("1. Feature engineering")
        f1, f2, f3 = st.columns(3)
        with f1:
            add_means = st.checkbox("Add construct means", value=True)
        with f2:
            add_interactions = st.checkbox("Add interaction features", value=True)
        with f3:
            remove_duplicates = st.checkbox("Remove duplicate rows", value=True)

        model_df = raw_df.copy()
        if remove_duplicates:
            model_df = model_df.drop_duplicates().copy()

        model_df, engineered = engineer_features(
            model_df,
            add_means=add_means,
            add_interactions=add_interactions and add_means,
        )

        predictor_candidates = [c for c in model_df.columns if c not in TARGET_COLUMNS]
        default_predictors = predictor_candidates.copy()

        selected_predictors = st.multiselect(
            "Predictor variables",
            predictor_candidates,
            default=default_predictors,
        )

        if engineered:
            st.caption("Engineered features: " + ", ".join(engineered))

        st.info("IB1, IB2 and IB3 are always excluded from the predictor set to prevent target leakage.")

        st.subheader("2. Feature selection")
        use_feature_selection = st.checkbox("Use mutual-information Top-K feature selection", value=False)
        if use_feature_selection:
            max_k = max(1, len(selected_predictors))
            feature_k = st.slider("Keep top K features", 1, max_k, min(10, max_k))
        else:
            feature_k = max(1, len(selected_predictors))

        st.subheader("3. Class imbalance")
        balance_mode = st.radio(
            "Balancing method",
            ["None", "Random Oversampling", "Class Weight"],
            horizontal=True,
            index=1,
        )
        st.caption("Compare methods rather than assuming oversampling is always best.")

        st.subheader("4. Models")
        available_models = get_available_models()
        default_models = [m for m in ["Gradient Boosting", "Random Forest", "SVM (RBF)"] if m in available_models]
        selected_models = st.multiselect("Models to compare", available_models, default=default_models)

        st.subheader("5. Validation")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            test_size = st.slider("Test size", 0.10, 0.40, 0.20, 0.05)
        with c2:
            cv_requested = st.slider("CV folds", 3, 7, 3)
        with c3:
            repeated_cv = st.checkbox("Repeated CV", value=False)
        with c4:
            repeats = st.slider("CV repeats", 2, 4, 2, disabled=not repeated_cv)

        use_optuna = st.checkbox("Use Optuna tuning", value=False, disabled=not OPTUNA_AVAILABLE)
        if use_optuna:
            optuna_trials = st.slider("Optuna trials per supported model", 5, 30, 8, 1)
            st.warning("Optuna + repeated CV + several models is CPU-heavy. Prefer running this locally.")
        else:
            optuna_trials = 0

        if not OPTUNA_AVAILABLE:
            st.caption("Optuna is unavailable in this environment. Add it to requirements.txt.")

        st.markdown("---")

        if st.button("🚀 Run ML Experiment", type="primary"):
            if not selected_predictors:
                st.error("Select at least one predictor.")
                st.stop()
            if not selected_models:
                st.error("Select at least one model.")
                st.stop()

            work = model_df.dropna(subset=[target]).copy()
            X = work[selected_predictors].copy()
            y_original = work[target].copy()

            target_encoder = LabelEncoder()
            y = target_encoder.fit_transform(y_original.astype(str))

            if len(target_encoder.classes_) < 2:
                st.error("The selected target contains only one class.")
                st.stop()

            counts = pd.Series(y).value_counts()
            if counts.min() < 2:
                st.error("At least one target class has fewer than 2 observations.")
                st.stop()

            try:
                X_train, X_test, y_train, y_test = train_test_split(
                    X,
                    y,
                    test_size=test_size,
                    stratify=y,
                    random_state=RANDOM_STATE,
                )
            except ValueError as error:
                st.error(f"Could not create a stratified split: {error}")
                st.stop()

            min_train_class = int(pd.Series(y_train).value_counts().min())
            cv_folds = min(cv_requested, min_train_class)
            if cv_folds < 2:
                st.error("Not enough training observations per class for cross-validation.")
                st.stop()

            cv = make_cv(cv_folds, repeated=repeated_cv, repeats=repeats)

            d1, d2, d3, d4 = st.columns(4)
            d1.metric("Training rows", f"{len(X_train):,}")
            d2.metric("Test rows", f"{len(X_test):,}")
            d3.metric("Predictors", X.shape[1])
            d4.metric("Classes", len(target_encoder.classes_))

            results = []
            trained = {}
            progress = st.progress(0)

            for index, model_name in enumerate(selected_models):
                st.write(f"Training **{model_name}**...")
                best_params = {}
                cv_mean = np.nan
                cv_std = np.nan

                can_tune = model_name not in {"Voting Ensemble", "Stacking Ensemble"}

                try:
                    if use_optuna and can_tune:
                        def objective(trial):
                            params = get_optuna_params(trial, model_name)
                            candidate = create_model(model_name, params=params, balance_mode=balance_mode)
                            pipeline = create_ml_pipeline(
                                X_train,
                                candidate,
                                balance_mode=balance_mode,
                                use_feature_selection=use_feature_selection,
                                feature_k=feature_k,
                                scale_features=model_needs_scaling(model_name),
                            )
                            scores = cross_val_score(
                                pipeline,
                                X_train,
                                y_train,
                                cv=cv,
                                scoring="f1_macro",
                                n_jobs=1,
                            )
                            return float(np.mean(scores))

                        study = optuna.create_study(direction="maximize")
                        study.optimize(objective, n_trials=optuna_trials, show_progress_bar=False)
                        best_params = study.best_params

                    candidate = create_model(model_name, params=best_params, balance_mode=balance_mode)
                    pipeline = create_ml_pipeline(
                        X_train,
                        candidate,
                        balance_mode=balance_mode,
                        use_feature_selection=use_feature_selection,
                        feature_k=feature_k,
                        scale_features=model_needs_scaling(model_name),
                    )

                    cv_scores = cross_val_score(
                        pipeline,
                        X_train,
                        y_train,
                        cv=cv,
                        scoring="f1_macro",
                        n_jobs=1,
                    )
                    cv_mean = float(np.mean(cv_scores))
                    cv_std = float(np.std(cv_scores))

                    pipeline.fit(X_train, y_train)
                    predictions = pipeline.predict(X_test)

                    accuracy = accuracy_score(y_test, predictions)
                    bal_accuracy = balanced_accuracy_score(y_test, predictions)
                    macro_precision = precision_score(y_test, predictions, average="macro", zero_division=0)
                    macro_recall = recall_score(y_test, predictions, average="macro", zero_division=0)
                    macro_f1 = f1_score(y_test, predictions, average="macro", zero_division=0)
                    weighted_f1 = f1_score(y_test, predictions, average="weighted", zero_division=0)

                    results.append(
                        {
                            "Model": model_name,
                            "CV Macro F1": cv_mean,
                            "CV Std": cv_std,
                            "Accuracy": accuracy,
                            "Balanced Accuracy": bal_accuracy,
                            "Macro Precision": macro_precision,
                            "Macro Recall": macro_recall,
                            "Macro F1": macro_f1,
                            "Weighted F1": weighted_f1,
                        }
                    )

                    labels = np.arange(len(target_encoder.classes_))
                    cm = confusion_matrix(y_test, predictions, labels=labels)
                    report = classification_report(
                        y_test,
                        predictions,
                        labels=labels,
                        target_names=[str(x) for x in target_encoder.classes_],
                        output_dict=True,
                        zero_division=0,
                    )

                    trained[model_name] = {
                        "pipeline": pipeline,
                        "predictions": predictions,
                        "y_test": y_test,
                        "confusion_matrix": cm,
                        "classification_report": report,
                        "classes": target_encoder.classes_,
                        "best_params": best_params,
                    }

                except Exception as error:
                    st.error(f"{model_name} failed: {error}")

                progress.progress((index + 1) / len(selected_models))

            if not results:
                st.error("No model completed successfully.")
                st.stop()

            results_df = pd.DataFrame(results).sort_values(
                ["Macro F1", "Balanced Accuracy"], ascending=False
            ).reset_index(drop=True)

            st.session_state.ml_results = results_df
            st.session_state.trained_models = trained
            st.session_state.target_name = target
            st.session_state.target_classes = target_encoder.classes_
            st.session_state.feature_notes = {
                "engineered": engineered,
                "feature_selection": use_feature_selection,
                "feature_k": feature_k if use_feature_selection else None,
                "balance_mode": balance_mode,
                "repeated_cv": repeated_cv,
                "cv_folds": cv_folds,
                "repeats": repeats if repeated_cv else 1,
            }

            st.success("✅ Experiment complete")
            st.dataframe(
                results_df.style.format(
                    {
                        "CV Macro F1": "{:.4f}",
                        "CV Std": "{:.4f}",
                        "Accuracy": "{:.4f}",
                        "Balanced Accuracy": "{:.4f}",
                        "Macro Precision": "{:.4f}",
                        "Macro Recall": "{:.4f}",
                        "Macro F1": "{:.4f}",
                        "Weighted F1": "{:.4f}",
                    }
                ),
                use_container_width=True,
            )

    # TAB 5 -----------------------------------------------------------------
    with tab5:
        st.header("📈 Results")

        if st.session_state.ml_results is None:
            st.info("Run an experiment in the ML Lab first.")
        else:
            results_df = st.session_state.ml_results.copy()
            trained = st.session_state.trained_models
            target_name = st.session_state.target_name
            notes = st.session_state.feature_notes or {}

            st.subheader(f"Target: {target_name}")
            st.caption(
                f"Balancing: {notes.get('balance_mode')} | "
                f"CV: {notes.get('cv_folds')} folds × {notes.get('repeats', 1)} repeat(s) | "
                f"Feature selection: {'Top-' + str(notes.get('feature_k')) if notes.get('feature_selection') else 'Off'}"
            )

            st.dataframe(
                results_df.style.format(
                    {
                        "CV Macro F1": "{:.4f}",
                        "CV Std": "{:.4f}",
                        "Accuracy": "{:.4f}",
                        "Balanced Accuracy": "{:.4f}",
                        "Macro Precision": "{:.4f}",
                        "Macro Recall": "{:.4f}",
                        "Macro F1": "{:.4f}",
                        "Weighted F1": "{:.4f}",
                    }
                ),
                use_container_width=True,
            )

            st.subheader("Accuracy Comparison")
            chart_df = results_df.sort_values("Accuracy")
            fig, ax = plt.subplots(figsize=(9, 5))
            bars = ax.barh(chart_df["Model"], chart_df["Accuracy"])
            ax.set_xlim(0, 1)
            ax.set_xlabel("Accuracy")
            for bar, value in zip(bars, chart_df["Accuracy"]):
                ax.text(value + 0.01, bar.get_y() + bar.get_height() / 2, f"{value:.3f}", va="center")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

            st.subheader("Macro F1 and Balanced Accuracy")
            comparison = results_df[["Model", "Macro F1", "Balanced Accuracy"]].set_index("Model")
            st.bar_chart(comparison)

            st.subheader("Cross-validation Stability")
            cv_display = results_df[["Model", "CV Macro F1", "CV Std"]].copy()
            st.dataframe(cv_display, use_container_width=True)
            st.caption("Lower CV Std means the model is more stable across validation folds/repeats.")

            st.subheader("Confusion Matrices")
            for model_name, data in trained.items():
                with st.expander(model_name, expanded=False):
                    fig, ax = plt.subplots(figsize=(6, 5))
                    sns.heatmap(
                        data["confusion_matrix"],
                        annot=True,
                        fmt="d",
                        cmap="Blues",
                        xticklabels=data["classes"],
                        yticklabels=data["classes"],
                        ax=ax,
                    )
                    ax.set_xlabel("Predicted")
                    ax.set_ylabel("Actual")
                    ax.set_title(f"{model_name} Confusion Matrix")
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close(fig)

                    report_df = pd.DataFrame(data["classification_report"]).transpose()
                    st.markdown("**Classification report**")
                    st.dataframe(report_df.round(4), use_container_width=True)

                    if data["best_params"]:
                        st.markdown("**Best Optuna parameters**")
                        st.json(data["best_params"])

            st.subheader("📥 Download")
            st.download_button(
                "Download results CSV",
                data=results_df.to_csv(index=False),
                file_name=f"{target_name}_improved_ml_results.csv",
                mime="text/csv",
            )

else:
    st.markdown(
        """
        ## Welcome 👋

        Upload your CSV file from the sidebar.

        This version can test several legitimate ways to improve predictive performance:

        - construct-level mean features
        - interaction features
        - mutual-information Top-K feature selection
        - RandomOverSampler versus class weighting versus no balancing
        - Random Forest / Extra Trees / Gradient Boosting / HistGradientBoosting
        - SVM and Logistic Regression
        - XGBoost / LightGBM / CatBoost when installed
        - soft-voting and stacking ensembles
        - stratified or repeated stratified cross-validation
        - optional Optuna tuning

        **IB1, IB2 and IB3 are never used as predictors for each other.**
        """
    )

st.markdown("---")
st.markdown(
    '<p style="text-align:center; color:gray;">Impulsive Buying Behaviour Analysis | MRes Artificial Intelligence</p>',
    unsafe_allow_html=True,
)
