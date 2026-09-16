# ============================================================
# IMPULSIVE BUYING BEHAVIOUR ANALYSIS
# Streamlit Machine Learning Application
# ============================================================

import warnings

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import chi2_contingency

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_val_score
)

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline as SklearnPipeline

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier
)

from imblearn.pipeline import Pipeline
from imblearn.over_sampling import RandomOverSampler


# ============================================================
# OPTIONAL ML LIBRARIES
# ============================================================

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False


try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False


try:
    from catboost import CatBoostClassifier
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False


try:
    import optuna
    OPTUNA_AVAILABLE = True
    optuna.logging.set_verbosity(optuna.logging.WARNING)
except ImportError:
    OPTUNA_AVAILABLE = False


# ============================================================
# SETTINGS
# ============================================================

warnings.filterwarnings("ignore")

RANDOM_STATE = 42

TARGET_COLUMNS = [
    "IB1",
    "IB2",
    "IB3"
]


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Impulsive Buying Analysis",
    page_icon="🛒",
    layout="wide"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 2.5rem;
        font-weight: bold;
        color: #2C3E50;
        text-align: center;
        margin-bottom: 0.5rem;
    }

    .subtitle {
        font-size: 1.1rem;
        color: #7F8C8D;
        text-align: center;
        margin-bottom: 2rem;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
    <p class="main-title">
    🛒 Impulsive Buying Behaviour Analysis
    </p>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <p class="subtitle">
    Machine Learning Analysis |
    MRes Artificial Intelligence |
    University of Wolverhampton
    </p>
    """,
    unsafe_allow_html=True
)


# ============================================================
# SESSION STATE
# ============================================================

if "df" not in st.session_state:
    st.session_state.df = None

if "ml_results" not in st.session_state:
    st.session_state.ml_results = None

if "trained_models" not in st.session_state:
    st.session_state.trained_models = {}

if "target_name" not in st.session_state:
    st.session_state.target_name = None


# ============================================================
# CRAMER'S V
# ============================================================

def cramers_v(x, y):

    data = pd.DataFrame(
        {
            "x": x,
            "y": y
        }
    ).dropna()

    if data.empty:
        return np.nan

    table = pd.crosstab(
        data["x"],
        data["y"]
    )

    if table.empty:
        return np.nan

    chi2 = chi2_contingency(
        table,
        correction=False
    )[0]

    n = table.to_numpy().sum()

    if n <= 1:
        return 0

    phi2 = chi2 / n

    r, k = table.shape

    phi2corr = max(
        0,
        phi2
        - ((k - 1) * (r - 1))
        / (n - 1)
    )

    rcorr = (
        r
        - ((r - 1) ** 2)
        / (n - 1)
    )

    kcorr = (
        k
        - ((k - 1) ** 2)
        / (n - 1)
    )

    denominator = min(
        kcorr - 1,
        rcorr - 1
    )

    if denominator <= 0:
        return 0

    return np.sqrt(
        phi2corr / denominator
    )


# ============================================================
# ONE-HOT ENCODER
# ============================================================

def make_one_hot_encoder():
    """
    Supports both newer and slightly older
    versions of scikit-learn.
    """

    try:
        return OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False
        )

    except TypeError:
        return OneHotEncoder(
            handle_unknown="ignore",
            sparse=False
        )


# ============================================================
# PREPROCESSING
# ============================================================

def create_preprocessor(X):

    numeric_features = (
        X
        .select_dtypes(include=np.number)
        .columns
        .tolist()
    )

    categorical_features = (
        X
        .select_dtypes(exclude=np.number)
        .columns
        .tolist()
    )


    numeric_pipeline = SklearnPipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                )
            )
        ]
    )


    categorical_pipeline = SklearnPipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent"
                )
            ),
            (
                "encoder",
                make_one_hot_encoder()
            )
        ]
    )


    transformers = []


    if numeric_features:

        transformers.append(
            (
                "numeric",
                numeric_pipeline,
                numeric_features
            )
        )


    if categorical_features:

        transformers.append(
            (
                "categorical",
                categorical_pipeline,
                categorical_features
            )
        )


    return ColumnTransformer(
        transformers=transformers,
        remainder="drop"
    )


# ============================================================
# CREATE MODEL
# ============================================================

def create_model(
    model_name,
    params=None,
    use_oversampling=True
):

    params = params or {}


    # --------------------------------------------------------
    # Random Forest
    # --------------------------------------------------------

    if model_name == "Random Forest":

        defaults = {
            "n_estimators": 200,
            "max_depth": 10,
            "min_samples_split": 2,
            "min_samples_leaf": 1,
            "random_state": RANDOM_STATE,
            "n_jobs": -1,
            "class_weight":
                None
                if use_oversampling
                else "balanced"
        }

        defaults.update(params)

        return RandomForestClassifier(
            **defaults
        )


    # --------------------------------------------------------
    # Extra Trees
    # --------------------------------------------------------

    elif model_name == "Extra Trees":

        defaults = {
            "n_estimators": 200,
            "max_depth": 10,
            "min_samples_split": 2,
            "min_samples_leaf": 1,
            "random_state": RANDOM_STATE,
            "n_jobs": -1,
            "class_weight":
                None
                if use_oversampling
                else "balanced"
        }

        defaults.update(params)

        return ExtraTreesClassifier(
            **defaults
        )


    # --------------------------------------------------------
    # Gradient Boosting
    # --------------------------------------------------------

    elif model_name == "Gradient Boosting":

        defaults = {
            "n_estimators": 100,
            "max_depth": 3,
            "learning_rate": 0.05,
            "random_state": RANDOM_STATE
        }

        defaults.update(params)

        return GradientBoostingClassifier(
            **defaults
        )


    # --------------------------------------------------------
    # XGBoost
    # --------------------------------------------------------

    elif model_name == "XGBoost":

        if not XGBOOST_AVAILABLE:
            raise ImportError(
                "XGBoost is not installed."
            )

        defaults = {
            "n_estimators": 200,
            "max_depth": 5,
            "learning_rate": 0.05,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "eval_metric": "mlogloss",
            "random_state": RANDOM_STATE,
            "n_jobs": -1,
            "verbosity": 0
        }

        defaults.update(params)

        return xgb.XGBClassifier(
            **defaults
        )


    # --------------------------------------------------------
    # LightGBM
    # --------------------------------------------------------

    elif model_name == "LightGBM":

        if not LIGHTGBM_AVAILABLE:
            raise ImportError(
                "LightGBM is not installed."
            )

        defaults = {
            "n_estimators": 200,
            "max_depth": 8,
            "learning_rate": 0.05,
            "random_state": RANDOM_STATE,
            "n_jobs": -1,
            "verbosity": -1
        }

        defaults.update(params)

        return lgb.LGBMClassifier(
            **defaults
        )


    # --------------------------------------------------------
    # CatBoost
    # --------------------------------------------------------

    elif model_name == "CatBoost":

        if not CATBOOST_AVAILABLE:
            raise ImportError(
                "CatBoost is not installed."
            )

        defaults = {
            "iterations": 200,
            "depth": 6,
            "learning_rate": 0.05,
            "random_state": RANDOM_STATE,
            "verbose": False,
            "allow_writing_files": False
        }

        defaults.update(params)

        return CatBoostClassifier(
            **defaults
        )


    else:

        raise ValueError(
            f"Unknown model: {model_name}"
        )


# ============================================================
# OPTUNA PARAMETERS
# ============================================================

def get_optuna_params(
    trial,
    model_name
):

    if model_name == "Random Forest":

        return {
            "n_estimators":
                trial.suggest_int(
                    "n_estimators",
                    100,
                    400
                ),

            "max_depth":
                trial.suggest_int(
                    "max_depth",
                    3,
                    20
                ),

            "min_samples_split":
                trial.suggest_int(
                    "min_samples_split",
                    2,
                    10
                ),

            "min_samples_leaf":
                trial.suggest_int(
                    "min_samples_leaf",
                    1,
                    5
                )
        }


    elif model_name == "Extra Trees":

        return {
            "n_estimators":
                trial.suggest_int(
                    "n_estimators",
                    100,
                    400
                ),

            "max_depth":
                trial.suggest_int(
                    "max_depth",
                    3,
                    20
                ),

            "min_samples_split":
                trial.suggest_int(
                    "min_samples_split",
                    2,
                    10
                ),

            "min_samples_leaf":
                trial.suggest_int(
                    "min_samples_leaf",
                    1,
                    5
                )
        }


    elif model_name == "Gradient Boosting":

        return {
            "n_estimators":
                trial.suggest_int(
                    "n_estimators",
                    50,
                    300
                ),

            "max_depth":
                trial.suggest_int(
                    "max_depth",
                    2,
                    6
                ),

            "learning_rate":
                trial.suggest_float(
                    "learning_rate",
                    0.01,
                    0.20,
                    log=True
                )
        }


    elif model_name == "XGBoost":

        return {
            "n_estimators":
                trial.suggest_int(
                    "n_estimators",
                    100,
                    400
                ),

            "max_depth":
                trial.suggest_int(
                    "max_depth",
                    3,
                    10
                ),

            "learning_rate":
                trial.suggest_float(
                    "learning_rate",
                    0.01,
                    0.20,
                    log=True
                ),

            "subsample":
                trial.suggest_float(
                    "subsample",
                    0.7,
                    1.0
                ),

            "colsample_bytree":
                trial.suggest_float(
                    "colsample_bytree",
                    0.7,
                    1.0
                )
        }


    elif model_name == "LightGBM":

        return {
            "n_estimators":
                trial.suggest_int(
                    "n_estimators",
                    100,
                    400
                ),

            "max_depth":
                trial.suggest_int(
                    "max_depth",
                    3,
                    15
                ),

            "learning_rate":
                trial.suggest_float(
                    "learning_rate",
                    0.01,
                    0.20,
                    log=True
                ),

            "num_leaves":
                trial.suggest_int(
                    "num_leaves",
                    10,
                    60
                )
        }


    elif model_name == "CatBoost":

        return {
            "iterations":
                trial.suggest_int(
                    "iterations",
                    100,
                    400
                ),

            "depth":
                trial.suggest_int(
                    "depth",
                    4,
                    10
                ),

            "learning_rate":
                trial.suggest_float(
                    "learning_rate",
                    0.01,
                    0.20,
                    log=True
                )
        }


    return {}


# ============================================================
# CREATE ML PIPELINE
# ============================================================

def create_ml_pipeline(
    X,
    model,
    use_oversampling
):

    preprocessor = create_preprocessor(X)

    steps = [
        (
            "preprocessing",
            preprocessor
        )
    ]


    if use_oversampling:

        steps.append(
            (
                "oversampling",
                RandomOverSampler(
                    random_state=RANDOM_STATE
                )
            )
        )


    steps.append(
        (
            "model",
            model
        )
    )


    return Pipeline(
        steps=steps
    )


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.title(
        "🎯 Prediction"
    )

    st.markdown("---")


    uploaded_file = st.file_uploader(
        "📂 Upload CSV dataset",
        type=["csv"]
    )


    if uploaded_file is not None:

        try:

            loaded_df = pd.read_csv(
                uploaded_file
            )

            st.session_state.df = (
                loaded_df
            )

            st.success(
                f"✅ Loaded "
                f"{len(loaded_df):,} rows"
            )

        except Exception as error:

            st.error(
                f"Could not read CSV: "
                f"{error}"
            )


    st.markdown("---")


    st.markdown(
        "### About This App"
    )


    st.info(
        """
        This application includes:

        📊 Data Overview

        🔍 Exploratory Data Analysis

        🔗 Cramér's V Correlation

        🤖 Machine Learning Models

        ⚙️ Optuna Hyperparameter Tuning

        📈 Performance Comparison

        🧩 Confusion Matrices

        📋 Classification Reports
        """
    )


# ============================================================
# MAIN APP
# ============================================================

if st.session_state.df is not None:

    df = st.session_state.df.copy()


    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        [
            "📊 Data Overview",
            "🔍 EDA",
            "🔗 Correlations",
            "🤖 ML Models",
            "📈 Results"
        ]
    )


    # ========================================================
    # TAB 1
    # DATA OVERVIEW
    # ========================================================

    with tab1:

        st.header(
            "📊 Data Overview"
        )


        col1, col2, col3, col4 = (
            st.columns(4)
        )


        col1.metric(
            "Rows",
            f"{df.shape[0]:,}"
        )


        col2.metric(
            "Columns",
            df.shape[1]
        )


        col3.metric(
            "Missing Values",
            f"{df.isna().sum().sum():,}"
        )


        memory_kb = (
            df
            .memory_usage(deep=True)
            .sum()
            / 1024
        )


        col4.metric(
            "Memory",
            f"{memory_kb:.1f} KB"
        )


        # ----------------------------------------------------
        # Data Preview
        # ----------------------------------------------------

        st.subheader(
            "Data Preview"
        )


        st.dataframe(
            df.head(10),
            use_container_width=True
        )


        # ----------------------------------------------------
        # Column Summary
        # ----------------------------------------------------

        st.subheader(
            "Column Summary"
        )


        column_info = pd.DataFrame(
            {
                "Column":
                    df.columns,

                "Data Type":
                    df.dtypes
                    .astype(str)
                    .values,

                "Unique Values":
                    df.nunique()
                    .values,

                "Missing":
                    df.isna()
                    .sum()
                    .values,

                "Missing %":
                    (
                        df.isna()
                        .mean()
                        * 100
                    )
                    .round(2)
                    .values
            }
        )


        st.dataframe(
            column_info,
            use_container_width=True
        )


        # ----------------------------------------------------
        # Target Summary
        # ----------------------------------------------------

        available_targets = [
            column
            for column in TARGET_COLUMNS
            if column in df.columns
        ]


        if available_targets:

            st.subheader(
                "Target Variable Summary"
            )


            for target in available_targets:

                st.markdown(
                    f"#### {target}"
                )

                counts = (
                    df[target]
                    .value_counts(
                        dropna=False
                    )
                    .sort_index()
                )

                summary_df = (
                    counts
                    .rename("Count")
                    .reset_index()
                )

                summary_df.columns = [
                    "Class",
                    "Count"
                ]


                st.dataframe(
                    summary_df,
                    use_container_width=True
                )


    # ========================================================
    # TAB 2
    # EDA
    # ========================================================

    with tab2:

        st.header(
            "🔍 Exploratory Data Analysis"
        )


        # ----------------------------------------------------
        # Variable Classification
        # ----------------------------------------------------

        st.subheader(
            "Variable Classification"
        )


        variable_information = []


        for column in df.columns:

            number_unique = (
                df[column].nunique()
            )


            if (
                df[column].dtype
                == "object"
            ):

                variable_type = (
                    "Categorical"
                )


            elif number_unique <= 10:

                variable_type = (
                    "Categorical / Ordinal"
                )


            else:

                variable_type = (
                    "Numeric"
                )


            variable_information.append(
                {
                    "Column":
                        column,

                    "Type":
                        variable_type,

                    "Unique Values":
                        number_unique
                }
            )


        st.dataframe(
            pd.DataFrame(
                variable_information
            ),
            use_container_width=True
        )


        # ----------------------------------------------------
        # Target Distributions
        # ----------------------------------------------------

        available_targets = [
            target
            for target in TARGET_COLUMNS
            if target in df.columns
        ]


        if available_targets:

            st.subheader(
                "Target Variable Distributions"
            )


            fig, axes = plt.subplots(
                1,
                len(available_targets),
                figsize=(
                    5
                    * len(available_targets),
                    4
                )
            )


            if (
                len(available_targets)
                == 1
            ):

                axes = [axes]


            for i, target in enumerate(
                available_targets
            ):

                counts = (
                    df[target]
                    .value_counts()
                    .sort_index()
                )


                axes[i].bar(
                    counts.index.astype(str),
                    counts.values
                )


                axes[i].set_title(
                    f"{target} Distribution"
                )


                axes[i].set_xlabel(
                    "Response"
                )


                axes[i].set_ylabel(
                    "Count"
                )


                for j, value in enumerate(
                    counts.values
                ):

                    axes[i].text(
                        j,
                        value,
                        str(value),
                        ha="center",
                        va="bottom"
                    )


            plt.tight_layout()

            st.pyplot(fig)

            plt.close(fig)


        # ----------------------------------------------------
        # Feature Histograms
        # ----------------------------------------------------

        st.subheader(
            "Feature Distributions"
        )


        numeric_columns = (
            df
            .select_dtypes(
                include=np.number
            )
            .columns
            .tolist()
        )


        if numeric_columns:

            selected_features = (
                st.multiselect(
                    "Select variables",
                    numeric_columns,
                    default=
                        numeric_columns[:6]
                )
            )


            if selected_features:

                n_columns = min(
                    3,
                    len(selected_features)
                )


                n_rows = (
                    len(selected_features)
                    + n_columns
                    - 1
                ) // n_columns


                fig, axes = plt.subplots(
                    n_rows,
                    n_columns,
                    figsize=(
                        5 * n_columns,
                        4 * n_rows
                    )
                )


                axes = np.array(
                    axes
                ).reshape(-1)


                for i, column in enumerate(
                    selected_features
                ):

                    axes[i].hist(
                        df[column]
                        .dropna(),
                        bins=15,
                        edgecolor="white"
                    )


                    axes[i].set_title(
                        column
                    )


                    axes[i].set_xlabel(
                        "Value"
                    )


                    axes[i].set_ylabel(
                        "Frequency"
                    )


                for i in range(
                    len(selected_features),
                    len(axes)
                ):

                    axes[i].set_visible(
                        False
                    )


                plt.tight_layout()

                st.pyplot(fig)

                plt.close(fig)


    # ========================================================
    # TAB 3
    # CRAMER'S V
    # ========================================================

    with tab3:

        st.header(
            "🔗 Correlation Analysis"
        )


        st.subheader(
            "Cramér's V"
        )


        st.caption(
            "Cramér's V measures association "
            "between categorical or ordinal variables."
        )


        categorical_columns = [
            column
            for column in df.columns
            if (
                df[column].dtype == "object"
                or
                df[column].nunique() <= 10
            )
        ]


        if len(categorical_columns) > 1:

            with st.spinner(
                "Calculating Cramér's V..."
            ):

                matrix = pd.DataFrame(
                    np.zeros(
                        (
                            len(categorical_columns),
                            len(categorical_columns)
                        )
                    ),
                    index=categorical_columns,
                    columns=categorical_columns
                )


                for i in range(
                    len(categorical_columns)
                ):

                    for j in range(
                        i,
                        len(categorical_columns)
                    ):

                        value = cramers_v(
                            df[
                                categorical_columns[i]
                            ],
                            df[
                                categorical_columns[j]
                            ]
                        )


                        matrix.iloc[
                            i,
                            j
                        ] = value


                        matrix.iloc[
                            j,
                            i
                        ] = value


                fig, ax = plt.subplots(
                    figsize=(14, 11)
                )


                sns.heatmap(
                    matrix,
                    annot=True,
                    cmap="YlOrRd",
                    fmt=".2f",
                    ax=ax
                )


                ax.set_title(
                    "Cramér's V Correlation Matrix"
                )


                plt.xticks(
                    rotation=45,
                    ha="right"
                )


                plt.yticks(
                    rotation=0
                )


                plt.tight_layout()


                st.pyplot(fig)


                plt.close(fig)


        else:

            st.info(
                "Not enough categorical or "
                "ordinal variables are available."
            )


    # ========================================================
    # TAB 4
    # MACHINE LEARNING
    # ========================================================

    with tab4:

        st.header(
            "🤖 Machine Learning Models"
        )


        available_targets = [
            target
            for target in TARGET_COLUMNS
            if target in df.columns
        ]


        if not available_targets:

            st.error(
                "No IB target variables were found. "
                "The dataset should contain "
                "IB1, IB2 and/or IB3."
            )


        else:

            # ------------------------------------------------
            # Model choices
            # ------------------------------------------------

            available_models = [
                "Random Forest",
                "Extra Trees",
                "Gradient Boosting"
            ]


            if XGBOOST_AVAILABLE:
                available_models.append(
                    "XGBoost"
                )


            if LIGHTGBM_AVAILABLE:
                available_models.append(
                    "LightGBM"
                )


            if CATBOOST_AVAILABLE:
                available_models.append(
                    "CatBoost"
                )


            col1, col2 = st.columns(2)


            with col1:

                target = st.selectbox(
                    "Target Variable",
                    available_targets
                )


            with col2:

                default_models = [
                    model
                    for model in [
                        "Random Forest",
                        "XGBoost",
                        "LightGBM"
                    ]
                    if model
                    in available_models
                ]


                selected_models = (
                    st.multiselect(
                        "Select Models",
                        available_models,
                        default=default_models
                    )
                )


            # ------------------------------------------------
            # Predictor Selection
            # ------------------------------------------------

            predictor_candidates = [
                column
                for column in df.columns
                if column
                not in TARGET_COLUMNS
            ]


            st.subheader(
                "Predictor Variables"
            )


            selected_predictors = (
                st.multiselect(
                    "Select features used for prediction",
                    predictor_candidates,
                    default=predictor_candidates
                )
            )


            st.info(
                "IB1, IB2 and IB3 are excluded from "
                "the predictor variables to prevent "
                "target leakage."
            )


            # ------------------------------------------------
            # Training Settings
            # ------------------------------------------------

            st.subheader(
                "Training Settings"
            )


            col3, col4, col5 = (
                st.columns(3)
            )


            with col3:

                test_size = st.slider(
                    "Test Size",
                    min_value=0.10,
                    max_value=0.40,
                    value=0.20,
                    step=0.05
                )


            with col4:

                use_oversampling = (
                    st.checkbox(
                        "Random Oversampling",
                        value=True
                    )
                )


            with col5:

                use_optuna = (
                    st.checkbox(
                        "Optuna Tuning",
                        value=False,
                        disabled=
                            not OPTUNA_AVAILABLE
                    )
                )


            if not OPTUNA_AVAILABLE:

                st.caption(
                    "Optuna is unavailable. "
                    "Add optuna to requirements.txt."
                )


            if use_optuna:

                optuna_trials = st.slider(
                    "Number of Optuna Trials",
                    min_value=5,
                    max_value=50,
                    value=10,
                    step=5
                )

            else:

                optuna_trials = 0


            cv_requested = st.slider(
                "Cross-validation Folds",
                min_value=3,
                max_value=10,
                value=5
            )


            # =================================================
            # TRAIN BUTTON
            # =================================================

            if st.button(
                "🚀 Train Models",
                type="primary"
            ):

                # --------------------------------------------
                # Validation
                # --------------------------------------------

                if not selected_models:

                    st.error(
                        "Please select at least "
                        "one model."
                    )

                    st.stop()


                if not selected_predictors:

                    st.error(
                        "Please select at least "
                        "one predictor variable."
                    )

                    st.stop()


                # --------------------------------------------
                # Remove rows missing target
                # --------------------------------------------

                model_df = (
                    df
                    .dropna(
                        subset=[target]
                    )
                    .copy()
                )


                # --------------------------------------------
                # X and y
                # --------------------------------------------

                X = model_df[
                    selected_predictors
                ].copy()


                y_original = (
                    model_df[target]
                )


                # --------------------------------------------
                # Encode target
                # --------------------------------------------

                target_encoder = (
                    LabelEncoder()
                )


                y = (
                    target_encoder
                    .fit_transform(
                        y_original.astype(str)
                    )
                )


                number_classes = (
                    len(
                        target_encoder.classes_
                    )
                )


                if number_classes < 2:

                    st.error(
                        "The selected target has "
                        "only one class."
                    )

                    st.stop()


                # --------------------------------------------
                # Class Counts
                # --------------------------------------------

                class_counts = (
                    pd.Series(y)
                    .value_counts()
                )


                if class_counts.min() < 2:

                    st.error(
                        "At least one class has fewer "
                        "than 2 observations. "
                        "More data are required."
                    )

                    st.stop()


                # --------------------------------------------
                # Train / Test Split
                # --------------------------------------------

                try:

                    (
                        X_train,
                        X_test,
                        y_train,
                        y_test
                    ) = train_test_split(
                        X,
                        y,
                        test_size=test_size,
                        stratify=y,
                        random_state=
                            RANDOM_STATE
                    )

                except ValueError as error:

                    st.error(
                        "Unable to create a stratified "
                        f"train/test split: {error}"
                    )

                    st.stop()


                # --------------------------------------------
                # CV folds
                # --------------------------------------------

                training_counts = (
                    pd.Series(y_train)
                    .value_counts()
                )


                minimum_training_class = (
                    int(
                        training_counts.min()
                    )
                )


                cv_folds = min(
                    cv_requested,
                    minimum_training_class
                )


                if cv_folds < 2:

                    st.error(
                        "There are not enough training "
                        "samples per class for "
                        "cross-validation."
                    )

                    st.stop()


                cv = StratifiedKFold(
                    n_splits=cv_folds,
                    shuffle=True,
                    random_state=
                        RANDOM_STATE
                )


                # --------------------------------------------
                # Dataset Information
                # --------------------------------------------

                st.write(
                    f"Training observations: "
                    f"**{len(X_train):,}**"
                )


                st.write(
                    f"Testing observations: "
                    f"**{len(X_test):,}**"
                )


                st.write(
                    f"Predictor variables: "
                    f"**{X.shape[1]}**"
                )


                st.write(
                    f"Target classes: "
                    f"**{number_classes}**"
                )


                st.write(
                    f"Cross-validation folds: "
                    f"**{cv_folds}**"
                )


                results = []

                trained_models = {}

                progress_bar = (
                    st.progress(0)
                )


                # =============================================
                # MODEL LOOP
                # =============================================

                for model_index, model_name in enumerate(
                    selected_models
                ):

                    st.markdown(
                        f"### {model_name}"
                    )


                    best_params = {}

                    cv_score = np.nan


                    # =========================================
                    # OPTUNA
                    # =========================================

                    if (
                        use_optuna
                        and OPTUNA_AVAILABLE
                    ):

                        with st.spinner(
                            f"Tuning {model_name}..."
                        ):

                            def objective(
                                trial
                            ):

                                trial_params = (
                                    get_optuna_params(
                                        trial,
                                        model_name
                                    )
                                )


                                model = (
                                    create_model(
                                        model_name,
                                        trial_params,
                                        use_oversampling
                                    )
                                )


                                pipeline = (
                                    create_ml_pipeline(
                                        X_train,
                                        model,
                                        use_oversampling
                                    )
                                )


                                scores = (
                                    cross_val_score(
                                        pipeline,
                                        X_train,
                                        y_train,
                                        cv=cv,
                                        scoring=
                                            "f1_macro",
                                        n_jobs=1
                                    )
                                )


                                return (
                                    scores.mean()
                                )


                            study = (
                                optuna
                                .create_study(
                                    direction=
                                        "maximize"
                                )
                            )


                            study.optimize(
                                objective,
                                n_trials=
                                    optuna_trials,
                                show_progress_bar=
                                    False
                            )


                            best_params = (
                                study.best_params
                            )


                            cv_score = (
                                study.best_value
                            )


                    # =========================================
                    # NORMAL CROSS VALIDATION
                    # =========================================

                    else:

                        model = create_model(
                            model_name,
                            use_oversampling=
                                use_oversampling
                        )


                        pipeline = (
                            create_ml_pipeline(
                                X_train,
                                model,
                                use_oversampling
                            )
                        )


                        with st.spinner(
                            f"Cross-validating "
                            f"{model_name}..."
                        ):

                            scores = (
                                cross_val_score(
                                    pipeline,
                                    X_train,
                                    y_train,
                                    cv=cv,
                                    scoring=
                                        "f1_macro",
                                    n_jobs=1
                                )
                            )


                        cv_score = (
                            scores.mean()
                        )


                    # =========================================
                    # FINAL MODEL
                    # =========================================

                    final_model = (
                        create_model(
                            model_name,
                            best_params,
                            use_oversampling
                        )
                    )


                    final_pipeline = (
                        create_ml_pipeline(
                            X_train,
                            final_model,
                            use_oversampling
                        )
                    )


                    with st.spinner(
                        f"Training final "
                        f"{model_name}..."
                    ):

                        final_pipeline.fit(
                            X_train,
                            y_train
                        )


                    # =========================================
                    # TEST SET
                    # =========================================

                    predictions = (
                        final_pipeline.predict(
                            X_test
                        )
                    )


                    # =========================================
                    # METRICS
                    # =========================================

                    accuracy = (
                        accuracy_score(
                            y_test,
                            predictions
                        )
                    )


                    balanced_accuracy = (
                        balanced_accuracy_score(
                            y_test,
                            predictions
                        )
                    )


                    precision_macro = (
                        precision_score(
                            y_test,
                            predictions,
                            average="macro",
                            zero_division=0
                        )
                    )


                    recall_macro = (
                        recall_score(
                            y_test,
                            predictions,
                            average="macro",
                            zero_division=0
                        )
                    )


                    macro_f1 = (
                        f1_score(
                            y_test,
                            predictions,
                            average="macro",
                            zero_division=0
                        )
                    )


                    weighted_f1 = (
                        f1_score(
                            y_test,
                            predictions,
                            average="weighted",
                            zero_division=0
                        )
                    )


                    results.append(
                        {
                            "Model":
                                model_name,

                            "CV Macro F1":
                                cv_score,

                            "Accuracy":
                                accuracy,

                            "Balanced Accuracy":
                                balanced_accuracy,

                            "Macro Precision":
                                precision_macro,

                            "Macro Recall":
                                recall_macro,

                            "Macro F1":
                                macro_f1,

                            "Weighted F1":
                                weighted_f1
                        }
                    )


                    # =========================================
                    # CONFUSION MATRIX
                    # =========================================

                    all_labels = np.arange(
                        number_classes
                    )


                    cm = confusion_matrix(
                        y_test,
                        predictions,
                        labels=all_labels
                    )


                    # =========================================
                    # CLASSIFICATION REPORT
                    # =========================================

                    report = (
                        classification_report(
                            y_test,
                            predictions,
                            labels=all_labels,
                            target_names=[
                                str(x)
                                for x
                                in target_encoder.classes_
                            ],
                            output_dict=True,
                            zero_division=0
                        )
                    )


                    trained_models[
                        model_name
                    ] = {
                        "pipeline":
                            final_pipeline,

                        "predictions":
                            predictions,

                        "y_test":
                            y_test,

                        "confusion_matrix":
                            cm,

                        "classification_report":
                            report,

                        "classes":
                            target_encoder.classes_,

                        "best_params":
                            best_params
                    }


                    progress_bar.progress(
                        (
                            model_index + 1
                        )
                        /
                        len(selected_models)
                    )


                # =============================================
                # SAVE RESULTS
                # =============================================

                results_df = (
                    pd.DataFrame(
                        results
                    )
                    .sort_values(
                        "Macro F1",
                        ascending=False
                    )
                    .reset_index(
                        drop=True
                    )
                )


                st.session_state.ml_results = (
                    results_df
                )


                st.session_state.trained_models = (
                    trained_models
                )


                st.session_state.target_name = (
                    target
                )


                st.success(
                    "✅ Training Complete!"
                )


                st.subheader(
                    "Model Performance"
                )


                st.dataframe(
                    results_df.style.format(
                        {
                            "CV Macro F1":
                                "{:.4f}",

                            "Accuracy":
                                "{:.4f}",

                            "Balanced Accuracy":
                                "{:.4f}",

                            "Macro Precision":
                                "{:.4f}",

                            "Macro Recall":
                                "{:.4f}",

                            "Macro F1":
                                "{:.4f}",

                            "Weighted F1":
                                "{:.4f}"
                        }
                    ),
                    use_container_width=True
                )


    # ========================================================
    # TAB 5
    # RESULTS
    # ========================================================

    with tab5:

        st.header(
            "📈 Results"
        )


        if (
            st.session_state.ml_results
            is None
        ):

            st.info(
                "👆 Train models first "
                "in the ML Models tab."
            )


        else:

            results_df = (
                st.session_state
                .ml_results
                .copy()
            )


            trained_models = (
                st.session_state
                .trained_models
            )


            target_name = (
                st.session_state
                .target_name
            )


            st.subheader(
                f"Target Variable: "
                f"{target_name}"
            )


            # ------------------------------------------------
            # Results Table
            # ------------------------------------------------

            st.subheader(
                "Performance Comparison"
            )


            st.dataframe(
                results_df.style.format(
                    {
                        "CV Macro F1":
                            "{:.4f}",

                        "Accuracy":
                            "{:.4f}",

                        "Balanced Accuracy":
                            "{:.4f}",

                        "Macro Precision":
                            "{:.4f}",

                        "Macro Recall":
                            "{:.4f}",

                        "Macro F1":
                            "{:.4f}",

                        "Weighted F1":
                            "{:.4f}"
                    }
                ),
                use_container_width=True
            )


            # ------------------------------------------------
            # Macro F1
            # ------------------------------------------------

            st.subheader(
                "Macro F1 Comparison"
            )


            plot_df = (
                results_df
                .sort_values(
                    "Macro F1"
                )
            )


            fig, ax = plt.subplots(
                figsize=(9, 5)
            )


            bars = ax.barh(
                plot_df["Model"],
                plot_df["Macro F1"]
            )


            ax.set_xlabel(
                "Macro F1 Score"
            )


            ax.set_xlim(
                0,
                1
            )


            ax.set_title(
                "Test Set Macro F1"
            )


            for bar, value in zip(
                bars,
                plot_df["Macro F1"]
            ):

                ax.text(
                    value + 0.01,
                    bar.get_y()
                    + bar.get_height()
                    / 2,
                    f"{value:.3f}",
                    va="center"
                )


            plt.tight_layout()

            st.pyplot(fig)

            plt.close(fig)


            # ------------------------------------------------
            # Balanced Accuracy
            # ------------------------------------------------

            st.subheader(
                "Balanced Accuracy"
            )


            balanced_df = (
                results_df
                .sort_values(
                    "Balanced Accuracy"
                )
            )


            fig, ax = plt.subplots(
                figsize=(9, 5)
            )


            bars = ax.barh(
                balanced_df["Model"],
                balanced_df[
                    "Balanced Accuracy"
                ]
            )


            ax.set_xlabel(
                "Balanced Accuracy"
            )


            ax.set_xlim(
                0,
                1
            )


            ax.set_title(
                "Balanced Accuracy by Model"
            )


            for bar, value in zip(
                bars,
                balanced_df[
                    "Balanced Accuracy"
                ]
            ):

                ax.text(
                    value + 0.01,
                    bar.get_y()
                    + bar.get_height()
                    / 2,
                    f"{value:.3f}",
                    va="center"
                )


            plt.tight_layout()

            st.pyplot(fig)

            plt.close(fig)


            # ------------------------------------------------
            # CV versus Test
            # ------------------------------------------------

            st.subheader(
                "Cross-validation vs Test Macro F1"
            )


            comparison_df = (
                results_df[
                    [
                        "Model",
                        "CV Macro F1",
                        "Macro F1"
                    ]
                ]
                .set_index(
                    "Model"
                )
            )


            st.bar_chart(
                comparison_df
            )


            # ------------------------------------------------
            # Confusion Matrices
            # ------------------------------------------------

            st.subheader(
                "Confusion Matrices"
            )


            for model_name, data in (
                trained_models.items()
            ):

                st.markdown(
                    f"### {model_name}"
                )


                cm = (
                    data[
                        "confusion_matrix"
                    ]
                )


                classes = (
                    data[
                        "classes"
                    ]
                )


                fig, ax = plt.subplots(
                    figsize=(6, 5)
                )


                sns.heatmap(
                    cm,
                    annot=True,
                    fmt="d",
                    cmap="Blues",
                    xticklabels=
                        classes,
                    yticklabels=
                        classes,
                    ax=ax
                )


                ax.set_xlabel(
                    "Predicted Class"
                )


                ax.set_ylabel(
                    "Actual Class"
                )


                ax.set_title(
                    f"{model_name} "
                    f"Confusion Matrix"
                )


                plt.tight_layout()

                st.pyplot(fig)

                plt.close(fig)


            # ------------------------------------------------
            # Classification Reports
            # ------------------------------------------------

            st.subheader(
                "Classification Reports"
            )


            for model_name, data in (
                trained_models.items()
            ):

                with st.expander(
                    f"{model_name} "
                    f"Classification Report"
                ):

                    report_df = (
                        pd.DataFrame(
                            data[
                                "classification_report"
                            ]
                        )
                        .transpose()
                    )


                    st.dataframe(
                        report_df.round(4),
                        use_container_width=True
                    )


            # ------------------------------------------------
            # Optuna Parameters
            # ------------------------------------------------

            parameter_models = [
                model_name
                for model_name, data
                in trained_models.items()
                if data["best_params"]
            ]


            if parameter_models:

                st.subheader(
                    "Optuna Best Parameters"
                )


                for model_name in (
                    parameter_models
                ):

                    with st.expander(
                        model_name
                    ):

                        st.json(
                            trained_models[
                                model_name
                            ][
                                "best_params"
                            ]
                        )


            # ------------------------------------------------
            # Download
            # ------------------------------------------------

            st.subheader(
                "📥 Download Results"
            )


            st.download_button(
                label=
                    "Download Model Results CSV",

                data=
                    results_df.to_csv(
                        index=False
                    ),

                file_name=
                    f"{target_name}_"
                    f"model_results.csv",

                mime=
                    "text/csv"
            )


# ============================================================
# WELCOME PAGE
# ============================================================

else:

    st.markdown(
        """
        ## Welcome! 👋

        This application analyses and predicts
        **impulsive buying behaviour using
        machine learning**.

        ### Features

        📊 Dataset overview

        🔍 Exploratory Data Analysis

        🔗 Cramér's V correlation

        🤖 Multiple machine-learning algorithms

        ⚖️ Random oversampling

        🔄 Stratified cross-validation

        ⚙️ Optuna hyperparameter optimisation

        📈 Model comparison

        🧩 Confusion matrices

        📋 Classification reports

        ### Target Variables

        The application supports:

        **IB1**

        **IB2**

        **IB3**

        When predicting an IB variable,
        IB1, IB2 and IB3 are excluded
        from the predictor variables to
        prevent target leakage.

        Upload your CSV file from the
        sidebar to start the analysis.
        """
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")


st.markdown(
    """
    <p style="
        text-align:center;
        color:gray;
    ">
    Impulsive Buying Behaviour Analysis |
    MRes Artificial Intelligence |
    University of Wolverhampton
    </p>
    """,
    unsafe_allow_html=True
)
