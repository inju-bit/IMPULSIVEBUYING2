"""
Impulsive Buying Behavior Analysis
Streamlit Web Application
By Inju Khadka - MRes Artificial Intelligence
University of Wolverhampton
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency
import warnings

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix
)
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier
)
from sklearn.cross_decomposition import PLSRegression
from imblearn.over_sampling import RandomOverSampler
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
import optuna

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

# Page config
st.set_page_config(
    page_title="Impulsive Buying Analysis - Inju Khadka",
    page_icon="🛒",
    layout="wide"
)

# Custom CSS
st.markdown("""
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
    .metric-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<p class="main-title">🛒 Impulsive Buying Behavior Analysis</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">| MRes Artificial Intelligence Research| Inju Khadka | University of Wolverhampton</p>', unsafe_allow_html=True)

# Session state
if 'df' not in st.session_state:
    st.session_state.df = None
if 'ml_results' not in st.session_state:
    st.session_state.ml_results = None
if 'trained_models' not in st.session_state:
    st.session_state.trained_models = {}

# Sidebar
with st.sidebar:
    st.title("🎯 PREDICTION")
    st.markdown("---")
    
    uploaded_file = st.file_uploader("📂 Upload your CSV data", type=['csv'])
    
    if uploaded_file:
        st.session_state.df = pd.read_csv(uploaded_file)
        st.success(f"✅ Loaded {len(st.session_state.df)} rows")
    
    st.markdown("---")
    st.markdown("### About This App")
    st.info("""
    This app analyzes impulsive buying behavior using:
    - 📊 Exploratory Data Analysis
    - 🔗 Cramér's V Correlation 
    - 🤖 Machine Learning Models
    - 📐 PLS-SEM Analysis
    """)
    
    st.markdown("---")
    st.markdown("**Developer:** Inju Khadka")
    st.markdown("**Student ID:** 2513941")
    st.markdown("**Supervisor:** Dr. Tahir Mahmood")
    st.markdown("**Submission Date:** 7 January 2026")
    st.markdown("**Program:** MRes Artificial Intelligence")
    st.markdown("**Module Code:** 7CS113")
    st.markdown("**University:** Wolverhampton")

# Main content
if st.session_state.df is not None:
    df = st.session_state.df.copy()
    
    # Create tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 Data Overview",
        "🔍 EDA",
        "🔗 Correlations",
        "🤖 ML Models",
        "📈 Results",
        "📐 PLS-SEM"
    ])
    
    # ============ TAB 1: DATA OVERVIEW ============
    with tab1:
        st.header("📊 Data Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Rows", df.shape[0])
        col2.metric("Columns", df.shape[1])
        col3.metric("Missing", df.isna().sum().sum())
        col4.metric("Memory", f"{df.memory_usage(deep=True).sum()/1024:.1f} KB")
        
        st.subheader("Data Preview")
        st.dataframe(df.head(10), use_container_width=True)
        
        st.subheader("Column Summary")
        col_info = pd.DataFrame({
            'Column': df.columns,
            'Type': df.dtypes.values,
            'Unique': df.nunique().values,
            'Missing': df.isna().sum().values
        })
        st.dataframe(col_info, use_container_width=True)
    
    # ============ TAB 2: EDA ============
    with tab2:
        st.header("🔍 Exploratory Data Analysis")
        
        # Variable types
        st.subheader("Variable Classification")
        
        var_info = []
        for col in df.columns:
            n_unique = df[col].nunique()
            if df[col].dtype == 'object':
                vtype = "Categorical"
            elif n_unique <= 10:
                vtype = "Categorical (Ordinal)"
            else:
                vtype = "Numeric"
            var_info.append({"Column": col, "Type": vtype, "Unique Values": n_unique})
        
        st.dataframe(pd.DataFrame(var_info), use_container_width=True)
        
        # Target distributions
        st.subheader("Target Variable Distributions (IB1, IB2, IB3)")
        
        targets = ['IB1', 'IB2', 'IB3']
        available = [t for t in targets if t in df.columns]
        
        if available:
            fig, axes = plt.subplots(1, len(available), figsize=(5*len(available), 4))
            if len(available) == 1:
                axes = [axes]
            
            colors = ['#3498db', '#9b59b6', '#1abc9c']
            for i, col in enumerate(available):
                counts = df[col].value_counts().sort_index()
                axes[i].bar(counts.index.astype(str), counts.values, color=colors[i])
                axes[i].set_title(f'{col} Distribution')
                axes[i].set_xlabel('Response')
                axes[i].set_ylabel('Count')
                for j, v in enumerate(counts.values):
                    axes[i].text(j, v + 1, str(v), ha='center')
            
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
        
        # Feature distributions
        st.subheader("Feature Distributions")
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if numeric_cols:
            selected = st.multiselect("Select features:", numeric_cols, default=numeric_cols[:6])
            
            if selected:
                n_cols = min(3, len(selected))
                n_rows = (len(selected) + n_cols - 1) // n_cols
                
                fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
                axes = np.array(axes).flatten()
                
                for i, col in enumerate(selected):
                    axes[i].hist(df[col].dropna(), bins=15, color='steelblue', edgecolor='white')
                    axes[i].set_title(col)
                    axes[i].set_xlabel('Value')
                    axes[i].set_ylabel('Frequency')
                
                for i in range(len(selected), len(axes)):
                    axes[i].set_visible(False)
                
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
    
    # ============ TAB 3: CORRELATIONS ============
    with tab3:
        st.header("🔗 Correlation Analysis")
        
        # Cramér's V
        st.subheader("Cramér's V (Categorical Correlation)")
        
        def cramers_v(x, y):
            table = pd.crosstab(x, y)
            chi2 = chi2_contingency(table, correction=False)[0]
            n = table.sum().sum()
            phi2 = chi2 / n
            r, k = table.shape
            phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))
            rcorr = r - ((r-1)**2)/(n-1)
            kcorr = k - ((k-1)**2)/(n-1)
            denom = min((kcorr-1), (rcorr-1))
            return np.sqrt(phi2corr / denom) if denom > 0 else 0
        
        cat_cols = [c for c in df.columns if df[c].nunique() <= 10]
        
        if len(cat_cols) > 1:
            with st.spinner("Computing Cramér's V matrix..."):
                cat_df = df[cat_cols].dropna()
                matrix = pd.DataFrame(np.zeros((len(cat_cols), len(cat_cols))),
                                     index=cat_cols, columns=cat_cols)
                
                for i in range(len(cat_cols)):
                    for j in range(i, len(cat_cols)):
                        v = cramers_v(cat_df[cat_cols[i]], cat_df[cat_cols[j]])
                        matrix.iloc[i, j] = matrix.iloc[j, i] = v
                
                fig, ax = plt.subplots(figsize=(12, 10))
                sns.heatmap(matrix, annot=True, cmap="YlOrRd", fmt=".2f", ax=ax)
                ax.set_title("Cramér's V Correlation Matrix", fontsize=14)
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
        
    
    
    # ============ TAB 4: ML MODELS ============
    with tab4:
        st.header("🤖 Machine Learning Models")
        
        targets = ['IB1', 'IB2', 'IB3']
        available = [t for t in targets if t in df.columns]
        
        if not available:
            st.warning("No target variables (IB1, IB2, IB3) found!")
        else:
            col1, col2 = st.columns(2)
            
            with col1:
                target = st.selectbox("Target Variable:", available)
            
            with col2:
                models = st.multiselect(
                    "Select Models:",
                    ["Random Forest", "Extra Trees", "Gradient Boosting",
                     "XGBoost", "LightGBM", "CatBoost"],
                    default=["Random Forest", "XGBoost"]
                )
            
            col3, col4 = st.columns(2)
            with col3:
                test_size = st.slider("Test Size:", 0.1, 0.4, 0.2, 0.05)
            with col4:
                use_optuna = st.checkbox("Optuna Tuning (slower)", value=False)
            
            oversample = st.checkbox("Use Oversampling", value=True)
            
            if st.button("🚀 Train Models", type="primary"):
                with st.spinner("Training... Please wait"):
                    
                    # Prepare data
                    df_ml = df.copy()
                    for col in df_ml.columns:
                        if df_ml[col].dtype == 'object':
                            df_ml[col] = LabelEncoder().fit_transform(df_ml[col].astype(str))
                    
                    if df_ml[target].min() >= 1:
                        df_ml[target] = df_ml[target].astype(int) - 1
                    
                    X = df_ml.drop(columns=[target])
                    y = df_ml[target]
                    
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=test_size, stratify=y, random_state=42
                    )
                    
                    if oversample:
                        ros = RandomOverSampler(random_state=42)
                        X_train, y_train = ros.fit_resample(X_train, y_train)
                    
                    # Model configurations
                    model_configs = {
                        "Random Forest": (RandomForestClassifier, 
                            {"n_estimators": 100, "max_depth": 8, "random_state": 42, "n_jobs": -1}),
                        "Extra Trees": (ExtraTreesClassifier,
                            {"n_estimators": 100, "max_depth": 8, "random_state": 42, "n_jobs": -1}),
                        "Gradient Boosting": (GradientBoostingClassifier,
                            {"n_estimators": 50, "max_depth": 4, "random_state": 42}),
                        "XGBoost": (xgb.XGBClassifier,
                            {"n_estimators": 100, "max_depth": 5, "eval_metric": "mlogloss",
                             "verbosity": 0, "random_state": 42, "n_jobs": -1}),
                        "LightGBM": (lgb.LGBMClassifier,
                            {"n_estimators": 100, "max_depth": 8, "verbose": -1,
                             "random_state": 42, "n_jobs": -1}),
                        "CatBoost": (CatBoostClassifier,
                            {"iterations": 100, "depth": 5, "verbose": False, "random_state": 42})
                    }
                    
                    results = []
                    trained = {}
                    
                    progress = st.progress(0)
                    
                    for i, name in enumerate(models):
                        st.write(f"Training {name}...")
                        
                        cls, params = model_configs[name]
                        
                        if use_optuna and name in ["Random Forest", "XGBoost", "LightGBM"]:
                            def objective(trial):
                                if name == "Random Forest":
                                    p = {"n_estimators": trial.suggest_int("n_estimators", 50, 300),
                                         "max_depth": trial.suggest_int("max_depth", 3, 12),
                                         "random_state": 42, "n_jobs": -1}
                                    m = RandomForestClassifier(**p)
                                elif name == "XGBoost":
                                    p = {"n_estimators": trial.suggest_int("n_estimators", 50, 300),
                                         "max_depth": trial.suggest_int("max_depth", 3, 10),
                                         "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
                                         "eval_metric": "mlogloss", "verbosity": 0,
                                         "random_state": 42, "n_jobs": -1}
                                    m = xgb.XGBClassifier(**p)
                                else:
                                    p = {"n_estimators": trial.suggest_int("n_estimators", 50, 300),
                                         "max_depth": trial.suggest_int("max_depth", 3, 12),
                                         "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
                                         "verbose": -1, "random_state": 42, "n_jobs": -1}
                                    m = lgb.LGBMClassifier(**p)
                                m.fit(X_train, y_train)
                                return f1_score(y_test, m.predict(X_test), average='weighted')
                            
                            study = optuna.create_study(direction="maximize")
                            study.optimize(objective, n_trials=5, show_progress_bar=False)
                            params.update(study.best_params)
                        
                        model = cls(**params)
                        model.fit(X_train, y_train)
                        preds = model.predict(X_test)
                        
                        results.append({
                            "Model": name,
                            "Accuracy": accuracy_score(y_test, preds),
                            "Precision": precision_score(y_test, preds, average='weighted', zero_division=0),
                            "Recall": recall_score(y_test, preds, average='weighted', zero_division=0),
                            "F1": f1_score(y_test, preds, average='weighted', zero_division=0)
                        })
                        
                        trained[name] = {"model": model, "preds": preds, "y_test": y_test}
                        progress.progress((i + 1) / len(models))
                    
                    st.session_state.ml_results = pd.DataFrame(results)
                    st.session_state.trained_models = trained
                    
                    st.success("✅ Training Complete!")
                    st.dataframe(
                        st.session_state.ml_results.style.highlight_max(
                            subset=['Accuracy', 'F1'], color='lightgreen'
                        ).format({
                            'Accuracy': '{:.4f}', 'Precision': '{:.4f}',
                            'Recall': '{:.4f}', 'F1': '{:.4f}'
                        }),
                        use_container_width=True
                    )
    
    # ============ TAB 5: RESULTS ============
    with tab5:
        st.header("📈 Results Visualization")
        
        if st.session_state.ml_results is not None:
            results_df = st.session_state.ml_results
            
            # Performance comparison
            st.subheader("Model Performance Comparison")
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig, ax = plt.subplots(figsize=(8, 5))
                colors = plt.cm.Paired(np.linspace(0, 1, len(results_df)))
                bars = ax.barh(results_df['Model'], results_df['Accuracy'], color=colors)
                ax.set_xlabel('Accuracy')
                ax.set_title('Model Accuracy')
                ax.set_xlim(0, 1)
                for bar, val in zip(bars, results_df['Accuracy']):
                    ax.text(val + 0.01, bar.get_y() + bar.get_height()/2,
                           f'{val:.3f}', va='center')
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
            
            with col2:
                fig, ax = plt.subplots(figsize=(8, 5))
                bars = ax.barh(results_df['Model'], results_df['F1'], color=colors)
                ax.set_xlabel('F1 Score')
                ax.set_title('Model F1 Score')
                ax.set_xlim(0, 1)
                for bar, val in zip(bars, results_df['F1']):
                    ax.text(val + 0.01, bar.get_y() + bar.get_height()/2,
                           f'{val:.3f}', va='center')
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
            
            # Confusion matrices
            st.subheader("Confusion Matrices")
            
            models = st.session_state.trained_models
            n = len(models)
            
            if n > 0:
                cols = min(3, n)
                rows = (n + cols - 1) // cols
                
                fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
                axes = np.array(axes).flatten()
                
                for i, (name, data) in enumerate(models.items()):
                    cm = confusion_matrix(data['y_test'], data['preds'])
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i])
                    axes[i].set_title(name)
                    axes[i].set_xlabel('Predicted')
                    axes[i].set_ylabel('Actual')
                
                for i in range(n, len(axes)):
                    axes[i].set_visible(False)
                
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
            
            # Loss chart
            st.subheader("F1 Loss Analysis (1 - F1)")
            results_df['Loss'] = 1 - results_df['F1']
            
            fig, ax = plt.subplots(figsize=(10, 5))
            sorted_df = results_df.sort_values('Loss')
            colors = plt.cm.Reds(np.linspace(0.3, 0.8, len(sorted_df)))
            bars = ax.bar(sorted_df['Model'], sorted_df['Loss'], color=colors)
            ax.set_ylabel('Loss (1 - F1)')
            ax.set_title('F1 Loss by Model (Lower = Better)')
            ax.set_ylim(0, 1)
            plt.xticks(rotation=45, ha='right')
            for bar, val in zip(bars, sorted_df['Loss']):
                ax.text(bar.get_x() + bar.get_width()/2, val + 0.02,
                       f'{val:.3f}', ha='center')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
            
            # Download
            st.subheader("📥 Download Results")
            st.download_button(
                "Download as CSV",
                results_df.to_csv(index=False),
                "ml_results.csv",
                "text/csv"
            )
        else:
            st.info("👆 Train models first in the ML Models tab!")
    
    # ============ TAB 6: PLS-SEM ============
    with tab6:
        st.header("📐 PLS-SEM Analysis")
        
        st.markdown("""
        **Partial Least Squares Structural Equation Modeling (PLS-SEM)** is used to test 
        theoretical relationships between latent constructs in behavioral research.
        """)
        
        # Block definitions (based on research model)
        blocks = {
            "PE": ["PE1", "PE2", "PE3", "PE4"],      # Physical Environment
            "SE": ["SE1", "SE2", "SE3"],              # Social Environment
            "TP": ["TP1", "TP2", "TP3"],              # Time Perspective
            "HB": ["HB1", "HB2", "HB3"],              # Hedonic Browsing
            "UB": ["UB1", "UB2", "UB3", "UB4", "UB5"] # Utilitarian Browsing
        }
        
        all_items = [item for items in blocks.values() for item in items]
        targets = ['IB1', 'IB2', 'IB3']
        available = [t for t in targets if t in df.columns]
        missing = [c for c in all_items if c not in df.columns]
        
        if missing:
            st.warning(f"Missing columns: {missing}")
        elif not available:
            st.warning("No target variables found!")
        else:
            st.subheader("Model Structure")
            st.code("""
    Latent Variables              Path Model
    ───────────────────────────────────────────────────
    PE1-PE4  →  PE (Physical Environment)    ──┐
                                               │
    SE1-SE3  →  SE (Social Environment)      ──┤
                                               │
    TP1-TP3  →  TP (Time Perspective)        ──┼──→  IB (Impulse Buying)
                                               │
    HB1-HB3  →  HB (Hedonic Browsing)        ──┤
                                               │
    UB1-UB5  →  UB (Utilitarian Browsing)    ──┘
            """)
            
            col1, col2 = st.columns(2)
            with col1:
                n_comp = st.slider("PLS Components:", 1, 5, 2)
            with col2:
                test_pls = st.slider("Test Size:", 0.1, 0.4, 0.2, key="pls")
            
            if st.button("🔬 Run PLS-SEM", type="primary"):
                with st.spinner("Running analysis..."):
                    
                    df_pls = df.copy()
                    for t in available:
                        if df_pls[t].min() >= 1:
                            df_pls[t] = df_pls[t].astype(int) - 1
                    
                    def compute_lv(data, block_items):
                        lv = pd.DataFrame()
                        for block, items in block_items.items():
                            X = data[items].values
                            w = np.ones(X.shape[1]) / X.shape[1]
                            scores = X @ w
                            scores = (scores - scores.mean()) / (scores.std() + 1e-8)
                            lv[block] = scores
                        return lv
                    
                    train_df, test_df = train_test_split(df_pls, test_size=test_pls, random_state=42)
                    
                    LV_train = compute_lv(train_df, blocks)
                    LV_test = compute_lv(test_df, blocks)
                    
                    block_names = list(blocks.keys())
                    X_train = LV_train[block_names].values
                    X_test = LV_test[block_names].values
                    Y_train = train_df[available].values
                    Y_test = test_df[available].values
                    
                    pls = PLSRegression(n_components=n_comp)
                    pls.fit(X_train, Y_train)
                    Y_pred = pls.predict(X_test)
                    Y_pred_class = np.clip(np.round(Y_pred).astype(int), 0, 4)
                    
                    # Metrics
                    metrics = []
                    for i, t in enumerate(available):
                        y_true = Y_test[:, i]
                        y_pred = Y_pred_class[:, i]
                        metrics.append({
                            "Target": t,
                            "Accuracy": accuracy_score(y_true, y_pred),
                            "Precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
                            "Recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
                            "F1": f1_score(y_true, y_pred, average="macro", zero_division=0)
                        })
                    
                    metrics_df = pd.DataFrame(metrics)
                    metrics_df['F1_Loss'] = 1 - metrics_df['F1']
                    
                    st.success("✅ PLS-SEM Complete!")
                    
                    # Results table
                    st.subheader("Metrics Table")
                    st.dataframe(
                        metrics_df.style.highlight_max(subset=['Accuracy', 'F1'], color='lightgreen')
                        .format({'Accuracy': '{:.4f}', 'Precision': '{:.4f}',
                                'Recall': '{:.4f}', 'F1': '{:.4f}', 'F1_Loss': '{:.4f}'}),
                        use_container_width=True
                    )
                    
                    # Path coefficients
                    st.subheader("Path Coefficients")
                    # pls.coef_ shape is (n_targets, n_features), need to transpose
                    coef_df = pd.DataFrame(pls.coef_.T, index=block_names, columns=available)
                    
                    fig, ax = plt.subplots(figsize=(10, 6))
                    sns.heatmap(coef_df, annot=True, cmap="RdYlGn", center=0, fmt=".3f", ax=ax)
                    ax.set_title("Path Coefficients: Latent Variables → Targets")
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()
                    
                    st.markdown("""
                    **Interpretation:**
                    - 🟢 Green = Positive influence on buying intention
                    - 🔴 Red = Negative influence on buying intention
                    - Higher absolute value = Stronger effect
                    """)
                    
                    # F1 Loss chart
                    st.subheader("F1 Loss by Target")
                    fig, ax = plt.subplots(figsize=(8, 5))
                    colors = ['#3498db', '#9b59b6', '#1abc9c']
                    bars = ax.bar(metrics_df['Target'], metrics_df['F1_Loss'], color=colors[:len(available)])
                    ax.set_ylabel('F1 Loss')
                    ax.set_title('PLS-SEM F1 Loss (Lower = Better)')
                    ax.set_ylim(0, 1)
                    for bar, val in zip(bars, metrics_df['F1_Loss']):
                        ax.text(bar.get_x() + bar.get_width()/2, val + 0.02, f'{val:.3f}', ha='center')
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()
                    
                    # Confusion matrices
                    st.subheader("Confusion Matrices")
                    fig, axes = plt.subplots(1, len(available), figsize=(5*len(available), 4))
                    if len(available) == 1:
                        axes = [axes]
                    
                    for i, t in enumerate(available):
                        cm = confusion_matrix(Y_test[:, i], Y_pred_class[:, i], labels=[0,1,2,3,4])
                        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                                   xticklabels=range(5), yticklabels=range(5), ax=axes[i])
                        axes[i].set_title(f'{t}')
                        axes[i].set_xlabel('Predicted')
                        axes[i].set_ylabel('Actual')
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()
                    
                    # Download
                    st.download_button(
                        "Download PLS-SEM Results",
                        metrics_df.to_csv(index=False),
                        "plssem_results.csv",
                        "text/csv"
                    )

else:
    # Welcome page
    st.markdown("""
    ## Welcome! 👋
    
    This is my MRes research project predicting and analyzing **impulsive buying behavior** using machine learning
    and structural equation modeling- PLS-SEM.
    
    ### What this app does:
    
    - **📊 Data Overview** - Check your dataset stats and preview
    - **🔍 EDA** - Explore variable distributions and patterns
    - **🔗 Correlations** - Cramér's V 
    - **🤖 ML Models** - Train and compare 6 different classifiers
    - **📈 Results** - Visualize model performance
    - **📐 PLS-SEM** - Test theoretical relationships between constructs
    
    ### How to use:
    
    1. Upload your CSV file using the sidebar
    2. Make sure it has columns: PE1-4, SE1-3, TP1-3, HB1-3, UB1-5, IB1-3
    3. Go through each tab to analyze your data
    
    ### Research Variables:
    
    | Code | Meaning |
    |------|---------|
    | PE | Physical Environment |
    | SE | Social Environment |
    | TP | Time Perspective |
    | HB | Hedonic Browsing |
    | UB | Utilitarian Browsing |
    | IB | Impulse Buying (Target) |
    
    ---
    **🧪 Try with Sample Data!**
    """)
    
    if st.button("Sample Data"):
        np.random.seed(42)
        n = 200
        sample = {
            'PE1': np.random.randint(1, 6, n), 'PE2': np.random.randint(1, 6, n),
            'PE3': np.random.randint(1, 6, n), 'PE4': np.random.randint(1, 6, n),
            'SE1': np.random.randint(1, 6, n), 'SE2': np.random.randint(1, 6, n),
            'SE3': np.random.randint(1, 6, n),
            'TP1': np.random.randint(1, 6, n), 'TP2': np.random.randint(1, 6, n),
            'TP3': np.random.randint(1, 6, n),
            'HB1': np.random.randint(1, 6, n), 'HB2': np.random.randint(1, 6, n),
            'HB3': np.random.randint(1, 6, n),
            'UB1': np.random.randint(1, 6, n), 'UB2': np.random.randint(1, 6, n),
            'UB3': np.random.randint(1, 6, n), 'UB4': np.random.randint(1, 6, n),
            'UB5': np.random.randint(1, 6, n),
            'IB1': np.random.randint(1, 6, n), 'IB2': np.random.randint(1, 6, n),
            'IB3': np.random.randint(1, 6, n),
        }
        st.session_state.df = pd.DataFrame(sample)
        st.rerun()

# Footer
st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:gray;'>Made by Inju Khadka | MRes Artificial Intelligence | University of Wolverhampton | 2025</p>",
    unsafe_allow_html=True
)
