"""
=========================================================================================
🔥 UNIVERSAL FLUID BUSINESS INTELLIGENCE SUITE WITH AUTOMATED PREDICTOR CHATBOX 🔥
=========================================================================================
Architectural Style: Streamlit Plotly-Native Architecture (Zero Matplotlib Dependency)
Features:
- 100% Dynamic UI: Headings, labels, and briefs adapt to any uploaded dataset.
- Target Discovery Engine: Auto-detects column contexts (Attrition, Churn, Status).
- Instant Pre-Training: Models train in the background immediately upon file load.
- Interactive Live Chat Predictor: On-the-fly risk profiling and segment analytics.
"""

import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc

# ---------------------------------------------------------------------------------------
# I. GLOBAL THEME DESIGN & HIGH-CONTRAST LIGHT LAYOUT INITIALIZATION
# ---------------------------------------------------------------------------------------
st.set_page_config(
    page_title="Universal Business Intelligence & Super-Learning Suite",
    layout="wide",
    page_icon="⚖️",
    initial_sidebar_state="expanded"
)

# Custom premium styling matrix forcing high-contrast dark text visibility on light grids
st.markdown("""
    <style>
    html, body, [data-testid="stAppViewContainer"] {
        background-color: #F8FAFC !important;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #FFFFFF !important;
        border: 1px solid #E2E8F0 !important;
        border-radius: 6px 6px 0px 0px;
        padding: 10px 20px !important;
        font-weight: bold !important;
    }
    .dynamic-header {
        background: linear-gradient(135deg, #0F172A 0%, #1E3A8A 100%);
        padding: 35px;
        border-radius: 12px;
        color: #FFFFFF !important;
        text-align: center;
        margin-bottom: 25px;
        box-shadow: 0 4px 12px rgba(30, 58, 138, 0.15);
    }
    .dynamic-header h1, .dynamic-header p { color: #FFFFFF !important; }
    .kpi-card {
        background-color: #FFFFFF !important;
        padding: 22px;
        border-radius: 12px;
        border-top: 6px solid #2563EB;
        box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05);
        text-align: center;
    }
    .kpi-card h2, .kpi-card p, .kpi-card span { color: #0F172A !important; }
    .dynamic-insight-card {
        background-color: #FFFFFF !important;
        border-left: 6px solid #10B981;
        padding: 20px;
        border-radius: 8px;
        margin-top: 15px;
        margin-bottom: 15px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.02);
    }
    .dynamic-insight-card h4, .dynamic-insight-card p, .dynamic-insight-card li, .dynamic-insight-card b { 
        color: #1E293B !important; 
    }
    .anomaly-alert-card {
        background-color: #FFFFFF !important;
        border-left: 6px solid #EF4444;
        padding: 22px;
        border-radius: 8px;
        margin-top: 15px;
        margin-bottom: 20px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.02);
    }
    .anomaly-alert-card h4, .anomaly-alert-card p, .anomaly-alert-card li, .anomaly-alert-card b { 
        color: #7F1D1D !important; 
    }
    h1, h2, h3, h4, h5, h6, p, label, span, div {
        color: #0F172A !important;
    }
    </style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------------------
# II. INTELLIGENT DATA SCHEMA DETECTION ENGINE
# ---------------------------------------------------------------------------------------
def auto_discover_schema(dataframe):
    schema = {
        'target_col': None,
        'age_col': None,
        'income_col': None,
        'group_col': None,
        'numeric_features': [],
        'categorical_features': []
    }
    cols = dataframe.columns.tolist()
    
    # 1. Discover target classification dimension (e.g. Attrition, Status, Churn)
    for c in cols:
        c_upper = str(c).upper().strip()
        if any(keyword in c_upper for keyword in ['STATUS', 'ATTRITION', 'CHURN', 'TARGET', 'OUTPUT', 'DEFAULT']):
            schema['target_col'] = c
            break
    if not schema['target_col']:
        schema['target_col'] = cols[-1]
        
    # 2. Assign core evaluation metrics axes
    for c in cols:
        if c == schema['target_col']: continue
        c_upper = str(c).upper().strip()
        
        if any(k in c_upper for k in ['AGE', 'BIRTH']):
            schema['age_col'] = c
        elif any(k in c_upper for k in ['INCOME', 'SALARY', 'EARNING', 'RATE', 'MONTHLYINCOME', 'DAILYRATE', 'HOURLYRATE']):
            schema['income_col'] = c
        elif any(k in c_upper for k in ['ZONE', 'DEPARTMENT', 'TEAM', 'BRANCH', 'OFFICE', 'LOCATION', 'STATE', 'JOBROLE']):
            schema['group_col'] = c
            
    # 3. Segregate layout arrays
    for c in cols:
        if c == schema['target_col']: continue
        if pd.api.types.is_numeric_dtype(dataframe[c]):
            schema['numeric_features'].append(c)
        else:
            schema['categorical_features'].append(c)
            
    if not schema['age_col'] and schema['numeric_features']: schema['age_col'] = schema['numeric_features'][0]
    if not schema['income_col'] and len(schema['numeric_features']) > 1: schema['income_col'] = schema['numeric_features'][1]
    if not schema['group_col'] and schema['categorical_features']: schema['group_col'] = schema['categorical_features'][0]
    
    return schema

# ---------------------------------------------------------------------------------------
# III. SMART DATA CLEANSE PIPELINE
# ---------------------------------------------------------------------------------------
@st.cache_data
def universal_cleanse_pipeline(file_source):
    if isinstance(file_source, str):
        try:
            raw_df = pd.read_csv(file_source)
        except FileNotFoundError:
            return None, None
    else:
        raw_df = pd.read_csv(file_source)
        
    df_clean = raw_df.copy()
    
    # Clean only numerical string vectors containing formatting marks (commas, currency signs)
    for col in df_clean.columns:
        if df_clean[col].dtype == 'object':
            sample_stripped = df_clean[col].astype(str).str.replace(r'[,\s\$₹]', '', regex=True)
            if sample_stripped.str.replace('.', '', 1).str.isnumeric().sum() > 0.8 * len(df_clean) and len(df_clean) > 0:
                df_clean[col] = pd.to_numeric(sample_stripped, errors='coerce')
                
    # Normalize string cases for uniform groupings
    for col in df_clean.select_dtypes(include=['object', 'category']).columns:
        df_clean[col] = df_clean[col].fillna('UNKNOWN').astype(str).str.strip().str.upper()
        
    schema = auto_discover_schema(df_clean)
    return df_clean, schema

# ---------------------------------------------------------------------------------------
# IV. CORE FILE RETRIEVAL SYSTEM & RUNTIME VARIABLE SETUP
# ---------------------------------------------------------------------------------------
st.sidebar.markdown("### 📁 Dataset Configuration Panel")
uploaded_file = st.sidebar.file_uploader("Upload Core Business File Ledger (CSV)", type=["csv"])

if uploaded_file is not None:
    df, schema = universal_cleanse_pipeline(uploaded_file)
else:
    df, schema = universal_cleanse_pipeline("EA.csv")
    if df is None:
        df, schema = universal_cleanse_pipeline("Insurance (1).csv")

if df is None:
    st.error("❌ Base template file registries missing! Please upload a data CSV via the sidebar container.")
    st.stop()

target_var = schema['target_col']
age_var = schema['age_col']
income_var = schema['income_col']
group_var = schema['group_col']

unique_target_classes = df[target_var].unique().tolist()
risk_class_counts = df[target_var].value_counts()
risk_anchor_target = risk_class_counts.index[-1] if len(risk_class_counts) > 1 else unique_target_classes[0]
safe_anchor_target = risk_class_counts.index[0] if len(risk_class_counts) > 1 else unique_target_classes[0]

# ---------------------------------------------------------------------------------------
# V. INSTANT BACKGROUND MACHINE LEARNING ENGINE (TRAINS AUTOMATICALLY OUT-OF-THE-BOX)
# ---------------------------------------------------------------------------------------
if 'dynamic_ml_memory' not in st.session_state or st.session_state.get('loaded_file') != uploaded_file:
    ml_df = df.copy()
    if age_var and age_var in ml_df.columns:
        ml_df['DYNAMIC_QUANTILE_BINS'] = pd.qcut(ml_df[age_var], q=min(4, ml_df[age_var].nunique()), duplicates='drop').astype(str)
    
    le_tgt = LabelEncoder()
    ml_df[target_var] = le_tgt.fit_transform(ml_df[target_var].astype(str))
    
    X_ml = ml_df.drop(columns=[target_var])
    if 'DYNAMIC_QUANTILE_BINS' in X_ml.columns: X_ml = X_ml.drop(columns=['DYNAMIC_QUANTILE_BINS'])
    y_ml = ml_df[target_var]
    
    num_cols = X_ml.select_dtypes(include=['int64', 'float64']).columns.tolist()
    cat_cols = X_ml.select_dtypes(include=['object', 'category']).columns.tolist()
    
    num_pipe = Pipeline(steps=[('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())])
    cat_pipe = Pipeline(steps=[('imputer', SimpleImputer(strategy='constant', fill_value='UNKNOWN')), ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])
    
    transformer_block = ColumnTransformer(transformers=[('num', num_pipe, num_cols), ('cat', cat_pipe, cat_cols)])
    
    X_train, X_test, y_train, y_test = train_test_split(X_ml, y_ml, test_size=0.30, random_state=42, stratify=y_ml)
    
    X_train_proc = transformer_block.fit_transform(X_train)
    X_test_proc = transformer_block.transform(X_test)
    
    classifiers = {
        'KNN Classifier': KNeighborsClassifier(n_neighbors=5, weights='distance'),
        'Decision Tree Model': DecisionTreeClassifier(random_state=42, max_depth=6, class_weight='balanced'),
        'Random Forest Ensemble': RandomForestClassifier(random_state=42, n_estimators=100, max_depth=8, class_weight='balanced'),
        'Gradient Boosting Machine': GradientBoostingClassifier(random_state=42, n_estimators=100, learning_rate=0.1, max_depth=4)
    }
    
    metrics_store = []
    roc_store = {}
    cm_store = {}
    
    target_risk_index = le_tgt.transform([str(risk_anchor_target).upper()])[0] if str(risk_anchor_target).upper() in le_tgt.classes_ else 1
    
    best_f1 = -1
    champion_name = None
    champion_model_instance = None
    
    for name, clf in classifiers.items():
        clf.fit(X_train_proc, y_train)
        preds = clf.predict(X_test_proc)
        train_preds = clf.predict(X_train_proc)
        probs = clf.predict_proba(X_test_proc)[:, 1] if hasattr(clf, "predict_proba") else preds
        
        f1_val = f1_score(y_test, preds, pos_label=target_risk_index, zero_division=0)
        if f1_val > best_f1:
            best_f1 = f1_val
            champion_name = name
            champion_model_instance = clf
        
        metrics_store.append({
            'Model Configuration': name,
            'Train Accuracy': accuracy_score(y_train, train_preds),
            'Test Accuracy': accuracy_score(y_test, preds),
            'Precision': precision_score(y_test, preds, pos_label=target_risk_index, zero_division=0),
            'Recall': recall_score(y_test, preds, pos_label=target_risk_index, zero_division=0),
            'F1-Score': f1_val
        })
        
        fpr, tpr, _ = roc_curve(y_test, probs, pos_label=target_risk_index)
        roc_store[name] = (fpr, tpr, auc(fpr, tpr))
        cm_store[name] = confusion_matrix(y_test, preds).tolist()
        
    st.session_state.dynamic_ml_memory = {
        'table': pd.DataFrame(metrics_store), 'roc': roc_store, 'cm': cm_store, 'labels': le_tgt.classes_.tolist()
    }
    st.session_state.model_pipeline_instance = transformer_block
    st.session_state.trained_champion_model = champion_model_instance
    st.session_state.ml_features_columns = X_ml.columns.tolist()
    st.session_state.target_label_encoder_instance = le_tgt
    st.session_state.loaded_file = uploaded_file

# ---------------------------------------------------------------------------------------
# VI. VOCABULARY ENGINE & MAIN HEADINGS MUTATION
# ---------------------------------------------------------------------------------------
if "ATTRITION" in str(target_var).upper():
    title_text = "Employee Attrition Risk & Retention Audit Suite"
    kpi_risk_title = f"ACTIVE ATTRITION COUNT ({risk_anchor_target})"
    kpi_safe_title = f"RETAINED STAFF COUNT ({safe_anchor_target})"
elif "STATUS" in str(target_var).upper() or "CLAIM" in str(target_var).upper():
    title_text = "Insurance Claim Settlement & Disparity Adjudication Suite"
    kpi_risk_title = f"REPUDIATED CLAIMS COUNT ({risk_anchor_target})"
    kpi_safe_title = f"APPROVED CLAIMS COUNT ({safe_anchor_target})"
else:
    title_text = f"Enterprise {str(target_var).replace('_', ' ').title()} Optimization Suite"
    kpi_risk_title = f"RISK OUTCOME INSTANCES ({risk_anchor_target})"
    kpi_safe_title = f"STABLE STATE INSTANCES ({safe_anchor_target})"

st.markdown(f'<div class="dynamic-header"><h1>⚖️ {title_text}</h1><p>Data-agnostic analytical pipeline running real-time automated tracking benchmarks on: <b>{target_var}</b></p></div>', unsafe_allow_html=True)

# ---------------------------------------------------------------------------------------
# VII. HIGH-CONTRAST FINANCIAL & OPERATIONAL KPI MATRICES
# ---------------------------------------------------------------------------------------
total_records = len(df)
total_risk_cases = len(df[df[target_var] == risk_anchor_target])
base_risk_rate = (total_risk_cases / total_records) * 100 if total_records > 0 else 0

kpi_grid1, kpi_grid2, kpi_grid3, kpi_grid4 = st.columns(4)
with kpi_grid1:
    st.markdown(f'<div class="kpi-card"><p style="color:#64748B;font-size:12px;margin:0;font-weight:700;">TOTAL DATASETS RECORDED</p><h2 style="color:#0F172A;margin:5px 0;">{total_records:,}</h2></div>', unsafe_allow_html=True)
with kpi_grid2:
    st.markdown(f'<div class="kpi-card" style="border-top-color:#EF4444;"><p style="color:#64748B;font-size:12px;margin:0;font-weight:700;">{kpi_risk_title}</p><h2 style="color:#EF4444;margin:5px 0;">{total_risk_cases:,}</h2><span style="color:#EF4444;font-size:12px;font-weight:bold;">{base_risk_rate:.2f}% Realization Rate</span></div>', unsafe_allow_html=True)
with kpi_grid3:
    st.markdown(f'<div class="kpi-card" style="border-top-color:#10B981;"><p style="color:#64748B;font-size:12px;margin:0;font-weight:700;">{kpi_safe_title}</p><h2 style="color:#10B981;margin:5px 0;">{total_records - total_risk_cases:,}</h2><span style="font-size:12px;color:#10B981;font-weight:bold;">{100-base_risk_rate:.2f}% Concentration</span></div>', unsafe_allow_html=True)
with kpi_grid4:
    unique_operational_segments = df[group_var].nunique() if group_var else 1
    st.markdown(f'<div class="kpi-card" style="border-top-color:#F59E0B;"><p style="color:#64748B;font-size:12px;margin:0;font-weight:700;">DISTINCT SUB-OPERATIONAL NODES</p><h2 style="color:#F59E0B;margin:5px 0;">{unique_operational_segments} Segments</h2><span style="font-size:12px;color:#64748B;">Categorized under {group_var}</span></div>', unsafe_allow_html=True)

# Define Core Architecture Slices Tabs
tab_descriptive, tab_diagnostic, tab_modeling, tab_findings = st.tabs([
    "📋 Descriptive Slices", "🔍 Disparity Diagnostics", "🤖 Super-Learning Classifiers", "💡 Dynamic Executive Summary"
])

# =======================================================================================
# TAB 1: DESCRIPTIVE CONTINGENCY SLICES
# =======================================================================================
with tab_descriptive:
    st.markdown(f"### 📋 Categorical Contingency Matrix vs `{target_var}`")
    selected_slice = st.selectbox("Choose Any Nominal Dimension to Project Proportional Distribution Splits:", schema['categorical_features'] if schema['categorical_features'] else df.columns.tolist())
    
    if selected_slice:
        count_ct = pd.crosstab(df[selected_slice], df[target_var])
        pct_ct = pd.crosstab(df[selected_slice], df[target_var], normalize='index') * 100
        
        split1, split2 = st.columns(2)
        with split1:
            st.markdown("#### Absolute Entry Count Distributions")
            st.dataframe(count_ct, use_container_width=True)
        with split2:
            st.markdown("#### Proportional Densities Across Rows (%)")
            st.dataframe(pct_ct.style.format("{:.2f}%"), use_container_width=True)
            
        fig_proportions = px.bar(df, x=selected_slice, color=target_var, barmode='group',
                                 color_discrete_sequence=px.colors.qualitative.Safe,
                                 title=f"Volumetric Sub-Category Metric Distribution Splitting: {selected_slice}")
        fig_proportions.update_layout(xaxis={'categoryorder': 'total descending'}, template="plotly_white")
        st.plotly_chart(fig_proportions, use_container_width=True)

# =======================================================================================
# TAB 2: DISPARITY DIAGNOSTICS (NATIVE PLOTLY CORE)
# =======================================================================================
with tab_diagnostic:
    st.markdown("### 🔍 Systemic Group Performance Disparity Probes")
    probe_selection = st.radio("Isolate Target Evaluation Attribute Array Layer:", ["Continuous Attribute Profile A", "Continuous Attribute Profile B", "Categorical Group Segments"], horizontal=True)
    
    if probe_selection == "Continuous Attribute Profile A" and age_var:
        st.subheader(f"Auditing Disparity Profile Layout: `{age_var}`")
        col_a1, col_a2 = st.columns(2)
        with col_a1:
            fig_box = px.box(df, x=target_var, y=age_var, color=target_var, points="outliers", color_discrete_sequence=['#0284C7', '#EF4444'])
            fig_box.update_layout(template="plotly_white")
            st.plotly_chart(fig_box, use_container_width=True)
        with col_a2:
            df['DYNAMIC_QUANTILE_BINS'] = pd.qcut(df[age_var], q=min(4, df[age_var].nunique()), duplicates='drop').astype(str)
            q_ct = pd.crosstab(df['DYNAMIC_QUANTILE_BINS'], df[target_var], normalize='index') * 100
            fig_q_bar = px.bar(q_ct.reset_index(), x='DYNAMIC_QUANTILE_BINS', y=q_ct.columns.tolist(), barmode='stack', color_discrete_sequence=['#0284C7', '#EF4444'])
            fig_q_bar.update_layout(template="plotly_white")
            st.plotly_chart(fig_q_bar, use_container_width=True)
            
    elif probe_selection == "Continuous Attribute Profile B" and income_var:
        st.subheader(f"Auditing Disparity Profile Layout: `{income_var}`")
        col_i1, col_i2 = st.columns(2)
        with col_i1:
            fig_inc_box = px.box(df, x=target_var, y=income_var, color=target_var, points="outliers", color_discrete_sequence=['#10B981', '#EF4444'])
            fig_inc_box.update_layout(template="plotly_white")
            st.plotly_chart(fig_inc_box, use_container_width=True)
        with col_i2:
            mean_financials = df.groupby(target_var)[income_var].mean().reset_index()
            fig_mean_inc = px.bar(mean_financials, x=target_var, y=income_var, color=target_var, color_discrete_sequence=['#10B981', '#EF4444'])
            fig_mean_inc.update_layout(template="plotly_white")
            st.plotly_chart(fig_mean_inc, use_container_width=True)
            
    elif probe_selection == "Categorical Group Segments" and group_var:
        st.subheader(f"Auditing Disparity Profile Layout: `{group_var}`")
        group_ct_pct = pd.crosstab(df[group_var], df[target_var], normalize='index') * 100
        if risk_anchor_target in group_ct_pct.columns:
            group_ct_pct_sorted = group_ct_pct.sort_values(by=risk_anchor_target, ascending=False)
        else:
            group_ct_pct_sorted = group_ct_pct
        
        fig_grp = px.bar(group_ct_pct_sorted.reset_index(), x=group_var, y=group_ct_pct_sorted.columns.tolist(), barmode='stack', color_discrete_sequence=['#0284C7', '#EF4444'])
        fig_grp.update_layout(template="plotly_white")
        st.plotly_chart(fig_grp, use_container_width=True)

# =======================================================================================
# TAB 3: AUTOMATED ML PERFORMANCE MATRIX MATRIX
# =======================================================================================
with tab_modeling:
    st.markdown("### 🤖 Parallel Super-Learning Classifiers Optimization Matrix")
    
    cache = st.session_state.dynamic_ml_memory
    st.dataframe(cache['table'].style.format({
        'Train Accuracy': "{:.2%}", 'Test Accuracy': "{:.2%}", 'Precision': "{:.2%}", 'Recall': "{:.2%}", 'F1-Score': "{:.2%}"
    }))
    
    fig_metrics = px.bar(cache['table'].melt(id_vars='Model Configuration'), x='Model Configuration', y='value', color='variable', barmode='group', template='plotly_white')
    st.plotly_chart(fig_metrics, use_container_width=True)
    
    cm_col1, cm_col2 = st.columns(2)
    with cm_col1:
        st.markdown("#### 🔄 Receiver Operating Characteristic (ROC) Trajectories")
        fig_roc = go.Figure()
        for name, (fpr, tpr, area) in cache['roc'].items():
            fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f"{name} (AUC = {area:.3f})"))
        fig_roc.update_layout(template='plotly_white', xaxis_title="False Positive Rate", yaxis_title="True Positive Rate")
        st.plotly_chart(fig_roc, use_container_width=True)
        
    with cm_col2:
        st.markdown("#### 🧩 Confusion Matrices Slices Grid")
        selected_matrix_model = st.selectbox("Select Model Configuration to View Confusion Cells Layer:", list(cache['cm'].keys()))
        if selected_matrix_model:
            matrix_array = np.array(cache['cm'][selected_matrix_model])
            fig_plotly_matrix = px.imshow(matrix_array, text_auto=True, x=cache['labels'], y=cache['labels'], color_continuous_scale='Blues')
            fig_plotly_matrix.update_layout(template='plotly_white', height=380)
            st.plotly_chart(fig_plotly_matrix, use_container_width=True)

# =======================================================================================
# TAB 4: EXECUTIVE SUMMARY FINDINGS (100% COMPLEX DATA-DRIVEN TEXT GENERATION)
# =======================================================================================
with tab_findings:
    st.markdown('<h2 style="color:#0F172A;">💡 Automated Strategic Briefing & Action Register</h2>', unsafe_allow_html=True)
    
    if age_var and age_var in df.columns:
        df['DYNAMIC_QUANTILE_BINS'] = pd.qcut(df[age_var], q=min(4, df[age_var].nunique()), duplicates='drop').astype(str)
        
    live_age_tab = pd.crosstab(df['DYNAMIC_QUANTILE_BINS'], df[target_var], normalize='index') * 100 if 'DYNAMIC_QUANTILE_BINS' in df.columns else pd.DataFrame()
    live_group_tab = pd.crosstab(df[group_var], df[target_var], normalize='index') * 100 if group_var else pd.DataFrame()
    
    st.markdown('<div class="anomaly-alert-card">', unsafe_allow_html=True)
    st.markdown('<h4>📌 Discovered Operational Sub-Group Disparities</h4>', unsafe_allow_html=True)
    
    if not live_group_tab.empty and risk_anchor_target in live_group_tab.columns:
        worst_grp = live_group_tab.sort_values(by=risk_anchor_target, ascending=False).index[0]
        worst_grp_val = live_group_tab.sort_values(by=risk_anchor_target, ascending=False)[risk_anchor_target].iloc[0]
        st.markdown(f"<li><b>Segment Disparity Variance Alert:</b> Structural processing indicates non-uniform behavior traits across tracking zones. The node segment matching <b>{worst_grp}</b> holds the maximum risk signature, tracking at a <b>{worst_grp_val:.2f}% validation density for the critical {risk_anchor_target} event lane</b>. This variance highlights a clear opportunity for processing standardization.</li>", unsafe_allow_html=True)
        
    if income_var:
        m_risk = df[df[target_var] == risk_anchor_target][income_var].mean()
        m_stable = df[df[target_var] == safe_anchor_target][income_var].mean() if len(df[df[target_var] == safe_anchor_target]) > 0 else 1
        st.markdown(f"<li><b>Continuous Metric Factor Scale Imbalance:</b> Accounts logging onto the <b>{risk_anchor_target}</b> risk path carry an average value score for metric `{income_var}` of <b>{m_risk:,.2f}</b>, compared to stable configurations tracking an average baseline parameters profile score of <b>{m_stable:,.2f}</b>.</li>", unsafe_allow_html=True)
        
    if not live_age_tab.empty and risk_anchor_target in live_age_tab.columns:
        worst_age = live_age_tab.sort_values(by=risk_anchor_target, ascending=False).index[0]
        worst_age_val = live_age_tab.sort_values(by=risk_anchor_target, ascending=False)[risk_anchor_target].iloc[0]
        st.markdown(f"<li><b>Continuous Bracket Disparities:</b> Maximum frequency concentrations matching the critical risk target state are localized inside the <b>{worst_age}</b> data quantile boundary layer, registering an active frequency baseline parameter score of <b>{worst_age_val:.2f}%</b>.</li>", unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<h3 style="color:#0F172A;">🛠️ Strategic Action Roadmap</h3>', unsafe_allow_html=True)
    st.markdown(f"""
    * **Enforce Centralized Process Guidelines:** Review and align validation standards across all components under `{group_var}` to eliminate localized variance and ensure equitable case resolutions.
    * **Deploy Continuous Exception Screeners:** Build automated secondary tracking loops for entries falling within high-risk ranges of `{income_var}` to protect against underlying systemic screening bias.
    * **Integrate Algorithmic Validation Audits:** Use the pre-trained **{champion_name}** model as an automated validation checkpoint to flag high-probability risk classifications for secondary human review.
    """)

# =======================================================================================
# VIII. INTEGRATED REAL-TIME BUSINESS PREDICTION CHATBOX
# =======================================================================================
st.sidebar.markdown("---")
st.sidebar.markdown("### 💬 Predictive Model Assistant Chatbox")
user_query = st.sidebar.text_input("Ask Assistant (e.g., 'Who will leave?', 'Check high risk profiles'):")

if user_query:
    clf = st.session_state.trained_champion_model
    proc_pipe = st.session_state.model_pipeline_instance
    feat_cols = st.session_state.ml_features_columns
    le_tgt = st.session_state.target_label_encoder_instance
    
    X_active = df[feat_cols]
    X_active_proc = proc_pipe.transform(X_active)
    
    raw_preds = clf.predict(X_active_proc)
    raw_probs = clf.predict_proba(X_active_proc)[:, 1] if hasattr(clf, "predict_proba") else raw_preds
    
    target_risk_idx_val = le_tgt.transform([str(risk_anchor_target).upper()])[0] if str(risk_anchor_target).upper() in le_tgt.classes_ else 1
    risk_indices = np.where(raw_preds == target_risk_idx_val)[0]
    
    st.sidebar.success(f"🎯 Query Processed via {type(clf).__name__} Engine!")
    st.sidebar.markdown(f"**Identified High-Risk System Entities:** `{len(risk_indices)} instances` found inside active data matrices arrays.")
    
    if len(risk_indices) > 0:
        high_risk_subset_df = df.iloc[risk_indices].copy()
        high_risk_subset_df['RISK_SCORE_%'] = raw_probs[risk_indices] * 100
        high_risk_subset_df = high_risk_subset_df.sort_values(by='RISK_SCORE_%', ascending=False)
        
        st.sidebar.markdown(f"**Top High-Risk Accounts List (`{risk_anchor_target}` Track):**")
        id_key = [c for c in ['POLICY_NO', 'EmployeeNumber', 'EMPLOYEENUMBER'] if c in df.columns]
        display_cols = id_key + [group_var, age_var] if id_key else [group_var, age_var]
        st.sidebar.dataframe(high_risk_subset_df[display_cols + ['RISK_SCORE_%']].head(10).style.format({'RISK_SCORE_%': "{:.1f}%"}))
        
        # Real-time visual projections map directly to the active analytics dashboard view layer
        st.markdown("---")
        st.markdown(f"### 📊 Chat Assistant Real-Time Analysis Grid: Isolated High-Risk Sub-Segment (`{risk_anchor_target}` Profile)")
        
        c_chat1, c_chat2 = st.columns(2)
        with c_chat1:
            fig_chat_bar = px.histogram(high_risk_subset_df, x=group_var, title=f"High-Risk Structural Density Count Slices by {group_var}", color_discrete_sequence=['#EF4444'], template='plotly_white')
            st.plotly_chart(fig_chat_bar, use_container_width=True)
        with c_chat2:
            fig_chat_scatter = px.scatter(high_risk_subset_df, x=age_var, y=income_var, size='RISK_SCORE_%', hover_data=display_cols, title=f"Risk Tracking Positioning Grid: {age_var} vs {income_var}", color_discrete_sequence=['#EF4444'], template='plotly_white')
            st.plotly_chart(fig_chat_scatter, use_container_width=True)
else:
    st.sidebar.info("💡 Type an investigation query above (e.g. 'Who will leave?' or 'Identify risk factors') to instantly isolate and graph high-probability data targets.")
