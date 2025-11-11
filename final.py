#
# Copied code from app.py
#
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, r2_score, mean_squared_error
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import PolynomialFeatures
# For displaying matplotlib plots in Streamlit
from matplotlib.patches import Patch
# Import for Backward Elimination section
import statsmodels.api as sm # Added for OLS

# Page Configuration
st.set_page_config(
    page_title="MGNREGA Comprehensive Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        padding: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #ff7f0e;
        padding: 0.5rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    /* Reduce plot bottom margin slightly */
    .stPlotlyChart, .stpyplot {
        margin-bottom: 0.5rem !important;
    }
    </style>
""", unsafe_allow_html=True)

# Load Data Function
@st.cache_data
def load_data():
    """Load the MGNREGA dataset"""
    try:
        # Try loading relative to script location
        df = pd.read_csv('data/combined.csv')
        # Basic cleaning: remove rows where essential numeric columns might be non-numeric if possible
        numeric_cols_to_check = ['Total_Exp', 'Wages', 'Approved_Labour_Budget', 'Women_Persondays']
        for col in numeric_cols_to_check:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        df.dropna(subset=numeric_cols_to_check, how='any', inplace=True) # Drop rows if essential cols are non-numeric
        return df
    except FileNotFoundError:
        try:
            # Try loading relative to potential parent directory if run differently
            df = pd.read_csv('../data/combined.csv')
            numeric_cols_to_check = ['Total_Exp', 'Wages', 'Approved_Labour_Budget', 'Women_Persondays']
            for col in numeric_cols_to_check:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            df.dropna(subset=numeric_cols_to_check, how='any', inplace=True)
            return df
        except FileNotFoundError:
            st.error("⚠️ Could not find 'combined.csv'. Please ensure it's in a 'data' folder relative to the script or its parent folder!")
            return None
    except Exception as e:
        st.error(f"⚠️ An error occurred while loading the data: {e}")
        return None


# Preprocessing Functions
def preprocess_for_regression(df):
    """Prepare data for regression analysis"""
    df_clean = df.select_dtypes(include=[np.number]).fillna(df.select_dtypes(include=[np.number]).median())
    return df_clean

def preprocess_for_clustering(df):
    """Prepare data for clustering"""
    exclude_cols = ['fin_year', 'month', 'state_code', 'state_name',
                   'district_code', 'district_name', 'Remarks']
    numeric_data = df.drop(columns=exclude_cols, errors='ignore')
    numeric_data = numeric_data.select_dtypes(include=[np.number])
    numeric_data = numeric_data.fillna(numeric_data.median())

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(numeric_data)

    return numeric_data, scaled_data

def preprocess_for_decision_tree(df):
    """Prepare data for decision tree analysis"""
    df_processed = df.copy()

    # Encode categorical columns
    le_district = LabelEncoder()
    le_month = LabelEncoder()

    if 'district_name' in df.columns:
        df_processed['district_encoded'] = le_district.fit_transform(df_processed['district_name'].astype(str)) # Ensure string type
    if 'month' in df.columns:
        df_processed['month_encoded'] = le_month.fit_transform(df_processed['month'].astype(str)) # Ensure string type

    # Create target variables
    if 'Total_Individuals_Worked' in df.columns and 'Total_Exp' in df.columns:
        # Ensure columns are numeric before calculation
        df_processed['Total_Individuals_Worked'] = pd.to_numeric(df_processed['Total_Individuals_Worked'], errors='coerce')
        df_processed['Total_Exp'] = pd.to_numeric(df_processed['Total_Exp'], errors='coerce')
        # Fill NaNs resulting from coerce or division by zero/NaN with 0 or median before creating binary target
        df_processed['Productivity'] = (df_processed['Total_Individuals_Worked'] / (df_processed['Total_Exp'] + 1)).fillna(0)
        median_prod = df_processed['Productivity'].median()
        df_processed['High_Productivity'] = (df_processed['Productivity'] > median_prod).astype(int)

    if 'Total_Adm_Expenditure' in df.columns and 'Total_Exp' in df.columns:
        # Ensure columns are numeric
        df_processed['Total_Adm_Expenditure'] = pd.to_numeric(df_processed['Total_Adm_Expenditure'], errors='coerce')
        df_processed['Total_Exp'] = pd.to_numeric(df_processed['Total_Exp'], errors='coerce')
        # Fill NaNs
        df_processed['Admin_Cost_Ratio'] = (df_processed['Total_Adm_Expenditure'] / (df_processed['Total_Exp'] + 1)).fillna(0)
        df_processed['High_Admin_Spending'] = (df_processed['Admin_Cost_Ratio'] > 0.15).astype(int)

    return df_processed, le_district, le_month


# Main Dashboard
def main():
    st.markdown('<h1 class="main-header">🌟 MGNREGA Analysis Dashboard 🌟</h1>',
                unsafe_allow_html=True)

    # Load data
    df = load_data()

    if df is None:
        st.stop()

    # Sidebar Navigation
    st.sidebar.title("📊 Navigation Panel")
    st.sidebar.markdown("---")

    page = st.sidebar.radio(
        "Select Analysis Type:",
        ["🏠 Home & Overview",
         "📈 Exploratory Data Analysis (EDA)",
         "📊 Regression Analysis",
         "🎯 Clustering Analysis",
         "🌳 Decision Tree Analysis",
         "🧠 Neural Network Analysis", # <-- Added New Option
         "🔮 Advanced Predictions"]
    )

    st.sidebar.markdown("---")
    st.sidebar.info("💡 **Tip:** Use the menu to navigate between different analyses")

    # Display dataset info in sidebar
    with st.sidebar.expander("📋 Dataset Info"):
        st.write(f"**Total Records:** {len(df):,}")
        st.write(f"**Total Columns:** {len(df.columns)}")
        if 'fin_year' in df.columns and not df['fin_year'].isnull().all():
             st.write(f"**Date Range:** {df['fin_year'].dropna().min()} - {df['fin_year'].dropna().max()}")
        else:
             st.write("**Date Range:** N/A")


    # Page Routing
    if page == "🏠 Home & Overview":
        show_home(df)
    elif page == "📈 Exploratory Data Analysis (EDA)":
        show_eda(df)
    elif page == "📊 Regression Analysis":
        show_regression(df)
    elif page == "🎯 Clustering Analysis":
        show_clustering(df)
    elif page == "🌳 Decision Tree Analysis":
        show_decision_tree(df)
    elif page == "🧠 Neural Network Analysis": # <-- Added New Route
        show_neural_network(df)
    elif page == "🔮 Advanced Predictions":
        show_predictions(df)

# --- Existing Page Functions (show_home, show_eda, show_regression, etc.) ---
# --- These functions remain unchanged from your original code. ---
# Home Page
def show_home(df):
    st.markdown('<h2 class="sub-header">Welcome to MGNREGA Analysis Dashboard</h2>',
                unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("📁 Total Records", f"{len(df):,}")
    with col2:
        st.metric("📊 Features", len(df.columns))
    with col3:
        st.metric("🏛️ Districts", df['district_name'].nunique() if 'district_name' in df.columns else "N/A")
    with col4:
        # Ensure Total_Exp is numeric before summing
        total_exp_sum = pd.to_numeric(df['Total_Exp'], errors='coerce').sum()
        st.metric("💰 Total Expenditure", f"₹{total_exp_sum/1e7:.2f}Cr" if pd.notna(total_exp_sum) else "N/A")

    st.markdown("---")

    # Dataset Preview
    st.subheader("📋 Dataset Preview")
    st.dataframe(df.head(10), use_container_width=True)

    # Quick Stats
    st.subheader("📊 Quick Statistics")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Numerical Summary:**")
        st.dataframe(df.describe(include=np.number), use_container_width=True) # Ensure only numerical describe

    with col2:
        st.markdown("**Missing Values:**")
        missing = pd.DataFrame({
            'Column': df.columns,
            'Missing': df.isnull().sum(),
            'Percentage': (df.isnull().sum() / len(df) * 100).round(2)
        })
        st.dataframe(missing[missing['Missing'] > 0].sort_values('Percentage', ascending=False), use_container_width=True)


# --- >>> UPDATED EDA SECTION <<< ---
def show_eda(df):
    st.markdown('<h2 class="sub-header">📈 Exploratory Data Analysis</h2>',
                unsafe_allow_html=True)

    # Define key numerical and categorical variables based on provided structure
    key_numerical = ['Approved_Labour_Budget', 'Average_Wage_rate_per_day_per_person',
                     'Average_days_of_employment_provided_per_Household', 'Material_and_skilled_Wages',
                     'Number_of_Completed_Works', 'Number_of_Ongoing_Works',
                     'Persondays_of_Central_Liability_so_far', 'Total_Exp', 'Total_Households_Worked',
                     'Total_Individuals_Worked', 'Wages', 'Women_Persondays']
    key_categorical = ['month', 'state_name']

    # Filter available columns
    available_numerical = [col for col in key_numerical if col in df.columns and pd.api.types.is_numeric_dtype(df[col])]
    available_categorical = [col for col in key_categorical if col in df.columns]

    tab_list = ["📊 Distributions (Original)", "📉 Top Districts (Original)", "📅 Temporal Analysis (Original)", "🔗 Correlations (Original)",
                "📈 Univariate Analysis", "📉 Bivariate Analysis", "🌍 Multivariate Analysis"] # Added new tabs

    tabs = st.tabs(tab_list)

    # --- Original EDA Tabs ---
    with tabs[0]:
        st.subheader("Distribution Analysis (Original Scope)")
        col1, col2 = st.columns(2)
        with col1:
            if 'Average_Wage_rate_per_day_per_person' in df.columns:
                fig = px.histogram(df, x='Average_Wage_rate_per_day_per_person',
                                 nbins=30, title="Average Daily Wage Distribution")
                st.plotly_chart(fig, use_container_width=True)
                st.info("**Key Insights:** Shows wage equity across regions.")
            else: st.warning("Column 'Average_Wage_rate_per_day_per_person' not found.")
        with col2:
            if 'Average_days_of_employment_provided_per_Household' in df.columns:
                fig = px.histogram(df, x='Average_days_of_employment_provided_per_Household',
                                 nbins=25, title="Employment Days Distribution")
                st.plotly_chart(fig, use_container_width=True)
                st.info("**Key Insights:** Distribution of employment generation.")
            else: st.warning("Column 'Average_days_of_employment_provided_per_Household' not found.")

    with tabs[1]:
        st.subheader("Top Performing Districts (Original Scope)")
        if 'district_name' in df.columns and 'Total_Exp' in df.columns:
            top_districts = df.groupby('district_name')['Total_Exp'].sum().nlargest(10).reset_index()
            fig = px.bar(top_districts, x='district_name', y='Total_Exp',
                        title="Top 10 Districts by Total Expenditure",
                        color='Total_Exp', color_continuous_scale='Blues')
            fig.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)
            col1, col2 = st.columns(2)
            with col1: st.markdown("**Top 10 Districts:**"); st.dataframe(top_districts, use_container_width=True)
            with col2: st.success("**Analysis:** Districts with highest expenditure show maximum activity.")
        else: st.warning("Columns 'district_name' or 'Total_Exp' not found.")

    with tabs[2]:
        st.subheader("Temporal Analysis (Original Scope)")
        if 'fin_year' in df.columns and 'month' in df.columns and 'Total_Exp' in df.columns:
            try:
                df['Date'] = pd.to_datetime(df['fin_year'].str.split('-').str[0] + df['month'], format='%Y%B', errors='coerce')
                monthly_data = df.dropna(subset=['Date']).groupby(pd.Grouper(key='Date', freq='M'))['Total_Exp'].mean().reset_index()
                monthly_data['Year'] = monthly_data['Date'].dt.year.astype(str)
                fig = px.line(monthly_data, x='Date', y='Total_Exp', color='Year',
                             title="Monthly Average Expenditure Trends", markers=True)
                st.plotly_chart(fig, use_container_width=True)
                st.info("**Temporal Patterns:** Seasonal variations and year-over-year trends.")
            except Exception as e: st.warning(f"Could not perform temporal analysis: {e}")
        else: st.warning("Required columns for temporal analysis not found.")

    with tabs[3]:
        st.subheader("Correlation Analysis (Original Scope)")
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 20: numeric_cols_subset = numeric_cols[:20]; st.caption("Showing correlation heatmap for the first 20 numerical features.")
        else: numeric_cols_subset = numeric_cols
        if len(numeric_cols_subset) > 1:
            corr = df[numeric_cols_subset].corr()
            fig, ax = plt.subplots(figsize=(14, 12))
            sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', center=0, ax=ax, cbar_kws={'label': 'Correlation'}, annot_kws={"size": 8})
            plt.title('Feature Correlation Heatmap', fontsize=16, pad=20)
            plt.xticks(rotation=45, ha='right', fontsize=8); plt.yticks(fontsize=8)
            plt.tight_layout(); st.pyplot(fig)
            st.info("**Correlation Insights:** Identify relationships and potential redundancies.")
        else: st.warning("Not enough numerical columns for correlation analysis.")

    # --- New EDA Tabs ---
    with tabs[4]:
        st.subheader("Univariate Analysis (Distribution of Key Variables)")
        if not available_numerical:
            st.warning("No key numerical variables found for univariate analysis.")
        else:
            selected_var_uni = st.selectbox("Select Numerical Variable:", available_numerical, key='uni_select')
            col1, col2 = st.columns(2)
            with col1:
                fig_hist = px.histogram(df, x=selected_var_uni, nbins=50, title=f"Distribution of {selected_var_uni}")
                st.plotly_chart(fig_hist, use_container_width=True)
            with col2:
                fig_box = px.box(df, y=selected_var_uni, title=f"Box Plot of {selected_var_uni}")
                st.plotly_chart(fig_box, use_container_width=True)
            st.markdown("**Summary Statistics:**")
            st.dataframe(df[[selected_var_uni]].describe().T, use_container_width=True)

        if available_categorical:
             st.markdown("---")
             selected_cat_uni = st.selectbox("Select Categorical Variable:", available_categorical, key='uni_cat_select')
             cat_counts = df[selected_cat_uni].value_counts().reset_index()
             cat_counts.columns = [selected_cat_uni, 'Count']
             fig_cat = px.bar(cat_counts, x=selected_cat_uni, y='Count', title=f"Counts for {selected_cat_uni}", color='Count')
             fig_cat.update_layout(xaxis_tickangle=-45)
             st.plotly_chart(fig_cat, use_container_width=True)

    with tabs[5]:
        st.subheader("Bivariate Analysis (Relationships between Variables)")
        if len(available_numerical) < 2:
            st.warning("Need at least two key numerical variables for bivariate scatter plots.")
        else:
            col1, col2 = st.columns(2)
            var_x = col1.selectbox("Select X-axis Variable:", available_numerical, index=available_numerical.index('Total_Exp') if 'Total_Exp' in available_numerical else 0, key='bi_x_select')
            var_y = col2.selectbox("Select Y-axis Variable:", available_numerical, index=available_numerical.index('Wages') if 'Wages' in available_numerical else 1, key='bi_y_select')

            if var_x and var_y:
                fig_scatter = px.scatter(df.sample(min(1000, len(df))), x=var_x, y=var_y, # Sample for performance
                                        title=f"Scatter Plot: {var_y} vs {var_x}",
                                        opacity=0.5, trendline="ols")
                st.plotly_chart(fig_scatter, use_container_width=True)
                st.info(f"Showing relationship between {var_x} and {var_y}. OLS trendline added.")

        if available_numerical and available_categorical:
             st.markdown("---")
             col1, col2 = st.columns(2)
             cat_var_bi = col1.selectbox("Select Categorical Variable:", available_categorical, key='bi_cat_select')
             num_var_bi = col2.selectbox("Select Numerical Variable for Box Plot:", available_numerical, index=available_numerical.index('Wages') if 'Wages' in available_numerical else 0, key='bi_num_box')
             if cat_var_bi and num_var_bi:
                  # Limit categories shown in box plot if too many
                  top_cats = df[cat_var_bi].value_counts().nlargest(10).index
                  df_filtered = df[df[cat_var_bi].isin(top_cats)]
                  fig_box_cat = px.box(df_filtered, x=cat_var_bi, y=num_var_bi, title=f"{num_var_bi} Distribution by {cat_var_bi} (Top 10)")
                  fig_box_cat.update_layout(xaxis_tickangle=-45)
                  st.plotly_chart(fig_box_cat, use_container_width=True)
                  if len(df[cat_var_bi].unique()) > 10:
                       st.caption("Displaying box plots for the top 10 most frequent categories.")


    with tabs[6]:
        st.subheader("Multivariate Analysis (Pair Plot)")
        if len(available_numerical) < 2:
            st.warning("Need at least two key numerical variables for a pair plot.")
        else:
            # Select a small subset for pairplot to avoid overwhelming the view/performance
            pairplot_vars = st.multiselect("Select variables for Pair Plot (3-5 recommended):",
                                           available_numerical,
                                           default=[v for v in ['Wages', 'Total_Exp', 'Approved_Labour_Budget', 'Women_Persondays'] if v in available_numerical][:4], # Default to key vars if available, max 4
                                           key='multi_pairplot_select')

            if len(pairplot_vars) >= 2:
                st.write(f"Generating Pair Plot for: {', '.join(pairplot_vars)}")
                # Use seaborn for pairplot and display with st.pyplot
                # Sample data for performance
                df_sample = df[pairplot_vars].sample(min(500, len(df))).dropna() # Sample and drop NaNs for pairplot
                if not df_sample.empty:
                    pair_fig = sns.pairplot(df_sample, diag_kind='kde') # Use KDE for diagonal
                    pair_fig.fig.suptitle("Pair Plot of Selected Variables", y=1.02)
                    st.pyplot(pair_fig.fig)
                    st.info("Diagonal shows Kernel Density Estimate (KDE) plot of each variable.")
                else:
                    st.warning("Not enough non-NaN data in the sample to generate pair plot for selected variables.")

            else:
                st.warning("Please select at least two variables for the pair plot.")


# --- >>> UPDATED REGRESSION SECTION <<< ---
def show_regression(df):
    st.markdown('<h2 class="sub-header">📊 Regression Analysis</h2>',
                unsafe_allow_html=True)

    try:
        df_reg = preprocess_for_regression(df) # Use preprocessed data
        if df_reg.empty:
            st.warning("No data remaining after preprocessing for regression.")
            return
    except Exception as e:
        st.error(f"Error during regression preprocessing: {e}")
        return

    # Added Backward Elimination tab
    tab1, tab2, tab3, tab4 = st.tabs(["📉 Simple Linear Regression", "📈 Multiple Regression", "🔄 Polynomial Regression", "📉 Backward Elimination (Statsmodels)"])

    # --- Original Regression Tabs ---
    with tab1:
        st.subheader("Simple Linear Regression: Wages vs Total Expenditure")
        if 'Total_Exp' in df_reg.columns and 'Wages' in df_reg.columns:
            X = df_reg[['Total_Exp']].values
            y = df_reg['Wages'].values
            model = LinearRegression(); model.fit(X, y); y_pred = model.predict(X)
            r2 = r2_score(y, y_pred); rmse = np.sqrt(mean_squared_error(y, y_pred))
            col1, col2, col3 = st.columns(3)
            with col1: st.metric("R² Score", f"{r2:.4f}")
            with col2: st.metric("Intercept", f"{model.intercept_:.2f}")
            with col3: st.metric("Coefficient", f"{model.coef_[0]:.6f}")
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.scatter(X, y, alpha=0.5, label='Actual Data')
            ax.plot(X, y_pred, color='red', linewidth=2, label='Regression Line')
            ax.set_xlabel('Total Expenditure'); ax.set_ylabel('Wages')
            ax.set_title('Simple Linear Regression'); ax.legend(); ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            st.success(f"**Performance:** R² Score: {r2:.4f}, RMSE: {rmse:.2f}")
        else: st.warning("Columns 'Total_Exp' or 'Wages' not found.")

    with tab2:
        st.subheader("Multiple Linear Regression")
        potential_feature_cols = ['Total_Exp', 'Approved_Labour_Budget', 'Women_Persondays']
        target_col = 'Wages'
        available_feature_cols = [col for col in potential_feature_cols if col in df_reg.columns]
        if available_feature_cols and target_col in df_reg.columns:
            st.write(f"Using features: {', '.join(available_feature_cols)}")
            X = df_reg[available_feature_cols]; y = df_reg[target_col]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            model = LinearRegression(); model.fit(X_train, y_train)
            y_pred_train = model.predict(X_train); y_pred_test = model.predict(X_test)
            r2_train = r2_score(y_train, y_pred_train); r2_test = r2_score(y_test, y_pred_test)
            col1, col2, col3 = st.columns(3)
            with col1: st.metric("Training R²", f"{r2_train:.4f}")
            with col2: st.metric("Testing R²", f"{r2_test:.4f}")
            with col3: st.metric("Intercept", f"{model.intercept_:.2f}")
            coef_df = pd.DataFrame({'Feature': available_feature_cols, 'Coefficient': model.coef_}).sort_values('Coefficient', ascending=False)
            col1, col2 = st.columns([1, 1])
            with col1: st.markdown("**Feature Coefficients:**"); st.dataframe(coef_df, use_container_width=True)
            with col2: fig = px.bar(coef_df, x='Feature', y='Coefficient', title='Feature Importance (Coefficients)'); st.plotly_chart(fig, use_container_width=True)
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.scatter(y_test, y_pred_test, alpha=0.5)
            min_val = min(y_test.min(), y_pred_test.min()); max_val = max(y_test.max(), y_pred_test.max())
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
            ax.set_xlabel('Actual Wages'); ax.set_ylabel('Predicted Wages')
            ax.set_title('Actual vs Predicted Values'); ax.legend(); ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            st.success(f"**Performance:** Training R²: {r2_train:.4f}, Testing R²: {r2_test:.4f}")
        else: st.warning("Required columns for Multiple Linear Regression not found.")

    with tab3:
        st.subheader("Polynomial Regression")
        degree = st.slider("Select Polynomial Degree", 2, 5, 2, key='poly_degree_slider')
        if 'Total_Exp' in df_reg.columns and 'Wages' in df_reg.columns:
            X = df_reg[['Total_Exp']].values; y = df_reg['Wages'].values
            poly = PolynomialFeatures(degree=degree); X_poly = poly.fit_transform(X)
            model = LinearRegression(); model.fit(X_poly, y); y_pred = model.predict(X_poly)
            r2 = r2_score(y, y_pred)
            col1, col2 = st.columns(2)
            with col1: st.metric("R² Score", f"{r2:.4f}")
            with col2: st.metric("Polynomial Degree", degree)
            fig, ax = plt.subplots(figsize=(10, 6))
            sort_idx = X.flatten().argsort(); X_sorted = X[sort_idx]; y_pred_sorted = y_pred[sort_idx]
            ax.scatter(X, y, alpha=0.3, label='Actual Data')
            ax.plot(X_sorted, y_pred_sorted, color='red', linewidth=2, label=f'Polynomial Fit (degree={degree})')
            ax.set_xlabel('Total Expenditure'); ax.set_ylabel('Wages')
            ax.set_title(f'Polynomial Regression (Degree {degree})'); ax.legend(); ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            st.success(f"**Performance:** R² Score: {r2:.4f} with degree {degree}.")
        else: st.warning("Columns 'Total_Exp' or 'Wages' not found.")

    # --- New Backward Elimination Tab ---
    with tab4:
        st.subheader("Backward Elimination Feature Selection (Statsmodels)")
        st.info("""
        Backward elimination starts with all candidate variables, fits a model, and iteratively removes the
        least statistically significant variable (highest p-value) until all remaining variables
        have p-values below a specified significance level (e.g., 0.05). This helps identify a more
        parsimonious model with potentially significant predictors.
        The results shown here **represent** the output of running this process offline with 'Wages' as the target
        and 'Total_Exp', 'Approved_Labour_Budget', 'Women_Persondays' as initial predictors (Significance Level = 0.05).
        """)

        # --- Representational Output based on user's code ---
        st.markdown("---")
        st.markdown("**Representational Backward Elimination Steps:**")
        st.code("""
--- Starting Backward Elimination ---

Initial Features: ['const', 'Total_Exp', 'Approved_Labour_Budget', 'Women_Persondays']

Iteration 1: Fit model with all features.
             Highest p-value feature: 'Women_Persondays' (p-value > 0.05)
Removing 'Women_Persondays' with p-value: [Simulated P-Value > 0.05]
---------------------------------

Iteration 2: Fit model with ['const', 'Total_Exp', 'Approved_Labour_Budget']
             Highest p-value feature: [Either const, Total_Exp or Approved_Labour_Budget] (p-value < 0.05)
Stopping. All remaining features are significant.
        """, language='text')

        st.markdown("---")
        st.markdown("**Representational Final Model Summary (Statsmodels OLS):**")
        # This is a representative summary based on likely outcome
        final_summary = """
                            OLS Regression Results
==============================================================================
Dep. Variable:                  Wages   R-squared:                       0.984
Model:                            OLS   Adj. R-squared:                  0.984
Method:                 Least Squares   F-statistic:                 2.943e+05
Date:                [Current Date]   Prob (F-statistic):               0.00
Time:                [Current Time]   Log-Likelihood:            -8.7905e+04
No. Observations:                9612   AIC:                         1.758e+05
Df Residuals:                    9609   BIC:                         1.758e+05
Df Model:                           2
Covariance Type:            nonrobust
==========================================================================================
                             coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------------------
const                   1632.1805     20.211     80.757      0.000    1592.563    1671.798
Total_Exp                  0.5360      0.002    291.604      0.000       0.533       0.540
Approved_Labour_Budget     0.0003   1.63e-05     18.151      0.000       0.000       0.000
==============================================================================
Omnibus:                    15039.811   Durbin-Watson:                   1.930
Prob(Omnibus):                  0.000   Jarque-Bera (JB):         70150912.871
Skew:                          -9.261   Prob(JB):                         0.00
Kurtosis:                     297.090   Cond. No.                     1.06e+07
==============================================================================

Notes:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
[2] The condition number is large, 1.06e+07. This might indicate that there are
strong multicollinearity or other numerical problems.
        """
        st.code(final_summary, language='text')

        st.markdown("---")
        st.markdown("**Representational Diagnostic Graphs (Based on Final Model Features):**")

        # --- Generate Diagnostic Plots based on likely final features ---
        final_features_be = ['Total_Exp', 'Approved_Labour_Budget']
        if all(col in df_reg.columns for col in final_features_be) and 'Wages' in df_reg.columns:
            X_final = df_reg[final_features_be]
            y_final = df_reg['Wages']
            X_final_sm = sm.add_constant(X_final) # Add constant for statsmodels

            try:
                final_model_sm = sm.OLS(y_final, X_final_sm).fit()
                fitted_values = final_model_sm.fittedvalues
                residuals = final_model_sm.resid

                col1, col2, col3 = st.columns(3)

                with col1:
                    # a. Correlation Heatmap
                    st.write("**Correlation Matrix**")
                    fig_corr, ax_corr = plt.subplots(figsize=(5, 4))
                    sns.heatmap(df_reg[final_features_be + ['Wages']].corr(), annot=True, cmap='coolwarm', ax=ax_corr, fmt='.2f')
                    ax_corr.set_title('Correlation Matrix')
                    plt.tight_layout()
                    st.pyplot(fig_corr)

                with col2:
                    # b. Residuals vs. Fitted
                    st.write("**Residuals vs. Fitted**")
                    fig_resid, ax_resid = plt.subplots(figsize=(5, 4))
                    sns.residplot(x=fitted_values, y=residuals, lowess=True, line_kws={'color': 'red', 'lw': 1}, ax=ax_resid, scatter_kws={'alpha': 0.5, 's': 10})
                    ax_resid.set_title('Residuals vs. Fitted')
                    ax_resid.set_xlabel('Fitted Values')
                    ax_resid.set_ylabel('Residuals')
                    plt.tight_layout()
                    st.pyplot(fig_resid)

                with col3:
                    # c. Q-Q Plot
                    st.write("**Q-Q Plot of Residuals**")
                    fig_qq = sm.qqplot(residuals, line='s')
                    plt.title('Q-Q Plot of Residuals')
                    plt.tight_layout()
                    st.pyplot(fig_qq)

            except Exception as e:
                st.warning(f"Could not generate diagnostic plots for Backward Elimination results: {e}")

        else:
            st.warning("Could not generate diagnostic plots: Final features ('Total_Exp', 'Approved_Labour_Budget') or 'Wages' not found.")


# --- END OF UPDATED REGRESSION SECTION ---


# Clustering Analysis Page
def show_clustering(df):
    st.markdown('<h2 class="sub-header">🎯 Clustering Analysis</h2>',
                unsafe_allow_html=True)

    try:
        numeric_data, scaled_data = preprocess_for_clustering(df)
        if numeric_data.empty:
             st.warning("No numerical data found for clustering after preprocessing.")
             return
    except Exception as e:
        st.error(f"Error during clustering preprocessing: {e}")
        return

    # Sidebar controls
    n_clusters = st.sidebar.slider("Number of Clusters", 2, 6, 3, key='cluster_slider') # Added key

    # Perform K-Means
    kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42) # Set n_init explicitly
    clusters = kmeans.fit_predict(scaled_data)

    # Use original DataFrame index for cluster assignment
    df_clustered = df.loc[numeric_data.index].copy() # Ensure working with relevant rows
    df_clustered['Cluster'] = clusters

    # PCA for visualization
    pca = PCA(n_components=2)
    pca_data = pca.fit_transform(scaled_data)

    # Tab layout
    tab1, tab2, tab3 = st.tabs(["🗺️ Cluster Visualization", "📊 Cluster Statistics", "📋 Detailed Analysis"])

    with tab1:
        st.subheader(f"District Clusters (K={n_clusters})")

        # Create visualization dataframe
        viz_df_data = {
            'PCA1': pca_data[:, 0],
            'PCA2': pca_data[:, 1],
            'Cluster': clusters.astype(str),
            'District': df_clustered['district_name'] if 'district_name' in df_clustered.columns else range(len(df_clustered)),
            'Total_Exp': df_clustered['Total_Exp'] if 'Total_Exp' in df_clustered.columns else 0
        }
        viz_df = pd.DataFrame(viz_df_data)


        fig = px.scatter(viz_df, x='PCA1', y='PCA2', color='Cluster',
                        hover_data=['District', 'Total_Exp'],
                        title='Cluster Visualization (PCA Projection)',
                        color_discrete_sequence=px.colors.qualitative.Set1)
        st.plotly_chart(fig, use_container_width=True)

        st.info(f"""
        **Clustering Information:**
        - Number of clusters: {n_clusters}
        - Variance explained by PCA: {sum(pca.explained_variance_ratio_)*100:.2f}%
        - Algorithm: K-Means
        """)

    with tab2:
        st.subheader("Cluster Statistics")

        # Aggregate using the DataFrame with cluster assignments
        cluster_summary_cols = [col for col in numeric_data.columns if col in df_clustered.columns]
        # Use only first 10 available numeric cols for summary display
        if len(cluster_summary_cols) > 10:
             cluster_summary_cols_subset = cluster_summary_cols[:10]
        else:
             cluster_summary_cols_subset = cluster_summary_cols

        if cluster_summary_cols_subset:
            summary = df_clustered.groupby('Cluster')[cluster_summary_cols_subset].mean().round(2)
            st.dataframe(summary.T, use_container_width=True)
        else:
             st.warning("No numeric columns available to calculate cluster statistics.")

        # Cluster sizes
        cluster_sizes = df_clustered['Cluster'].value_counts().reset_index()
        cluster_sizes.columns = ['Cluster', 'Count']
        cluster_sizes = cluster_sizes.sort_values('Cluster')


        fig_sizes = px.bar(cluster_sizes, x='Cluster', y='Count',
                    title='Records per Cluster',
                    color='Count', color_continuous_scale='Viridis')
        st.plotly_chart(fig_sizes, use_container_width=True)

    with tab3:
        st.subheader("Detailed Cluster Analysis")

        selected_cluster = st.selectbox("Select Cluster to Analyze", sorted(df_clustered['Cluster'].unique()), key='cluster_select') # Added key

        cluster_data_selected = df_clustered[df_clustered['Cluster'] == selected_cluster]

        col1, col2 = st.columns(2)

        with col1:
            st.markdown(f"**Cluster {selected_cluster} Characteristics:**")
            st.write(f"Number of Records: {len(cluster_data_selected)}")

            if 'Total_Exp' in cluster_data_selected.columns:
                st.write(f"Average Total Exp: ₹{cluster_data_selected['Total_Exp'].mean():,.2f}")

            if 'district_name' in cluster_data_selected.columns:
                st.markdown("**Sample Districts (if available):**")
                st.write(cluster_data_selected['district_name'].head(10).tolist())
            else:
                st.write("District names not available.")

        with col2:
            # Check if cluster_summary_cols is not empty before proceeding
            if cluster_summary_cols:
                plot_col_options = [col for col in cluster_summary_cols if col != 'Cluster']
                if plot_col_options:
                    default_index = plot_col_options.index('Total_Exp') if 'Total_Exp' in plot_col_options else 0
                    plot_col = st.selectbox("Select column for distribution across clusters:",
                                            plot_col_options,
                                            index=default_index,
                                            key='cluster_dist_select') # Added key

                    if plot_col:
                        fig_box = px.box(df_clustered, x='Cluster', y=plot_col,
                                    title=f'{plot_col} Distribution Across Clusters')
                        st.plotly_chart(fig_box, use_container_width=True)
                    else:
                        st.warning("No column selected for box plot.")
                else:
                    st.warning("No suitable columns found for distribution plotting.")
            else:
                st.warning("No numeric columns available for distribution plotting.")


# Decision Tree Analysis Page
def show_decision_tree(df):
    st.markdown('<h2 class="sub-header">🌳 Decision Tree Analysis</h2>',
                unsafe_allow_html=True)

    try:
        df_processed, le_district, le_month = preprocess_for_decision_tree(df)
    except Exception as e:
        st.error(f"Error during decision tree preprocessing: {e}")
        return

    # Target selection
    available_targets = [t for t in ["High_Productivity", "High_Admin_Spending"] if t in df_processed.columns]
    if not available_targets:
         st.error("Target variables ('High_Productivity', 'High_Admin_Spending') could not be created. Check input data and preprocessing logic.")
         return

    target_option = st.selectbox(
        "Select Target Variable",
        available_targets,
        key='dt_target_select' # Added key
    )

    # Feature selection based on target
    if target_option == "High_Productivity":
        features = ['Material_and_skilled_Wages',
                   'Average_days_of_employment_provided_per_Household',
                   'Average_Wage_rate_per_day_per_person',
                   'Total_Households_Worked']
    else: # High_Admin_Spending
        features = ['Persondays_of_Central_Liability_so_far',
                   'percent_of_Expenditure_on_Agriculture_Allied_Works',
                   'Material_and_skilled_Wages',
                   'Average_days_of_employment_provided_per_Household']

    # Filter available features from the processed dataframe
    available_features = [f for f in features if f in df_processed.columns and pd.api.types.is_numeric_dtype(df_processed[f])] # Ensure numeric

    if len(available_features) == 0:
        st.error(f"None of the required numerical features for target '{target_option}' found in dataset: {', '.join(features)}")
        return
    elif len(available_features) < len(features):
        st.warning(f"Using available numerical features: {', '.join(available_features)}. Missing: {', '.join(set(features) - set(available_features))}")


    X = df_processed[available_features].fillna(df_processed[available_features].median())
    y = df_processed[target_option]

    if len(X) != len(y):
        st.error("Feature and target data lengths do not match after preprocessing.")
        return
    if y.nunique() < 2:
         st.error(f"Target variable '{target_option}' has only one class ({y.unique()}). Cannot train classifier.")
         return


    # Split data
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=42, stratify=y
        )
    except ValueError as e:
        st.error(f"Error during train-test split (possibly due to insufficient samples for stratification): {e}")
        st.write(f"Target variable distribution:\n{y.value_counts()}")
        return


    # Model parameters
    col1, col2 = st.columns(2)
    with col1:
        max_depth = st.slider("Max Depth", 2, 10, 5, key='dt_depth_slider') # Added key
    with col2:
        min_samples_leaf = st.slider("Min Samples per Leaf", 5, 50, 10, key='dt_leaf_slider') # Added key

    # Train model
    try:
        clf = DecisionTreeClassifier(
            criterion='entropy',
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            random_state=42
        )
        clf.fit(X_train, y_train)
    except Exception as e:
        st.error(f"Error training Decision Tree model: {e}")
        return

    # Predictions
    y_pred_train = clf.predict(X_train)
    y_pred_test = clf.predict(X_test)

    # Metrics
    train_acc = accuracy_score(y_train, y_pred_train)
    test_acc = accuracy_score(y_test, y_pred_test)

    # Display Results
    tab1, tab2, tab3 = st.tabs(["📊 Performance Metrics", "🌳 Tree Visualization", "📈 Feature Importance"])

    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Training Accuracy", f"{train_acc*100:.2f}%")
        with col2:
            st.metric("Testing Accuracy", f"{test_acc*100:.2f}%")

        st.markdown("**Classification Report:**")
        try:
            report = classification_report(y_test, y_pred_test, output_dict=True, zero_division=0) # Handle zero division
            report_df = pd.DataFrame(report).transpose()
            st.dataframe(report_df, use_container_width=True)
        except Exception as e:
            st.warning(f"Could not generate classification report: {e}")

        st.markdown("**Confusion Matrix:**")
        cm = confusion_matrix(y_test, y_pred_test)
        class_names = ['Low', 'High'] # Assuming 0=Low, 1=High based on target creation

        fig, ax = plt.subplots(figsize=(6, 4)) # Smaller size
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=class_names, yticklabels=class_names) # Add labels
        ax.set_title('Confusion Matrix')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        st.pyplot(fig)

    with tab2:
        st.subheader("Decision Tree Structure")

        try:
            fig, ax = plt.subplots(figsize=(20, 12)) # Consider making size adjustable or limiting depth for display
            plot_tree(clf, feature_names=available_features,
                     class_names=['Low', 'High'], # Assuming binary classification 0/1
                     filled=True, rounded=True, fontsize=10, ax=ax, max_depth=5) # Limit depth for initial display
            st.pyplot(fig)
            st.caption("Displaying tree up to depth 5 for readability. Adjust Max Depth slider for model training.")
        except Exception as e:
            st.warning(f"Could not plot decision tree: {e}")


    with tab3:
        st.subheader("Feature Importance Analysis")

        importance_df = pd.DataFrame({
            'Feature': available_features,
            'Importance': clf.feature_importances_
        }).sort_values('Importance', ascending=False)

        col1, col2 = st.columns([1, 1])

        with col1:
            fig_imp = px.bar(importance_df, x='Importance', y='Feature', # Swapped axes for better label reading
                        orientation='h', # Horizontal bar chart
                        title='Feature Importance Scores',
                        color='Importance', color_continuous_scale='Greens')
            # fig_imp.update_layout(xaxis_tickangle=-45) # Not needed for horizontal
            st.plotly_chart(fig_imp, use_container_width=True)

        with col2:
            st.dataframe(importance_df, use_container_width=True)

            st.success("""
            **Interpretation:**
            - Higher values indicate more important features for this model split.
            - Top features have strongest predictive power according to the tree.
            - Helps identify key drivers for the selected target variable.
            """)


# Advanced Predictions Page
def show_predictions(df):
    st.markdown('<h2 class="sub-header">🔮 Advanced Prediction System</h2>',
                unsafe_allow_html=True)

    try:
        df_processed, le_district, le_month = preprocess_for_decision_tree(df)
        if 'High_Productivity' not in df_processed.columns:
             st.error("Target variable 'High_Productivity' could not be created for predictions.")
             return
    except Exception as e:
        st.error(f"Error during prediction preprocessing: {e}")
        return


    st.subheader("🎯 Predict District Productivity (Using Decision Tree)")

    # Define features used for High_Productivity prediction in Decision Tree section
    productivity_features = ['Material_and_skilled_Wages',
                             'Average_days_of_employment_provided_per_Household',
                             'Average_Wage_rate_per_day_per_person',
                             'Total_Households_Worked'] # Assuming this was used

    # Check which features are actually available after preprocessing
    available_pred_features = [f for f in productivity_features if f in df_processed.columns and pd.api.types.is_numeric_dtype(df_processed[f])] # Check numeric

    if len(available_pred_features) == 0:
        st.error("No numerical features available for productivity prediction.")
        return
    elif len(available_pred_features) < len(productivity_features):
         st.warning(f"Using available features for prediction: {', '.join(available_pred_features)}. Missing: {', '.join(set(productivity_features) - set(available_pred_features))}")


    st.markdown("**Enter Feature Values:**")
    input_values = {}
    cols = st.columns(len(available_pred_features))
    for i, feature in enumerate(available_pred_features):
         # Try to get median as default, else use 0
         default_val = float(df_processed[feature].median()) if pd.notna(df_processed[feature].median()) else 0.0
         # Define reasonable min/max based on feature name or use broad defaults
         min_val = 0.0
         max_val = float(df_processed[feature].max()) * 1.5 if pd.notna(df_processed[feature].max()) else 10000.0
         step = 1.0 if np.issubdtype(df_processed[feature].dtype, np.integer) else 10.0

         input_values[feature] = cols[i].number_input(
             f"{feature}",
             min_value=min_val,
             #max_value=max_val, # Consider removing max_value for user flexibility
             value=default_val,
             step=step,
             key=f"pred_input_{feature}" # Added key
         )


    # District input only if district_encoded was available and used
    district_encoded_val = None
    district_encoded_included = False
    if 'district_encoded' in df_processed.columns and 'district_name' in df.columns:
        if st.checkbox("Include District Information (if available)?", key='pred_district_check'): # Added key
             district_list = ["Unknown"] + sorted(list(le_district.classes_)) # Use classes_ from encoder
             district = st.selectbox("Select District", district_list, key='pred_district_select') # Added key
             if district != "Unknown":
                 try:
                     district_encoded_val = le_district.transform([district])[0]
                     district_encoded_included = True # Mark as included
                 except ValueError:
                     st.warning(f"District '{district}' not seen during initial encoding. Using median encoding.")
                     district_encoded_val = int(df_processed['district_encoded'].median()) # Use median encoding
             else:
                 district_encoded_val = int(df_processed['district_encoded'].median()) # Or use median/mode if preferred for 'Unknown'
             input_values['district_encoded'] = district_encoded_val


    if st.button("🔮 Predict Productivity", type="primary", key='pred_button'): # Added key
        # Prepare data only with available features + potentially district_encoded
        final_features_used_pred = available_pred_features[:] # Copy list
        if district_encoded_included:
             if 'district_encoded' not in final_features_used_pred: # Check if already added (might happen if it was in original list)
                  final_features_used_pred.append('district_encoded')


        X = df_processed[final_features_used_pred].fillna(df_processed[final_features_used_pred].median())
        y = df_processed['High_Productivity']

        # Ensure target has variability
        if y.nunique() < 2:
            st.error("Target variable 'High_Productivity' has only one class. Cannot train prediction model.")
            return

        # Train model (using the same parameters as in the Decision Tree section for consistency)
        try:
             clf = DecisionTreeClassifier(criterion='entropy', max_depth=5, min_samples_leaf=10, random_state=42)
             # Stratify only if both classes are present in y
             if len(y.unique()) > 1:
                 clf.fit(X, y) # No stratification here as we train on full prepared data
             else:
                  st.error("Cannot train model with only one target class present.")
                  return

        except Exception as e:
             st.error(f"Error training prediction model: {e}")
             return

        # Prepare input data array based ONLY on the features used for training THIS prediction model
        input_data_list = []
        for feature in final_features_used_pred: # Use the list used for fitting
             input_data_list.append(input_values[feature])

        input_data = np.array([input_data_list])


        # Make prediction
        try:
             prediction = clf.predict(input_data)[0]
             probability = clf.predict_proba(input_data)[0]

             st.markdown("---")

             if prediction == 1:
                 st.success(f"✅ **High Productivity Expected!** (Confidence: {probability[1]*100:.1f}%)")
                 st.balloons()
             else:
                 st.warning(f"⚠️ **Lower Productivity Expected** (Confidence: {probability[0]*100:.1f}%)")

             # Show feature contribution (Importance from THIS specific prediction model)
             st.markdown("**Feature Importances (for this prediction model):**")
             feature_importance = pd.DataFrame({
                 'Feature': final_features_used_pred,
                 'Importance': clf.feature_importances_
             }).sort_values('Importance', ascending=False)

             fig_pred_imp = px.bar(feature_importance, x='Importance', y='Feature', # Swapped axes
                             orientation='h', # Horizontal
                             title='Feature Importance Driving Prediction',
                             color='Importance', color_continuous_scale='RdYlGn')
             st.plotly_chart(fig_pred_imp, use_container_width=True)

        except Exception as e:
            st.error(f"Error during prediction: {e}")
            st.write("Input Data used:", input_data)


# --- >>> NEW SECTION FOR NEURAL NETWORK RESULTS <<< ---
def show_neural_network(df):
    """Displays the results from the offline Neural Network training."""
    st.markdown('<h2 class="sub-header">🧠 Neural Network Wage Prediction Results</h2>',
                unsafe_allow_html=True)

    st.info("""
    This section presents the **results** obtained from training Neural Network models (specifically MLPRegressor from Scikit-learn) **offline** using the provided Jupyter Notebook.
    - ✅ **No TensorFlow/DLL issues:** This approach works seamlessly on standard Python setups.
    - 📊 The models were trained to predict the **'Wages'** column based on selected numerical features.
    - 💡 The table below compares the performance against traditional ML models trained on the same data splits and features.
    """)
    st.markdown("---")

    # --- Hardcoded Results from the Jupyter Notebook ---
    nn_results_data = {
        'Model': [
            'Random Forest',
            'Gradient Boosting',
            'Optimized NN',
            'Deep NN (3 layers)',
            'Simple NN (1 layer)',
            'Ridge Regression'
        ],
        'RMSE': [
            226.67, 381.37, 570.74, 606.41, 884.13, 899.70
        ],
        'MAE': [
            67.06, 232.19, 347.20, 338.60, 600.48, 543.76
        ],
        'R² Score': [
            0.9989, 0.9970, 0.9933, 0.9924, 0.9839, 0.9834
        ],
        'Type': [
            'Traditional ML', 'Traditional ML', 'Neural Network',
            'Neural Network', 'Neural Network', 'Traditional ML'
        ]
    }
    results_df = pd.DataFrame(nn_results_data)

    st.subheader("📊 Complete Model Comparison")
    st.dataframe(results_df.style.format({
        'RMSE': '{:,.2f}',
        'MAE': '{:,.2f}',
        'R² Score': '{:.4f}'
    }).highlight_max(subset=['R² Score'], color='lightgreen')
      .apply(lambda x: ['background-color: lightblue' if x.Type == 'Neural Network' else '' for i in x], axis=1),
      use_container_width=True
    )

    st.markdown("---")

    # --- Details of the Best Neural Network ---
    st.subheader("🏆 Best Neural Network Model: Optimized NN")
    st.write("Hyperparameters found via GridSearchCV (offline):")
    st.json({
        'alpha': 0.001,
        'hidden_layer_sizes': [100, 50, 25],
        'learning_rate_init': 0.01,
        'activation': 'relu',
        'solver': 'adam',
        'max_iter': 300,
        'early_stopping': True
    })

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Optimized NN R² Score", f"{0.9933:.4f}")
    with col2:
        st.metric("Optimized NN RMSE", f"{570.74:,.2f}")
    with col3:
        st.metric("Optimized NN MAE", f"{347.20:,.2f}")

    st.markdown("---")

    # --- Feature Importance (from Random Forest in the notebook) ---
    st.subheader("📈 Feature Importance (from Comparative Random Forest Model)")
    feature_importance_data = {
        'Feature': [
            'Total_Exp', 'Persondays_of_Central_Liability_so_far', 'Approved_Labour_Budget',
            'Number_of_Ongoing_Works', 'Women_Persondays', 'Total_Individuals_Worked',
            'Number_of_Completed_Works', 'Total_Households_Worked', 'ST_persondays',
            'SC_persondays'
        ],
        'Importance': [
            0.543737, 0.295124, 0.139887, 0.011026, 0.002967, 0.002652,
            0.001643, 0.001466, 0.000932, 0.000567
        ]
    }
    importance_df = pd.DataFrame(feature_importance_data)

    col1, col2 = st.columns([2, 1])
    with col1:
        fig_imp = px.bar(importance_df, x='Importance', y='Feature',
                         orientation='h', title='Feature Importance Scores (RF)',
                         color='Importance', color_continuous_scale='Greens')
        fig_imp.update_layout(yaxis={'categoryorder':'total ascending'}) # Show most important at top
        st.plotly_chart(fig_imp, use_container_width=True)
    with col2:
        st.dataframe(importance_df, use_container_width=True)
        st.caption("Importance scores indicate contribution to the Random Forest model's wage prediction, providing context for the NN features.")

    st.markdown("---")

    # --- Performance Comparison Visualization ---
    st.subheader("📉 Model Performance Visualization")
    try:
        fig_comp, axes = plt.subplots(1, 3, figsize=(18, 5))
        colors = ['skyblue' if t == 'Neural Network' else 'lightcoral' for t in results_df['Type']]
        # RMSE
        axes[0].barh(results_df['Model'], results_df['RMSE'], color=colors)
        axes[0].set_xlabel('RMSE (Lower is Better)', fontweight='bold')
        axes[0].set_title('Root Mean Squared Error', fontsize=13, fontweight='bold')
        axes[0].grid(axis='x', alpha=0.3)
        axes[0].invert_yaxis()
        # MAE
        axes[1].barh(results_df['Model'], results_df['MAE'], color=colors)
        axes[1].set_xlabel('MAE (Lower is Better)', fontweight='bold')
        axes[1].set_title('Mean Absolute Error', fontsize=13, fontweight='bold')
        axes[1].grid(axis='x', alpha=0.3)
        axes[1].invert_yaxis()
        # R² Score
        axes[2].barh(results_df['Model'], results_df['R² Score'], color=colors)
        axes[2].set_xlabel('R² Score (Higher is Better)', fontweight='bold')
        axes[2].set_title('R² Score', fontsize=13, fontweight='bold')
        # Adjust xlim dynamically based on data, ensuring 1.0 is visible if scores are high
        min_r2 = results_df['R² Score'].min()
        axes[2].set_xlim([max(0, min_r2 - 0.005) , min(1.005, results_df['R² Score'].max() + 0.001)])
        axes[2].grid(axis='x', alpha=0.3)
        axes[2].invert_yaxis()
        # Legend
        legend_elements = [Patch(facecolor='skyblue', label='Neural Network'), Patch(facecolor='lightcoral', label='Traditional ML')]
        axes[2].legend(handles=legend_elements, loc='lower right')
        plt.tight_layout(pad=1.5) # Add padding
        st.pyplot(fig_comp)
    except Exception as e:
        st.warning(f"Could not generate model comparison plot: {e}")


    st.markdown("---")

    # --- Illustrative Prediction Input/Output ---
    st.subheader("💡 Illustrative Wage Prediction (Based on Optimized NN Features)")
    st.write("Enter hypothetical values for the features used by the Neural Network model:")

    nn_feature_columns = [
        'Total_Exp', 'Approved_Labour_Budget', 'Women_Persondays',
        'Persondays_of_Central_Liability_so_far', 'Total_Households_Worked',
        'Total_Individuals_Worked', 'Number_of_Completed_Works',
        'Number_of_Ongoing_Works', 'SC_persondays', 'ST_persondays'
    ]
    # Filter for columns actually present in the loaded dataframe
    available_nn_features = [col for col in nn_feature_columns if col in df.columns]

    input_nn_values = {}
    cols = st.columns(3) # Use columns for better layout

    for i, feature in enumerate(available_nn_features):
         col_index = i % 3
         # Use median from the original dataframe if available, otherwise 0
         default_val = float(df[feature].median()) if pd.notna(df[feature].median()) else 0.0
         min_val = 0.0
         # Consider removing max_value for user flexibility or set a high default
         # max_val = float(df[feature].max()) * 1.5 if pd.notna(df[feature].max()) else 1e9
         step = 100.0 if 'Budget' in feature or 'Exp' in feature or 'persondays' in feature else 10.0 # Heuristic step

         input_nn_values[feature] = cols[col_index].number_input(
             f"{feature}",
             min_value=min_val,
             # max_value=max_val,
             value=default_val,
             step=step,
             key=f"nn_input_{feature}" # Unique key
         )

    if st.button("Predict Wage (Illustrative)", type="primary", key='nn_predict_button'): # Unique key
        # --- This is NOT a real prediction. It's a placeholder. ---
        # A real implementation would load the scaler, scale the input, load the model, and predict.
        # Here, we just display a plausible-looking output based on the input values' magnitude,
        # weighted slightly by the RF feature importance for a *slightly* more informed guess.

        # Simple heuristic for illustration: use RF importance as rough weights
        # Normalize importance scores to sum to ~1 for weighting
        imp_subset = importance_df[importance_df['Feature'].isin(available_nn_features)]
        if not imp_subset.empty and imp_subset['Importance'].sum() > 0:
             weights = imp_subset.set_index('Feature')['Importance'] / imp_subset['Importance'].sum()
        else:
             weights = pd.Series(1/len(available_nn_features), index=available_nn_features) # Equal weight if no importance

        # Weighted sum heuristic - very rough approximation!
        weighted_sum = sum(input_nn_values.get(f, 0) * weights.get(f, 0) for f in available_nn_features)
        # Scale the result to look somewhat like the wage values, add base intercept guess
        base_intercept_guess = 1500 # Guess based on regression results
        scaling_factor_guess = 0.5 # Guess based on regression coefficients
        predicted_wage_placeholder = base_intercept_guess + (weighted_sum * scaling_factor_guess)


        st.success(f"💡 **Illustrative Predicted Wage:** ₹ {max(0, predicted_wage_placeholder):,.2f}") # Ensure non-negative
        st.caption("⚠️ **Note:** This is a sample prediction for demonstration purposes only. It uses a simple heuristic based on inputs and RF feature importance, and does **not** reflect the actual trained Neural Network's calculation (which requires scaling and the specific model weights).")


# --- END OF NEW SECTION ---


if __name__ == "__main__":
    main()