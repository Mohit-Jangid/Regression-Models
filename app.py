# app.py 
# """--> if you want to run the app in your local machine then follow the below steps
#         1.  Place both app.py and requirements.txt files at same directory
#         2.  open the command prompt and run 'python -m venv venv' to create a virtual environment
#         3.  run the command 'venv\scripts\activate' to activate the virtual environment
#         4.  run the command 'pip install -r requirements.txt' to install all the dependencies
#         5.  run the command 'streamlit run app.py' to open the dashboard in localhost url"""


# Import necessary libraries

import streamlit as st
import time
# from st_pages import add_page_title           
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objs as go
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, root_mean_squared_error
from sklearn.preprocessing import StandardScaler
from itertools import combinations
import pickle
import io


# Visualization Class

class Visualize:
    def __init__(self, df):
        self.df = df

    def detect_outliers(self, series):
        q1 = np.percentile(series, 25)
        q3 = np.percentile(series, 75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        return (series < lower_bound) | (series > upper_bound)

    def scatter_with_marginal_histogram(self):
        st.subheader("📌 Scatter Plot with Marginal Histogram")
        
        # Identify numeric columns
        numeric_cols = [col for col in self.df.columns if np.issubdtype(self.df[col].dropna().dtype, np.number)]

        if not numeric_cols:
            st.warning("No numeric columns found in the dataset.")
            return

        # Dropdown to select column
        col1, col2 = st.columns([1, 4])
        with col1:
            selected_col = st.selectbox("Select a numeric column to visualize", numeric_cols)

        series = self.df[selected_col].dropna()
        y_values = series.values
        x_values = np.arange(len(y_values))

        # Outlier Detection
        is_outlier = self.detect_outliers(series)

        # Create DataFrame for plotting
        plot_df = pd.DataFrame({
            "Index": x_values,
            selected_col: y_values,
            "Outlier": np.where(is_outlier, "Outlier", "Normal")
        })

        # Create scatter plot
        fig = px.scatter(
            plot_df,
            x="Index",
            y=selected_col,
            color="Outlier",
            marginal_y="histogram",
            opacity=0.8,
            color_discrete_map={"Normal": "teal", "Outlier": "red"},
            template="plotly_white",
            title=f"<b>Scatter with Histogram: {selected_col}</b>",
            labels={"Index": "Index", selected_col: selected_col, "Outlier": "Data Type"}
        )

        fig.update_traces(marker=dict(size=7), selector=dict(mode='markers'))

        # Mean and Median
        mean_val = np.mean(y_values)
        median_val = np.median(y_values)

        fig.add_hline(
            y=mean_val,
            line_dash="dash",
            line_color="orange",
            annotation_text="Mean",
            annotation_position="bottom left",
        )

        fig.add_hline(
            y=median_val,
            line_dash="dash",
            line_color="purple",
            annotation_text="Median",
            annotation_position="top left",
        )

        fig.update_layout(height=650, hovermode="closest")

        # Display metrics and chart
        st.markdown(f"""
        <div style='font-family:Arial; font-size:14px;'>
            <b style="color: teal;">Mean:</b> {mean_val:.2f} &nbsp;&nbsp;&nbsp;
            <b style="color: purple;">Median:</b> {median_val:.2f}
        </div>
        """, unsafe_allow_html=True)

        st.plotly_chart(fig, use_container_width=True, key=f"scatter_with_marginal_histogram{str(time.time())}")


    def pair_plot(self):
        st.subheader("🔗 Pair Plot")

        # Select numeric columns
        numeric_cols = [col for col in self.df.columns if np.issubdtype(self.df[col].dropna().dtype, np.number)]

        if len(numeric_cols) < 2:
            st.warning("Need at least 2 numerical columns for a pair plot.")
            return
        
        # Optionally allow the user to select which numeric columns to include
        selected_cols = st.multiselect("Select numeric columns for pair plot", options=numeric_cols, default=numeric_cols[:4])

        if len(selected_cols) < 2:
            st.info("Please select at least two numeric columns.")
            return
        
        data = self.df[selected_cols].dropna()

        # Outlier detection
        outlier_flags = pd.DataFrame(False, index=data.index, columns=selected_cols)
        for col in selected_cols:
            outlier_flags[col] = self.detect_outliers(data[col])

        combined_outlier = outlier_flags.any(axis=1)

        plot_df = data.copy()
        plot_df["Data_Type"] = np.where(combined_outlier, "Outlier", "Normal")

        # Plot
        fig = px.scatter_matrix(
            plot_df,
            dimensions=selected_cols,
            color="Data_Type",
            opacity=0.6,
            color_discrete_map={"Normal": "teal", "Outlier": "red"},
            title="<b>Pair Plot with Outlier Detection</b>",
            labels={col: col for col in selected_cols},
            height=1000,
            width=1000,
        )

        fig.update_traces(
            diagonal_visible=True,
            marker=dict(size=4),
            selected_marker=dict(opacity=1),
            unselected_marker=dict(opacity=0.3),
        )

        fig.update_layout(
            template="plotly_white",
            hovermode="closest",
            dragmode="select",
            margin=dict(l=70, r=70, b=70, t=100),
            title_x=0.5,
            font=dict(family="Arial", size=12),
            legend=dict(
                title="",
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
            ),
        )

        st.plotly_chart(fig, use_container_width=True, key=f"pair_plot{str(time.time())}")

    def correlation_heatmap(self):
        st.subheader("🧮 Correlation Heatmap")

        numeric_cols = [col for col in self.df.columns if np.issubdtype(self.df[col].dropna().dtype, np.number)]

        if len(numeric_cols) < 2:
            st.warning("At least two numerical columns are required for a correlation heatmap.")
            return

        selected_cols = st.multiselect(
            "Select numeric columns to include in the heatmap:",
            options=numeric_cols,
            default=numeric_cols[:4],
        )

        if len(selected_cols) < 2:
            st.info("Please select at least two numeric columns.")
            return

        selected_data = self.df[selected_cols].dropna()

        corr_matrix = selected_data.corr().round(2)

        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.index,
            text=corr_matrix.values.round(2).astype(str),
            texttemplate="%{text}",
            textfont=dict(size=18,color='black',family='Arial'),
            colorscale="RdBu",
            zmin=-1,
            zmax=1,
            colorbar=dict(title="Correlation", tickfont=dict(size=12)),
            hovertemplate="Correlation between %{x} and %{y}: %{z}<extra></extra>",
        ))

        fig.update_layout(
            title="<b>📊 Correlation Heatmap</b>",
            xaxis=dict(title=""),
            yaxis=dict(title=""),
            template="plotly_white",
            height=600,
            width=800,
            margin=dict(t=60, l=60, r=40, b=60),
            font=dict(family="Arial", size=12),
        )

        st.plotly_chart(fig, use_container_width=True, key=f"correlation_heatmap{str(time.time())}")

    def interactive_relation_scatter(self):
        st.subheader("🔍 Interactive Relation: Scatter Plot")

        numeric_cols = self.df.select_dtypes(include=["number"]).columns.tolist()

        if len(numeric_cols) < 2:
            st.error("Need at least two numeric columns for scatter plot.")
            return

        col1, col2 = st.columns(2)
        with col1:
            x_col = st.selectbox("Select X-axis", numeric_cols, index=0)
        with col2:
            y_col = st.selectbox("Select Y-axis", numeric_cols, index=1)

        if x_col == y_col:
            st.warning("Select different columns for X and Y axes.")
            return
        
        data = self.df[[x_col, y_col]].dropna()
        data["Outlier"] = np.where(self.detect_outliers(data[x_col]) | self.detect_outliers(data[y_col]),
        "Outlier", "Normal")

        fig = px.scatter(
            data, x=x_col, y=y_col, color="Outlier",
            color_discrete_map={"Normal": "teal", "Outlier": "red"},
            opacity=0.7,
            template="plotly_white",
            title=f"<b>{x_col} vs {y_col}</b>",
            height=600
        )

        fig.add_vline(x=data[x_col].mean(), line_dash="dash", line_color="orange", annotation_text="Mean", annotation_position="top left")
        fig.add_vline(x=data[x_col].median(), line_dash="dash", line_color="purple", annotation_text="Median", annotation_position="bottom left")
        fig.add_hline(y=data[y_col].mean(), line_dash="dash", line_color="orange", annotation_text="Mean", annotation_position="top right")
        fig.add_hline(y=data[y_col].median(), line_dash="dash", line_color="purple", annotation_text="Median", annotation_position="bottom right")

        st.plotly_chart(fig, use_container_width=True, key=f"interactive_relation_scatter{str(time.time())}")


    def interactive_relation_box(self):
        st.subheader("📦 Interactive Relation: Box Plot (Binned)")
        
        numeric_cols = self.df.select_dtypes(include=["number"]).columns.tolist()

        if len(numeric_cols) < 2:
            st.error("Need at least two numeric columns for scatter plot.")
            return

        col1, col2 = st.columns(2)
        with col1:
            x_col = st.selectbox("Select X-axis", numeric_cols, index=0)
        with col2:
            y_col = st.selectbox("Select Y-axis", numeric_cols, index=1)

        if x_col == y_col:
            st.warning("Select different columns for X and Y axes.")
            return

        bins = st.slider("Select number of bins", min_value=2, max_value=20, value=5, step=1)

        data = self.df[[x_col, y_col]].dropna()
        data["Bins"] = pd.cut(data[x_col], bins=bins).astype(str)
        data["Outlier"] = np.where(self.detect_outliers(data[x_col]) | self.detect_outliers(data[y_col]),
        "Outlier", "Normal")

        fig = px.box(
            data, x="Bins", y=y_col, points="all", color="Outlier",
            color_discrete_map={"Normal": "teal", "Outlier": "red"},
            title=f"<b>{x_col} (binned) vs {y_col}</b>",
            template="plotly_white",
            height=600
        )

        st.plotly_chart(fig, use_container_width=True, key=f"interactive_relation_box{str(time.time())}")


# Regression Models Class

class Model:
    def __init__(self, df, feature_columns, target_column):
        self.df = df = df.fillna(df.mean(numeric_only=True))
        self.feature_columns = feature_columns
        self.target_column = target_column
        self.X = df[feature_columns]
        self.y = df[target_column]
        self.scaler = StandardScaler()
        self.X_scaled = pd.DataFrame(self.scaler.fit_transform(self.X),columns=self.feature_columns)
        self.model = None

    def train_model(self, model_type='Linear Regression Model', **kwargs):
        if model_type == 'Linear Regression Model':
            self.model = LinearRegression()
        elif model_type == 'Ridge Regression Model':
            self.model = Ridge(alpha=kwargs.get('alpha', 1.0))
        elif model_type == 'Lasso Regression Model':
            self.model = Lasso(alpha=kwargs.get('alpha', 1.0))
        elif model_type == 'RandomForest Regression Model':
            self.model = RandomForestRegressor(
                n_estimators=kwargs.get('n_estimators', 100),
                random_state=28
            )
        elif model_type == 'DecisionTree Regression Model':
            self.model = DecisionTreeRegressor(
                max_depth=kwargs.get('max_depth', 10),
                random_state=28
            )

        else:
            raise ValueError("Unsupported model type")

        self.model.fit(self.X_scaled, self.y)
        self.y_pred = self.model.predict(self.X_scaled)
        return self.model

    def _prepare_feature_importance(self):
        if hasattr(self.model, 'coef_'):
            importance = pd.Series(self.model.coef_, index=self.feature_columns)
        elif hasattr(self.model, 'feature_importances_'):
            importance = pd.Series(self.model.feature_importances_, index=self.feature_columns)
        else: 
            st.write("Model does not support feature importances")
            return None

        df = pd.DataFrame({
            'Feature': self.feature_columns,
            'Importance': importance,
            'Standardized Importance': importance / np.std(self.X, axis=0),
            'Mean': self.X.mean(),
            'Std': self.X.std(),
        }).sort_values(by='Importance', key=abs, ascending=False)

        return df

    def feature_importance_table(self):
        self.importance_df = self._prepare_feature_importance()
        if self.importance_df is not None:
            st.dataframe(self.importance_df, use_container_width=True)

    def feature_importance_plot(self):
        self.importance_df = self._prepare_feature_importance()
        if self.importance_df is not None:
            fig = px.bar(self.importance_df, x='Importance', y='Feature', orientation='h',
                        title='🔍 Feature Importance', color='Importance',
                        color_continuous_scale='Viridis', height=500)
            fig.update_layout(template='plotly_white')

            max_abs_value = max(abs(self.importance_df['Importance'].min()), abs(self.importance_df['Importance'].max()))
            fig.update_xaxes(range=[-max_abs_value, max_abs_value])
            st.plotly_chart(fig, use_container_width=True, key=f"feature_importance_plot{str(time.time())}")


    def generate_equation(self):
        if not hasattr(self.model, 'coef_'):
            st.warning("This model does not support equation extraction.")
            return
        
        # intercept = self.y.mean()
        intercept = self.model.intercept_
        pivot_values = self.df[self.feature_columns].mean()
        coef = self.model.coef_ / np.std(self.X, axis=0)
        terms = [f"({coef:.4f} * ({feat} - {pivot_values[feat]:.3f}))" for coef, feat in zip(coef, self.feature_columns)]
        equation = " + ".join(terms)
        full_eq = f"{self.target_column} = {intercept:.4f} + {equation}"
        st.code(full_eq, language='python')

    def user_input_and_predict(self, model_obj, feature_columns):

        columns = st.columns(len(feature_columns))

        user_inputs = {}
        for i, col in enumerate(feature_columns):
            with columns[i]:
                user_inputs[col] = st.number_input(
                    f"{col}", value=0.0, format="%.4f", key=f"input_{col}"
                )

        input_df = pd.DataFrame([user_inputs])

        # st.dataframe(input_df, use_container_width=True)

        if st.button("🔮 Predict"):
            model_pred = model_obj.predict_user_input(input_df)
            st.success(f"📊 Predicted Output: **{model_pred:.2f}**")
            return model_pred
        
    def predict_user_input(self, input_df):
        input_scaled = self.scaler.transform(input_df)
        return self.model.predict(input_scaled)[0]

    def evaluate_model(self):

        mae = mean_absolute_error(self.y, self.y_pred)
        mse = mean_squared_error(self.y, self.y_pred)
        rmse = root_mean_squared_error(self.y, self.y_pred)
        r2 = r2_score(self.y, self.y_pred)*100
        adjusted_r2 = (1 - (1 - (r2/100)) * (len(self.y) - 1) / (len(self.y) - len(self.feature_columns) - 1))*100
        
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Adjusted R² Score", f"{adjusted_r2:.2f}"+"%")
        col2.metric("R² Score", f"{r2:.2f}"+"%")
        col3.metric("MAE", f"{mae:.2f}")
        col4.metric("MSE", f"{mse:.2f}")
        col5.metric("RMSE", f"{rmse:.2f}")

    def plot_actual_vs_predicted(self):
        fig = px.scatter(x=self.y, y=self.y_pred, labels={'x': 'Actual Values', 'y': 'Predicted Values'}, title='📈 Actual vs Predicted', color_discrete_sequence=['teal'])
        
        min_val = min(self.y.min(), self.y_pred.min())
        max_val = max(self.y.max(), self.y_pred.max())
        fig.add_trace(go.Scatter(x=[min_val, max_val], y=[min_val, max_val], mode='lines', name='Ideal Line', line=dict(color='orange')))
        fig.update_traces(marker=dict(size=6, opacity=0.6))
        fig.update_layout(template='plotly_white')
        st.plotly_chart(fig, use_container_width=True, key=f"plot_actual_vs_predicted{str(time.time())}")

    def residual_histogram(self):
        residuals = self.y - self.y_pred
        fig = px.histogram(residuals, nbins=30, title='📊 Residual Histogram', color_discrete_sequence=['teal'])
        fig.add_vline(x=residuals.mean(), line_dash="dash", line_color="green",
                    annotation_text="Mean", annotation_position="top left")
        fig.add_vline(x=residuals.median(), line_dash="dot", line_color="red",
                    annotation_text="Median", annotation_position="top right")
        fig.update_layout(template='plotly_white', xaxis_title='Residuals', yaxis_title='Count')
        st.plotly_chart(fig, use_container_width=True, key=f"residual_histogram{str(time.time())}")

    def residual_scatter(self):
        residuals = self.y - self.y_pred
        fig = px.scatter(x=self.y_pred, y=residuals, labels={'x': 'Predicted Value', 'y': 'Residual'}, title='📉 Residuals vs Predicted', color_discrete_sequence=['teal'])
        fig.add_hline(y=0, line_dash="dash", line_color="red")
        fig.add_hline(y=residuals.mean(), line_dash="dot", line_color="green",
                    annotation_text="Mean Residual", annotation_position="top left")
        fig.update_layout(template='plotly_white')
        st.plotly_chart(fig, use_container_width=True, key=f"residual_scatter{str(time.time())}")
    
    def residual_scatter_with_histogram(self):
        residuals = self.y - self.y_pred
        fig = px.scatter(
            x=self.y_pred,
            y=residuals,
            labels={'x': 'Predicted Value', 'y': 'Residual'},
            title='📊 Residuals vs Predicted with Histogram',
            color_discrete_sequence=['teal'],
            marginal_y='histogram',
        )
        
        fig.add_hline(y=0, line_dash="dash", line_color="red")
        fig.add_hline(y=residuals.mean(), line_dash="dot", line_color="green",
                    annotation_text="Mean Residual", annotation_position="top left")

        fig.update_layout(template='plotly_white')
        st.plotly_chart(fig, use_container_width=True, key=f"residual_scatter_with_histogram{str(time.time())}")

    def learning_curve(self):

        train_sizes, train_scores, test_scores = learning_curve(
            self.model,
            self.X,
            self.y,
            cv=10,
            scoring='r2',
            train_sizes=np.linspace(0.1, 1.0, 5),
            shuffle=True,
            random_state=42
        )

        train_scores_mean = np.mean(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)

        # interactive plot
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=train_sizes,
            y=train_scores_mean,
            mode='lines+markers',
            name='Train R² Score',
            line=dict(color='green'),
            hovertemplate='Train Size: %{x}<br>R² Score: %{y:.3f}<extra></extra>'
        ))

        fig.add_trace(go.Scatter(
            x=train_sizes,
            y=test_scores_mean,
            mode='lines+markers',
            name='Validation R² Score',
            line=dict(color='orange'),
            hovertemplate='Train Size: %{x}<br>R² Score: %{y:.3f}<extra></extra>'
        ))

        # layout options
        fig.update_layout(
            title="📈 Learning Curve by Selected Model",
            xaxis_title="Training Set Size",
            yaxis_title="R² Score",
            template='plotly_white',
            legend=dict(orientation="h", y=1.1, x=0.5, xanchor='center'),
            margin=dict(t=80)
        )

        st.plotly_chart(fig, use_container_width=True, key=f"learning_curve{str(time.time())}")


    def train_on_all_subsets(self, model_type='Linear Regression Model', **kwargs):
        results = []
        n = len(self.df)

        for r in range(1, len(self.feature_columns) + 1):
            for subset in combinations(self.feature_columns, r):
                # Create new Model instance for this subset
                subset_model = Model(self.df, list(subset), self.target_column)

                # Train with selected model type
                subset_model.train_model(model_type=model_type, **kwargs)

                # --- Evaluate (reuse your evaluate_model math without Streamlit UI) ---
                mae = mean_absolute_error(subset_model.y, subset_model.y_pred)
                mse = mean_squared_error(subset_model.y, subset_model.y_pred)
                rmse = np.sqrt(mse)
                r2 = r2_score(subset_model.y, subset_model.y_pred)
                adj_r2 = (1 - (1 - r2) * (n - 1) / (n - len(subset) - 1)) if n > len(subset) + 1 else r2

                # --- Equation or importance ---
                if hasattr(subset_model.model, 'coef_'):
                    # Use your generate_equation logic but without Streamlit output
                    intercept = subset_model.model.intercept_
                    pivot_values = subset_model.df[list(subset)].mean()
                    coefs = subset_model.model.coef_ / np.std(subset_model.X, axis=0)
                    terms = [f"({coef:.4f} * ({feat} - {pivot_values[feat]:.3f}))"
                            for coef, feat in zip(coefs, subset)]
                    equation = f"{self.target_column} = {intercept:.4f} + " + " + ".join(terms)
                elif hasattr(subset_model.model, 'feature_importances_'):
                    importances = dict(zip(subset, subset_model.model.feature_importances_))
                    equation = f"Feature importances: {importances}"
                else:
                    equation = "Equation/importance not available."

                # --- Collect results ---
                results.append({
                    'subset': len(results) + 1,
                    'features': list(subset),
                    'r2': r2 * 100,          # %
                    'adj_r2': adj_r2 * 100,  # %
                    'mae': mae,
                    'mse': mse,
                    'rmse': rmse,
                    'equation': equation,
                    'model_type': model_type,
                    'model': subset_model
                })

        # Sort by adjusted R² (descending)
        results = sorted(results, key=lambda x: x['adj_r2'], reverse=True)
        return results

    def prepare_subset_results_dataframe(self, results):
        feature_columns = self.feature_columns
        records = []
        for res in results:
            row = {
                'Adjusted R²': res['adj_r2'],
                'R²': res['r2'],
                'MSE': res['mse'],
                'MAE': res['mae'],
                'Equation': res['equation']
            }
            # Fill feature coefficients (standardized)
            model = res['model']
            if hasattr(model.model, 'coef_'):
                coefs = model.model.coef_ / model.X.std()
                for f, c in zip(res['features'], coefs):
                    row[f] = c
            records.append(row)

        df = pd.DataFrame(records)
        
        # Ensure all feature columns are present
        for feat in feature_columns:
            if feat not in df.columns:
                df[feat] = np.nan

        df = df[['Adjusted R²', 'R²'] + feature_columns + ['MSE', 'MAE', 'Equation']]
        return df.fillna(0)
    
    @staticmethod
    def color_coefficients(val, max_abs_val=None):
        if pd.isna(val):
            return ""
        
        # Default to raw scaling if no max provided
        if max_abs_val is None or max_abs_val == 0:
            max_abs_val = abs(val)
        
        # Base color by sign
        base_color = "0, 200, 0" if val > 0 else "200, 0, 0"
        
        # Normalize opacity (max 0.4)
        opacity = min(0.4, abs(val) / max_abs_val)
        
        return f"background-color: rgba({base_color}, {opacity:.2f});"


# Dashboard Frontend

st.set_page_config(page_title="Regression Dashboard", page_icon="⚡", initial_sidebar_state="auto", layout="wide")

st.title("📊 Data Visualization & 📉 Regression Model Dashboard")
st.subheader("📁 Upload Your Dataset")


# --- Styling Functions ---
def style_missing_and_outliers(df, visualizer, show_missing=True, show_outliers=True):
    styled_df = pd.DataFrame("", index=df.index, columns=df.columns)

    # Outlier detection
    if show_outliers:
        numeric_cols = df.select_dtypes(include='number').columns
        for col in numeric_cols:
            outlier_mask = visualizer.detect_outliers(df[col])
            for i in df.index:
                if outlier_mask[i]:
                    styled_df.at[i, col] += 'background-color: rgba(255,165,0,0.25); outline: 2px solid orange;'


    # Missing value detection
    if show_missing:
        for col in df.columns:
            for i in df.index:
                if pd.isna(df.at[i, col]):
                    styled_df.at[i, col] += 'background-color: rgba(255,0,0,0.25); outline: 2px solid red;'

    return styled_df

def mark_outliers(df, visualizer):
    outlier_flags = []
    for col in df.select_dtypes(include='number').columns:
        mask = visualizer.detect_outliers(df[col])
        outlier_flags.append(mask)

    if outlier_flags:
        combined_mask = np.logical_or.reduce(outlier_flags)
        df["Outlier"] = np.where(combined_mask, "Outlier", "Normal")
    else:
        df["Outlier"] = "Normal"

    return df.drop(columns="Outlier")


# --- File Upload Logic ---
uploaded_file = st.file_uploader("Choose a CSV or Excel file", type=["csv", "xlsx"])

if uploaded_file is not None:
    try:
        # Read the file
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith(".xlsx"):
            df = pd.read_excel(uploaded_file)
        else:
            st.error("Unsupported file format")
            st.stop()

        st.success("File uploaded successfully!")

        # Remove unnamed index columns
        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

        # Visualizer instance for outlier detection
        visualizer = Visualize(df)

        # Mark outliers (adds "Outlier" column)
        df = mark_outliers(df, visualizer)

        st.subheader("🔍 Data Preview")

        # Toggle options
        # st.subheader("🛠️ Highlight Options")
        col1, col2 = st.columns(2)
        show_missing = col1.checkbox("Highlight Missing Values", value=True)
        show_outliers = col2.checkbox("Highlight Outliers", value=True)

        # Styling
        styled = df.style.apply(lambda _: style_missing_and_outliers(df, visualizer, show_missing, show_outliers), axis=None)

        st.dataframe(styled, use_container_width=True)
        # st.write(styled)

        # Show dataset shape
        st.write(f"**Shape:** {df.shape[0]} rows × {df.shape[1]} columns")

        # Data types
        st.subheader("📊 Column Summary (Data Types & Missing Values)")

        # Get data types & Missing Values
        dtypes_df = df.dtypes.reset_index().rename(columns={'index': 'Column', 0: 'Data Type'})
        dtypes_df['Data Type'] = dtypes_df['Data Type'].astype(str)

        # Get missing values and percentages
        missing_vals = df.isnull().sum().reset_index()
        missing_vals.columns = ['Column', 'Missing Values']
        missing_vals['% Missing Values'] = ((missing_vals['Missing Values'] / len(df)) * 100).round(2)

        # Merge data type info with missing values
        summary_df = pd.merge(dtypes_df, missing_vals, on='Column', how='left')
        summary_df[['Missing Values', '% Missing Values']] = summary_df[['Missing Values', '% Missing Values']].fillna(0)
        summary_df['Missing Values'] = summary_df['Missing Values'].astype(int)

        # Display the summary table
        st.dataframe(summary_df.style.format({'% Missing Values': '{:.2f}%'}))

        col1, col2 = st.columns([1, 3])
        with col1:
            columns_to_drop = st.multiselect("Select irrelavant columns to delete", df.columns)
        
        if columns_to_drop:
            df = df.drop(columns=columns_to_drop)
            st.write("Updated Data:")
            st.dataframe(df)

        def handle_missing_values(df, strategy):
            if strategy == "Drop missing values":
                df = df.dropna()
            if strategy == "Fill with Zero":
                df = df.fillna(0)
            if strategy == "Fill with mean value":
                df = df.fillna(df.mean())
            if strategy == "Fill with median value":
                df = df.fillna(df.median())
            if strategy == "Fill with mode value":
                df = df.fillna(df.mode().iloc[0])
            return df

        col1, col2 = st.columns([1, 3])
        with col1:
            missing_value_strategy = st.selectbox("How to handle missing values?", 
                                                ["Drop missing values", "Fill with Zero", "Fill with mean value", "Fill with median value", "Fill with mode value"], index=None, placeholder="Choose an option")
        
        if missing_value_strategy:
            df = handle_missing_values(df, missing_value_strategy)
            
            st.write("Updated Data:")
            st.dataframe(df)

    except Exception as e:
        st.error(f"Error loading file: {e}")

else:
    st.info("Please upload a CSV or Excel file to proceed.")

# Check if the DataFrame is loaded
if 'df' in locals() or 'df' in globals():

    st.subheader("🌀 Select what you want")

    col1, col2 = st.columns([1, 3])
    with col1:
        Type = st.selectbox("To Visualize or to Predict",["Data Visualization", "Model Prediction"], index=None, placeholder="Choose an option")

    if Type == "Model Prediction":
        # Only show selection if data is successfully loaded
        if 'df' in locals() or 'df' in globals():

            viz = Visualize(df)

            col1, col2 = st.columns([1, 4])
            with col1:
                outlier_strategy = st.selectbox(
                    "How to handle outliers?",
                    ["Do nothing", "Remove outliers", "Replace with median", "Replace with mean"],
                    index=None,
                    placeholder="Choose an option"
                )

            def handle_outliers(df, outlier_strategy, viz):
                for col in df.select_dtypes(include=np.number).columns:
                    outlier_mask = viz.detect_outliers(df[col])
                    if outlier_strategy == "Remove outliers":
                        df = df[~outlier_mask]
                    elif outlier_strategy == "Replace with median":
                        median = df[col].median()
                        df.loc[outlier_mask, col] = median
                    elif outlier_strategy == "Replace with mean":
                        mean = df[col].mean()
                        df.loc[outlier_mask, col] = mean
                return df
            
            if outlier_strategy and outlier_strategy != "Do nothing":
                df = handle_outliers(df, outlier_strategy, viz)
                st.write("Update Data after handling outliers:")
                st.dataframe(df)

            st.subheader("🎯 Select Target and Feature Columns")
            
            # Identify numeric columns
            numeric_cols = [col for col in df.columns if np.issubdtype(df[col].dropna().dtype, np.number)]
            
            col1, col2 = st.columns([2, 3]) 
            with col1:
                st.markdown("#### 🧮 Inputs")
                # Select target column
                target_column = st.selectbox(
                    "Select the target column (dependent variable)", 
                    options=numeric_cols
                )
            
                # Select feature columns
                feature_columns = st.multiselect(
                    "Select one or more feature columns (independent variables)",
                    options=[col for col in numeric_cols if col != target_column]
                )
            
            with col2:
                st.markdown("#### 📊 Summary")
            
                if feature_columns:
                    st.write("**Target column:**")
                    st.code(target_column, language="text")
            
                    st.write(f"**Feature columns ({len(feature_columns)}):**")
                    for col in feature_columns:
                        st.code(col, language="text")
                else:
                    st.warning("Please select at least one feature column to proceed.")

            # Initialize session state
            if "model_trained" not in st.session_state:
                st.session_state.model_trained = False
            if "model_obj" not in st.session_state:
                st.session_state.model_obj = None
            if "prev_model_type" not in st.session_state:
                st.session_state.prev_model_type = None
            if "prev_params" not in st.session_state:
                st.session_state.prev_params = {}

            if feature_columns and target_column:
                col1, col2 = st.columns([1, 4])
            
                with col1:
                    with st.container():
                        model_type = st.selectbox(
                            "🏗️ Choose your Model", 
                            ["Linear Regression Model", "Ridge Regression Model", "Lasso Regression Model", 
                             "RandomForest Regression Model", "DecisionTree Regression Model"],
                            label_visibility="visible"
                        )
            
                        # Add spacing
                        st.markdown("")
            
                        # Place the button directly below the selectbox, same width
                        train_clicked = st.button("🚀 Train Model", use_container_width=True)
            
                with col2:
                    model_params = {}
            
                    if model_type in ["Ridge Regression Model", "Lasso Regression Model"]:
                        model_params['alpha'] = st.slider("Alpha", 0.01, 10.0, 1.0)
                    elif model_type == "RandomForest Regression Model":
                        model_params['n_estimators'] = st.slider("Number of estimators", 10, 500, 100)
                    elif model_type == "DecisionTree Regression Model":
                        model_params['max_depth'] = st.slider("Max depth", 1, 50, 10)
            
                # Reset training if config changes
                if (
                    model_type != st.session_state.prev_model_type or
                    model_params != st.session_state.prev_params
                ):
                    st.session_state.model_trained = False
            
                st.session_state.prev_model_type = model_type
                st.session_state.prev_params = model_params
            
                if train_clicked:
                    st.session_state.model_obj = Model(df, feature_columns, target_column)
                    st.session_state.model_obj.train_model(model_type=model_type, **model_params)
                    st.session_state.model_trained = True
                    st.success("✅ Model Trained Successfully!")
                    # st.balloons()

                # Check if model is trained before showing output options
                if st.session_state.get("model_trained"):
                    model_obj = st.session_state.model_obj

                    st.subheader("🧪 Choose Outputs to Display")

                    # Required session states
                    if "selected_outputs" not in st.session_state:
                        st.session_state.selected_outputs = []

                    if "ctrl_mode" not in st.session_state:
                        st.session_state.ctrl_mode = False
                    
                    st.session_state.ctrl_mode = st.checkbox("Simulate Ctrl Key Held (for multi-select)")

                    sections = [
                        ("Feature Importance Chart", "📊"),
                        ("Feature Importance Table", "🔢"),
                        ("Regression Equation", "🧮"),
                        ("Model Evaluation", "📈"),
                        ("Actual vs Predicted Plot", "📉"),
                        ("Residual Scatter with Histogram Plot", "📉"),
                        ("Residual Histogram", "📊"),
                        ("Residual Scatter Plot", "🌀"),
                        ("Learning Curve", "📚"),
                        ("Make Prediction with Custom Inputs", "🧠"),
                        # ("Run All Subset Regression", "🔁")
                    ]

                    # CSS for buttons with hover and active effect
                    st.markdown("""
                        <style>
                        .stButton > button {
                            background-color: transparent;
                            color: #0366d6;
                            border: 2px solid #0366d6;
                            padding: 0.4rem 1rem;
                            border-radius: 8px;
                            margin-bottom: 0.5rem;
                            width: 100%;
                            text-align: left;
                            font-weight: 600;
                            transition: all 0.2s ease-in-out;
                            white-space: nowrap;
                            overflow: hidden;
                            text-overflow: ellipsis;
                            cursor: pointer;
                        }
                        .stButton > button:hover {
                            background-color: #0366d6;
                            color: white;
                        }
                        .stButton.active > button {
                            background-color: #0366d6 !important;
                            color: white !important;
                            border-color: #024a9c !important;
                        }
                        </style>
                    """, unsafe_allow_html=True)

                    col1, col2 = st.columns([1, 4])

                    with col1:
                        for label, icon in sections:
                            key = f"btn_{label.replace(' ', '_')}"
                            is_selected = label in st.session_state.selected_outputs
                            clicked = st.button(f"{icon} {label}", key=key)

                            if clicked:
                                if st.session_state.ctrl_mode:
                                    # Multi-select logic
                                    if is_selected:
                                        st.session_state.selected_outputs.remove(label)
                                    else:
                                        st.session_state.selected_outputs.append(label)
                                else:
                                    # Single select logic
                                    if is_selected and len(st.session_state.selected_outputs) == 1:
                                        st.session_state.selected_outputs = []
                                    else:
                                        st.session_state.selected_outputs = [label]

                            # Active button highlight
                            if is_selected:
                                st.markdown(f"""
                                    <style>
                                    div[data-testid="stButton"][key="{key}"] > button {{
                                        background-color: #0366d6 !important;
                                        color: white !important;
                                        border-color: #024a9c !important;
                                    }}
                                    </style>
                                """, unsafe_allow_html=True)

                    with col2:
                        if st.session_state.selected_outputs:
                            for label in st.session_state.selected_outputs:
                            
                                if label == "Feature Importance Table":
                                    st.subheader("🔢 Feature Importance Table")
                                    model_obj.feature_importance_table()

                                if label == "Feature Importance Chart":
                                    st.subheader("📊 Feature Importance Chart")
                                    model_obj.feature_importance_plot()

                                if label == "Regression Equation":
                                    st.subheader("🧮 Regression Equation")
                                    model_obj.generate_equation()

                                if label == "Model Evaluation":
                                    st.subheader("📈 Model Evaluation")
                                    model_obj.evaluate_model()

                                if label == "Actual vs Predicted Plot":
                                    st.subheader("📉 Actual vs Predicted Plot")
                                    model_obj.plot_actual_vs_predicted()

                                if label == "Residual Scatter with Histogram Plot":
                                    st.subheader("📉 Residuals vs Predicted Scatter with Histogram")
                                    model_obj.residual_scatter_with_histogram()

                                if label == "Residual Histogram":
                                    st.subheader("📊 Residual Histogram")
                                    model_obj.residual_histogram()

                                if label == "Residual Scatter Plot":
                                    st.subheader("🌀 Residuals vs Predicted Scatter")
                                    model_obj.residual_scatter()

                                if label == "Learning Curve":
                                    st.subheader("📚 Learning Curve")
                                    model_obj.learning_curve()

                                if label == "Make Prediction with Custom Inputs":
                                    st.subheader("🧠 Make Prediction with Custom Inputs")
                                    model_obj.user_input_and_predict(model_obj, feature_columns)



                    st.markdown("## 🔁 All Subset Regression")

                    col1, col2 = st.columns([1, 3])
                    with col1:
                        run_subsets = st.button("Run All Subset Regression", use_container_width=True)
                    
                    if run_subsets:
                        all_subsets_results = model_obj.train_on_all_subsets(model_type=model_type, **model_params)
                        st.session_state.all_subsets_results = all_subsets_results
                        st.success("✅ All Subset Regression completed!")

                    if "all_subsets_results" in st.session_state:
                        results = st.session_state.all_subsets_results

                        st.markdown("## Subset Results with Interactive Plots")

                        for idx, result in enumerate(st.session_state.all_subsets_results):
                            subset_no = result['subset']
                            model = result['model']
                            feature = (", ".join(result['features']))

                            with st.expander(f"Subset :-  {feature}"):
                                # Show metrics
                                st.markdown(f"""
                                **Features**: {", ".join(result['features'])}  
                                **R²**: {round(result['r2'], 4)}  
                                **Adjusted R²**: {round(result['adj_r2'], 4)}  
                                **MAE**: {round(result['mae'], 4)}  
                                **MSE**: {round(result['mse'], 4)}  
                                **Equation**: `{result['equation']}`
                                """)

                                # Tab-based chart display
                                tab_labels = [
                                    "🔥 Feature Importance Plot",
                                    "📋 Feature Importance Table",
                                    "📈 Actual vs Predicted",
                                    "📊 Residual Scatter",
                                    "📉 Residual Histogram",
                                    "🟦 Residual Scatter + Histogram",
                                    "📐 Learning Curve"
                                ]
                                tabs = st.tabs(tab_labels)

                                with tabs[0]:
                                    model.feature_importance_plot()
                                with tabs[1]:
                                    model.feature_importance_table()
                                with tabs[2]:
                                    model.plot_actual_vs_predicted()
                                with tabs[3]:
                                    model.residual_scatter()
                                with tabs[4]:
                                    model.residual_histogram()
                                with tabs[5]:
                                    model.residual_scatter_with_histogram()
                                with tabs[6]:
                                    model.learning_curve()

                        st.markdown("## 📊 Subset Regression Summary Table")

                        if "all_subsets_results" in st.session_state:
                            results = st.session_state.all_subsets_results
                            feature_columns = model_obj.feature_columns

                            # Convert results to DataFrame
                            df = model_obj.prepare_subset_results_dataframe(results)

                            # Find max absolute coefficient in these columns
                            max_abs_val = df[feature_columns].abs().max().max()

                            # Apply styling with normalized opacity
                            styled_df = df.style \
                                .map(lambda v: Model.color_coefficients(v, max_abs_val), subset=feature_columns) \
                                .format({
                                    'Adjusted R²': '{:.4f}',
                                    'R²': '{:.4f}',
                                    'MSE': '{:.4f}',
                                    'MAE': '{:.4f}',
                                })

                            st.dataframe(styled_df, use_container_width=True, hide_index=True)

                            # st.download_button("📥 Download Results as CSV", df.to_csv(index=False), file_name="subset_results.csv")



    elif Type == "Data Visualization":

        if 'df' in locals() or 'df' in globals():
            viz = Visualize(df)

        # Dropdown to select plot type
        col1, col2 = st.columns([1, 3])
        with col1:
            plot_type = st.selectbox("📊 Select Visualization Type", options=[
                "Scatter with Marginal Histogram",
                "Pair Plot",
                "Correlation Heatmap",
                "Interactive Relation Scatter",
                "Interactive Relation Box"
            ])
        
        # Display selected plot
        if plot_type == "Scatter with Marginal Histogram":
            viz.scatter_with_marginal_histogram()
        if plot_type == "Pair Plot":
            viz.pair_plot()
        if plot_type == "Correlation Heatmap":
            viz.correlation_heatmap()
        if plot_type == "Interactive Relation Scatter":
            viz.interactive_relation_scatter()
        if plot_type == "Interactive Relation Box":
            viz.interactive_relation_box()


            
