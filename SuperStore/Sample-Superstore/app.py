from pathlib import Path
import warnings

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.stats.diagnostic import acorr_ljungbox


warnings.filterwarnings("ignore", category=UserWarning)

st.set_page_config(
    page_title="Superstore Sales Intelligence",
    page_icon="📈",
    layout="wide"
)


@st.cache_data
def load_data():

    file_path = Path(__file__).parent / "data" / "Sample - Superstore.csv"

    df = pd.read_csv(file_path, encoding="latin1")

    df["Order Date"] = pd.to_datetime(df["Order Date"])

    if df["Sales"].isnull().any():
        raise ValueError("The Sales column contains missing values.")

    monthly_sales = (
        df.set_index("Order Date")["Sales"]
        .resample("MS")
        .sum()
        .reset_index()
    )

    monthly_sales.columns = ["Month", "Sales"]

    if len(monthly_sales) < 36:
        raise ValueError("At least 36 months of sales data are needed.")

    return monthly_sales


@st.cache_resource
def load_models():

    models_dir = Path(__file__).parent / "models"

    model_files = {
        "Linear Regression": "linear_regression_model.pkl",
        "Holt-Winters": "holt_winters_model.pkl",
        "Seasonal Naive": "seasonal_naive_model.pkl",
        "SARIMA": "sarima_model.pkl"
    }

    missing_files = [
        filename
        for filename in model_files.values()
        if not (models_dir / filename).exists()
    ]

    if missing_files:
        raise FileNotFoundError(
            "Missing model files: " + ", ".join(missing_files)
        )

    models = {}

    for model_name, filename in model_files.items():

        models[model_name] = joblib.load(
            models_dir / filename
        )

    return models


def generate_forecast(model_name, models, monthly_sales):

    periods = 6
    sales_count = len(monthly_sales)

    if model_name == "Linear Regression":

        future_months = np.arange(
            sales_count + 1,
            sales_count + periods + 1
        ).reshape(-1, 1)

        forecast_values = models[model_name].predict(
            future_months
        )

    elif model_name == "Holt-Winters":

        forecast_values = models[model_name].forecast(periods)

    elif model_name == "Seasonal Naive":

        last_season = np.asarray(
            models[model_name]["last_season"]
        ).reshape(-1)

        if len(last_season) == 0:
            raise ValueError(
                "Seasonal Naive model has no saved seasonal values."
            )

        forecast_values = np.tile(
            last_season,
            int(np.ceil(periods / len(last_season)))
        )[:periods]

    elif model_name == "SARIMA":

        forecast_values = models[model_name].forecast(periods)

    else:
        raise ValueError(
            f"Unknown forecasting model: {model_name}"
        )

    forecast_values = np.asarray(
        forecast_values
    ).reshape(-1)

    if len(forecast_values) != periods:
        raise ValueError(
            "The selected model did not return six forecast values."
        )

    if not np.isfinite(forecast_values).all():
        raise ValueError(
            "The selected model returned invalid forecast values."
        )

    forecast_dates = pd.date_range(
        start=monthly_sales["Month"].max() + pd.offsets.MonthBegin(1),
        periods=periods,
        freq="MS"
    )

    return pd.DataFrame({
        "Month": forecast_dates,
        "Forecast Sales": forecast_values
    })


@st.cache_data
def run_forecasting(monthly_sales):

    sales = monthly_sales["Sales"].reset_index(drop=True)

    initial_train_size = 24
    forecast_horizon = 3

    sarima_configs = [
        {
            "order": (1, 1, 1),
            "seasonal_order": (1, 1, 1, 12)
        },
        {
            "order": (1, 1, 0),
            "seasonal_order": (1, 1, 1, 12)
        },
        {
            "order": (0, 1, 1),
            "seasonal_order": (1, 1, 1, 12)
        },
        {
            "order": (1, 1, 1),
            "seasonal_order": (0, 1, 1, 12)
        }
    ]

    actual_values = []
    linear_values = []
    hw_values = []
    seasonal_values = []
    months = []

    sarima_results = {
        str(config): []
        for config in sarima_configs
    }

    for split_index in range(
        initial_train_size,
        len(sales) - forecast_horizon + 1,
        forecast_horizon
    ):

        train = sales.iloc[:split_index]

        test = sales.iloc[
            split_index:split_index + forecast_horizon
        ]

        X_train = np.arange(
            len(train)
        ).reshape(-1, 1)

        X_test = np.arange(
            len(train),
            len(train) + len(test)
        ).reshape(-1, 1)

        linear_model = LinearRegression()

        linear_model.fit(
            X_train,
            train
        )

        linear_predictions = linear_model.predict(X_test)

        hw_model = ExponentialSmoothing(
            train,
            trend="add",
            seasonal="add",
            seasonal_periods=12,
            initialization_method="estimated"
        ).fit()

        hw_predictions = hw_model.forecast(len(test))

        seasonal_predictions = train.iloc[
            -12:-12 + len(test)
        ].to_numpy()

        for config in sarima_configs:

            config_key = str(config)

            try:

                sarima_model = SARIMAX(
                    train,
                    order=config["order"],
                    seasonal_order=config["seasonal_order"],
                    enforce_stationarity=False,
                    enforce_invertibility=False
                ).fit(
                    disp=False,
                    maxiter=200
                )

                predictions = sarima_model.forecast(len(test))

                sarima_results[config_key].extend(
                    predictions.tolist()
                )

            except Exception:

                sarima_results[config_key].extend(
                    [np.nan] * len(test)
                )

        actual_values.extend(test.tolist())

        linear_values.extend(
            linear_predictions.tolist()
        )

        hw_values.extend(
            hw_predictions.tolist()
        )

        seasonal_values.extend(
            seasonal_predictions.tolist()
        )

        months.extend(
            monthly_sales["Month"].iloc[
                split_index:split_index + forecast_horizon
            ].tolist()
        )

    actual = np.array(actual_values)

    linear_predictions = np.array(linear_values)
    hw_predictions = np.array(hw_values)
    seasonal_predictions = np.array(seasonal_values)

    def calculate_metrics(actual, predicted):

        valid = np.isfinite(predicted)

        actual_valid = actual[valid]
        predicted_valid = predicted[valid]

        if len(actual_valid) == 0:

            return {
                "MAE": np.nan,
                "RMSE": np.nan,
                "MAPE": np.nan
            }

        valid_non_zero = actual_valid != 0

        mape = (
            np.mean(
                np.abs(
                    (
                        actual_valid[valid_non_zero]
                        - predicted_valid[valid_non_zero]
                    )
                    / actual_valid[valid_non_zero]
                )
            ) * 100
            if valid_non_zero.any()
            else np.nan
        )

        return {
            "MAE": mean_absolute_error(
                actual_valid,
                predicted_valid
            ),
            "RMSE": np.sqrt(
                mean_squared_error(
                    actual_valid,
                    predicted_valid
                )
            ),
            "MAPE": mape
        }

    model_rows = [
        {
            "Model": "Linear Regression",
            **calculate_metrics(actual, linear_predictions)
        },
        {
            "Model": "Holt-Winters",
            **calculate_metrics(actual, hw_predictions)
        },
        {
            "Model": "Seasonal Naive",
            **calculate_metrics(actual, seasonal_predictions)
        }
    ]

    sarima_config_metrics = []

    for config in sarima_configs:

        config_key = str(config)

        predictions = np.array(
            sarima_results[config_key]
        )

        result = calculate_metrics(
            actual,
            predictions
        )

        sarima_config_metrics.append({
            "Configuration": config_key,
            "Order": str(config["order"]),
            "Seasonal Order": str(config["seasonal_order"]),
            **result
        })

    sarima_tuning = pd.DataFrame(
        sarima_config_metrics
    )

    valid_configs = sarima_tuning.dropna(
        subset=["MAPE"]
    )

    if valid_configs.empty:

        raise ValueError(
            "All SARIMA configurations failed. "
            "Check the model settings and data."
        )

    best_config_row = valid_configs.loc[
        valid_configs["MAPE"].idxmin()
    ]

    best_config = next(
        config
        for config in sarima_configs
        if str(config) == best_config_row["Configuration"]
    )

    best_sarima_predictions = np.array(
        sarima_results[str(best_config)]
    )

    model_rows.append({
        "Model": "SARIMA (Tuned)",
        **calculate_metrics(
            actual,
            best_sarima_predictions
        )
    })

    metrics = pd.DataFrame(model_rows)

    final_sarima_model = SARIMAX(
        sales,
        order=best_config["order"],
        seasonal_order=best_config["seasonal_order"],
        enforce_stationarity=False,
        enforce_invertibility=False
    ).fit(
        disp=False,
        maxiter=200
    )

    sarima_residuals = pd.Series(
        final_sarima_model.resid
    ).replace(
        [np.inf, -np.inf],
        np.nan
    ).dropna()

    residual_diagnostics = pd.DataFrame({
        "Residual": sarima_residuals.values
    })

    comparison = pd.DataFrame({
        "Month": months,
        "Actual Sales": actual,
        "Linear Regression": linear_predictions,
        "Holt-Winters": hw_predictions,
        "Seasonal Naive": seasonal_predictions,
        "SARIMA (Tuned)": best_sarima_predictions
    })

    return (
        metrics,
        comparison,
        sarima_tuning,
        best_config,
        residual_diagnostics
    )


st.title("SUPERSTORE / SALES INTELLIGENCE")

st.caption(
    "Historical performance  /  Forecasting  /  Business insights"
)


try:

    monthly_sales = load_data()

    saved_models = load_models()

    (
        metrics,
        comparison,
        sarima_tuning,
        best_sarima_config,
        residual_diagnostics
    ) = run_forecasting(monthly_sales)

except Exception as error:

    st.error(
        f"Unable to load data, models or forecasts: {error}"
    )

    st.stop()


best_model = metrics.loc[
    metrics["MAPE"].idxmin(),
    "Model"
]

st.sidebar.title("NAVIGATION")

page = st.sidebar.radio(
    "Explore",
    [
        "Overview",
        "Sales Forecast",
        "Model Comparison"
    ]
)

st.sidebar.divider()

st.sidebar.caption("BEST VALIDATED MODEL")

st.sidebar.subheader(best_model)

st.sidebar.caption(
    "Selected using historical walk-forward validation."
)


if page == "Overview":

    st.subheader("Sales overview")

    latest_sales = monthly_sales["Sales"].iloc[-1]

    previous_sales = monthly_sales["Sales"].iloc[-2]

    if previous_sales != 0:

        monthly_growth = (
            (latest_sales - previous_sales) / previous_sales
        ) * 100

    else:

        monthly_growth = 0

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Total sales",
        f"{monthly_sales['Sales'].sum():,.0f}"
    )

    col2.metric(
        "Average monthly sales",
        f"{monthly_sales['Sales'].mean():,.0f}"
    )

    col3.metric(
        "Monthly observations",
        len(monthly_sales)
    )

    col4.metric(
        "Monthly sales growth",
        f"{monthly_growth:+.2f}%"
    )

    st.caption(
        f"Compared with the previous month | "
        f"Latest month: "
        f"{monthly_sales['Month'].iloc[-1].strftime('%B %Y')}"
    )

    st.divider()

    st.subheader("Historical monthly sales")

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=monthly_sales["Month"],
        y=monthly_sales["Sales"],
        mode="lines",
        name="Monthly sales",
        line=dict(color="#F28C45", width=3),
        fill="tozeroy",
        fillcolor="rgba(242, 140, 69, 0.08)",
        hovertemplate=(
            "<b>%{x|%B %Y}</b><br>"
            "Sales: %{y:,.2f}<extra></extra>"
        )
    ))

    fig.update_layout(
        template="plotly_dark",
        height=450,
        xaxis_title="",
        yaxis_title="Sales",
        hovermode="x unified",
        showlegend=False,
        margin=dict(l=10, r=10, t=20, b=10)
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.divider()

    st.subheader("Business insights")

    highest_month = monthly_sales.loc[
        monthly_sales["Sales"].idxmax()
    ]

    lowest_month = monthly_sales.loc[
        monthly_sales["Sales"].idxmin()
    ]

    col1, col2 = st.columns(2)

    col1.metric(
        "Highest sales month",
        highest_month["Month"].strftime("%B %Y"),
        f"{highest_month['Sales']:,.2f}"
    )

    col2.metric(
        "Lowest sales month",
        lowest_month["Month"].strftime("%B %Y"),
        f"{lowest_month['Sales']:,.2f}"
    )


elif page == "Sales Forecast":

    st.subheader("Six-month sales forecast")

    selected_model = st.selectbox(
        "Select forecasting model",
        [
            "Holt-Winters",
            "Linear Regression",
            "Seasonal Naive",
            "SARIMA"
        ]
    )

    try:

        forecast = generate_forecast(
            selected_model,
            saved_models,
            monthly_sales
        )

    except Exception as error:

        st.error(
            f"Unable to generate forecast: {error}"
        )

        st.stop()

    metric_model_names = {
        "Linear Regression": "Linear Regression",
        "Holt-Winters": "Holt-Winters",
        "Seasonal Naive": "Seasonal Naive",
        "SARIMA": "SARIMA (Tuned)"
    }

    validation_model = metric_model_names[selected_model]

    selected_mape = metrics.loc[
        metrics["Model"] == validation_model,
        "MAPE"
    ].iloc[0]

    st.caption(
        f"Six-month forecast generated using the saved "
        f"{selected_model} model."
    )

    col1, col2 = st.columns(2)

    col1.metric(
        "Forecast total",
        f"{forecast['Forecast Sales'].sum():,.2f}"
    )

    col2.metric(
        "Average forecast",
        f"{forecast['Forecast Sales'].mean():,.2f}"
    )

    st.divider()

    st.subheader("Forecast insights")

    first_forecast = forecast["Forecast Sales"].iloc[0]

    last_forecast = forecast["Forecast Sales"].iloc[-1]

    peak_forecast = forecast.loc[
        forecast["Forecast Sales"].idxmax()
    ]

    latest_actual = monthly_sales["Sales"].iloc[-1]

    forecast_change = (
        (first_forecast - latest_actual) / latest_actual
    ) * 100 if latest_actual != 0 else 0

    forecast_trend = (
        (last_forecast - first_forecast) / first_forecast
    ) * 100 if first_forecast != 0 else 0

    if forecast_trend > 0:

        trend_label = "Upward"

        trend_description = (
            f"Forecast sales increase by {forecast_trend:.2f}% "
            "across the six-month period."
        )

    elif forecast_trend < 0:

        trend_label = "Downward"

        trend_description = (
            f"Forecast sales decrease by {abs(forecast_trend):.2f}% "
            "across the six-month period."
        )

    else:

        trend_label = "Stable"

        trend_description = (
            "Forecast sales remain unchanged across the six-month period."
        )

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Forecast direction",
        trend_label,
        f"{forecast_trend:+.2f}%"
    )

    col2.metric(
        "Peak forecast month",
        peak_forecast["Month"].strftime("%B %Y"),
        f"{peak_forecast['Forecast Sales']:,.2f}"
    )

    col3.metric(
        "Change from latest actual",
        f"{forecast_change:+.2f}%"
    )

    st.caption(trend_description)

    if pd.notna(selected_mape):

        st.info(
            f"{selected_model} recorded a MAPE of "
            f"{selected_mape:.2f}% during historical "
            "walk-forward validation. Future forecast errors may differ."
        )

    else:

        st.warning(
            "Historical validation MAPE is unavailable for this model."
        )

    st.divider()

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=monthly_sales["Month"],
        y=monthly_sales["Sales"],
        mode="lines",
        name="Historical sales",
        line=dict(color="#AEB6BF", width=2)
    ))

    fig.add_trace(go.Scatter(
        x=forecast["Month"],
        y=forecast["Forecast Sales"],
        mode="lines+markers",
        name=f"{selected_model} forecast",
        line=dict(color="#F28C45", dash="dash", width=3),
        marker=dict(size=8)
    ))

    fig.update_layout(
        template="plotly_dark",
        height=500,
        xaxis_title="Month",
        yaxis_title="Sales",
        hovermode="x unified"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.subheader("Forecast values")

    st.dataframe(
        forecast.style.format({
            "Forecast Sales": "{:,.2f}"
        }),
        use_container_width=True,
        hide_index=True
    )

    st.download_button(
        "Download forecast",
        forecast.to_csv(index=False),
        file_name=(
            f"{selected_model.lower().replace(' ', '_')}_forecast.csv"
        ),
        mime="text/csv"
    )


elif page == "Model Comparison":

    st.subheader("Forecasting model performance")

    st.caption(
        "Models are evaluated using expanding-window time-series "
        "validation across multiple three-month forecasting periods."
    )

    st.dataframe(
        metrics.style.format({
            "MAE": "{:,.2f}",
            "RMSE": "{:,.2f}",
            "MAPE": "{:.2f}%"
        }),
        use_container_width=True,
        hide_index=True
    )

    st.success(
        f"Best model based on MAPE: {best_model}"
    )

    st.divider()

    st.subheader("SARIMA parameter tuning")

    st.caption(
        "Four SARIMA configurations were tested using the same "
        "walk-forward validation periods."
    )

    st.dataframe(
        sarima_tuning[
            ["Order", "Seasonal Order", "MAE", "RMSE", "MAPE"]
        ].style.format({
            "MAE": "{:,.2f}",
            "RMSE": "{:,.2f}",
            "MAPE": "{:.2f}%"
        }),
        use_container_width=True,
        hide_index=True
    )

    st.info(
        f"Selected SARIMA configuration: "
        f"order={best_sarima_config['order']}, "
        f"seasonal_order={best_sarima_config['seasonal_order']}"
    )

    st.divider()

    st.subheader("Model comparison — MAPE")

    model_colors = {
        "Linear Regression": "#777D80",
        "Holt-Winters": "#F28C45",
        "Seasonal Naive": "#5B9BD5",
        "SARIMA (Tuned)": "#8E7CC3"
    }

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=metrics["Model"],
        y=metrics["MAPE"],
        marker_color=[
            model_colors[model]
            for model in metrics["Model"]
        ],
        text=metrics["MAPE"].round(2).astype(str) + "%",
        textposition="outside"
    ))

    fig.update_layout(
        template="plotly_dark",
        yaxis_title="MAPE (%)",
        height=400,
        showlegend=False
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.divider()

    st.subheader("Actual vs predicted sales")

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=comparison["Month"],
        y=comparison["Actual Sales"],
        mode="lines+markers",
        name="Actual sales",
        line=dict(color="#F2F2F2", width=3)
    ))

    prediction_colors = {
        "Linear Regression": "#777D80",
        "Holt-Winters": "#F28C45",
        "Seasonal Naive": "#5B9BD5",
        "SARIMA (Tuned)": "#8E7CC3"
    }

    for model_name, color in prediction_colors.items():

        fig.add_trace(go.Scatter(
            x=comparison["Month"],
            y=comparison[model_name],
            mode="lines",
            name=model_name,
            line=dict(color=color, width=2)
        ))

    fig.update_layout(
        template="plotly_dark",
        height=450,
        xaxis_title="Month",
        yaxis_title="Sales",
        hovermode="x unified"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.divider()

    st.subheader("SARIMA residual diagnostics")

    residuals = residual_diagnostics["Residual"].astype(float)

    col1, col2 = st.columns(2)

    col1.metric(
        "Mean residual",
        f"{residuals.mean():,.2f}"
    )

    col2.metric(
        "Residual standard deviation",
        f"{residuals.std():,.2f}"
    )

    st.caption(
        "These are in-sample residuals from the SARIMA model fitted "
        "to the full historical dataset. They are separate from the "
        "walk-forward validation forecast errors."
    )

    st.divider()

    st.subheader("Residual distribution")

    fig = go.Figure()

    fig.add_trace(go.Histogram(
        x=residuals,
        nbinsx=20,
        marker_color="#F28C45",
        name="Residual distribution"
    ))

    fig.update_layout(
        template="plotly_dark",
        height=350,
        xaxis_title="Residual",
        yaxis_title="Frequency",
        showlegend=False
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.subheader("Residuals over time")

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        y=residuals,
        mode="lines+markers",
        name="SARIMA residuals",
        line=dict(color="#F28C45", width=2)
    ))

    fig.add_hline(
        y=0,
        line_dash="dash",
        line_color="white"
    )

    fig.update_layout(
        template="plotly_dark",
        height=350,
        xaxis_title="Observation",
        yaxis_title="Residual",
        showlegend=False
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.caption(
        "Residuals should ideally fluctuate around zero without "
        "a clear trend, repeating seasonal pattern or systematic structure."
    )

    st.divider()

    st.subheader("Residual autocorrelation")

    max_lags = min(
        12,
        len(residuals) // 2 - 1
    )

    if max_lags >= 1:

        acf_values = acf(
            residuals,
            nlags=max_lags,
            fft=False
        )

        pacf_values = pacf(
            residuals,
            nlags=max_lags,
            method="ywm"
        )

        confidence_limit = 1.96 / np.sqrt(len(residuals))

        col1, col2 = st.columns(2)

        with col1:

            st.markdown("**Autocorrelation Function (ACF)**")

            acf_fig = go.Figure()

            acf_fig.add_trace(go.Bar(
                x=list(range(len(acf_values))),
                y=acf_values,
                marker_color="#F28C45",
                name="ACF"
            ))

            acf_fig.add_hline(
                y=confidence_limit,
                line_dash="dash",
                line_color="#F5D76E"
            )

            acf_fig.add_hline(
                y=-confidence_limit,
                line_dash="dash",
                line_color="#F5D76E"
            )

            acf_fig.add_hline(
                y=0,
                line_color="white",
                line_width=1
            )

            acf_fig.update_layout(
                template="plotly_dark",
                height=350,
                xaxis_title="Lag",
                yaxis_title="Autocorrelation",
                showlegend=False
            )

            st.plotly_chart(
                acf_fig,
                use_container_width=True
            )

        with col2:

            st.markdown("**Partial Autocorrelation Function (PACF)**")

            pacf_fig = go.Figure()

            pacf_fig.add_trace(go.Bar(
                x=list(range(len(pacf_values))),
                y=pacf_values,
                marker_color="#5B9BD5",
                name="PACF"
            ))

            pacf_fig.add_hline(
                y=confidence_limit,
                line_dash="dash",
                line_color="#F5D76E"
            )

            pacf_fig.add_hline(
                y=-confidence_limit,
                line_dash="dash",
                line_color="#F5D76E"
            )

            pacf_fig.add_hline(
                y=0,
                line_color="white",
                line_width=1
            )

            pacf_fig.update_layout(
                template="plotly_dark",
                height=350,
                xaxis_title="Lag",
                yaxis_title="Partial autocorrelation",
                showlegend=False
            )

            st.plotly_chart(
                pacf_fig,
                use_container_width=True
            )

        st.caption(
            "The dashed lines show approximate 95% confidence limits. "
            "Bars outside these limits may indicate statistically "
            "significant autocorrelation at individual lags."
        )

    st.divider()

    st.subheader("Ljung–Box test")

    try:

        model_df = (
            best_sarima_config["order"][0]
            + best_sarima_config["order"][2]
            + best_sarima_config["seasonal_order"][0]
            + best_sarima_config["seasonal_order"][2]
        )

        test_lag = min(
            12,
            len(residuals) // 2 - 1
        )

        if test_lag <= model_df:

            st.warning(
                "There are not enough residual observations for the "
                "selected Ljung–Box test lag and model degrees of freedom."
            )

        else:

            ljung_box_results = acorr_ljungbox(
                residuals,
                lags=[test_lag],
                model_df=model_df,
                return_df=True
            )

            lb_statistic = ljung_box_results["lb_stat"].iloc[0]

            lb_pvalue = ljung_box_results["lb_pvalue"].iloc[0]

            col1, col2, col3 = st.columns(3)

            col1.metric(
                "Ljung–Box statistic",
                f"{lb_statistic:.4f}"
            )

            col2.metric(
                "P-value",
                f"{lb_pvalue:.4f}"
            )

            col3.metric(
                "Test lag",
                test_lag
            )

            if lb_pvalue < 0.05:

                st.warning(
                    "The p-value is below 0.05. Reject the null hypothesis. "
                    "There is evidence of remaining residual autocorrelation."
                )

            else:

                st.success(
                    "The p-value is at least 0.05. There is insufficient "
                    "evidence to reject the null hypothesis of no residual "
                    "autocorrelation at the tested lags."
                )

            st.caption(
                "Null hypothesis (H₀): Residuals have no autocorrelation "
                "up to the tested lag. Alternative hypothesis (H₁): "
                "Residuals contain autocorrelation. Significance level: 5%."
            )

    except Exception as error:

        st.warning(
            f"Could not complete the Ljung–Box test: {error}"
        )

    st.divider()

    st.download_button(
        "Download validation predictions",
        comparison.to_csv(index=False),
        file_name="model_validation_predictions.csv",
        mime="text/csv"
    )


st.divider()

st.caption("SUPERSTORE SALES FORECASTING PROJECT")