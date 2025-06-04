"""
Interactive Model Comparison Dashboard for Weather Regulation Prediction

This module creates comprehensive dashboards for comparing multiple models,
analyzing their performance, and providing insights for decision making.
"""

import dash
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import dash_table, dcc, html
from dash.dependencies import Input, Output, State
from plotly.subplots import make_subplots

from results.results_manager import ResultsManager
from visualization.plots import ModelVisualizer, WeatherVisualizer


class ModelComparisonDashboard:
    """Interactive dashboard for model comparison and analysis"""

    def __init__(self, results_path: str = "./results"):
        """Initialize dashboard"""
        self.results_manager = ResultsManager(results_path)
        self.model_visualizer = ModelVisualizer()
        self.weather_visualizer = WeatherVisualizer()

        # Initialize Dash app
        self.app = dash.Dash(
            __name__, external_stylesheets=[dbc.themes.BOOTSTRAP], suppress_callback_exceptions=True
        )

        # Color scheme
        self.colors = {
            "background": "#f8f9fa",
            "text": "#212529",
            "primary": "#0d6efd",
            "secondary": "#6c757d",
            "success": "#198754",
            "danger": "#dc3545",
            "warning": "#ffc107",
            "info": "#0dcaf0",
        }

    def create_layout(self):
        """Create dashboard layout"""
        self.app.layout = dbc.Container(
            [
                # Header
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                html.H1(
                                    "Weather Regulation Prediction - Model Comparison Dashboard",
                                    className="text-center mb-4",
                                ),
                                html.Hr(),
                            ]
                        )
                    ]
                ),
                # Experiment selector
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                html.Label("Select Experiments to Compare:"),
                                dcc.Dropdown(
                                    id="experiment-selector",
                                    options=self._get_experiment_options(),
                                    value=[],
                                    multi=True,
                                    placeholder="Select one or more experiments...",
                                ),
                            ],
                            width=8,
                        ),
                        dbc.Col(
                            [
                                html.Label("Comparison Metric:"),
                                dcc.Dropdown(
                                    id="metric-selector",
                                    options=[
                                        {"label": "Accuracy", "value": "test_accuracy"},
                                        {"label": "F1 Score", "value": "test_f1"},
                                        {"label": "Precision", "value": "test_precision"},
                                        {"label": "Recall", "value": "test_recall"},
                                        {"label": "AUC", "value": "test_auc"},
                                    ],
                                    value="test_accuracy",
                                ),
                            ],
                            width=4,
                        ),
                    ],
                    className="mb-4",
                ),
                # Main content tabs
                dcc.Tabs(
                    id="main-tabs",
                    value="overview",
                    children=[
                        dcc.Tab(label="Overview", value="overview"),
                        dcc.Tab(label="Model Performance", value="performance"),
                        dcc.Tab(label="Detailed Comparison", value="comparison"),
                        dcc.Tab(label="Feature Analysis", value="features"),
                        dcc.Tab(label="Training Analysis", value="training"),
                        dcc.Tab(label="Weather Analysis", value="weather"),
                        dcc.Tab(label="Reports", value="reports"),
                    ],
                ),
                # Tab content
                html.Div(id="tab-content", className="mt-4"),
                # Hidden divs for storing data
                dcc.Store(id="experiment-data"),
                dcc.Store(id="comparison-data"),
            ],
            fluid=True,
        )

    def _get_experiment_options(self):
        """Get available experiments for dropdown"""
        experiments_df = self.results_manager.list_experiments()

        if len(experiments_df) == 0:
            return []

        options = []
        for _, exp in experiments_df.iterrows():
            options.append(
                {
                    "label": f"{exp['name']} ({exp['timestamp']}) - Best: {exp['best_accuracy']:.3f}",
                    "value": exp["experiment_id"],
                }
            )

        return options

    def setup_callbacks(self):
        """Setup dashboard callbacks"""

        @self.app.callback(
            Output("experiment-data", "data"),
            Output("comparison-data", "data"),
            Input("experiment-selector", "value"),
            Input("metric-selector", "value"),
        )
        def load_experiment_data(experiment_ids, metric):
            """Load experiment data when selection changes"""
            if not experiment_ids:
                return None, None

            experiments = {}
            for exp_id in experiment_ids:
                exp = self.results_manager.load_experiment_result(exp_id)
                if exp:
                    experiments[exp_id] = exp

            # Create comparison data
            comparison_df = self.results_manager.compare_experiments(experiment_ids, metric)

            # Convert to JSON-serializable format
            exp_data = {}
            for exp_id, exp in experiments.items():
                exp_data[exp_id] = {
                    "name": exp.experiment_name,
                    "best_model": exp.best_model,
                    "best_accuracy": exp.best_accuracy,
                    "models": list(exp.model_results.keys()),
                }

            return exp_data, comparison_df.to_dict() if comparison_df is not None else None

        @self.app.callback(
            Output("tab-content", "children"),
            Input("main-tabs", "value"),
            State("experiment-data", "data"),
            State("comparison-data", "data"),
            State("experiment-selector", "value"),
            State("metric-selector", "value"),
        )
        def render_tab_content(active_tab, exp_data, comp_data, exp_ids, metric):
            """Render content based on selected tab"""

            if active_tab == "overview":
                return self._render_overview(exp_data, comp_data)

            elif active_tab == "performance":
                return self._render_performance(exp_ids, metric)

            elif active_tab == "comparison":
                return self._render_comparison(exp_ids)

            elif active_tab == "features":
                return self._render_features(exp_ids)

            elif active_tab == "training":
                return self._render_training(exp_ids)

            elif active_tab == "weather":
                return self._render_weather(exp_ids)

            elif active_tab == "reports":
                return self._render_reports(exp_ids)

            return html.Div("Select a tab to view content")

    def _render_overview(self, exp_data, comp_data):
        """Render overview tab"""
        if not exp_data:
            return html.Div("Please select experiments to compare", className="text-center mt-5")

        # Summary cards
        cards = []
        for _, data in exp_data.items():
            card = dbc.Card(
                [
                    dbc.CardHeader(html.H5(data["name"])),
                    dbc.CardBody(
                        [
                            html.P(f"Best Model: {data['best_model']}"),
                            html.P(f"Best Accuracy: {data['best_accuracy']:.3f}"),
                            html.P(f"Models Trained: {len(data['models'])}"),
                        ]
                    ),
                ],
                className="mb-3",
            )
            cards.append(dbc.Col(card, width=4))

        # Comparison table
        if comp_data:
            comp_df = pd.DataFrame.from_dict(comp_data)
            table = dash_table.DataTable(
                data=comp_df.reset_index().to_dict("records"),
                columns=[{"name": i, "id": i} for i in comp_df.reset_index().columns],
                style_cell={"textAlign": "left"},
                style_data_conditional=[
                    {"if": {"row_index": "odd"}, "backgroundColor": "rgb(248, 248, 248)"}
                ],
                style_header={"backgroundColor": "rgb(230, 230, 230)", "fontWeight": "bold"},
            )
        else:
            table = html.Div("No comparison data available")

        return html.Div(
            [
                html.H3("Experiment Overview"),
                dbc.Row(cards),
                html.Hr(),
                html.H4("Model Performance Comparison"),
                table,
            ]
        )

    def _render_performance(self, exp_ids, metric):
        """Render performance comparison tab"""
        if not exp_ids:
            return html.Div("Please select experiments to compare", className="text-center mt-5")

        # Load experiments
        all_results = []
        for exp_id in exp_ids:
            exp = self.results_manager.load_experiment_result(exp_id)
            if exp and exp.comparison_metrics is not None:
                df = exp.comparison_metrics.copy()
                df["experiment"] = exp.experiment_name
                all_results.append(df)

        if not all_results:
            return html.Div("No performance data available")

        results_df = pd.concat(all_results)

        # Create visualizations
        # 1. Bar chart comparison
        fig_bar = px.bar(
            results_df,
            x="model",
            y=metric,
            color="experiment",
            barmode="group",
            title=f'{metric.replace("_", " ").title()} Comparison',
            height=500,
        )

        # 2. Radar chart
        metrics = ["accuracy", "precision", "recall", "f1_score"]
        radar_data = []

        for exp_name in results_df["experiment"].unique():
            exp_data = results_df[results_df["experiment"] == exp_name]
            for _, row in exp_data.iterrows():
                values = [row.get(m, 0) for m in metrics]
                radar_data.append(
                    {
                        "model": f"{exp_name} - {row['model']}",
                        "values": values + [values[0]],  # Close the polygon
                        "metrics": metrics + [metrics[0]],
                    }
                )

        fig_radar = go.Figure()
        for data in radar_data:
            fig_radar.add_trace(
                go.Scatterpolar(
                    r=data["values"], theta=data["metrics"], fill="toself", name=data["model"]
                )
            )

        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            showlegend=True,
            title="Multi-Metric Performance Comparison",
            height=600,
        )

        # 3. Box plot for metric distribution
        fig_box = px.box(
            results_df,
            x="experiment",
            y=metric,
            color="experiment",
            title=f'{metric.replace("_", " ").title()} Distribution by Experiment',
            height=400,
        )

        return html.Div(
            [
                html.H3("Performance Analysis"),
                dbc.Row([dbc.Col(dcc.Graph(figure=fig_bar), width=12)]),
                dbc.Row(
                    [
                        dbc.Col(dcc.Graph(figure=fig_radar), width=8),
                        dbc.Col(dcc.Graph(figure=fig_box), width=4),
                    ]
                ),
            ]
        )

    def _render_comparison(self, exp_ids):
        """Render detailed comparison tab"""
        if not exp_ids or len(exp_ids) < 2:
            return html.Div(
                "Please select at least 2 experiments to compare", className="text-center mt-5"
            )

        # Load experiments
        experiments = []
        for exp_id in exp_ids[:2]:  # Compare first 2
            exp = self.results_manager.load_experiment_result(exp_id)
            if exp:
                experiments.append(exp)

        if len(experiments) < 2:
            return html.Div("Could not load experiment data")

        # Side-by-side comparison
        comparison_items = []

        # Best models comparison
        comparison_items.append(
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.H5(experiments[0].experiment_name),
                            html.P(f"Best Model: {experiments[0].best_model}"),
                            html.P(f"Accuracy: {experiments[0].best_accuracy:.3f}"),
                        ],
                        width=6,
                        className="text-center",
                    ),
                    dbc.Col(
                        [
                            html.H5(experiments[1].experiment_name),
                            html.P(f"Best Model: {experiments[1].best_model}"),
                            html.P(f"Accuracy: {experiments[1].best_accuracy:.3f}"),
                        ],
                        width=6,
                        className="text-center",
                    ),
                ],
                className="mb-4",
            )
        )

        # Confusion matrices comparison
        if (
            experiments[0].best_model in experiments[0].model_results
            and experiments[1].best_model in experiments[1].model_results
        ):
            cm1 = experiments[0].model_results[experiments[0].best_model].confusion_matrix
            cm2 = experiments[1].model_results[experiments[1].best_model].confusion_matrix

            if cm1 is not None and cm2 is not None:
                fig_cm = make_subplots(
                    rows=1,
                    cols=2,
                    subplot_titles=[experiments[0].experiment_name, experiments[1].experiment_name],
                )

                # Add confusion matrices
                fig_cm.add_trace(
                    go.Heatmap(z=cm1, colorscale="Blues", showscale=False), row=1, col=1
                )
                fig_cm.add_trace(
                    go.Heatmap(z=cm2, colorscale="Blues", showscale=True), row=1, col=2
                )

                fig_cm.update_layout(title="Confusion Matrix Comparison", height=400)

                comparison_items.append(dcc.Graph(figure=fig_cm))

        # Training time comparison
        time_data = []
        for exp in experiments:
            for model_name, model_result in exp.model_results.items():
                time_data.append(
                    {
                        "experiment": exp.experiment_name,
                        "model": model_name,
                        "training_time": model_result.training_time,
                    }
                )

        time_df = pd.DataFrame(time_data)
        fig_time = px.bar(
            time_df,
            x="model",
            y="training_time",
            color="experiment",
            barmode="group",
            title="Training Time Comparison (seconds)",
            height=400,
        )

        comparison_items.append(dcc.Graph(figure=fig_time))

        return html.Div([html.H3("Detailed Experiment Comparison"), *comparison_items])

    def _render_features(self, exp_ids):
        """Render feature analysis tab"""
        if not exp_ids:
            return html.Div("Please select experiments to analyze", className="text-center mt-5")

        # Load feature importance data
        importance_data = {}

        for exp_id in exp_ids:
            exp = self.results_manager.load_experiment_result(exp_id)
            if exp:
                for model_name, model_result in exp.model_results.items():
                    if model_result.feature_importance is not None:
                        key = f"{exp.experiment_name} - {model_name}"
                        importance_data[key] = model_result.feature_importance

        if not importance_data:
            return html.Div("No feature importance data available")

        # Create feature importance visualization
        fig = self.model_visualizer.plot_feature_importance(
            importance_data, top_n=15, title="Feature Importance Comparison", interactive=True
        )

        # Feature correlation analysis
        # (Would need actual feature data for this)

        return html.Div(
            [
                html.H3("Feature Analysis"),
                dcc.Graph(figure=fig),
                html.Hr(),
                html.H4("Feature Statistics"),
                html.P("Feature correlation and statistical analysis would go here"),
            ]
        )

    def _render_training(self, exp_ids):
        """Render training analysis tab"""
        if not exp_ids:
            return html.Div("Please select experiments to analyze", className="text-center mt-5")

        # Load training histories
        histories = {}

        for exp_id in exp_ids:
            exp = self.results_manager.load_experiment_result(exp_id)
            if exp:
                for model_name, model_result in exp.model_results.items():
                    if model_result.training_history:
                        key = f"{exp.experiment_name} - {model_name}"
                        histories[key] = model_result.training_history

        if not histories:
            return html.Div("No training history data available")

        # Create training history visualization
        fig = self.model_visualizer.plot_training_history(
            histories,
            metrics=["loss", "accuracy"],
            title="Training History Comparison",
            interactive=True,
        )

        return html.Div(
            [
                html.H3("Training Analysis"),
                dcc.Graph(figure=fig),
                html.Hr(),
                html.H4("Training Statistics"),
                self._create_training_stats_table(exp_ids),
            ]
        )

    def _create_training_stats_table(self, exp_ids):
        """Create training statistics table"""
        stats_data = []

        for exp_id in exp_ids:
            exp = self.results_manager.load_experiment_result(exp_id)
            if exp:
                for model_name, model_result in exp.model_results.items():
                    stats_data.append(
                        {
                            "Experiment": exp.experiment_name,
                            "Model": model_name,
                            "Training Time (s)": f"{model_result.training_time:.2f}",
                            "Final Accuracy": (
                                f"{model_result.test_accuracy:.3f}"
                                if model_result.test_accuracy
                                else "N/A"
                            ),
                            "Parameters": model_result.config.get("n_params", "N/A"),
                        }
                    )

        if not stats_data:
            return html.Div("No training statistics available")

        df = pd.DataFrame(stats_data)
        return dash_table.DataTable(
            data=df.to_dict("records"),
            columns=[{"name": i, "id": i} for i in df.columns],
            style_cell={"textAlign": "left"},
            style_data_conditional=[
                {"if": {"row_index": "odd"}, "backgroundColor": "rgb(248, 248, 248)"}
            ],
        )

    def _render_weather(self, exp_ids):
        """Render weather analysis tab"""
        # This would show weather-specific analysis
        # For now, showing placeholder

        return html.Div(
            [
                html.H3("Weather Analysis"),
                html.P(
                    "Weather pattern analysis and regulation correlation would be displayed here"
                ),
                html.Hr(),
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                html.H5("Weather Severity Distribution"),
                                html.P("Chart showing distribution of weather severity scores"),
                            ],
                            width=6,
                        ),
                        dbc.Col(
                            [
                                html.H5("Regulation Patterns"),
                                html.P("Analysis of regulation patterns by weather conditions"),
                            ],
                            width=6,
                        ),
                    ]
                ),
            ]
        )

    def _render_reports(self, exp_ids):
        """Render reports tab"""
        if not exp_ids:
            return html.Div(
                "Please select experiments to generate reports", className="text-center mt-5"
            )

        # Export options
        export_formats = ["csv", "excel", "json", "latex"]

        return html.Div(
            [
                html.H3("Generate Reports"),
                html.P("Export experiment results in various formats"),
                html.Hr(),
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                html.Label("Select Experiment:"),
                                dcc.Dropdown(
                                    id="report-experiment",
                                    options=[
                                        {"label": exp_id, "value": exp_id} for exp_id in exp_ids
                                    ],
                                    value=exp_ids[0] if exp_ids else None,
                                ),
                            ],
                            width=6,
                        ),
                        dbc.Col(
                            [
                                html.Label("Export Format:"),
                                dcc.Dropdown(
                                    id="report-format",
                                    options=[
                                        {"label": fmt.upper(), "value": fmt}
                                        for fmt in export_formats
                                    ],
                                    value="excel",
                                ),
                            ],
                            width=6,
                        ),
                    ]
                ),
                html.Br(),
                dbc.Button("Generate Report", id="generate-report", color="primary", size="lg"),
                html.Div(id="report-status", className="mt-3"),
            ]
        )

    def run(self, debug=False, port=8050):
        """Run the dashboard"""
        self.create_layout()
        self.setup_callbacks()

        # Add report generation callback
        @self.app.callback(
            Output("report-status", "children"),
            Input("generate-report", "n_clicks"),
            State("report-experiment", "value"),
            State("report-format", "value"),
        )
        def generate_report(n_clicks, exp_id, format):
            if not n_clicks or not exp_id:
                return ""

            try:
                output_path = self.results_manager.export_results(exp_id, format)
                return dbc.Alert(
                    f"Report successfully generated: {output_path}",
                    color="success",
                    dismissable=True,
                )
            except Exception as e:
                return dbc.Alert(
                    f"Error generating report: {str(e)}", color="danger", dismissable=True
                )

        self.app.run_server(debug=debug, port=port)


# Standalone dashboard creation function
def create_dashboard(
    results_path: str = "./results", experiments: list[str] | None = None
) -> ModelComparisonDashboard:
    """Create and configure dashboard"""
    dashboard = ModelComparisonDashboard(results_path)

    # Pre-select experiments if provided
    if experiments:
        # This would require modifying the dashboard to accept initial values
        pass

    return dashboard


# Quick launch function
def launch_dashboard(results_path: str = "./results", port: int = 8050):
    """Quick function to launch dashboard"""
    dashboard = create_dashboard(results_path)
    print("\nLaunching Model Comparison Dashboard...")
    print(f"Open your browser to: http://localhost:{port}")
    dashboard.run(debug=False, port=port)
