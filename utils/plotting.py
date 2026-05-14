import numpy as np
import plotly.graph_objects as go


def plot_trial_results(model_name, mean_curve, std_curve=None):
    """Plot mean steps-until-failure per trial, with optional ±1 std band."""
    trials = np.arange(len(mean_curve))
    fig = go.Figure()

    if std_curve is not None:
        fig.add_trace(go.Scatter(
            x=trials, y=mean_curve + std_curve,
            mode="lines", line=dict(width=0),
            showlegend=False, hoverinfo="skip",
        ))
        fig.add_trace(go.Scatter(
            x=trials, y=mean_curve - std_curve,
            mode="lines", line=dict(width=0),
            fill="tonexty", fillcolor="rgba(0, 100, 200, 0.2)",
            name="±1 std", hoverinfo="skip",
        ))

    fig.add_trace(go.Scatter(
        x=trials, y=mean_curve,
        mode="lines+markers",
        name="Mean steps until failure",
        line=dict(color="rgb(0, 100, 200)"),
    ))

    fig.update_layout(
        title=f"{model_name} Learning Curve",
        xaxis_title="Trial",
        yaxis_title="Steps until failure",
        template="plotly_white",
    )

    fig.show(config={"toImageButtonOptions": {"format": "png", "scale": 3}})
