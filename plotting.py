import plotly.express as px
import plotly.graph_objects as go
import plotly.colors as pc


def plot_hourly(
    data, hours=None, title="Hourly Data", yaxis_title="Value", color="blue", line=True, bar=True
):
    """
    Plots hourly data as a bar chart using Plotly.

    Parameters:
    - data: List of values to plot (length should be 24 for 24 hours).
    - hours: Optional list of hour labels (e.g., ["00:00", "01:00", ..., "23:00"]).
             If None, it defaults to ["0h", "1h", ..., "23h"].
    - title: Title of the plot.
    - yaxis_title: Label for the y-axis.
    """
    if hours is None:
        hours = [f"{i}h" for i in range(len(data))]

    fig = go.Figure()
    if bar:
        fig.add_trace(go.Bar(x=hours, y=data, marker_color=color))
    if line:
        fig.add_trace(go.Scatter(x=hours, y=data, mode="lines+markers", line=dict(color=color)))

    fig.update_layout(
        title=title,
        width=600,
        height=400,
        xaxis_title="Hour of Day",
        yaxis_title=yaxis_title,
        xaxis=dict(tickmode="linear", tick0=0, dtick=1),
        template="plotly_dark",
        plot_bgcolor="#1e1e2e",
        paper_bgcolor="#1e1e2e",
        showlegend=False,
    )
    fig.show()


def plot_stacked_generators(
    generators_by_hour, hours=None, title="Hourly Generation Mix", yaxis_title="Output (MWh)", palette=pc.qualitative.Plotly
):
    """
    Plots a stacked bar chart of generator outputs by hour.
    
    Parameters:
    - generators_by_hour: List of lists, where each inner list contains tuples of (generator, output) for that hour.
    - hours: Optional list of hour labels. If None, it defaults to ["0h", "1h", ..., "23h"].
    - title: Title of the plot.
    - yaxis_title: Label for the y-axis.
    - palette: List of colors to use for the generators.
    """

    if hours is None:
        hours = [f"{i}h" for i in range(len(generators_by_hour))]

    # Collect all unique generators (preserving merit order)
    seen = {}
    for gens_in_hour in generators_by_hour:
        for gen, _ in gens_in_hour:
            if gen.name not in seen:
                seen[gen.name] = gen
    generators = list(seen.values())

    # Assign colors to generators based on their order in the merit stack
    colours = [palette[i % len(palette)] for i in range(len(generators))]

    gen_outputs = {gen.name: [] for gen in generators}
    for gens_in_hour in generators_by_hour:
        dispatched = {gen.name: output for gen, output in gens_in_hour}
        for gen in generators:
            gen_outputs[gen.name].append(dispatched.get(gen.name, 0))

    fig = go.Figure()

    for i, gen in enumerate(generators):
        fig.add_trace(
            go.Bar(
                x=hours,
                y=gen_outputs[gen.name],
                name=f"{gen.type.capitalize()} ({gen.name.upper()})",
                marker_color=colours[i % len(colours)],
            )
        )

        
    fig.update_layout(
        barmode="stack",
        title=title,
        xaxis=dict(title="Hour of Day", tickmode="linear", tick0=0, dtick=1),
        yaxis=dict(title=yaxis_title),
        template="plotly_dark",
        plot_bgcolor="#1e1e2e",
        paper_bgcolor="#1e1e2e",
        width=900,
        height=500,
    )
    fig.show()

def plot_schedule_heatmap(df, title="Generator Power Output Schedule (MW) - UCLP Solution"):
    fig = px.imshow(df, aspect='auto', color_continuous_scale='Blues', title=title)
    fig.update_layout(xaxis_title="Hour", yaxis_title="Generator")
    fig.show()

def plot_multiple_series(series_dict, title="Multiple Series Plot", yaxis_title="Value", line=True, bar=False, hours=None, colors=pc.qualitative.Plotly):
    fig = go.Figure()
    if hours is None:
        hours = [f"{i}h" for i in range(len(next(iter(series_dict.values()))))]
        
    for i, (name, data) in enumerate(series_dict.items()):
        color = colors[i % len(colors)]
        if line:
            fig.add_trace(go.Scatter(x=hours, y=data, mode="lines+markers", name=name, line=dict(color=color)))
        if bar:
            fig.add_trace(go.Bar(x=hours, y=data, name=name, marker_color=color, opacity=0.5))  
    fig.update_layout(
        title=title,
        xaxis_title="Hour of Day",
        yaxis_title=yaxis_title,
        xaxis=dict(tickmode="linear", tick0=0, dtick=1),
        template="plotly_dark",
        plot_bgcolor="#1e1e2e",
        paper_bgcolor="#1e1e2e",
    )

    fig.show()
