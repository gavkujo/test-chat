import plotly.graph_objs as go
import PyPDF2
from io import BytesIO
import pandas as pd
from datetime import timedelta
from helpers.datasources import S_series
from helpers.data_structures import merge_pdfs_to_bytes

def plot_combi_S(ids, df_S, Y_dtick=500):
    colours = ['rgba(255, 0, 0, 1)', 'rgba(0, 0, 255, 1)', 'rgba(0, 255, 0, 1)', 'rgba(255, 255, 0, 1)',
               'rgba(128, 0, 128, 1)', 'rgba(255, 165, 0, 1)', 'rgba(255, 192, 203, 1)', 'rgba(0, 128, 128, 1)',
               'rgba(0, 255, 150, 1)', 'rgba(0, 255, 255, 1)', 'rgba(255, 0, 255, 1)', 'rgba(75, 0, 130, 1)',
               'rgba(64, 224, 208, 1)', 'rgba(255, 215, 0, 1)', 'rgba(192, 192, 192, 1)', 'rgba(238, 130, 238, 1)',
               'rgba(128, 0, 0, 1)', 'rgba(128, 128, 0, 1)', 'rgba(0, 0, 128, 1)', 'rgba(255, 127, 80, 1)',
               'rgba(112, 128, 144, 1)', 'rgba(221, 160, 221, 1)', 'rgba(250, 128, 114, 1)', 'rgba(240, 255, 255, 1)',
               'rgba(230, 230, 250, 1)', 'rgba(224, 17, 95, 1)', 'rgba(80, 200, 120, 1)', 'rgba(255, 191, 0, 1)',
               'rgba(234, 224, 200, 1)', 'rgba(54, 69, 79, 1)', 'rgba(255, 255, 240, 1)', 'rgba(220, 20, 60, 1)',
               'rgba(15, 82, 186, 1)', 'rgba(255, 204, 0, 1)', 'rgba(0, 168, 107, 1)', 'rgba(255, 0, 127, 1)',
               'rgba(205, 127, 50, 1)', 'rgba(127, 255, 212, 1)', 'rgba(210, 105, 30, 1)', 'rgba(128, 128, 128, 1)']

    df_S['Settlement (mm)'] = pd.to_numeric(df_S['Settlement (mm)'], errors='coerce')
    # df_S.dropna(subset=['Settlement (mm)'], inplace=True)

    start_date = min(df_S["Date"])
    end_date = max(df_S["Date"]) + + timedelta(days=15)

    traces = []
    for i, id_value in enumerate(ids):
        trace_settlement = go.Scatter(
            x=df_S[df_S['id'] == id_value]['Date'],
            y=df_S[df_S['id'] == id_value]['Settlement (mm)'],
            xaxis='x1',
            yaxis='y1',
            name=id_value,
            mode='lines+markers',
            line=dict(width=2, color=colours[i], dash="solid"),
            connectgaps=False,
            legendgroup="l1",
            showlegend=True
        )
        traces.append(trace_settlement)

    x_axis = dict(
        title='Date',
        range=[start_date, end_date],
        tickangle=45,
        showline=True,
        mirror=True,
        showgrid=True,
        tickfont=dict(size=10)
    )

    y_axis = dict(
        showline=True,
        zeroline=True,
        showgrid=True,
        mirror=True,
        ticks="outside",
        dtick=Y_dtick,
        ticklen=6,
        tickcolor="black",
        minor=dict(ticklen=6, tickcolor="black"),
        gridcolor='#d4d4d4',
        tickfont=dict(size=10)
    )

    layout = dict(
        width=1200,
        height=600,
        autosize=True,
        title=dict(
            text='Settlement Trend',
            font=dict(size=16),
            x=0.5,
            xanchor='center'
        ),
        margin=dict(l=10, r=10, t=30, b=0.1),
        xaxis1=dict(x_axis, **dict(domain=[0, 1], anchor='y1', dtick="M1", showticklabels=True, title='Date')),
        yaxis1=dict(y_axis, **dict(domain=[0.2, 1], anchor='x1', title='Settlement (mm)')),
        plot_bgcolor='rgba(229, 236, 246, 100)'
    )

    fig_gen = dict(data=traces, layout=layout)

    fig = go.Figure(fig_gen)
    fig.update_layout(legend=dict(
        orientation="h",
        yanchor="top",
        y=0.02,
        xanchor="left",
        x=0,
    ))
    # fig.show()
    pg1 = fig.to_image(format="pdf")
    return pg1

def y_tick_interval(value, multiple):
    return value - (value % multiple)

def reporter_Settlement(ids, max_date, Y_dtick=None):
    df_settlement = S_series(ids, max_date)
    if Y_dtick is None:
        Y_dtick = y_tick_interval(abs(max(df_settlement["Settlement (mm)"])), 25)

    tmp = BytesIO()

    merger = PyPDF2.PdfMerger(fileobj=tmp)
    pg1 = plot_combi_S(ids, df_settlement, Y_dtick)
    merger.append(PyPDF2.PdfReader(BytesIO(pg1)))

    return merge_pdfs_to_bytes(merger)
