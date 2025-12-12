import pprint
import plotly.graph_objs as go
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import PyPDF2
import os
from io import BytesIO
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib import colors
from reportlab.platypus import Table, TableStyle
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase import pdfmetrics

from helpers.asaoka import Asaoka_data
from helpers.datasources import S_series, SM_metrics
from helpers.data_structures import merge_pdfs_to_bytes


def plot_pg_0(df_overview):
    if 'OpenSans' not in pdfmetrics.getRegisteredFontNames():
        base_dir = os.getcwd()
        regular_font_path = os.path.join(base_dir, 'static', 'fonts', 'OpenSans-Regular.ttf')
        bold_font_path = os.path.join(base_dir, 'static', 'fonts', 'OpenSans-Bold.ttf')

        pdfmetrics.registerFont(TTFont('OpenSans', regular_font_path))
        pdfmetrics.registerFont(TTFont('OpenSans-Bold', bold_font_path))


    latest_date = max(pd.to_datetime(df_overview["Last Read"]))
    week = latest_date.isocalendar()[1]

    tmp_pg0 = BytesIO()
    c = canvas.Canvas(tmp_pg0, pagesize=A4)
    width, height = A4

    c.setFont("OpenSans-Bold", 16)
    text_1 = f"Asaoka Assessment Summary (data as at Week {week})"
    c.drawString(50, height - 40, text_1)

    data_list = [["Settlement \nPlate", "Last Read", "Latest Settlement\n(m)", "Latest GroundLevel\n (mCD)",
                  "Asaoka DOC\n(%)", "Remarks"]] + df_overview.values.tolist()

    col_widths = [100, 80, 100, 110, 100, 60]
    table = Table(data_list, colWidths=col_widths)

    def get_text_color(value):
        if value == "OK":
            return colors.green
        elif value == "Not OK":
            return colors.red
        else:
            return colors.black

    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.Color(229 / 255, 236 / 255, 246 / 255)),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'OpenSans-Bold'),
        ('FONTNAME', (0, 1), (-1, -1), 'OpenSans'),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.Color(229 / 255, 236 / 255, 246 / 255)),
        ('GRID', (0, 0), (-1, -1), 1, colors.Color(50 / 255, 50 / 255, 50 / 255)),
    ]))

    for i, row in enumerate(data_list[1:], start=1):  # Skip header row
        text_color = get_text_color(row[5])
        table.setStyle(TableStyle([
            ('TEXTCOLOR', (5, i), (5, i), text_color),
        ]))

    table_width, table_height = table.wrapOn(c, width, height)
    table.drawOn(c, (width - table_width) / 2, height - table_height - 90)

    c.save()
    tmp_pg0.seek(0)

    return tmp_pg0



def plot_pg_1(id, df_S, df_A, table_S, table_A, SCD, ASD, trendline, Y_dtick=500):
    m, b = trendline["m"], trendline["b"]

    table_header_1 = go.Table(
        domain=dict(x=[0, 1],
                    y=[0.95, 1.0]),
        header=dict(height=30,
                    values=[f'{id} Settlement and ground level Series'],
                    line=dict(color='rgb(0,0,0)'),
                    font=dict(color=['rgb(0,0,0)'], size=14),
                    fill=dict(color='rgba(229, 236, 246, 100)')))

    table_header_2 = go.Table(
        domain=dict(x=[0, 1],
                    y=[0.51, 0.56]),
        header=dict(height=30,
                    values=[f'{id} Asaoka Plot'],
                    line=dict(color='rgb(0,0,0)'),
                    font=dict(color=['rgb(0,0,0)'], size=14),
                    fill=dict(color='rgba(229, 236, 246, 100)')))

    table_summary = go.Table(
        domain=dict(x=[0, 1],
                    y=[0.87, 0.95]),
        header=dict(height=30,
                    values=list(table_S.keys()),
                    line=dict(color='rgb(50, 50, 50)'),
                    font=dict(color=['rgb(45, 45, 45)'] * 5, size=14),
                    fill=dict(color='rgba(229, 236, 246, 100)')),
        cells=dict(values=list(table_S.values()),
                   line=dict(color='#506784'),
                   font=dict(color=['rgb(40, 40, 40)'] * 5, size=12),
                   fill=dict(color='rgba(229, 236, 246, 100)'),
                   height=30)
    )

    table_Asaoka = go.Table(
        domain=dict(x=[0, 1],
                    y=[0.41, 0.51]),
        header=dict(values=list(table_A.keys()),
                    line=dict(color='rgb(50, 50, 50)'),
                    font=dict(color=['rgb(45, 45, 45)'] * 5, size=14),
                    fill=dict(color='rgba(229, 236, 246, 100)')),
        cells=dict(values=list(table_A.values()),
                   line=dict(color='#506784'),
                   font=dict(color=['rgb(40, 40, 40)'] * 5, size=12),
                   fill=dict(color='rgba(229, 236, 246, 100)')
                   ))

    trace_settlement = go.Scatter(
        x=df_S['Date'],
        y=df_S['Settlement (mm)'],
        xaxis='x1',
        yaxis='y1',
        mode='lines',
        line=dict(width=2, color="#EF553B", dash="solid"),
        connectgaps=True,
        name=f'Settlement (mm)',
        legendgroup="l1"
    )

    trace_GL = go.Scatter(
        x=df_S['Date'],
        y=df_S['Ground Level (mCD)'],
        xaxis='x1',
        yaxis='y12',
        mode='lines',
        line=dict(width=2, color="#00cc96", dash="solid"),
        connectgaps=True,
        name=f'Ground Level (mCD)',
        legendgroup="l1"
    )

    trace_SCD = go.Scatter(
        x=[SCD, SCD],
        y=[0, max(df_S['Ground Level (mCD)']) + 1],
        xaxis='x1',
        yaxis='y12',
        mode='lines',
        line=dict(width=2, color="#0000cc", dash="dash"),
        connectgaps=True,
        name=f'Surcharge Completion Date (SCD)',
        legendgroup="l1"
    )

    trace_ASD = go.Scatter(
        x=[ASD, ASD],
        y=[0, max(df_S['Ground Level (mCD)']) + 1],
        xaxis='x1',
        yaxis='y12',
        mode='lines',
        line=dict(width=2, color="#800080", dash="solid"),
        connectgaps=True,
        name=f'Assessment Start Date (ASD)',
        legendgroup="l1"
    )

    trace_Asaoka = go.Scatter(
        x=df_A['St-1'],
        y=df_A['St'],
        xaxis='x2',
        yaxis='y2',
        mode='markers',
        name=f'{id} Asaoka Plot',
        showlegend=False
    )

    trace_int = go.Scatter(
        x=[min(df_A['St-1']) - 10, max(df_A['St']) + 10],
        y=[min(df_A['St-1']) - 10, max(df_A['St']) + 10],
        xaxis='x2',
        yaxis='y2',
        mode='lines',
        line=dict(width=2, color="#0000cc", dash="solid"),
        connectgaps=True,
        showlegend=False
    )
    trace_fit = go.Scatter(
        x=df_A['St-1'],
        y=m * df_A['St-1'] + b,
        xaxis='x2',
        yaxis='y2',
        mode='lines',
        line=dict(width=2, color="rgba(0, 255, 136)", dash="solid"),
        connectgaps=True,
        showlegend=False
    )

    x_axis = dict(
        showline=True,
        zeroline=False,
        showgrid=True,
        mirror=True,
        range=[min(df_S["Date"]), max(df_S["Date"] + timedelta(days=15))],
        minor=dict(ticklen=5, dtick="D1", tickcolor="black"),
        ticklen=4,
        gridcolor='#d4d4d4',
        tickfont=dict(size=10)
    )

    axis_asaoka = dict(
        showline=True,
        zeroline=False,
        showgrid=True,
        mirror=True,
        range=[min(df_S["Date"]), max(df_S["Date"] + timedelta(days=15))],
        ticklen=4,
        gridcolor='#d4d4d4',
        tickfont=dict(size=10)
    )

    y_axis_sec = dict(
        showline=False,
        zeroline=False,
        showgrid=False,
        mirror=True,
        dtick=2.00,
        ticks="outside",
        ticklen=6,
        tickcolor="black",
        minor=dict(ticklen=6, tickcolor="black"),
        gridcolor='#ffffff',
        tickfont=dict(size=10)
    )

    y_axis = dict(
        showline=True,
        zeroline=False,
        showgrid=True,
        mirror=True,
        dtick=Y_dtick,
        minor=dict(ticklen=6, tickcolor="black"),
        ticks="outside",
        ticklen=6,
        tickcolor="black",
        gridcolor='#ffffff',
        tickfont=dict(size=10)
    )

    layout = dict(
        width=800,
        height=1200,
        autosize=True,
        title=f'{id} Asaoka Assessment',
        xaxis1=dict(x_axis, **dict(domain=[0, 1], anchor='y1', dtick="M1", showticklabels=True, title='Date')),
        xaxis2=dict(axis_asaoka, **dict(domain=[0, 1], anchor='y2', title="S<sub>T-1</sub>")),

        yaxis1=dict(y_axis, **dict(domain=[0.67, 0.87], anchor='x1', title='Settlement (mm)')),
        yaxis12=dict(y_axis_sec, **dict(domain=[0.67, 0.87], anchor='x1', title='Ground level (mCD)',
                                        overlaying='y', side='right')),
        yaxis2=dict(axis_asaoka, **dict(domain=[0.16, 0.41], anchor='x2', title="S<sub>T</sub>")),
        plot_bgcolor='rgba(229, 236, 246, 100)'
    )

    fig_gen = dict(
        data=[table_header_1, table_header_2, trace_settlement, trace_GL, trace_ASD, trace_SCD, table_summary,
              trace_Asaoka, trace_int, trace_fit, table_Asaoka], layout=layout)

    fig = go.Figure(fig_gen)
    fig.update_layout(legend=dict(
        orientation="h",
        yanchor="top",
        y=0.65,
        xanchor="left",
        x=0
    ))

    pg1 = fig.to_image(format="pdf", width=800, height=1124)
    return pg1


def plot_pg_2(id, df_plot):
    dp_limit = ''
    if len(df_plot) > 30:
        df_plot = df_plot.tail(30)
        dp_limit = ' (Showing latest 30 datapoints)'
    table_header = go.Table(
        domain=dict(x=[0, 1],
                    y=[0.95, 1.0]),
        header=dict(height=30,
                    values=[f'{id} Asaoka Ordered Pairs{dp_limit}'],
                    line=dict(color='rgb(255,255,255)'),
                    font=dict(color=['rgb(0,0,0)'], size=14),
                    fill=dict(color='rgba(229, 236, 246, 100)')))

    table_pairs = go.Table(
        domain=dict(x=[0, 1],
                    y=[0, 0.95]),
        header=dict(height=30,
                    values=["T", "St", "T-1", "St-1"],
                    line=dict(color='rgb(50, 50, 50)'),
                    font=dict(color=['rgb(45, 45, 45)'] * 5, size=14),
                    fill=dict(color='rgba(229, 236, 246, 100)')),
        cells=dict(values=list(df_plot[i] for i in ["T", "St", "T-1", "St-1"]),
                   line=dict(color='#506784'),
                   font=dict(color=['rgb(40, 40, 40)'] * 5, size=12),
                   fill=dict(color='rgba(229, 236, 246, 100)'),
                   height=30)
    )

    layout = dict(
        width=800,
        height=1200,
        autosize=True,
        plot_bgcolor='rgba(229, 236, 246, 100)'
    )

    fig_gen = dict(data=[table_header, table_pairs], layout=layout)
    fig = go.Figure(fig_gen)
    fig.update_layout(legend=dict(
        orientation="h",
        yanchor="top",
        y=0.63,
        xanchor="left",
        x=0
    ))

    pg2 = fig.to_image(format="pdf", width=800, height=1124)
    return pg2


def reporter_Asaoka(ids, SCD: str, ASD: str, max_date, n=4, asaoka_days=7, dtick=500):
    df_settlement = S_series(ids, max_date)
    tmp = BytesIO()
    merger = PyPDF2.PdfMerger(fileobj=tmp)

    period = 0
    DOC_lst = []

    # SCD = datetime.strptime(SCD, "%Y-%m-%d")
    # ASD = datetime.strptime(ASD, "%Y-%m-%d")

    for id in ids:
        try:
            asaoka_results = Asaoka_data(id, SCD, ASD, max_date, asaoka_days, period, n)
            pprint.pp(asaoka_results)
            SM_data = SM_metrics(id)
            df = df_settlement[df_settlement["id"] == id]

            series = asaoka_results["pairs"]
            dates = asaoka_results["dates"]
            df_Asaoka = pd.DataFrame(series, columns=['St-1', 'St'])
            df_Asaoka[["T-1", "T"]] = dates

            m = round(asaoka_results["m"], 3)
            b = round(asaoka_results["b"], 2)
            sf = asaoka_results["Asaoka_pred"]
            equation = f"y = {m}x + {b}"
            DOC = min(round(asaoka_results["DOC"], 2), 100)
            latest_date = asaoka_results["Latest_date"]
            latest_GL = asaoka_results["Latest_GL"]

            DOC_lst.append([id, str(str(latest_date)[:10]), str(round(SM_data["Final_S"],2)), str(round(SM_data["GL"], 2)), DOC])

            table_S = {
                "Easting": SM_data["Easting"],
                "Northing": SM_data["Northing"],
                "Latest Reading": str(round(SM_data["Final_S"], 3)) + ' m',
                "Ground Level": str(round(SM_data["GL"], 2)) + ' mCD',
                "DOC": str(DOC) + ' %'
            }

            table_Asaoka = {
                "Equation": str(equation),
                "Asaoka sf (m)": str(sf),
                "DOC (%)": str(DOC),
                "Assessment From": str(str(ASD)[:10])
            }

            trendline = {"m": m,
                         "b": b}

            pg1 = plot_pg_1(id, df, df_Asaoka, table_S, table_Asaoka, SCD, ASD, trendline, dtick)
            pg2 = plot_pg_2(id, df_Asaoka)

            merger.append(PyPDF2.PdfReader(BytesIO(pg1)))
            merger.append(PyPDF2.PdfReader(BytesIO(pg2)))

        except:
            return None

        # except KeyError:
        #    return {"result": f"Report for {id} cannot be generated. (Unspecified SCD)"}
        # except TypeError:
        #     return {f"Report for {id} cannot be generated. (Invalid SCD)"}
        # except ZeroDivisionError:
        #     return {f"Report for {id} cannot be generated. (Missing data. Check SCD)"}

    DOC_df = pd.DataFrame(DOC_lst, columns=["Settlement Plate", "Last Read", "Latest Settlement (m)",
    "Latest GL (mCD)", "DOC (%)"])

    DOC_df["Remarks"] = np.where(DOC_df["DOC (%)"] > 90, "OK", "Not OK")

    pg0 = plot_pg_0(DOC_df)
    merger.append(PyPDF2.PdfReader(pg0))

    return merge_pdfs_to_bytes(merger)
