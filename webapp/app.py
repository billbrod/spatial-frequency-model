import sfm
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output


app = dash.Dash(__name__, suppress_callback_exceptions=True)

app.layout = html.Div(children=[
    html.H1('Hello Dash'),

    html.Div([
        'Slope',
        dcc.Slider(
            id='slope-slider', min=0, max=1, value=.1, marks={i/10: str(i/10) for i in range(11)},
            step=None,
        )], style={'width': '33%', 'padding': '20px 20px'}
    ),
    html.Div([
        dcc.Graph(
            id='period-graph',
        )], style={'width': '33%', 'display': 'inline-block'}),

    html.Div([
        dcc.Graph(
            id='period-contour-graph',
        )], style={'width': '33%', 'display': 'inline-block'}),

    html.Div([
        dcc.Graph(
            id='amp-contour-graph',
        )
    ], style={'width': '33%', 'display': 'inline-block'}),
])


@app.callback(
    Output('period-graph', 'figure'),
    [Input('slope-slider', 'value')]
)
def period_graph(slope):
    """
    """
    model = sfm.model.LogGaussianDonut('full', sf_ecc_slope=slope, sf_ecc_intercept=.35,
                                       rel_mode_cardinals=.1)
    per_df = sfm.analyze_model.create_preferred_period_df(model, 'relative')
    return {
        'data': [
            dict(x=per_df.query("`Stimulus type`==@i")['Eccentricity (deg)'],
                 y=per_df.query("`Stimulus type`==@i")['Preferred period (dpc)'],
                 name=i)
            for i in per_df['Stimulus type'].unique()
        ],
        'layout': {
            'xaxis': {'title': 'Eccentricity (deg)'},
            'yaxis': {'title': 'Preferred period (dpc)'},
            'transition': {'duration': 500},
        }
    }


@app.callback(
    Output('period-contour-graph', 'figure'),
    [Input('slope-slider', 'value')]
)
def period_contour_graph(slope):
    """
    """
    model = sfm.model.LogGaussianDonut('full', sf_ecc_slope=slope, sf_ecc_intercept=.35,
                                       rel_mode_cardinals=.1)
    contour_df = sfm.analyze_model.create_preferred_period_contour_df(model, 'relative',
                                                                      period_target=[1])
    return {
        'data': [
            dict(theta=contour_df.query("`Stimulus type`==@i")['Retinotopic angle (rad)'],
                 thetaunit='radians',
                 r=contour_df.query("`Stimulus type`==@i")['Eccentricity (deg)'],
                 name=i, type='scatterpolar')
            for i in contour_df['Stimulus type'].unique()
        ],
        'layout': {
            'angularaxis': {'title': 'Retinotopic angle', 'thetaunit': 'radians'},
            'radialaxis': {'title': 'Eccentricity (deg)'},
            'title': 'Iso-preferred-period contours',
            'transition': {'duration': 500},
        }
    }


@app.callback(
    Output('amp-contour-graph', 'figure'),
    [Input('slope-slider', 'value')]
)
def amp_contour_graph(slope):
    """
    """
    model = sfm.model.LogGaussianDonut('full', sf_ecc_slope=slope, sf_ecc_intercept=.35,
                                       rel_mode_cardinals=.1)
    amp_df = sfm.analyze_model.create_max_amplitude_df(model, 'relative')
    return {
        'data': [
            dict(theta=amp_df.query("`Stimulus type`==@i")['Retinotopic angle (rad)'],
                 thetaunit='radians',
                 r=amp_df.query("`Stimulus type`==@i")['Max amplitude'],
                 name=i, type='scatterpolar')
            for i in amp_df['Stimulus type'].unique()
        ],
        'layout': {
            'angularaxis': {'title': 'Retinotopic angle'},
            'radialaxis': {'title': 'Max amplitude'},
            'transition': {'duration': 500},
        }
    }


if __name__ == '__main__':
    app.run_server(debug=True)
