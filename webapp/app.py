import sfm
import functions
import dash
import dash_core_components as dcc
import os.path as op
import dash_html_components as html
from dash.dependencies import Input, Output


app = dash.Dash(__name__, suppress_callback_exceptions=True)

eqts_dir = op.join(op.dirname(op.realpath(__file__)), '..', 'equations')
main_eqt = functions.get_svg(op.join(eqts_dir, 'main.svg'))

app.layout = html.Div([
    html.H1('Spatial Frequency Preferences Model'),
    html.Div(id='model-name'),
    html.Div([html.Img(src=main_eqt)]),
    html.Div([html.Img(id='period-eqt')]),
    html.Div([html.Img(id='amp-eqt')]),

    html.Div([
        u'\u03c3',
        dcc.Slider(
            id='sigma-slider', min=.25, max=3, value=1, marks={i/4: str(i/4) for i in range(1, 13)},
            step=None,
        ),
        dcc.Markdown('*a*'),
        dcc.Slider(
            id='slope-slider', min=0, max=1, value=.1, marks={i/10: str(i/10) for i in range(11)},
            step=None,
        ),
        dcc.Markdown('*b*'),
        dcc.Slider(
            id='intercept-slider', min=0, max=1, value=.3, marks={i/10: str(i/10) for i in range(11)},
            step=None,
        ),
    ], style={'width': '30%', 'padding': '20px 20px', 'display': 'inline-block'}
    ),
    html.Div([
        dcc.Markdown('*p<sub>1</sub>*', dangerously_allow_html=True),
        dcc.Slider(
            id='p1-slider', min=-.5, max=.5, value=0,
            marks={(i-5)/10: str((i-5)/10) for i in range(11)},
            step=None,
        ),
        dcc.Markdown('*p<sub>2</sub>*', dangerously_allow_html=True),
        dcc.Slider(
            id='p2-slider', min=-.5, max=.5, value=0,
            marks={(i-5)/10: str((i-5)/10) for i in range(11)},
            step=None,
        ),
        dcc.Markdown('*p<sub>3</sub>*', dangerously_allow_html=True),
        dcc.Slider(
            id='p3-slider', min=-.5, max=.5, value=0,
            marks={(i-5)/10: str((i-5)/10) for i in range(11)},
            step=None,
        ),
        dcc.Markdown('*p<sub>4</sub>*', dangerously_allow_html=True),
        dcc.Slider(
            id='p4-slider', min=-.5, max=.5, value=0,
            marks={(i-5)/10: str((i-5)/10) for i in range(11)},
            step=None,
        ),
    ], style={'width': '30%', 'padding': '20px 20px', 'display': 'inline-block'}
    ),
    html.Div([
        dcc.Markdown('*A<sub>1</sub>*', dangerously_allow_html=True),
        dcc.Slider(
            id='A1-slider', min=-.5, max=.5, value=0,
            marks={(i-5)/10: str((i-5)/10) for i in range(11)},
            step=None,
        ),
        dcc.Markdown('*A<sub>2</sub>*', dangerously_allow_html=True),
        dcc.Slider(
            id='A2-slider', min=-.5, max=.5, value=0,
            marks={(i-5)/10: str((i-5)/10) for i in range(11)},
            step=None,
        ),
        dcc.Markdown('*A<sub>3</sub>*', dangerously_allow_html=True),
        dcc.Slider(
            id='A3-slider', min=-.5, max=.5, value=0,
            marks={(i-5)/10: str((i-5)/10) for i in range(11)},
            step=None,
        ),
        dcc.Markdown('*A<sub>4</sub>*', dangerously_allow_html=True),
        dcc.Slider(
            id='A4-slider', min=-.5, max=.5, value=0,
            marks={(i-5)/10: str((i-5)/10) for i in range(11)},
            step=None,
        ),
    ], style={'width': '30%', 'padding': '20px 20px', 'display': 'inline-block'}
    ),
    html.Div([
        dcc.Graph(id='rel-period-graph'), dcc.Graph(id='abs-period-graph')],
             style={'width': '33%', 'display': 'inline-block'}),

    html.Div([
        dcc.Graph(id='rel-period-contour-graph'), dcc.Graph(id='abs-period-contour-graph')],
             style={'width': '33%', 'display': 'inline-block'}),

    html.Div([
        dcc.Graph(id='rel-amp-contour-graph'), dcc.Graph(id='abs-amp-contour-graph')],
             style={'width': '33%', 'display': 'inline-block'}),
])


@app.callback(
    Output('rel-period-graph', 'figure'),
    [Input('sigma-slider', 'value'),
     Input('slope-slider', 'value'), Input('intercept-slider', 'value'),
     Input('p1-slider', 'value'), Input('p2-slider', 'value'), Input('p3-slider', 'value'),
     Input('p4-slider', 'value'), Input('A1-slider', 'value'), Input('A2-slider', 'value'),
     Input('A3-slider', 'value'), Input('A4-slider', 'value'), ]
)
def rel_period_graph(sigma, a, b, p1, p2, p3, p4, A1, A2, A3, A4):
    """
    """
    model = sfm.model.LogGaussianDonut('full', 'full', True, sigma, a, b, p1, p2, p3, p4,
                                       A1, A2, A3, A4)
    return functions.cartesian_plot(model, 'relative')


@app.callback(
    Output('abs-period-graph', 'figure'),
    [Input('sigma-slider', 'value'),
     Input('slope-slider', 'value'), Input('intercept-slider', 'value'),
     Input('p1-slider', 'value'), Input('p2-slider', 'value'), Input('p3-slider', 'value'),
     Input('p4-slider', 'value'), Input('A1-slider', 'value'), Input('A2-slider', 'value'),
     Input('A3-slider', 'value'), Input('A4-slider', 'value'), ]
)
def abs_period_graph(sigma, a, b, p1, p2, p3, p4, A1, A2, A3, A4):
    """
    """
    model = sfm.model.LogGaussianDonut('full', 'full', True, sigma, a, b, p1, p2, p3, p4,
                                       A1, A2, A3, A4)
    return functions.cartesian_plot(model, 'absolute')


@app.callback(
    Output('rel-period-contour-graph', 'figure'),
    [Input('sigma-slider', 'value'),
     Input('slope-slider', 'value'), Input('intercept-slider', 'value'),
     Input('p1-slider', 'value'), Input('p2-slider', 'value'), Input('p3-slider', 'value'),
     Input('p4-slider', 'value'), Input('A1-slider', 'value'), Input('A2-slider', 'value'),
     Input('A3-slider', 'value'), Input('A4-slider', 'value'), ]
)
def rel_period_contour_graph(sigma, a, b, p1, p2, p3, p4, A1, A2, A3, A4):
    """
    """
    model = sfm.model.LogGaussianDonut('full', 'full', True, sigma, a, b, p1, p2, p3, p4,
                                       A1, A2, A3, A4)
    return functions.polar_plot(model, 'relative', 'period')


@app.callback(
    Output('abs-period-contour-graph', 'figure'),
    [Input('sigma-slider', 'value'),
     Input('slope-slider', 'value'), Input('intercept-slider', 'value'),
     Input('p1-slider', 'value'), Input('p2-slider', 'value'), Input('p3-slider', 'value'),
     Input('p4-slider', 'value'), Input('A1-slider', 'value'), Input('A2-slider', 'value'),
     Input('A3-slider', 'value'), Input('A4-slider', 'value'), ]
)
def abs_period_contour_graph(sigma, a, b, p1, p2, p3, p4, A1, A2, A3, A4):
    """
    """
    model = sfm.model.LogGaussianDonut('full', 'full', True, sigma, a, b, p1, p2, p3, p4,
                                       A1, A2, A3, A4)
    return functions.polar_plot(model, 'absolute', 'period')


@app.callback(
    Output('rel-amp-contour-graph', 'figure'),
    [Input('sigma-slider', 'value'),
     Input('slope-slider', 'value'), Input('intercept-slider', 'value'),
     Input('p1-slider', 'value'), Input('p2-slider', 'value'), Input('p3-slider', 'value'),
     Input('p4-slider', 'value'), Input('A1-slider', 'value'), Input('A2-slider', 'value'),
     Input('A3-slider', 'value'), Input('A4-slider', 'value'), ]
)
def rel_amp_contour_graph(sigma, a, b, p1, p2, p3, p4, A1, A2, A3, A4):
    """
    """
    model = sfm.model.LogGaussianDonut('full', 'full', True, sigma, a, b, p1, p2, p3, p4,
                                       A1, A2, A3, A4)
    return functions.polar_plot(model, 'relative', 'amp')


@app.callback(
    Output('abs-amp-contour-graph', 'figure'),
    [Input('sigma-slider', 'value'),
     Input('slope-slider', 'value'), Input('intercept-slider', 'value'),
     Input('p1-slider', 'value'), Input('p2-slider', 'value'), Input('p3-slider', 'value'),
     Input('p4-slider', 'value'), Input('A1-slider', 'value'), Input('A2-slider', 'value'),
     Input('A3-slider', 'value'), Input('A4-slider', 'value'), ]
)
def abs_amp_contour_graph(sigma, a, b, p1, p2, p3, p4, A1, A2, A3, A4):
    """
    """
    model = sfm.model.LogGaussianDonut('full', 'full', True, sigma, a, b, p1, p2, p3, p4,
                                       A1, A2, A3, A4)
    return functions.polar_plot(model, 'absolute', 'amp')


@app.callback(
    Output('model-name', 'children'),
    [Input('slope-slider', 'value'), Input('intercept-slider', 'value'),
     Input('p1-slider', 'value'), Input('p2-slider', 'value'), Input('p3-slider', 'value'),
     Input('p4-slider', 'value'), Input('A1-slider', 'value'), Input('A2-slider', 'value'),
     Input('A3-slider', 'value'), Input('A4-slider', 'value'), ]
)
def model_name(a, b, p1, p2, p3, p4, A1, A2, A3, A4):
    if A1 == 0 and A2 == 0 and A3 == 0 and A4 == 0:
        if a == 0 and p1 == 0 and p2 == 0 and p3 == 0 and p4 == 0:
            name = 'constant iso'
        elif b == 0 and p1 == 0 and p2 == 0 and p3 == 0 and p4 == 0:
            name = 'scaling iso'
        elif p1 == 0 and p2 == 0 and p3 == 0 and p4 == 0:
            name = 'full iso'
        elif p1 == 0 and p2 == 0:
            name = 'full relative'
        elif p3 == 0 and p4 == 0:
            name = 'full absolute'
        else:
            name = 'full full'
    else:
        if A3 == 0 and A4 == 0 and p3 == 0 and p4 == 0:
            name = 'full absolute amps'
        elif A1 == 0 and A2 == 0 and p1 == 0 and p2 == 0:
            name = 'full relative amps'
        else:
            name = 'full full amps'
    return [f'Model name: {name}']


@app.callback(
    Output('period-eqt', 'src'),
    [Input('slope-slider', 'value'), Input('intercept-slider', 'value'),
     Input('p1-slider', 'value'), Input('p2-slider', 'value'), Input('p3-slider', 'value'),
     Input('p4-slider', 'value'), ]
)
def period_eqt(a, b, p1, p2, p3, p4):
    """
    """
    if a == 0 and p1 == 0 and p2 == 0 and p3 == 0 and p4 == 0:
        path = op.join(eqts_dir, 'period-constant-iso.svg')
    elif b == 0 and p1 == 0 and p2 == 0 and p3 == 0 and p4 == 0:
        path = op.join(eqts_dir, 'period-scaling-iso.svg')
    elif p1 == 0 and p2 == 0 and p3 == 0 and p4 == 0:
        path = op.join(eqts_dir, 'period-full-iso.svg')
    elif p1 == 0 and p2 == 0:
        path = op.join(eqts_dir, 'period-full-relative.svg')
    elif p3 == 0 and p4 == 0:
        path = op.join(eqts_dir, 'period-full-absolute.svg')
    else:
        path = op.join(eqts_dir, 'period-full-full.svg')
    return functions.get_svg(path)


@app.callback(
    Output('amp-eqt', 'src'),
    [Input('A1-slider', 'value'), Input('A2-slider', 'value'), Input('A3-slider', 'value'),
     Input('A4-slider', 'value'), ]
)
def amp_eqt(A1, A2, A3, A4):
    """
    """
    if A1 == 0 and A2 == 0 and A3 == 0 and A4 == 0:
        path = op.join(eqts_dir, 'amp-none.svg')
    elif A1 == 0 and A2 == 0:
        path = op.join(eqts_dir, 'amp-relative.svg')
    elif A3 == 0 and A4 == 0:
        path = op.join(eqts_dir, 'amp-absolute.svg')
    else:
        path = op.join(eqts_dir, 'amp-full.svg')
    return functions.get_svg(path)


if __name__ == '__main__':
    app.run_server(debug=True)
