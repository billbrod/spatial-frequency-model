"""Dash app for exploring the spatial frequency preferences model

Usage:

`python app.py`

App will be running at 0.0.0.0:8050 (equivalent to localhost:8050) and
can be viewed in the browser

This script accepts no arguments

"""
import sfm
import functions
import dash
import numpy as np
from dash import dcc
import os.path as op
from dash import html
from dash.dependencies import Input, Output


app = dash.Dash(__name__, suppress_callback_exceptions=True,
                url_base_pathname='/spatial-frequency-model/')

image_dir = op.join(op.dirname(op.realpath(__file__)), '..', 'images')
eqts_dir = op.join(image_dir, 'equations')
main_eqt = functions.get_svg(op.join(eqts_dir, 'main.svg'))
rel_legend = functions.get_svg(op.join(image_dir, 'stimulus-legend-relative.svg'))
abs_legend = functions.get_svg(op.join(image_dir, 'stimulus-legend-absolute.svg'))

app.layout = html.Div([
    html.H1('Spatial Frequency Preferences Model'),
    # Graph showing single voxel response and components to change the
    # reference frame and voxel location
    html.Div([dcc.RadioItems(id='ref-frame', value='relative',
                             options=[{'label': i.capitalize(), 'value': i}
                                      for i in ['relative', 'absolute']]),
              dcc.Input(id='vox-ecc', type='number', placeholder='Voxel eccentricity (deg)',
                        min=0, max=90),
              dcc.Input(id='vox-angle', type='number', placeholder='Voxel angle (deg)',
                        min=0, max=360, step=1),
              dcc.Graph(id='pdf')],
             style={'width': '45%', 'display': 'inline-block', 'float': 'right',
                    'padding-right': '50px'}),
    # Equations of the model
    html.Div([html.P(id='model-type'), html.Img(src=main_eqt), html.Img(id='period-eqt'),
              html.Img(id='amp-eqt')],
             style={'width': '45%', 'display': 'inline-block', 'float': 'left',
                    'padding-bottom': '55px', 'padding-top': '75px', 'padding-left': '50px'}),

    # Pictures of legends
    html.Div([html.P(children=['Example stimuli']), html.Img(src=rel_legend), html.Img(src=abs_legend)],
             style={'display': 'inline-block', 'width': '45%',
                    'padding-left': '50px', 'padding-bottom': '20px'}),

    # First group of parameter sliders
    html.Div([
        dcc.Markdown('\u03c3'),
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
    ], style={'width': '31.5%', 'padding': '20px 1vw', 'display': 'inline-block', 'float': 'left'}
    ),
    # Second group of parameter sliders
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
    ], style={'width': '31.5%', 'padding': '20px 1vw', 'display': 'inline-block'}
    ),
    # Third group of parameter sliders
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
    ], style={'width': '31.5%', 'padding': '20px 0vw', 'display': 'inline-block', 'float': 'right'}
    ),
    # Graphs
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
    """Plot showing the relative reference frame preferred period

    Parameters
    ----------
    sigma : float
        model's sigma parameter, giving its standard deviation in
        octaves
    a : float
        model's sf_ecc_slope parameter, giving the slope of the linear
        relationship between preferred period and eccentricity
    b : float
        model's sf_ecc_intercept parameter, giving the intercept of the
        linear relationship between preferred period and eccentricity
    p1 : float
        model's abs_mode_cardinals parameter, controlling the effect of
        the cardinal absolute orientations on preferred period (i.e.,
        horizontal vs. vertical)
    p2 : float
        model's abs_mode_obliques parameter, controlling the effect of
        the absolute cardinals vs obliques orientations on preferred
        period (i.e., horizontal/vertical vs. diagonal)
    p3 : float
        model's rel_mode_cardinals parameter, controlling the effect of
        the cardinal relative orientations on preferred period (i.e.,
        radial vs. angular)
    p4 : float
        model's rel_mode_obliques parameter, controlling the effect of
        the relative cardinals vs obliques orientations on preferred
        period (i.e., radial/angular vs. spirals)
    A1 : float
        model's abs_amplitude_cardinals parameter, controlling the
        effect of the cardinal absolute orientations on max amplitude
        (i.e., horizontal vs. vertical)
    A2 : float
        model's abs_amplitude_obliques parameter, controlling the effect
        of the absolute cardinals vs obliques orientations on max
        amplitude (i.e., horizontal/vertical vs. diagonal)
    A3 : float
        model's rel_amplitude_cardinals parameter, controlling the
        effect of the cardinal relative orientations on max amplitude
        (i.e., radial vs. angular)
    A4 : float
        model's rel_amplitude_obliques parameter, controlling the effect
        of the relative cardinals vs obliques orientations on max
        amplitude (i.e., radial/angular vs. spirals)

    Returns
    -------
    figure : dict
        dict defining the plot, to be passed as the function object to a
        dcc.Graph component

    """
    model = sfm.model.LogGaussianDonut('full', 'full', 'full', sigma, a, b, p1, p2, p3, p4,
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
    """Plot showing the absolute reference frame preferred period

    Parameters
    ----------
    sigma : float
        model's sigma parameter, giving its standard deviation in
        octaves
    a : float
        model's sf_ecc_slope parameter, giving the slope of the linear
        relationship between preferred period and eccentricity
    b : float
        model's sf_ecc_intercept parameter, giving the intercept of the
        linear relationship between preferred period and eccentricity
    p1 : float
        model's abs_mode_cardinals parameter, controlling the effect of
        the cardinal absolute orientations on preferred period (i.e.,
        horizontal vs. vertical)
    p2 : float
        model's abs_mode_obliques parameter, controlling the effect of
        the absolute cardinals vs obliques orientations on preferred
        period (i.e., horizontal/vertical vs. diagonal)
    p3 : float
        model's rel_mode_cardinals parameter, controlling the effect of
        the cardinal relative orientations on preferred period (i.e.,
        radial vs. angular)
    p4 : float
        model's rel_mode_obliques parameter, controlling the effect of
        the relative cardinals vs obliques orientations on preferred
        period (i.e., radial/angular vs. spirals)
    A1 : float
        model's abs_amplitude_cardinals parameter, controlling the
        effect of the cardinal absolute orientations on max amplitude
        (i.e., horizontal vs. vertical)
    A2 : float
        model's abs_amplitude_obliques parameter, controlling the effect
        of the absolute cardinals vs obliques orientations on max
        amplitude (i.e., horizontal/vertical vs. diagonal)
    A3 : float
        model's rel_amplitude_cardinals parameter, controlling the
        effect of the cardinal relative orientations on max amplitude
        (i.e., radial vs. angular)
    A4 : float
        model's rel_amplitude_obliques parameter, controlling the effect
        of the relative cardinals vs obliques orientations on max
        amplitude (i.e., radial/angular vs. spirals)

    Returns
    -------
    figure : dict
        dict defining the plot, to be passed as the function object to a
        dcc.Graph component

    """
    model = sfm.model.LogGaussianDonut('full', 'full', 'full', sigma, a, b, p1, p2, p3, p4,
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
    """Plot showing the relative reference frame preferred period contours

    This shows the eccentricity at which the model has preferred period
    1, for each stimulus orientation and retinal angle.

    Parameters
    ----------
    sigma : float
        model's sigma parameter, giving its standard deviation in
        octaves
    a : float
        model's sf_ecc_slope parameter, giving the slope of the linear
        relationship between preferred period and eccentricity
    b : float
        model's sf_ecc_intercept parameter, giving the intercept of the
        linear relationship between preferred period and eccentricity
    p1 : float
        model's abs_mode_cardinals parameter, controlling the effect of
        the cardinal absolute orientations on preferred period (i.e.,
        horizontal vs. vertical)
    p2 : float
        model's abs_mode_obliques parameter, controlling the effect of
        the absolute cardinals vs obliques orientations on preferred
        period (i.e., horizontal/vertical vs. diagonal)
    p3 : float
        model's rel_mode_cardinals parameter, controlling the effect of
        the cardinal relative orientations on preferred period (i.e.,
        radial vs. angular)
    p4 : float
        model's rel_mode_obliques parameter, controlling the effect of
        the relative cardinals vs obliques orientations on preferred
        period (i.e., radial/angular vs. spirals)
    A1 : float
        model's abs_amplitude_cardinals parameter, controlling the
        effect of the cardinal absolute orientations on max amplitude
        (i.e., horizontal vs. vertical)
    A2 : float
        model's abs_amplitude_obliques parameter, controlling the effect
        of the absolute cardinals vs obliques orientations on max
        amplitude (i.e., horizontal/vertical vs. diagonal)
    A3 : float
        model's rel_amplitude_cardinals parameter, controlling the
        effect of the cardinal relative orientations on max amplitude
        (i.e., radial vs. angular)
    A4 : float
        model's rel_amplitude_obliques parameter, controlling the effect
        of the relative cardinals vs obliques orientations on max
        amplitude (i.e., radial/angular vs. spirals)

    Returns
    -------
    figure : dict
        dict defining the plot, to be passed as the function object to a
        dcc.Graph component

    """
    model = sfm.model.LogGaussianDonut('full', 'full', 'full', sigma, a, b, p1, p2, p3, p4,
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
    """Plot showing the absolute reference frame preferred period contours

    This shows the eccentricity at which the model has preferred period
    1, for each stimulus orientation and retinal angle.

    Parameters
    ----------
    sigma : float
        model's sigma parameter, giving its standard deviation in
        octaves
    a : float
        model's sf_ecc_slope parameter, giving the slope of the linear
        relationship between preferred period and eccentricity
    b : float
        model's sf_ecc_intercept parameter, giving the intercept of the
        linear relationship between preferred period and eccentricity
    p1 : float
        model's abs_mode_cardinals parameter, controlling the effect of
        the cardinal absolute orientations on preferred period (i.e.,
        horizontal vs. vertical)
    p2 : float
        model's abs_mode_obliques parameter, controlling the effect of
        the absolute cardinals vs obliques orientations on preferred
        period (i.e., horizontal/vertical vs. diagonal)
    p3 : float
        model's rel_mode_cardinals parameter, controlling the effect of
        the cardinal relative orientations on preferred period (i.e.,
        radial vs. angular)
    p4 : float
        model's rel_mode_obliques parameter, controlling the effect of
        the relative cardinals vs obliques orientations on preferred
        period (i.e., radial/angular vs. spirals)
    A1 : float
        model's abs_amplitude_cardinals parameter, controlling the
        effect of the cardinal absolute orientations on max amplitude
        (i.e., horizontal vs. vertical)
    A2 : float
        model's abs_amplitude_obliques parameter, controlling the effect
        of the absolute cardinals vs obliques orientations on max
        amplitude (i.e., horizontal/vertical vs. diagonal)
    A3 : float
        model's rel_amplitude_cardinals parameter, controlling the
        effect of the cardinal relative orientations on max amplitude
        (i.e., radial vs. angular)
    A4 : float
        model's rel_amplitude_obliques parameter, controlling the effect
        of the relative cardinals vs obliques orientations on max
        amplitude (i.e., radial/angular vs. spirals)

    Returns
    -------
    figure : dict
        dict defining the plot, to be passed as the function object to a
        dcc.Graph component

    """
    model = sfm.model.LogGaussianDonut('full', 'full', 'full', sigma, a, b, p1, p2, p3, p4,
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
    """Plot showing the relative reference frame max amplitude

    This shows the model's maximum amplitude for each stimulus
    orientation and retinal angle.

    Parameters
    ----------
    sigma : float
        model's sigma parameter, giving its standard deviation in
        octaves
    a : float
        model's sf_ecc_slope parameter, giving the slope of the linear
        relationship between preferred period and eccentricity
    b : float
        model's sf_ecc_intercept parameter, giving the intercept of the
        linear relationship between preferred period and eccentricity
    p1 : float
        model's abs_mode_cardinals parameter, controlling the effect of
        the cardinal absolute orientations on preferred period (i.e.,
        horizontal vs. vertical)
    p2 : float
        model's abs_mode_obliques parameter, controlling the effect of
        the absolute cardinals vs obliques orientations on preferred
        period (i.e., horizontal/vertical vs. diagonal)
    p3 : float
        model's rel_mode_cardinals parameter, controlling the effect of
        the cardinal relative orientations on preferred period (i.e.,
        radial vs. angular)
    p4 : float
        model's rel_mode_obliques parameter, controlling the effect of
        the relative cardinals vs obliques orientations on preferred
        period (i.e., radial/angular vs. spirals)
    A1 : float
        model's abs_amplitude_cardinals parameter, controlling the
        effect of the cardinal absolute orientations on max amplitude
        (i.e., horizontal vs. vertical)
    A2 : float
        model's abs_amplitude_obliques parameter, controlling the effect
        of the absolute cardinals vs obliques orientations on max
        amplitude (i.e., horizontal/vertical vs. diagonal)
    A3 : float
        model's rel_amplitude_cardinals parameter, controlling the
        effect of the cardinal relative orientations on max amplitude
        (i.e., radial vs. angular)
    A4 : float
        model's rel_amplitude_obliques parameter, controlling the effect
        of the relative cardinals vs obliques orientations on max
        amplitude (i.e., radial/angular vs. spirals)

    Returns
    -------
    figure : dict
        dict defining the plot, to be passed as the function object to a
        dcc.Graph component

    """
    model = sfm.model.LogGaussianDonut('full', 'full', 'full', sigma, a, b, p1, p2, p3, p4,
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
    """Plot showing the absolute reference frame max amplitude

    This shows the model's maximum amplitude for each stimulus
    orientation and retinal angle.

    Parameters
    ----------
    sigma : float
        model's sigma parameter, giving its standard deviation in
        octaves
    a : float
        model's sf_ecc_slope parameter, giving the slope of the linear
        relationship between preferred period and eccentricity
    b : float
        model's sf_ecc_intercept parameter, giving the intercept of the
        linear relationship between preferred period and eccentricity
    p1 : float
        model's abs_mode_cardinals parameter, controlling the effect of
        the cardinal absolute orientations on preferred period (i.e.,
        horizontal vs. vertical)
    p2 : float
        model's abs_mode_obliques parameter, controlling the effect of
        the absolute cardinals vs obliques orientations on preferred
        period (i.e., horizontal/vertical vs. diagonal)
    p3 : float
        model's rel_mode_cardinals parameter, controlling the effect of
        the cardinal relative orientations on preferred period (i.e.,
        radial vs. angular)
    p4 : float
        model's rel_mode_obliques parameter, controlling the effect of
        the relative cardinals vs obliques orientations on preferred
        period (i.e., radial/angular vs. spirals)
    A1 : float
        model's abs_amplitude_cardinals parameter, controlling the
        effect of the cardinal absolute orientations on max amplitude
        (i.e., horizontal vs. vertical)
    A2 : float
        model's abs_amplitude_obliques parameter, controlling the effect
        of the absolute cardinals vs obliques orientations on max
        amplitude (i.e., horizontal/vertical vs. diagonal)
    A3 : float
        model's rel_amplitude_cardinals parameter, controlling the
        effect of the cardinal relative orientations on max amplitude
        (i.e., radial vs. angular)
    A4 : float
        model's rel_amplitude_obliques parameter, controlling the effect
        of the relative cardinals vs obliques orientations on max
        amplitude (i.e., radial/angular vs. spirals)

    Returns
    -------
    figure : dict
        dict defining the plot, to be passed as the function object to a
        dcc.Graph component

    """
    model = sfm.model.LogGaussianDonut('full', 'full', 'full', sigma, a, b, p1, p2, p3, p4,
                                       A1, A2, A3, A4)
    return functions.polar_plot(model, 'absolute', 'amp')


@app.callback(
    Output('pdf', 'figure'),
    [Input('sigma-slider', 'value'),
     Input('slope-slider', 'value'), Input('intercept-slider', 'value'),
     Input('p1-slider', 'value'), Input('p2-slider', 'value'), Input('p3-slider', 'value'),
     Input('p4-slider', 'value'), Input('A1-slider', 'value'), Input('A2-slider', 'value'),
     Input('A3-slider', 'value'), Input('A4-slider', 'value'), Input('ref-frame', 'value'),
     Input('vox-ecc', 'value'), Input('vox-angle', 'value')]
)
def pdf_graph(sigma, a, b, p1, p2, p3, p4, A1, A2, A3, A4, reference_frame, vox_ecc, vox_angle):
    """Plot showing the responses of a single voxel

    This plot shows the response the model predicts for a single voxel.

    Parameters
    ----------
    sigma : float
        model's sigma parameter, giving its standard deviation in
        octaves
    a : float
        model's sf_ecc_slope parameter, giving the slope of the linear
        relationship between preferred period and eccentricity
    b : float
        model's sf_ecc_intercept parameter, giving the intercept of the
        linear relationship between preferred period and eccentricity
    p1 : float
        model's abs_mode_cardinals parameter, controlling the effect of
        the cardinal absolute orientations on preferred period (i.e.,
        horizontal vs. vertical)
    p2 : float
        model's abs_mode_obliques parameter, controlling the effect of
        the absolute cardinals vs obliques orientations on preferred
        period (i.e., horizontal/vertical vs. diagonal)
    p3 : float
        model's rel_mode_cardinals parameter, controlling the effect of
        the cardinal relative orientations on preferred period (i.e.,
        radial vs. angular)
    p4 : float
        model's rel_mode_obliques parameter, controlling the effect of
        the relative cardinals vs obliques orientations on preferred
        period (i.e., radial/angular vs. spirals)
    A1 : float
        model's abs_amplitude_cardinals parameter, controlling the
        effect of the cardinal absolute orientations on max amplitude
        (i.e., horizontal vs. vertical)
    A2 : float
        model's abs_amplitude_obliques parameter, controlling the effect
        of the absolute cardinals vs obliques orientations on max
        amplitude (i.e., horizontal/vertical vs. diagonal)
    A3 : float
        model's rel_amplitude_cardinals parameter, controlling the
        effect of the cardinal relative orientations on max amplitude
        (i.e., radial vs. angular)
    A4 : float
        model's rel_amplitude_obliques parameter, controlling the effect
        of the relative cardinals vs obliques orientations on max
        amplitude (i.e., radial/angular vs. spirals)
    reference_frame : {'relative', 'absolute'}
        Which reference frame to show.
    vox_ecc : float
        Eccentricity of the voxel whose response we're displaying
    vox_angle : float
        Polar angle of the voxel whose response we're displaying

    Returns
    -------
    figure : dict
        dict defining the plot, to be passed as the function object to a
        dcc.Graph component

    """
    model = sfm.model.LogGaussianDonut('full', 'full', 'full', sigma, a, b, p1, p2, p3, p4,
                                       A1, A2, A3, A4)
    if vox_ecc is None:
        vox_ecc = 1
    if vox_angle is None:
        vox_angle = 0
    vox_angle = np.deg2rad(vox_angle)
    return functions.pdf_plot(model, reference_frame, vox_ecc, vox_angle)


@app.callback(
    Output('model-type', 'children'),
    [Input('slope-slider', 'value'), Input('intercept-slider', 'value'),
     Input('p1-slider', 'value'), Input('p2-slider', 'value'), Input('p3-slider', 'value'),
     Input('p4-slider', 'value'), Input('A1-slider', 'value'), Input('A2-slider', 'value'),
     Input('A3-slider', 'value'), Input('A4-slider', 'value'), ]
)
def model_type(a, b, p1, p2, p3, p4, A1, A2, A3, A4):
    """Get the model type

    We fit 9 different variants of the model, which differ from each
    other in the number of parameters we fit (the others are assumed to
    be constant, at 0). This function returns a str describing which the
    model type for the set of parameters. The model's are nested
    versions of each other (i.e., the most complicated model can fit all
    parameters, and thus the simpler models are all instances of the
    most complicated one), so we return the simplest possible model type
    (that is, the model with the fewest free parameters). For example,
    if all the ps and As are 0, and a and b are nonzero, the model could
    technically be any of: full iso, full absolute, full relative, full
    full, full absolute amps, full relative amps, or full full
    amps. However, because full iso is the simplest (with only two free
    parameters), this is the type returned.

    Parameters
    ----------
    sigma : float
        model's sigma parameter, giving its standard deviation in
        octaves
    a : float
        model's sf_ecc_slope parameter, giving the slope of the linear
        relationship between preferred period and eccentricity
    b : float
        model's sf_ecc_intercept parameter, giving the intercept of the
        linear relationship between preferred period and eccentricity
    p1 : float
        model's abs_mode_cardinals parameter, controlling the effect of
        the cardinal absolute orientations on preferred period (i.e.,
        horizontal vs. vertical)
    p2 : float
        model's abs_mode_obliques parameter, controlling the effect of
        the absolute cardinals vs obliques orientations on preferred
        period (i.e., horizontal/vertical vs. diagonal)
    p3 : float
        model's rel_mode_cardinals parameter, controlling the effect of
        the cardinal relative orientations on preferred period (i.e.,
        radial vs. angular)
    p4 : float
        model's rel_mode_obliques parameter, controlling the effect of
        the relative cardinals vs obliques orientations on preferred
        period (i.e., radial/angular vs. spirals)
    A1 : float
        model's abs_amplitude_cardinals parameter, controlling the
        effect of the cardinal absolute orientations on max amplitude
        (i.e., horizontal vs. vertical)
    A2 : float
        model's abs_amplitude_obliques parameter, controlling the effect
        of the absolute cardinals vs obliques orientations on max
        amplitude (i.e., horizontal/vertical vs. diagonal)
    A3 : float
        model's rel_amplitude_cardinals parameter, controlling the
        effect of the cardinal relative orientations on max amplitude
        (i.e., radial vs. angular)
    A4 : float
        model's rel_amplitude_obliques parameter, controlling the effect
        of the relative cardinals vs obliques orientations on max
        amplitude (i.e., radial/angular vs. spirals)

    Returns
    -------
    type : str
        Model type

    """
    if A1 == 0 and A2 == 0 and A3 == 0 and A4 == 0:
        if a == 0 and p1 == 0 and p2 == 0 and p3 == 0 and p4 == 0:
            name = '1 (constant period-iso amps-iso)'
        elif b == 0 and p1 == 0 and p2 == 0 and p3 == 0 and p4 == 0:
            name = '2 (scaling period-iso amps-iso)'
        elif p1 == 0 and p2 == 0 and p3 == 0 and p4 == 0:
            name = '3 (full period-iso amps-iso)'
        elif p1 == 0 and p2 == 0:
            name = '4 (full period-absolute amps-iso)'
        elif p3 == 0 and p4 == 0:
            name = '5 (full period-relative amps-iso)'
        else:
            name = '6 (full period-full amps-iso)'
    elif A3 == 0 and A4 == 0:
        if p1 == 0 and p2 == 0 and p3 == 0 and p4 == 0:
            name = '7 (full period-iso amps-absolute)'
        if p3 == 0 and p4 == 0:
            name = '8 (full period-absolute amps-absolute)'
        else:
            name = '9 (full period-full amps-absolute)'
    else:
       
        if p1 == 0 and p2 == 0 and p3 == 0 and p4 == 0 and A1 == 0 and A2 == 0:
            name = '3-alt (full period-iso amps-relative)'
        elif p3 == 0 and p4 == 0 and A1 == 0 and A2 == 0:
            name = '5-alt (full period-relative amps-relative)'
        elif A1 == 0 and A2 == 0:
            name = '6-alt (full period-full amps-relative)'
        elif p1 == 0 and p2 == 0 and p3 == 0 and p4 == 0:
            name = '7-alt (full period-iso amps-full)'
        else:
            name = '9-alt (full period-full amps-full)'
    return [f'Model type: {name}']


@app.callback(
    Output('period-eqt', 'src'),
    [Input('slope-slider', 'value'), Input('intercept-slider', 'value'),
     Input('p1-slider', 'value'), Input('p2-slider', 'value'), Input('p3-slider', 'value'),
     Input('p4-slider', 'value'), ]
)
def period_eqt(a, b, p1, p2, p3, p4):
    """Return the src for the image showing the period equation

    In order to help explain what the model type means, we change the
    equation showing the preferred period to gray out those parameters
    that are fixed at 0. All of these equations have been pre-generated
    as svg images, so this function just checks the relevant parameters,
    grabs the appropriate svg file, and converts it into the proper
    format for passing to the dcc.Img component

    Parameters
    ----------
    sigma : float
        model's sigma parameter, giving its standard deviation in
        octaves
    a : float
        model's sf_ecc_slope parameter, giving the slope of the linear
        relationship between preferred period and eccentricity
    b : float
        model's sf_ecc_intercept parameter, giving the intercept of the
        linear relationship between preferred period and eccentricity
    p1 : float
        model's abs_mode_cardinals parameter, controlling the effect of
        the cardinal absolute orientations on preferred period (i.e.,
        horizontal vs. vertical)
    p2 : float
        model's abs_mode_obliques parameter, controlling the effect of
        the absolute cardinals vs obliques orientations on preferred
        period (i.e., horizontal/vertical vs. diagonal)
    p3 : float
        model's rel_mode_cardinals parameter, controlling the effect of
        the cardinal relative orientations on preferred period (i.e.,
        radial vs. angular)
    p4 : float
        model's rel_mode_obliques parameter, controlling the effect of
        the relative cardinals vs obliques orientations on preferred
        period (i.e., radial/angular vs. spirals)

    Returns
    -------
    src : str
        The str to pass as src to a dcc.Img component. Note that this
        will contain the decoded svg image (as text) and so will be
        quite long

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
    """Return the src for the image showing the max amplitude equation

    In order to help explain what the model type means, we change the
    equation showing the max amplitude to gray out those parameters that
    are fixed at 0. All of these equations have been pre-generated as
    svg images, so this function just checks the relevant parameters,
    grabs the appropriate svg file, and converts it into the proper
    format for passing to the dcc.Img component

    Parameters
    ----------
    A1 : float
        model's abs_amplitude_cardinals parameter, controlling the
        effect of the cardinal absolute orientations on max amplitude
        (i.e., horizontal vs. vertical)
    A2 : float
        model's abs_amplitude_obliques parameter, controlling the effect
        of the absolute cardinals vs obliques orientations on max
        amplitude (i.e., horizontal/vertical vs. diagonal)
    A3 : float
        model's rel_amplitude_cardinals parameter, controlling the
        effect of the cardinal relative orientations on max amplitude
        (i.e., radial vs. angular)
    A4 : float
        model's rel_amplitude_obliques parameter, controlling the effect
        of the relative cardinals vs obliques orientations on max
        amplitude (i.e., radial/angular vs. spirals)

    Returns
    -------
    src : str
        The str to pass as src to a dcc.Img component. Note that this
        will contain the decoded svg image (as text) and so will be
        quite long

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
    app.run_server(host='0.0.0.0', debug=True, port=8050)
