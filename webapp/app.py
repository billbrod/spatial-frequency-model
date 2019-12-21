import sfm
import dash
import dash_core_components as dcc
import dash_html_components as html


model = sfm.model.LogGaussianDonut('full', sf_ecc_slope=.15, sf_ecc_intercept=.35,
                                   rel_mode_cardinals=.1)
per_df = sfm.analyze_model.create_preferred_period_df(model, 'relative')
contour_df = sfm.analyze_model.create_preferred_period_contour_df(model, 'relative',
                                                                  period_target=[1])
amp_df = sfm.analyze_model.create_max_amplitude_df(model, 'relative')

app = dash.Dash(__name__)

app.layout = html.Div(children=[
    html.H1('Hello Dash'),

    html.Div([
        dcc.Graph(
            id='period-graph',
            figure={
                'data': [
                    dict(x=per_df.query("`Stimulus type`==@i")['Eccentricity (deg)'],
                         y=per_df.query("`Stimulus type`==@i")['Preferred period (dpc)'],
                         name=i)
                    for i in per_df['Stimulus type'].unique()
                ],
                'layout': {
                    'xaxis': {'title': 'Eccentricity (deg)'},
                    'yaxis': {'title': 'Preferred period (dpc)'},
                }
            }
        )], style={'width': '33%', 'display': 'inline-block'}),

    html.Div([
        dcc.Graph(
            id='period-contour-graph',
            figure={
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
                }
            }
        )], style={'width': '33%', 'display': 'inline-block'}),

    html.Div([
        dcc.Graph(
            id='amp-contour-graph',
            figure={
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
                }
            }
        )
    ], style={'width': '33%', 'display': 'inline-block'})
])

if __name__ == '__main__':
    app.run_server(debug=True)
