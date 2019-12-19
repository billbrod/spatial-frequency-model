import sfm
import dash
import dash_core_components as dcc
import dash_html_components as html


model = sfm.model.LogGaussianDonut()
df = sfm.analyze_model.create_preferred_period_df(model)


app = dash.Dash(__name__)

app.layout = html.Div(children=[
    html.H1(children='Hello Dash'),

    html.Div(children='''
        Dash: A web application framework for Python.
    '''),

    dcc.Graph(
        id='example-graph',
        figure={
            'data': [
                dict(x=df.query("`Stimulus type`==@i")['Eccentricity (deg)'],
                     y=df.query("`Stimulus type`==@i")['Eccentricity (deg)'],
                     name=i)
                for i in df['Stimulus type'].unique()
            ],
            'layout': {
                'title': 'Dash Data Visualization'
            }
        }
    )
])

if __name__ == '__main__':
    app.run_server(debug=True)
