import sfm
import base64
import numpy as np


def cartesian_plot(model, reference_frame, ):
    """
    """
    df = sfm.analyze_model.create_preferred_period_df(model, reference_frame,
                                                      eccentricity=np.linspace(0, 11, 48))
    # average over retinotopic angle for this plot
    gb_cols = ['Orientation (rad)', 'Eccentricity (deg)', 'reference_frame', 'Stimulus type']
    df = df.groupby(gb_cols).mean().reset_index()
    pal = sfm.plotting.stimulus_type_palette(reference_frame)
    return {
        'data': [
            dict(x=df.query("`Stimulus type`==@i")['Eccentricity (deg)'],
                 y=df.query("`Stimulus type`==@i")['Preferred period (dpc)'],
                 line={'color': pal[i]}, name=i)
            for i in df['Stimulus type'].unique()
        ],
        'layout': {
            'xaxis': {'title': 'Eccentricity (deg)'},
            'yaxis': {'title': 'Preferred period (dpc)'},
            'title': 'Preferred period at each eccentricity',
        }
    }


def polar_plot(model, reference_frame, r):
    """
    """
    pal = sfm.plotting.stimulus_type_palette(reference_frame)
    if r == 'amp':
        df = sfm.analyze_model.create_max_amplitude_df(model, reference_frame)
        return {
            'data': [
                dict(theta=df.query("`Stimulus type`==@i")['Retinotopic angle (rad)'],
                     thetaunit='radians',
                     r=df.query("`Stimulus type`==@i")['Max amplitude'],
                     name=i, type='scatterpolar', line={'color': pal[i]})
                for i in df['Stimulus type'].unique()
            ],
            'layout': {
                'angularaxis': {'title': 'Retinotopic angle'},
                'radialaxis': {'title': 'Max amplitude'},
                'title': 'Max amplitude as a function of retinotopic angle',
            }
        }
    elif r == 'period':
        df = sfm.analyze_model.create_preferred_period_contour_df(model, reference_frame,
                                                                  period_target=[1])
        return {
            'data': [
                dict(theta=df.query("`Stimulus type`==@i")['Retinotopic angle (rad)'],
                     thetaunit='radians', line={'color': pal[i]},
                     r=df.query("`Stimulus type`==@i")['Eccentricity (deg)'],
                     name=i, type='scatterpolar')
                for i in df['Stimulus type'].unique()
            ],
            'layout': {
                'angularaxis': {'title': 'Retinotopic angle', 'thetaunit': 'radians'},
                'radialaxis': {'title': 'Eccentricity (deg)'},
                'title': 'Iso-preferred-period contours (preferred period = 1 dpc)',
            }
        }


def get_svg(path):
    """from https://github.com/plotly/dash/issues/537
    """
    svg = base64.b64encode(open(path, 'rb').read())
    return 'data:image/svg+xml;base64,{}'.format(svg.decode())
