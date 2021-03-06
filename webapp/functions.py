"""helper functions for the dash app
"""
import sfm
import base64
import numpy as np


def pdf_plot(model, reference_frame, vox_ecc, vox_angle):
    """Create plotly plot showing response of a single voxel

    Parameters
    ----------
    model : sfm.model.LogGaussianDonut
        single instance of the spatial frequency model
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
    data_dict = []
    orientations = np.linspace(0, np.pi, 4, endpoint=False)
    pal = sfm.plotting.stimulus_type_palette(reference_frame)
    if reference_frame == 'relative':
        pal.pop('mixtures')
        # in this case orientations contains relative orientations, that
        # is theta - phi (sf_angle - vox_angle), and so we have to add
        # phi / vox_angle back in order to get the actual sf_angle
        orientations = [o + vox_angle for o in orientations]
    elif reference_frame == 'absolute':
        pal.pop('off-diagonal')
    sf_mag = np.logspace(-10, 10, 1000, base=2)
    vox_ecc = vox_ecc * np.ones_like(sf_mag)
    vox_angle = vox_angle * np.ones_like(sf_mag)
    # the data component of our figure dict is a list, with one per line
    # in the plot. in our case, that's the different stimuli.
    for i, (k, v) in enumerate(pal.items()):
        sf_angle = orientations[i] * np.ones_like(sf_mag)
        response = model.evaluate(sf_mag, sf_angle, vox_ecc, vox_angle)
        data_dict.append(dict(x=sf_mag, y=response, name=k, line={'color': v}))
    return {
        'data': data_dict,
        'layout': {
            'xaxis': {'title': 'Spatial frequency (cpd)', 'type': 'log', 'basex': 2},
            'yaxis': {'title': 'Voxel response (arbitrary units)'},
            'title': 'Response of voxel at %.1f deg ecc, %.1f deg polar angle' %
            (vox_ecc[0], np.rad2deg(vox_angle[0])),
            'height': '400'
        }
    }


def cartesian_plot(model, reference_frame):
    """Create plotly plot showing preferred period as function of eccentricity

    We average over the retinotopic angle here, otherwise things look
    strange.

    Parameters
    ----------
    model : sfm.model.LogGaussianDonut
        single instance of the spatial frequency model
    reference_frame : {'relative', 'absolute'}
        Which reference frame to show.

    Returns
    -------
    figure : dict
        dict defining the plot, to be passed as the function object to a
        dcc.Graph component

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
    """Create plotly plot showing r as function of retinal angle

    This function is called for either creating the preferred period
    contour plots (`r='period'`) or the max amplitude plot(`r='amp'`).

    Ideally, we would label the radial and angular axes, but that's not
    working right now

    Parameters
    ----------
    model : sfm.model.LogGaussianDonut
        single instance of the spatial frequency model
    reference_frame : {'relative', 'absolute'}
        Which reference frame to show.
    r : {'amp', 'period'}
        the data to show on the radial axis of the polar plot

    Returns
    -------
    figure : dict
        dict defining the plot, to be passed as the function object to a
        dcc.Graph component

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
        df = sfm.analyze_model.create_preferred_period_df(model, reference_frame, eccentricity=[5],
                                                          retinotopic_angle=np.linspace(0, 2*np.pi, 49))
        return {
            'data': [
                dict(theta=df.query("`Stimulus type`==@i")['Retinotopic angle (rad)'],
                     thetaunit='radians', line={'color': pal[i]},
                     r=df.query("`Stimulus type`==@i")['Preferred period (dpc)'],
                     name=i, type='scatterpolar')
                for i in df['Stimulus type'].unique()
            ],
            'layout': {
                'angularaxis': {'title': 'Retinotopic angle'},
                'radialaxis': {'title': 'Eccentricity (deg)'},
                'title': 'Preferred period at eccentricity 5 deg',
            }
        }


def get_svg(path):
    """Get an svg image into format to properly display

    This formats the svg image found at `path` into the str necessary
    for displaying in a dcc.Img component.

    from https://github.com/plotly/dash/issues/537

    Parameters
    ----------
    path : str
        path to the svg image

    Returns
    -------
    src : str
        The str to pass as src to a dcc.Img component. Note that this
        will contain the decoded svg image (as text) and so will be
        quite long
    """
    svg = base64.b64encode(open(path, 'rb').read())
    return 'data:image/svg+xml;base64,{}'.format(svg.decode())
