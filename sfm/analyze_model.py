#!/usr/bin/python
"""code to analyze the outputs of our 2d tuning model
"""
import pandas as pd
import numpy as np
import torch
import re
import os
import glob
from . import model as sfp_model


def load_LogGaussianDonut(save_path_stem):
    """this loads and returns the actual model, given the saved parameters, for analysis
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # we try and infer model type from the path name, which we can do assuming we used the
    # Snakefile to generate saved model.
    vary_amps_label = save_path_stem.split('_')[-1]
    if vary_amps_label == 'vary':
        vary_amps = True
    elif vary_amps_label == 'constant':
        vary_amps = False
    ecc_type = save_path_stem.split('_')[-2]
    ori_type = save_path_stem.split('_')[-3]
    model = sfp_model.LogGaussianDonut(ori_type, ecc_type, vary_amps)
    model.load_state_dict(torch.load(save_path_stem + '_model.pt', map_location=device.type))
    model.eval()
    model.to(device)
    return model


def load_single_model(save_path_stem, load_results_df=True):
    """load in the model, loss df, and model df found at the save_path_stem

    we also send the model to the appropriate device
    """
    try:
        if load_results_df:
            results_df = pd.read_csv(save_path_stem + '_results_df.csv')
        else:
            results_df = pd.read_csv(save_path_stem + '_results_df.csv', nrows=1)
    except FileNotFoundError as e:
        if load_results_df:
            raise e
        results_df = None
    loss_df = pd.read_csv(save_path_stem + '_loss.csv')
    model_history_df = pd.read_csv(save_path_stem + "_model_history.csv")
    if 'test_subset' not in loss_df.columns or 'test_subset' not in model_history_df.columns:
        # unclear why this happens, it's really strange
        if not save_path_stem.split('_')[-4].startswith('c'):
            raise Exception("Can't grab test_subset from path %s!" % save_path_stem)
        # this will give it the same spacing as the original version
        test_subset = ', '.join(save_path_stem.split('_')[-4][1:].split(','))
        if "test_subset" not in loss_df.columns:
            loss_df['test_subset'] = test_subset
        if "test_subset" not in model_history_df.columns:
            model_history_df['test_subset'] = test_subset
    model = load_LogGaussianDonut(save_path_stem)
    return model, loss_df, results_df, model_history_df


def combine_models(base_path_template, load_results_df=True):
    """load in many models and combine into dataframes

    returns: model_df, loss_df, results_df

    base_path_template: path template where we should find the results. should contain no string
    formatting symbols (e.g., "{0}" or "%s") but should contain at least one '*' because we will
    use glob to find them (and therefore should point to an actual file when passed to glob, one
    of: the loss df, model df, or model paramters).

    load_results_df: boolean. Whether to load the results_df or not. Set False if your results_df
    are too big and you're worried about having them all in memory. In this case, the returned
    results_df will be None.
    """
    models = []
    loss_df = []
    results_df = []
    model_history_df = []
    path_stems = []
    for p in glob.glob(base_path_template):
        path_stem = (p.replace('_loss.csv', '').replace('_model.pt', '')
                     .replace('_results_df.csv', '').replace('_model_history.csv', ''))
        # we do this to make sure we're not loading in the outputs of a model twice (by finding
        # both its loss.csv and its results_df.csv, for example)
        if path_stem in path_stems:
            continue
        # based on how these are saved, we can make some assumptions and grab extra info from their
        # paths
        metadata = {}
        if 'tuning_2d_simulated' in path_stem:
            metadata['modeling_goal'] = path_stem.split(os.sep)[-2]
        elif 'tuning_2d_model' in path_stem:
            metadata['session'] = path_stem.split(os.sep)[-2]
            metadata['subject'] = path_stem.split(os.sep)[-3]
            metadata['modeling_goal'] = path_stem.split(os.sep)[-4]
            metadata['mat_type'] = path_stem.split(os.sep)[-5]
            metadata['atlas_type'] = path_stem.split(os.sep)[-6]
            metadata['task'] = re.search('_(task-[a-z0-9]+)_', path_stem).groups()[0]
            metadata['indicator'] = str((metadata['subject'], metadata['session'], metadata['task'])).replace("'", "")
        path_stems.append(path_stem)
        model, loss, results, model_history = load_single_model(path_stem,
                                                                load_results_df=load_results_df)
        for k, v in metadata.items():
            if results is not None:
                results[k] = v
            loss[k] = v
            model_history[k] = v
        results_df.append(results)
        loss_df.append(loss)
        model_history_df.append(model_history)
        tmp = loss.head(1)
        tmp = tmp.drop(['epoch_num', 'batch_num', 'loss'], 1)
        tmp['model'] = model
        for name, val in model.named_parameters():
            tmper = tmp.copy()
            tmper['model_parameter'] = name
            tmper['fit_value'] = val.cpu().detach().numpy()
            if results is not None:
                if 'true_model_%s' % name in results.columns:
                    tmper['true_value'] = results['true_model_%s' % name].unique()[0]
            models.append(tmper)
    loss_df = pd.concat(loss_df).reset_index(drop=True)
    model_history_df = pd.concat(model_history_df).reset_index(drop=True)
    if load_results_df:
        results_df = pd.concat(results_df).reset_index(drop=True).drop('index', 1)
    else:
        results_df = None
    models = pd.concat(models)
    return models, loss_df, results_df, model_history_df


def _finish_feature_df(df, reference_frame='absolute'):
    """helper function to clean up the feature dataframes

    This helper function cleans up the feature dataframes so that they
    can be more easily used for plotting with feature_df_plot and
    feature_df_polar_plot functions. It performs the following actions:

    1. Adds reference_frame as column.

    2. Converts retinotopic angles to human-readable labels (only if
       default retinotopic angles used).

    3. Adds "Stimulus type" as column, giving human-readable labels
       based on "Orientation" columns.

    Parameters
    ----------
    df : pd.DataFrame
        The feature dataframe to finish up
    reference_frame : {'absolute, 'relative'}
        The reference frame of df

    Returns
    -------
    df : pd.DataFrame
        The cleaned up dataframe

    """
    if isinstance(df, list):
        df = pd.concat(df).reset_index(drop=True)
    df['reference_frame'] = reference_frame
    angle_ref = np.linspace(0, np.pi, 4, endpoint=False)
    angle_labels = ['0', r'$\frac{\pi}{4}$', r'$\frac{\pi}{2}$', r'$\frac{3\pi}{4}$']
    rel_labels = ['annuli', 'forward spiral', 'pinwheels', 'reverse spiral']
    abs_labels = ['vertical', 'forward diagonal', 'horizontal', 'reverse diagonal']
    if np.array_equiv(angle_ref, df["Retinotopic angle (rad)"].unique()):
        df["Retinotopic angle (rad)"] = df["Retinotopic angle (rad)"].map(dict((k, v) for k, v in
                                                                               zip(angle_ref,
                                                                                   angle_labels)))
    if reference_frame == 'relative':
        df["Stimulus type"] = df["Orientation (rad)"].map(dict((k, v) for k, v in
                                                               zip(angle_ref, rel_labels)))
    elif reference_frame == 'absolute':
        df["Stimulus type"] = df["Orientation (rad)"].map(dict((k, v) for k, v in
                                                               zip(angle_ref, abs_labels)))
    return df


def create_preferred_period_df(model, reference_frame='absolute',
                               retinotopic_angle=np.linspace(0, np.pi, 4, endpoint=False),
                               orientation=np.linspace(0, np.pi, 4, endpoint=False),
                               eccentricity=np.linspace(0, 11, 11)):
    """Create dataframe summarizing preferred period as function of eccentricity

    Generally, you should not call this function directly, but use
    create_feature_df. Differences from that function: this functions
    requires the initialized model and only creates the info for a
    single model, while create_feature_df uses the models dataframe to
    initialize models itself, and combines the outputs across multiple
    indicators.

    This function creates a dataframe summarizing the specified model's
    preferred period as a function of eccentricity, for multiple
    stimulus orientations (in either absolute or relative reference
    frames) and retinotopic angles. This dataframe is then used for
    creating plots to summarize the model.

    Unless you have something specific in mind, you can trust the
    default options for retinotopic_angle, orientation, and
    eccentricity.

    Parameters
    ----------
    model : sfp.model.LogGaussianDonut
        a single, initialized model, which we will summarize.
    reference_frame : {"absolute", "relative"}, optional
        Whether we use the absolute or relative reference frame in the
        feature dataframe; that is whether we consider conventional
        gratings (absolute, orientation is relative to
        vertical/horizontal), or our log-polar gratings (relative,
        orientation is relative to fovea).
    retinotopic_angle : np.array, optional
        Array specifying which retinotopic angles to find the preferred
        period for. If you don't care about retinotopic angle and just
        want to summarize the model's overall features, you should use
        the default (which includes all angles where the model can have
        different preferences, based on its parametrization) and then
        average over them.
    orientation : np.array, optional
        Array specifying which stimulus orientations to find the
        preferred period for. Note that the meaning of these
        orientations will differ depending on the value of
        reference_frame; you should most likely plot and interpret the
        output based on the "Stimulus type" column instead (which
        include strings like 'vertical'/'horizontal' or
        'annuli'/'pinwheels'). However, this mapping can only happen if
        the values in orientation line up with our stimuli (0, pi/4,
        pi/2, 3*pi/2), and thus it's especially recommended that you use
        the default value for this argument. If you don't care about
        orientation and just want to summarize the model's overall
        features, you should use the default (which includes all
        orientations where the model can have different preferences,
        based on its parametrization) and then average over them.
    eccentricity : np.array, optional
        Array specifying which eccentricities to find the preferred
        period for. The default values span the range of measurements
        for our experiment, but you can certainly go higher if you
        wish. Note that, for creating the plot of preferred period as a
        function of eccentricity, the model's predictions will always be
        linear and so you most likely only need 2 points. More are
        included because you may want to examine the preferred period at
        specific eccentricities

    Returns
    -------
    preferred_period_df : pd.DataFrame
        Dataframe containing preferred period of the model, to use with
        sfp.plotting.feature_df_plot for plotting preferred period as a
        function of eccentricity.

    """
    df = []
    for o in orientation:
        if reference_frame == 'absolute':
            tmp = model.preferred_period(eccentricity, retinotopic_angle, o)
        elif reference_frame == 'relative':
            tmp = model.preferred_period(eccentricity, retinotopic_angle, rel_sf_angle=o)
        tmp = pd.DataFrame(tmp.detach().numpy(), index=retinotopic_angle, columns=eccentricity)
        tmp = tmp.reset_index().rename(columns={'index': 'Retinotopic angle (rad)'})
        tmp['Orientation (rad)'] = o
        df.append(pd.melt(tmp, ['Retinotopic angle (rad)', 'Orientation (rad)'],
                          var_name='Eccentricity (deg)', value_name='Preferred period (dpc)'))
    return _finish_feature_df(df, reference_frame)


def create_max_amplitude_df(model, reference_frame='absolute',
                            retinotopic_angle=np.linspace(0, 2*np.pi, 49),
                            orientation=np.linspace(0, np.pi, 4, endpoint=False)):
    """Create dataframe summarizing max amplitude as function of retinotopic angle

    Generally, you should not call this function directly, but use
    create_feature_df. Differences from that function: this functions
    requires the initialized model and only creates the info for a
    single model, while create_feature_df uses the models dataframe to
    initialize models itself, and combines the outputs across multiple
    indicators.

    This function creates a dataframe summarizing the specified model's
    maximum amplitude as a function of retinotopic angle, for multiple
    stimulus orientations (in either absolute or relative reference
    frames). This dataframe is then used for creating plots to summarize
    the model.

    Unless you have something specific in mind, you can trust the
    default options for retinotopic_angle and orientation.

    Parameters
    ----------
    model : sfp.model.LogGaussianDonut
        a single, initialized model, which we will summarize.
    reference_frame : {"absolute", "relative"}, optional
        Whether we use the absolute or relative reference frame in the
        feature dataframe; that is whether we consider conventional
        gratings (absolute, orientation is relative to
        vertical/horizontal), or our log-polar gratings (relative,
        orientation is relative to fovea).
    retinotopic_angle : np.array, optional
        Array specifying which retinotopic angles to find the preferred
        period for. Note that the sampling of retinotopic angle is much
        finer than for create_preferred_period_df (and goes all the way
        to 2*pi), because this is what we will use as the dependent
        variable in our plotsl
    orientation : np.array, optional
        Array specifying which stimulus orientations to find the
        preferred period for. Note that the meaning of these
        orientations will differ depending on the value of
        reference_frame; you should most likely plot and interpret the
        output based on the "Stimulus type" column instead (which
        include strings like 'vertical'/'horizontal' or
        'annuli'/'pinwheels'). However, this mapping can only happen if
        the values in orientation line up with our stimuli (0, pi/4,
        pi/2, 3*pi/2), and thus it's especially recommended that you use
        the default value for this argument. If you don't care about
        orientation and just want to summarize the model's overall
        features, you should use the default (which includes all
        orientations where the model can have different preferences,
        based on its parametrization) and then average over them.

    Returns
    -------
    max_amplitude_df : pd.DataFrame
        Dataframe containing maximum amplitude of the model, to use with
        sfp.plotting.feature_df_polar_plot for plotting max amplitude as
        a function of retinotopic angle.

    """
    if reference_frame == 'absolute':
        tmp = model.max_amplitude(retinotopic_angle, orientation).detach().numpy()
    elif reference_frame == 'relative':
        tmp = model.max_amplitude(retinotopic_angle, rel_sf_angle=orientation).detach().numpy()
    tmp = pd.DataFrame(tmp, index=retinotopic_angle, columns=orientation)
    tmp = tmp.reset_index().rename(columns={'index': 'Retinotopic angle (rad)'})
    df = pd.melt(tmp, ['Retinotopic angle (rad)'], var_name='Orientation (rad)',
                 value_name='Max amplitude')
    return _finish_feature_df(df, reference_frame)


def create_feature_df(models, feature_type='preferred_period', reference_frame='absolute',
                      **kwargs):
    """Create dataframe to summarize the predictions made by our models

    The point of this dataframe is to generate plots (using
    plotting.feature_df_plot and plotting.feature_df_polar_plot) to
    easily visualize what the parameters of our model mean, either for
    demonstrative purposes or with the parameters fit to actual data.

    This is used to create a feature data frame that combines info
    across multiple models, using the "indicator" column to separate
    them, and serves as a wrapper around two other functions:
    create_preferred_period_df and create_max_amplitude_df (based on the
    value of the feature_type arg). We loop through the unique
    indicators in the models dataframe and instantiate a model for each
    one (thus, each indicator must only have one associated model). We
    then create dataframes summarizing the relevant features, add the
    indicator, and, concatenate.

    The intended use of these dataframes is to create plots showing the
    models' predictions for (using bootstraps to get confidence
    intervals to show variability across subjects):
    
    1. preferred period as a function of eccentricity:

    ```
    pref_period = create_feature_df(models, feature_type='preferred_period')
    sfp.plotting.feature_df_plot(pref_period)
    # average over retinotopic angle
    sfp.plotting.feature_df_plot(pref_period, col=None, 
                                 pre_boot_gb_func=np.mean)
    ```

    2. preferred period as a function of retinotopic angle and stimulus
       orientation:

    ```
    pref_period_contour = create_feature_df(models, 
                                            feature_type='preferred_period_contour')
    sfp.plotting.feature_df_polar_plot(pref_period_contour)
    ```

    3. max amplitude as a function of retinotopic angle and stimulus
       orientation:

    ```
    max_amp = create_feature_df(models, feature_type='max_amplitude')
    sfp.plotting.feature_df_polar_plot(max_amp, col=None, r='Max amplitude')
    ```

    Parameters
    ----------
    models : pd.DataFrame
        dataframe summarizing model fits across many subjects / sessions
        (as created by analyze_model.combine_models function). Must
        contain the indicator columns and a row for each of the model's
        11 parameters
    feature_type : {"preferred_period", "preferred_period_contour", "max_amplitude"}, optional
        Which feature dataframe to create. Determines which function we
        call, from create_preferred_period_df and create_max_amplitude_df
    reference_frame : {"absolute", "relative"}, optional
        Whether we use the absolute or relative reference frame in the
        feature dataframe; that is whether we consider conventional
        gratings (absolute, orientation is relative to
        vertical/horizontal), or our log-polar gratings (relative,
        orientation is relative to fovea).
    kwargs : {retinotopic_angle, orientation, eccentricity, period_target}
        passed to the various create_*_df functions. See their
        docstrings for more info. if not set, use the defaults.

    Returns
    -------
    feature_df : pd.DataFrame
        Dataframe containing specified feature info

    """
    df = []
    for ind in models.indicator.unique():
        m = sfp_model.LogGaussianDonut.init_from_df(models.query('indicator==@ind'))
        if feature_type == 'preferred_period':
            df.append(create_preferred_period_df(m, reference_frame, **kwargs))
        elif feature_type == 'preferred_period_contour':
            df.append(create_preferred_period_df(m, reference_frame, eccentricity=[5],
                                                 retinotopic_angle=np.linspace(0, 2*np.pi, 49),
                                                 **kwargs))
        elif feature_type == 'max_amplitude':
            df.append(create_max_amplitude_df(m, reference_frame, **kwargs))
        df[-1]['indicator'] = ind
    return pd.concat(df).reset_index(drop=True)
