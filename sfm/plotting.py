#!/usr/bin/python
"""high-level functions to make relevant plots
"""
import itertools
import seaborn as sns

LOGPOLAR_SUPERCLASS_ORDER = ['radial', 'forward spiral', 'angular', 'reverse spiral', 'mixtures']
CONSTANT_SUPERCLASS_ORDER = ['vertical', 'forward diagonal', 'horizontal', 'reverse diagonal',
                             'off-diagonal']
PARAM_ORDER = (['sigma', 'sf_ecc_slope', 'sf_ecc_intercept'] +
               ['%s_%s_%s' % (i, j, k) for j, i, k in
                itertools.product(['amplitude', 'mode'], ['abs', 'rel'],
                                  ['cardinals', 'obliques'])])


def stimulus_type_palette(reference_frame):
    palette = {}
    if isinstance(reference_frame, str):
        reference_frame = [reference_frame]
    if 'relative' in reference_frame:
        pal = sns.color_palette('deep', 5)
        palette.update(dict(zip(LOGPOLAR_SUPERCLASS_ORDER, pal)))
    if 'absolute' in reference_frame:
        pal = sns.color_palette('cubehelix', 5)
        palette.update(dict(zip(CONSTANT_SUPERCLASS_ORDER, pal)))
    return palette


def stimulus_type_order(reference_frame):
    order = []
    if isinstance(reference_frame, str):
        reference_frame = [reference_frame]
    for t in reference_frame:
        order.extend({'relative': LOGPOLAR_SUPERCLASS_ORDER,
                      'absolute': CONSTANT_SUPERCLASS_ORDER}[t])
    return order


def feature_df_plot(feature_df, hue="Stimulus type", col='Retinotopic angle (rad)', row=None,
                    plot_func=sns.lineplot, x='Eccentricity (deg)', y='Preferred period (dpc)',
                    yticks=[0, 1, 2], xticks=[0, 2, 4, 6, 8, 10], height=4, aspect=1,
                    title='Preferred period', top=.85, pal=None, col_order=None, row_order=None,
                    ylim=None, xlim=None, ci=68, n_boot=10000, col_wrap=None, pre_boot_gb_func=None,
                    pre_boot_gb_cols=['indicator', 'reference_frame', 'Stimulus type',
                                      'Eccentricity (deg)']):
    """Create plot from feature_df

    This function takes the feature_df created by
    sfp.analyze_model.create_feature_df and makes summary plots. The
    default should do more or less what you want it to, but there's a
    lot of customizability.

    Note that this makes a non-polar plot (it plots y as a function of
    x), and so the intended use is for the preferred_period
    feature_df. For the preferred period and max amplitude contours, use
    feature_df_polar_plot

    The majority of the arguments are passed right to sns.FacetGrid

    Parameters
    ----------
    feature_df : pd.DataFrame
        The feature dataframe, containing the preferred period as a
        function of eccentricity, at multiple stimulus orientation and
        retinotopic angles
    hue : str, optional
        a column in feature_df, which feature to use for the hue of plot
    col : str, optional
        a column in feature_df, which feature to facet on the columns
    row : str, optional
        a column in feature_df, which feature to facet on the rows
    plot_func : callable, optional
        The plot function to map on the FacetGrid. First two args should
        be x and y, should accept ci and n_boot kwargs (many seaborn
        plotting functions would work for this)
    x : str, optional
        a column in feature_df, which feature to plot on the x-axis
    y : str, optional
        a column in feature_df, which feature to plot on the y-axis
    {y, x}ticks : list, optional
        list of floats, which y- and x-ticks to include on the plot
    height : float, optional
        The height of each individual subplot
    aspect : float, optional
        The aspect ratio of each individual subplot
    title : str or None, optional
        The super-title of the plot. If None, we don't add a
        super-title, and we will not adjust the subplot spacing
    top : float, optional
        The amount to adjust the subplot spacing at the top so the title
        is above the subplots (with a call to
        g.fig.subplots_adjust(top=top)). If title is None, this is
        ignored.
    pal : palette name, list, dict, or None, optional
        palette to pass to sns.FacetGrid for specifying the colors to
        use. if None and hue=="Stimulus type", we use the defaults given
        by sfp.plotting.stimulus_type_palette.
    {col, row}_order : list or None, optional
        the order for the columns and rows. If None, we use the default
    {y, x}lim : tuples or None, optional
        if not None, the limits for the y- and x-axes for all subplots.
    ci : int, optional
        the size of the confidence intervals to plot. see the docstring
        of plot_func for more details
    n_boot : int, optional
        the number of bootstraps to use for creating the confidence
        intervals. see the docstring of plot_func for more details
    col_wrap : int or None, optional
        'wrap' the column variable at this width, so that the column
        facets span multiple rows. will throw an exception if col_wrap
        and row are both not None
    pre_boot_gb_func : callable or None, optional
        feature_df contains a lot of info, and you may want to collapse
        over some of those dimensions. In order to make sure those
        dimensions are collapsed over appropriately, this function can
        perform an (optional) groupby before creating the FacetGrid. If
        this is not None, we will create the plot with
        feature_df.groupby(pre_boot_gb_cols).apply(pre_boot_gb_func).reset_index(). The
        intended use case is for, e.g., averaging over all retinotopic
        angles.
    pre_boot_gb_cols : list, optional
        The columns to use for the optional groupby. See above for more
        details

    Returns
    -------
    g : sns.FacetGrid
        The FacetGrid containing the plot

    """
    if pal is None and hue == 'Stimulus type':
        pal = stimulus_type_palette(feature_df.reference_frame.unique())
    if col_order is None and col == 'Stimulus type':
        col_order = stimulus_type_order(feature_df.reference_frame.unique())
    if row_order is None and row == 'Stimulus type':
        row_order = stimulus_type_order(feature_df.reference_frame.unique())
    if pre_boot_gb_func is not None:
        feature_df = feature_df.groupby(pre_boot_gb_cols).apply(pre_boot_gb_func).reset_index()
    g = sns.FacetGrid(feature_df, hue=hue, col=col, row=row, height=height, aspect=aspect,
                      palette=pal, xlim=xlim, ylim=ylim, col_wrap=col_wrap, col_order=col_order,
                      row_order=row_order)
    g.map(plot_func, x, y, ci=ci, n_boot=n_boot)
    g.add_legend()
    for ax in g.axes.flatten():
        ax.axhline(color='gray', linestyle='--')
        ax.axvline(color='gray', linestyle='--')
        if yticks is not None:
            ax.set_yticks(yticks)
        if xticks is not None:
            ax.set_xticks(xticks)
    if title is not None:
        g.fig.suptitle(title)
        g.fig.subplots_adjust(top=top)
    return g


def feature_df_polar_plot(feature_df, hue="Stimulus type", col='Preferred period (dpc)', row=None,
                          plot_func=sns.lineplot, theta='Retinotopic angle (rad)',
                          r='Eccentricity (deg)', r_ticks=None, theta_ticks=None, height=4,
                          aspect=1, title='Preferred period contours', top=.76, pal=None,
                          col_order=None, row_order=None, title_position=[.5, 1.15], ylabelpad=30,
                          legend_position=None, ylim=None, xlim=None, ci=68, n_boot=10000,
                          col_wrap=None, pre_boot_gb_func=None,
                          pre_boot_gb_cols=['indicator', 'reference_frame', 'Stimulus type',
                                            'Eccentricity (deg)']):
    """Create polar plot from feature_df

    This function takes the feature_df created by
    sfp.analyze_model.create_feature_df and makes summary plots. The
    default should do more or less what you want it to, but there's a
    lot of customizability.

    Note that this makes a polar plot (it plots r as a function of
    theta), and so the intended use is for the preferred period and max
    amplitude contours feature_df. For the preferred period, use
    feature_df_plot

    The majority of the arguments are passed right to sns.FacetGrid

    Parameters
    ----------
    feature_df : pd.DataFrame
        The feature dataframe, containing the preferred period as a
        function of eccentricity, at multiple stimulus orientation and
        retinotopic angles
    hue : str, optional
        a column in feature_df, which feature to use for the hue of plot
    col : str, optional
        a column in feature_df, which feature to facet on the columns
    row : str, optional
        a column in feature_df, which feature to facet on the rows
    plot_func : callable, optional
        The plot function to map on the FacetGrid. First two args should
        be x and y, should accept ci and n_boot kwargs (many seaborn
        plotting functions would work for this)
    theta : str, optional
        a column in feature_df, which feature to plot as polar angle
    r : str, optional
        a column in feature_df, which feature to plot as distance from
        the origin
    {r, theta}ticks : list, optional
        list of floats, which r- and theta-ticks to include on the plot
    height : float, optional
        The height of each individual subplot
    aspect : float, optional
        The aspect ratio of each individual subplot
    title : str or None, optional
        The super-title of the plot. If None, we don't add a
        super-title, and we will not adjust the subplot spacing
    top : float, optional
        The amount to adjust the subplot spacing at the top so the title
        is above the subplots (with a call to
        g.fig.subplots_adjust(top=top)). If title is None, this is
        ignored.
    pal : palette name, list, dict, or None, optional
        palette to pass to sns.FacetGrid for specifying the colors to
        use. if None and hue=="Stimulus type", we use the defaults given
        by sfp.plotting.stimulus_type_palette.
    {col, row}_order : list or None, optional
        the order for the columns and rows. If None, we use the default
    title_position : 2-tuple, optional
        The position (in x, y) of each subplots' title (not the
        super-title)
    ylabelpad : int
        number of pixels to "pad" the y-label by, so that it doesn't
        overlap with the polar plot
    legend_position : 2-tuple or None, optional
        if not None, the x, y position of the legend. if None, use
        default position
    {y, x}lim : tuples or None, optional
        if not None, the limits for the y- and x-axes for all subplots.
    ci : int, optional
        the size of the confidence intervals to plot. see the docstring
        of plot_func for more details
    n_boot : int, optional
        the number of bootstraps to use for creating the confidence
        intervals. see the docstring of plot_func for more details
    col_wrap : int or None, optional
        'wrap' the column variable at this width, so that the column
        facets span multiple rows. will throw an exception if col_wrap
        and row are both not None
    pre_boot_gb_func : callable or None, optional
        feature_df contains a lot of info, and you may want to collapse
        over some of those dimensions. In order to make sure those
        dimensions are collapsed over appropriately, this function can
        perform an (optional) groupby before creating the FacetGrid. If
        this is not None, we will create the plot with
        feature_df.groupby(pre_boot_gb_cols).apply(pre_boot_gb_func).reset_index(). The
        intended use case is for, e.g., averaging over all retinotopic
        angles.
    pre_boot_gb_cols : list, optional
        The columns to use for the optional groupby. See above for more
        details

    Returns
    -------
    g : sns.FacetGrid
        The FacetGrid containing the plot

    """
    if pal is None and hue == 'Stimulus type':
        pal = stimulus_type_palette(feature_df.reference_frame.unique())
    if col_order is None and col == 'Stimulus type':
        col_order = stimulus_type_order(feature_df.reference_frame.unique())
    if row_order is None and row == 'Stimulus type':
        row_order = stimulus_type_order(feature_df.reference_frame.unique())
    if pre_boot_gb_func is not None:
        feature_df = feature_df.groupby(pre_boot_gb_cols).apply(pre_boot_gb_func).reset_index()
    g = sns.FacetGrid(feature_df, col=col, hue=hue, row=row, subplot_kws={'projection': 'polar'},
                      despine=False, height=height, aspect=aspect, palette=pal, xlim=xlim,
                      ylim=ylim, col_wrap=col_wrap, col_order=col_order, row_order=row_order)
    g.map(plot_func, theta, r, ci=ci, n_boot=n_boot)
    for i, ax in enumerate(g.axes.flatten()):
        ax.title.set_position(title_position)
        if i == 0:
            ax.yaxis.labelpad = ylabelpad
        if r_ticks is not None:
            ax.set_yticks(r_ticks)
        if theta_ticks is not None:
            ax.set_xticks(theta_ticks)
    if legend_position is not None:
        g.add_legend(bbox_to_anchor=legend_position)
    else:
        g.add_legend()
    if title is not None:
        g.fig.suptitle(title)
        g.fig.subplots_adjust(top=top)
    return g
