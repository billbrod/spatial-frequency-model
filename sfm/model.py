#!/usr/bin/python
"""2d tuning model
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import warnings
import itertools
import re
from scipy import stats


def _cast_as_tensor(x):
    if type(x) == pd.Series:
        x = x.values
    # needs to be float32 to work with the Hessian calculations
    return torch.tensor(x, dtype=torch.float32)


def _cast_as_param(x, requires_grad=True):
    return torch.nn.Parameter(_cast_as_tensor(x), requires_grad=requires_grad)


def _cast_args_as_tensors(args, on_cuda=False):
    return_args = []
    for v in args:
        if not torch.is_tensor(v):
            v = _cast_as_tensor(v)
        if on_cuda:
            v = v.cuda()
        return_args.append(v)
    return return_args


def _check_and_reshape_tensors(x, y):
    if (x.ndimension() == 1 and y.ndimension() == 1) and (x.shape != y.shape):
        x = x.repeat(len(y), 1)
        y = y.repeat(x.shape[1], 1).transpose(0, 1)
    return x, y


def _check_log_gaussian_params(param_vals, train_params, orientation_type, eccentricity_type,
                               vary_amplitude):
        if orientation_type in ['relative', 'iso']:
            for param, angle in itertools.product(['mode', 'amplitude'],
                                                  ['cardinals', 'obliques']):
                if param_vals['abs_%s_%s' % (param, angle)] != 0:
                    warnings.warn("When orientation_type is %s, all absolute variables must"
                                  " be 0, correcting this..." % orientation_type)
                    param_vals['abs_%s_%s' % (param, angle)] = 0
                train_params['abs_%s_%s' % (param, angle)] = False
        if orientation_type in ['absolute', 'iso']:
            for param, angle in itertools.product(['mode', 'amplitude'],
                                                  ['cardinals', 'obliques']):
                if param_vals['rel_%s_%s' % (param, angle)] != 0:
                    warnings.warn("When orientation_type is %s, all relative variables must"
                                  " be 0, correcting this..." % orientation_type)
                    param_vals['rel_%s_%s' % (param, angle)] = 0
                train_params['rel_%s_%s' % (param, angle)] = False
        if orientation_type not in ['relative', 'absolute', 'iso', 'full']:
            raise Exception("Don't know how to handle orientation_type %s!" % orientation_type)
        if not vary_amplitude:
            for ori, angle in itertools.product(['abs', 'rel'], ['cardinals', 'obliques']):
                if param_vals['%s_amplitude_%s' % (ori, angle)] != 0:
                    warnings.warn("When vary_amplitude is False, all amplitude variables must"
                                  " be 0, correcting this...")
                    param_vals['%s_amplitude_%s' % (ori, angle)] = 0
                train_params['%s_amplitude_%s' % (ori, angle)] = False
        if eccentricity_type == 'scaling':
            if param_vals['sf_ecc_intercept'] != 0:
                warnings.warn("When eccentricity_type is scaling, sf_ecc_intercept must be 0! "
                              "correcting...")
                param_vals['sf_ecc_intercept'] = 0
            train_params['sf_ecc_intercept'] = False
        elif eccentricity_type == 'constant':
            if param_vals['sf_ecc_slope'] != 0:
                warnings.warn("When eccentricity_type is constant, sf_ecc_slope must be 0! "
                              "correcting...")
                param_vals['sf_ecc_slope'] = 0
            train_params['sf_ecc_slope'] = False
        elif eccentricity_type != 'full':
            raise Exception("Don't know how to handle eccentricity_type %s!" % eccentricity_type)
        return param_vals, train_params


class LogGaussianDonut(torch.nn.Module):
    """simple LogGaussianDonut in pytorch

    orientation_type, eccentricity_type, vary_amplitude: together specify what
    kind of model to train

    orientation_type: {iso, absolute, relative, full}.
    - iso: model is isotropic, predictions identical for all orientations.
    - absolute: model can fit differences in absolute orientation, that is, in Cartesian
      coordinates, such that sf_angle=0 correponds to "to the right"
    - relative: model can fit differences in relative orientation, that is, in retinal polar
      coordinates, such that sf_angle=0 corresponds to "away from the fovea"
    - full: model can fit differences in both absolute and relative orientations

    eccentricity_type: {scaling, constant, full}.
    - scaling: model's relationship between preferred period and eccentricity is exactly scaling,
      that is, the preferred period is equal to the eccentricity.
    - constant: model's relationship between preferred period and eccentricity is exactly constant,
      that is, it does not change with eccentricity but is flat.
    - full: model discovers the relationship between eccentricity and preferred period, though it
      is constrained to be linear (i.e., model solves for a and b in $period = a * eccentricity +
      b$)

    vary_amplitude: boolean. whether to allow the model to fit the parameters that control
    amplitude as a function of orientation (whether this depends on absolute orientation, relative
    orientation, or both depends on the value of `orientation_type`)

    all other parameters are initial values. whether they will be fit or not (i.e., whether they
    have `requires_grad=True`) depends on the values of `orientation_type`, `eccentricity_type` and
    `vary_amplitude`

    when you call this model, sf_angle should be the (absolute) orientation of the grating, so that
    sf_angle=0 corresponds to "to the right". That is, regardless of whether the model considers
    the absolute orientation, relative orientation, neither or both to be important, you always
    call it with the absolute orientation.

    """
    def __init__(self, orientation_type='iso', eccentricity_type='full', vary_amplitude=True,
                 sigma=.4, sf_ecc_slope=1, sf_ecc_intercept=0, abs_mode_cardinals=0,
                 abs_mode_obliques=0, rel_mode_cardinals=0, rel_mode_obliques=0,
                 abs_amplitude_cardinals=0, abs_amplitude_obliques=0, rel_amplitude_cardinals=0,
                 rel_amplitude_obliques=0):
        super().__init__()
        train_kwargs = {}
        kwargs = {}
        for ori, param, angle in itertools.product(['abs', 'rel'], ['mode', 'amplitude'],
                                                   ['cardinals', 'obliques']):
            train_kwargs['%s_%s_%s' % (ori, param, angle)] = True
            kwargs['%s_%s_%s' % (ori, param, angle)] = eval('%s_%s_%s' % (ori, param, angle))
        for var in ['slope', 'intercept']:
            train_kwargs['sf_ecc_%s' % var] = True
            kwargs['sf_ecc_%s' % var] = eval("sf_ecc_%s" % var)
        kwargs, train_kwargs = _check_log_gaussian_params(kwargs, train_kwargs, orientation_type,
                                                          eccentricity_type, vary_amplitude)

        self.orientation_type = orientation_type
        amp_vary_label = {False: 'constant', True: 'vary'}[vary_amplitude]
        self.eccentricity_type = eccentricity_type
        self.vary_amplitude = vary_amplitude
        self.model_type = '%s_donut_%s_amps-%s' % (eccentricity_type, orientation_type,
                                                   amp_vary_label)
        self.sigma = _cast_as_param(sigma)

        self.abs_amplitude_cardinals = _cast_as_param(kwargs['abs_amplitude_cardinals'],
                                                      train_kwargs['abs_amplitude_cardinals'])
        self.abs_amplitude_obliques = _cast_as_param(kwargs['abs_amplitude_obliques'],
                                                     train_kwargs['abs_amplitude_obliques'])
        self.rel_amplitude_cardinals = _cast_as_param(kwargs['rel_amplitude_cardinals'],
                                                      train_kwargs['rel_amplitude_cardinals'])
        self.rel_amplitude_obliques = _cast_as_param(kwargs['rel_amplitude_obliques'],
                                                     train_kwargs['rel_amplitude_obliques'])
        self.abs_mode_cardinals = _cast_as_param(kwargs['abs_mode_cardinals'],
                                                 train_kwargs['abs_mode_cardinals'])
        self.abs_mode_obliques = _cast_as_param(kwargs['abs_mode_obliques'],
                                                train_kwargs['abs_mode_obliques'])
        self.rel_mode_cardinals = _cast_as_param(kwargs['rel_mode_cardinals'],
                                                 train_kwargs['rel_mode_cardinals'])
        self.rel_mode_obliques = _cast_as_param(kwargs['rel_mode_obliques'],
                                                train_kwargs['rel_mode_obliques'])
        self.sf_ecc_slope = _cast_as_param(kwargs['sf_ecc_slope'],
                                           train_kwargs['sf_ecc_slope'])
        self.sf_ecc_intercept = _cast_as_param(kwargs['sf_ecc_intercept'],
                                               train_kwargs['sf_ecc_intercept'])

    @classmethod
    def init_from_df(cls, df):
        """initialize from the dataframe we make summarizing the models

        the df must only contain a single model (that is, it should only have 11 rows, one for each
        parameter value, and a unique value for the column fit_model_type)

        """
        fit_model_type = df.fit_model_type.unique()
        if len(fit_model_type) > 1 or len(df) != 11:
            raise Exception("df must contain exactly one model!")
        params = {}
        for i, row in df.iterrows():
            params[row.model_parameter] = row.fit_value
        parse_string = r'([a-z]+)_donut_([a-z]+)_amps-([a-z]+)'
        eccentricity_type, orientation_type, amp_vary_label = re.findall(parse_string,
                                                                         fit_model_type[0])[0]
        return cls(orientation_type, eccentricity_type,
                   {'vary': True, 'constant': False}[amp_vary_label], **params)

    def __str__(self):
        # so we can see the parameters
        return ("{0}(sigma: {1:.03f}, sf_ecc_slope: {2:.03f}, sf_ecc_intercept: {3:.03f}, "
                "abs_amplitude_cardinals: {4:.03f}, abs_amplitude_obliques: {5:.03f}, "
                "abs_mode_cardinals: {6:.03f}, abs_mode_obliques: {7:.03f}, "
                "rel_amplitude_cardinals: {8:.03f}, rel_amplitude_obliques: {9:.03f}, "
                "rel_mode_cardinals: {10:.03f}, rel_mode_obliques: {11:.03f})").format(
                    type(self).__name__, self.sigma, self.sf_ecc_slope, self.sf_ecc_intercept,
                    self.abs_amplitude_cardinals, self.abs_amplitude_obliques,
                    self.abs_mode_cardinals, self.abs_mode_obliques, self.rel_amplitude_cardinals,
                    self.rel_amplitude_obliques, self.rel_mode_cardinals, self.rel_mode_obliques)

    def __repr__(self):
        return self.__str__()
    
    def prepare_image_computable(self, energy, filters, stim_radius_degree=12):
        """prepare for the image computable version of the model

        Parameters
        ----------
        energy : np.ndarray
            energy has shape (num_classes, n_scales, n_orientations, *img_size) and 
            contains the energy (square and absolute value the complex valued output of 
            SteerablePyramidFreq; equivalently, square and sum the output of the quadrature pair of 
            filters that make up the pyramid) for each image, at each scale and orientation. the energy
            has all been upsampled to the size of the initial image.
        filters : np.ndarray
            filters has shape (max_ht, n_orientations, *img_size) and is the fourier transform of the 
            filters at each scale and orientation, zero-padded so they all have the same size. we only 
            have one set of filters (instead of one per stimulus class) because the same pyramid was 
            used for each of them; we ensure this by getting the filters for each stimulus class and 
            checking that they're individually equal to the average across classes.
        stim_radius_degree : int
            the radius of the stimuli (in degrees), necessary for converting between pixels and 
            degrees.

        """
        self.stim_radius_degree = stim_radius_degree
        if energy.shape[-2] != energy.shape[-1]:
            raise Exception("For now, this only works on square input images!")
        self.image_size = energy.shape[-1]
        filters, energy = _cast_args_as_tensors([filters, energy], self.sigma.is_cuda)
        self.energy = energy.unsqueeze(0)
        # this is the l1 norm
        norm_weights = filters.abs().sum((2,3), keepdim=True)
        norm_weights = norm_weights[0] / norm_weights
        self.filters = filters * norm_weights
        x = np.linspace(-self.stim_radius_degree, self.stim_radius_degree, self.image_size)
        x, y = np.meshgrid(x, x)
        # we want to try and delete things to save memory
        del norm_weights, energy, filters
        self.visual_space = np.dstack((x, y))

    def _create_mag_angle(self, extent=(-10, 10), n_samps=1001):
        x = torch.linspace(extent[0], extent[1], n_samps)
        x, y = torch.meshgrid(x, x)
        r = torch.sqrt(torch.pow(x, 2) + torch.pow(y, 2))
        th = torch.atan2(y, x)
        return r, th

    def create_image(self, vox_ecc, vox_angle, extent=None, n_samps=None):
        vox_ecc, vox_angle = _cast_args_as_tensors([vox_ecc, vox_angle], self.sigma.is_cuda)
        if vox_ecc.ndimension() == 0:
            vox_ecc = vox_ecc.unsqueeze(-1)
            vox_angle = vox_angle.unsqueeze(-1)
        if extent is None:
            extent = (-self.stim_radius_degree, self.stim_radius_degree)
        if n_samps is None:
            n_samps = self.image_size
        r, th = self._create_mag_angle(extent, n_samps)
        return self.evaluate(r.repeat(len(vox_ecc), 1, 1), th.repeat(len(vox_ecc), 1, 1),
                             vox_ecc, vox_angle)

    def preferred_period_contour(self, preferred_period, vox_angle, sf_angle=None,
                                 rel_sf_angle=None):
        """return eccentricity that has specified preferred_period for given sf_angle, vox_angle

        either sf_angle or rel_sf_angle can be set
        """
        if ((sf_angle is None and rel_sf_angle is None) or
            (sf_angle is not None and rel_sf_angle is not None)):
            raise Exception("Either sf_angle or rel_sf_angle must be set!")
        if sf_angle is None:
            sf_angle = rel_sf_angle
            rel_flag = True
        else:
            rel_flag = False
        preferred_period, sf_angle, vox_angle = _cast_args_as_tensors(
            [preferred_period, sf_angle, vox_angle], self.sigma.is_cuda)
        # we can allow up to two of these variables to be non-singletons.
        if sf_angle.ndimension() == 1 and preferred_period.ndimension() == 1 and vox_angle.ndimension() == 1:
            # if this is False, then all of them are the same shape and we have no issues
            if sf_angle.shape != preferred_period.shape != vox_angle.shape:
                raise Exception("Only two of these variables can be non-singletons!")
        else:
            sf_angle, preferred_period = _check_and_reshape_tensors(sf_angle, preferred_period)
            preferred_period, vox_angle = _check_and_reshape_tensors(preferred_period, vox_angle)
            sf_angle, vox_angle = _check_and_reshape_tensors(sf_angle, vox_angle)
        if not rel_flag:
            rel_sf_angle = sf_angle - vox_angle
        else:
            rel_sf_angle = sf_angle
            sf_angle = rel_sf_angle + vox_angle
        orientation_effect = (1 + self.abs_mode_cardinals * torch.cos(2 * sf_angle) +
                              self.abs_mode_obliques * torch.cos(4 * sf_angle) +
                              self.rel_mode_cardinals * torch.cos(2 * rel_sf_angle) +
                              self.rel_mode_obliques * torch.cos(4 * rel_sf_angle))
        return (preferred_period / orientation_effect - self.sf_ecc_intercept) / self.sf_ecc_slope

    def preferred_period(self, vox_ecc, vox_angle, sf_angle=None, rel_sf_angle=None):
        """return preferred period for specified voxel at given orientation
        """
        if ((sf_angle is None and rel_sf_angle is None) or
            (sf_angle is not None and rel_sf_angle is not None)):
            raise Exception("Either sf_angle or rel_sf_angle must be set!")
        if sf_angle is None:
            sf_angle = rel_sf_angle
            rel_flag = True
        else:
            rel_flag = False
        sf_angle, vox_ecc, vox_angle = _cast_args_as_tensors([sf_angle, vox_ecc, vox_angle],
                                                             self.sigma.is_cuda)
        # we can allow up to two of these variables to be non-singletons.
        if sf_angle.ndimension() == 1 and vox_ecc.ndimension() == 1 and vox_angle.ndimension() == 1:
            # if this is False, then all of them are the same shape and we have no issues
            if sf_angle.shape != vox_ecc.shape != vox_angle.shape:
                raise Exception("Only two of these variables can be non-singletons!")
        else:
            sf_angle, vox_ecc = _check_and_reshape_tensors(sf_angle, vox_ecc)
            vox_ecc, vox_angle = _check_and_reshape_tensors(vox_ecc, vox_angle)
            sf_angle, vox_angle = _check_and_reshape_tensors(sf_angle, vox_angle)
        if not rel_flag:
            rel_sf_angle = sf_angle - vox_angle
        else:
            rel_sf_angle = sf_angle
            sf_angle = rel_sf_angle + vox_angle
        eccentricity_effect = self.sf_ecc_slope * vox_ecc + self.sf_ecc_intercept
        orientation_effect = (1 + self.abs_mode_cardinals * torch.cos(2 * sf_angle) +
                              self.abs_mode_obliques * torch.cos(4 * sf_angle) +
                              self.rel_mode_cardinals * torch.cos(2 * rel_sf_angle) +
                              self.rel_mode_obliques * torch.cos(4 * rel_sf_angle))
        return torch.clamp(eccentricity_effect * orientation_effect, min=1e-6)

    def preferred_sf(self, sf_angle, vox_ecc, vox_angle):
        return 1. / self.preferred_period(vox_ecc, vox_angle, sf_angle)

    def max_amplitude(self, vox_angle, sf_angle=None, rel_sf_angle=None):
        if ((sf_angle is None and rel_sf_angle is None) or
            (sf_angle is not None and rel_sf_angle is not None)):
            raise Exception("Either sf_angle or rel_sf_angle must be set!")
        if sf_angle is None:
            sf_angle = rel_sf_angle
            rel_flag = True
        else:
            rel_flag = False
        sf_angle, vox_angle = _cast_args_as_tensors([sf_angle, vox_angle], self.sigma.is_cuda)
        sf_angle, vox_angle = _check_and_reshape_tensors(sf_angle, vox_angle)
        if not rel_flag:
            rel_sf_angle = sf_angle - vox_angle
        else:
            rel_sf_angle = sf_angle
            sf_angle = rel_sf_angle + vox_angle
        amplitude = (1 + self.abs_amplitude_cardinals * torch.cos(2*sf_angle) +
                     self.abs_amplitude_obliques * torch.cos(4*sf_angle) +
                     self.rel_amplitude_cardinals * torch.cos(2*rel_sf_angle) +
                     self.rel_amplitude_obliques * torch.cos(4*rel_sf_angle))
        return torch.clamp(amplitude, min=1e-6)

    def evaluate(self, sf_mag, sf_angle, vox_ecc, vox_angle):
        sf_mag, = _cast_args_as_tensors([sf_mag], self.sigma.is_cuda)
        # if ecc_effect is 0 or below, then log2(ecc_effect) is infinity or undefined
        # (respectively). to avoid that, we clamp it 1e-6. in practice, if a voxel ends up here
        # that means the model predicts 0 response for it.
        preferred_period = self.preferred_period(vox_ecc, vox_angle, sf_angle)
        pdf = torch.exp(-((torch.log2(sf_mag) + torch.log2(preferred_period))**2) /
                        (2*self.sigma**2))
        amplitude = self.max_amplitude(vox_angle, sf_angle)
        return amplitude * pdf

    def image_computable_weights(self, vox_ecc, vox_angle):
        vox_ecc, vox_angle = _cast_args_as_tensors([vox_ecc, vox_angle], self.sigma.is_cuda)
        vox_tuning = self.create_image(vox_ecc.unsqueeze(-1).unsqueeze(-1),
                                       vox_angle.unsqueeze(-1).unsqueeze(-1),
                                       (-self.stim_radius_degree, self.stim_radius_degree),
                                       self.image_size)
        vox_tuning = vox_tuning.unsqueeze(1).unsqueeze(1)
        return torch.sum(vox_tuning * self.filters, (-1, -2), keepdim=True).unsqueeze(1)

    def create_prfs(self, vox_ecc, vox_angle, vox_sigma):
        vox_ecc, vox_angle, vox_sigma = _cast_args_as_tensors([vox_ecc, vox_angle, vox_sigma],
                                                              self.sigma.is_cuda)
        if vox_ecc.ndimension() == 0:
            vox_ecc = vox_ecc.unsqueeze(-1)
        if vox_angle.ndimension() == 0:
            vox_angle = vox_angle.unsqueeze(-1)
        if vox_sigma.ndimension() == 0:
            vox_sigma = vox_sigma.unsqueeze(-1)
        vox_x = vox_ecc * np.cos(vox_angle)
        vox_y = vox_ecc * np.sin(vox_angle)
        prfs = []
        for x, y, s in zip(vox_x, vox_y, vox_sigma):
            prf = stats.multivariate_normal((x, y), s)
            prfs.append(prf.pdf(self.visual_space))
        return _cast_args_as_tensors([prfs], self.sigma.is_cuda)[0].unsqueeze(1)
    
    def image_computable(self, inputs):
        # the different features will always be indexed along the last axis (we don't know whether
        # this is 2d (stimulus_class, features) or 3d (voxels, stimulus_class, features))
        # to be used as index, must be long type
        stim_class = inputs.select(-1, 0)
        if stim_class.ndimension() == 2:
            stim_class = stim_class.mean(0)
        stim_class = torch.as_tensor(stim_class, dtype=torch.long)
        vox_ecc = inputs.select(-1, 1).mean(-1)
        if vox_ecc.ndimension() == 0:
            vox_ecc = vox_ecc.unsqueeze(-1)
        vox_angle = inputs.select(-1, 2).mean(-1)
        if vox_angle.ndimension() == 0:
            vox_angle = vox_angle.unsqueeze(-1)
        vox_sigma = inputs.select(-1, 3).mean(-1)
        if vox_sigma.ndimension() == 0:
            vox_sigma = vox_sigma.unsqueeze(-1)
        weights = self.image_computable_weights(vox_ecc, vox_angle)
        reweighted_energy = torch.sum(weights * self.energy[:, stim_class], (2, 3))

        prfs = self.create_prfs(vox_ecc, vox_angle, vox_sigma)
        predictions = torch.sum(prfs * reweighted_energy, (-1, -2))
        # reweighted_energy is big, so we want to delete it to try and save memory
        del reweighted_energy
        return predictions

    def forward(self, inputs):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        # the different features will always be indexed along the last axis (we don't know whether
        # this is 2d (stimulus_class, features) or 3d (voxels, stimulus_class, features))
        sf_mag = inputs.select(-1, 0)
        sf_angle = inputs.select(-1, 1)
        vox_ecc = inputs.select(-1, 2)
        vox_angle = inputs.select(-1, 3)
        return self.evaluate(sf_mag, sf_angle, vox_ecc, vox_angle)


def show_image(donut, voxel_eccentricity=1, voxel_angle=0, extent=(-5, 5), n_samps=1001,
               cmap="Reds", show_colorbar=True, ax=None, **kwargs):
    """wrapper function to plot the image from a given donut

    This shows the spatial frequency selectivity implied by the donut at a given voxel eccentricity
    and angle, if appropriate (eccentricity and angle ignored if donut is ConstantLogGuassianDonut)

    donut: a LogGaussianDonut

    extent: 2-tuple of floats. the range of spatial frequencies to visualize `(min, max)`. this
    will be the same for x and y
    """
    if ax is None:
        plt.imshow(
            donut.create_image(voxel_eccentricity, voxel_angle, extent, n_samps=n_samps).detach()[0],
            extent=(extent[0], extent[1], extent[0], extent[1]), cmap=cmap,
            origin='lower', **kwargs)
        ax = plt.gca()
    else:
        ax.imshow(
            donut.create_image(voxel_eccentricity, voxel_angle, extent, n_samps=n_samps).detach()[0],
            extent=(extent[0], extent[1], extent[0], extent[1]), cmap=cmap,
            origin='lower', **kwargs)
    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)
    ax.set_frame_on(False)
    if show_colorbar:
        plt.colorbar()
    return ax


def weighted_normed_loss(predictions, target):
    """takes in the predictions and target, returns weighted norm loss

    note all of these must be tensors, not numpy arrays

    target must contain both the targets and the precision (along the last axis)

    if we weren't multiplying by the precision, this would be equivalent to cosine distance (times
    a constant: num_classes / 2)

    """
    precision = target.select(-1, 1)
    target = target.select(-1, 0)
    # we occasionally have an issue where the predictions are really small (like 1e-200), which
    # gives us a norm of 0 and thus a normed_predictions of infinity, and thus an infinite loss.
    # the point of renorming is that multiplying by a scale factor won't change our loss, so we do
    # that here to avoid this issue
    if 0 in predictions.norm(2, -1, True):
        warnings.warn("Predictions too small to normalize correctly, multiplying it be 1e100")
        predictions = predictions * 1e100
    # we norm / average along the last dimension, since that means we do it across all stimulus
    # classes for a given voxel. we don't know whether these tensors will be 1d (single voxel, as
    # returned by our FirstLevelDataset) or 2d (multiple voxels, as returned by the DataLoader)
    normed_predictions = predictions / predictions.norm(2, -1, True)
    normed_target = target / target.norm(2, -1, True)
    # this isn't really necessary (all the values along that dimension should be identical, based
    # on how we calculated it), but just in case. and this gets it in the right shape
    precision = precision.mean(-1, True)
    squared_error = precision * (normed_predictions - normed_target)**2
    return squared_error.mean()
