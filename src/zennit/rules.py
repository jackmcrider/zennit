# This file is part of Zennit
# Copyright (C) 2019-2021 Christopher J. Anders
#
# zennit/rules.py
#
# Zennit is free software: you can redistribute it and/or modify it under
# the terms of the GNU Lesser General Public License as published by the Free
# Software Foundation; either version 3 of the License, or (at your option) any
# later version.
#
# Zennit is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for
# more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this library. If not, see <https://www.gnu.org/licenses/>.
'''Rules based on Hooks'''
import torch

from .core import Hook, BasicHook, Stabilizer, expand, ParamMod


def zero_bias(zero_params=None):
    '''Add `'bias'` to `zero_params`, where `zero_params` is a string or a list of strings.

    Parameters
    ----------
    zero_params: str or list of str, optional
        Name or names, to which ``'bias'`` should be added.

    Returns
    -------
    list of str
        Supplied ``zero_params``, with the string ``'bias'`` appended.
    '''
    if zero_params is None:
        return ['bias']
    if isinstance(zero_params, str):
        zero_params = [zero_params]
    if 'bias' in zero_params:
        return zero_params
    return list(zero_params) + ['bias']


class ClampMod(ParamMod):
    '''ParamMod to clamp module parameters.

    Parameters
    ----------
    min: float or None, optional
        Minimum float value for which the parameters should be clamped, or None if no clamping should be done.
    max: float or None, optional
        Maximum float value for which the parameters should be clamped, or None if no clamping should be done.
    kwargs: dict[str, object]
        Additional keyword arguments used for :py:class:`ParamMod`.
    '''
    def __init__(self, min=None, max=None, **kwargs):
        def modifier(param, name):
            return param.clamp(min=min, max=max)
        super().__init__(modifier, **kwargs)


class GammaMod(ParamMod):
    '''ParamMod to modify module parameters as in the Gamma rule. Adds the scaled, clamped parameters to the parameter
    itself.

    Parameters
    ----------
    gamma: float, optional
        Gamma scaling parameter, by which the clamped parameters are multiplied.
    min: float or None, optional
        Minimum float value for which the parameters should be clamped, or None if no clamping should be done.
    max: float or None, optional
        Maximum float value for which the parameters should be clamped, or None if no clamping should be done.
    kwargs: dict[str, object]
        Additional keyword arguments used for :py:class:`ParamMod`.
    '''
    def __init__(self, gamma=0.25, min=None, max=None, **kwargs):
        def modifier(param, name):
            return param + gamma * param.clamp(min=min, max=max)
        super().__init__(modifier, **kwargs)


class NoMod(ParamMod):
    '''ParamMod that does not modify the parameters. Allows other modification flags.

    Parameters
    ----------
    kwargs: dict[str, object]
        Additional keyword arguments used for :py:class:`ParamMod`.
    '''
    def __init__(self, **kwargs):
        super().__init__((lambda param, _: param), **kwargs)


class Epsilon(BasicHook):
    '''LRP Epsilon rule :cite:p:`bach2015pixel`.
    Setting ``(epsilon=0)`` produces the LRP-0 rule :cite:p:`bach2015pixel`.
    LRP Epsilon is most commonly used in middle layers, LRP-0 is most commonly used in upper layers
    :cite:p:`montavon2019layer`.
    Sometimes higher values of ``epsilon`` are used, therefore it is not always only a stabilizer value.

    Parameters
    ----------
    epsilon: callable or float, optional
        Stabilization parameter. If ``epsilon`` is a float, it will be added to the denominator with the same sign as
        each respective entry. If it is callable, a function ``(input: torch.Tensor) -> torch.Tensor`` is expected, of
        which the output corresponds to the stabilized denominator. Note that this is called ``stabilizer`` for all
        other rules.
    zero_params: list[str], optional
        A list of parameter names that shall set to zero. If `None` (default), no parameters are set to zero.
    '''
    def __init__(self, epsilon=1e-6, zero_params=None):
        stabilizer_fn = Stabilizer.ensure(epsilon)
        super().__init__(
            input_modifiers=[lambda input: input],
            param_modifiers=[NoMod(zero_params=zero_params)],
            output_modifiers=[lambda output: output],
            gradient_mapper=(lambda out_grad, outputs: out_grad / stabilizer_fn(outputs[0])),
            reducer=(lambda inputs, gradients: inputs[0] * gradients[0]),
        )


class Gamma(BasicHook):
    '''Generalized LRP Gamma rule :cite:p:`montavon2019layer,andeol2021learning`.
    The gamma parameter scales the added positive/negative parts of the weights.
    The original Gamma rule :cite:p:`montavon2019layer` may only be used with positive inputs.
    The generalized version is equivalent to the original Gamma when there are only positive inputs, but may also be
    used for negative inputs.

    Parameters
    ----------
    gamma: float, optional
        Multiplier for added positive weights.
    stabilizer: callable or float, optional
        Stabilization parameter. If ``stabilizer`` is a float, it will be added to the denominator with the same sign
        as each respective entry. If it is callable, a function ``(input: torch.Tensor) -> torch.Tensor`` is expected,
        of which the output corresponds to the stabilized denominator.
    zero_params: list[str], optional
        A list of parameter names that shall set to zero. If `None` (default), no parameters are set to zero.
    '''
    def __init__(self, gamma=0.25, stabilizer=1e-6, zero_params=None):
        mod_kwargs = {'zero_params': zero_params}
        mod_kwargs_nobias = {'zero_params': zero_bias(zero_params)}
        stabilizer_fn = Stabilizer.ensure(stabilizer)
        super().__init__(
            input_modifiers=[
                lambda input: input.clamp(min=0),
                lambda input: input.clamp(max=0),
                lambda input: input.clamp(min=0),
                lambda input: input.clamp(max=0),
                lambda input: input,
            ],
            param_modifiers=[
                GammaMod(gamma, min=0., **mod_kwargs),
                GammaMod(gamma, max=0., **mod_kwargs_nobias),
                GammaMod(gamma, max=0., **mod_kwargs),
                GammaMod(gamma, min=0., **mod_kwargs_nobias),
                NoMod(),
            ],
            output_modifiers=[lambda output: output] * 5,
            gradient_mapper=(
                lambda out_grad, outputs: [
                    output * out_grad / stabilizer_fn(denom)
                    for output, denom in (
                        [(outputs[4] > 0., sum(outputs[:2]))] * 2
                        + [(outputs[4] < 0., sum(outputs[2:4]))] * 2
                    )
                ] + [torch.zeros_like(out_grad)]
            ),
            reducer=(
                lambda inputs, gradients: sum(input * gradient for input, gradient in zip(inputs[:4], gradients[:4]))
            ),
        )


class ZPlus(BasicHook):
    '''LRP ZPlus rule :cite:p:`bach2015pixel,montavon2017explaining`.
    It is the same as using :py:class:`~zennit.rules.AlphaBeta` with ``(alpha=1, beta=0)``

    Parameters
    ----------
    stabilizer: callable or float, optional
        Stabilization parameter. If ``stabilizer`` is a float, it will be added to the denominator with the same sign
        as each respective entry. If it is callable, a function ``(input: torch.Tensor) -> torch.Tensor`` is expected,
        of which the output corresponds to the stabilized denominator.
    zero_params: list[str], optional
        A list of parameter names that shall set to zero. If `None` (default), no parameters are set to zero.

    Notes
    -----
    Note that the original deep Taylor Decomposition (DTD) specification of the ZPlus Rule
    :cite:p:`montavon2017explaining` only considers positive inputs, as they are used in ReLU Networks.
    This implementation is effectively alpha=1, beta=0, where negative inputs are allowed.
    '''
    def __init__(self, stabilizer=1e-6, zero_params=None):
        stabilizer_fn = Stabilizer.ensure(stabilizer)
        super().__init__(
            input_modifiers=[
                lambda input: input.clamp(min=0),
                lambda input: input.clamp(max=0),
            ],
            param_modifiers=[
                ClampMod(min=0., zero_params=zero_params),
                ClampMod(max=0., zero_params=zero_bias(zero_params)),
            ],
            output_modifiers=[lambda output: output] * 2,
            gradient_mapper=(lambda out_grad, outputs: [out_grad / stabilizer_fn(sum(outputs))] * 2),
            reducer=(lambda inputs, gradients: inputs[0] * gradients[0] + inputs[1] * gradients[1]),
        )


class AlphaBeta(BasicHook):
    '''LRP AlphaBeta rule :cite:p:`bach2015pixel`.
    The AlphaBeta rule weights positive (alpha) and negative (beta) contributions.
    Most common parameters are ``(alpha=1, beta=0)`` and ``(alpha=2, beta=1)``.
    It is most commonly used for lower layers :cite:p:`montavon2019layer`.

    Parameters
    ----------
    alpha: float, optional
        Multiplier for the positive output term.
    beta: float, optional
        Multiplier for the negative output term.
    stabilizer: callable or float, optional
        Stabilization parameter. If ``stabilizer`` is a float, it will be added to the denominator with the same sign
        as each respective entry. If it is callable, a function ``(input: torch.Tensor) -> torch.Tensor`` is expected,
        of which the output corresponds to the stabilized denominator.
    zero_params: list[str], optional
        A list of parameter names that shall set to zero. If `None` (default), no parameters are set to zero.

    '''
    def __init__(self, alpha=2., beta=1., stabilizer=1e-6, zero_params=None):
        if alpha < 0 or beta < 0:
            raise ValueError("Both alpha and beta parameters must be positive!")
        if (alpha - beta) != 1.:
            raise ValueError("The difference of parameters alpha - beta must equal 1!")
        mod_kwargs = {'zero_params': zero_params}
        mod_kwargs_nobias = {'zero_params': zero_bias(zero_params)}
        stabilizer_fn = Stabilizer.ensure(stabilizer)

        super().__init__(
            input_modifiers=[
                lambda input: input.clamp(min=0),
                lambda input: input.clamp(max=0),
                lambda input: input.clamp(min=0),
                lambda input: input.clamp(max=0),
            ],
            param_modifiers=[
                ClampMod(min=0., **mod_kwargs),
                ClampMod(max=0., **mod_kwargs_nobias),
                ClampMod(max=0., **mod_kwargs),
                ClampMod(min=0., **mod_kwargs_nobias),
            ],
            output_modifiers=[lambda output: output] * 4,
            gradient_mapper=(
                lambda out_grad, outputs: [
                    out_grad / stabilizer_fn(denom)
                    for denom in ([sum(outputs[:2])] * 2 + [sum(outputs[2:])] * 2)
                ]
            ),
            reducer=(
                lambda inputs, gradients: (
                    alpha * (inputs[0] * gradients[0] + inputs[1] * gradients[1])
                    - beta * (inputs[2] * gradients[2] + inputs[3] * gradients[3])
                )
            ),
        )


class ZBox(BasicHook):
    '''LRP ZBox rule :cite:p:`montavon2017explaining`.
    The ZBox rule is intended for "boxed" input pixel space.
    Generally, the lowest and highest *possible* values are used, i.e. ``(low=0., high=1.)`` for raw image data in
    the float data type.
    Neural network inputs are often normalized to match an isotropic gaussian distribution with mean 0 and variance 1,
    which means that the lowest and highest values also need to be adapted.
    For image data, this generally happens per channel, for which case ``low`` and ``high`` can be passed as tensors
    with shape ``(1, 3, 1, 1)``, which will be broadcasted as expected.

    Parameters
    ----------
    low: :py:class:`torch.Tensor` or float
        Lowest pixel values of input. Subject to broadcasting.
    high: :py:class:`torch.Tensor` or float
        Highest pixel values of input. Subject to broadcasting.
    stabilizer: callable or float, optional
        Stabilization parameter. If ``stabilizer`` is a float, it will be added to the denominator with the same sign
        as each respective entry. If it is callable, a function ``(input: torch.Tensor) -> torch.Tensor`` is expected,
        of which the output corresponds to the stabilized denominator.
    zero_params: list[str], optional
        A list of parameter names that shall set to zero. If `None` (default), no parameters are set to zero.

    '''
    def __init__(self, low, high, stabilizer=1e-6, zero_params=None):
        def sub(positive, *negatives):
            return positive - sum(negatives)

        mod_kwargs = {'zero_params': zero_params}
        stabilizer_fn = Stabilizer.ensure(stabilizer)

        super().__init__(
            input_modifiers=[
                lambda input: input,
                lambda input: expand(low, input.shape, cut_batch_dim=True).to(input),
                lambda input: expand(high, input.shape, cut_batch_dim=True).to(input),
            ],
            param_modifiers=[
                NoMod(**mod_kwargs),
                ClampMod(min=0., **mod_kwargs),
                ClampMod(max=0., **mod_kwargs),
            ],
            output_modifiers=[lambda output: output] * 3,
            gradient_mapper=(lambda out_grad, outputs: (out_grad / stabilizer_fn(sub(*outputs)),) * 3),
            reducer=(lambda inputs, gradients: sub(*(input * gradient for input, gradient in zip(inputs, gradients)))),
        )


class Pass(Hook):
    '''Unmodified pass-through rule.
    If the rule of a layer shall not be any other, is elementwise and shall not be the gradient, the `Pass` rule simply
    passes upper layer relevance through to the lower layer.
    '''
    def backward(self, module, grad_input, grad_output):
        '''Pass through the upper gradient, skipping the one for this layer.'''
        return grad_output


class Norm(BasicHook):
    '''Normalize and weight by input contribution.
    This is essentially the same as the LRP :py:class:`~zennit.rules.Epsilon` rule :cite:p:`bach2015pixel` with a fixed
    epsilon only used as a stabilizer, and without the need of the attached layer to have parameters ``weight`` and
    ``bias``.
    '''
    def __init__(self, stabilizer=1e-6):
        stabilizer_fn = Stabilizer.ensure(stabilizer)
        super().__init__(
            input_modifiers=[lambda input: input],
            param_modifiers=[NoMod(param_keys=[])],
            output_modifiers=[lambda output: output],
            gradient_mapper=(lambda out_grad, outputs: out_grad / stabilizer_fn(outputs[0])),
            reducer=(lambda inputs, gradients: inputs[0] * gradients[0]),
        )


class WSquare(BasicHook):
    '''LRP WSquare rule :cite:p:`montavon2017explaining`.
    It is most commonly used in the first layer when the values are not bounded :cite:p:`montavon2019layer`.

    Parameters
    ----------
    stabilizer: callable or float, optional
        Stabilization parameter. If ``stabilizer`` is a float, it will be added to the denominator with the same sign
        as each respective entry. If it is callable, a function ``(input: torch.Tensor) -> torch.Tensor`` is expected,
        of which the output corresponds to the stabilized denominator.
    zero_params: list[str], optional
        A list of parameter names that shall set to zero. If `None` (default), no parameters are set to zero.
    '''
    def __init__(self, stabilizer=1e-6, zero_params=None):
        stabilizer_fn = Stabilizer.ensure(stabilizer)
        super().__init__(
            input_modifiers=[torch.ones_like],
            param_modifiers=[
                ParamMod((lambda param, _: param ** 2), zero_params=zero_params),
            ],
            output_modifiers=[lambda output: output],
            gradient_mapper=(lambda out_grad, outputs: out_grad / stabilizer_fn(outputs[0])),
            reducer=(lambda inputs, gradients: gradients[0]),
        )


class Flat(BasicHook):
    '''LRP Flat rule :cite:p:`lapuschkin2019unmasking`.
    It is essentially the same as the LRP :py:class:`~zennit.rules.WSquare` rule, but with all parameters set to ones.

    Parameters
    ----------
    stabilizer: callable or float, optional
        Stabilization parameter. If ``stabilizer`` is a float, it will be added to the denominator with the same sign
        as each respective entry. If it is callable, a function ``(input: torch.Tensor) -> torch.Tensor`` is expected,
        of which the output corresponds to the stabilized denominator.
    zero_params: list[str], optional
        A list of parameter names that shall set to zero. If `None` (default), no parameters are set to zero.
    '''
    def __init__(self, stabilizer=1e-6, zero_params=None):
        mod_kwargs = {'zero_params': zero_bias(zero_params), 'require_params': False}
        stabilizer_fn = Stabilizer.ensure(stabilizer)
        super().__init__(
            input_modifiers=[torch.ones_like],
            param_modifiers=[
                ParamMod((lambda param, name: torch.ones_like(param)), **mod_kwargs),
            ],
            output_modifiers=[lambda output: output],
            gradient_mapper=(lambda out_grad, outputs: out_grad / stabilizer_fn(outputs[0])),
            reducer=(lambda inputs, gradients: gradients[0]),
        )


class ReLUDeconvNet(Hook):
    '''DeconvNet ReLU rule :cite:p:`zeiler2014visualizing`.'''
    def backward(self, module, grad_input, grad_output):
        '''Modify ReLU gradient according to DeconvNet :cite:p:`zeiler2014visualizing`.'''
        return (grad_output[0].clamp(min=0),)


class ReLUGuidedBackprop(Hook):
    '''GuidedBackprop ReLU rule :cite:p:`springenberg2015striving`.'''
    def backward(self, module, grad_input, grad_output):
        '''Modify ReLU gradient according to GuidedBackprop :cite:p:`springenberg2015striving`.'''
        return (grad_input[0] * (grad_output[0] > 0.),)


class ReLUBetaSmooth(Hook):
    '''Modify ReLU gradient to smooth softplus gradient :cite:p:`dombrowski2019explanations`. Used to obtain meaningful
    surrogate gradients to compute higher order gradients with ReLUs. Equivalent to changing the gradient to be the
    (scaled) logistic function (sigmoid).

    Parameters
    ----------
    beta_smooth: float, optional
        The beta parameter for the softplus gradient (i.e. ``sigmoid(beta * input)``). Defaults to ``10``.
    '''
    def __init__(self, beta_smooth=10.):
        super().__init__()
        self.beta_smooth = beta_smooth

    def copy(self):
        '''Return a copy of this hook with the same beta parameter.'''
        return self.__class__(beta_smooth=self.beta_smooth)

    def forward(self, module, input, output):
        '''Remember the input for the backward pass.'''
        self.stored_tensors['input'] = input

    def backward(self, module, grad_input, grad_output):
        '''Modify ReLU gradient to the smooth softplus gradient :cite:p:`dombrowski2019explanations`.'''
        return (torch.sigmoid(self.beta_smooth * self.stored_tensors['input'][0]) * grad_output[0],)


class TakesMostBase(Hook):
    '''Base class for TakesMost rules.
    This class provides a common interface for rule variants that utilize a softmax-like weighting of the input
    contributions based on their magnitude.

    Parameters
    ----------
    beta: float
        Beta parameter for controlling the sensitivity of the softmax weighting.

    Methods
    -------
    max_fn(input, kernel_size, stride, padding, dilation):
        Computes the maximum value in a local window for each entry in the input tensor.
    sum_fn(input, kernel, stride, padding, dilation):
        Computes the sum of elements in a local window for each entry in the input tensor.
    forward(module, input, output):
        Stores the input for later use in the backward pass.
    backward(module, grad_input, grad_output):
        Modifies the gradient based on the softmax weighting of the input contributions.
    '''
    def __init__(self, beta=1.0):
        super().__init__()
        self.beta = beta
        self.stored_tensors = {}

    def copy(self):
        '''Return a copy of this hook with the same beta parameter.'''
        return self.__class__(beta=self.beta)

    def max_fn(self, input, kernel_size, stride, padding, dilation):
        '''Computes the maximum value in a local window for each entry in the input tensor.'''
        raise NotImplementedError("Implement in subclass")

    def sum_fn(self, input, kernel_size, stride, padding, dilation):
        '''Computes the sum of elements in a local window for each entry in the input tensor.'''
        raise NotImplementedError("Implement in subclass")

    def forward(self, module, input, output):
        '''Stores the input for later use in the backward pass.'''
        self.stored_tensors['input'] = input

    def backward(self, module, grad_input, grad_output):
        '''Modifies the gradient based on the softmax-like weighting of input contributions.'''
        stored_input = self.stored_tensors['input'][0]

        kernel_size = module.kernel_size
        stride = module.stride
        padding = module.padding
        dilation = module.dilation

        # For numerical stability, we subtract the maximum value from the input
        max_val = self.max_fn(self.beta * stored_input, kernel_size, stride, padding, dilation)
        exp_input = torch.exp(self.beta * stored_input - max_val)
        summed_elements = self.sum_fn(exp_input, kernel_size, stride=stride, padding=padding, dilation=dilation)
        softmax_output = exp_input / summed_elements

        return (softmax_output * grad_output[0],)


class MinTakesMost1d(TakesMostBase):
    '''1D variant of TakesMost rule that weights the smallest contributions the most.
    This rule is a 1D variant of TakesMostBase, but weights the smallest input contributions the most.

    Methods
    -------
    __init__(beta=1.0):
        Initializes the MinTakesMost1d class with a negative beta value.
    '''
    def __init__(self, beta=1.0):
        super().__init__(-beta)

    def max_fn(self, input, kernel_size, stride, padding, dilation):
        '''Computes the maximum value in a local window for each entry in the input tensor.'''
        return torch.nn.functional.max_pool1d(input, kernel_size, stride=stride, padding=padding, dilation=dilation)

    def sum_fn(self, input, kernel_size, stride, padding, dilation):
        '''Computes the sum of elements in a local window for each entry in the input tensor.'''
        in_channels = input.shape[1]
        kernel = torch.ones((in_channels, 1, kernel_size), device=input.device)
        return torch.nn.functional.conv1d(input, weight=kernel, stride=stride, padding=padding, dilation=dilation,
                                          groups=in_channels)


class MaxTakesMost1d(TakesMostBase):
    '''1D variant of TakesMost rule that weights the largest contributions the most.
    This rule is a 1D variant of TakesMostBase, but weights the largest input contributions the most.

    Methods
    -------
    __init__(beta=1.0):
        Initializes the MaxTakesMost1d class.
    '''
    def max_fn(self, input, kernel_size, stride, padding, dilation):
        '''Computes the maximum value in a local window for each entry in the input tensor.'''
        return torch.nn.functional.max_pool1d(input, kernel_size, stride=stride, padding=padding, dilation=dilation)

    def sum_fn(self, input, kernel_size, stride, padding, dilation):
        '''Computes the sum of elements in a local window for each entry in the input tensor.'''
        in_channels = input.shape[1]
        kernel = torch.ones((in_channels, 1, kernel_size), device=input.device)
        return torch.nn.functional.conv1d(input, weight=kernel, stride=stride, padding=padding, dilation=dilation,
                                          groups=in_channels)


class MinTakesMost2d(TakesMostBase):
    '''2D variant of TakesMost rule that weights the smallest contributions the most.
    This rule is a 2D variant of TakesMostBase, but weights the smallest input contributions the most.

    Methods
    -------
    __init__(beta=1.0):
        Initializes the MinTakesMost2d class with a negative beta value.
    '''
    def __init__(self, beta=1.0):
        super().__init__(-beta)

    def max_fn(self, input, kernel_size, stride, padding, dilation):
        '''Computes the maximum value in a local window for each entry in the input tensor.'''
        return torch.nn.functional.max_pool2d(input, kernel_size, stride=stride, padding=padding, dilation=dilation)

    def sum_fn(self, input, kernel_size, stride, padding, dilation):
        '''Computes the sum of elements in a local window for each entry in the input tensor.'''
        in_channels = input.shape[1]
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        kernel = torch.ones((in_channels, 1, *kernel_size), device=input.device)
        return torch.nn.functional.conv2d(input, weight=kernel, stride=stride, padding=padding, dilation=dilation,
                                          groups=in_channels)


class MaxTakesMost2d(TakesMostBase):
    '''2D variant of TakesMost rule that weights the largest contributions the most.
    This rule is a 2D variant of TakesMostBase, but weights the largest input contributions the most.

    Methods
    -------
    __init__(beta=1.0):
        Initializes the MaxTakesMost2d class.
    '''
    def max_fn(self, input, kernel_size, stride, padding, dilation):
        '''Computes the maximum value in a local window for each entry in the input tensor.'''
        # return torch.nn.functional.max_pool2d(input, kernel_size, stride=stride, padding=padding, dilation=dilation)
        return input.max().view(1,1,1,1)

    def sum_fn(self, input, kernel_size, stride, padding, dilation):
        '''Computes the sum of elements in a local window for each entry in the input tensor.'''
        in_channels = input.shape[1]
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        if isinstance(stride, int):
            stride = (stride, stride)
        kernel = torch.ones((in_channels, 1, *kernel_size), device=input.device)
        summed_tensor = torch.nn.functional.conv2d(input, weight=kernel, stride=stride, padding=padding,
                                                   dilation=dilation, groups=in_channels)
        expanded_sum = torch.nn.functional.conv_transpose2d(summed_tensor, weight=kernel, stride=stride,
                                                            padding=padding, dilation=dilation, groups=in_channels)
        pad_height = input.shape[2] - expanded_sum.shape[2]
        pad_width = input.shape[3] - expanded_sum.shape[3]
        if pad_height > 0 or pad_width > 0:
            expanded_sum = torch.nn.functional.pad(expanded_sum, (0, pad_width, 0, pad_height))

        return expanded_sum

    def backward(self, module, grad_input, grad_output):
        '''Modifies the gradient based on the softmax-like weighting of input contributions.'''
        stored_input = self.stored_tensors['input'][0]
        if isinstance(module.stride, int):
            stride = (module.stride, module.stride)
        else:
            stride = module.stride

        kernel_size = module.kernel_size
        padding = module.padding
        dilation = module.dilation

        max_val = self.max_fn(self.beta * stored_input, kernel_size, stride, padding, dilation)
        exp_input = torch.exp(self.beta * stored_input - max_val)
        summed_elements = self.sum_fn(exp_input, kernel_size, stride=stride, padding=padding, dilation=dilation)
        softmax_output = exp_input / summed_elements
        softmax_output[summed_elements == 0] = 0

        in_channels = stored_input.shape[1]
        kernel = torch.ones((in_channels, 1, kernel_size, kernel_size), device=stored_input.device)
        expanded_grad_output = torch.nn.functional.conv_transpose2d(grad_output[0], weight=kernel, stride=stride,
                                                                    padding=padding, dilation=dilation,
                                                                    groups=in_channels)
        pad_height = stored_input.shape[2] - expanded_grad_output.shape[2]
        pad_width = stored_input.shape[3] - expanded_grad_output.shape[3]
        if pad_height > 0 or pad_width > 0:
            expanded_grad_output = torch.nn.functional.pad(expanded_grad_output, (0, pad_width, 0, pad_height))

        return (softmax_output * expanded_grad_output,)
