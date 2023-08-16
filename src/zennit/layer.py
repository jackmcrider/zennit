# This file is part of Zennit
# Copyright (C) 2019-2021 Christopher J. Anders
#
# zennit/layer.py
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
'''Additional Utility Layers'''
import torch


class Sum(torch.nn.Module):
    '''Compute the sum along an axis.

    Parameters
    ----------
    dim : int
        Dimension over which to sum.
    '''
    def __init__(self, dim=-1):
        super().__init__()
        self.dim = dim

    def forward(self, input):
        '''Computes the sum along a dimension.'''
        return torch.sum(input, dim=self.dim)


class Distance(torch.nn.Module):
    '''Compute distance between inputs and centroids.'''
    def __init__(self, centroids, power=2):
        super().__init__()
        self.centroids = torch.nn.Parameter(centroids)
        self.power = power

    def forward(self, input):
        """Computes the nearest centroid for each input.

        :param input: Data points
        :returns: Index of nearest centroid

        """
        distance = torch.cdist(input, self.centroids)**self.power
        return distance


class NeuralizedKMeans(torch.nn.Module):
    '''Neuralized K-Means layer. Actually a tensor-matrix product.'''
    def __init__(self, W, b):
        super().__init__()
        self.W = torch.nn.Parameter(W)
        self.b = torch.nn.Parameter(b)

    def forward(self, x):
        x = torch.einsum('nd,kjd->nkj', x, self.W) + self.b
        return x


class LogMeanExpPool(torch.nn.Module):
    """Computes the log mean exp pooling.

    :param beta: scaling parameter
    :param dim: dimension over which to pool
    :returns: log mean exp pooled tensor

    """
    def __init__(self, beta=1., dim=-1):
        super().__init__()
        self.dim = dim
        self.beta = beta

    def forward(self, input):
        N = input.shape[self.dim]
        return (torch.logsumexp(self.beta * input, dim=self.dim) -
                torch.log(torch.tensor(N, dtype=input.dtype))) / self.beta
