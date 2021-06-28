from math import sqrt

import torch
from torch.nn import functional as F


def strotss_loss(content_vec, style_vec, output_vec, style_yuv_vec, output_yuv_vec, alpha):
    assert 1 == len(content_vec) == len(style_vec) == len(output_vec) == len(style_yuv_vec) == len(output_yuv_vec)

    content_vec = content_vec.squeeze(0)
    style_vec = style_vec.squeeze(0)
    output_vec = output_vec.squeeze(0)
    style_yuv_vec = style_yuv_vec.squeeze(0)
    output_yuv_vec = output_yuv_vec.squeeze(0)
    ###################################################################################

    style_cost_matrix = _cosine_cost_matrix(output_vec, style_vec)
    palette_cost_matrix = (_euclidian_cost_matrix(output_yuv_vec, style_yuv_vec) +
                           _cosine_cost_matrix(output_yuv_vec, style_yuv_vec))

    loss_style = _remd(style_cost_matrix)
    loss_moment = _moment_loss(output_vec, style_vec)
    loss_content = _content_loss(output_vec, content_vec)
    loss_palette = _remd(palette_cost_matrix)

    loss = alpha * loss_content + loss_moment + loss_style + (loss_palette / alpha)
    return loss / (2. + alpha + (1.0 / alpha))


def _remd(cost_matrix):
    lhs, _ = cost_matrix.min(0)
    rhs, _ = cost_matrix.min(1)
    return torch.max(lhs.mean(), rhs.mean())


def _content_loss(output_vec, content_vec):
    cost_matrix_output = _cosine_cost_matrix(output_vec, output_vec)
    cost_matrix_content = _cosine_cost_matrix(content_vec, content_vec)

    return F.l1_loss(cost_matrix_output, cost_matrix_content)


def _moment_loss(output_vec, style_vec):
    n_output = len(output_vec)
    n_style = len(style_vec)

    # Row vector to column vector
    output_vec = output_vec.T
    style_vec = style_vec.T

    # Mean
    output_mu = torch.mean(output_vec, dim=1, keepdim=True)
    style_mu = torch.mean(style_vec, dim=1, keepdim=True)

    # Covariance
    output_vec = output_vec - output_mu
    style_vec = style_vec - style_mu
    output_cov = (output_vec @ output_vec.T) / (n_output - 1)
    style_cov = (style_vec @ style_vec.T) / (n_style - 1)

    return F.l1_loss(output_mu, style_mu) + F.l1_loss(output_cov, style_cov)


def _cosine_cost_matrix(x, y):
    x = F.normalize(x, dim=1)
    y = F.normalize(y, dim=1)
    return 1. - x @ y.T


def _euclidian_cost_matrix(x, y):
    return torch.cdist(x, y) / sqrt(x.size(1))
