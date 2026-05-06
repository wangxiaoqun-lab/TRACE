# The script is from scGPT (https://github.com/bowang-lab/scGPT), copyright (c) 2022 suber

import torch
import torch.nn.functional as F


def masked_mse_loss(
    input: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """
    Compute the masked MSE loss between input and target.
    """
    mask = mask.float()
    loss = F.mse_loss(input * mask, target * mask, reduction="sum")
    return loss / mask.sum()


def criterion_neg_log_bernoulli(
    input: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """
    Compute the negative log-likelihood of Bernoulli distribution
    """
    mask = mask.float()
    bernoulli = torch.distributions.Bernoulli(probs=input)
    masked_log_probs = bernoulli.log_prob((target > 0).float()) * mask
    return -masked_log_probs.sum() / mask.sum()


def masked_relative_error(
    input: torch.Tensor, target: torch.Tensor, mask: torch.LongTensor
) -> torch.Tensor:
    """
    Compute the masked relative error between input and target.
    """
    assert mask.any()
    loss = torch.abs(input[mask] - target[mask]) / (target[mask] + 1e-6)
    return loss.mean()

from torch import Tensor, nn
from torch.distributions import NegativeBinomial
from torch.autograd import Function

# FROM SCVI
def nb(target: Tensor, mu: Tensor, theta: Tensor, eps=1e-8):
    """
    Computes the negative binomial (NB) loss.

    This function was adapted from scvi-tools.

    Args:
        target (Tensor): Ground truth data.
        mu (Tensor): Means of the negative binomial distribution (must have positive support).
        theta (Tensor): Inverse dispersion parameter (must have positive support).
        eps (float, optional): Numerical stability constant. Defaults to 1e-8.

    Returns:
        Tensor: NB loss value.
    """
    if theta.ndimension() == 1:
        theta = theta.view(1, theta.size(0))

    log_theta_mu_eps = torch.log(theta + mu + eps)
    res = (
        theta * (torch.log(theta + eps) - log_theta_mu_eps)
        + target * (torch.log(mu + eps) - log_theta_mu_eps)
        + torch.lgamma(target + theta)
        - torch.lgamma(theta)
        - torch.lgamma(target + 1)
    )

    return -res.mean()


def nb_dist(x: Tensor, mu: Tensor, theta: Tensor, eps=1e-8):
    """
    nb_dist Computes the negative binomial distribution.

    Args:
        x (Tensor): Torch Tensor of observed data.
        mu (Tensor): Torch Tensor of means of the negative binomial distribution (must have positive support).
        theta (Tensor): Torch Tensor of inverse dispersion parameter (must have positive support).
        eps (float, optional): Numerical stability constant. Defaults to 1e-8.

    Returns:
        Tensor: Negative binomial loss value.
    """
    loss = -NegativeBinomial(mu=mu, theta=theta).log_prob(x)
    return loss


def zinb(
    target: Tensor,
    mu: Tensor,
    theta: Tensor,
    pi: Tensor,
    eps=1e-8,
):
    """
    Computes zero-inflated negative binomial (ZINB) loss.

    This function was modified from scvi-tools.

    Args:
        target (Tensor): Torch Tensor of ground truth data.
        mu (Tensor): Torch Tensor of means of the negative binomial (must have positive support).
        theta (Tensor): Torch Tensor of inverse dispersion parameter (must have positive support).
        pi (Tensor): Torch Tensor of logits of the dropout parameter (real support).
        eps (float, optional): Numerical stability constant. Defaults to 1e-8.

    Returns:
        Tensor: ZINB loss value.
    """
    #  uses log(sigmoid(x)) = -softplus(-x)
    softplus_pi = F.softplus(-pi)
    # eps to make it positive support and taking the log
    log_theta_mu_eps = torch.log(theta + mu + eps)
    pi_theta_log = -pi + theta * (torch.log(theta + eps) - log_theta_mu_eps)

    case_zero = F.softplus(pi_theta_log) - softplus_pi
    mul_case_zero = torch.mul((target < eps).type(torch.float32), case_zero)

    case_non_zero = (
        -softplus_pi
        + pi_theta_log
        + target * (torch.log(mu + eps) - log_theta_mu_eps)
        + torch.lgamma(target + theta)
        - torch.lgamma(theta)
        - torch.lgamma(target + 1)
    )
    mul_case_non_zero = torch.mul((target > eps).type(torch.float32), case_non_zero)

    res = mul_case_zero + mul_case_non_zero
    # we want to minize the loss but maximize the log likelyhood
    return -res.mean()

class AdversarialDiscriminatorLoss(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_cls: int,
        nlayers: int = 3,
        activation: callable = nn.LeakyReLU,
        reverse_grad: bool = True,
    ):
        """
        Discriminator for the adversarial training for batch correction.

        Args:
            d_model (int): The size of the input tensor.
            n_cls (int): The number of classes.
            nlayers (int, optional): The number of layers in the discriminator. Defaults to 3.
            activation (callable, optional): The activation function. Defaults to nn.LeakyReLU.
            reverse_grad (bool, optional): Whether to reverse the gradient. Defaults
        """
        super().__init__()
        # module list
        self.decoder = nn.ModuleList()
        for _ in range(nlayers - 1):
            self.decoder.append(nn.Linear(d_model, d_model))
            self.decoder.append(nn.LayerNorm(d_model))
            self.decoder.append(activation())
        self.out_layer = nn.Linear(d_model, n_cls)
        self.reverse_grad = reverse_grad

    def forward(self, x: Tensor, batch_labels: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, embsize]
        """
        if self.reverse_grad:
            x = grad_reverse(x, lambd=1.0)
        for layer in self.decoder:
            x = layer(x)
        x = self.out_layer(x)
        return F.cross_entropy(x, batch_labels)


class GradReverse(Function):
    @staticmethod
    def forward(ctx, x: Tensor, lambd: float) -> Tensor:
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: Tensor) -> tuple[Tensor, None]:
        return grad_output.neg() * ctx.lambd, None


def grad_reverse(x: Tensor, lambd: float = 1.0) -> Tensor:
    """
    grad_reverse Reverses the gradient of the input tensor.

    Args:
        x (Tensor): The input tensor whose gradient is to be reversed.
        lambd (float, optional): The scaling factor for the reversed gradient. Defaults to 1.0.

    Returns:
        Tensor: The input tensor with its gradient reversed during the backward pass.
    """
    return GradReverse.apply(x, lambd)