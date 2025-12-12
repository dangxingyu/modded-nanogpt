
import torch
from torch import Tensor

# def zeropower_via_newtonschulz5(G: Tensor) -> Tensor:
#     """
#     Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt to use a
#     quintic iteration whose coefficients are selected to maximize the slope at zero. For the purpose
#     of minimizing steps, it turns out to be empirically effective to keep increasing the slope at
#     zero even beyond the point where the iteration no longer converges all the way to one everywhere
#     on the interval. This iteration therefore does not produce UV^T but rather something like US'V^T
#     where S' is diagonal with S_{ii}' ∈ [1 - l, 1 + r], which turns out not to hurt model
#     performance at all relative to UV^T, where USV^T = G is the SVD.
#     """
#     assert G.ndim >= 2 # batched Muon implementation by @scottjmaddox, and put into practice in the record by @YouJiacheng
#     X = G.bfloat16()
#     if G.size(-2) > G.size(-1):
#         X = X.mT

#     # Ensure spectral norm is at most 1
#     X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
#     # Perform the NS iterations
#     for a, b, c in [
#         (4.0848, -6.8946, 2.9270),
#         (3.9505, -6.3029, 2.6377),
#         (3.7418, -5.5913, 2.3037),
#         (2.8769, -3.1427, 1.2046),
#         (2.8366, -3.0525, 1.2012),
#     ]:
#         A = X @ X.mT
#         B = b * A + c * A @ A # quintic computation strategy adapted from suggestion by @jxbz, @leloykun, and @YouJiacheng
#         X = a * X + B @ X

#     if G.size(-2) > G.size(-1):
#         X = X.mT
#     return X

# def update(acc_bf16_view_u16: Tensor, mantissa: Tensor, momentum_buffer: Tensor, grad: Tensor, momentum: Tensor, eff_lr: Tensor, eff_weight_decay: Tensor):
#     assert acc_bf16_view_u16.dtype == mantissa.dtype == torch.uint16
#     grad = grad.float()
#     momentum_buffer.copy_(momentum * momentum_buffer + (1 - momentum) * grad)
#     v = zeropower_via_newtonschulz5(momentum * momentum_buffer + (1 - momentum) * grad)
#     acc_m_u32 = (acc_bf16_view_u16.to(torch.uint32) << 16) | mantissa.to(torch.uint32)
#     # For batched parameters (e.g., qkvo_w with shape [4, hdim, dim]), compute norm over last 2 dims
#     norm_dims = (-2, -1) if grad.ndim >= 2 else None
#     fro_norm = acc_m_u32.view(torch.float32).norm(p='fro', dim=norm_dims, keepdim=True)
#     v_norm = v.norm(p='fro', dim=norm_dims, keepdim=True)
#     acc_m_u32.view(torch.float32).add_(other=v, alpha=-eff_lr * fro_norm / v_norm)
#     fro_norm_new = acc_m_u32.view(torch.float32).norm(p='fro', dim=norm_dims, keepdim=True)
#     acc_m_u32.view(torch.float32).mul_(fro_norm / fro_norm_new)

#     acc_bf16_view_u16.copy_((acc_m_u32 >> 16).to(torch.uint16))
#     mantissa.copy_(acc_m_u32.to(torch.uint16))



@torch.compile
def run(a, v):
    fro_a = a.norm(p='fro', dim=(-2, -1), keepdim=True)
    fro_v = v.norm(p='fro', dim=(-2, -1), keepdim=True)

    a_2 = a.clone()
    v_2 = v.clone()

    a.add_(other=v, alpha= -fro_a / fro_v)

    for i in range(a_2.shape[0]):
        a_2[i].add_(other=v_2[i], alpha=-fro_a[i] / fro_v[i])

    return a, a_2


a = torch.randn(2, 3, 5)
v = torch.randn(2, 3, 5)
a_2, a_2_2 = run(a, v)
print(a)
print(a_2)
