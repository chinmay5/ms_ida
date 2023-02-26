import math
import torch

from torch_geometric.utils import softmax
from torch import Tensor
from torch_geometric.nn import TransformerConv
from torch_geometric.typing import OptTensor
import torch.nn.functional as F
import torch.nn as nn


class TransformerConvNoEdge(TransformerConv):
    def __init__(
            self,
            in_channels,
            out_channels,
            heads=1,
            concat=True,
            beta=False,
            dropout=0.,
            edge_dim=None,
            bias=True,
            root_weight=True,
            **kwargs,
    ):
        super(TransformerConvNoEdge, self).__init__(in_channels=in_channels, out_channels=out_channels, heads=heads,
                                                    concat=concat, beta=beta, dropout=dropout, edge_dim=edge_dim,
                                                    bias=bias, root_weight=root_weight, **kwargs)

    def message(self, query_i: Tensor, key_j: Tensor, value_j: Tensor,
                edge_attr: OptTensor, index: Tensor, ptr: OptTensor,
                size_i):

        if self.lin_edge is not None:
            assert edge_attr is not None
            edge_attr = self.lin_edge(edge_attr).view(-1, self.heads,
                                                      self.out_channels)
            key_j = key_j + edge_attr

        alpha = (query_i * key_j).sum(dim=-1) / math.sqrt(self.out_channels)
        alpha = softmax(alpha, index, ptr, size_i)
        self._alpha = alpha
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        out = value_j
        # We just ignore the edge_attr completely.
        # if edge_attr is not None:
        #     out = out + edge_attr

        out = out * alpha.view(-1, self.heads, 1)
        return out

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, heads={self.heads})')


class STEFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return (input >= 0.5).float()

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


class STESigmoidThresholding(nn.Module):
    def __init__(self):
        super(STESigmoidThresholding, self).__init__()

    def forward(self, x):
        x = STEFunction.apply(x)
        return x


