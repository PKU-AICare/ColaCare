from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils

from models.utils import get_last_visit


class Sparsemax(nn.Module):
    """Sparsemax function."""

    def __init__(self, dim=None):
        super(Sparsemax, self).__init__()

        self.dim = -1 if dim is None else dim

    def forward(self, input):
        original_size = input.size()
        input = input.view(-1, input.size(self.dim))

        dim = 1
        number_of_logits = input.size(dim)

        input = input - torch.max(input, dim=dim, keepdim=True)[0].expand_as(input)

        zs = torch.sort(input=input, dim=dim, descending=True)[0]
        range = torch.arange(
            start=1, end=number_of_logits + 1, dtype=torch.float32
        ).view(1, -1)
        range = range.expand_as(zs)

        bound = 1 + range * zs
        cumulative_sum_zs = torch.cumsum(zs, dim)
        is_gt = torch.gt(bound, cumulative_sum_zs).type(input.type())
        k = torch.max(is_gt * range, dim, keepdim=True)[0]

        zs_sparse = is_gt * zs
        taus = (torch.sum(zs_sparse, dim, keepdim=True) - 1) / k
        taus = taus.expand_as(input)

        self.output = torch.max(torch.zeros_like(input), input - taus)

        output = self.output.view(original_size)

        return output

    def backward(self, grad_output):
        dim = 1

        nonzeros = torch.ne(self.output, 0)
        sum = torch.sum(grad_output * nonzeros, dim=dim) / torch.sum(nonzeros, dim=dim)
        self.grad_input = nonzeros * (grad_output - sum.expand_as(grad_output))

        return self.grad_input


class CausalConv1d(torch.nn.Conv1d):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        dilation=1,
        groups=1,
        bias=True,
    ):
        self.__padding = (kernel_size - 1) * dilation

        super(CausalConv1d, self).__init__(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=self.__padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )

    def forward(self, input):
        result = super(CausalConv1d, self).forward(input)
        if self.__padding != 0:
            return result[:, :, : -self.__padding]
        return result


class Recalibration(nn.Module):
    def __init__(
        self, channel, reduction=9, use_h=True, use_c=True, activation="sigmoid"
    ):
        super(Recalibration, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.use_h = use_h
        self.use_c = use_c
        scale_dim = 0
        self.activation = activation

        self.nn_c = nn.Linear(channel, channel // reduction)
        scale_dim += channel // reduction

        self.nn_rescale = nn.Linear(scale_dim, channel)
        self.sparsemax = Sparsemax(dim=1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        b, c, t = x.size()

        y_origin = x.permute(0, 2, 1).reshape(b * t, c).contiguous()
        se_c = self.nn_c(y_origin)
        se_c = torch.relu(se_c)
        y = se_c

        y = self.nn_rescale(y).view(b, t, c).permute(0, 2, 1).contiguous()
        if self.activation == "sigmoid":
            y = torch.sigmoid(y)
        elif self.activation == "sparsemax":
            y = self.sparsemax(y)
        else:
            y = self.softmax(y)
        return x * y.expand_as(x), y.permute(0, 2, 1)


class AdaCareLayer(nn.Module):
    """AdaCare layer.

    Paper: Liantao Ma et al. Adacare: Explainable clinical health status representation learning
    via scale-adaptive feature extraction and recalibration. AAAI 2020.

    This layer is used in the AdaCare model. But it can also be used as a
    standalone layer.

    Args:
        input_dim: the input feature size.
        hidden_dim: the hidden dimension of the GRU layer. Default is 128.
        kernel_size: the kernel size of the causal convolution layer. Default is 2.
        kernel_num: the kernel number of the causal convolution layer. Default is 64.
        r_v: the number of the reduction rate for the original feature calibration. Default is 4.
        r_c: the number of the reduction rate for the convolutional feature recalibration. Default is 4.
        activation: the activation function for the recalibration layer (sigmoid, sparsemax, softmax). Default is "sigmoid".
        dropout: dropout rate. Default is 0.5.

    Examples:
        >>> from pyhealth.models import AdaCareLayer
        >>> input = torch.randn(3, 128, 64)  # [batch size, sequence len, feature_size]
        >>> layer = AdaCareLayer(64)
        >>> c, _, inputatt, convatt = layer(input)
        >>> c.shape
        torch.Size([3, 64])
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        kernel_size: int = 2,
        kernel_num: int = 64,
        r_v: int = 4,
        r_c: int = 4,
        activation: str = "sigmoid",
        rnn_type: str = "gru",
        dropout: float = 0.5,
    ):
        super(AdaCareLayer, self).__init__()

        if activation not in ["sigmoid", "softmax", "sparsemax"]:
            raise ValueError(
                "Only sigmoid, softmax and sparsemax are supported for activation."
            )
        if rnn_type not in ["gru", "lstm"]:
            raise ValueError("Only gru and lstm are supported for rnn_type.")

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.kernel_num = kernel_num
        self.r_v = r_v
        self.r_c = r_c
        self.dropout = dropout

        self.nn_conv1 = CausalConv1d(input_dim, kernel_num, kernel_size, 1, 1)
        self.nn_conv3 = CausalConv1d(input_dim, kernel_num, kernel_size, 1, 3)
        self.nn_conv5 = CausalConv1d(input_dim, kernel_num, kernel_size, 1, 5)
        torch.nn.init.xavier_uniform_(self.nn_conv1.weight)
        torch.nn.init.xavier_uniform_(self.nn_conv3.weight)
        torch.nn.init.xavier_uniform_(self.nn_conv5.weight)

        self.nn_convse = Recalibration(
            3 * kernel_num, r_c, use_h=False, use_c=True, activation="sigmoid"
        )
        self.nn_inputse = Recalibration(
            input_dim, r_v, use_h=False, use_c=True, activation=activation
        )

        if rnn_type == "gru":
            self.rnn = nn.GRU(input_dim + 3 * kernel_num, hidden_dim)
        else:
            self.rnn = nn.LSTM(input_dim + 3 * kernel_num, hidden_dim)
        # self.nn_output = nn.Linear(hidden_dim, output_dim)
        self.nn_dropout = nn.Dropout(dropout)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(
        self,
        x: torch.tensor,
        mask: Optional[torch.tensor] = None,
    ) -> Tuple[torch.tensor, torch.tensor, torch.tensor, torch.tensor]:
        """Forward propagation.

        Args:
            x: a tensor of shape [batch size, sequence len, input_dim].
            mask: an optional tensor of shape [batch size, sequence len], where
                1 indicates valid and 0 indicates invalid.

        Returns:
            last_output: a tensor of shape [batch size, input_dim] representing the
                patient embedding.
            output: a tensor of shape [batch size, sequence_len, input_dim] representing the patient embedding at each time step.
            inputatt: a tensor of shape [batch size, sequence_len, input_dim] representing the feature importance of the input.
            convatt: a tensor of shape [batch size, sequence_len, 3 * kernel_num] representing the feature importance of the convolutional features.
        """
        # batch_size = x.size(0)
        # time_step = x.size(1)
        # feature_dim = x.size(2)

        conv_input = x.permute(0, 2, 1)
        conv_res1 = self.nn_conv1(conv_input)
        conv_res3 = self.nn_conv3(conv_input)
        conv_res5 = self.nn_conv5(conv_input)

        conv_res = torch.cat((conv_res1, conv_res3, conv_res5), dim=1)
        conv_res = self.relu(conv_res)

        convse_res, convatt = self.nn_convse(conv_res)
        inputse_res, inputatt = self.nn_inputse(x.permute(0, 2, 1))
        concat_input = torch.cat((convse_res, inputse_res), dim=1).permute(0, 2, 1)
        output, _ = self.rnn(concat_input)
        last_output = get_last_visit(output, mask)
        if self.dropout > 0.0:
            last_output = self.nn_dropout(last_output)
        return last_output, output, inputatt, convatt

class AdaCare(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        kernel_size: int = 2,
        kernel_num: int = 64,
        r_v: int = 4,
        r_c: int = 4,
        activation: str = "sigmoid",
        rnn_type: str = "gru",
        dropout: float = 0.5,
        **kwargs
    ):
        super(AdaCare, self).__init__()

        if activation not in ["sigmoid", "softmax", "sparsemax"]:
            raise ValueError(
                "Only sigmoid, softmax and sparsemax are supported for activation."
            )
        if rnn_type not in ["gru", "lstm"]:
            raise ValueError("Only gru and lstm are supported for rnn_type.")

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.kernel_num = kernel_num
        self.r_v = r_v
        self.r_c = r_c
        self.dropout = dropout
        self.adacare_layer = AdaCareLayer(input_dim, hidden_dim, kernel_size, kernel_num, r_v, r_c, activation, rnn_type, dropout)
    def forward(
        self,
        x: torch.tensor,
        mask: Optional[torch.tensor] = None,
    ):
        # batch_size, time_steps, _ = x.size()
        # out = torch.zeros((batch_size, time_steps, self.hidden_dim))
        # for cur_time in range(time_steps-1, time_steps):
        #     cur_x = x[:, :cur_time+1, :]
        #     cur_mask = mask[:, :cur_time+1]
        #     cur_out, _, _, _ = self.adacare_layer(cur_x, cur_mask)
        #     out[:, cur_time, :] = cur_out
        out, _, feature_attn, _ = self.adacare_layer(x, mask)
        return out, feature_attn