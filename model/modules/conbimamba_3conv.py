'''
The following code contains components adapted from the "Mamba-in-Speech" open-source project
Original project repository: https://github.com/Tonyyouyou/Mamba-in-Speech
Original authors: Zhang, Xiangyu; Zhang, Qiquan; Liu, Hexin; Xiao, Tianyi; Qian, Xinyuan; 
                  Ahmed, Beena; Ambikairajah, Eliathamby; Li, Haizhou; Epps, Julien
Original work is associated with the paper: "Mamba in Speech: Towards an Alternative to Self-Attention"
(arXiv preprint arXiv:2405.12609, 2024)

Modifications made: [Briefly describe your modifications here if applicable]
- [Example: Adapted the Mamba architecture for specific speech recognition tasks]
- [Example: Optimized the training pipeline for our dataset]
'''

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch import Tensor
from typing import Tuple
from mamba_ssm.modules.mamba_simple import Mamba

class Swish(nn.Module):
    def __init__(self):
        super(Swish, self).__init__()
    
    def forward(self, inputs: Tensor) -> Tensor:
        return inputs * inputs.sigmoid()


class GLU(nn.Module):
    def __init__(self, dim: int) -> None:
        super(GLU, self).__init__()
        self.dim = dim

    def forward(self, inputs: Tensor) -> Tensor:
        outputs, gate = inputs.chunk(2, dim=self.dim)
        return outputs * gate.sigmoid()

    
class ResidualConnectionModule(nn.Module):
    def __init__(self, module: nn.Module, module_factor: float = 1.0, input_factor: float = 1.0):
        super(ResidualConnectionModule, self).__init__()
        self.module = module
        self.module_factor = module_factor
        self.input_factor = input_factor

    def forward(self, inputs: Tensor) -> Tensor:
        return (self.module(inputs) * self.module_factor) + (inputs * self.input_factor)


class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super(Linear, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        init.xavier_uniform_(self.linear.weight)
        if bias:
            init.zeros_(self.linear.bias)

    def forward(self, x: Tensor) -> Tensor:
        return self.linear(x)


class View(nn.Module):
    def __init__(self, shape: tuple, contiguous: bool = False):
        super(View, self).__init__()
        self.shape = shape
        self.contiguous = contiguous

    def forward(self, x: Tensor) -> Tensor:
        if self.contiguous:
            x = x.contiguous()

        return x.view(*self.shape)


class Transpose(nn.Module):
    def __init__(self, shape: Tuple[int, int]):
        super(Transpose, self).__init__()
        self.shape = shape

    def forward(self, x: Tensor) -> Tensor:
        return x.transpose(*self.shape)
    
class FeedForwardModule(nn.Module):
    def __init__(
            self,
            encoder_dim: int = 512,
            expansion_factor: int = 4,
            dropout_p: float = 0.1,
    ) -> None:
        super(FeedForwardModule, self).__init__()
        self.sequential = nn.Sequential(
            nn.LayerNorm(encoder_dim),
            Linear(encoder_dim, encoder_dim * expansion_factor, bias=True),
            Swish(),
            nn.Dropout(p=dropout_p),
            Linear(encoder_dim * expansion_factor, encoder_dim, bias=True),
            nn.Dropout(p=dropout_p),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return self.sequential(inputs)
    
class DepthwiseConv1d(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int,
            stride: int = 1,
            padding: int = 0,
            bias: bool = False,
    ) -> None:
        super(DepthwiseConv1d, self).__init__()
        assert out_channels % in_channels == 0, "out_channels should be constant multiple of in_channels"
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            groups=in_channels,
            stride=stride,
            padding=padding,
            bias=bias,
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return self.conv(inputs)


class PointwiseConv1d(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            stride: int = 1,
            padding: int = 0,
            bias: bool = True,
    ) -> None:
        super(PointwiseConv1d, self).__init__()
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=stride,
            padding=padding,
            bias=bias,
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return self.conv(inputs)

class MultiScaleConformerConvModule(nn.Module):
    def __init__(
        self,
        in_channels: int,
        kernel_sizes: list = [15, 31, 63],
        expansion_factor: int = 2,
        dropout_p: float = 0.1,
        merge_mode: str = 'gated',
    ):
        super().__init__()
        self.merge_mode = merge_mode
        if self.merge_mode not in ["avg", "concat", "gated"]:
            raise ValueError("merge_mode must be 'avg', 'concat', or 'gated'")

        self.norm = nn.LayerNorm(in_channels)
        self.transpose = Transpose(shape=(1, 2))  # B,T,C -> B,C,T

        self.pre_pointwise = PointwiseConv1d(in_channels, in_channels * expansion_factor)
        self.glu = GLU(dim=1)  

        self.branches = nn.ModuleList()
        if self.merge_mode == "gated":
            self.gating_units = nn.ModuleList()
            for kernel_size in kernel_sizes:
                assert (kernel_size - 1) % 2 == 0, "kernel_size should be odd for SAME padding"
                self.branches.append(
                    DepthwiseConv1d(
                        in_channels,
                        in_channels,
                        kernel_size=kernel_size,
                        padding=(kernel_size - 1) // 2,
                    )
                )
                self.gating_units.append(nn.Sigmoid())
        else:
            for kernel_size in kernel_sizes:
                assert (kernel_size - 1) % 2 == 0, "kernel_size should be odd for SAME padding"
                self.branches.append(
                    DepthwiseConv1d(
                        in_channels,
                        in_channels,
                        kernel_size=kernel_size,
                        padding=(kernel_size - 1) // 2,
                    )
                )

        if self.merge_mode == "concat":
            self.output_proj = PointwiseConv1d(len(kernel_sizes) * in_channels, in_channels)

        self.batch_norm = nn.BatchNorm1d(in_channels)
        self.activation = Swish()
        self.post_pointwise = PointwiseConv1d(in_channels, in_channels)
        self.dropout = nn.Dropout(p=dropout_p)

    def forward(self, inputs: Tensor) -> Tensor:
        x = self.norm(inputs)       
        x = self.transpose(x)       
        x = self.pre_pointwise(x)  
        x = self.glu(x)        


        branch_outputs = [branch(x) for branch in self.branches]  # List[(B, C, T)]

        if self.merge_mode == "gated":
            gated_outputs = []
            for i, output in enumerate(branch_outputs):
                gate = self.gating_units[i](output)  
                gated_output = output * gate  
                gated_outputs.append(gated_output)
            out = sum(gated_outputs)  
        elif self.merge_mode == "avg":
            out = sum(branch_outputs) / len(branch_outputs)
        else:  # concat
            out = torch.cat(branch_outputs, dim=1)  
            out = self.output_proj(out)  

        out = self.batch_norm(out)
        out = self.activation(out)
        out = self.post_pointwise(out)
        out = self.dropout(out)

        return out.transpose(1, 2) 


class Conv2dSubampling(nn.Module): # abandon
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super(Conv2dSubampling, self).__init__()
        self.sequential = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2),
            nn.ReLU(),
        )

    def forward(self, inputs: Tensor, input_lengths: Tensor) -> Tuple[Tensor, Tensor]:
        outputs = self.sequential(inputs.unsqueeze(1))
        batch_size, channels, subsampled_lengths, sumsampled_dim = outputs.size() # B,C,T,F

        outputs = outputs.permute(0, 2, 1, 3) # B,T,C,F
        outputs = outputs.contiguous().view(batch_size, subsampled_lengths, channels * sumsampled_dim) # B,T,C*F

        output_lengths = input_lengths >> 2 # input_lengths // 4

        output_lengths -= 1

        return outputs, output_lengths
    
class RelPositionalEncoding(nn.Module):

    def __init__(self, d_model: int = 512, max_len: int = 5000) -> None:
        super(RelPositionalEncoding, self).__init__()
        self.d_model = d_model
        self.pe = None
        self.extend_pe(torch.tensor(0.0).expand(1, max_len))

    def extend_pe(self, x: Tensor) -> None:
        if self.pe is not None:
            if self.pe.size(1) >= x.size(1) * 2 - 1:
                if self.pe.dtype != x.dtype or self.pe.device != x.device:
                    self.pe = self.pe.to(dtype=x.dtype, device=x.device)
                return

        pe_positive = torch.zeros(x.size(1), self.d_model)
        pe_negative = torch.zeros(x.size(1), self.d_model)
        position = torch.arange(0, x.size(1), dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2, dtype=torch.float32) * -(math.log(10000.0) / self.d_model)
        )
        pe_positive[:, 0::2] = torch.sin(position * div_term)
        pe_positive[:, 1::2] = torch.cos(position * div_term)
        pe_negative[:, 0::2] = torch.sin(-1 * position * div_term)
        pe_negative[:, 1::2] = torch.cos(-1 * position * div_term)

        pe_positive = torch.flip(pe_positive, [0]).unsqueeze(0)
        pe_negative = pe_negative[1:].unsqueeze(0)
        pe = torch.cat([pe_positive, pe_negative], dim=1)
        self.pe = pe.to(device=x.device, dtype=x.dtype)

    def forward(self, x: Tensor) -> Tensor:
        self.extend_pe(x)
        assert self.pe is not None
        pos_emb = self.pe[:,self.pe.size(1) // 2 - x.size(1) + 1 : self.pe.size(1) // 2 + x.size(1),
        ]
        return pos_emb
    
class ExBimamba(nn.Module):
    def __init__(
        self,
        d_model,
        d_state=16,
        d_conv=4,
        expand=2,
        device=None,
        dtype=None,
        Amatrix_type='default'
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.forward_mamba = Mamba(d_model=self.d_model, d_state=self.d_state, d_conv=self.d_conv, expand=self.expand)    
        self.backward_mamba = Mamba(d_model=self.d_model, d_state=self.d_state, d_conv=self.d_conv, expand=self.expand)
        self.output_proj = nn.Linear(2*self.d_model, self.d_model)     
    
    def forward(self, hidden_input):
        forward_output = self.forward_mamba(hidden_input)
        backward_output = self.backward_mamba(hidden_input.flip([1]))
        res = torch.cat((forward_output, backward_output.flip([1])), dim=-1)
        res = self.output_proj(res)
        return res
    
class ConbimambaBlock(nn.Module):
    def __init__(
            self,
            encoder_dim: int = 512,
            num_attention_heads: int = 8,
            feed_forward_expansion_factor: int = 4,
            conv_expansion_factor: int = 2,
            feed_forward_dropout_p: float = 0.1,
            attention_dropout_p: float = 0.1,
            conv_dropout_p: float = 0.1,
            # conv_kernel_size: int = 31,
            conv_kernel_sizes: list = [15,31,63],
            half_step_residual: bool = True,
            merge_mode: str = 'avg'
    ):
        super(ConbimambaBlock, self).__init__()
        if half_step_residual:
            self.feed_forward_residual_factor = 0.5
        else:
            self.feed_forward_residual_factor = 1

        self.sequential = nn.Sequential(
            ResidualConnectionModule(
                module=FeedForwardModule(
                    encoder_dim=encoder_dim,
                    expansion_factor=feed_forward_expansion_factor,
                    dropout_p=feed_forward_dropout_p,
                ),
                module_factor=self.feed_forward_residual_factor,
            ),
            ResidualConnectionModule(
                module=ExBimamba(
                    d_model=encoder_dim,
                ),
            ),


            ResidualConnectionModule(
                module=MultiScaleConformerConvModule(
                    in_channels=encoder_dim,
                    kernel_sizes=conv_kernel_sizes,  
                    expansion_factor=conv_expansion_factor,
                    dropout_p=conv_dropout_p,
                    merge_mode=merge_mode,
                ),
            ),


            ResidualConnectionModule(
                module=FeedForwardModule(
                    encoder_dim=encoder_dim,
                    expansion_factor=feed_forward_expansion_factor,
                    dropout_p=feed_forward_dropout_p,
                ),
                module_factor=self.feed_forward_residual_factor,
            ),
            nn.LayerNorm(encoder_dim),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return self.sequential(inputs)


class ConBiMambaEncoder(nn.Module):
    def __init__(
            self,
            input_dim: int = 768,  
            encoder_dim: int = 256,  
            num_layers: int = 6,
            num_attention_heads: int = 8,
            feed_forward_expansion_factor: int = 2,
            conv_expansion_factor: int = 2,
            input_dropout_p: float = 0.1,
            feed_forward_dropout_p: float = 0.1,
            attention_dropout_p: float = 0.1,
            conv_dropout_p: float = 0.1,
            # conv_kernel_size: int = 31,
            conv_kernel_sizes: list = [15, 31, 63],
            half_step_residual: bool = True,
            use_1d_subsample: bool = False,
            merge_mode: str = 'avg'
            
    ):
        super(ConBiMambaEncoder, self).__init__()
        
        self.use_1d_subsample = use_1d_subsample
        if self.use_1d_subsample:
            self.conv1d_subsample = nn.Conv1d(
                in_channels=input_dim,
                out_channels=input_dim,
                kernel_size=3,
                stride=2,
                padding=1,
                groups=input_dim  
            )
        
        self.input_projection = nn.Sequential(
            Linear(input_dim, encoder_dim), 
            nn.Dropout(p=input_dropout_p),
        )
        
        self.layers = nn.ModuleList([ConbimambaBlock(
            encoder_dim=encoder_dim,
            num_attention_heads=num_attention_heads,
            feed_forward_expansion_factor=feed_forward_expansion_factor,
            conv_expansion_factor=conv_expansion_factor,
            feed_forward_dropout_p=feed_forward_dropout_p,
            attention_dropout_p=attention_dropout_p,
            conv_dropout_p=conv_dropout_p,
            # conv_kernel_size=conv_kernel_size,
            conv_kernel_sizes = conv_kernel_sizes,
            half_step_residual=half_step_residual,
            merge_mode=merge_mode,
        ) for _ in range(num_layers)])

    def count_parameters(self) -> int:
        return sum([p.numel() for p in self.parameters()])

    def update_dropout(self, dropout_p: float) -> None:
        for name, child in self.named_children():
            if isinstance(child, nn.Dropout):
                child.p = dropout_p


    def forward(self, inputs: Tensor, input_lengths: Tensor) -> Tuple[Tensor, Tensor]:
        outputs = inputs 
        output_lengths = input_lengths  

        if self.use_1d_subsample:
            outputs = outputs.transpose(1, 2) 
            outputs = self.conv1d_subsample(outputs)  
            outputs = outputs.transpose(1, 2)  
            output_lengths = (output_lengths + 1) // 2  
        
        outputs = self.input_projection(outputs)

        layer_outputs = []   

        for layer in self.layers:
            outputs = layer(outputs)
            layer_outputs.append(outputs)
        return layer_outputs, outputs, output_lengths
    
class ConBiMamba(nn.Module):
    def __init__(
            self,
            num_classes: int,
            input_dim: int = 80,
            encoder_dim: int = 512,
            num_encoder_layers: int = 17,
            num_attention_heads: int = 8,
            feed_forward_expansion_factor: int = 4,
            conv_expansion_factor: int = 2,
            input_dropout_p: float = 0.1,
            feed_forward_dropout_p: float = 0.1,
            attention_dropout_p: float = 0.1,
            conv_dropout_p: float = 0.1,
            # conv_kernel_size: int = 31,
            conv_kernel_sizes: list = [15, 31, 63],
            half_step_residual: bool = True,
            merge_mode: str = 'avg'
    ) -> None:
        super(ConBiMamba, self).__init__()
        self.encoder = ConBiMambaEncoder(
            input_dim=input_dim,
            encoder_dim=encoder_dim,
            num_layers=num_encoder_layers,
            num_attention_heads=num_attention_heads,
            feed_forward_expansion_factor=feed_forward_expansion_factor,
            conv_expansion_factor=conv_expansion_factor,
            input_dropout_p=input_dropout_p,
            feed_forward_dropout_p=feed_forward_dropout_p,
            attention_dropout_p=attention_dropout_p,
            conv_dropout_p=conv_dropout_p,
            # conv_kernel_size=conv_kernel_size,
            conv_kernel_sizes=conv_kernel_sizes,
            half_step_residual=half_step_residual,
            merge_mode=merge_mode
        )
        self.fc = Linear(encoder_dim, num_classes, bias=False)

    def count_parameters(self) -> int:
        """ Count parameters of encoder """
        return self.encoder.count_parameters()

    def update_dropout(self, dropout_p) -> None:
        """ Update dropout probability of model """
        self.encoder.update_dropout(dropout_p)

    def forward(self, inputs: Tensor, input_lengths: Tensor) -> Tuple[Tensor, Tensor]:
        _, encoder_outputs, encoder_output_lengths = self.encoder(inputs, input_lengths)
        outputs = self.fc(encoder_outputs)
        outputs = nn.functional.log_softmax(outputs, dim=-1)
        return outputs, encoder_output_lengths

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_classes = 4
    input_dim = 768
    encoder_dim = 256
    seq_len = 100
    batch_size = 2

    model = ConBiMamba(
        num_classes=num_classes,
        input_dim=input_dim,
        encoder_dim=encoder_dim,
        num_encoder_layers=6,
        conv_kernel_sizes=[15, 31, 63],
        merge_mode='gated'
    ).to(device) 

    x = torch.randn((batch_size, seq_len, input_dim)).to(device)
    input_lengths = torch.full((batch_size,), seq_len, dtype=torch.long).to(device)

    outputs, out_lengths = model(x, input_lengths)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {outputs.shape}")       # (B, T, num_classes)
    print(f"Output lengths: {out_lengths}")
    print(model)
