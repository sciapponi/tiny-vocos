import torch
from torch import nn 
import torch.nn.functional as F
import torch 
from torch import nn, sin, pow
from torch.nn import Parameter
from torch.distributions.exponential import Exponential

class InstanceNorm(nn.Module):

    def __init__(self, epsilon=1e-8):
        """
            @notice: avoid in-place ops.
            https://discuss.pytorch.org/t/encounter-the-runtimeerror-one-of-the-variables-needed-for-gradient-computation-has-been-modified-by-an-inplace-operation/836/3
        """
        super(InstanceNorm, self).__init__()
        self.epsilon = epsilon

    def forward(self, x):
        x   = x - torch.mean(x, (2, 3), True)
        tmp = torch.mul(x, x) # or x ** 2
        tmp = torch.rsqrt(torch.mean(tmp, (2, 3), True) + self.epsilon)
        return x * tmp


class Snake(nn.Module):
    '''         
    Implementation of the serpentine-like sine-based periodic activation function:
    .. math::
         Snake_a := x + \frac{1}{a} sin^2(ax) = x - \frac{1}{2a}cos{2ax} + \frac{1}{2a}
    This activation function is able to better extrapolate to previously unseen data,
    especially in the case of learning periodic functions

    Shape:
        - Input: (N, *) where * means, any number of additional
          dimensions
        - Output: (N, *), same shape as the input
        
    Parameters:
        - a - trainable parameter
    
    References:
        - This activation function is from this paper by Liu Ziyin, Tilman Hartwig, Masahito Ueda:
        https://arxiv.org/abs/2006.08195
        
    Examples:
        >>> a1 = snake(256)
        >>> x = torch.randn(256)
        >>> x = a1(x)
    '''
    def __init__(self, in_features, a=None, trainable=True):
        '''
        Initialization.
        Args:
            in_features: shape of the input
            a: trainable parameter
            trainable: sets `a` as a trainable parameter
            
            `a` is initialized to 1 by default, higher values = higher-frequency, 
            5-50 is a good starting point if you already think your data is periodic, 
            consider starting lower e.g. 0.5 if you think not, but don't worry, 
            `a` will be trained along with the rest of your model
        '''
        super(Snake,self).__init__()
        self.in_features = in_features if isinstance(in_features, list) else [in_features]

        # Initialize `a`
        if a is not None:
            self.a = Parameter(torch.ones(self.in_features) * a) # create a tensor out of alpha
        else:            
            m = Exponential(torch.tensor([0.1]))
            self.a = Parameter((m.rsample(self.in_features)).squeeze()) # random init = mix of frequencies

        self.a.requiresGrad = trainable # set the training of `a` to true

    def forward(self, x):
        '''
        Forward pass of the function.
        Applies the function to the input elementwise.
        Snake ∶= x + 1/a* sin^2 (xa)
        '''
        return  x + (1.0/self.a) * pow(sin(x * self.a), 2)

def autopad(k, p=None):  # kernel, padding
    # Pad to 'same'
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p

class SnakeXiConv(nn.Module):
    # XiNET convolution with Snake Activation Function
    def __init__(self, c1, c2, k=3, s=1, p=None, g=1, act=True, compression=4, attention=True, skip_tensor_in=None, skip_channels=1, pool=None, pool_stride=None, upsampling=1, attention_k=3, attention_lite=True, norm='batchnorm', dropout_rate=0,  skip_k=1):  # ch_in, ch_out, kernel, stride, padding, groups
        super().__init__()
        self.compression = compression
        self.attention = attention
        self.attention_lite = attention_lite
        self.attention_lite_ch_in = c2//compression
        self.pool = pool
        self.norm = norm
        self.dropout_rate = dropout_rate
        self.upsampling = upsampling

        self.compression_conv = nn.Conv2d(c1, c2//compression, 1, 1,  groups=g, padding='same', bias=False)
        self.main_conv = nn.Conv2d(c2//compression if compression>1 else c1, c2, k, s,  groups=g, padding='same' if s==1 else autopad(k, p), bias=False)
        # self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())
        self.act = Snake()
        
        if attention:
            if attention_lite:
                self.att_pw_conv= nn.Conv2d(c2, self.attention_lite_ch_in, 1, 1, groups=g, padding='same', bias=False)
            self.att_conv = nn.Conv2d(c2 if not attention_lite else self.attention_lite_ch_in, c2, attention_k, 1, groups=g, padding='same', bias=False)
            self.att_act = nn.Sigmoid()

        if pool:
            self.mp = nn.MaxPool2d(pool, pool_stride)
        if skip_tensor_in:
            self.skip_conv = nn.Conv2d(skip_channels, c2//compression, skip_k, 1,  groups=g, padding='same', bias=False)

        if norm=='instancenorm':
            self.bn = InstanceNorm()
        elif norm=='batchnorm':
            self.bn = nn.BatchNorm2d(c2)

        if dropout_rate>0:
            self.do = nn.Dropout(dropout_rate)
        


    def forward(self, x):
        s = None
        # skip connection
        if isinstance(x, list):
            s = F.adaptive_avg_pool2d(x[1], output_size=x[0].shape[2:])
            s = self.skip_conv(s)
            x = x[0]
        #     print(f'Skip shape {s.shape}')
        # print(f'Input shape {x.shape}')

        # compression convolution
        if self.compression > 1:
            x = self.compression_conv(x)
            
        if s is not None:
            # print(f'Tensor shape {x.shape}')
            x = x+s

        if self.pool:
            x = self.mp(x)
        if self.upsampling > 1:
            x = F.interpolate(x, scale_factor=self.upsampling, mode='bilinear')
        # main conv and activation
        x = self.main_conv(x)
        if self.norm:
            x = self.bn(x)
        x = self.act(x)

        # attention conv
        if self.attention:
            if self.attention_lite:
                att_in=self.att_pw_conv(x)
            else:
                att_in=x
            y = self.att_act(self.att_conv(att_in))
            x = x*y

        
        if self.dropout_rate > 0:
            x = self.do(x)

        # print(f'Output shape {x.shape} \n')
        return x
