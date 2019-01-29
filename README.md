# pytorch-densenet
Simple Code Implementation of ["DenseNet"](https://arxiv.org/pdf/1608.06993.pdf) architecture using PyTorch.


![](https://github.com/hoya012/pytorch-densenet/blob/master/assets/1.PNG)

For simplicity, i write codes in `ipynb`. So, you can easliy test my code.

*Last update : 2019/1/29*

## Contributor
* hoya012

## Requirements
Python 3.5
```
numpy
matplotlib
torch=1.0.0
torchvision
torchsummary
```

## Usage
You only run `DenseNet-BC-CIFAR10.ipynb`.
For training, testing, i used `CIFAR-10` Dataset.

## DenseBlock and other blocks impelemtation.
In DenseNet, there are many DenseBlock. This is my simple implemenatation.

### Bottleneck layer
```
class bottleneck_layer(nn.Sequential):
  def __init__(self, nin, growth_rate, drop_rate=0.2):    
      super(bottleneck_layer, self).__init__()
      
      self.add_module('conv_1x1', bn_relu_conv(nin=nin, nout=growth_rate*4, kernel_size=1, stride=1, padding=0, bias=False))
      self.add_module('conv_3x3', bn_relu_conv(nin=growth_rate*4, nout=growth_rate, kernel_size=3, stride=1, padding=1, bias=False))
      
      self.drop_rate = drop_rate
      
  def forward(self, x):
      bottleneck_output = super(bottleneck_layer, self).forward(x)
      if self.drop_rate > 0:
          bottleneck_output = F.dropout(bottleneck_output, p=self.drop_rate, training=self.training)
          
      bottleneck_output = torch.cat((x, bottleneck_output), 1)
      
      return bottleneck_output
```

### Transition layer
```
class Transition_layer(nn.Sequential):
  def __init__(self, nin, theta=0.5):    
      super(Transition_layer, self).__init__()
      
      self.add_module('conv_1x1', bn_relu_conv(nin=nin, nout=int(nin*theta), kernel_size=1, stride=1, padding=0, bias=False))
      self.add_module('avg_pool_2x2', nn.AvgPool2d(kernel_size=2, stride=2, padding=0))
```

### DenseBlock
![](https://github.com/hoya012/pytorch-densenet/blob/master/assets/2.png)
```
class DenseBlock(nn.Sequential):
  def __init__(self, nin, num_bottleneck_layers, growth_rate, drop_rate=0.2):
      super(DenseBlock, self).__init__()
                        
      for i in range(num_bottleneck_layers):
          nin_bottleneck_layer = nin + growth_rate * i
          self.add_module('bottleneck_layer_%d' % i, bottleneck_layer(nin=nin_bottleneck_layer, growth_rate=growth_rate, drop_rate=drop_rate))
```

## DenseNet architecture for CIFAR-10
![](https://github.com/hoya012/pytorch-densenet/blob/master/assets/3.png)
