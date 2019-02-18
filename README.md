# pytorch-densenet
Simple Code Implementation of ["DenseNet"](https://arxiv.org/pdf/1608.06993.pdf) architecture using PyTorch.


![](https://github.com/hoya012/pytorch-densenet/blob/master/assets/1.PNG)

For simplicity, i write codes in `ipynb`. So, you can easliy test my code.

*Last update : 2019/1/29*

## Contributor
* hoya012

## 0. Requirements
```
python=3.5
numpy
matplotlib
torch=1.0.0
torchvision
torchsummary
```

## 1. Usage
You only run `DenseNet-BC-CIFAR10.ipynb`. 

Or you can use Google Colab for free!! This is [colab link](https://colab.research.google).

After downloading ipynb, just upload to your google drive. and run!

For training, testing, i used `CIFAR-10` Dataset.

## 2. Paper Review & Code implementation Blog Posting (Korean Only)
[“DenseNet Tutorial [1] Paper Review & Implementation details”](https://hoya012.github.io/blog/DenseNet-Tutorial-1/)  
[“DenseNet Tutorial [2] PyTorch Code Implementation”](https://hoya012.github.io/blog/DenseNet-Tutorial-2/)


## 3. DenseNet and other layers impelemtation.
In DenseNet, there are many DenseBlock. This is my simple implemenatation.


### Bottleneck layer
```python
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
```python
class Transition_layer(nn.Sequential):
  def __init__(self, nin, theta=0.5):    
      super(Transition_layer, self).__init__()
      
      self.add_module('conv_1x1', bn_relu_conv(nin=nin, nout=int(nin*theta), kernel_size=1, stride=1, padding=0, bias=False))
      self.add_module('avg_pool_2x2', nn.AvgPool2d(kernel_size=2, stride=2, padding=0))
```

### DenseBlock
![](https://github.com/hoya012/pytorch-densenet/blob/master/assets/2.png)
```python
class DenseBlock(nn.Sequential):
  def __init__(self, nin, num_bottleneck_layers, growth_rate, drop_rate=0.2):
      super(DenseBlock, self).__init__()
                        
      for i in range(num_bottleneck_layers):
          nin_bottleneck_layer = nin + growth_rate * i
          self.add_module('bottleneck_layer_%d' % i, bottleneck_layer(nin=nin_bottleneck_layer, growth_rate=growth_rate, drop_rate=drop_rate))
```

## 4. DenseNet architecture for CIFAR-10
![](https://github.com/hoya012/pytorch-densenet/blob/master/assets/3.png)

The DenseNet architecture for CIFAR-10 differs from the architecture table presented in the paper. The DenseNet architecture applicable to CIFAR-10 is shown in the figure above.
