# MobileNetV2_C

C implementation of MobileNetV2 using only static libraries.

File hierarchy: main -> actions -> blocks -> layers -> operations

## Work so far

 - Loaded pre-trained network in Python and tested on a random cat image, generating a correct answer.
 - Extracted weights from first layer and also exported the cat image.
 - Developed a 2D-Convolution function in C to emulate the first layer (Conv2D).
 - Ran the layer in C with the weights and image from Python, and wrote one of the resulting 112x112 images to a file.
 - Compared first layer convolved image to the equivalent one in Python, resulting in the exact same image. Thus, proving the first layer is working correctly.
 - Repeated the same process for Batch Normalization, ReLU6, Depthwise, Pad and Add layers.
 - Completed Stride 2 Block and Stride 1 Block.
 - Completed all main blocks.
 - Completed Pooling and Softmax functions.
 - Completed Fully Connected Network.
 - Inference finished and tested.
----
 - Implemented training in Python for comparison with C implementation
 - Formulating training equations

## Training Equations

From the [MobileNetV2 Article](https://arxiv.org/abs/1801.04381):
>We train our models using TensorFlow. We use the standard RMSPropOptimizer with both decay and momentum set to 0.9. We use batch normalization after every layer, and the standard weight decay is set to 0.00004. Following MobileNetV1 setup we use initial learning rate of 0.045, and learning rate decay rate of 0.98 per epoch. We use 16 GPU asynchronous workers, and a batch size of 96.

A "batch size of 96" means this is not stochastic gradient descent, but mini-batch gradient descent. Hence, 96 images are randomly selected from the training set, each image is passed through the network and their respective gradients are averaged, and then the parameters are updated using the mean gradient. This process is repeated until the dataset ends, which constitutes one epoch.

### Fully Connected Layer

Gradient is affected by: Cross-entropy loss, Softmax activation, stardard activation, and stardard weight decay. Final parameter change is additionally affected by learning rate and RMSProp. Assuming standard learning rate decay.

Terminology:
 - I is the value of a neuron in the input layer
 - Z is the activation of a neuron in the output layer *before softmax*
 - O is the activation of a neuron in the output layer *after softmax*
 - L is the cross-entropy loss of the classification values
 - W represents the weights, B represents the biases
 - T is the target of each classification neuron (in one-hot encoding this is either 0 or 1)
 - i is the index of a neuron in the input layer (1 - 1280)
 - j is the index of a neuron in the output layer (1 - 1000)
 - c is the index of the correct prediction as stated by the label
 - Lambda is the standard weight decay (= 0.00004)
 - Delta is the [Kronecker Delta](https://en.wikipedia.org/wiki/Kronecker_delta)

Cross-entropy loss (with standard weight decay):
```math
L = -\sum_{j}[T_{j}ln(O_{j})] + \frac{\lambda W_{ij}^{2}}{2} \rightarrow \text{one-hot} \rightarrow -ln(O_{c}) + \frac{\lambda W_{ij}^{2}}{2}
```

Softmax:
```math
Softmax(Z_{j}) = \frac{e^{Z_{j}}}{\sum_{k}e^{Z_{k}}} = \frac{e^{Z_{j}}}{S}
```

Activation:
```math
Z_{j} = \sum_{i}(I_{i} \cdot W_{ij}) + B_{j}
```

Weight gradient:
```math
\frac{\partial L}{\partial W_{ij}} = -\frac{\partial Z_{j}}{\partial W_{ij}} \cdot \frac{\partial L}{\partial Z_{j}} + \frac{\partial \left [\frac{\lambda W_{ij}^2}{2}  \right ]}{\partial W_{ij}} = -\frac{\partial Z_{j}}{\partial W_{ij}} \cdot \frac{\partial L}{\partial Z_{j}} + \lambda W_{ij}
```
```math
\frac{\partial Z_{j}}{\partial W_{ij}} = \frac{\partial [\sum_{p}(I_{p}W_{pj}+B_{j}) ] }{\partial W_{ij}} =  \sum_{p}\left [ I_{p} \cdot \frac{\partial W_{pj}}{\partial W_{ij}} \right ] = \sum_{p}\left [ I_{p} \delta_{ip} \right ] = I_{i}
```
```math
\frac{\partial L}{\partial Z_{j}} = \frac{\partial [ln(\frac{e^{Z_{c}}}{S})]}{\partial Z_{j}} = \frac{\partial \left [Z_{c} - ln(S)  \right ]}{\partial Z_{j}} = \delta_{jc} - \frac{1}{S} \cdot \frac{\partial S}{\partial Z_{j}} = \delta_{jc} - \frac{1}{S} \cdot \frac{\partial [\sum_{k}(e^{z_{k}})]}{\partial Z_{j}} = \delta_{jc} - \frac{1}{S} \cdot \sum_{k}[\delta_{jk}e^{Z_{k}}] = \delta_{jc} - \frac{e^{Z_{j}}}{S} = \delta_{jc} - O_{j}
```
```math
\frac{\partial L}{W_{ij}} = -I_{i} (\delta_{jc} - O_{j}) + \lambda W_{ij} = I_{i} (O_{j} -\delta_{jc}) + \lambda W_{ij}
```

Bias gradient (no standard weight decay):
```math
\frac{\partial L}{\partial B_{j}} = -\frac{\partial Z_{j}}{\partial B_{j}} \cdot \frac{\partial L}{\partial Z_{j}}
```
```math
\frac{\partial Z_{j}}{\partial B_{j}} = \frac{\partial [ \sum_{p} ( I_{p} \cdot W_{pj} ) + B_{j}  ] }{\partial B_{j}} = 1
```
```math
\frac{\partial L}{B_{j}} = -(\delta_{jc} - O_{j}) = O_{j} - \delta_{jc}
```

Backpropagated Error:
```math
\frac{\partial L}{\partial I_{i}} = -\sum_{j} \left [\frac{\partial Z_{j}}{\partial I_{i}} \cdot \frac{\partial L}{\partial Z_{j}} \right ]
```
```math
\frac{\partial Z_{j}}{\partial I_{i}} = \frac{\partial [ \sum_{p} ( I_{p} \cdot W_{pj} ) + B_{j}  ] }{\partial I_{i}} = W_{ij}
```
```math
\frac{\partial L}{\partial I_{i}} = - \sum_{j} \left [W_{ij}(\delta_{jc} - O_{j})\right ]  = \sum_{j} \left [W_{ij}(O_{j} - \delta_{jc})  \right ]
```

### Average Pooling Layer

Error backpropagates through the derivative of this function.

Terminology:
 - I is the value of an input to the pooling layer (7x7x1280)
 - O is the value of an output to the pooling layer (1280)
 - i is the index of the first dimention of the input (1 - 7)
 - y is the index of the second dimention of the input (1 - 7)
 - k is the index of the dimention of the output (1280)
 - A is the Global Average Pooling function

Global Average Pooling equation:
```math
A(I_{k}) = \frac{1}{49} \sum_{i}\left [\sum_{j}(I_{ijk})  \right ]
```

Differential:
```math
\frac{\partial A(O_{k})}{\partial I_{pqr}} = \frac{1}{49} \sum_{i}\left [\sum_{j}\left ( \frac{\partial I_{ijk}}{\partial I_{pqr}} \right )  \right ] = \frac{1}{49} \sum_{i}\left [\sum_{j}\left ( \delta_{ip} \delta_{jq} \delta_{kr} \right )  \right ] = \frac{\delta_{kr}}{49}
```

This means that for each output value, its error is multiplied by 1/49 and assigned to each value of the corresponding 7x7 block.

### ReLU6

Error backpropagates through the derivative of this function.

Terminology:
 - I is the value of an input to the ReLU6 layer
 - O is the value of an output to the ReLU6 layer
 - i, j and k are the dimentions of the input and output

ReLU6 equation:
 ```math
 ReLU6(I_{ijk}) = \begin{cases}
0 & \text{ if } I_{ijk} < 0 \\ 
I_{ijk} & \text{ if } 0 < I_{ijk} < 6 \\ 
6 & \text{ if } I_{ijk} > 6 
\end{cases}
 ```

Differential:
 ```math
 ReLU6'(I_{ijk}) = \begin{cases}
0 & \text{ if } I_{ijk} < 0 \\ 
1 & \text{ if } 0 < I_{ijk} < 6 \\ 
0 & \text{ if } I_{ijk} > 6 
\end{cases}
 ```

This means that for each output value, its error is multiplied by 0 when it is below 0 or above 6 and multiplied by 1 when it is in between these two values.

### Batch Normalization

Batch Normalization is different during inference or during training. During inference, the mean and standard variation are calculated based on only the image in question, using the moving averages previously calculated. During training, the mean and variance are instead calculated based on the current mini-batch, and the moving averages are updated.

Here, the trainable parameters are gamma and beta, and the error also backpropagates.

Terminology:
 - I and O are the the values of an input and output to the BN layer respectivelly
 - i, j and k are the indexes of the dimensions of the input and output
 - x and y are the indexes of the first two dimentions of all mini-batch members
 - Mu is the moving mean and Sigma squared is the moving variance
 - Gamma and Beta are the trainable parameters "scale" and "shift" respectivelly
 - Rho is the momentum of the moving equations (= 0.999)
 - Epsilon is a small constant for numerical stability (= 0.001)
 - T is the total number of neurons across dimensions i and j of all mini-batch members
 - mb is the index representing the current mini-batch
 - t is the index representing epochs

Batch Normalization Moving Equations:

```math
\mu_{mb} = \frac{\sum_{xy}I_{xy}}{T}
```
```math
\sigma_{mb}^2 = \frac{\sum_{xy}(I_{xy} - \mu_{mb})^2}{T}
```
```math
\mu_{mb} = \mu_{mb-1} \cdot \rho + (1 - \rho) \mu_{mb}
```
```math
\sigma_{mb}^{2} = \sigma_{mb-1}^{2} \cdot \rho + (1 - \rho) \sigma_{mb}^{2}
```

Output Equations:
```math
\begin{cases}
\hat{I}_{ij} = \frac{I_{ij} - \mu_{mb}}{\sqrt{\sigma^2_{mb}+\epsilon}} & \text{ if } t=0 \\ 
\hat{I}_{ij} = \frac{I_{ij} - \mu}{\sqrt{\sigma^2+\epsilon}} & \text{ if } t \neq 0 
\end{cases}
```
```math
O_{ij} = \gamma \hat{I}_{ij} + \beta
```

Gradient of Beta:
```math
\frac{\partial L}{\partial \beta} = \sum_{ij} \left [\frac{\partial L}{\partial O_{ij}} \cdot \frac{\partial O_{ij}}{\partial \beta} \right ] = \sum_{ij} \left [\frac{\partial L}{\partial O_{ij}} \right ]
```

Gradient of Gamma:
```math
\frac{\partial L}{\partial \gamma_{k}} = \sum_{ij} \left [\frac{\partial L}{\partial O_{ij}} \cdot \frac{\partial O_{ij}}{\partial \gamma} \right ] = \sum_{ij} \left [\frac{\partial L}{\partial O_{ij}} \cdot \hat{I}_{ij} \right ]
```

Backpropagating Error:
```math
\frac{\partial L}{\partial I_{ij}} = \frac{\partial L}{\partial I_{ij}} \cdot \frac{\partial \hat{I}_{ij}}{\partial I_{ij}} + \frac{\partial L}{\partial \sigma^2} \cdot \frac{\partial \sigma^2}{\partial I_{ij}} + \frac{\partial L}{\partial \mu} \cdot \frac{\partial \mu}{\partial I_{ij}}
```
```math
\frac{\partial L}{\partial \hat{I}_{ij}} = \frac{\partial L}{\partial O_{ij}} \cdot \frac{\partial O_{ij}}{\partial \hat{I}_{ij}} = \frac{\partial L}{\partial O_{ij}} \gamma
```
```math
\frac{\partial L}{\partial \sigma^2} = \sum_{ij} \left [\frac{\partial L}{\partial \hat{I}_{ij}} \cdot \frac{\partial \hat{I}_{ij}}{\partial \sigma^2}  \right ] = \sum_{ij} \left [\frac{\partial L}{\partial \hat{I}_{ij}} (I_{ij} - \mu) \frac{-1}{2} (\sigma^2 + \epsilon)^{-3/2}  \right ]
```
```math
\frac{\partial L}{\partial \mu} = \sum_{ij} \left [\frac{\partial L}{\partial \hat{I}_{ij}} \cdot \frac{\partial \hat{I}_{ij}}{\partial \mu} \right ] + \frac{\partial L}{\partial \sigma^2} \cdot \frac{\partial \sigma^2}{\partial \mu} = \sum_{ij} \left [\frac{\partial L}{\partial \hat{I}_{ij}} \cdot \frac{-1}{\sqrt{\sigma^2 + \epsilon}}  \right ] + \frac{\partial L}{\partial \sigma^2} \cdot \frac{\sum_{ij}[-2(I_{ij}-\mu)]}{d^2} = \sum_{ij} \left [\frac{\partial L}{\partial \hat{I}_{ij}} \cdot \frac{-1}{\sqrt{\sigma^2 + \epsilon}}  \right ]
```
```math
\frac{\partial \hat{I}_{ij}}{\partial I_{ij}} = \frac{1}{\sqrt{\sigma^2 + \epsilon}} ;   \frac{\partial \sigma^2}{\partial I_{ij}} = \frac{2(I_{ij}-\mu)}{d^2} ; \frac{\partial \mu}{\partial I_{ij}} = \frac{1}{d^2}
```
```math
\frac{\partial L}{\partial I_{ij}} = \frac{\partial L}{\partial \hat{I}_{ij}} \frac{1}{\sqrt{\sigma^2 + \epsilon}} + \frac{\partial L}{\partial \sigma^2} \frac{2(I_{ij}-\mu)}{d^2} + \frac{\partial L}{\partial \mu} \frac{1}{d^2}
```

### 2D-Convolution

Terminology:
 - I is the value of an input to the convolution function.
 - O is the value of an output to the convolution function.
 - L is the loss function (representing backpropagated error)
 - i and j are indexes representing the dimensions of a filter
 - p and q are indexes representing the dimensions of an output
 - a and b are indexes representing the dimensions of an input
 - k is the number of filters (and by extension the number of channels of the output)
 - c is the number of channels of the input
 - s is an index representing stride and pad is an index representing padding

Weight Chain Rule:
```math
\frac{\partial L}{\partial W_{ijk}} = \sum_{pq} \left [\frac{\partial O_{pqk}}{\partial W_{ijk}} \frac{\partial L}{\partial O_{pqk}}  \right ]
```
```math
a = s \cdot p + i - pad
```
```math
b = s \cdot q + j - pad
```
```math
O_{pqk} = \sum_{ijc} \left [ I_{abc} \cdot W_{ijk} \right ]
```
```math
\frac{\partial O_{pqk}}{\partial W_{xyk}} = \frac{\partial \left [\sum_{ijc}\left ( I_{abc} \cdot W_{ijk} \right )  \right ]}{\partial W_{xyk}} = \sum_{ijc} \left ( I_{abc} \delta_{ix} \delta_{jy} \right ) \Rightarrow \frac{\partial O_{pqk}}{\partial W_{ijk}} = \sum_{ijc} I_{s \cdot p + i - pad, s \cdot q + j - pad, c}
```

Final correction formula:
```math
\frac{\partial L}{\partial W_{ijk}} = \sum_{pq} \left [\frac{\partial O_{pqk}}{\partial W_{ijk}} \frac{\partial L}{\partial O_{pqk}}  \right ] = \sum_{pq} \left [  \frac{\partial L}{\partial O_{pqk}} \sum_{ijc} I_{abc} \right ]
```

Therefore, to get the gradient of a specific weight: for each output, you multiply the backpropagated error of that neuron by the input that this weight connects to (which depends on stride), and sum everything up (this doesn't reflect very well in the math).

Backpropagated Error:
```math
\frac{\partial L}{\partial I_{abc}} = \sum_{pq} \left [\frac{\partial O_{pqk}}{\partial I_{abc}} \frac{\partial L}{\partial O_{pqk}}  \right ]
```
```math
\frac{\partial O_{pqk}}{\partial I_{xyz}} = \frac{\partial \left [\sum_{ijc}\left ( I_{abc} \cdot W_{ijk} \right )  \right ]}{\partial I_{xyz}} = \sum_{ijc} \left ( W_{ijk} \delta_{ax} \delta_{by} \delta_{cz} \right ) \Rightarrow \frac{\partial O_{pqk}}{\partial I_{abc}} = \sum_{ij} W_{ijk}
```
```math
\frac{\partial L}{\partial I_{abc}} = \sum_{pq} \left [\frac{\partial O_{pqk}}{\partial I_{abc}} \frac{\partial L}{\partial O_{pqk}}  \right ] = \sum_{pq} \left [  \frac{\partial L}{\partial O_{pqk}} \sum_{ij} W_{ijk} \right ]
```
Where, in this case, pq is the index of each neuron in the output layer *which connects to this specific input* and ij is the index of the specific weight *which connects that output to this input*. I'm still figuring out a better matemathical way to write this.

### Other Layers

Depthwise separable convolution is the same as 2D-convolution except the connections between channels are missing and the number of neurons in the output layer is different. The logic is the same, however.

The Add layer has a derivative of one and therefore doesn't affect the backpropagated error, only carries it.

### Correction Equations

Terminology: 
 - P is is a generic parameter of the network (3.4 million)
 - mb is the index representing mini-batches
 - t is the index representing epochs
 - Mu is the *momentum* and Rho is the *decay* hyperparameter of RMSProp (both equal to 0.9)
 - Eta is the *learning rate* (Initial = 0.045) and Delta is the *learning rate decay* (0.98 per epoch)
 - Epsilon is a constant for numerical stability (assuming Keras default = 1e-07)

RMSProp:
```math
E_{P, mb} = \mu E_{P, mb-1} + (1 - \rho) \cdot \left ( \frac{\partial L}{\partial P} \right )^2
```

Final correction equations:
```math
P_{mb} = P_{mb-1} - \frac{\eta_{0} \cdot \delta^{t}}{\sqrt{E_{t}}+\epsilon} \cdot \frac{1}{96} \sum_{96} \frac{\partial L}{\partial P}
```









