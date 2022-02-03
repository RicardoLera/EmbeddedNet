# EmbeddedNet

EmbeddedNet is a Convolutional Neural Network written in C capable of **Inference**, **Fine-Tuning** and **Transfer Learning**. It is based on the [MobileNetV2 Architecture](https://arxiv.org/abs/1801.04381) and utilizes Stochastic Gradient Descent techniques as well as Layer Freezing to achieve a more efficient training process in embedded systems. For that purpose, it also exclusively utilizes static libraries.

A Keras application of MobileNetV2 is used for pre-training as well as development and testing.

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
 - Implemented training in Python for comparison with C implementation.
 - Formulated training equations.
 - Adapted system to Stochastic Gradient Descent.
 - Implemented training for Fully Connected layer and observed a slight increase in precision with a Learning Rate of 0.000045.
 - Repeated the same process for Average Pooling, ReLU6, Batch Normalization, 2D-Convolution and Depthwise Convolution.
 - Training Finished
 - Implemented Layer Freezing
 - Profiled perfomance with Gprof - Bottleneck is at convolution functions: Managed to improve backprop_conv2d at the cost of memory.
 - Profiled memory usage with Massif - Peak memory at 5,211,736B due to backprop_fc's gradient allocation.
 
## Training Equations

From the [MobileNetV2 Article](https://arxiv.org/abs/1801.04381):
>We train our models using TensorFlow. We use the standard RMSPropOptimizer with both decay and momentum set to 0.9. We use batch normalization after every layer, and the standard weight decay is set to 0.00004. Following MobileNetV1 setup we use initial learning rate of 0.045, and learning rate decay rate of 0.98 per epoch. We use 16 GPU asynchronous workers, and a batch size of 96.

A "batch size of 96" means this is not stochastic gradient descent, but mini-batch gradient descent. Hence, 96 images are randomly selected from the training set, each image is passed through the network and their respective gradients are averaged, and then the parameters are updated using the mean gradient. This process is repeated until the dataset ends, which constitutes one epoch.

We will not be using this process, as it is more efficient for training the network from scratch, which would be unfeasible in an embedded system. Instead, the fine-tuning process will utilize stochastic gradient descent with a much smaller dataset, Layer Freezing, as well as a much lower value of the Learning Rate. RMSPropOptimizer function is still implemented.

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
<p align="center">
<img src="https://render.githubusercontent.com/render/math?math={\displaystyle \color{gray}\   L = -\sum_{j}[T_{j}ln(O_{j})] + \frac{\lambda W_{ij}^{2}}{2} \rightarrow \text{one-hot} \rightarrow -ln(O_{c}) + \frac{\lambda W_{ij}^{2}}{2}   }">
</p>

Softmax:
<p align="center">
<img src="https://render.githubusercontent.com/render/math?math={\displaystyle \color{gray}\   Softmax(Z_{j}) = \frac{e^{Z_{j}}}{\sum_{k}e^{Z_{k}}} = \frac{e^{Z_{j}}}{S}   }">
</p>

Activation:
<p align="center">
<img src="https://render.githubusercontent.com/render/math?math={\displaystyle \color{gray}\   Z_{j} = \sum_{i}(I_{i} \cdot W_{ij}) + B_{j}   }">
</p>

Weight gradient:
<p align="center">
<img src="https://render.githubusercontent.com/render/math?math={\displaystyle \color{gray}\   \frac{\partial L}{\partial W_{ij}} = -\frac{\partial Z_{j}}{\partial W_{ij}} \cdot \frac{\partial L}{\partial Z_{j}} + \frac{\partial \left [\frac{\lambda W_{ij}^2}{2}  \right ]}{\partial W_{ij}} = -\frac{\partial Z_{j}}{\partial W_{ij}} \cdot \frac{\partial L}{\partial Z_{j}} + \lambda W_{ij}  }">
</p>
<p align="center">
<img src="https://render.githubusercontent.com/render/math?math={\displaystyle \color{gray}\   \frac{\partial Z_{j}}{\partial W_{ij}} = \frac{\partial [\sum_{p}(I_{p}W_{pj}+B_{j}) ] }{\partial W_{ij}} =  \sum_{p}\left [ I_{p} \cdot \frac{\partial W_{pj}}{\partial W_{ij}} \right ] = \sum_{p}\left [ I_{p} \delta_{ip} \right ] = I_{i}   }">
</p>
<p align="center">
<img src="https://render.githubusercontent.com/render/math?math={\displaystyle \color{gray}\   \frac{\partial L}{\partial Z_{j}} = \frac{\partial [ln(\frac{e^{Z_{c}}}{S})]}{\partial Z_{j}} = \frac{\partial \left [Z_{c} - ln(S)  \right ]}{\partial Z_{j}} = \delta_{jc} - \frac{1}{S} \cdot \frac{\partial S}{\partial Z_{j}} = \delta_{jc} - \frac{1}{S} \cdot \frac{\partial [\sum_{k}(e^{z_{k}})]}{\partial Z_{j}} = \delta_{jc} - \frac{1}{S} \cdot \sum_{k}[\delta_{jk}e^{Z_{k}}] = \delta_{jc} - \frac{e^{Z_{j}}}{S} = \delta_{jc} - O_{j}   }">
</p>
<p align="center">
<img src="https://render.githubusercontent.com/render/math?math={\displaystyle \color{gray}\   \frac{\partial L}{W_{ij}} = -I_{i} (\delta_{jc} - O_{j}) + \lambda W_{ij} = I_{i} (O_{j} -\delta_{jc}) + \lambda W_{ij}   }">
</p>

Bias gradient (no standard weight decay):
<p align="center">
<img src="https://render.githubusercontent.com/render/math?math={\displaystyle \color{gray}\   \frac{\partial L}{\partial B_{j}} = -\frac{\partial Z_{j}}{\partial B_{j}} \cdot \frac{\partial L}{\partial Z_{j}}   }">
</p>
<p align="center">
<img src="https://render.githubusercontent.com/render/math?math={\displaystyle \color{gray}\   \frac{\partial Z_{j}}{\partial B_{j}} = \frac{\partial [ \sum_{p} ( I_{p} \cdot W_{pj} ) + B_{j}  ] }{\partial B_{j}} = 1   }">
</p>
<p align="center">
<img src="https://render.githubusercontent.com/render/math?math={\displaystyle \color{gray}\   \frac{\partial L}{B_{j}} = -(\delta_{jc} - O_{j}) = O_{j} - \delta_{jc}   }">
</p>

Backpropagated Error:
<p align="center">
<img src="https://render.githubusercontent.com/render/math?math={\displaystyle \color{gray}\   \frac{\partial L}{\partial I_{i}} = -\sum_{j} \left [\frac{\partial Z_{j}}{\partial I_{i}} \cdot \frac{\partial L}{\partial Z_{j}} \right ]   }">
</p>
<p align="center">
<img src="https://render.githubusercontent.com/render/math?math={\displaystyle \color{gray}\   \frac{\partial Z_{j}}{\partial I_{i}} = \frac{\partial [ \sum_{p} ( I_{p} \cdot W_{pj} ) + B_{j}  ] }{\partial I_{i}} = W_{ij}   }">
</p>
<p align="center">
<img src="https://render.githubusercontent.com/render/math?math={\displaystyle \color{gray}\   \frac{\partial L}{\partial I_{i}} = - \sum_{j} \left [W_{ij}(\delta_{jc} - O_{j})\right ]  = \sum_{j} \left [W_{ij}(O_{j} - \delta_{jc})  \right ]   }">
</p>

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
<p align="center">
<img src="https://render.githubusercontent.com/render/math?math={\displaystyle \color{gray}\   A(I_{k}) = \frac{1}{49} \sum_{i}\left [\sum_{j}(I_{ijk})  \right ]   }">
</p>

Differential:
<p align="center">
<img src="https://render.githubusercontent.com/render/math?math={\displaystyle \color{gray}\   \frac{\partial A(O_{k})}{\partial I_{pqr}} = \frac{1}{49} \sum_{i}\left [\sum_{j}\left ( \frac{\partial I_{ijk}}{\partial I_{pqr}} \right )  \right ] = \frac{1}{49} \sum_{i}\left [\sum_{j}\left ( \delta_{ip} \delta_{jq} \delta_{kr} \right )  \right ] = \frac{\delta_{kr}}{49}   }">
</p>

This means that for each output value, its error is multiplied by 1/49 and assigned to each value of the corresponding 7x7 block.

### ReLU6

Error backpropagates through the derivative of this function.

Terminology:
 - I is the value of an input to the ReLU6 layer
 - O is the value of an output to the ReLU6 layer
 - i, j and k are the dimentions of the input and output

ReLU6 equation:
<p align="center">
<img src="https://render.githubusercontent.com/render/math?math={\displaystyle \color{gray}\    ReLU6(I_{ijk}) = \begin{cases} 0 \qquad \text{ if } \ I_{ijk} < 0 \\ I_{ijk} \quad  \text{ if } \ 0 < I_{ijk} < 6 \\ 6 \qquad \text{ if } \ I_{ijk} > 6 \end{cases}   }">
</p>

Differential:
<p align="center">
<img src="https://render.githubusercontent.com/render/math?math={\displaystyle \color{gray}\    ReLU6'(I_{ijk}) = \begin{cases}0 \qquad \text{ if } \ I_{ijk} < 0 \\ 1 \qquad \text{ if } \ 0 < I_{ijk} < 6 \\ 0 \qquad \text{ if } \ I_{ijk} > 6 \end{cases}   }">
</p>

This means that for each output value, its error is multiplied by 0 when it is below 0 or above 6 and multiplied by 1 when it is in between these two values.

### Batch Normalization

During pre-training, the mean and variance were calculated based on the mini-batch. Those values might still be used in frozen layers. During fine-tuning, the mean and standard variation are calculated based on only the image in question.

Here, the trainable parameters are gamma and beta, and the error also backpropagates.

Terminology:
 - I and O are the the values of an input and output to the BN layer respectivelly
 - i, j and k are the indexes of the dimensions of the input and output
 - Mu is the moving mean and Sigma squared is the moving variance
 - Gamma and Beta are the trainable parameters "scale" and "shift" respectivelly
 - Rho is the momentum of the moving equations (= 0.999)
 - Epsilon is a small constant for numerical stability (= 0.001)
 - T is the total number of neurons across dimensions i and j
 - t is the index representing epochs

Batch Normalization Moving Equations:

<p align="center">
<img src="https://render.githubusercontent.com/render/math?math={\displaystyle \color{gray}\   \mu = \frac{\sum_{ij}I_{ij}}{T}  }">
</p>
<p align="center">
<img src="https://render.githubusercontent.com/render/math?math={\displaystyle \color{gray}\   \sigma^2 = \frac{\sum_{ij}(I_{ij} - \mu_{t})^2}{T}  }">
</p>
<p align="center">
<img src="https://render.githubusercontent.com/render/math?math={\displaystyle \color{gray}\   \mu_{t} = \mu_{t-1} \cdot \rho + (1 - \rho) \frac{\sum_{ij}I_{ij}}{T}  }">
</p>
<p align="center">
<img src="https://render.githubusercontent.com/render/math?math={\displaystyle \color{gray}\   \sigma_{t}^{2} = \sigma_{t-1}^{2} \cdot \rho + (1 - \rho) \frac{\sum_{ij}(I_{ij} - \mu_{t})^2}{T}  }">
</p>

Output Equation:
<p align="center">
<img src="https://render.githubusercontent.com/render/math?math={\displaystyle \color{gray}\   \hat{I}_{ij} = \frac{I_{ij} - \mu}{\sqrt{\sigma^2+\epsilon}}   }">
</p>

Gradient of Beta:
<p align="center">
<img src="https://render.githubusercontent.com/render/math?math={\displaystyle \color{gray}\   \frac{\partial L}{\partial \beta} = \sum_{ij} \left [\frac{\partial L}{\partial O_{ij}} \cdot \frac{\partial O_{ij}}{\partial \beta} \right ] = \sum_{ij} \left [\frac{\partial L}{\partial O_{ij}} \right ]   }">
</p>

Gradient of Gamma:
<p align="center">
<img src="https://render.githubusercontent.com/render/math?math={\displaystyle \color{gray}\   \frac{\partial L}{\partial \gamma_{k}} = \sum_{ij} \left [\frac{\partial L}{\partial O_{ij}} \cdot \frac{\partial O_{ij}}{\partial \gamma} \right ] = \sum_{ij} \left [\frac{\partial L}{\partial O_{ij}} \cdot \hat{I}_{ij} \right ]   }">
</p>

Backpropagating Error:
<p align="center">
<img src="https://render.githubusercontent.com/render/math?math={\displaystyle \color{gray}\   \frac{\partial L}{\partial I_{ij}} = \frac{\partial L}{\partial \hat{I}_{ij}} \cdot \frac{\partial \hat{I}_{ij}}{\partial I_{ij}} + \frac{\partial L}{\partial \sigma^2} \cdot \frac{\partial \sigma^2}{\partial I_{ij}} + \frac{\partial L}{\partial \mu} \cdot \frac{\partial \mu}{\partial I_{ij}}   }">
</p>
<p align="center">
<img src="https://render.githubusercontent.com/render/math?math={\displaystyle \color{gray}\   \frac{\partial L}{\partial \hat{I}_{ij}} = \frac{\partial L}{\partial O_{ij}} \cdot \frac{\partial O_{ij}}{\partial \hat{I}_{ij}} = \frac{\partial L}{\partial O_{ij}} \gamma   }">
</p>
<p align="center">
<img src="https://render.githubusercontent.com/render/math?math={\displaystyle \color{gray}\   \frac{\partial L}{\partial \sigma^2} = \sum_{ij} \left [\frac{\partial L}{\partial \hat{I}_{ij}} \cdot \frac{\partial \hat{I}_{ij}}{\partial \sigma^2}  \right ] = \sum_{ij} \left [\frac{\partial L}{\partial \hat{I}_{ij}} (I_{ij} - \mu) \frac{-1}{2} (\sigma^2 + \epsilon)^{-3/2}  \right ]   }">
</p>
<p align="center">
<img src="https://render.githubusercontent.com/render/math?math={\displaystyle \color{gray}\   \frac{\partial L}{\partial \mu} = \sum_{ij} \left [\frac{\partial L}{\partial \hat{I}_{ij}} \cdot \frac{\partial \hat{I}_{ij}}{\partial \mu} \right ] + \frac{\partial L}{\partial \sigma^2} \cdot \frac{\partial \sigma^2}{\partial \mu} = \sum_{ij} \left [\frac{\partial L}{\partial \hat{I}_{ij}} \cdot \frac{-1}{\sqrt{\sigma^2 + \epsilon}}  \right ] + \frac{\partial L}{\partial \sigma^2} \cdot \frac{\sum_{ij}[-2(I_{ij}-\mu)]}{d^2} = \sum_{ij} \left [\frac{\partial L}{\partial \hat{I}_{ij}} \cdot \frac{-1}{\sqrt{\sigma^2 + \epsilon}}  \right ]   }">
</p>
<p align="center">
<img src="https://render.githubusercontent.com/render/math?math={\displaystyle \color{gray}\   \frac{\partial \hat{I}_{ij}}{\partial I_{ij}} = \frac{1}{\sqrt{\sigma^2 + \epsilon}} \qquad   \frac{\partial \sigma^2}{\partial I_{ij}} = \frac{2(I_{ij}-\mu)}{d^2} \qquad \frac{\partial \mu}{\partial I_{ij}} = \frac{1}{d^2}   }">
</p>
<p align="center">
<img src="https://render.githubusercontent.com/render/math?math={\displaystyle \color{gray}\   \frac{\partial L}{\partial I_{ij}} = \frac{\partial L}{\partial \hat{I}_{ij}} \frac{1}{\sqrt{\sigma^2 + \epsilon}} + \frac{\partial L}{\partial \sigma^2} \frac{2(I_{ij}-\mu)}{d^2} + \frac{\partial L}{\partial \mu} \frac{1}{d^2}   }">
</p>

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
<p align="center">
<img src="https://render.githubusercontent.com/render/math?math={\displaystyle \color{gray}\   \frac{\partial L}{\partial W_{ijk}} = \sum_{pq} \left [\frac{\partial O_{pqk}}{\partial W_{ijk}} \frac{\partial L}{\partial O_{pqk}}  \right ]   }">
</p>
<p align="center">
<img src="https://render.githubusercontent.com/render/math?math={\displaystyle \color{gray}\   a = s \cdot p + i - pad   }">
</p>
<p align="center">
<img src="https://render.githubusercontent.com/render/math?math={\displaystyle \color{gray}\   b = s \cdot q + j - pad   }">
</p>
<p align="center">
<img src="https://render.githubusercontent.com/render/math?math={\displaystyle \color{gray}\   O_{pqk} = \sum_{ijc} \left [ I_{abc} \cdot W_{ijk} \right ]   }">
</p>
<p align="center">
<img src="https://render.githubusercontent.com/render/math?math={\displaystyle \color{gray}\   \frac{\partial O_{pqk}}{\partial W_{xyk}} = \frac{\partial \left [\sum_{ijc}\left ( I_{abc} \cdot W_{ijk} \right )  \right ]}{\partial W_{xyk}} = \sum_{ijc} \left ( I_{abc} \delta_{ix} \delta_{jy} \right ) \Rightarrow \frac{\partial O_{pqk}}{\partial W_{ijk}} = \sum_{ijc} I_{s \cdot p + i - pad, s \cdot q + j - pad, c}   }">
</p>

Final correction formula:
<p align="center">
<img src="https://render.githubusercontent.com/render/math?math={\displaystyle \color{gray}\   \frac{\partial L}{\partial W_{ijk}} = \sum_{pq} \left [\frac{\partial O_{pqk}}{\partial W_{ijk}} \frac{\partial L}{\partial O_{pqk}}  \right ] = \sum_{pq} \left [  \frac{\partial L}{\partial O_{pqk}} \sum_{ijc} I_{abc} \right ]   }">
</p>

Therefore, to get the gradient of a specific weight: for each output, you multiply the backpropagated error of that neuron by the input that this weight connects to (which depends on stride), and sum everything up (this doesn't reflect very well in the math).

Backpropagated Error:
<p align="center">
<img src="https://render.githubusercontent.com/render/math?math={\displaystyle \color{gray}\   \frac{\partial L}{\partial I_{abc}} = \sum_{pq} \left [\frac{\partial O_{pqk}}{\partial I_{abc}} \frac{\partial L}{\partial O_{pqk}}  \right ]   }">
</p>
<p align="center">
<img src="https://render.githubusercontent.com/render/math?math={\displaystyle \color{gray}\   \frac{\partial O_{pqk}}{\partial I_{xyz}} = \frac{\partial \left [\sum_{ijc}\left ( I_{abc} \cdot W_{ijk} \right )  \right ]}{\partial I_{xyz}} = \sum_{ijc} \left ( W_{ijk} \delta_{ax} \delta_{by} \delta_{cz} \right ) \Rightarrow \frac{\partial O_{pqk}}{\partial I_{abc}} = \sum_{ij} W_{ijk}   }">
</p>
<p align="center">
<img src="https://render.githubusercontent.com/render/math?math={\displaystyle \color{gray}\   \frac{\partial L}{\partial I_{abc}} = \sum_{pq} \left [\frac{\partial O_{pqk}}{\partial I_{abc}} \frac{\partial L}{\partial O_{pqk}}  \right ] = \sum_{pq} \left [  \frac{\partial L}{\partial O_{pqk}} \sum_{ij} W_{ijk} \right ]   }">
</p>

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
<p align="center">
<img src="https://render.githubusercontent.com/render/math?math={\displaystyle \color{gray}\   E_{P, t} = \mu E_{P, t-1} + (1 - \rho) \cdot \left ( \frac{\partial L}{\partial P} \right )^2   }">
</p>

Final correction equations:
<p align="center">
<img src="https://render.githubusercontent.com/render/math?math={\displaystyle \color{gray}\   P_{t} = P_{t-1} - \frac{\eta_{0} \cdot \delta^{t}}{\sqrt{E_{t}}+\epsilon} \cdot \frac{\partial L}{\partial P}   }">
</p>
