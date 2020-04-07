整理重要的文献专用~

```
## 二值网络
BinaryConnect(BC): 
1.仅在前向传播计算激活值和反向传播计算梯度2个操作对权值W二值化，权值更新时权值W不二值化
2.提出stochastic binarization（相对deterministic）

Binarized Neural Network(BNN):
1.从第一层到倒数第二层对权值W和激活值A都进行二值化，第一层的初始激活值为所有二值化权值和真实值输入的乘累加和
2.在激活值二值化之前batch normalization
3.数据集：MNIST，Cifar-10，SVHN

Xnor-Net(BWN，XNOR):
1.binary-weight-networks + xnor-net
2.，BWN只有权值二值化，xnor权值和激活值都二值化
3.数据集：ImageNet

## 量化
Differentiable Soft Quantization(DSQ):
1.提出一种介于全精度和量化中间的函数，BP可以计算梯度（通常意义的round()量化没法计算）
2.随着batch增加，函数逐渐接近的量化函数。设置观测参数α，加入loss函数计算最优解，获得更准确的量化函数
3.用DSQ做BNN的效果没差别，BP计算梯度不一样，用α作为参数，可以观察不同参数对量化的敏感度
4.重点是对量化过程做了研究，引入新的量化函数提升精度，github上没有代码
5.线性量化(uniform)

LQ-Net:
1.量化误差作为指标加入loss迭代学习，获得更高精度的量化区间
2.非线性量化

```


# **BNN**
### Binary neural networks: A survey


*Haotong Qin , Ruihao Gong , Xianglong Liu , Xiao Bai , Jingkuan Song , Nicu Sebe  
Beihang University, University of Electronic Science and Technology of China, Chengdu, University of Trento, Italy*

**ABSTRACT**  
The binary neural network, largely saving the storage and computation, serves as a promising technique for deploying deep models on resource-limited devices. However, the binarization inevitably causes severe information loss, and even worse, its discontinuity brings difficulty to the optimization of the deep network. To address these issues, a variety of algorithms have been proposed, and achieved satisfying progress in recent years. _**In this paper**_, we present a comprehensive survey of these algorithms, mainly categorized into the native solutions directly conducting binarization, and the optimized ones using techniques like minimizing the quantization error, improving the network loss function, and reducing the gradient error. We also investigate other practical aspects of binary neural networks such as the hardware-friendly design and the training tricks. Then, we give the evaluation and discussions on different tasks, including image classification, object detection and semantic segmentation. Finally, the challenges that may be faced in future research are prospected.

----
### BinaryConnect: Training Deep Neural Networks with binary weights during propagations  


*Matthieu Courbariaux , Yoshua Bengio , Jean-Pierre David  
Ecole Polytechnique de Montre ́al, Universite ́ de Montre ́al*  

**ABSTRACT**  

Deep Neural Networks (DNN) have achieved state-of-the-art results in a wide range of tasks, with the best results obtained with large training sets and large models. In the past, GPUs enabled these breakthroughs because of their greater computational speed. In the future, _**faster computation at both training and test time**_ is likely to be crucial for further progress and for consumer applications on low-power devices. As a result, there is much interest in research and development of dedicated hardware for Deep Learning (DL). _**Binary weights**_, i.e., _**weights which are constrained to only two possible values (e.g. -1 or 1)**_, would bring great benefits to specialized DL hardware by _**replacing many multiply-accumulate operations by simple accumulations**_, as _**multipliers are the most space and power-hungry components of the digital implementation of neural networks**_. We introduce BinaryConnect, a method which consists in training a DNN with binary weights during the forward and backward propagations, while retaining precision of the stored weights in which gradients are accumulated. Like other dropout schemes, we show that BinaryConnect acts as regularizer and we obtain near state-of-the-art results with BinaryConnect on the permutation-invariant _**MNIST, CIFAR-10 and SVHN**_.

----
### Binarized Neural Networks: Training Neural Networks with Weights and Activations Constrained to +1 or −1

*Matthieu Courbariaux , Itay Hubara , Daniel Soudry , Ran El-Yaniv , Yoshua Bengio  
Universite ́ de Montre ́al, Technion - Israel Institute of Technology, Columbia University, CIFAR Senior Fellow*

**ABSTRACT** 

We introduce a method to train _**Binarized Neural Networks (BNNs) neural networks with binary weights and activations**_ at run-time. At training-time the binary weights and activations are used for computing the parameters gradients. _**During the forward pass, BNNs drastically reduce memory size and accesses, and replace most arithmetic operations with bit-wise operations, which is expected to substantially improve power-efficiency**_. To validate the effectiveness of BNNs we conduct two sets of experiments on the Torch7 and Theano frameworks. On both, BNNs achieved nearly state-of-the-art results over the _**MNIST, CIFAR-10 and SVHN datasets**_. Last but not least, _**we wrote a binary matrix multiplication GPU kernel with which it is possible to run our MNIST BNN 7 times faster than with an unoptimized GPU kernel, without suffering any loss in classification accuracy**_. The code for training and running our BNNs is available on-line.


----
### XNOR-Net: ImageNet Classification Using Binary Convolutional Neural Networks  

*Mohammad Rastegari†, Vicente Ordonez†, Joseph Redmon∗, Ali Farhadi†∗  
Allen Institute for AI†, University of Washington∗*

**ABSTRACT** 

We propose two efficient approximations to standard convolutional neural networks: _**Binary-Weight-Networks**_ and _**XNOR-Networks**_. In Binary-Weight-Networks, the filters are approximated with binary values resulting in _**32× memory saving**_. In XNOR-Networks, both the filters and the input to convolutional layers are binary. XNOR-Networks approximate convolutions using primarily binary operations. This results in _**58× faster convolutional operations and 32× memory savings**_. XNOR-Nets offer the possibility of running state-of-the-art networks on CPUs (rather than GPUs) in real-time. Our binary networks are simple, accurate, efficient, and work on challenging visual tasks. We evaluate our approach on the ImageNet classification task. The classification accuracy with a Binary-Weight-Network version of _**AlexNet**_ is only _**2.9%**_ less than the full-precision AlexNet (in top-1 measure). We compare our method with recent network binarization methods, BinaryConnect and BinaryNets, and outperform these methods by large margins on ImageNet, more than 16% in top-1 accuracy.


## QUANTIZATION
### Differentiable Soft Quantization: Bridging Full-Precision and Low-Bit Neural Networks

*Ruihao Gong1,2 Xianglong Liu1∗ Shenghu Jiang1,2 Tianxiang Li2,3 Peng Hu2 Jiazhen Lin 2 Fengwei Yu2 Junjie Yan2  
1State Key Laboratory of Software Development Environment, Beihang University 2SenseTime Group Limited 3Beijing Institute of Technology*  

**ABSTRACT** 

Hardware-friendly network quantization (e.g., binary/uniform quantization) can efficiently accelerate the inference and meanwhile reduce memory consumption of the deep neural networks, which is crucial for model deployment on resource-limited devices like mobile phones. However, due to the discreteness of low-bit quantization, existing quantization methods often face the unstable training process and severe performance degradation. To address this problem, in this paper we propose Differentiable Soft Quantization (DSQ) to bridge the gap between the full-precision and low-bit networks. _**DSQ can automatically evolve during training to gradually approximate the standard quantization**_. Owing to its differentiable property, DSQ can help pursue the accurate gradients in backward propagation, and reduce the quantization loss in forward process with an appropriate clipping range. Extensive experiments over several popular network structures show that training low-bit neural networks with DSQ can consistently outperform state-of-the-art quantization methods. Besides, our first efficient implementation for deploying 2 to 4-bit DSQ on devices with ARM architecture achieves up to 1.7× speed up, compared with the open-source 8-bit high-performance inference framework NCNN.


----
### LQ-Nets: Learned Quantization for Highly Accurate and Compact Deep Neural Networks

*Dongqing Zhang∗, Jiaolong Yang∗, Dongqiangzi Ye∗, and Gang Hua  
Microsoft Research*

**ABSTRACT**  

Although weight and activation quantization is an effective approach for Deep Neural Network (DNN) compression and has a lot of potentials to increase inference speed leveraging bit-operations, there is still a noticeable gap in terms of prediction accuracy between the quantized model and the full-precision model. To address this gap, we propose to _**jointly train**_ a quantized, bit-operation-compatible DNN and its associated quantizers, as opposed to using fixed, handcrafted quantization schemes such as uniform or logarithmic quantization. Our method for learning the quantizers applies to both network weights and activations with arbitrary-bit precision, and _**our quantizers are easy to train**_. The comprehensive experiments on CIFAR-10 and ImageNet datasets show that our method works consistently well for various network structures such as AlexNet, VGG-Net, GoogLeNet, ResNet, and DenseNet, surpassing previous quantization methods in terms of accuracy by an appreciable margin. Code available at https://github.com/Microsoft/LQ-Nets
