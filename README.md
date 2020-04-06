整理重要的文献专用~

```
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
