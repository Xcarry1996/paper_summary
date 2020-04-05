整理重要的文献专用~

# **BNN**
### Binary neural networks: A survey


*Haotong Qin , Ruihao Gong , Xianglong Liu , Xiao Bai , Jingkuan Song , Nicu Sebe  
Beihang University, University of Electronic Science and Technology of China, Chengdu, University of Trento, Italy*

**ABSTRACT**  
The binary neural network, largely saving the storage and computation, serves as a promising technique for deploying deep models on resource-limited devices. However, the binarization inevitably causes severe information loss, and even worse, its discontinuity brings difficulty to the optimization of the deep network. To address these issues, a variety of algorithms have been proposed, and achieved satisfying progress in recent years. _**In this paper, we present a comprehensive survey of these algorithms, mainly categorized into the native solutions directly conducting binarization, and the optimized ones using techniques like minimizing the quantization error, improving the network loss function, and reducing the gradient error. We also investigate other practical aspects of binary neural networks such as the hardware-friendly design and the training tricks. Then, we give the evaluation and discussions on different tasks, including image classification, object detection and semantic segmentation. Finally, the challenges that may be faced in future research are prospected.**_

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
