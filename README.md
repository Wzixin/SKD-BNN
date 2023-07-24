# SKD-BNN

This project is the PyTorch implementation of our paper : Self-Knowledge Distillation enhanced Binary Neural Networks using Underutilized Information.

**Datasets augmentation.** (1) CIFAR-10: During training, we employ normal data augmentation techniques, including padding four pixels on each side of the images, random cropping, and random horizontal flipping. For testing, we evaluate a single view of the original image. (2) ImageNet: For the training stage, we implement random cropping and random horizontal flipping as the augmentation methods. During testing, we exclusively employed a 224 × 224 center cropping approach for evaluation.

**Network structures.** We evaluate SKD-BNN with widely popular network structures, including ResNet-18, ResNet-20 and VGG-Small. Hardtanh is selected as the activation function. To ensure a fair comparison, we binarize all convolution layers except for the first and last layers of the networks consistently.

**Training Details.** Our SKD-BNN is trained from scratch without any pre-trained model, aligning with the typical one-stage methods. We follow the training scheme in IR-Net [1]. Stochastic gradient descent (SGD) with a momentum of 0.9, and the learning rate is adjusted using the cosine annealing scheduler. For CIFAR-10, the batch size is 128, and for ImageNet, it is 256. The temperature Tem is set to 4 for all experiments.

**Dependencies.**

- Python == 3.7
- Pytorch == 1.3.0
- GPU == Tesla V100

**Accuracy.** 

CIFAR-10:

|   Model   | Bit-Width (W/A) | Top-1 Acc. (%) |
| --------- | --------------- | ------------ |
| VGG-Small | 1 / 1           | 91.4         |
| ResNet-18 | 1 / 1           | 87.2         |
| ResNet-18 | 1 / 1           | 93.0         | 

ImageNet:

|   Model   | Bit-Width (W/A) | Top-1 Acc. (%) |
| --------- | --------------- | --------- |
| ResNet-18 | 1 / 1           | 59.7      |

**Reference.** 

[1] Haotong Qin, Ruihao Gong, Xianglong Liu, Mingzhu Shen,
Ziran Wei, Fengwei Yu, and Jingkuan Song. Forward and
backward information retention for accurate binary neural networks. In 2020 IEEE/CVF Conference on Computer Vision and Pattern Recognition, CVPR 2020, Seattle, WA, USA, June 13-19, 2020, pages 2247-2256, 2020.

## Citation

If you find our code useful for your research, please consider citing:

    @article{XXXXXX,
      title={Self-Knowledge Distillation enhanced Binary Neural Networks using Underutilized Information},
      DOI={XXXXXX},
      author={XXXXXX},
      journal={XXXXXX},
      year={XXXXXX},
      month={XXXXXX}
    }


//                                                                                                                                                   
//                                                                                                                                                   
//     SSSSSSSSSSSSSSS KKKKKKKKK    KKKKKKKDDDDDDDDDDDDD                         BBBBBBBBBBBBBBBBB   NNNNNNNN        NNNNNNNNNNNNNNNN        NNNNNNNN
//   SS:::::::::::::::SK:::::::K    K:::::KD::::::::::::DDD                      B::::::::::::::::B  N:::::::N       N::::::NN:::::::N       N::::::N
//  S:::::SSSSSS::::::SK:::::::K    K:::::KD:::::::::::::::DD                    B::::::BBBBBB:::::B N::::::::N      N::::::NN::::::::N      N::::::N
//  S:::::S     SSSSSSSK:::::::K   K::::::KDDD:::::DDDDD:::::D                   BB:::::B     B:::::BN:::::::::N     N::::::NN:::::::::N     N::::::N
//  S:::::S            KK::::::K  K:::::KKK  D:::::D    D:::::D                    B::::B     B:::::BN::::::::::N    N::::::NN::::::::::N    N::::::N
//  S:::::S              K:::::K K:::::K     D:::::D     D:::::D                   B::::B     B:::::BN:::::::::::N   N::::::NN:::::::::::N   N::::::N
//   S::::SSSS           K::::::K:::::K      D:::::D     D:::::D                   B::::BBBBBB:::::B N:::::::N::::N  N::::::NN:::::::N::::N  N::::::N
//    SS::::::SSSSS      K:::::::::::K       D:::::D     D:::::D ---------------   B:::::::::::::BB  N::::::N N::::N N::::::NN::::::N N::::N N::::::N
//      SSS::::::::SS    K:::::::::::K       D:::::D     D:::::D -:::::::::::::-   B::::BBBBBB:::::B N::::::N  N::::N:::::::NN::::::N  N::::N:::::::N
//         SSSSSS::::S   K::::::K:::::K      D:::::D     D:::::D ---------------   B::::B     B:::::BN::::::N   N:::::::::::NN::::::N   N:::::::::::N
//              S:::::S  K:::::K K:::::K     D:::::D     D:::::D                   B::::B     B:::::BN::::::N    N::::::::::NN::::::N    N::::::::::N
//              S:::::SKK::::::K  K:::::KKK  D:::::D    D:::::D                    B::::B     B:::::BN::::::N     N:::::::::NN::::::N     N:::::::::N
//  SSSSSSS     S:::::SK:::::::K   K::::::KDDD:::::DDDDD:::::D                   BB:::::BBBBBB::::::BN::::::N      N::::::::NN::::::N      N::::::::N
//  S::::::SSSSSS:::::SK:::::::K    K:::::KD:::::::::::::::DD                    B:::::::::::::::::B N::::::N       N:::::::NN::::::N       N:::::::N
//  S:::::::::::::::SS K:::::::K    K:::::KD::::::::::::DDD                      B::::::::::::::::B  N::::::N        N::::::NN::::::N        N::::::N
//   SSSSSSSSSSSSSSS   KKKKKKKKK    KKKKKKKDDDDDDDDDDDDD                         BBBBBBBBBBBBBBBBB   NNNNNNNN         NNNNNNNNNNNNNNN         NNNNNNN
//                                                                                                                                                   
//                                                                                                                                                   
//                                                                                                                                                   
//                                                                                                                                                   
//                                                                                                                                                   
//                                                                                                                                                   
//                                                                                                                                                   
