# SKD-BNN

This project is the PyTorch implementation of our paper : Self-Knowledge Distillation enhanced Binary Neural Networks using Underutilized Information.

**Datasets augmentation:** 
(1) MNIST: This dataset comprises a training set of 60,000 samples and a test set of 10,000 28 × 28 grayscale images representing digits from 0 to 9. To preserve the challenge level of the benchmark, convolutions, data augmentation, preprocessing, or other operations are not applied.
(2) CIFAR-10: The CIFAR-10 dataset consists of 60,000 RGB images with sizes of 32 × 32, covering 10 classes: airplanes, automobiles, birds, cats, deer, dogs, frogs, horses, ships, and trucks. The dataset contains 50,000 images for training and 10,000 images for testing. During the training process, normal data augmentation techniques, including padding four pixels on each side of the images, random cropping, and random horizontal flipping, are employed. For testing, a single view of each original image is evaluated.
(3) ImageNet: The ImageNet dataset contains approximately 1.2 million training images and 50,000 validation images distributed across 1000 categories. For the training stage, random cropping and resizing to 224 × 224, along with random horizontal flipping, are applied as the augmentation methods. During testing, a 224 × 224 center cropping approach is exclusively employed for evaluation purposes.

**Network structures:**
On the MNIST dataset, to ensure fair comparisons, the same multilayer perceptron (MLP) network architecture as that described [1] is adopted for the CNN; this architecture consists of 3 hidden layers with 2048 binary units. On CIFAR-10 and ImageNet, SKD-BNN is evaluated in comparison with widely popular network structures, including ResNet-18, ResNet-20 and VGG-Small. To ensure a fair comparison, all the convolution layers except for the first and last layers of the networks are consistently binarized. For ResNet-20, the double-skip connections proposed in Bi-Real are adopted [2].

**Training Details:**
The proposed SKD-BNN approach is trained from scratch without a pretrained model and is aligned with the typical one-stage methods. On the MNIST dataset, regularization is applied to the model via dropout. The square hinge loss is minimized using the Adam adaptive learning rate method, consistent with those in BNN [1]. On CIFAR-10 and ImageNet, the training scheme of IR-Net is followed [3]. HardTanh is selected as the activation function. The stochastic gradient descent (SGD) algorithm has a momentum parameter of 0.9, and the learning rate is adjusted using the cosine annealing scheduler. For CIFAR-10, the batch size is 128, and for MNIST and ImageNet, it is 256. The temperature T em is set to 4.0 for all the experiments.

**Dependencies:**

- Pytorch==2.1.1
- Torchvision == 0.16.1
- pytorch-cuda=12.1

**Accuracy:** 

CIFAR-10:
|   Model   | Bit-Width (W/A) | Top-1 Acc. (%) |
| --------- | --------------- | ------------ |
| VGG-Small | 1 / 1           | 91.4         |
| ResNet-20 | 1 / 1           | 87.2         |
| ResNet-18 | 1 / 1           | 93.0         | 

ImageNet:
|   Model   | Bit-Width (W/A) | Top-1 Acc. (%) |
| --------- | --------------- | ------------ |
| ResNet-18 | 1 / 1           | 59.7         |

**Reference:** 
[1] Courbariaux M, Hubara I, Soudry D, et al (2016) Binarized neural networks: Training deep neural networks with weights and activations constrained to +1 or -1. arXiv preprint arXiv:1602.02830

[2] Liu Z, Wu B, Luo W, et al (2018) Bi-real net: Enhancing the performance of 1-bit cnns with improved representational capability and advanced training algorithm. In: Proceedings of the European Conference on Computer Vision (ECCV), pp 747–763, https://doi.org/10.1007/978-3-030-01267-0_44

[3] Qin H, Gong R, Liu X, et al (2020) Forward and backward information retention for accurate binary neural networks. In: Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pp 2247–2256, https://doi.org/10.1109/CVPR42600.2020.00232

<!-- ## Citation

If you find our code useful for your research, please consider citing:

    @article{XXXXXX,
      title={Self-Knowledge Distillation enhanced Binary Neural Networks using Underutilized Information},
      DOI={XXXXXX},
      author={XXXXXX},
      journal={XXXXXX},
      year={XXXXXX},
      month={XXXXXX}
    } -->

## Wzixin

                                                                                        
                       ,--.                                           ,--.         ,--. 
      .--.--.      ,--/  /|    ,---,                  ,---,.        ,--.'|       ,--.'| 
     /  /    '. ,---,': / '  .'  .' `\              ,'  .'  \   ,--,:  : |   ,--,:  : | 
    |  :  /`. / :   : '/ / ,---.'     \     ,---,.,---.' .' |,`--.'`|  ' :,`--.'`|  ' : 
    ;  |  |--`  |   '   ,  |   |  .`\  |  ,'  .' ||   |  |: ||   :  :  | ||   :  :  | | 
    |  :  ;_    '   |  /   :   : |  '  |,---.'   ,:   :  :  /:   |   \ | ::   |   \ | : 
     \  \    `. |   ;  ;   |   ' '  ;  :|   |    |:   |    ; |   : '  '; ||   : '  '; | 
      `----.   \:   '   \  '   | ;  .  |:   :  .' |   :     \'   ' ;.    ;'   ' ;.    ; 
      __ \  \  ||   |    ' |   | :  |  ':   |.'   |   |   . ||   | | \   ||   | | \   | 
     /  /`--'  /'   : |.  \'   : | /  ; `---'     '   :  '; |'   : |  ; .''   : |  ; .' 
    '--'.     / |   | '_\.'|   | '` ,/            |   |  | ; |   | '`--'  |   | '`--'   
      `--'---'  '   : |    ;   :  .'              |   :   /  '   : |      '   : |       
                ;   |,'    |   ,.'                |   | ,'   ;   |.'      ;   |.'       
                '---'      '---'                  `----'     '---'        '---'         
                                                                                        
                                                                                       
