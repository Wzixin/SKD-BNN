# SKD-BNN

This project is the PyTorch implementation of our paper : Self-Knowledge Distillation enhanced Binary Neural Networks using Underutilized Information.

## The core of our code is located in /SKD_BNN.py. As of now, our paper has not been accepted yet, and this part is waiting to be uploaded.

**Datasets augmentation:** (1) CIFAR-10: During training, we employ normal data augmentation techniques, including padding four pixels on each side of the images, random cropping, and random horizontal flipping. For testing, we evaluate a single view of the original image. (2) ImageNet: For the training stage, we implement random cropping and random horizontal flipping as the augmentation methods. During testing, we exclusively employed a 224 × 224 center cropping approach for evaluation.

**Network structures:** We evaluate SKD-BNN with widely popular network structures, including ResNet-18, ResNet-20 and VGG-Small. Hardtanh is selected as the activation function. To ensure a fair comparison, we binarize all convolution layers except for the first and last layers of the networks consistently.

**Training Details:** Our SKD-BNN is trained from scratch without any pre-trained model, aligning with the typical one-stage methods. We follow the training scheme in IR-Net [1]. Stochastic gradient descent (SGD) with a momentum of 0.9, and the learning rate is adjusted using the cosine annealing scheduler. For CIFAR-10, the batch size is 128, and for ImageNet, it is 256. The temperature Tem is set to 4 for all experiments.

**Dependencies:**

- Ubuntu == 18.04
- GPU == NVIDIA V100
- GPU Driver == 460.106.00
- CUDA == 11.2.2
- cuDNN == 8.2.1
- Python == 3.8
- Pytorch == 1.9.1
- Torchvision == 0.10.0

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

[1] Haotong Qin, Ruihao Gong, Xianglong Liu, Mingzhu Shen,
Ziran Wei, Fengwei Yu, and Jingkuan Song. Forward and
backward information retention for accurate binary neural networks. In 2020 IEEE/CVF Conference on Computer Vision and Pattern Recognition, CVPR 2020, Seattle, WA, USA, June 13-19, 2020, pages 2247-2256, 2020.

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
                                                                                        
                                                                                       
