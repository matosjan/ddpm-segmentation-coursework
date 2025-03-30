
# Semantic Segmentation using Diffusion Models

This repository is a fork of the [original repository](https://github.com/yandex-research/ddpm-segmentation?tab=readme-ov-file), of the method described in [Label-Efficient Semantic Segmentation with Diffusion Models](https://arxiv.org/abs/2112.03126), enhanced with modifications based on research conducted as part of my coursework at the Faculty of Computer Science, Higher School of Economics.

## Overview

Semantic segmentation is a fundamental task in computer vision, where each pixel of an image is classified into a specific semantic class. Recent advancements demonstrate that generative diffusion models, specifically Denoising Diffusion Probabilistic Models (DDPM), effectively capture semantic information, which can significantly improve segmentation tasks, particularly in label-scarce scenarios.

This project explored DDPM-based segmentation methods, identified weaknesses in the existing state-of-the-art method, and proposed modifications to enhance its performance.

## Conducted Work

-   **Theoretical Study**: Studied foundational papers on diffusion models to build a comprehensive understanding of their principles and applications.
    
-   **Analysis of Baseline Method**: Studied the original method leveraging DDPMs for semantic segmentation, where pixel representations were extracted from intermediate activations of a pretrained DDPM (U-Net backbone).
    
-   **Identification of Weaknesses**: Discovered significant issues including background noise, pixel misclassification, and limited robustness.
    
-   **Hypothesis Testing and Improvements**:
    
    -   **Data Augmentation**: Applied pixel-level and geometric augmentations during classifier training to increase model generalization and reduce overfitting.
        
    -   **Regularization Techniques**: Integrated AdamW optimizer and Dropout layers into the classifier ensemble to further mitigate overfitting.
        
    -   **Test-Time Augmentation**: Implemented ensemble predictions on augmented test inputs to stabilize segmentation masks and reduce background noise.
        

## Results

The implemented modifications consistently improved segmentation performance across multiple datasets. For comparison, baseline metrics were independently reproduced to ensure reliable and fair evaluation:
| Dataset    | Baseline mIoU | Modified mIoU |
|------------|---------------|---------------|
| Horse-21   | 64.03%        | 65.33%        |
| FFHQ-34    | 57.91%        | 58.12%        |
| Cat-15     | 57.53%        | 59.54%        |
| Bedroom-28 | 50.05%        | 52.00%        |

## Conclusion

The modifications introduced in this project significantly enhanced the existing DDPM-based semantic segmentation method, improving its robustness and overall accuracy.

## Full Report

The complete coursework report is available in this [PDF document](https://github.com/matosjan/ddpm-segmentation-coursework/blob/main/report.pdf)
    
## References

-   [Label-Efficient Semantic Segmentation with Diffusion Models](https://arxiv.org/abs/2112.03126)
    
