# Frequency-Conditioned Prompting for Test-Time Adaptation


This project implements Frequency based Visual Prompt TTA, a lightweight Test-Time Adaptation (TTA) method that operates in the frequency domain. Instead of updating model parameters (which can be unstable), it learns input-dependent visual prompts in the Fourier domain to adapt images to new distributions at test-time.

---

## Overview

Test-Time Adaptation (TTA) focuses on improving model performance under distribution shifts without access to source data. this project introduces a frequency-conditioned prompting method that:
1.  **Extracts Radial Features**: Uses a `RadialFrequencyExtractor` to compute mean energy across concentric radial bins in the Fourier domain.
2.  **Generates Prompts**: Employs a lightweight MLP to generate input-specific prompts based on these frequency signatures.
3.  **Adapts Amplitude**: Modifies the amplitude of the image's FFT while preserving the phase, which captures critical structural information.
4.  **Minimizes Entropy**: Optimizes the prompt using entropy minimization at test-time while keeping the backbone model frozen.
 ---

## Key Results

Based on our evaluation on **CIFAR-10-C** and **PACS** datasets:

| Dataset | Findings |
| :--- | :--- |
| **CIFAR-10-C** | Highly effective for corruptions like **Shot Noise** and **Blur**. The method successfully identifies and suppresses high-frequency artifacts. |
| **PACS** | Shows stability but limited improvement/degradation on complex domain shifts (e.g., Sketch, Cartoon). Suggests frequency-only adaptation may need augmentation for semantic shifts. |

> **Observation**: The method mainly modifies high-frequency components corresponding to noise, as seen in the qualitative difference maps in the report.

---




## References

1. **VPT**: Visual Prompt Tuning (ECCV 2022)
2. **Tent**: Fully Test-Time Adaptation by Entropy Minimization (ICLR 2021)
3. **VPTTA**: Prompt-based Test-Time Adaptation (arXiv 2023)

---

## Contact
by [vhemanth06](https://github.com/vhemanth06) and [MohammedAbdurRehman11015](https://github.com/MohammedAbdurRehman11015).
