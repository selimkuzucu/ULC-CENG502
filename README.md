# Uncertainty-Aware Learning Against Label Noise on Imbalanced Datasets

This readme file is an outcome of the [CENG502 (Spring 2023)](https://ceng.metu.edu.tr/~skalkan/ADL/) project for reproducing a paper without an implementation. See [CENG502 (Spring 20223) Project List](https://github.com/CENG502-Projects/CENG502-Spring2023) for a complete list of all paper reproduction projects.

# 1. Introduction

Authored by Yingsong Huang, Bing Bai, Shengwei Zhao, Kun Bai and Fei Wang, "Uncertainty-Aware Learning Against Label Noise on Imbalanced Datasets" was published in AAAI 2022. The goal of the paper is to come up with a novel method to overcome the drawbacks of noise and class imbalance, especially when they co-exist. 

My goal with this repository is to provide a code for the presented novel technique and reproduce some of the results.

## 1.1. Paper summary

Since a significant portion of the real-life applications involve dealing with noisy labels,learning against label noise is a common challenge in tne machine learning literature. Recent literature aims to address this problem through considering the output probabilities of the models and/or the loss values to separate clean and noisy samples with the aim of treating them differently.

However, these techniques fail to fully address some of the most important real-life cases especially in the presence of class imbalance, even when the imbalance rate is in the range of 1:5.

The authors first conjecture that this might be due to the fact that the existing literature fails to consider the predictive uncertainty of the model's while solely focusing on the output probabilities.

Furthermore, the authors claim that these existing class-agnostic approaches are not working well on imbalanced datasets as inter-class loss distribution varies significantly across classes in imbalanced datasets.

Motivated by these shortcomings, the authors propose a novel framework, "Uncertainty-aware Label Correction (ULC)", to address these gaps.

# 2. The method and my interpretation

## 2.1. The original method

<p align="center">

![Screenshot 2023-06-18 at 21 32 54](https://github.com/selimkuzucu/ULC-CENG502/assets/56355561/5c838db7-b2fd-44dd-8fa4-1d300350478a)

</p>

The pseudocode for the proposed method, coined as "Uncertainty-aware Label Correction (ULC)"




- Uncertainty-aware Label Correction (ULC) has two major novelties:
- **"Epistemic Uncertainty-Aware Class-specific Noise Modeling (EUCS)"** module and the **"Aleatoric Uncertainty-aware Learning (AUL)"**

### 2.1.1 Epistemic Uncertainty-Aware Class-specific Noise Modeling (EUCS)
- First of these is the **"Epistemic Uncertainty-Aware Class-specific Noise Modeling (EUCS)"** module. With this module, the authors aim to fit the inter-class discrepancy on the loss distribution.
- Initially, the authors obtain the epistemic uncertainty estimations for each of the samples through utilizing MC Dropout [REF]. With MC Dropout, _T_ stochastic forward passes are performed with dropout enabled for each of the input samples during the test time. Following obtaining the output probabilities from each of these passes, taking their entropy and then normalizing it would be yielding the epistemic uncertainty for that particular sample.
- After the epistemic uncertainty estimation, the authors fit a GMM with to the each class's loss distribution, then compute the probability of having the mean of the component with the lower $\mu$ given each samples' loss, i.e $p(\mu_{j0} | l_i)$ for class j and sample i.
- Based on these two steps, the authors then come up with the following equation to determine the probability of a given sample i being clean or noisy:

<p align="center">
$\omega_i = (1-\epsilon)r p(\mu_{j0} | l_i)^{1-r}$
</p>

where $\epsilon_i$ corresponds to the epistemic uncertainty for that sample and $r$ being a hyperparameter to weight the uncertainty and the probability from GMM.

- The authors then apply thresholding based on hyperparameter $\tau$ to decide which samples are considered clean and which samples are considered noisy. At this stage, the noisy samples are discarded as being unlabeled.
- Finally, label of the cleaned samples are also refined based on the following equation:

<p align="center">
$y_i = \omega_i y_i^{~} + (1-\omega_i) \hat{y_i}$
</p>

where $y_i^{~}$ is the label possibly with noise and  $\hat{y_i} = \frac{1}{T} \sum_{t} softmax(f(x_i, W))$ 

![Screenshot 2023-06-18 at 21 41 25](https://github.com/selimkuzucu/ULC-CENG502/assets/56355561/1d7bbae3-c668-4981-b250-34073a4ec724)

The pseudocode for the EULC module of the proposed framework




### 2.1.2 Aleatoric Uncertainty-aware Learning (AUL)

- In this second module, the authors aim to utilize an objective akin to the one proposed by Kendall [REF].
- The authors claim that this module is particularly important as the noise modeling achieved by the EULC module is not sufficient to account for the residual noise that may contribute to overfitting in certain cases.
- Specifically, the authors aim to model aleatoric uncertainty through logit corruption with Gaussian noise, which leads to a learned loss attenuation as described more in detail in Kendall [REF]. This learned loss attenuation is particularly helpful while learning against noise and providing robustness as it attenuates the effects of corrupted labels.
- Two different types of noise are considered with the assumption of independence between them: Instance-dependent noise and class-dependent noise. Formally, the corresponding corruption process can be observed from the following equation:

<p align="center">
  $\hat{v}_i(W) = \delta^{x_i}(W) + \delta^y f_{W}(x_i)$
</p>

where $\hat{v}_i(;)$ stands for the $i^{th}$ logit, $\delta^{x_i}$ stands for the instance-dependent noise factor and $\delta^y$ stands for the class-dependent noise factor.

- Then, these corrupted logits are passed through a softmax layer to obtain output proabilities:

<p align="center">
  $\hat{y_i} = softmax(\delta^{x_i}(W) + \delta^y f_{W}(x_i))$
</p>

- Furthermore, the authors assume that both $\delta^{x_i}$ and $\delta^y$ are drawn from their respective Gaussian distributions (with the assumption of independence between them) and reserve an additional head from the network to effectively predict and learn $\sigma^{x_i}$ of the Gaussian that we sample the $\delta^{x_i}$ from.

- Finally, for the objective part, the authors leverage the aforementioned bulletpoints to define the following objectives for the labeled and unlabeled samples:

<p align="center">
  $l_x = \frac{-1}{X'} \sum_{x_i, y_i} (y_i)^{T} log \frac{1}{T} \sum_t \hat{y_{it}}(x_i ; W, \sigma^{x_i}, \sigma^{Y})$
</p>

<p align="center">
  $l_u = \frac{1}{U'} \sum_{x_i, y_i} || y_i - \frac{1}{T} \sum_t \hat{y_{it}}(x_i ; W, \sigma^{x_i}, \sigma^{Y}) ||^2_2$
</p>

where $X'$ stands for the labeled inputs (following MixMatch[REF] augmentation), $l_x$ stands for the labeled inputs' objective, , $U'$ stands for the unlabeled input (following MixMatch[REF] augmentation), $l_u$ stands for the unlabeled inputs' objective, and $\hat{y_{it}}$ stands for the softmax output of the $t^{th}$ sampling pass from the network.

- Introducing a balancing hyperparameter $\lambda_u$ between these two objectives and combining them yields the objective that the paper utilizes:

<p align="center">
  $l_c = l_x + \lambda_u l_u$
</p>


![Screenshot 2023-06-19 at 00 21 58](https://github.com/selimkuzucu/ULC-CENG502/assets/56355561/dfb4afce-5dd3-4be6-8c0d-9efe0f517d96)

The pseudocode for the AUL module of the proposed framework




## 2.2. Our interpretation 

- First and foremost, this paper has strict prerequisites of understanding other works such as DivideMix[REF], Kendall[REF], MixMatch[REF], MC Dropout[REF].
- This is primarily due to the fact that crucial procedures such as epistemic uncertainty quantification and/or aleatoric uncertainty-based loss attenuation are not explained in a detailed manner and the authors rather refer the reader to the relevant citations.
- Furthermore, the structure of the objective functions for both the labeled and unlabeled samples do not seem trivial from a first sight but makes much more sense after carefully reading DivideMix[REF] as their structure and principle is extremely similar to it.



![Screenshot 2023-06-19 at 00 27 16](https://github.com/selimkuzucu/ULC-CENG502/assets/56355561/d2215987-08b2-4c96-a012-b88fa593d1a6)

Objectives for labeled and unlabeled samples respectively from DivideMix[REF]




- Other important details, such as the co-teaching strategy with two networks, the semi-supervised learning strategy for the unlabeled samples and many more design choices are explained in a more detailed manner in the works cited by the authors. Since this work is not the one to introduce them, neither the authors of the paper nor this repository's discussion sections will be providing details on those parts.

- Finally, the overall pipeline and the algorithm borrows heavily from the DivideMix[REF] work, while also introducing many novelties to it at the same time, especially from the uncertainty quantification literature. Hence, my suggestion to anyone willing to use this repository is to first thoroughly go through the aforementioned citations in the first bulletpoint of this section.

  
![Screenshot 2023-06-19 at 00 31 16](https://github.com/selimkuzucu/ULC-CENG502/assets/56355561/ec35bd1e-b3b4-4b2b-9e1c-f5232ce47842)

The pseudocode for the DivideMix[REF], provided here for the sake of highlighting the similarities



# 3. Experiments and results

## 3.1. Experimental setup

In the paper, training settings are desribed as follows:

<p align="center">
  We train the network using SGD with a momentum of 0.9 for 300 epochs; warm-up 30 epochs for CIFAR-10 and 40 epochs for CIFAR-100. In the Cloth- ing1M experiments, we use ResNet-50 with ImageNet pre- trained weights, following the previous work (Li, Socher, and Hoi 2020). The warm-up period is 1 epoch for Cloth- ing1M. 𝜏 is set as 0.6 for 90% noise ratio and 0.5 for oth- ers. 𝜆𝑢 is validated from {0, 25, 50, 150}. Generally, the hy- perparameters setting for MixMatch is inherited from Di- videMix without heavily tuning, because the SSL part is not our focus and can be replaced by other alternatives. We leverage MC-dropout [REF] to esti- mate uncertainty, setting 𝑇 to 10 and the dropout rate to 0.3. The uncertainty ratio 𝑟 is set as 0.1 to obtain the final clean probability.
</p>


## 3.2. Running the code

@TODO: Explain your code & directory structure and how other people can run it.

## 3.3. Results

@TODO: Present your results and compare them to the original paper. Please number your figures & tables as if this is a paper.

# 4. Conclusion

@TODO: Discuss the paper in relation to the results in the paper and your results.

# 5. References

@TODO: Provide your references here.
[1] original paper
[2] dividemix - thanks for the implementation
[3] kendall and gal
[4] gal and ghahramani

# Contact
Please don't hesitate to ask any questions!

Selim Kuzucu - selim686kuzucu@gmail.com
