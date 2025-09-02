---
layout: post
title: "양자화"
subtitle: AI 양자화
gh-repo: james-hyeok-kim/james-hyeok-kim.github.io.git
gh-badge: [star, fork, follow]
tags: [양자화, Quantization]
comments: true
mathjax: true
author: James Kim
---

[Reference](https://junstar92.tistory.com/519)

# Quantization

<p align="center">
<img width="1500" height="450" alt="image" src="https://github.com/user-attachments/assets/fb7d474d-d9ef-4e63-84b1-bad9fb25149e" />
</p>

<p align="center">
<img width="820" height="432" alt="image" src="https://github.com/user-attachments/assets/37031fd4-aef5-4c0e-9ae2-299917682a1a" />
</p>


위 그림의 왼쪽이 FP32일 때의 숫자이고 오른쪽은 formula를 통해 변환한 INT8에 해당하는 값입니다. 

실제로는 FP32과 INT8의 숫자는 딥러닝 모델의 weight에 해당함.

위 도표를 통해서 아래 식 (1), (2), (4)을 통해 정의한 Formula를 연산이 가능함.

$$
f_q(x,s,z)=Clip(round(\frac{x}{s})+z) \quad (1)
$$

* $fq(x,s,z)$: quantized value
* clip(): clip the values in a range (ex. 0 - 255)
* x: real value (float32)
* s: scale, 기존 실수값 범위 / 양자화 할 정수값 범위
* z: zero-point integer

$$
s=\frac{FP_{Max} − FP_{Min}}{Int_{Max}−Int_{Min}} \quad (2)
$$


위 표에서 최댓값은 4.67 이고 최솟값은 -4.75

FP32를 INT8(Unsigned) 로 변경한다고 가정하면, α 

$$
​Int_{Min}=0, \quad Int_{Max}=255
$$

$$
S = \frac{4.67−(−4.75)}{255−0} = \frac{9.42}{255}  =0.037 \quad (3)
$$


* Zero Point 식 정리

$$
z=round \left(\frac{FP_{Max} * Int_{Min} - Fp_{Min} * Int_{Max}}{FP_{Max} - FP_{Min}} \right)
$$

위 값을 이용하여 z를 구하면,

$$
z=round \left(\frac{4.67∗0−(−4.75)∗255}{4.67−(−4.75)} \right)=129
$$

구한 s,z 를 이용해 x=−3.57 을 식(1)을 통해 양자화를 진행하면 아래와 같다

$$
q=round \left(\frac{x}{s} \right)+z=round \left(\frac{−3.57}{0.037} \right)+129=33
$$


clipping 은 범위를 초과하는 값을 범위안에 가지도록함

$$
\text{clip}(x, \text{Int}_{\text{Min}}, \text{Int}_{\text{Max}}) =
\begin{cases}
\text{Int}_{\text{Min}} & \text{if } x < \text{Int}_{\text{Min}} \\
x & \text{if } \text{Int}_{\text{Min}} \le x \le \text{Int}_{\text{Max}} \\
\text{Int}_{\text{Max}} & \text{if } x > \text{Int}_{\text{Max}}
\end{cases}
$$
​

이와 같은 quantization 기법을 uniform quantization이라고 부르며, 결과로 얻어지는 quantized values는 균등하게 분포된다.

## Quantizatin 종류

### Uniform vs Non-uniform
<p align="center">
<img width="803" height="365" alt="image" src="https://github.com/user-attachments/assets/c127fe8a-7dfd-4618-8344-72587f0be4c1" />
</p>

### Symmetric vs Asymmetric
<p align="center">
<img width="1114" height="331" alt="image" src="https://github.com/user-attachments/assets/47cfec7a-b725-4630-a8bb-a231e6274f47" />
</p>

### Static vs Dynamic
* Run Time vs Offline

### Quantization Granularity
<p align="center">
<img width="1104" height="667" alt="image" src="https://github.com/user-attachments/assets/0cbe13fd-1777-459a-b025-d6076c1f9b22" />
</p>

* Layerwise Quantization
* Groupwise Quantization
* Channelwise Quantization
* Sub-Channelwise Quantization




