## todo
3. detailed manuscript
4. simple report on usage and experiment
7. include gifs here
12. better logging
13. add comment and fix typo
14. clear print

slope! if outside the boundary, we trained the slope
play with freeze to check the slope

"""
四种方法：
1. 等距插值 RatioInterval {'epoch': 9999, 'loss': 3.3225504125766747e-07}
2. 等距区间+优化参数 RatioIntervalTunedParam {'epoch': 9999, 'loss': 8.87078329014912e-07}
3. 可调插值点 TunedInterval {'epoch': 9999, 'loss': 1.2212714750603482e-07}
4. 可调区间+优化参数 TunedIntervalTunedParam {'epoch': 9999, 'loss': 2.3224330902849033e-07}

如何控制不乱序
1. 乱序重排
2. 控制变化区间
    1. 梯度控制（不如第二种直接）
    2. 限制区间（记录原来的区间，clamp）

torch.linspace 没有梯度，要手动去算！
TODO: clamp_(min, max) need for forward?
"""

Sigmoid函数拟合
- 分段函数：一次、二次、三次
- 评价标准：绝对拟合误差、最大的绝对拟合误差、误差分布
- 拟合方式：手算、梯度优化
- 进阶要求：$\lambda$, speed, performance, channel-wise?

导出函数
- Function
  - 前向、后向
  - 如何高效计算？
  - 确实没有必要重新弄，在原来的基础上写好就行。不能简单弄，因为y值和target_func有关，还是得提取出来emm
  - just freeze
- 如何提取参数
  - 先不考虑怎么提取参数，先运算通，看看需求怎么样
  - 提取参数的话便于直接运算，可能吗，不得不牛顿插值

Experiment
- 函数
  - N, bl, br, min, max
- 初始
  - 函数拟合精度、误差、速度
- 进阶
  - 简单的图像分类任务速度、性能

## motivation

you may theorectically work out the solution for interpolation. however, results may differ and get extremely hard (for me) given different input distributions and criterions. so a simple workaround is using gradient descent to optimize given arbitrary objective function, input distribution, evaluation criterion.

we can see the powerful fitting ability with gradient descent!

![](./assets/equidistant_tuned_values.gif)

## method

given objective function (`objective_func`), the number of intervals (`N`) and degree of polynomial (`degree`), we want to interpolate with polynomials within intervals, and interpolate linearly outside the left / right most boundary (`bl`, `br`). so we need to work out the parameters for interpolation points for each interval (`sample_points`) and the slope for outside linear interpolation (`sl`, `sr`).

a tricky way to find nearly optimal parameters is neural-network-like optimization. with the problem formulation as below

$$ min\ loss(y_{ref}, y_{pre}) $$
$$ st. \ y_{ref} = objective\_func(x) $$
$$ y_{pre} = linear_{left}(x) * I_{x\in left} + linear_{right}(x) * I_{x\in right} \sum_i interpolation_i(x) * I_{x\in i} $$
$$ linear_{left}(x) = y_{l} + s_l * (x-b_l) $$
$$ linear_{right}(x) = y_{r} + s_r * (x-b_r) $$
$$ interpolation_i (x)=\sum_j y_j\frac{\prod_{k\ne j}(x-x_k)}{\prod_{k\ne j}(x_j - x_k)} $$

we can derive gradients using backpropagation to update parameters just like training a nerual network!

to this end, we have four methods to implement: whether sample points are equally distributed and whether sample values are derive from objective function or learned.

### equidistant

`sample_points` are equally distributed and `intervals` are of the same length. `sample_values`, the corresponding values of `sample_points`, are calculated with the `objective_func`. 

so in this setting, we only need to optimize: 1) left / right boundary `bl` / `br` which control the `sample_points` distribution between them. 2) left / right slope `sl` / `sr` for outside linear interpolation.

![](./assets/equidistant.gif)

### equidistant_tuned_values

`sample_points` are equally distributed and `intervals` are of the same length. `sample_values`, however, are learned with the initial values from `Sigmoid`. 

so in this setting, we only need to optimize: 1) left / right boundary `bl` / `br` which control the `sample_points` distribution between them. 2) `sample_values` for `sample_points`. 3) left / right slope `sl` / `sr` for outside linear interpolation.

![](./assets/equidistant_tuned_values.gif)

### adaptive

`sample_points` are NOT equally distributed and `intervals` are NOT of the same length. `sample_values` are calculated with the `objective_func`. 

so in this setting, we only need to optimize: 1) `sample_points` for interval interpolation. 2) left / right slope `sl` / `sr` for outside linear interpolation.

for implementation, `sample_points` would sometimes cross each other, leading to instability of learning. so `sample_points_buffer` is proposed to restrict step size of single point. ie. `sample_points_buffer[i-1]` <= `sample_points[i]` <= `sample_points_buffer[i+1]`

![](./assets/adaptive.gif)

### adaptive_tuned_values

`sample_points` are NOT equally distributed and `intervals` are NOT of the same length. `sample_values`, are learned with the initial values from `Sigmoid`. 

so in this setting, we only need to optimize: 1) `sample_points` for interval interpolation. 2) `sample_values` for `sample_points`. 3) left / right slope `sl` / `sr` for outside linear interpolation.

![](./assets/adaptive_tuned_values.gif)

### note

blue line is the `objective_func`. orange line in the interpolation. red dots stand for `intervals` and green dots stand for `sample_points`. we can summarize the differences between these methods:

|method|equidistant|equidistant_tuned_values|adaptive|adaptive_tuned_values|
|:-:|:-:|:-:|:-:|:-:|
|`bl` / `br`|learnable|-|learnable|-|
|`sample_points`|x|learnable|x|learnable|
|`sample_values`|x|learnable|x|learnable|
|`sl` / `sr`|learnable|learnable|learnable|learnable|

we may draw some simple conclusions from the above demo
1. `*_tuned_values` models perform poorly if not properly initialized
2. `adaptive*` models can sometimes jitter, tiny distance between `sample_points` cause the ill-posed problem!
3. `adaptive_tuned_values` should be the most powerful model. however, it would be the most challenging to learn if not properly configured. `equidistant` would always deliver stable and relatively good performance, recommended to try first!
4. `*_tuned_values` models are proposed not just to interpolate, but to fit the `objective_func`. but in theory, they are somehow the same with non-`*_tuned_values` models the if given more `sample_points`.


## usage

you can freeze

0. setup

install necessary packages by

```bash
pip install -r requirements.txt
```

1. fit a objective function

how to add arbitrary functions


2. test fitting results

`Sigmoid` would be fair

3. pair with neural networks

you can load the parameters and write a static function which is a lot faster.

## study, investigation
may include a simple study on some cases

relative speed

performance on simple cases

error distribution for some activations

## troubleshooting

### ill-posed problem

when is bumping, try reducing N, since there might be 
nearly ill-posed

```bash
python main.py --model adaptive --objective_func relu --xmin -12 --xmax 12 --epoch 1000
```

![](./assets/progress_jitter.gif)

simply reduce N to 5 (default is 10) would do the trick

```bash
python main.py --model adaptive --objective_func relu --xmin -12 --xmax 12 --epoch 1000 --N 5
```

![](./assets/progress_N5.gif)

or using equidistant method

```bash
python main.py --model equidistant --objective_func relu --xmin -12 --xmax 12 --epoch 1000
```

![](./assets/progress_equi.gif)

### training with cuda

### gif generation may take a long time

## directories
- assets/
- src/
  - activations/
  - interpolation/
  - registry.py
  - seed.py
- main.py
- fitting_test.py
- nn_test.py
- requirements.txt

## reference
- Piecewice Linear Activation function
  - GitHub: https://github.com/MrGoriay/pwlu-pytorch
- torchpwl
  - GitHub: https://github.com/PiotrDabkowski/torchpwl
