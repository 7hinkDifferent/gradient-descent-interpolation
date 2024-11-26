## todo
3. detailed manuscript
4. simple report on usage and experiment
7. include gifs here
12. better logging
13. add comment and fix typo
14. clear print

slope! if outside the boundary, we trained the slope
play with freeze to check the slope

we can see the powerful fitting ability of gradient descent!

gif generation may take a long time.

## motivation
you may theorectically work out the solution for interpolation. however, results may differ and get extremely hard (for me) given different input distributions and criterions. so a simple workaround is using gradient descent to optimize given arbitrary objective function, input distribution, evaluation criterion.

## method

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


## usage

0. setup

install necessary packages by

```bash
pip install -r requirements.txt
```

1. fit a objective function

how to add arbitrary functions


2. test fitting results



3. pair with neural networks

you can load the parameters and write a static function which is a lot faster.

## study, investigation
may include a simple study on some cases

relative speed

performance on simple cases

error distribution for some activations

## troubleshooting

1. ill-posed problem
when is bumping, try reducing N, since there might be 
nearly ill-posed

```bash
python main.py --model adaptive --objective_func relu --xmin -12 --xmax 12 --epoch 1000
```

![](./assets/progress_jitter.gif)

simply using equidistant
```bash
python main.py --model equidistant --objective_func relu --xmin -12 --xmax 12 --epoch 1000
```

![](./assets/progress_equi.gif)

or reduce N to 5 (default is 10)

```bash
python main.py --model adaptive --objective_func relu --xmin -12 --xmax 12 --epoch 1000 --N 5
```

![](./assets/progress_N5.gif)

2. training with cuda

3. 

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