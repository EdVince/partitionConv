# partitionConv

将Conv2d切成四个块做分块计算，目的是想要降低计算的存储占用

## Implement Detail

1. 代码基于Stable-Diffusion-NCNN魔改得来
2. 主要就是对kernel=3，stride=1，padding=1的Conv2d进行四分块的计算
3. 先用4个crop切出来四块，要多切一行确保padding不会错
4. 对切出来的四个块分别应用原始的Conv2d
5. 对计算完的四个块，再做一次crop，把多出来的边切掉
6. 最后两两做w维度的concat，最后再做h的concat还原

## Reult

模型比较大，计算的时候内存有浮动，我跑了好多次，平均下来感觉内存就小了500M左右，还没浮动的幅度大，感觉像是翻车了。