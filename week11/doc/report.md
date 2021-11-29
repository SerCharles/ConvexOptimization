---
typora-root-url: ./
---

1.直接使用Heavy ball method，解得

x = (9.30869147423489e-10, -8.9984017584314e-11)

y = 8.381148558431732e-19

有x的轨迹变化如图：

![](/momentum_x.png)

k-f(k)的半对数图如下：

![momentum_semilog](/momentum_semilog.png)

使用固定步长的梯度下降法，取步长为1/121，能解得

x = (9.966786470976668e-09, 0.0)

y = 4.966841627902177e-17

对比二者的k-f(k)半对数图如下：

![momentum_compare](/momentum_compare.png)

可以看出，Heavy ball方法远远比普通的梯度下降法要快，前者100多次迭代即可收敛，后者需要两千多次迭代。

2.