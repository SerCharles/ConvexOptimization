---
typora-root-url: ./
---

1.
$$
\nabla f(x) = (10x_1-\frac{5e^{-x_1-x_2}}{1+e^{-x_1-x_2}}, x_2-\frac{5e^{-x_1-x_2}}{1+e^{-x_1-x_2}})\\
\nabla^{2} f(x) = \left(\begin{array}{l}
10 + \frac{5e^{-x_1-x_2}}{(1+e^{-x_1-x_2})^2}&\frac{5e^{-x_1-x_2}}{(1+e^{-x_1-x_2})^2}\\
\frac{5e^{-x_1-x_2}}{(1+e^{-x_1-x_2})^2},&1+\frac{5e^{-x_1-x_2}}{(1+e^{-x_1-x_2})^2}
\end{array}\right) \\
根据牛顿法，有 \Delta x_{nt} = -\nabla^{2} f(x)^{-1}\nabla f(x),  \lambda^{2} = \nabla f(x)^{T}\nabla^{2} f(x)^{-1} \nabla f(x)\\
我们取\alpha = 0.4, \beta = 0.5 ，即可带入用牛顿法进行运算。
$$

结果为x=（0.11246719，1.12467185），y=1.969725574672439

x位置的变化曲线如图

![Q1_x](/Q1_x.png)

k与ln(f(k))的变化曲线如图

![Q1_y](/Q1_y.png)

2.(1)

![Q2_proof](/Q2_proof.jpg)



​    (2)
$$
\nabla f(x)_j = \sum_{i=1}^{n}\frac{a_i}{1-a_i^T x} + (\frac{2x_j}{1-x_j^2})\\
\nabla^2 f(x) = \sum_{i=1}^{n}\frac{a_i a_i^T}{(1-a_i^T x)^2} + diag(\frac{2(1 + x_i^2)}{(1-x_i^2)^2})\\
根据牛顿法，有 \Delta x_{nt} = -\nabla^{2} f(x)^{-1}\nabla f(x), \lambda^{2} = \nabla f(x)^{T}\nabla^{2} f(x)^{-1} \nabla f(x),可以带入牛顿法进行运算\\
$$
N=50时，解得y=-116.50112380347852

k-ln(f(xk)-p*)曲线如图

![Q2_d_50](/Q2_d_50.png)

k-tk曲线如图

![](/Q2_t_50.png)

N=100时，y=-298.8589389811623

k-ln(f(xk)-p*)曲线如图

![Q2_d_100](/Q2_d_100.png)

k-tk曲线如图

![Q2_t_100](/Q2_t_100.png)

