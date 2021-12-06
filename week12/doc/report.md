---
typora-root-url: ./
---

1.

![Q1](/Q1.jpg)

2.(1)

![Q2](/Q2.jpg)

(2)

采用alpha=1000时，收敛到f(x)=0.01991554109994043，此时x结果见根目录下result/Q2/x.csv文件

log(k)和log(||x_k - x*||^2)的图像如下：

![q2_k_dist](/q2_k_dist.png)

log(k)和log(f(x_k))的图像如下：

![q2_k_fx](/q2_k_fx.png)



3.(1)
$$
min_{x}\frac{1}{2}||Ax-b||_2^2 + ||x||_1的次梯度g(x)为\\
A^TAx - A^Tb+sign(x),其中sign(x)_i=\left\{\begin{aligned}{}1 \quad x_i>0\\
0 \quad x_i=0\\-1\quad x_i<0\end{aligned}\right.\\

$$
具体推导如下：

![Q3](/Q3.jpg)

解得最优的h(x)=12.522447423125461,x存储在根目录下的result/Q3/x1.csv中

log(k)和log(||x_k+1 - x_k||_2)的图像如下：

![q3_k_dist_1](/q3_k_dist_1.png)

log(k)和log(||x_k-x*||_2)的图像如下：

![q3_k_dist_best_1](/q3_k_dist_best_1.png)

log(x)和log(h(x))的图像如下：

![q3_k_hx_1](/q3_k_hx_1.png)

(2)
$$
对f做泰勒展开，有f(y)=f(x) + \nabla f(x)^T(y - x) + \frac{1}{2}(y - x)^T\nabla^2f(x)(y - x)\\
\geq f(x) + \nabla f(x)^T(y - x) + \frac{1}{2}m(y-x)^T(y-x),其中\nabla f(x)^2 = A^TA\\
因此有对于任意x，y \in dom(f)， 有(y-x)^T(A^TA-mI)(y-x) \geq 0, 也就是A^TA-mI半正定\\
m最大值为A^TA最小特征值
$$
解得m=0.7569455027845506， 最优的h(x)=12.52240996025351,x存储在根目录下的result/Q3/x2.csv中

log(k)和log(||x_k+1 - x_k||_2)的图像如下：

![q3_k_dist_2](/q3_k_dist_2.png)

log(k)和log(||x_k-x*||_2)的图像如下：

![q3_k_dist_best_2](/q3_k_dist_best_2.png)

log(k)和log(h(x_k))的图像如下：

![q3_k_hx_2](/q3_k_hx_2.png)