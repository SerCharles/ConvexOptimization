---
typora-root-url: ./
---
1.
$$
\nabla f(x) = (10x_1-\frac{5e^{-x_1-x_2}}{1+e^{-x_1-x_2}}, x_2-\frac{5e^{-x_1-x_2}}{1+e^{-x_1-x_2}})\\
\nabla^{2} f(x) = \begin{bmatrix}{}
10 + \frac{5e^{-x_1-x_2}}{(1+e^{-x_1-x_2})^2}&\frac{5e^{-x_1-x_2}}{(1+e^{-x_1-x_2})^2}\\
\frac{5e^{-x_1-x_2}}{(1+e^{-x_1-x_2})^2}&1+\frac{5e^{-x_1-x_2}}{(1+e^{-x_1-x_2})^2}
\end{bmatrix} \\
根据牛顿法，有 \Delta x_{nt} = -\nabla^{2} f(x)^{-1}\nabla f(x),  \lambda^{2} = \nabla f(x)^{T}\nabla^{2} f(x)^{-1} \nabla f(x)\\
我们取\alpha = 0.4, \beta = 0.5 ，即可带入用牛顿法进行运算。
$$
结果为x=（0.11246719， 1.12467185），y=1.969725574672439

x位置的变化曲线如图

![Q1_x](/Q1_x.png)

k与ln(f(k))的变化曲线如图

![Q1_y](/Q1_y.png)

2.(1)

![Q2_proof](/Q2_proof.jpg)

（2）

$$
\nabla f(x)_j = \sum_{i=1}^{n}\frac{a_i}{1-a_i^T x} + (\frac{2x_j}{1-x_j^2})\\
\nabla^2 f(x) = \sum_{i=1}^{n}\frac{a_i a_i^T}{(1-a_i^T x)^2} + diag(\frac{2(1 + x_i^2)}{(1-x_i^2)^2})\\
根据牛顿法，有 \Delta x_{nt} = -\nabla^{2} f(x)^{-1}\nabla f(x), \lambda^{2} = \nabla f(x)^{T}\nabla^{2} f(x)^{-1} \nabla f(x),可以带入牛顿法进行运算\\
$$

N=50时，解得y=-116.49478021150344，对应的x保存在作业根目录的result/Q2/A_50.csv中

k-ln(f(xk)-p*)曲线如图

![Q2_d_50](/Q2_d_50.png)

k-tk曲线如图

![](/Q2_t_50.png)

N=100时，y=-298.83964731578664，对应的x保存在作业根目录的result/Q2/A_100.csv中

k-ln(f(xk)-p*)曲线如图

![Q2_d_100](/Q2_d_100.png)

k-tk曲线如图

![Q2_t_100](/Q2_t_100.png)

3.
$$
该问题KKT条件为：\\
\begin{numcases}{}
Px^* + q + A^T\upsilon^*=0\\
Ax^*-b=0
\end{numcases}\\
因此牛顿法下降方向有\begin{bmatrix}{}
\nabla^2 f(x) & A^T\\
A& 0
\end{bmatrix} 
\begin{bmatrix}{}\Delta x_{nt}\\
w
\end{bmatrix} = \begin{bmatrix}{}-\nabla f(x)\\
0
\end{bmatrix}\\
其中\nabla f(x) = Px + q, \nabla^2 f(x) = P\\
用牛顿法解得x^*后，可以带入求解对偶问题最优解v^*=-(AA^T)^{-1}A(Px^*+q)。\\由于该问题Hesse矩阵半正定且满足Slater条件，因此强对偶，原问题和对偶问题最优解同时取到而且最优值一样。\\
对于初值问题x_0满足Ax_0=b, 我们将A扩展到A_{extend} = \begin{bmatrix}{} A_{left} &A_{right}\\
0 & I
\end{bmatrix}, \\将B扩展到b_{extend} = \begin{bmatrix}{} b\\
0
\end{bmatrix}， \\有A_{extend}可逆，且A_{extend}^{-1}b_{extend}为Ax=b的解。\\
我们取\alpha = 0.4, \beta = 0.5 ，即可带入用牛顿法进行运算。
$$
解得y=56957.0794509534，对应的原问题最优解x保存在作业根目录的result/Q3/x.csv中，对应的对偶问题最优解u保存在作业根目录的result/Q3/u.csv中。

k-ln(f(xk)-p*)曲线如图

![Q3_d](E:\Programming\ConvexOptimization\week10\doc\Q3_d.png)

k-tk曲线如图

![Q3_t](E:\Programming\ConvexOptimization\week10\doc\Q3_t.png)

可以看出，第一次迭代t取得1时，牛顿法可以一次取得二次优化的最优解。
