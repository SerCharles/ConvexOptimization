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

可以看出，Heavy ball方法远远比普通的梯度下降法要快，前者100多次迭代即可收敛，后者需要两千多次迭代。不过并不是一直如此，一开始Heavy ball方法迭代速度并没有这样快，经历了一个较慢的过程之后才迅速下降。

2.

障碍函数法：
$$
首先，对于原问题，有障碍函数\phi(x)=-\sum_{i=1}^{n}log(x_i)\\
加上障碍函数后，转化为求解等式约束凸优化问题：\\
minimize \quad g(x)\\
s.t.Ax=b\\
其中g(x)=t(\frac{1}{2}x^TPx + q^Tx) -\sum_{i=1}^{n}log(x_i)\\
\nabla g(x) = t(Px+q)-\begin{bmatrix}{}
\frac{1}{x1}
\\
\frac{1}{x2}
\\
...\\
\frac{1}{x_n}
\end{bmatrix}\\
\nabla^2 g(x) = tP + diag(\frac{1}{x_i^2})\\
牛顿法下降方向\Delta x_{nt}和对偶问题最优解\nu^*满足如下方程：\\
\begin{bmatrix}{}
\nabla^2 g(x) & A^T
\\
A & 0
\end{bmatrix}
\begin{bmatrix}{}
\Delta x_{nt} \\
\nu^*
\end{bmatrix}=
\begin{bmatrix}{}
-\nabla g(x)\\
0
\end{bmatrix}
\\
解得x^*后，根据KKT条件，\lambda^* = Px^* + q^T - A^T\nu^*
$$

$$
而需要首先求可行初值x^{(0)}，使用阶段1方法\\
即求解优化问题：\\
minimize \quad s:\\
s.t. -x_i - s \leq 0， i= 1 ... n\\
Ax=b\\
障碍函数\phi_0(x)=-\sum_{i=1}^{n}log(x_i + s)\\
问题转化为求解等式约束凸优化问题:\\
minimize \quad g_0(x, s)\\
s.t. Ax=b\\
其中g_0(x) = t * s -\sum_{i=1}^{n}log(x_i + s)\\
\nabla g_0(x, s) = 
\begin{bmatrix}{}
-\frac{1}{x_1 + s}\\
-\frac{1}{x_2 + s}\\
...\\
t - \sum_{i=1}^{n}\frac{1}{x_i + s}
\end{bmatrix}\\
\nabla^2 g_0(x, s) = 
\begin{bmatrix}{}
diag(\frac{1}{(s + x_i) ^ 2}) & M\\
M^T & \sum_{i=1}^{n}\frac{1}{(s + x_i)^2}
\end{bmatrix}, M = 
\begin{bmatrix}{}
\frac{1}{(x_1 + s)^2}\\
\frac{1}{(x_2 + s)^2}\\
...\\
\frac{1}{(x_n + s)^2}
\end{bmatrix}\\
牛顿法下降方向\Delta_0 x_{nt}满足如下方程：\\
\begin{bmatrix}{}
\nabla^2 g_0(x, s) & A^T \\
A & 0
\end{bmatrix}
\begin{bmatrix}{}
\Delta_0 x_{nt}\\
w
\end{bmatrix}
=\begin{bmatrix}{}
-\nabla g_0(x, s) \\
0
\end{bmatrix}\\
初始值选择s^{(0)}=-min(x_i) + 1，牛顿法收敛得到的x_0^*就是原问题求解的初始值x^{(0)}
$$

