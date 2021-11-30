---
typora-root-url: ./
---

1.

使用梯度下降法，使用alpha=4/121, beta=81/121的回溯直线搜索，能解得

x = (5.273082108419521e-09, 7.602272040793395e-11)

y = 1.4191670161978175e-17

直接使用Heavy ball method，解得

x = (9.30869147423489e-10, -8.9984017584314e-11)

y = 8.381148558431732e-19

有x的轨迹变化如图：

![](/momentum_x.png)

k-f(k)的半对数图如下：

![momentum_semilog](/momentum_semilog.png)



对比二者的k-f(k)半对数图, 可以看出，Heavy ball方法远远比普通的梯度下降法要快，前者100多次迭代即可收敛，后者需要两千多次迭代。不过并不是一直如此，一开始Heavy ball方法迭代速度并没有这样快，经历了一个较慢的过程之后才迅速下降。

2.障碍函数法：
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
t\nu^*
\end{bmatrix}=
\begin{bmatrix}{}
-\nabla g(x)\\
0
\end{bmatrix}
\\
解得x^*后，根据KKT条件，有
\lambda^* = Px^* + q + A^T\nu^*
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
其中g_0(x, s) = t * s -\sum_{i=1}^{n}log(x_i + s)\\
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
\nabla^2 g_0(x, s) & A_e^T \\
A_e & 0
\end{bmatrix}
\begin{bmatrix}{}
\Delta_0 x_{nt}\\
w
\end{bmatrix}
=\begin{bmatrix}{}
-\nabla g_0(x, s) \\
0
\end{bmatrix}\\
其中A_e = \begin{bmatrix}{}A&0\end{bmatrix}\\
初始值选择s^{(0)}=-min(x_i) + 1，牛顿法收敛得到的x_0^*就是原问题求解的初始值x^{(0)}
$$

$$
对于初值问题x_0满足Ax_0=b, 我们将A扩展到A_{extend} = \begin{bmatrix}{} A_{left} &A_{right}\\
0 & I
\end{bmatrix}, \\将B扩展到b_{extend} = \begin{bmatrix}{} b\\
0
\end{bmatrix}， \\有A_{extend}可逆，且A_{extend}^{-1}b_{extend}为Ax=b的解。\\
$$

我们的完整流程如下：

1. 对A进行扩展，求得满足等式约束Ax=b，但是不满足不等式约束的初值x0。
2. 使用阶段1方法，取初始t=1000，u=10，收敛指标为1e-8。使用回溯直线搜索的牛顿法进行迭代，alpha=0.4，beta=0.5，迭代终止指标为1e-12。能求得满足不等式和等式约束的初值x1。
3. 使用障碍函数法，取初始t=1，u=10，收敛指标为1e-8，使用回溯直线搜索的牛顿法进行迭代，alpha=0.4，beta=0.5，迭代终止指标为1e-12。能求得最优解x*和对应的对偶最优解、最优值。

解得最优值p*=214982.08024465642，最优解和对偶最优解储存在根目录下result/barrier的对应文件中。

对数对偶间隙log(n/t)与牛顿迭代次数k的关系如下：

![step](E:\Programming\ConvexOptimization\week11\doc\step.png)

原对偶内点法：
$$
x，\lambda， \nu的更新方向\Delta x, \Delta \lambda, \Delta \nu的更新方向满足如下方程：\\

\begin{bmatrix}{}
\Delta x\\
\Delta \lambda\\
\Delta \nu
\end{bmatrix}=
-\begin{bmatrix}{}
\nabla^2f(x) & -I &A^T\\
diag(\lambda) & diag(x) & 0\\
A & 0 & 0
\end{bmatrix}^{-1}
\begin{bmatrix}{}
r_{dual}\\
r_{cent}\\
r_{pri}
\end{bmatrix}\\
其中
\begin{bmatrix}{}
r_{dual}\\
r_{cent}\\
r_{pri}
\end{bmatrix}
=\begin{bmatrix}{}
\nabla f(x) - \lambda + A^T\nu\\
(\lambda_i x_i) - (1/t)1\\
Ax - b
\end{bmatrix}\\
代理对偶间隙\hat{\eta}(x, \lambda) = \sum_{i=1}{n}x_i\lambda_i(其中x_i >0, \lambda_i >=0)
$$
我们取u=10，使用牛顿法进行优化，使用标准的基于残差范数的回溯直线搜索搜索步长，取alpha=0.4，beta=0.5。

解得最优值为214982.0768222267，最优解和对偶最优解储存在根目录下result/primal_dual的对应文件中。

迭代次数k和对数代理对偶间隙、对数残差的关系图如下：

![log_eta](E:\Programming\ConvexOptimization\week11\doc\log_eta.png)

![log_residue](E:\Programming\ConvexOptimization\week11\doc\log_residue.png)

