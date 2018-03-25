.. _bayes:

.. toctree::
  :maxdepth: 4

ベイズ推論
===================

ベイズ推定と尤度関数(likelihood function)
---------------------------------------------

  ベイズ推論の上で未知パラメータ :math:`\theta` の推定を行う。
  訓練データ :math:`D` と求めたい未知パラメータ :math:`\theta` に対して、
  同時確率は以下で与えられる。

  .. math::
    p(D,\theta) = p(D|\theta)p(\theta)

  ここで、 :math:`p(D|\theta)` を :math:`\theta` に関する尤度関数という。
  このとき、訓練データ :math:`D` に対するパラメータ :math:`\theta` の推定値は以下で与えられる。

  .. math::
    p(\theta|D) = \frac{p(D|\theta)p(\theta)}{p(D)} = \frac{p(D,\theta)}{p(D)}

予測分布(predictive distribution)
--------------------------------------

  上記の方程式から未知パラメータ :math:`\theta` が求められるが、このパラメータと
  観測済みデータ :math:`D` 使って今後得られるであろうデータ :math:`x^*` の分布を
  求めるための方程式は以下である。

  .. math::
    p(x^*|D) = \int p(x^*|\theta)p(\theta|D)d\theta


1次元ガウス分布におけるベイズ推定
---------------------------------------

ガウス分布(Gaussian distribution, Normal distribution)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. math::
  \mathcal{N}(\mu, \sigma^2) = \frac{1}{\sqrt{2\pi}\sigma^2}\exp(\frac{(x-\mu)^2}{2\sigma^2})

平均が未知パラメータの場合
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
データの集合 :math:`D=\{x_i\in \mathcal{R}| i \in [N] \}` に対して、
その分布をガウス分布と仮定できるとする。また、分散 :math:`\sigma` は既知として
平均 :math:`\mu` を推定する。
このとき、データ :math:`x \in D` に対して以下が成立する。

  .. math::
    p(x|\mu) = \mathcal{N}(x|\mu, \sigma^2)

上記の共役事前分布として、以下の分布を考える。

  .. math::
    p(\mu) = \mathcal{N}(\mu|m, \sigma_0^2)

このとき、データ集合 :math:`D` に対する平均値 :math:`\mu` は以下で与えられる。

  .. math::
    p(\mu|D) &\propto& p(D|\mu)p(\mu) \\
             &=& \{\prod_{i=1}^N p(x_i|\mu)\}p(\mu) \\
             &=& \{\prod_{i=1}^N \mathcal{N}(x_i|\mu, \sigma^2)\}\mathcal{N}(\mu|m, \sigma_0^2) \\
             &=& \{\prod_{i=1}^N \frac{1}{\sqrt{2\pi}\sigma^2}\exp(\frac{(x_i-\mu)^2}{2\sigma^2}) \} \frac{1}{\sqrt{2\pi}\sigma_0^2}\exp(\frac{(\mu-m)^2}{2\sigma_0^2}) \\

ここで、辺々対数を取る。

  .. math::
    \ln p(\mu|D) &=& \sum_{i=1}^N \ln \{ \frac{1}{\sqrt{2\pi}\sigma^2}\exp(\frac{(x_i-\mu)^2}{2\sigma^2}) \} + \ln \{ \frac{1}{\sqrt{2\pi}\sigma_0^2}\exp(\frac{(\mu-m)^2}{2\sigma_0^2}) \} \\
                 &=& \ln \frac{N}{\sqrt{2\pi}\sigma^2} + \sum_{i=1}^N \frac{(x_i-\mu)^2}{2\sigma^2}
                 + \ln \frac{N}{\sqrt{2\pi}\sigma_0^2} + \frac{(\mu-m)^2}{2\sigma_0^2} \\
                 &=& \sum_{i=1}^N \frac{(x_i-\mu)^2}{2\sigma^2} + \frac{(\mu-m)^2}{2\sigma_0^2} + const^{'}\\
                 &=& \frac{1}{2\sigma^2}\sum_{i=1}^N (x_i^2-2x_i\mu+\mu^2) + \frac{1}{2\sigma_0^2}(\mu^2-2m\mu+m^2) + const^{'}\\
                 &=& \frac{1}{2\sigma^2}\sum_{i=1}^N (\mu^2-2x_i\mu) + \frac{1}{2\sigma_0^2}(\mu^2-2m\mu) + const \\
                 &=& \frac{1}{2} (\frac{N}{\sigma^2}+\frac{1}{\sigma_0^2}) \mu^2 - (\frac{\sum_{i=1}^N x_i}{\sigma^2} + \frac{m}{\sigma_0^2})\mu + const \\

一方、事後分布がガウス分布となる事が既知であるから以下を得る。

  .. math::
    \ln p(\mu|D) &=& \ln \mathcal{N}(\mu|\hat{m}, \hat{\sigma}^2) \\
                 &=& \ln \frac{1}{\sqrt{2\pi}\hat{\sigma}^2}\exp(\frac{(\mu-\hat{m})^2}{2\hat{\sigma}^2}) \\
                 &=& \frac{(\mu-\hat{m})^2}{2\hat{\sigma}^2} + const^{'}\\
                 &=& \frac{1}{2\hat{\sigma}^2}(\mu^2-2\hat{m}\mu+\hat{m}^2)+ const^{'} \\
                 &=& \frac{1}{2\hat{\sigma}^2}\mu^2 - \frac{\hat{m}}{\hat{\sigma}^2}\mu + const \\

上記の2つの結果から以下を得る。

  .. math::
    \frac{N}{\sigma^2}+\frac{1}{\sigma_0^2} &=& \frac{1}{2\hat{\sigma}^2} \\

  .. math::
    \frac{\sum_{i=1}^N x_i}{\sigma^2} + \frac{m}{\sigma_0^2} &=& \frac{\hat{m}}{\hat{\sigma}^2} \\

以上の結果より、未知パラメータ :math:`\mu` は上を満たすような :math:`\hat{m}, \hat{\sigma}` によって
以下で与えられる。

  .. math::
    p(\mu|D) &=& \mathcal{N}(\mu|\hat{m}, \hat{\sigma}^2) \\
             &=& \frac{1}{\sqrt{2\pi}\hat{\sigma}^2}\exp(\frac{(\mu-\hat{m})^2}{2\hat{\sigma}^2}) \\

多次元ガウス分布におけるベイズ推定
---------------------------------------

n次元ガウス分布(Gaussian distribution, Normal distribution)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. math::
  \mathcal{N}(\boldsymbol{\mu}, \Sigma) = \frac{1}{\sqrt{2\pi}^n\sqrt{|\Sigma|}}\exp(-\frac{1}{2}(\boldsymbol{x}-\boldsymbol{\mu})^{T}\Sigma^{-1}(\boldsymbol{x}-\boldsymbol{\mu}))

平均が未知パラメータの場合
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

ベイズ推論に基づく線形回帰
-------------------------------

線形回帰モデル
^^^^^^^^^^^^^^^^^
| データ :math:`\boldsymbol{x_i} \in X` とラベルデータ :math:`y_i \in Y` に対して、
| 回帰係数 :math:`\boldsymbol{w}` を推定して、回帰を行う。ただし、データ数は :math:`n` とする。
| このときの回帰モデルは以下とする。

  .. math::
    y_i = \boldsymbol{w} \boldsymbol{x_i}+\epsilon_{i} ,~~~~~
    i \in \{1,2,\dots,n\}

| ここで、 :math:`\epsilon_{i}` は定数項(誤差項)とする。
| 変数について、データ :math:`\boldsymbol{x_i}, \boldsymbol{w}` は :math:`m` 次元ベクトルであるが、
| :math:`y_i` はスカラーである。

  .. math::
    y_i \in \mathcal{R} ,~
    \boldsymbol{x_i} = \left(
      \begin{array}{c}
        x_{i1}  \\
        x_{i2}  \\
        \vdots  \\
        x_{im}  \\
      \end{array}
    \right) \in \mathcal{R^m} ,~
    \boldsymbol{w} = \left(
      \begin{array}{c}
        w_{1}  \\
        w_{2}  \\
        \vdots  \\
        w_{m}  \\
      \end{array}
    \right) \in \mathcal{R^m}

モデルの学習
^^^^^^^^^^^^^^^^

| 上記の記述の上で :math:`\boldsymbol{w}` を推定する。

  .. math::
    p(\boldsymbol{w}|Y, X) &=& \frac{p(\boldsymbol{w})\prod_{i=1}^{n}p(y_i|\boldsymbol{x}_i, \boldsymbol{w})}{p(Y|X)} \\
    &\propto& p(\boldsymbol{w})\prod_{i=1}^{n}p(y_i|\boldsymbol{x}_i, \boldsymbol{w}) \\

| 対数をとる。

  .. math::
    \ln p(\boldsymbol{w}|Y, X) &=& \ln p(\boldsymbol{w})\prod_{i=1}^{n}p(y_i|\boldsymbol{x}_i, \boldsymbol{w}) \\
    &=& \ln \mathcal{N}(\boldsymbol{w}|\boldsymbol{m},\Sigma) + \sum_{i=1}^{n} \ln \mathcal{N}(y_i|\mathcal{w}^T\mathcal{x}_i, \sigma^2) \\
    &=& -\frac{1}{2}(\boldsymbol{w}-\boldsymbol{m})^{T}\Sigma^{-1}(\boldsymbol{w}-\boldsymbol{m}) + \sum_{i=1}^{n} \ln \mathcal{N}(y_i|\mathcal{w}^T\mathcal{x}_i, \sigma^2)  + const^{'} \\

| ここで、共分散行列は対称行列であるため、以下のように変形できる。

  .. math::
    &=& -\frac{1}{2}(\boldsymbol{w}^{T}\Sigma^{-1}\boldsymbol{w}-2\boldsymbol{w}^{T}\Sigma^{-1}\boldsymbol{m})+\sum_{i=1}^{n} \ln \mathcal{N}(y_i|\mathcal{w}^T\mathcal{x}_i, \sigma^2) + const^{''} \\
    &=& -\frac{1}{2}(\boldsymbol{w}^{T}\Sigma^{-1}\boldsymbol{w}-2\boldsymbol{w}^{T}\Sigma^{-1}\boldsymbol{m})+\sum_{i=1}^{n} \frac{1}{2\sigma^2}(y_i-\boldsymbol{w}^T\boldsymbol{x})^2 + const^{'''} \\
    &=& -\frac{1}{2}(\boldsymbol{w}^{T}\Sigma^{-1}\boldsymbol{w}-2\boldsymbol{w}^{T}\Sigma^{-1}\boldsymbol{m})+\sum_{i=1}^{n} \frac{1}{2\sigma^2}((\boldsymbol{w}^T\boldsymbol{x})^2-2\boldsymbol{w}^{T}\boldsymbol{x}_iy_i) + const \\

| 更に、 :math:`(\boldsymbol{w}^{T}\boldsymbol{x}_i)^2 = \boldsymbol{w}^{T}\boldsymbol{x}_i\boldsymbol{x}_i^{T}\boldsymbol{w}` より、以下を得る。

  .. math::
    &=& -\frac{1}{2}(\boldsymbol{w}^{T}\Sigma^{-1}\boldsymbol{w}-2\boldsymbol{w}^{T}\Sigma^{-1}\boldsymbol{m})+ \frac{1}{2\sigma^2} \sum_{i=1}^{n}(\boldsymbol{w}^{T}\boldsymbol{x}_i\boldsymbol{x}_i^{T}\boldsymbol{w}-2\boldsymbol{w}^{T}\boldsymbol{x}_iy_i) + const \\
    &=& -\boldsymbol{w}^{T}(\Sigma^{-1}\boldsymbol{m}+\sum_{i=1}^n\boldsymbol{x}_iy_i) - \frac{1}{2} \boldsymbol{w}^{T}(\Sigma^{-1}+\frac{1}{\sigma^2}(\sum_{i=1}^n \boldsymbol{x}_i\boldsymbol{x}_i^{T}))\boldsymbol{w} + const \\

| 一方で、事後分布がガウス分布に従うことがわかっているので、以下が得られる。

  .. math::
    p(\boldsymbol{w}|Y, X) &=& \mathcal{N}(\boldsymbol{w}|\hat{\boldsymbol{m}}, \hat{\Sigma}) \\
    &=& -\frac{1}{2}(\boldsymbol{w}-\hat{\boldsymbol{m}})^{T}\hat{\Sigma}^{-1}(\boldsymbol{w}-\hat{\boldsymbol{m}}) + const^{'} \\
    &=& -\boldsymbol{w}^{T}\hat{\Sigma}^{-1}\hat{\boldsymbol{m}} -\frac{1}{2}\boldsymbol{w}^{T}\hat{\Sigma}^{-1}\hat{\boldsymbol{w}} + const \\

| したがって、以下を解くことによって :math:`\boldsymbol{w}` の事後分布を推定できる。

  .. math::
    \hat{\boldsymbol{m}} &=& \hat{\Sigma}(\Sigma^{-1}\boldsymbol{m}+\sum_{i=1}^n\boldsymbol{x}_iy_i) \\
    \hat{\Sigma}^{-1} &=& \Sigma^{-1}+\frac{1}{\sigma^2}(\sum_{i=1}^n \boldsymbol{x}_i\boldsymbol{x}_i^{T}) \\

学習後のモデルによる予測
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

モデルの評価
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
