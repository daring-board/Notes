.. linear_regression:

.. toctree::
  :maxdepth: 4

線形回帰
=============

ベイズ推論に基づく線形回帰
-------------------------------

モデル
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

| 回帰係数 :math:`\boldsymbol{w}` の事前分布はガウス分布に従うものとし、以下のようにあらわす。

  .. math::
    p(\boldsymbol{w}) = \mathcal{N}(\boldsymbol{w}|\boldsymbol{m}, \Sigma)

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
| 予測データ :math:`x^*` が与えられたときに、ラベル :math:`y^*` の値を予測したい。
| このとき、学習データ :math:`X, Y` に対して、予測データ :math:`x^*` のラベルが :math:`y^*` として
| 得られる確率を求めればよい。また、その確率は以下のように表される。

  .. math::
    p(y^*|\boldsymbol{x}^*, Y, X)

ここまでの議論で、回帰係数 :math:`\boldsymbol{w}` の事前分布は以下を仮定し、
  .. math::
    p(\boldsymbol{w}) = \mathcal{N}(\boldsymbol{w}|\boldsymbol{m}, \Sigma)

回帰係数 :math:`\boldsymbol{w}` の事後分布として以下を得ている。
  .. math::
    p(\boldsymbol{w}|Y, X) = \mathcal{N}(\boldsymbol{w}|\hat{\boldsymbol{m}}, \hat{\Sigma})

| このとき、回帰係数 :math:`\boldsymbol{w}` が事前分布に従うと仮定した上で
| 予測データ :math:`x^*` のラベルが :math:`y^*` として得られる確率を求める。

  .. math::
    p(w|y^*, \boldsymbol{x}^*) &=& \frac{p(\boldsymbol{w})p(y^*|\boldsymbol{x}^*, \boldsymbol{w})}{p(y^*|\boldsymbol{x}^*)} \\
    \ln p(y^*|\boldsymbol{x}^*) &=& \ln p(y^*|\boldsymbol{x}^*, \boldsymbol{w}) - \ln p(w|y^*, \boldsymbol{x}^*) + const \\

| 計算はここまでの計算結果を利用すればよいので省略し、以下の結果を得る。

  .. math::
    p(y^*|\boldsymbol{x}^*) = \mathcal{N}(y^*|\mu^*, (\sigma^*)^2)

ただし、
  .. math::
    \mu^* &=& \boldsymbol{m}^{T}\boldsymbol{x}^* \\
    (\sigma^*)^2 &=& \sigma^2 + \boldsymbol{x}^{*T}\Sigma\boldsymbol{x}^* \\

| ここで、パラメータ :math:`\boldsymbol{m}, \Sigma` は回帰係数 :math:`\boldsymbol{w}` に
| 従うと仮定した場合の値である。
| 一方で、回帰係数 :math:`\boldsymbol{w}` が事後分布に従うと仮定した場合、以下のように表現される。

.. math::
  p(y^*|\boldsymbol{x}^*, Y, X) &=& \mathcal{N}(y^*|\mu^*, (\sigma^*)^2)  \\
  \mu^* &=& \hat{\boldsymbol{m}}^{T}\boldsymbol{x}^* \\
  (\sigma^*)^2 &=& \sigma^2 + \boldsymbol{x}^{*T}\hat{\Sigma}\boldsymbol{x}^* \\

モデルの評価
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
モデルの評価には以下のような周辺尤度を用いる。

  .. math::
    p(Y|X) &=& \frac{p(\boldsymbol{w})\prod_{i=1}^{n}p(y_i|\boldsymbol{x}_i, \boldsymbol{w})}{p(\boldsymbol{w}|Y, X)} \\
    \ln p(Y|X) &=& -\frac{1}{2}\{ \frac{1}{\sigma^2}\sum_{i=1}^{n}y_i^2 + 2\ln\sigma + \ln2\pi + \boldsymbol{m}^{T}\Sigma\boldsymbol{m} - \ln|\Sigma| -  \hat{\boldsymbol{m}}^{T}\Sigma\hat{\boldsymbol{m}} + \ln|\hat{\Sigma}|\}
