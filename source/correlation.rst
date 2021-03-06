.. _correlation:

  .. toctree::
    :maxdepth: 4

相関関係
=============

相関関係とは、
  2つのデータに対して、一方の値が大きいときに他方の値が大きいまたは小さいの関係を言う。
  相関関係は2つのデータがお互いに影響を与え合っている相互関連性である。

.. attention::
  | 相関関係に対して因果関係という関係があるが、因果関係は相関関係とは異なり、
  | 一方のデータが他方のデータに影響を与えるような一方向性の関係である。
  | 原因と結果の関係である。
  |


データの種類について
---------------------------

相関関係を統計的に推定する場合、各データを確率変数とみなす。
データには以下の3種類が考えられる。

* 計量尺度
* 順序尺度
* 名義尺度

計量尺度とは
^^^^^^^^^^^^^^^^^^^

| 実数で取得されるデータであり、連続値となる。

順序尺度とは
^^^^^^^^^^^^^^^^^^^

| データに順序関係があるような離散データである。
| 例えば、「ゲームは好きか」という質問に対して回答が以下のような5段階のデータ。
| 好き・やや好き・どちらでもない・やや嫌い・嫌い

名義尺度とは
^^^^^^^^^^^^^^^^^^^

| データに順序関係が存在しないような離散データである。
| 例えば、都道府県というデータでは東京都と大阪府には順序関係はない。


相関係数
-------------------

| 上記の各種類のデータに対して、確率変数同士の相関関係の強さを表す値を相関係数という。


計量尺度×計量尺度
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

| まず計量尺度同士の相関係数を求める方法について説明する。
| 計量尺度同士の相関係数はピアソンの積率相関係数を用いる。
| ピアソンの積率相関係数は以下の数式で表される値 :math:`r` である。

.. math::
  r &=& \frac{S_{xy}}{ \sqrt{S_{xx}S_{yy}} } \\
  S_{xx} &=& \sum_{i=1}^{n}(x_i-E(X))^2 \\
  S_{xy} &=& \sum_{i=1}^{n}(x_i-E(X))(y_i-E(Y)) \\
  (-1 \leq r \leq 1) \\

| このとき、各確率変数の関与率(寄与率)は相関係数 :math:`r` を用いて以下のように表される。

.. math::
  r^{2} &=& \frac{S_{xy}}{S_{xx}S_{yy}} \\
  (0 \leq r^2 \leq 1) \\

順序尺度×順序尺度
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

| この場合はスピアマンの順位相関係数を用いる。
| スピアマンの順位相関係数は以下の数式で表される値 :math:`r` である。

.. math::
  r &=& \frac{S_{rxry}}{ \sqrt{S_{rxrx}S_{ryry}} } \\
  S_{rxrx} &=& \sum_{i=1}^{n}(r_{xi}-E(R_{X}))^2 \\
  S_{rxry} &=& \sum_{i=1}^{n}(r_{xi}-E(R_{X}))(r_{yi}-E(R_Y)) \\
  (-1 \leq r \leq 1) \\

| 上記の数式はピアソンの積率相関係数を離散化された順序尺度に適用している。
| 他に順序尺度の相関を求める手法としてポリコリック相関係数 [#]_ が存在する。


ポリコリック相関係数(Polychoric Correlation)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
| ポリコリック相関係数は順序尺度の相関に用いる事を想定しており、
| 背後に連続的な値があり、その値は正規分布に従うような値である事を仮定して考える。

変数 :math:`X, Y` について、以下を仮定する。

.. math::
  \rho &:& X, Y の相関係数 \\
  a_i, i \in \{ 1, 2, \cdots , s \} &:& Xに対するカテゴリの閾値 \\
  b_j, j \in \{ 1, 2, \cdots , r \} &:& Yに対するカテゴリの閾値 \\
  n_{ij} \in \mathcal{N}    &:& Xのカテゴリがi かつ Yのカテゴリがjのデータの度数 \\

上記の仮定の上で最尤推定法によって相関係数 :math:`\rho` を推定する。
尤度方程式は以下となる。

.. math::
  L &=& C \prod_{i=1}^{s} \prod_{j=1}^{r} \pi_{ij}^{n_{ij}} \\

対数尤度関数に変換する。

.. math::
  l &=& \ln(C) + \sum_{i=1}^{s} \sum_{j=1}^{r} n_{ij} \ln(\pi_{ij}) \\

ここで、 :math:`\pi_{ij}` は以下で与えられる。

.. math::
  \pi_{ij} = \Phi_2(a_i, b_j) - \Phi_2(a_{i-1}, b_j) - \Phi_2(a_i, b_{j-1}) + \Phi_2(a_{i-1}, b_{j-1})

ただし、 :math:`\Phi_2(u, v)` は相関係数 :math:`\rho` の上での2次元正規分布関数である。

.. attention::
  | ここで、順序尺度の相関と計量尺度の相関は数学的には本質が同じである。
  | しかし、科学的には別の意味を持つ事に注意。
  | ピアソンの積率相関係数とスピアマンの順位相関係数の違いについては
  | 以下を参照。
  | http://www.snap-tck.com/room04/c01/stat/stat05/stat0503.html

名義尺度×名義尺度
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

| この場合はクラメールの連関係数を用いる。
| クラメールの連関係数は以下の数式で表される値 :math:`r` である。

.. math::
  r^{2} &=& \frac{\chi_{0}}{N(s-1)} \\

| 変数同士の独立性を検定している。
| :math:`\chi^2` 検定については以下を参照。
| https://bellcurve.jp/statistics/course/9496.html

.. [#] http://kosugitti.sakura.ne.jp/wp/wp-content/uploads/2013/08/polynote.pdf
