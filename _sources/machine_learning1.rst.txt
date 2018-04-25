.. _machine_learning1:

.. toctree::
  :maxdepth: 4

機械学習1
========================

ランキング学習
--------------------

| 本ページではランキング学習について述べる。
| ランキング学習はクエリと検索対象群に対して、検索対象の順序付けを行う学習である。

.. math::
  rank = f(q, T)

| ここで、 :math:`q` はクエリであり、 :math:`T` は検索対象集合とする。
| 検索対象集合 :math:`T` の要素が1つで学習する場合、PairWiseなランキング学習といい、
| 検索対象集合 :math:`T` の要素が複数で学習する場合、ListWiseなランキング学習という。
| ニューラルネットワークを用いたランキング学習にListNet [#]_ とRankNet [#]_ が存在する。
|

.. [#] `ListNet <https://qiita.com/koreyou/items/a69750696fd0b9d88608>`_
.. [#] `RankNet <http://cympfh.cc/paper/ranknet.html>`_
