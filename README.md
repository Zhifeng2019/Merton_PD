# Merton_PD
PD based on Merton Model

这是微信公众号“猫咪风控”的文章《用Python实现基于Merton模型的违约概率》所使用的Python代码。包含两个主要功能：计算基于Merton模型的PD；利用历史数据估计参数mu和sigma。参数估计采用了MLE的办法。

主要的函数在dtd.py。main.py提供了使用该工具的例子。sample_data.csv作为测试数据。
