import pandas as pd 
from dtd import implied_volatility, implied_mu, merton_pd

## 获取样本数据，该数据已经按照所需的格式准备好。
data = pd.read_csv("sample_data.csv")
data["trading_date"] = pd.to_datetime(data["trading_date"])
data.set_index("trading_date", inplace=True)
data.sort_index(inplace=True)

equity = data["equity"].values 
liab = data["liability"].values 
rfr = data["rfr"].values 
gap = data["gap"].values 

## 计算mu和sigma
vol, flag = implied_volatility(equity, liab, rfr, gap, term=1)
vol = vol[0]
mu = implied_mu(vol, equity, liab, rfr, gap, term=1)

'''
计算隐含资产价值、 DTD和PD。
用历史数据估计的参数事实上只能计算最后一天的值，其余的PD需要里用当天往前一年的数据来得到参数。
以下，我们仍然使用同样的参数来计算整个历史，仅仅作为测试来查看历史变动，并不是正确做法。
'''
mertonPD, DTD, impAsset = merton_pd(equity, liab, rfr, vol, mu, term=1)

mertonPD = pd.Series(mertonPD, index=data.index)
DTD = pd.Series(DTD, index=data.index)
impAsset = pd.Series(impAsset, index=data.index)