import numpy as np 
from scipy.stats import norm
from scipy import optimize


def bs_call_price(
	asset, 
	liab, 
	rfr, 
	vol, 
	term=1):
	# BSM期权公式
	## 输入变量
	# asset: 资产价值
	# liab:	负债
	# rfr:	无风险利率
	# vol:	波动率
	# term:	期限tau，默认为1
	d1 = (np.log(asset/liab )+(rfr+0.5*vol**2)*term)/(vol*np.sqrt(term))
	d2 = d1-vol*np.sqrt(term)
	px = asset*norm.cdf(d1)-np.exp(-rfr*term)*liab*norm.cdf(d2)
	return px


def bs_call_price_jac(
	asset, 
	liab, 
	rfr, 
	vol, 
	term=1):
	# BSM期权公式关于资产的求导，加速求解过程
	## 输入变量
	# asset: 资产价值
	# liab:	负债
	# rfr:	无风险利率
	# vol:	波动率
	# term:	期限tau，默认为1
	d1 = (np.log(asset/liab )+(rfr+0.5*vol**2)*term)/(vol*np.sqrt(term))
	return norm.cdf(d1)


def implied_asset(
	equity, 
	liab, 
	rfr, 
	vol, 
	term=1, 
	initial=None):
	# 使用牛顿法求解BSM方程 E=g(V; L, r, sigma, tau)
	## 输入变量
	# equity: 市值
	# liab:	负债
	# rfr:	无风险利率
	# vol:	波动率
	# term:	期限tau，默认为1
	# initial:	初始值，默认为负债+市值
	if initial is None:
		initial = equity+liab

	# 目标函数：f(V)=g(V)-E，求解f(V)=0
	ff = lambda x: bs_call_price(x, 
		liab, rfr, vol, term)-equity
	# 目标函数一阶导数：f'(V)=g'(V)
	fp = lambda x: bs_call_price_jac(x, 
		liab, rfr, vol, term)
	# 使用牛顿法
	impAsset = optimize.newton(ff, initial, fprime=fp)
	return impAsset


def merton_pd(
	equity, 
	liab, 
	rfr, 
	vol, 
	mu=None, 
	term=1):
	# 计算基于Merton模型的违约概率
	## 输入变量
	# equity: 市值
	# liab:	负债
	# rfr:	无风险利率
	# vol:	波动率
	# mu:	漂移项，如果mu为None，我们设mu=0.5vol^2
	# term:	期限tau，默认一年

	# 输出结果
	# 违约概率(dp)，DTD(dtd)，隐含资产价值(impAsset)
	impAsset = implied_asset(equity, liab, rfr, vol, term)
	if mu is not None:
		r = (mu-0.5*vol**2)*term
	else:
		r = 0
	# DTD
	dtd = (np.log(impAsset/liab)+r)/(vol*np.sqrt(term))
	# PD
	dp = norm.cdf(-dtd)
	return dp, dtd, impAsset


def objective_functionn(
	vol,
	equity,
	liab,
	rfr,
	gap,
	term=1):
	# 基于Meron模型的资产价值演化似然函数的相反数，将最大化问题转化为最小化问题
	## 输入变量：numpy的ndarray，从时刻t_0到t_n共n+1个点。
	# vol:	波动率
	# equity: 市值
	# liab:	负债
	# rfr:	无风险利率
	# gap:	相邻两个市值的交易日间隔，一般为一个交易日。间隔数目需要除以一年的总交易日数目。
	# term:	期限tau，默认一年

	## 输出结果：
	# 似然函数的相反数
	# 似然函数的相反数的导数
	n = len(equity)-1	# 总样本个数（i到i+1的变化为一个样本点）
	t = sum(gap[1:])	# 总时间，一般为1，因为已经和term同步过。
	# 隐含资产
	impAsset = implied_asset(equity, liab, rfr, vol, term)
	# 隐含资产下的d1，d2
	d1 = (np.log(impAsset/liab )+(rfr+0.5*vol**2)*term)/(vol*np.sqrt(term))
	d2 = d1-vol*np.sqrt(term)
	vChg = impAsset[1:]/impAsset[:-1]	# V_t/V_{t-1}
	vChgLog = np.log(vChg)	# log(V_t/V_{t-1})
	vChgLogSum = sum(vChgLog)	# sum(log(# V_t/V_{t-1}))

	h = -np.sqrt(term)*norm.pdf(d1)/norm.cdf(d1)	# 辅助因子h=V'/V
	hChg = h[1:]-h[:-1]
	q = h/(vol*np.sqrt(term))-d2/vol 	# 辅助因子 q=d1'/d1

	# 似然函数的6个项，与文章对应
	l1 = -0.5*n/np.log(2*np.pi)
	l2 = -n*np.log(vol)-0.5*sum(np.log(gap[1:]))
	l3 = -sum(np.log(impAsset[1:]))
	l4 = -sum(np.log(norm.cdf(d1[1:])))
	l5 = -1/(2*vol**2)*sum(1/gap[1:]*(vChgLog)**2)
	l6 = 1/(2*t*vol**2)*vChgLogSum**2

	lkh = l1+l2+l3+l4+l5+l6

	# 似然函数导数的6个项，与文章对应
	p1 = -n/vol-sum(h[1:])
	p2 = 1/np.sqrt(term)*sum(h[1:]*q[1:])
	p3 = 1/vol**3*sum(1/gap[1:]*(vChgLog)**2)
	p4 = -1/vol**2*sum(hChg/gap[1:]*vChgLog)
	p5 = -1/(vol**3*t)*vChgLogSum**2
	p6 = 1/(vol**2*t)*vChgLogSum*sum(hChg)

	lprime = p1+p2+p3+p4+p5+p6

	return -lkh, -lprime


def implied_volatility(
	equity, 
	liab, 
	rfr,
	gap, 
	term=1, 
	initial=0.05):
	# 利用MLE求解波动率
	## 输入变量
	# 必须为numpy的ndarray！
	# equity: 市值
	# liab:	负债
	# rfr:	无风险利率
	# gap:	相邻两个市值的交易日间隔，一般为一个交易日。间隔数目需要除以一年的总交易日数目。
	# term:	期限tau，默认一年	
	# initial:	优化的起始值，默认为0.05
	## 输出结果
	# 波动率、优化结果的状态（来自fmin_tnc)
	res = optimize.fmin_tnc(
		lambda x: objective_functionn(x, 
			equity, liab, rfr, gap, term), 
		x0=initial, bounds=[(0.0001,100)], disp=0)
	return res[0], res[2]


def implied_mu(
	vol,
	equity, 
	liab, 
	rfr, 
	gap,
	term=1):
	# 利用MLE的解析解得到mu
	## 输入变量
	# 必须为numpy的ndarray！
	# vol:	波动率
	# equity: 市值
	# liab:	负债
	# rfr:	无风险利率
	# gap:	相邻两个市值的交易日间隔，一般为一个交易日。间隔数目需要除以一年的总交易日数目。
	# term:	期限tau，默认一年

	## 输出结果
	# 漂移项mu
	impAsset = implied_asset(equity, liab, rfr, vol, term)
	vChg = impAsset[1:]/impAsset[:-1]
	return 0.5*vol**2+1/sum(gap[1:])*sum(np.log(vChg))
	
	






