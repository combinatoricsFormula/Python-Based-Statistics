________________________________________
	Data Loading
________________________________________
		Common Data Formats:
			CSV: pandas.read_csv()
			Excel: pandas.read_excel(), openpyxl, xlrd
			SQL Databases: pandas.read_sql(), sqlite3, sqlalchemy
			Big Data: pyspark, dask
			NoSQL: pymongo (MongoDB), redis-py
			Web Data: requests, beautifulsoup4, scrapy
			JSON: json, pandas.read_json()
			XML: xml.etree.ElementTree, lxml
			Parquet: pyarrow, dask.dataframe
________________________________________
	Data Cleaning
________________________________________
		Handling Missing Values:
			df.isnull(), df.notnull()
			Imputation:
			SimpleImputer: sklearn.impute.SimpleImputer()
			KNN Imputer: sklearn.impute.KNNImputer()
			MICE: fancyimpute.IterativeImputer()
			Interpolation: df.interpolate()
			Drop Missing: df.dropna()
			Replace Missing: df.fillna()
			Outliers Handling:
			Z-score: scipy.stats.zscore()
			IQR Method: np.percentile(), scipy.stats.iqr()
			Isolation Forest: sklearn.ensemble.IsolationForest()
			DBSCAN for Outlier Detection: sklearn.cluster.DBSCAN()
			Duplicate Data:
			df.drop_duplicates()
________________________________________
	Data Transformation and Standardization
________________________________________
		Scaling Techniques:
			Standardization (Z-score): sklearn.preprocessing.StandardScaler()
			Min-Max Scaling: sklearn.preprocessing.MinMaxScaler()
			Robust Scaling: sklearn.preprocessing.RobustScaler()
			Quantile Transformation: sklearn.preprocessing.QuantileTransformer()
			Log Transformation: np.log()
			Box-Cox Transformation: scipy.stats.boxcox()
			Yeo-Johnson Transformation: scipy.stats.yeojohnson()
			Handling Skew:
			
			Log-Normalization: scipy.stats.lognorm()
			Power Transformation: sklearn.preprocessing.PowerTransformer()
________________________________________
	Descriptive Statistics
		Central Tendency:
			Mean: np.mean()
			Median: np.median()
			Mode: scipy.stats.mode()
			Geometric Mean: scipy.stats.gmean()
			Harmonic Mean: scipy.stats.hmean()
			Dispersion:
			Variance: np.var()
			Standard Deviation: np.std()
			Range: df.max() - df.min()
			IQR (Interquartile Range): scipy.stats.iqr()
			Mean Absolute Deviation: scipy.stats.median_absolute_deviation()
			Shape:
			Skewness: scipy.stats.skew()
			Kurtosis: scipy.stats.kurtosis()
			Quantiles and Percentiles:
			np.percentile()
			df.quantile()
			Correlation and Covariance:
			Pearson: scipy.stats.pearsonr()
			Spearman: scipy.stats.spearmanr()
			Kendall Tau: scipy.stats.kendalltau()
			Covariance: np.cov()
________________________________________
	Probability Distributions
		Common Distributions:
			Normal Distribution: scipy.stats.norm()
			Poisson Distribution: scipy.stats.poisson()
			Binomial Distribution: scipy.stats.binom()
			Geometric Distribution: scipy.stats.geom()
			Exponential Distribution: scipy.stats.expon()
			Gamma Distribution: scipy.stats.gamma()
			Beta Distribution: scipy.stats.beta()
			Uniform Distribution: scipy.stats.uniform()
			Weibull Distribution: scipy.stats.weibull_min()
			Pareto Distribution: scipy.stats.pareto()
			Log-Normal Distribution: scipy.stats.lognorm()
			Student's t-distribution: scipy.stats.t()
	Special Functions:
			pnorm (CDF for Normal): scipy.stats.norm.cdf()
			qnorm (Quantile for Normal): scipy.stats.norm.ppf()
			dpois (PDF for Poisson): scipy.stats.poisson.pmf()
			ppois (CDF for Poisson): scipy.stats.poisson.cdf()
			dbinom (PMF for Binomial): scipy.stats.binom.pmf()
			pbinom (CDF for Binomial): scipy.stats.binom.cdf()
			dgamma (PDF for Gamma): scipy.stats.gamma.pdf()
			pgamma (CDF for Gamma): scipy.stats.gamma.cdf()
			dchisq (PDF for Chi-Square): scipy.stats.chi2.pdf()
			pchisq (CDF for Chi-Square): scipy.stats.chi2.cdf()
		Goodness of Fit:

			Kolmogorov-Smirnov Test: scipy.stats.ks_2samp()
			Anderson-Darling Test: scipy.stats.anderson()
			Chi-Square Test: scipy.stats.chisquare()
			Shapiro-Wilk Test: scipy.stats.shapiro()
________________________________________
	Data Visualization
		Basic Plots:

			Histograms: matplotlib.pyplot.hist(), seaborn.histplot()
			Box Plots: seaborn.boxplot(), matplotlib.pyplot.boxplot()
			Scatter Plots: matplotlib.pyplot.scatter(), seaborn.scatterplot()
			Line Plots: matplotlib.pyplot.plot(), seaborn.lineplot()
		Advanced Plots:

			Pair Plots: seaborn.pairplot()
			Heatmaps: seaborn.heatmap()
			Violin Plots: seaborn.violinplot()
			Density Plots: seaborn.kdeplot()
			Contour Plots: matplotlib.pyplot.contour()
		Interactive Visualization:

			Plotly: plotly.express
			Bokeh: bokeh.plotting
			Altair: alt.Chart()
			Dash: dash.Dash()
		Dimensionality Reduction:

		PCA: sklearn.decomposition.PCA()
		t-SNE: sklearn.manifold.TSNE()
		UMAP: umap.UMAP()
________________________________________
	Statistical Inference and Hypothesis Testing
		Hypothesis Testing:

			t-test: scipy.stats.ttest_ind()
			ANOVA: scipy.stats.f_oneway()
			Mann-Whitney U Test: scipy.stats.mannwhitneyu()
			Chi-Square Test: scipy.stats.chisquare()
			Kolmogorov-Smirnov Test: scipy.stats.ks_2samp()
			Fisher's Exact Test: scipy.stats.fisher_exact()
		Confidence Intervals:

			For Mean: scipy.stats.t.interval()
			For Proportions: Calculate using sample proportions, e.g., statsmodels.stats.proportion.proportion_confint()
			Bootstrapping: sklearn.utils.resample()
		Bayesian Inference:

			Markov Chain Monte Carlo (MCMC): PyMC3, PyStan, emcee
			Bayes Theorem: P(A|B) = (P(B|A) * P(A)) / P(B)
________________________________________
	Regression and Modeling
________________________________________
		Linear Regression: 
			sklearn.linear_model.LinearRegression()
			Logistic Regression: sklearn.linear_model.LogisticRegression()
			Ridge and Lasso Regression: sklearn.linear_model.Ridge(), Lasso()
			Polynomial Regression: sklearn.preprocessing.PolynomialFeatures()
			K-Nearest Neighbors Regression: sklearn.neighbors.KNeighborsRegressor()
			Support Vector Machines: sklearn.svm.SVR()
			Decision Trees: sklearn.tree.DecisionTreeRegressor()
			Random Forests: sklearn.ensemble.RandomForestRegressor()
			Gradient Boosting: sklearn.ensemble.GradientBoostingRegressor()
