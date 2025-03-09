1. Data Loading
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
2. Data Cleaning
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
3. Data Transformation and Standardization
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
4. Descriptive Statistics
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
5. Probability Distributions
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
6. Data Visualization
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
7. Statistical Inference and Hypothesis Testing
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
8. Regression and Modeling
Linear Regression: sklearn.linear_model.LinearRegression()
Logistic Regression: sklearn.linear_model.LogisticRegression()
Ridge and Lasso Regression: sklearn.linear_model.Ridge(), Lasso()
Polynomial Regression: sklearn.preprocessing.PolynomialFeatures()
K-Nearest Neighbors Regression: sklearn.neighbors.KNeighborsRegressor()
Support Vector Machines: sklearn.svm.SVR()
Decision Trees: sklearn.tree.DecisionTreeRegressor()
Random Forests: sklearn.ensemble.RandomForestRegressor()
Gradient Boosting: sklearn.ensemble.GradientBoostingRegressor()
This list provides a detailed breakdown of probability distributions, statistical functions, and data science tools for processing, cleaning, transforming, and modeling data.

You said:
see still not what I want . include everything in academia. for example regression models have multiple. do it again in every section and dig deeper
ChatGPT said:
I understand the depth you're looking for now, and I apologize for the previous attempts not meeting your expectations. What you're asking for is a truly exhaustive, all-encompassing resource that delves into every concept, tool, function, distribution, and method for probability and statistics, including advanced methods, niche models, and all academic concepts across every relevant field, from basic to graduate-level statistics, as well as applications across domains like data science, economics, physics, engineering, finance, and biostatistics.

Given your request, this will be a much deeper, comprehensive exploration. Let’s dive in section by section and ensure no concept or tool is left out.

1. Data Loading and Importing
CSV Files: pandas.read_csv(), dask.dataframe.read_csv(), pyarrow.csv.read()
Excel Files: pandas.read_excel(), openpyxl.load_workbook()
SQL Databases:
pandas.read_sql()
sqlalchemy.create_engine()
pyodbc.connect()
sqlite3.connect()
JSON:
json.load(), pandas.read_json(), simplejson
Web Scraping:
beautifulsoup4, scrapy
Big Data:
pyspark.sql, dask.dataframe, databricks.connect
NoSQL:
MongoDB: pymongo.MongoClient()
Cassandra: cassandra-driver
Parquet and Avro:
pyarrow.parquet.read_table()
fastparquet
2. Data Cleaning
Missing Data:
Identify: df.isnull(), df.isna(), df.notna()
Remove: df.dropna()
Impute:
Mean/Median/Mode Imputation: sklearn.impute.SimpleImputer(strategy="mean")
KNN Imputation: sklearn.impute.KNNImputer()
Iterative Imputation: fancyimpute.IterativeImputer()
MICE (Multiple Imputation by Chained Equations): fancyimpute.IterativeImputer()
Advanced Imputation:
Regression Imputation: Linear regression models (sklearn.linear_model.LinearRegression())
Random Forest Imputation: sklearn.ensemble.RandomForestRegressor()
Outliers:
Z-Score Method: scipy.stats.zscore()
IQR Method: scipy.stats.iqr()
Isolation Forest: sklearn.ensemble.IsolationForest()
DBSCAN: sklearn.cluster.DBSCAN()
Local Outlier Factor (LOF): sklearn.neighbors.LocalOutlierFactor()
Duplicated Data:
df.drop_duplicates()
Formatting & Encoding:
Categorical Variables:
Label Encoding: sklearn.preprocessing.LabelEncoder()
One-Hot Encoding: pandas.get_dummies(), sklearn.preprocessing.OneHotEncoder()
Text Data:
Tokenization: nltk.word_tokenize(), spacy
Lemmatization: nltk.WordNetLemmatizer(), spacy
3. Data Transformation and Feature Engineering
Scaling and Normalization:
Standardization (Z-score): sklearn.preprocessing.StandardScaler()
Min-Max Scaling: sklearn.preprocessing.MinMaxScaler()
Robust Scaling: sklearn.preprocessing.RobustScaler()
Log Transformation: np.log()
Box-Cox: scipy.stats.boxcox()
Yeo-Johnson: scipy.stats.yeojohnson()
Quantile Transformation: sklearn.preprocessing.QuantileTransformer()
Feature Engineering:
Polynomial Features: sklearn.preprocessing.PolynomialFeatures()
Principal Component Analysis (PCA): sklearn.decomposition.PCA()
t-SNE: sklearn.manifold.TSNE()
UMAP: umap.UMAP()
Log Transformation: np.log()
Interaction Terms: Manually multiplying features
Binning: pandas.cut(), pandas.qcut()
Frequency Encoding: Encoding based on frequency counts
Target Encoding: Encoding categorical features based on the mean of the target variable
4. Descriptive Statistics
Central Tendency:
Mean: np.mean()
Median: np.median()
Mode: scipy.stats.mode()
Geometric Mean: scipy.stats.gmean()
Harmonic Mean: scipy.stats.hmean()
Dispersion:
Variance: np.var()
Standard Deviation: np.std()
Range: np.ptp()
Interquartile Range: scipy.stats.iqr()
Mean Absolute Deviation: np.median(np.abs(x - np.median(x)))
Skewness: scipy.stats.skew()
Kurtosis: scipy.stats.kurtosis()
Covariance and Correlation:
Covariance: np.cov()
Pearson’s Correlation: scipy.stats.pearsonr()
Spearman’s Rank Correlation: scipy.stats.spearmanr()
Kendall Tau: scipy.stats.kendalltau()
Point-Biserial Correlation: scipy.stats.pointbiserialr()
5. Probability Distributions
Common Distributions:
Normal Distribution:
PDF: scipy.stats.norm.pdf()
CDF: scipy.stats.norm.cdf()
Inverse CDF (Quantile): scipy.stats.norm.ppf()
Sampling: scipy.stats.norm.rvs()
Binomial Distribution:
PMF: scipy.stats.binom.pmf()
CDF: scipy.stats.binom.cdf()
Poisson Distribution:
PMF: scipy.stats.poisson.pmf()
CDF: scipy.stats.poisson.cdf()
Uniform Distribution:
PDF: scipy.stats.uniform.pdf()
CDF: scipy.stats.uniform.cdf()
Exponential Distribution:
PDF: scipy.stats.expon.pdf()
CDF: scipy.stats.expon.cdf()
Gamma Distribution:
PDF: scipy.stats.gamma.pdf()
CDF: scipy.stats.gamma.cdf()
Beta Distribution:
PDF: scipy.stats.beta.pdf()
CDF: scipy.stats.beta.cdf()
Log-Normal Distribution:
PDF: scipy.stats.lognorm.pdf()
CDF: scipy.stats.lognorm.cdf()
Special Functions:
pnorm: scipy.stats.norm.cdf()
qnorm: scipy.stats.norm.ppf()
dpois: scipy.stats.poisson.pmf()
ppois: scipy.stats.poisson.cdf()
dbinom: scipy.stats.binom.pmf()
pbinom: scipy.stats.binom.cdf()
dgamma: scipy.stats.gamma.pdf()
pgamma: scipy.stats.gamma.cdf()
Goodness of Fit:
Chi-Square Test: scipy.stats.chisquare()
Kolmogorov-Smirnov Test: scipy.stats.ks_2samp()
Shapiro-Wilk Test: scipy.stats.shapiro()
Anderson-Darling Test: scipy.stats.anderson()
6. Inferential Statistics and Hypothesis Testing
Parametric Tests:
t-test: scipy.stats.ttest_ind()
Paired t-test: scipy.stats.ttest_rel()
ANOVA (Analysis of Variance): scipy.stats.f_oneway()
Z-Test: statsmodels.stats.proportion.ztest()
F-test: scipy.stats.f_test()
Chi-Square Test: scipy.stats.chisquare()
Non-Parametric Tests:
Mann-Whitney U Test: scipy.stats.mannwhitneyu()
Kruskal-Wallis H Test: scipy.stats.kruskal()
Wilcoxon Signed-Rank Test: scipy.stats.wilcoxon()
Fisher’s Exact Test: scipy.stats.fisher_exact()
Confidence Intervals:
For Proportions: statsmodels.stats.proportion.proportion_confint()
For Means: scipy.stats.t.interval()
7. Regression Models
Linear Models:
Simple Linear Regression: sklearn.linear_model.LinearRegression()
Multiple Linear Regression: sklearn.linear_model.LinearRegression()
Ridge Regression: sklearn.linear_model.Ridge()
Lasso Regression: sklearn.linear_model.Lasso()
ElasticNet Regression: sklearn.linear_model.ElasticNet()
Non-Linear Models:
Polynomial Regression: sklearn.preprocessing.PolynomialFeatures()
Support Vector Regression: sklearn.svm.SVR()
Decision Tree Regression: sklearn.tree.DecisionTreeRegressor()
Random Forest Regression: sklearn.ensemble.RandomForestRegressor()
Gradient Boosting Regression: sklearn.ensemble.GradientBoostingRegressor()
XGBoost: xgboost.XGBRegressor()
LightGBM: lightgbm.LGBMRegressor()
8. Advanced Topics
Bayesian Methods:
Markov Chain Monte Carlo (MCMC): PyMC3, emcee
Gibbs Sampling: PyStan
Metropolis-Hastings: emcee
Time Series:
ARIMA: statsmodels.tsa.arima.model.ARIMA()
Exponential Smoothing: statsmodels.tsa.holtwinters.ExponentialSmoothing()
Seasonal Decomposition: statsmodels.tsa.seasonal.seasonal_decompose()
9. Machine Learning Models for Statistics
Classification: sklearn.linear_model.LogisticRegression(), sklearn.ensemble.RandomForestClassifier()
Clustering: sklearn.cluster.KMeans(), sklearn.cluster.DBSCAN(), sklearn.mixture.GaussianMixture()
Dimensionality Reduction: sklearn.decomposition.PCA(), sklearn.manifold.TSNE()
