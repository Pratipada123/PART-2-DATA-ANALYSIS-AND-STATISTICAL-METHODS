# Week 6: Probability and Statistical Testing
# Theory: Study probability distributions (normal, Poisson), hypothesis testing (ttest, chi-squared), and confidence intervals.
# Hands-On: Perform hypothesis testing using SciPy, simulate probability distributions with NumPy.
# Client Project: Perform statistical analysis for comparing two business strategies (e.g., t-test).

# If needed (usually in a new environment):
# pip install numpy scipy matplotlib

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Reproducibility so your numbers match each run
rng = np.random.default_rng(42)
rng

# PARAMETERS
mu, sigma, n = 50, 10, 2000

# SIMULATE
normal_data = rng.normal(loc=mu, scale=sigma, size=n)

# QUICK STATS
print(f"Normal: mean≈{normal_data.mean():.2f}, std≈{normal_data.std(ddof=1):.2f}")

# PLOT (one plot, no custom colors)
plt.figure()
plt.hist(normal_data, bins=30, density=True, alpha=0.6)
xs = np.linspace(normal_data.min(), normal_data.max(), 400)
plt.plot(xs, stats.norm.pdf(xs, loc=mu, scale=sigma))
plt.title("Normal(μ=50, σ=10): Histogram + Theoretical PDF")
plt.xlabel("Value"); plt.ylabel("Density")
plt.show()

# PARAMETERS
lam, n_pois = 5, 2000

# SIMULATE
poisson_data = rng.poisson(lam=lam, size=n_pois)

# QUICK STATS
print(f"Poisson: mean≈{poisson_data.mean():.2f}, var≈{poisson_data.var(ddof=1):.2f}")

# EMPIRICAL PMF vs THEORETICAL PMF
vals, counts = np.unique(poisson_data, return_counts=True)
emp_pmf = counts / n_pois
k = np.arange(vals.min(), vals.max()+1)

plt.figure()
plt.bar(vals, emp_pmf, width=0.9, alpha=0.6, label="Empirical PMF")
plt.plot(k, stats.poisson.pmf(k, mu=lam), marker='o', label="Theoretical PMF")
plt.title("Poisson(λ=5): Empirical vs Theoretical PMF")
plt.xlabel("k (count)"); plt.ylabel("Probability")
plt.legend()
plt.show()

t_stat, p_val = stats.ttest_1samp(normal_data, popmean=50)
print(f"One-sample t-test vs 50: t={t_stat:.3f}, p={p_val:.4f}")
print("Decision:", "Reject H0" if p_val < 0.05 else "Fail to reject H0")

# SIMULATE TWO GROUPS (different means)
groupA = rng.normal(loc=75, scale=10, size=40)
groupB = rng.normal(loc=80, scale=12, size=45)

t_stat, p_val = stats.ttest_ind(groupA, groupB, equal_var=False)  # Welch's t-test
print(f"Welch two-sample t-test: t={t_stat:.3f}, p={p_val:.4f}")
print("Decision:", "Reject H0" if p_val < 0.05 else "Fail to reject H0")

observed = np.array([10, 12, 8, 11, 9, 10])
expected = np.full(6, 10)

chi2, p_val = stats.chisquare(f_obs=observed, f_exp=expected)
print(f"Chi-square GOF (fair die): χ²={chi2:.3f}, p={p_val:.4f}")
print("Decision:", "Reject H0 (not fair)" if p_val < 0.05 else "Fail to reject H0 (no evidence of unfairness)")

# Contingency table:
# rows = Campaign A, Campaign B; columns = Yes, No
table = np.array([[30, 20],
                  [45, 15]])

chi2, p_val, dof, expected = stats.chi2_contingency(table, correction=False)
print(f"Chi-square Independence: χ²={chi2:.3f}, df={dof}, p={p_val:.4f}")
print("Expected counts:\n", np.round(expected, 2))
print("Decision:", "Reject H0 (association present)" if p_val < 0.05 else "Fail to reject H0 (no evidence of association)")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

# load csv
df = pd.read_csv(r"C:\Users\prati\OneDrive\Datafiles\test_data.csv")
df

# check the sanity
print("Shape:", df.shape)
print("\nInfo:")
print(df.info())
print("\nMissing values per column:")
print(df.isna().sum())

print("\nGroup counts:")
print(df["group"].value_counts())

# check duplicate user_ids
dupes = df["user_id"].duplicated().sum()
print(f"\nDuplicate user_id rows: {dupes}")

desc = df.groupby("group")["revenue"].agg(
    mean="mean", median="median", std="std", count="count", min="min", max="max")
desc

# box plot
df.boxplot(column="revenue", by="group")
plt.title("Revenue Comparison: Strategy A vs B")
plt.suptitle("")  # remove automatic subtitle
plt.xlabel("Strategy (group)")
plt.ylabel("Revenue")
plt.show()

# normality
group_a = df.loc[df["group"]=="A", "revenue"].values
group_b = df.loc[df["group"]=="B", "revenue"].values

shapiro_a = stats.shapiro(group_a)
shapiro_b = stats.shapiro(group_b)

print(f"Shapiro-Wilk A: stat={shapiro_a.statistic:.3f}, p={shapiro_a.pvalue:.4f}")
print(f"Shapiro-Wilk B: stat={shapiro_b.statistic:.3f}, p={shapiro_b.pvalue:.4f}")

# equal variances
lev_stat, lev_p = stats.levene(group_a, group_b, center="median")
print(f"Levene's test: stat={lev_stat:.3f}, p={lev_p:.4f}")
use_equal_var = lev_p >= 0.05  # True if variances are not significantly different
use_equal_var

# t-test (safe default): equal_var=False
t_stat, p_val = stats.ttest_ind(group_a, group_b, equal_var=False)
print(f"t = {t_stat:.3f}, p = {p_val:.6f}")

# Mean difference (B - A)
mean_a, mean_b = np.mean(group_a), np.mean(group_b)
diff = mean_b - mean_a

# Standard errors for Welch CI
s1_sq = np.var(group_a, ddof=1)
s2_sq = np.var(group_b, ddof=1)
n1, n2  = len(group_a), len(group_b)
se_diff = np.sqrt(s1_sq/n1 + s2_sq/n2)

# Welch-Satterthwaite degrees of freedom
df_welch = (s1_sq/n1 + s2_sq/n2)**2 / ((s1_sq**2)/((n1**2)*(n1-1)) + (s2_sq**2)/((n2**2)*(n2-1)))

# 95% CI
alpha = 0.05
t_crit = stats.t.ppf(1 - alpha/2, df_welch)
ci_low, ci_high = diff - t_crit*se_diff, diff + t_crit*se_diff

# Effect size: Cohen's d and Hedges' g
sp = np.sqrt(((n1-1)*s1_sq + (n2-1)*s2_sq) / (n1 + n2 - 2))   # pooled SD
cohens_d = diff / sp
J = 1 - 3 / (4*(n1 + n2) - 9)                                  # small-sample correction
hedges_g = cohens_d * J

print(f"Mean A = {mean_a:.3f}, Mean B = {mean_b:.3f}")
print(f"Mean difference (B - A) = {diff:.3f}")
print(f"95% CI for difference = [{ci_low:.3f}, {ci_high:.3f}]  (df≈{df_welch:.2f})")
print(f"Cohen's d = {cohens_d:.3f}, Hedges' g = {hedges_g:.3f}")

# Decision & business interpretation (auto-print)
alpha = 0.05
decision = "Reject H0 (significant difference)" if p_val < alpha else "Fail to reject H0 (no significant difference)"

print("---------- RESULT ----------")
print(f"Test: Welch's t-test (equal_var=False)")
print(f"t = {t_stat:.3f}, df ≈ {df_welch:.2f}, p = {p_val:.6f}")
print(f"Decision @ α={alpha}: {decision}")
print(f"Interpretation: On average, Strategy B is {diff:.2f} units "
      f"{'higher' if diff>0 else 'lower'} than Strategy A "
      f"[95% CI {ci_low:.2f} to {ci_high:.2f}].")
print(f"Effect size (Hedges' g): {hedges_g:.2f} "
      f"→ {'small' if abs(hedges_g)<0.5 else 'moderate' if abs(hedges_g)<0.8 else 'large'} practical impact.")