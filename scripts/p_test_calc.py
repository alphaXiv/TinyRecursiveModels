from statsmodels.stats.proportion import proportions_ztest

# successes and total observations
count = [round(0.5275 * 1000), round(0.43 * 1000)]
nobs = [1000, 1000]

# one-tailed test: is Model B better?
z_stat, p_value = proportions_ztest(count, nobs, alternative='larger')

print("z =", z_stat, "p =", p_value)

