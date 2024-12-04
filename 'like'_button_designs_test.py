# Test the impact of two different 'like' button designs
import numpy as np
import scipy.stats as stats

np.random.seed(42)

# A组：2500用户，点击率为20%
A_group = np.random.binomial(1, 0.2, 2500)

# B组：2500用户，点击率为23%
B_group = np.random.binomial(1, 0.23, 2500)

# 计算两组的平均点击率
A_click_rate = A_group.mean()
B_click_rate = B_group.mean()

# t检验
t_stat, p_value = stats.ttest_ind(A_group, B_group)

print(f"A组点击率: {A_click_rate}")
print(f"B组点击率: {B_click_rate}")
print(f"t统计量: {t_stat}")
print(f"p值: {p_value}")

if p_value < 0.05:
    print("结果有显著差异，B组设计更有效")
else:
    print("结果没有显著差异")
