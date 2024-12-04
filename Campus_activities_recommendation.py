# Recommend campus activities based on the user's interest
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity

ratings = pd.read_csv('user_ratings.csv', index_col=0)

# 使用SVD进行矩阵分解
svd = TruncatedSVD(n_components=20, random_state=42)
svd_matrix = svd.fit_transform(ratings)

user_similarity = cosine_similarity(svd_matrix)

# 假设我们想为第n个用户推荐活动
user_id = n-1 
similar_users = user_similarity[user_id]

# 找到与该用户最相似的前5个用户
top_similar_users = np.argsort(similar_users)[-6:-1]

print(f"与用户{user_id+1}最相似的用户ID: {top_similar_users + 1}")
