import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
df = pd.read_csv('/Users/bitengjiao/vscode/AI期末项目/music_rating.csv',encoding='GBK')
train_df, test_df = train_test_split(df, test_size=0.3, random_state=42)
R_df = train_df.pivot_table(index='music_id', columns='user_id', values='rating')
R_df = R_df.fillna(0)
music_id = 46615
user_id = 97
rating = R_df.loc[music_id, user_id]
print('用户 %d 对音乐 %d 的评分为：%d' % (user_id, music_id, rating))
model_knn = NearestNeighbors(metric='cosine', algorithm='brute')
model_knn.fit(R_df)
similarity_matrix = 1 - model_knn.kneighbors_graph(R_df, n_neighbors=5, mode='distance').toarray()
def predict_rating(user_id, music_id):
    # 获取该用户对所有音乐的评分记录
    user_ratings = R_df.loc[:, user_id]
    # 寻找与该音乐最相似的K个音乐
    k = 10  # 可自行设置
    similarities = similarity_matrix[music_id-1, :]
    top_k_idx = similarities.argsort()[-k:]
    # 获取这K个相似音乐的评分记录
    similar_ratings = R_df.iloc[top_k_idx, user_id-1]
    # 计算该用户对这K个音乐的加权评分
    weighted_ratings = similar_ratings * similarities[top_k_idx]
    # 返回加权评分的平均值作为预测评分
    return np.sum(weighted_ratings) / np.sum(similarities[top_k_idx])
test_df['predicted_rating'] = test_df.apply(lambda row: predict_rating(row['user_id'], row['music_id']), axis=1)
mse = np.mean((test_df['predicted_rating'] - test_df['rating'])**2)
print('均方误差为：%.4f' % mse)
