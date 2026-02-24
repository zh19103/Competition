# 导入核心库
import pandas as pd
import chromadb
from sentence_transformers import SentenceTransformer

# 1. 加载测试岗位数据
df = pd.read_csv("test_jobs.csv")
# 将每条岗位数据拼接成文本片段
df["text"] = df.apply(lambda row: 
    f"岗位：{row['岗位名称']}，技能要求：{row['技能要求']}，薪资：{row['薪资范围']}，晋升路径：{row['晋升路径']}", 
    axis=1)
text_chunks = df["text"].tolist()  # 提取文本片段

# 2. 初始化向量化模型和本地向量库
model = SentenceTransformer('all-MiniLM-L6-v2')  # 轻量模型，WSL运行无压力
# 向量库数据存在本地，可持久化
db = chromadb.PersistentClient(path="./a13_vector_db")
collection = db.get_or_create_collection(name="test_jobs")

# 3. 文本向量化并入库
embeddings = model.encode(text_chunks)
ids = [f"job_{i}" for i in range(len(text_chunks))]
collection.add(
    embeddings=embeddings,
    documents=text_chunks,
    ids=ids
)

# 4. 测试检索（模拟学生提问："我学Python，适合什么岗位？"）
query = "我学Python，适合什么岗位？"
query_embedding = model.encode([query])
results = collection.query(
    query_embeddings=query_embedding,
    n_results=2  # 返回最相关的2个岗位
)

# 5. 打印检索结果
print("检索到的相关岗位：")
for doc in results["documents"][0]:
    print("-", doc)
