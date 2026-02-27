# 岗位检索RAG系统
基于千问3.5 + Chroma向量库的岗位智能检索与总结系统

---

## 核心功能
- 🎯 **精准检索**：基于向量相似度的岗位检索，支持阈值过滤 / 自定义返回条数
- 🤖 **智能总结**：千问3.5生成贴合需求的岗位总结（薪资 / 城市 / 岗位类型）
- 📱 **双模式运行**：命令行 / 交互模式，适配不同使用场景
- 🛡️ **健壮兼容**：自动纠正模型名，适配多版本SDK，无调用失败风险

---

## 快速开始

## 1. 环境准备
### 安装依赖
pip install chromadb sentence-transformers dashscope pandas openpyxl

### 配置千问API Key（Linux/Mac）
export DASHSCOPE_API_KEY="你的阿里云灵积API Key"

## 2. 构建向量库
### 删除旧库（首次运行可跳过）
rm -rf examples/.chroma

### 构建新库（替换为你的Excel文件路径）
python examples/build_index_from_excel.py --excel jobs.xls
## 3. 运行检索
### 命令行模式：检索"10k"相关岗位，返回前5条
python examples/query.py --q "10k" --top_k 5

## 交互模式
python examples/query.py

### 核心参数
参数	                                说明	          示例
--q	检索关键词（必填）	                 --q            "10k 北京"
--top_k	返回条数（0 = 不限）	          --top_k         5
--min_sim	相似度阈值（0~1，默认 0.5）  	--min_sim       0.6
--all	返回全部高相似度结果	            --all
--model	千问模型名（默认qwen-plus）	    --model         qwen-plus

## 常见问题
1,千问调用失败：Model not exist
原因：模型名拼写错误，千问 3.5 官方正确名称为 qwen-plus（非 qwen-3.5-plus）
解决：使用 --model qwen-plus 参数，或修改代码默认模型名

2,模型下载慢 / 卡住
解决：设置国内镜像源后重新运行：
export HF_ENDPOINT=https://hf-mirror.com  # Linux/Mac
set HF_ENDPOINT=https://hf-mirror.com     # Windows

3,向量库构建报错：--excel 参数必填
解决：运行构建脚本时指定 Excel 文件路径：
python examples/build_index_from_excel.py --excel jobs.xls
