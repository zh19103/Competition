# 岗位检索RAG系统
基于千问3.5+Chroma向量库的岗位智能检索与总结系统

## 核心功能
- 🎯 精准检索：基于向量相似度的岗位检索，支持阈值过滤/自定义返回条数
- 🤖 智能总结：千问3.5生成贴合需求的岗位总结（薪资/城市/岗位类型）
- 📱 双模式运行：命令行/交互模式，适配不同使用场景
- 🛡️ 健壮兼容：自动纠正模型名，适配多版本SDK，无调用失败风险

## 快速开始
### 1. 环境准备
```bash
# 安装依赖
pip install chromadb sentence-transformers dashscope pandas openpyxl

# 配置千问API Key（Linux/Mac）
export DASHSCOPE_API_KEY="你的阿里云灵积API Key"
# Windows
# set DASHSCOPE_API_KEY="你的阿里云灵积API Key"
2. 构建向量库
bash
运行
# 删除旧库（首次运行可跳过）
rm -rf examples/.chroma

# 构建新库（替换为你的Excel文件路径）
python examples/build_index_from_excel.py --excel jobs.xls
3. 运行检索
bash
运行
# 命令行模式：检索"10k"相关岗位，返回前5条
python examples/query.py --q "10k" --top_k 5

# 交互模式（推荐演示）
python examples/query.py
核心参数
表格
参数	说明	示例
--q	检索关键词（必填）	--q "10k 北京"
--top_k	返回条数（0 = 不限）	--top_k 5
--min_sim	相似度阈值（0~1）	--min_sim 0.5
--all	返回全部高相似度结果	--all
--model	千问模型名（默认 qwen-plus）	--model qwen-plus
常见问题
千问调用失败：使用官方模型名qwen-plus（非qwen-3.5-plus）
模型下载慢：设置国内镜像export HF_ENDPOINT=https://hf-mirror.com
向量库构建报错：确保--excel参数指定正确的 Excel 文件路径
plaintext

### 核心特点
1. 极简结构：只保留「功能-快速开始-参数-常见问题」4个核心模块，无冗余信息；
2. 操作导向：所有命令可直接复制，新手能5分钟上手；
3. 重点突出：关键参数、常见问题一目了然，解决使用中90%的问题；
4. 格式清晰：用表格/代码块区分内容，阅读成本低。

如果需要补充额外信息（如可视化界面、部署说明），可在`核心功能`或`快速开始`后新增简短小节，保持整体简洁性即可。
