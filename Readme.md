# 🏥 AI 健康科普互动助手 (HealthBot)

HealthBot 是一个基于 **LangGraph + Azure OpenAI Chat** 的互动式健康教育机器人，能够帮助用户学习特定健康主题、生成通俗摘要、提出练习题，并对用户回答进行评分与反馈。系统支持多轮循环学习，并可持续保存状态，适合作为医学科普教育或患者教育工具。

---

## 🔹 功能特点

- **主题搜索**：通过自定义 ResearcherAgent 调用 Tavily 搜索指定健康主题。
- **摘要生成**：基于搜索结果生成 3-4 段易懂科普总结。
- **练习题生成**：根据摘要生成开放式高难度问题。
- **答案评分**：使用 AI 对用户回答进行 A/B/C/F 评分并提供改进建议。
- **循环学习**：用户可选择继续学习新主题或结束会话。
- **多轮状态管理**：通过 MemorySaver 保存状态，实现中断恢复。

---

## 🔹 技术栈

- **后端**：
  - Python 3.12+
  - [LangGraph](https://github.com/langgraph/langgraph) 状态图管理
  - [LangChain](https://www.langchain.com/) 与 Azure OpenAI Chat 模型
  - Pydantic 数据验证与结构化输出
  - 异步执行 (`asyncio`)  
- **前端**：
  - [Streamlit](https://streamlit.io/) 交互式界面
  - 聊天式 UI 展示摘要、题目和评分结果

---

## 🔹 安装

```bash
# 克隆项目
git clone <repo_url>
cd healthbot

# 安装依赖
pip install -r requirements.txt

# 配置环境变量 (.env)
AZURE_OPENAI_CHAT_DEPLOYMENT_NAME=<your_deployment_name>
AZURE_OPENAI_API_VERSION=<api_version>