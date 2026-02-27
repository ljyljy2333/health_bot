import streamlit as st
import asyncio
import uuid
from langgraph.types import Command
from health_bot import health_bot

# --- 1. 页面配置 ---
st.set_page_config(page_title="AI 健康科普助手", page_icon="🏥", layout="wide")

# --- 2. 初始化 Session State ---
if "bot" not in st.session_state:
    st.session_state.bot = health_bot()
    # 每次重置或首次进入分配一个唯一的 thread_id
    st.session_state.thread_id = str(uuid.uuid4())
    st.session_state.messages = []

# 配置项
config = {"configurable": {"thread_id": st.session_state.thread_id}}

# --- 3. 侧边栏布局 ---
with st.sidebar:
    st.header("⚙️ 控制面板")
    st.write(f"当前会话 ID: `{st.session_state.thread_id[:8]}...`")

    # 显示后端的流程图
    if st.checkbox("显示逻辑流程图", value=False):
        try:
            st.image("healthbot.png", caption="Health Bot Workflow")
        except:
            st.warning("流程图文件不存在")
        # 实时查看后端状态（调试用）
        # 便于调用的 config
    current_config = {"configurable": {"thread_id": st.session_state.thread_id}}
    snapshot = st.session_state.bot.graph.get_state(current_config)
    st.write("🔍 后端下一步节点:", snapshot.next)
    st.divider()
    if st.button("🗑️ 清空聊天记录并重置", use_container_width=True):
        st.session_state.thread_id = str(uuid.uuid4())  # 换个新 ID，后端会自动创建新状态
        st.session_state.messages = []
        st.rerun()


# --- 4. 工具函数 ---
def add_message(role, content, title=None):
    st.session_state.messages.append({"role": role, "content": content, "title": title})


async def run_and_display(inputs_or_command):
    """通用执行函数：带加载动画和节点监控"""
    # 使用 st.status 显示 AI 思考过程
    with st.status("🚀 AI 正在处理中...", expanded=True) as status:
        async for event in st.session_state.bot.graph.astream(
                inputs_or_command,
                {"configurable": {"thread_id": st.session_state.thread_id}},
                stream_mode="updates"
        ):
            # 获取当前正在运行的节点名
            for node_name, values in event.items():
                status.write(f"✅ 节点 **{node_name}** 处理完成...")

                if node_name == "summarize":
                    add_message("assistant", values["summary"], "📖 知识摘要")
                elif node_name == "quiz":
                    add_message("assistant", values["quiz_question"], "❓ 随堂测试")
                elif node_name == "grade":

                    add_message("assistant", values["grade"], "📝 评分结果")

        status.update(label="✨ 处理完成！", state="complete", expanded=False)

    st.rerun()


# --- 5. 主界面渲染 ---
async def main():
    st.title("🏥 AI 健康科普互动助手")
    st.caption("基于 LangGraph 的医学教育工作流 - 实时响应版")
    st.divider()

    # 渲染历史聊天记录
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            if msg.get("title"):
                st.subheader(msg["title"])
            st.write(msg["content"])

    # 获取当前状态
    snapshot = st.session_state.bot.graph.get_state({"configurable": {"thread_id": st.session_state.thread_id}})
    next_steps = snapshot.next

    # --- 交互逻辑判断 ---

    # 场景 1：初始启动
    if not next_steps:
        if subject := st.chat_input("你想了解什么健康主题？"):
            add_message("user", subject)
            await run_and_display({"subject": subject, "iteration_count": 0})

    # 场景 2：有中断待处理
    else:
        current_node = next_steps[0]

        # A. 回答问题
        if current_node == "human_input":
            if user_answer := st.chat_input("请根据摘要回答问题 (输入 exit 退出)..."):
                add_message("user", user_answer)
                await run_and_display(Command(resume=user_answer))

        # B. 决策是否继续 (should_continue 挂在 grade 或其后续决策上)
        elif "should_continue" in str(snapshot.tasks[0].name) or current_node == "ask_continue":

            with st.chat_message("assistant"):
                st.info("🌟 学习已完成。您对这个结果满意吗？")
                c1, c2 = st.columns(2)
                if c1.button("✅ 学习新主题", use_container_width=True):
                    await run_and_display(Command(resume="yes"))
                if c2.button("🛑 结束本次对话", use_container_width=True):
                    await run_and_display(Command(resume="no"))

        # C. 输入新主题
        elif current_node == "collect_new_subject":
            if new_sub := st.chat_input("请输入下一个课题名称："):
                add_message("user", f"新课题：{new_sub}")
                await run_and_display(Command(resume=new_sub))


if __name__ == "__main__":
    asyncio.run(main())