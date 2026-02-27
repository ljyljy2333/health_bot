import functools

from dotenv import load_dotenv, find_dotenv
from langchain import hub
from langgraph.prebuilt import ToolNode


import os
from langchain_openai import AzureChatOpenAI

# 导入基本消息类、用户消息类和工具消息类
from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    ToolMessage,
    AIMessage
)
# 导入聊天提示模板和消息占位符
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
# 导入状态图相关的常量和类
from langgraph.graph import END, StateGraph, START
# 导入操作符和类型注解
import operator
from typing import Annotated, Sequence, TypedDict, List
from langchain_community.tools.tavily_search import TavilySearchResults

import asyncio
from typing import Literal
# 导入过滤器

"""
执行Agent的执行器, 输入为字典形式的消息，输出为字典形式的消息。主要功能：
1. 加载环境变量
2. 创建AzureChatOpenAI实例，并绑定TavilySearchResults工具
3. 创建聊天提示模板，并绑定到AzureChatOpenAI实例
4. 定义AgentState类型
5. 定义Agent节点函数，用于处理消息
6. 定义路由器，用于选择下一个节点
7. 定义状态图，并编译
8. 调用状态图的ainvoke方法，执行Agent的执行
9. 保存状态图的图片到文件
10. 异步执行状态图，并打印输出结果
:param messages_dict: 输入的消息字典
:return: 输出的消息字典

"""

_ = load_dotenv(find_dotenv(), verbose=True,override=True)
# 创建TavilySearchResults工具，设置最大结果数为1
tools = [TavilySearchResults(max_results=2)]
# 定义AgentState类型
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]

class ResearcherAgent:
    def __init__(self):
        self.llm = AzureChatOpenAI(
            azure_deployment=os.environ["AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"],
            api_version=os.environ["AZURE_OPENAI_API_VERSION"],
            temperature=0
        )

        self.tool_node = ToolNode(tools)  # 自动找到调用哪个工具，传入生成的参数

        self.graph = self._build_graph()
        graph_png = self.graph.get_graph().draw_mermaid_png()
        with open("csv_searcher.png", "wb") as f:
            f.write(graph_png)

    # 定义一个工具函数，用于缩短工具消息的内容
    def _shrink_tool_message(self,msg: ToolMessage) -> ToolMessage:
        return ToolMessage(
            tool_call_id=msg.tool_call_id,
            name=msg.name,
            content="tool result omitted, deleted internally"
        )

    # 定义一个工具函数，用于过替换 ToolMessage 类型的消息
    def _filter_state(self,state: AgentState):
        last_msg = state["messages"][-1]

        msgs = []
        for m in state["messages"][:-1]:
            if isinstance(m, ToolMessage):
                msgs.append(self._shrink_tool_message(m))
            else:
                msgs.append(m)

        return {
            "messages": msgs + [last_msg],
        }

    # --- 节点函数 ---
    #节点模版
    async def agent_node(self,state: AgentState,agent):
        # 判断是否调用工具
        new_state = self._filter_state(state)

        result = await agent.ainvoke({"messages": new_state["messages"]})  # messgeplaceholder

        result = AIMessage(**result.dict(exclude={"type", "name"}))
        return {
            "messages": [result],
        }
    # 定义搜索节点函数
    async def research_node(self, state: AgentState):

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system",
                 "You are a helpful assistant.You must refer to the result of toolcall. If there is no result of toolcall, call tools."),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )

        agent = prompt | self.llm.bind_tools(tools)

        # 👇 直接调用 agent_node
        return await self.agent_node(state, agent=agent)
    def router(self,state: AgentState) -> Literal["call_tool", "__end__"]:
        # 这是路由器
        messages = state["messages"]
        last_message = messages[-1]
        # 检查 last_message 是否包含工具调用（tool calls）
        if last_message.tool_calls:
            return "call_tool"
        else:
            return "__end__"
    def _build_graph(self):
        workflow = StateGraph(AgentState)
        workflow.add_node("researcher", self.research_node)
        workflow.add_node("call_tool", self.tool_node)

        workflow.add_edge(START, "researcher")
        workflow.add_edge("call_tool", "researcher")

        workflow.add_conditional_edges(
            "researcher",
            self.router,
            {
                "call_tool": "call_tool",
                "__end__": END,
            }
        )

        return workflow.compile()

    def run(self, messages_dict: dict):
        return self.graph.ainvoke(messages_dict,
        {"recursion_limit": 50})
# # 创建Agent实例
# agent = ResearcherAgent()
#
# all_messages = await graph.ainvoke(
#     messages_dict,
#     {"recursion_limit": 50},
# )
#
# print(all_messages)
# print("ToolMessage:"+all_messages["messages"][-2].content)
# return all_messages
