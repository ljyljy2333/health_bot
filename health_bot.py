import operator
import asyncio
import os
import json
from typing import Annotated, List, Literal, TypedDict, Union
from dotenv import load_dotenv, find_dotenv

from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.checkpoint.memory import MemorySaver

from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt
# 导入你现有的执行器
from agent_executor import ResearcherAgent

_ = load_dotenv(find_dotenv(), verbose=True, override=True)
llm = AzureChatOpenAI(azure_deployment=os.environ["AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"],
                          api_version=os.environ["AZURE_OPENAI_API_VERSION"], temperature=0, verbose=True)
# --- 1. Pydantic Schemas (数据校验) 结构化输出模型 ---
class GradeResult(BaseModel):
    """评分结果"""
    score: str = Field(description="Grades A, B, C, D, or F")
    justification: str = Field(description="评分的详细理由")
# --- 2. 状态定义 ---
class HealthBotState(TypedDict):
    subject: str  # 用户想要学习的主题
    search_results: str  # Tavily 搜索到的原始结果
    summary: str  # 3-4 段的摘要
    quiz_question: str  # 基于摘要生成的题目
    user_answer: str  # 用户录入的答案
    grade: str  # 评分 (A/B/C...) 及理由
    continue_choice: str  # 用户是否要继续学习
    iteration_count: int  # 控制循环
    is_finished: bool  # 是否退出工作流



# --- 3. Tools (工具实现) ---



# --- 4. 核心类实现 ---
class health_bot:
    def __init__(self):
        self.llm = AzureChatOpenAI(
            azure_deployment=os.environ["AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"],
            api_version=os.environ["AZURE_OPENAI_API_VERSION"],
            temperature=0
        )
        self.graph = self._build_graph()
        graph_png = self.graph.get_graph().draw_mermaid_png()
        with open("healthbot.png", "wb") as f:
            f.write(graph_png)

    # --- 节点函数 ---
    async def search_node(self,state: HealthBotState):
        """节点 1: 使用 Tavily 搜索主题信息"""
        print(f"--- SEARCHING ABOUT {state['subject']} INFORMATION ---")
        # 构造发送给 agent_executor 的消息
        # agent_executor 内部使用了 TavilySearchResults
        query = f"Provide detailed medical/health information about {state['subject']} for patient education."
        # 调用 agent_executor
        researcherAgent = ResearcherAgent()
        response = await researcherAgent.run({"messages": [HumanMessage(content=query)]})
        # 解析 agent_executor 的响应
        # 提取 Tavily 的搜索结果（通常在最后一条 AI 消息或工具消息中）
        toolcall = response["messages"][-3].additional_kwargs
        results = response["messages"][-2].content
        return {"search_results": json.loads(results)}

    async def summarize_node(self,state: HealthBotState):
        """节点 2: 总结搜索结果 (3-4 段)"""
        print("--- GENERATING A SUMMARY ---")

        prompt = ChatPromptTemplate.from_template(
            """You are a health education assistant. Please write a 3-4 paragraph popular science summary based on the search results below.
            Requirements:
            1. Strictly only use the provided search results and do not add external knowledge.
            2. The tone is easy to understand, suitable for patients to read.

            SEARCH RESULTS：{context}
            """
        )
        chain = prompt |self.llm
        response = await chain.ainvoke({"context": state['search_results']})
        return {"summary": response.content}

    async def quiz_node(self,state: HealthBotState):
        """节点 3: 基于摘要生成测试题"""
        print("--- PRACTICE QUESTIONS ARE BEING GENERATED ---")

        prompt = ChatPromptTemplate.from_template(
            """Based on the following summary, generate a very challenging but fair question.
            Requirements:
            1. Questions must be able to be answered just by reading the summary.
            2. Don't give options, it's an open-ended question.

            Abstract content:{summary}
            """
        )
        chain = prompt | llm
        response = await chain.ainvoke({"summary": state['summary']})
        return {"quiz_question": response.content}

    # async def human_input_node(state: HealthBotState):
    #     """节点 4: 模拟用户回答 (在实际 Streamlit 中这里是 UI 交互)"""
    #     print(f"\n[question]: {state['quiz_question']}")
    #     # 模拟异步获取用户输入
    #     user_input = input("Please enter your answer (or type 'exit'): ")
    #     return {"user_answer": user_input, "is_finished": user_input.lower() == 'exit'}

    async def human_input_node(self,state: HealthBotState):
        # 这里的字符串会保存在任务状态中，前端可以通过 get_state().next 看到
        user_answer = interrupt({
            "question": state['quiz_question'],
            "instruction": "请阅读摘要并回答问题"
        })

        return {
            "user_answer": user_answer,
            "is_finished": user_answer.lower() == "exit"
        }


    async def grade_node(self,state: HealthBotState):
        """节点 5: 评分并给出理由"""
        print("--- IS BEING SCORED ---")
        prompt = ChatPromptTemplate.from_template(
            """You are an unbiased mentor. Please rate the user's responses based on the summary provided.

            References (summary): {summary}
            Question: {question}
            User answer: {answer}

            Please give a grade (A/B/C... ) and brief suggestions for improvement.
            """
        )
        # 使用结构化输出满足 Review 要求
        grader = prompt | llm.with_structured_output(GradeResult)
        result = await grader.ainvoke({
            "summary": state['summary'],
            "question": state['quiz_question'],
            "answer": state['user_answer']
        })

        full_grade = f"Grade: {result.score}\nJustification: {result.justification}"
        print(f"\n[SCORING RESULTS]:\n{full_grade}\n")
        return {"grade": full_grade}

    async def collect_subject_node(self,state: HealthBotState):
        """新节点：专门负责收集或更新学习主题"""
        # 如果是第一次运行，subject 已经由初始输入提供
        # 如果是从 grade 节点跳回来的，我们需要询问新主题

        if state.get("is_finished") is False and state.get("grade"):
            new_subject = interrupt(

                "\nIf it detects that you want to continue learning, enter a new study topic: "
            )
            return {"subject": new_subject, "grade": None, "summary": None}  # 重置状态
        return {}

    async def ask_continue_node(self, state: HealthBotState):
        if state.get("is_finished"):
            return "__end__"
        choice = interrupt("Do you want to learn a new topic? (yes/no)")
        return {"continue_choice": choice}
    # --- 4. 构建图逻辑 ---

    async def should_continue(self,state: HealthBotState) -> Literal["search", "__end__"]:
        """条件边：决定是重新开始还是结束"""
        if state["continue_choice"].lower() == "yes":
            return "collect_new_subject"

        return "__end__"

    def _build_graph(self):
        workflow = StateGraph(HealthBotState)

        # 添加节点
        workflow.add_node("search", self.search_node)
        workflow.add_node("summarize", self.summarize_node)
        workflow.add_node("quiz", self.quiz_node)
        workflow.add_node("human_input", self.human_input_node)
        workflow.add_node("grade", self.grade_node)
        workflow.add_node("ask_continue", self.ask_continue_node)

        workflow.add_node("collect_new_subject", self.collect_subject_node)

        # 构建边
        workflow.add_edge(START, "collect_new_subject")  # 从收集主题开始
        workflow.add_edge("collect_new_subject", "search")
        workflow.add_edge("search", "summarize")
        workflow.add_edge("summarize", "quiz")
        workflow.add_edge("quiz", "human_input")
        workflow.add_edge("human_input", "grade")
        workflow.add_edge("grade", "ask_continue")

        # 条件边缘：决定重启或结束
        workflow.add_conditional_edges(
            "ask_continue",
            self.should_continue,
            {
                "collect_new_subject": "collect_new_subject",
                "__end__": END
            }
        )

        return workflow.compile(checkpointer=MemorySaver())
    async def run_bot(self,initial_subject: str):
        inputs = {"subject": initial_subject, "iteration_count": 0}
        config = {"recursion_limit": 50,
                  "configurable": {"thread_id": "healthbot-session-1"}
                  }

        async for event in self.graph.astream(inputs, config=config):
            print(event)
            print(self.graph.get_state(config).next)


    # async for event in app.astream(inputs, config=config):
    #     for k, v in event.items():
    #         if k == "summarize":
    #             print("\n[MODEL SUMMARY]:\n", v["summary"])


if __name__ == '__main__':
    subject = "Diabetes Diet Management"
    health_bot=health_bot()

    asyncio.run(health_bot.run_bot(subject))
    # async def main(subject: str):
    #     async for event in health_bot.run_bot(subject):
    #         print(event)
    # asyncio.run(main(subject))
