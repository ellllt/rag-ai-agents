from typing import Any
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel, Field
from constants import ROLES
from state import State
from chat_model import llm
from retriever import retriever

def selection_node(state: State) -> dict[str, Any]:
    query = state.query
    role_options = "\n".join([f"{k}. {v['name']}: {v['description']}" for k, v in ROLES.items()])
    prompt = ChatPromptTemplate.from_template(
        """
        質問を分析し、最も適切な回答担当ロールを選択してください。

        選択肢:
        {role_options}

        回答は選択肢の番号(1、2、3、4、5、または6)のみを返してください。

        質問: {query}
        """.strip()
    )
    chain = prompt | llm.with_config(configurable=dict(max_tokens=1)) | StrOutputParser()
    role_number = chain.invoke({"role_options": role_options, "query": query})

    selected_role = ROLES[role_number.strip()]["name"]
    return {"current_role": selected_role}

def answering_node(state: State) -> dict[str, Any]:
    query = state.query
    role = state.current_role
    role_details = "\n".join([f"- {v['name']}: {v['details']}" for v in ROLES.values()])
    context = retriever.invoke(query)

    prompt = ChatPromptTemplate.from_template(
        """
        あなたは{role}として回答してください。以下の質問に対して、あなたの役割に基づいた適切な回答を提供してください。
        また、以下の文脈だけを踏まえて質問に回答してください。
        資料から適切な回答が見付けられない場合は、社内ルールにないのでわからないことを伝えてください。
        わからない場合は「文脈」という単語を使用せずに回答してください。
        文脈はマークダウン形式ではなく、テキストで分かりやすく表示してください。
        また、必要に応じて改行を使い、視覚的に整理された形で出力してください。

        役割の詳細:
        {role_details}

        文脈:
        {context}

        質問: {query}

        回答:
        """.strip()
    )
    chain = prompt | llm | StrOutputParser()

    answer = chain.invoke({
        "role": role,
        "role_details": role_details,
        "context": context,
        "query": query
    })


    return {"messages": [answer]}

class Judgement(BaseModel):
    judge: bool = Field(default=False, description="判定結果")
    reason: str = Field(default="", description="判定理由")

def check_node(state: State) -> dict[str, Any]:
    query = state.query
    answer = state.messages[-1]
    prompt = ChatPromptTemplate.from_template(
        """
        以下の回答の品質をチェックし、問題がある場合は'False'、問題がない場合は'True'を回答してください。
        例外として、文脈から適切な回答が見付からないという回答の場合は'True'を回答してください。
        また、その判断理由も説明してください。

        ユーザーからの質問: {query}
        回答: {answer}
        """.strip()
    )
    chain = prompt | llm.with_structured_output(Judgement)
    result: Judgement = chain.invoke({"query": query, "answer": answer})

    return {
        "current_judge": result.judge,
        "judgement_reason": result.reason
    }

