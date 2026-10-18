def build_citation_sources(evidence_chunks: list[dict]) -> list[dict]:
    """按完整 Evidence 列表的顺序构造稳定的引用编号映射。"""

    citation_sources = []

    for index, chunk in enumerate(evidence_chunks, start=1):

        citation_id = f"[资料{index}]"

        citation_sources.append(
            {
                "citation_id": citation_id,
                "chunk_id": chunk["id"],
                "source": chunk["source"]
            }
        )

    return citation_sources


def _build_context(evidence_chunks: list[dict]) -> tuple[str, list[dict]]:
    """把 Evidence 组装成给 LLM 的上下文，同时保留引用编号到真实 Chunk 的映射。"""

    citation_sources = build_citation_sources(evidence_chunks)
    context_parts = []

    for chunk, citation_source in zip(evidence_chunks, citation_sources):

        context_parts.append(
            f"{citation_source['citation_id']}\n"
            f"来源：{chunk['source']}\n"
            f"内容：{chunk['text']}"
        )

    return "\n\n".join(context_parts), citation_sources


def build_single_evidence_prompt(
    query: str,
    evidence_chunk: dict
) -> str:
    """构造只允许使用一个 Evidence 的结论生成 Prompt。"""

    return f"""
你是一名银行业务问答助手。

请只根据下面当前这一条 Evidence 判断并回答，不要使用常识、经验或其他资料补充。

回答要求：

1. 如果当前 Evidence 能直接支持用户问题中的某一条相关结论，只输出这一条结论；不要求完整回答整个用户问题。
2. 结论不超过 60 个中文字，使用自己的话简短概括，不要复制整段原文。
3. 不要输出标题、编号、项目符号、解释、来源编号、方括号引用或其他标记。
4. 如果当前 Evidence 与用户问题没有可直接支持的结论，只输出：NO_ANSWER

当前 Evidence：
来源：{evidence_chunk['source']}
内容：{evidence_chunk['text']}

用户问题：
{query}

请直接输出一条结论或 NO_ANSWER：
""".strip()


def build_rag_prompt(
    query: str,
    evidence_chunks: list[dict]
) -> tuple[str, list[dict]]:
    """构造正式 RAG 生成 Prompt。"""

    context, citation_sources = _build_context(evidence_chunks)

    prompt = f"""
你是一名银行业务问答助手。

请只根据下面提供的参考资料回答用户问题。

回答要求：

1. 围绕用户问题回答，不要按参考资料逐份总结；有足够信息时，只输出 1～3 条结论，每条约 30～60 个中文字并单独一行。
2. 用自己的话简短概括，不要复制整段原文。
3. 不要输出标题、编号、项目符号、规则说明或占位文字。
4. 每行先写结论正文并以句号结束，随后紧接真实引用；引用必须位于行末，引用后不得再有文字。
5. 每条只引用真正支持该结论的资料，引用编号必须来自下面的参考资料。
6. 只要至少一条参考资料能够回答问题，就应根据它回答；只有所有参考资料都无法回答时，才只回答：
“根据当前提供的资料，无法确定。”

参考资料：

{context}

用户问题：
{query}

请直接输出结论行：
""".strip()

    return prompt, citation_sources


def build_citation_repair_prompt(
    query: str,
    evidence_chunks: list[dict]
) -> tuple[str, list[dict]]:
    """首轮引用校验失败后，只根据原问题与 Evidence 重新生成答案。"""

    context, citation_sources = _build_context(evidence_chunks)

    prompt = f"""
请完全忽略此前的回答，只根据下面的参考资料和用户问题重新生成答案。

回答要求：

1. 围绕用户问题回答，不要按参考资料逐份总结；有足够信息时，只输出 1～3 条结论，每条约 30～60 个中文字并单独一行。
2. 用自己的话简短概括，不要复制整段原文。
3. 不要输出标题、编号、项目符号、规则说明或占位文字。
4. 每行先写结论正文并以句号结束，随后紧接真实引用；引用必须位于行末，引用后不得再有文字。
5. 重新选择真正支持每条结论的资料编号，不沿用其他结论的引用。
6. 引用编号必须来自下面的参考资料。
7. 只要至少一条参考资料能够回答问题，就应根据它回答；只有所有参考资料都无法回答时，才只回答：
“根据当前提供的资料，无法确定。”

参考资料：

{context}

用户问题：
{query}

请直接输出结论行：
""".strip()

    return prompt, citation_sources
