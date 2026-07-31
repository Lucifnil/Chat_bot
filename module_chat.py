from collections.abc import Iterator  # Iterator 表示函数会逐段产出内容，而不是一次返回全部内容。
import re  # 导入正则表达式模块，用来查找和删除 <think> 标签中的内容。
import ollama  # 导入 Ollama 的 Python 库，用来和本机运行的大模型通信。

# 创建一个 Ollama 客户端。127.0.0.1 表示本机，11434 是 Ollama 默认服务端口。
# timeout=60 表示请求超过 60 秒未响应时停止等待，避免网页无限卡住。
new_ollama = ollama.Client(host='http://127.0.0.1:11434', timeout=60)
DEFAULT_MODEL = 'glm-4.7-flash:q4_K_M'  # 无法读取模型列表时使用的默认模型。


def strip_think(text: str) -> str:
    """从文本中删除页面显示的推理过程，用于保存干净的历史消息。"""
    # re.sub(查找规则, 替换内容, 原文本) 会把匹配的内容替换掉。
    # 【思考过程开始】和【思考过程结束】是页面中可见的推理边界标记。
    # (?:【思考过程结束】|$) 表示匹配到结束标记，或匹配到文本结尾。
    # 因此即使模型只输出了 <think> 而没有输出 </think>，推理内容也不会进入下一轮对话。
    # strip() 会去除结果开头和结尾多余的空格、换行。
    return re.sub(r'【思考过程开始】.*?(?:【思考过程结束】|$)', '', text, flags=re.DOTALL).strip()


def get_local_models() -> list[str]:
    """读取本机 Ollama 已安装的模型名称；服务不可用时返回默认模型。"""
    try:
        # list().models 中每项的 model 字段就是 Ollama 使用的完整模型名称。
        models = [model.model for model in new_ollama.list().models]
        return models or [DEFAULT_MODEL]
    except Exception:
        # 页面仍可打开，之后实际提问时再显示具体连接错误。
        return [DEFAULT_MODEL]


def chat_stream(
    messages: list[dict],
    model: str,
    temperature: float,
    num_predict: int
) -> Iterator[str]:
    """流式调用本地模型，每生成一小段文字就立即返回一小段。

    参数 messages 是不含推理过程的聊天历史；model、temperature 和 num_predict
    分别来自页面侧边栏。Iterator[str] 表示调用者会持续拿到文本片段。
    """
    try:
        # 调用 chat 方法向模型发送整个对话历史。
        # stream=True 表示不要等待完整回答生成完毕，而是逐段接收回答。
        stream = new_ollama.chat(
            model=model,  # 由侧边栏选择的本机模型名称。
            messages=messages,  # 发送给模型的历史消息列表，每项包含 role 和 content。
            stream=True,  # 开启流式输出，让网页能实时显示模型的思考和回答。
            options={
                'temperature': temperature,  # 值越高，回答越有发散性；值越低，回答越稳定。
                'num_predict': num_predict  # 限制本轮最多生成的 token 数，避免回答无限变长。
            }
        )

        # stream 是一个可遍历对象，模型每生成一段内容就会得到一个 chunk（数据块）。
        # 用于记录是否已经输出推理开始标记，避免每个数据块都重复输出标记。
        is_thinking = False
        has_content = False  # 记录本轮是否生成过正式回答，避免只有推理时被误判为成功。
        done_reason = ''  # Ollama 在最后一个数据块中给出的结束原因，例如 stop 或 length。
        for chunk in stream:
            # Ollama 的流式数据块是 ChatResponse 对象，message 中可能有两类文字：
            # thinking 是模型的推理过程，content 是最终回答。不同模型的支持情况不同。
            # 使用 getattr 安全读取属性：字段不存在时返回空字符串，而不会让程序报错。
            message = chunk.message
            thinking = getattr(message, 'thinking', '') or ''
            content = getattr(message, 'content', '') or ''

            # 使用中文标记而不是 <think> 标签：Streamlit 会把尖括号标签当作 HTML 隐藏。
            # 第一次收到推理内容时才输出开始标记，后续数据块直接追加推理文字。
            if thinking:
                if not is_thinking:
                    yield '\n\n【思考过程开始】\n\n'
                    is_thinking = True
                yield thinking

            # 模型开始输出正式回答时，先输出推理结束标记，再输出回答内容。
            # 保存历史时 strip_think 会删除完整的推理块，只保留正式回答。
            # yield 会暂停函数并把内容交出去，下次循环再继续执行。
            if content:
                if is_thinking:
                    yield '\n\n【思考过程结束】\n\n'
                    is_thinking = False
                has_content = True
                yield content

            # 只有最后一个数据块通常才会包含 done_reason，因此每次安全地更新它。
            done_reason = getattr(chunk, 'done_reason', '') or done_reason

        # 有些模型只返回推理就结束；此时补上结束标记，保证历史清理规则能正确匹配。
        if is_thinking:
            yield '\n\n【思考过程结束】'

        # num_predict 会同时限制推理和正式回答。若推理耗尽上限，明确提示用户调大长度。
        # 这里在所有数据块读取完成后判断，才能确认模型最终确实没有输出 content。
        if not has_content and done_reason == 'length':
            raise RuntimeError(
                '模型在生成正式回答前已用完最大生成长度。请在侧边栏调大“最大生成长度”后重试。'
            )
    except Exception as e:
        # 保留上方主动抛出的提示，避免被误包装成“Ollama 服务未运行”。
        if isinstance(e, RuntimeError):
            raise
        # 例如 Ollama 未启动、模型不存在或网络连接失败时，会执行这里。
        # 抛出异常交由页面显示；这样错误文字不会被误存为一条助手历史消息。
        raise RuntimeError(f"模型暂时无法响应，请检查 Ollama 服务是否运行。错误详情：{e}") from e


def compress_history(messages: list[dict], model: str) -> str:
    """将较早对话压缩成一条简短记忆，减少后续请求的上下文长度。

    参数 messages 只应传入准备被替换的旧消息；model 是当前侧边栏选择的模型。
    成功时返回摘要文本，失败时返回空字符串，由调用方继续保留旧消息。
    """
    # 把旧消息格式化成易读文本，再请模型仅保留用户偏好、问题背景和已确定结论。
    conversation = '\n'.join(
        f"{message['role']}: {message['content']}" for message in messages
    )
    # 将角色和内容拼成多行文本，例如“user: 问题”，让总结模型能区分说话人。
    prompt = (
        '请将以下早期对话压缩为简洁记忆。只保留后续回答需要的用户偏好、背景、'
        '已确认的事实和未完成事项；不要编造信息，不要写思考过程。\n\n'
        f'{conversation}'
    )
    try:
        # 这里不使用 stream=True：摘要只需要最终完整文本，不需要在页面逐字展示。
        response = new_ollama.chat(
            model=model,  # 使用当前选中的模型压缩早期聊天记录。
            messages=[{'role': 'user', 'content': prompt}]
        )
        return (response.message.content or '').strip()
    except Exception:
        # 压缩失败时返回空文本；调用方会保留原始消息，避免丢失对话内容。
        return ''