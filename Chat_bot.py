import streamlit as st  # 导入 Streamlit，并使用 st 作为简写来创建网页界面。
from module_chat import chat_stream, compress_history, get_local_models, strip_think  # 导入模型调用和历史清理函数。

# 设置浏览器标签、页面图标和宽屏布局；必须放在所有其他 Streamlit 调用之前。
st.set_page_config(page_title='能工智人', page_icon='🤖', layout='wide')

# 在网页顶部显示应用标题。
st.title('能工智人')
# 欢迎语既显示在页面上，也会作为第一条助手消息保存到两份聊天记录中。
WELCOME_MESSAGE = '参见老大，有何吩咐？'
# RECENT_MESSAGE_COUNT 是压缩时不动的“近期原文”数量，数值越大，近期上下文越完整。
RECENT_MESSAGE_COUNT = 12  # 压缩后保留最近 12 条原始消息，维持近期上下文的细节。
# COMPRESS_TRIGGER_COUNT 是开始压缩的门槛，避免每产生一两条消息就额外请求模型做总结。
COMPRESS_TRIGGER_COUNT = 20  # 消息超过 20 条时，将较早内容压缩为一条记忆。

# 在侧边栏选择模型和生成参数，避免每次换模型或调节回答风格都修改源代码。
with st.sidebar:
    # sidebar 会在网页左侧创建独立区域，里面的控件值会在每次页面重跑时重新取得。
    st.header('模型设置')
    # 从 Ollama 查询已安装模型；查询失败时函数会返回默认模型，保证页面仍可打开。
    local_models = get_local_models()
    # selectbox 返回用户当前选择的模型名称，之后会传给 chat_stream 和 compress_history。
    selected_model = st.selectbox('选择模型', local_models)
    # slider 返回浮点数。temperature 越低答案越稳定，越高则表达更有变化。
    temperature = st.slider('回答随机性', min_value=0.0, max_value=1.5, value=0.7, step=0.1)
    # 推理模型会将思考过程也计入长度；较大的默认值能减少“只输出思考、没有答案”。
    # num_predict 是单轮最多生成的 token 数，不是汉字或字节数。
    num_predict = st.slider('最大生成长度', min_value=256, max_value=4096, value=2048, step=256)

# session_state 是 Streamlit 提供的会话存储空间。
# Streamlit 每次交互都会重新运行脚本，所以聊天记录要放在这里才能保留。
if 'messages' not in st.session_state:
    # 第一次打开网页时，创建 messages 列表，并先放入一条助手欢迎语。
    # role 表示消息角色：assistant 是助手，user 是用户。
    # content 是实际显示和发送给模型的文字内容。
    st.session_state['messages'] = [
        {'role': 'assistant', 'content': WELCOME_MESSAGE}
    ]

# display_messages 专门保存页面显示的完整内容，其中包含推理过程。
# messages 只保存发送给模型的干净内容，避免推理过程影响下一轮回答。
# 需要两份列表，是因为“页面展示内容”和“模型下一轮应看到的内容”并不相同。
if 'display_messages' not in st.session_state:
    # copy() 创建新的列表，避免两个 session_state 键指向同一个列表对象。
    st.session_state['display_messages'] = st.session_state['messages'].copy()

# 按照保存顺序，显示本次会话中的完整聊天记录。
for msg in st.session_state['display_messages']:
    # st.chat_message 会根据 role 显示用户或助手样式的聊天气泡。
    st.chat_message(msg['role']).write(msg['content'])

# 在页面底部显示输入框；用户按下回车发送后，prompt 会得到输入的文字。
if prompt := st.chat_input('请输入你的问题'):
    # 删除输入文字两侧的空格和换行，防止仅输入空格时也发送请求。
    prompt = prompt.strip()
    if prompt:
        # 把用户问题添加到历史列表。之后本轮请求会携带这条问题。
        st.session_state['messages'].append({'role': 'user', 'content': prompt})
        # 同时添加到显示列表，保证刷新后用户问题仍会显示。
        st.session_state['display_messages'].append({'role': 'user', 'content': prompt})
        # 记录“加入本轮用户问题后”的列表长度，失败时据此精确撤销这一条消息。
        model_message_count = len(st.session_state['messages'])

        # 立即在当前页面显示用户刚发送的问题，不必等下次页面刷新。
        with st.chat_message('user'):
            st.write(prompt)

        # 创建一个助手消息气泡，用来显示模型正在生成的内容。
        with st.chat_message('assistant'):
            # chat_stream 会不断返回模型新生成的文字；write_stream 会实时追加到页面。
            # res 是模型本轮最终生成的完整文本，其中仍包含 <think> 推理过程。
            try:
                # 将四个参数传入：干净历史、当前模型、随机性和最大生成长度。
                # write_stream 会遍历生成器，并把 yield 出来的每个文本片段追加到聊天气泡中。
                res = st.write_stream(
                    chat_stream(
                        st.session_state['messages'],
                        selected_model,
                        temperature,
                        num_predict
                    )
                )
            except RuntimeError as error:
                # 调用失败时只在页面显示错误，不把错误保存到聊天历史中。
                st.error(str(error))
                # 移除本轮刚加入的用户问题，防止下一轮出现连续两条 user 消息。
                # [:model_message_count - 1] 会保留本轮用户问题加入之前的所有消息。
                st.session_state['messages'] = st.session_state['messages'][:model_message_count - 1]
                res = None

        # 只有成功生成回复时才保存；避免错误信息成为下一轮提示词的一部分。
        if isinstance(res, str) and res.strip():
            # display_messages 保存完整回复，刷新页面后仍可以看到推理过程。
            st.session_state['display_messages'].append(
                {'role': 'assistant', 'content': res}
            )
            # messages 保存清理后的正式回答，供下一轮模型调用使用。
            st.session_state['messages'].append(
                {'role': 'assistant', 'content': strip_think(res)}
            )

            # 历史过长时压缩早期对话：保留欢迎语、已有记忆和最近消息。
            # 这样不会简单删除旧内容，同时也不会让发送给模型的上下文无限增长。
            if len(st.session_state['messages']) > COMPRESS_TRIGGER_COUNT:
                # 切片 [-RECENT_MESSAGE_COUNT:] 取最后 12 条，保留正在讨论的话题细节。
                recent_messages = st.session_state['messages'][-RECENT_MESSAGE_COUNT:]
                # 其余较早消息交给模型总结；原消息只有在总结成功后才会被替换。
                old_messages = st.session_state['messages'][:-RECENT_MESSAGE_COUNT]
                summary = compress_history(old_messages, selected_model)
                if summary:
                    # system 角色告诉模型这是长期记忆，不是用户刚刚提出的新问题。
                    st.session_state['messages'] = [
                        {'role': 'system', 'content': f'早期对话记忆：{summary}'}
                    ] + recent_messages
        elif res == '':
            # 模型正常结束却没有输出内容时，撤销本轮用户问题，避免留下不完整上下文。
            st.warning('模型没有返回内容，请重新提问。')
            # 此处与异常分支相同，恢复到发送本轮问题之前的干净历史。
            st.session_state['messages'] = st.session_state['messages'][:model_message_count - 1]

# 点击按钮后，同时清除页面显示记录和发送给模型的对话记录。
if st.button('清空对话'):
    # 两个列表都必须重置：一个负责模型记忆，一个负责页面展示。
    st.session_state['messages'] = [
        {'role': 'assistant', 'content': WELCOME_MESSAGE}
    ]
    st.session_state['display_messages'] = [
        {'role': 'assistant', 'content': WELCOME_MESSAGE}
    ]
    # 立即重新运行脚本，使旧聊天气泡从页面中消失并显示新的欢迎语。
    st.rerun()