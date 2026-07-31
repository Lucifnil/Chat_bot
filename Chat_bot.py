import datetime
import re
import sqlite3
from pathlib import Path

import streamlit as st

from module_chat import chat_stream, compress_history, get_local_models, strip_think


st.set_page_config(page_title='能工智人', page_icon='🤖', layout='wide')

WELCOME_MESSAGE = '参见老大，有何吩咐？'
RECENT_MESSAGE_COUNT = 12
COMPRESS_TRIGGER_COUNT = 20
DB_PATH = Path(__file__).with_name('history.db')


def connect_db() -> sqlite3.Connection:
    return sqlite3.connect(DB_PATH)


def init_db() -> None:
    with connect_db() as conn:
        conn.execute(
            '''
            CREATE TABLE IF NOT EXISTS chat_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                role TEXT NOT NULL,
                full_content TEXT NOT NULL,
                clean_content TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            '''
        )
        conn.execute(
            '''
            CREATE TABLE IF NOT EXISTS chat_memory (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                summary TEXT NOT NULL,
                last_message_id INTEGER NOT NULL
            )
            '''
        )


def save_message_to_db(role: str, full_content: str, clean_content: str) -> int:
    with connect_db() as conn:
        cursor = conn.execute(
            'INSERT INTO chat_history (role, full_content, clean_content) VALUES (?, ?, ?)',
            (role, full_content, clean_content),
        )
        return int(cursor.lastrowid)


def delete_message_from_db(message_id: int) -> None:
    with connect_db() as conn:
        conn.execute('DELETE FROM chat_history WHERE id = ?', (message_id,))


def clear_db() -> None:
    with connect_db() as conn:
        conn.execute('DELETE FROM chat_history')
        conn.execute('DELETE FROM chat_memory')
        conn.execute(
            'INSERT INTO chat_history (role, full_content, clean_content) VALUES (?, ?, ?)',
            ('assistant', WELCOME_MESSAGE, WELCOME_MESSAGE),
        )


def load_history_from_db() -> tuple[list[dict], list[dict]]:
    with connect_db() as conn:
        rows = conn.execute(
            'SELECT id, role, full_content, clean_content FROM chat_history ORDER BY id'
        ).fetchall()
        memory = conn.execute(
            'SELECT summary, last_message_id FROM chat_memory WHERE id = 1'
        ).fetchone()

    display_messages = [
        {'role': role, 'content': clean_content}
        for _, role, _, clean_content in rows
    ]
    if memory:
        summary, last_message_id = memory
        messages = [{'role': 'system', 'content': f'早期对话记忆：{summary}'}]
        rows = [row for row in rows if row[0] > last_message_id]
    else:
        messages = []
    messages.extend({'role': role, 'content': clean_content} for _, role, _, clean_content in rows)
    return messages, display_messages


def load_export_messages() -> list[dict]:
    with connect_db() as conn:
        rows = conn.execute(
            'SELECT role, clean_content FROM chat_history ORDER BY id'
        ).fetchall()
    return [{'role': role, 'content': content} for role, content in rows]


def compress_and_update_db(selected_model: str) -> bool:
    with connect_db() as conn:
        memory = conn.execute(
            'SELECT summary, last_message_id FROM chat_memory WHERE id = 1'
        ).fetchone()
        last_message_id = memory[1] if memory else 0
        source_rows = conn.execute(
            '''
            SELECT id, role, clean_content FROM chat_history
            WHERE id > ? ORDER BY id
            ''',
            (last_message_id,),
        ).fetchall()

    if len(source_rows) <= RECENT_MESSAGE_COUNT:
        return False

    old_rows = source_rows[:-RECENT_MESSAGE_COUNT]
    old_messages = []
    if memory:
        old_messages.append({'role': 'system', 'content': f'早期对话记忆：{memory[0]}'})
    old_messages.extend(
        {'role': role, 'content': clean_content}
        for _, role, clean_content in old_rows
    )
    summary = compress_history(old_messages, selected_model)
    if not summary:
        return False

    with connect_db() as conn:
        conn.execute(
            '''
            INSERT INTO chat_memory (id, summary, last_message_id) VALUES (1, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                summary = excluded.summary,
                last_message_id = excluded.last_message_id
            ''',
            (summary, old_rows[-1][0]),
        )
    return True


def export_history_to_markdown(messages: list[dict]) -> str:
    role_names = {'user': '用户', 'assistant': '助手', 'system': '系统记忆'}
    content = '# 能工智人 对话记录\n\n'
    for message in messages:
        role_name = role_names.get(message['role'], message['role'])
        content += f"### {role_name}\n\n{message['content']}\n\n"
    return content


def parse_markdown_to_messages(markdown_text: str) -> list[dict]:
    pattern = r'### (用户|助手)\n\n(.*?)(?=\n### |\Z)'
    return [
        {
            'role': 'user' if role_name == '用户' else 'assistant',
            'content': content.strip(),
        }
        for role_name, content in re.findall(pattern, markdown_text, re.DOTALL)
        if content.strip()
    ]


def import_messages(messages: list[dict], replace: bool) -> None:
    with connect_db() as conn:
        if replace:
            conn.execute('DELETE FROM chat_history')
            conn.execute('DELETE FROM chat_memory')
            conn.execute(
                'INSERT INTO chat_history (role, full_content, clean_content) VALUES (?, ?, ?)',
                ('assistant', WELCOME_MESSAGE, WELCOME_MESSAGE),
            )
        for message in messages:
            conn.execute(
                'INSERT INTO chat_history (role, full_content, clean_content) VALUES (?, ?, ?)',
                (message['role'], message['content'], message['content']),
            )
        conn.execute('DELETE FROM chat_memory')


init_db()
if 'messages' not in st.session_state:
    messages, display_messages = load_history_from_db()
    if not messages:
        clear_db()
        messages, display_messages = load_history_from_db()
    st.session_state['messages'] = messages
    st.session_state['display_messages'] = display_messages

with st.sidebar:
    st.header('模型设置')
    selected_model = st.selectbox('选择模型', get_local_models())
    temperature = st.slider('回答随机性', 0.0, 1.5, 0.7, 0.1)
    num_predict = st.slider('最大生成长度', 256, 4096, 2048, 256)

st.title('能工智人')
for message in st.session_state['display_messages']:
    st.chat_message(message['role']).write(message['content'])

if prompt := st.chat_input('请输入你的问题'):
    prompt = prompt.strip()
    if prompt:
        st.session_state['messages'].append({'role': 'user', 'content': prompt})
        st.session_state['display_messages'].append({'role': 'user', 'content': prompt})
        user_message_id = save_message_to_db('user', prompt, prompt)

        with st.chat_message('user'):
            st.write(prompt)

        with st.chat_message('assistant'):
            try:
                response = st.write_stream(
                    chat_stream(
                        st.session_state['messages'],
                        selected_model,
                        temperature,
                        num_predict,
                    )
                )
            except RuntimeError as error:
                st.error(str(error))
                response = None

        if isinstance(response, str) and response.strip():
            clean_response = strip_think(response)
            st.session_state['display_messages'].append(
                {'role': 'assistant', 'content': response}
            )
            st.session_state['messages'].append(
                {'role': 'assistant', 'content': clean_response}
            )
            save_message_to_db('assistant', clean_response, clean_response)
            if len(st.session_state['messages']) > COMPRESS_TRIGGER_COUNT:
                if compress_and_update_db(selected_model):
                    st.session_state['messages'], _ = load_history_from_db()
        else:
            if response == '':
                st.warning('模型没有返回内容，请重新提问。')
            delete_message_from_db(user_message_id)
            st.session_state['messages'].pop()
            st.session_state['display_messages'].pop()

with st.sidebar:
    st.header('对话管理')
    export_messages = load_export_messages()
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    st.download_button(
        '导出对话记录',
        data=export_history_to_markdown(export_messages),
        file_name=f'chat_history_{timestamp}.md',
        mime='text/markdown',
    )

    uploaded_file = st.file_uploader('导入对话记录（Markdown）', type=['md'])
    if uploaded_file is not None:
        imported_messages = parse_markdown_to_messages(uploaded_file.getvalue().decode('utf-8'))
        if imported_messages:
            st.caption(f'已解析 {len(imported_messages)} 条消息')
            has_existing_conversation = len(load_export_messages()) > 1
            import_mode = st.radio(
                '导入方式',
                ('合并', '覆盖') if has_existing_conversation else ('覆盖',),
                horizontal=True,
            )
            if st.button('确认导入'):
                import_messages(imported_messages, replace=import_mode == '覆盖')
                st.session_state['messages'], st.session_state['display_messages'] = load_history_from_db()
                st.rerun()
        else:
            st.warning('未找到可导入的用户或助手消息。')

    if 'confirm_clear' not in st.session_state:
        st.session_state['confirm_clear'] = False
    if st.button('清空对话'):
        st.session_state['confirm_clear'] = True
    if st.session_state['confirm_clear']:
        st.warning('确定要清空所有对话记录吗？此操作不可撤销。')
        confirm_column, cancel_column = st.columns(2)
        with confirm_column:
            if st.button('确认清空'):
                clear_db()
                st.session_state['messages'], st.session_state['display_messages'] = load_history_from_db()
                st.session_state['confirm_clear'] = False
                st.rerun()
        with cancel_column:
            if st.button('取消'):
                st.session_state['confirm_clear'] = False
                st.rerun()