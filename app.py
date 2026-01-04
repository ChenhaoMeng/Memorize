import streamlit as st
import pandas as pd
import uuid
import json
import re
import csv
from datetime import date, timedelta
from io import StringIO
from github import Github, GithubException
from openai import OpenAI

# ==============================================================================
# 1. 配置与初始化
# ==============================================================================

st.set_page_config(
    page_title="MemoFlow - AI制卡版",
    page_icon="⚡",
    layout="wide"
)

# 10列标准数据结构
REQUIRED_COLUMNS = [
    'id', 'term', 'definition', 'context', 
    'last_review', 'next_review', 'interval', 
    'repetitions', 'ease_factor', 'status'
]

# 初始化 Session State
if 'data' not in st.session_state:
    st.session_state.data = pd.DataFrame(columns=REQUIRED_COLUMNS)
if 'show_answer' not in st.session_state:
    st.session_state.show_answer = False
if 'current_book' not in st.session_state:
    st.session_state.current_book = None 
if 'book_list' not in st.session_state:
    st.session_state.book_list = []

# ==============================================================================
# 2. 核心逻辑：SM-2 算法 & GitHub 同步
# ==============================================================================

class SRSManager:
    @staticmethod
    def calculate_next_review(row, quality):
        """SM-2 算法 (增加严格的类型校验)"""
        try:
            reps = int(float(row.get('repetitions', 0)))
            ef = float(row.get('ease_factor', 2.5))
            interval = int(float(row.get('interval', 0)))
        except (ValueError, TypeError):
            reps, ef, interval = 0, 2.5, 0

        if quality >= 3:
            if reps == 0: interval = 1
            elif reps == 1: interval = 6
            else: interval = int(interval * ef)
            reps += 1
            ef = ef + (0.1 - (5 - quality) * (0.08 + (5 - quality) * 0.02))
        else:
            reps = 0
            interval = 1
        
        if ef < 1.3: ef = 1.3

        return {
            'last_review': date.today().strftime('%Y-%m-%d'),
            'next_review': (date.today() + timedelta(days=interval)).strftime('%Y-%m-%d'),
            'interval': interval,
            'repetitions': reps,
            'ease_factor': round(ef, 2),
            'status': 'learning' if reps < 3 else 'review'
        }

class GitHubSync:
    def __init__(self, token, repo_name):
        self.token = token
        self.repo_name = repo_name
        self.gh = Github(token)
        self.data_dir = "data" 

    def get_repo(self):
        return self.gh.get_repo(self.repo_name)

    def list_books(self):
        try:
            repo = self.get_repo()
            contents = repo.get_contents(self.data_dir)
            return [f.name for f in contents if f.name.endswith(".csv")]
        except Exception: return []

    def pull_data(self, filename):
        try:
            repo = self.get_repo()
            path = f"{self.data_dir}/{filename}"
            contents = repo.get_contents(path)
            csv_str = contents.decoded_content.decode("utf-8")
            df = pd.read_csv(StringIO(csv_str), quoting=csv.QUOTE_MINIMAL)
            
            # --- 关键修复：先补齐列，再清洗数据 ---
            for col in REQUIRED_COLUMNS:
                if col not in df.columns:
                    if col in ['interval', 'repetitions']: df[col] = 0
                    elif col == 'ease_factor': df[col] = 2.5
                    else: df[col] = ""
            
            # 强制类型转换，防止 NaN 导致的 ValueError
            df['repetitions'] = pd.to_numeric(df['repetitions'], errors='coerce').fillna(0).astype(int)
            df['interval'] = pd.to_numeric(df['interval'], errors='coerce').fillna(0).astype(int)
            df['ease_factor'] = pd.to_numeric(df['ease_factor'], errors='coerce').fillna(2.5).astype(float)
            df['next_review'] = df['next_review'].replace("", date.today().strftime('%Y-%m-%d'))
            
            return df[REQUIRED_COLUMNS] # 保证列顺序一致
        except Exception as e:
            st.error(f"加载失败: {e}")
            return pd.DataFrame(columns=REQUIRED_COLUMNS)

    def push_data(self, df, filename):
        try:
            repo = self.get_repo()
            path = f"{self.data_dir}/{filename}"
            # 使用 QUOTE_NONNUMERIC 确保带逗号的内容被引号包裹
            csv_content = df.to_csv(index=False, quoting=csv.QUOTE_NONNUMERIC)
            try:
                contents = repo.get_contents(path)
                repo.update_file(contents.path, f"Update {filename}", csv_content, contents.sha)
            except GithubException:
                repo.create_file(path, f"Create {filename}", csv_content)
            return True, "同步成功"
        except Exception as e: return False, str(e)

# ==============================================================================
# 3. LLM 服务
# ==============================================================================

def stream_llm_explanation(api_key, base_url, model_name, term, context, mode, placeholder):
    if not api_key:
        placeholder.error("⚠️ 请配置 API Key")
        return
    client = OpenAI(api_key=api_key, base_url=base_url)
    prompts = {
        "explain": f"请简要解释术语 '{term}'。背景：{context}。要求：1.一句话定义。2.生活类比。3.三个关键点。Markdown格式。",
        "examples": f"请为 '{term}' 生成3个例句（初级/中级/高级），包含中文翻译。",
        "quiz": f"基于 '{term}' 出一道填空题，答案用 || 包裹。"
    }
    try:
        stream = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompts[mode]}],
            stream=True 
        )
        full_response = ""
        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                full_response += chunk.choices[0].delta.content
                placeholder.markdown(full_response + "▌")
        placeholder.markdown(full_response)
    except Exception as e: placeholder.error(f"API Error: {e}")

def generate_ai_card(api_key, base_url, model_name, term):
    if not api_key: return None, "⚠️ API Key 未配置"
    client = OpenAI(api_key=api_key, base_url=base_url)
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "system", "content": "你是一个词典编纂者，请输出 JSON。"},
                      {"role": "user", "content": f"分析单词: {term}，包含 definition 字段和 context(Markdown格式) 字段。"}],
            response_format={"type": "json_object"} 
        )
        return json.loads(response.choices[0].message.content), None
    except Exception as e: return None, str(e)

# ==============================================================================
# 4. Streamlit UI
# ==============================================================================

sec_gh_token = st.secrets.get("GITHUB_TOKEN", "")
sec_repo_name = st.secrets.get("GITHUB_REPO", "")
sec_api_key = st.secrets.get("LLM_API_KEY", "")
sec_base_url = st.secrets.get("LLM_BASE_URL", "https://api.deepseek.com")
sec_model = st.secrets.get("LLM_MODEL", "deepseek-chat")

syncer = None

with st.sidebar:
    st.header("🗂️ 词书管理")
    with st.expander("🔐 仓库配置", expanded=not sec_repo_name): 
        gh_token = st.text_input("GitHub Token", value=sec_gh_token, type="password")
        repo_name = st.text_input("Repo Name", value=sec_repo_name)
    
    if gh_token and repo_name:
        syncer = GitHubSync(gh_token, repo_name)

    if syncer:
        if st.button("🔄 刷新列表"): st.session_state.book_list = syncer.list_books()
        selected_book = st.selectbox("选择词书", options=st.session_state.book_list)

        if st.button("📥 加载词书", type="primary"):
            if selected_book:
                st.session_state.data = syncer.pull_data(selected_book)
                st.session_state.current_book = selected_book
                st.rerun()

        st.divider()
        tab_ai, tab_csv = st.tabs(["✨ AI制卡", "📄 CSV导入"])
        
        with tab_ai:
            ai_term = st.text_input("新单词")
            if st.button("🪄 生成"):
                res, err = generate_ai_card(sec_api_key, sec_base_url, sec_model, ai_term)
                if res:
                    new_row = {col: "" for col in REQUIRED_COLUMNS}
                    new_row.update({'id': str(uuid.uuid4()), 'term': ai_term, 
                                    'definition': res.get('definition', ''), 'context': res.get('context', ''),
                                    'next_review': date.today().strftime('%Y-%m-%d'), 'ease_factor': 2.5, 'status': 'new', 'interval':0, 'repetitions':0})
                    st.session_state.data = pd.concat([st.session_state.data, pd.DataFrame([new_row])], ignore_index=True)
                    st.success(f"已添加 {ai_term}")
        
        with tab_csv:
            up = st.file_uploader("CSV文件", type=['csv'])
            if up and st.button("确认导入"):
                new_df = pd.read_csv(up)
                new_df['id'] = [str(uuid.uuid4()) for _ in range(len(new_df))]
                st.session_state.data = pd.concat([st.session_state.data, new_df], ignore_index=True).fillna("")
                st.success("导入成功")

        if st.button("☁️ 保存进度", type="primary", use_container_width=True):
            if st.session_state.current_book:
                success, msg = syncer.push_data(st.session_state.data, st.session_state.current_book)
                if success: st.toast("同步成功")
                else: st.error(msg)

# --- 复习界面 ---
st.title("🧠 记忆训练场")

if not st.session_state.current_book:
    st.info("👈 请加载词书")
    st.stop()

df = st.session_state.data
today = date.today().strftime('%Y-%m-%d')
due_df = df[(df['next_review'] <= today) | (df['status'] == 'new')]

if not due_df.empty:
    current_index = due_df.index[0]
    card = df.loc[current_index]
    
    with st.container(border=True):
        st.subheader(f"单词: {card['term']}")
        with st.expander("助教面板"):
            res_box = st.empty()
            if st.button("💡 解释"): stream_llm_explanation(sec_api_key, sec_base_url, sec_model, card['term'], card['context'], "explain", res_box)

        if not st.session_state.show_answer:
            if st.button("👁️ 显示答案", use_container_width=True):
                st.session_state.show_answer = True
                st.rerun()
        else:
            st.info(f"**定义**: {card['definition']}")
            st.write(card['context'])
            
            c1, c2, c3 = st.columns(3)
            def submit(q):
                res = SRSManager.calculate_next_review(card, q)
                for k, v in res.items(): st.session_state.data.at[current_index, k] = v
                st.session_state.show_answer = False
                # 注意：回调内不写 rerun，回调结束后系统自动刷
            
            c1.button("🔴 忘记", on_click=submit, args=(0,), use_container_width=True)
            c2.button("🟡 模糊", on_click=submit, args=(3,), use_container_width=True)
            c3.button("🟢 掌握", on_click=submit, args=(5,), use_container_width=True)
else:
    st.balloons()
    st.success("复习完成！")
    st.dataframe(df)
