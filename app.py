import streamlit as st
import pandas as pd
import uuid
import json
import re
import csv
from datetime import date, timedelta, datetime
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

# 定义数据结构标准 (10列)
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
        """SM-2 算法 - 增加空值容错"""
        # 使用 pd.pd.isna() 检查或简单转换，确保即使是空值也能变成 0 或默认值
        try:
            reps = int(row['repetitions']) if pd.notna(row['repetitions']) else 0
            ef = float(row['ease_factor']) if pd.notna(row['ease_factor']) else 2.5
            interval = int(row['interval']) if pd.notna(row['interval']) else 0
        except (ValueError, TypeError):
            reps = 0
            ef = 2.5
            interval = 0
            
        if ef < 1.3: ef = 1.3

        next_date = date.today() + timedelta(days=interval)
        return {
            'last_review': date.today().strftime('%Y-%m-%d'),
            'next_review': next_date.strftime('%Y-%m-%d'),
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
            books = [f.name for f in contents if f.name.endswith(".csv")]
            return books
        except Exception:
            return []

    def pull_data(self, filename):
        try:
            repo = self.get_repo()
            path = f"{self.data_dir}/{filename}"
            contents = repo.get_contents(path)
            csv_str = contents.decoded_content.decode("utf-8")
            df = pd.read_csv(StringIO(csv_str), quoting=csv.QUOTE_MINIMAL)
            
            # --- 新增：填充数值列的空值，防止 int() 转换失败 ---
            num_cols = ['repetitions', 'interval']
            df[num_cols] = df[num_cols].fillna(0).astype(int)
            df['ease_factor'] = df['ease_factor'].fillna(2.5).astype(float)
            df['next_review'] = df['next_review'].fillna(date.today().strftime('%Y-%m-%d'))
            # ----------------------------------------------

            for col in REQUIRED_COLUMNS:
                if col not in df.columns: df[col] = None
            return df
        except Exception as e:
            st.error(f"读取失败: {e}")
            return pd.DataFrame(columns=REQUIRED_COLUMNS)
    def push_data(self, df, filename):
        try:
            repo = self.get_repo()
            path = f"{self.data_dir}/{filename}"
            
            # 💡 关键：强制对所有非数值字段使用双引号，防止 context 里的逗号导致列数错误
            csv_content = df.to_csv(index=False, quoting=csv.QUOTE_NONNUMERIC)
            
            try:
                contents = repo.get_contents(path)
                repo.update_file(contents.path, f"Update {filename}", csv_content, contents.sha)
                return True, "同步成功"
            except GithubException:
                repo.create_file(path, f"Create {filename}", csv_content)
                return True, "创建并同步成功"
        except Exception as e:
            return False, str(e)

# ==============================================================================
# 3. LLM 服务集成 (流式输出 + 结构化生成)
# ==============================================================================

def stream_llm_explanation(api_key, base_url, model_name, term, context, mode, placeholder):
    """
    流式生成解释，直接更新 UI。
    """
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
            temperature=0.7,
            stream=True 
        )
        
        full_response = ""
        for chunk in stream:
            if chunk.choices:
                content = chunk.choices[0].delta.content
                if content:
                    full_response += content
                    placeholder.markdown(full_response + "▌")
        
        placeholder.markdown(full_response)
        
    except Exception as e:
        placeholder.error(f"API Error: {str(e)}")

def generate_ai_card(api_key, base_url, model_name, term):
    if not api_key: return None, "⚠️ API Key 未配置"
    client = OpenAI(api_key=api_key, base_url=base_url)

    system_prompt = "你是一位资深语言专家，请为单词生成详尽的学习卡片数据。输出为 JSON 格式。"
    user_prompt = f"""
    请分析单词："{term}"
    输出 JSON 格式（包含 definition 和 context 字段）。
    context 必须是详细的 Markdown，包含词性、搭配、例句和辨析。
    """

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.7,
            response_format={"type": "json_object"} 
        )
        data = json.loads(response.choices[0].message.content)
        return data, None
    except Exception as e:
        return None, str(e)

# ==============================================================================
# 4. Streamlit UI 布局
# ==============================================================================

# 读取 Secrets
sec_gh_token = st.secrets.get("GITHUB_TOKEN", "")
sec_repo_name = st.secrets.get("GITHUB_REPO", "")
sec_api_key = st.secrets.get("LLM_API_KEY", "")
sec_base_url = st.secrets.get("LLM_BASE_URL", "https://api.deepseek.com") # 默认地址
sec_model = st.secrets.get("LLM_MODEL", "deepseek-chat")

syncer = None

with st.sidebar:
    st.header("🗂️ 词书管理")
    
    with st.expander("🔐 仓库配置", expanded=not sec_repo_name): 
        gh_token = st.text_input("GitHub Token", value=sec_gh_token, type="password")
        repo_name = st.text_input("Repo Name", value=sec_repo_name, placeholder="username/repo")
    
    if gh_token and repo_name:
        if "/" not in repo_name:
            st.error("格式：用户名/仓库名")
        else:
            syncer = GitHubSync(gh_token, repo_name)

    st.divider()

    if syncer:
        if st.button("🔄 刷新词书列表"):
            st.session_state.book_list = syncer.list_books()
        
        book_options = st.session_state.book_list
        selected_book = st.selectbox("选择当前词书", options=book_options, index=0 if book_options else None)

        if st.button("📥 加载选中词书", type="primary"):
            if selected_book:
                st.session_state.data = syncer.pull_data(selected_book)
                st.session_state.current_book = selected_book
                st.success(f"已加载: {selected_book}")

        st.divider()
        st.subheader("➕ 生产力工具")
        tab_ai, tab_csv, tab_new = st.tabs(["✨ AI制卡", "📄 CSV追加", "🆕 建新书"])
        
        with tab_ai:
            if st.session_state.current_book:
                ai_term = st.text_input("输入要制作的词")
                if st.button("🪄 生成并添加"):
                    with st.spinner("AI 正在思考..."):
                        res, err = generate_ai_card(sec_api_key, sec_base_url, sec_model, ai_term)
                        if res:
                            new_row = {
                                'id': str(uuid.uuid4()), 'term': ai_term,
                                'definition': res.get('definition', ''), 'context': res.get('context', ''),
                                'last_review': '', 'next_review': date.today().strftime('%Y-%m-%d'),
                                'interval': 0, 'repetitions': 0, 'ease_factor': 2.5, 'status': 'new'
                            }
                            st.session_state.data = pd.concat([st.session_state.data, pd.DataFrame([new_row])], ignore_index=True)
                            st.success(f"已添加: {ai_term}")
                        else: st.error(err)
            else: st.warning("请先加载词书")

        with tab_csv:
            uploaded_file = st.file_uploader("导入CSV (term,definition,context)", type=['csv'])
            if uploaded_file and st.session_state.current_book:
                if st.button("确认CSV追加"):
                    new_df = pd.read_csv(uploaded_file)
                    new_df['id'] = [str(uuid.uuid4()) for _ in range(len(new_df))]
                    new_df['next_review'] = date.today().strftime('%Y-%m-%d')
                    new_df['status'] = 'new'
                    for col in REQUIRED_COLUMNS:
                        if col not in new_df.columns: new_df[col] = ""
                    st.session_state.data = pd.concat([st.session_state.data, new_df[REQUIRED_COLUMNS]], ignore_index=True)
                    st.success("CSV 追加成功")

        if st.button("☁️ 保存当前词书进度", type="primary"):
            if st.session_state.current_book:
                success, msg = syncer.push_data(st.session_state.data, st.session_state.current_book)
                if success: st.toast("同步成功", icon="✅")
                else: st.error(msg)

# --- 复习界面 ---
st.title("🧠 记忆训练场")

if not st.session_state.current_book:
    st.info("👈 请在左侧选择或新建一个词书开始学习")
    st.stop()

# 计算待复习队列
df = st.session_state.data
df['next_review'] = df['next_review'].fillna(date.today().strftime('%Y-%m-%d'))
today_str = date.today().strftime('%Y-%m-%d')
due_mask = (df['next_review'] <= today_str) | (df['status'] == 'new')
review_queue = df[due_mask]

col1, col2, col3 = st.columns(3)
col1.metric("今日待复习", len(review_queue))
col2.metric("当前词书", st.session_state.current_book)
col3.metric("总记录", len(df))

st.divider()

if len(review_queue) > 0:
    current_index = review_queue.index[0]
    card = df.loc[current_index]
    
    with st.container(border=True):
        st.markdown(f"### 📇 {card['term']}")
        
        with st.expander("🤖 助教面板"):
            t1, t2, t3 = st.tabs(["💡 解释", "📝 例句", "❓ 测试"])
            with t1:
                if st.button("生成详细解释"):
                    res_box = st.empty()
                    stream_llm_explanation(sec_api_key, sec_base_url, sec_model, card['term'], card['context'], "explain", res_box)
            with t2:
                if st.button("生成更多例句"):
                    res_box = st.empty()
                    stream_llm_explanation(sec_api_key, sec_base_url, sec_model, card['term'], card['context'], "examples", res_box)
            with t3:
                if st.button("即兴小测验"):
                    res_box = st.empty()
                    stream_llm_explanation(sec_api_key, sec_base_url, sec_model, card['term'], card['context'], "quiz", res_box)
        
        st.write("---")

        if not st.session_state.show_answer:
            if st.button("👁️ 显示答案", use_container_width=True, type="primary"):
                st.session_state.show_answer = True
                st.rerun()
        else:
            st.success(f"**定义**：{card['definition']}")
            if card['context']: st.info(f"**背景/备注**：\n\n{card['context']}")
            
            c1, c2, c3 = st.columns(3)
            def submit_review(quality):
                new_state = SRSManager.calculate_next_review(card, quality)
                for k, v in new_state.items():
                    st.session_state.data.at[current_index, k] = v
                st.session_state.show_answer = False
                st.toast("进度已更新")
                st.rerun()
            
            with c1: st.button("🔴 忘记", on_click=submit_review, args=(0,), use_container_width=True)
            with c2: st.button("🟡 模糊", on_click=submit_review, args=(3,), use_container_width=True)
            with c3: st.button("🟢 掌握", on_click=submit_review, args=(5,), use_container_width=True)
else:
    st.balloons()
    st.success("🎉 太棒了！当前词书已全部复习完成！")

