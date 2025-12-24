import streamlit as st
import pandas as pd
import uuid
from datetime import date, timedelta, datetime
import os
from io import StringIO
from github import Github, GithubException
from openai import OpenAI

# ==============================================================================
# 1. 配置与初始化
# ==============================================================================

st.set_page_config(
    page_title="MemoFlow",
    page_icon="🧠",
    layout="wide"
)

# 定义数据结构标准
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
if 'current_card_id' not in st.session_state:
    st.session_state.current_card_id = None

# ==============================================================================
# 2. 核心逻辑：SM-2 算法 & GitHub 同步
# ==============================================================================

class SRSManager:
    @staticmethod
    def calculate_next_review(row, quality):
        """SM-2 算法实现"""
        reps = int(row['repetitions'])
        ef = float(row['ease_factor'])
        interval = int(row['interval'])

        if quality >= 3:
            if reps == 0:
                interval = 1
            elif reps == 1:
                interval = 6
            else:
                interval = int(interval * ef)
            reps += 1
            ef = ef + (0.1 - (5 - quality) * (0.08 + (5 - quality) * 0.02))
        else:
            reps = 0
            interval = 1
        
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
    def __init__(self, token, repo_name, file_path="data/vocab.csv"):
        self.token = token
        self.repo_name = repo_name
        self.file_path = file_path
        self.gh = Github(token)

    def pull_data(self):
        try:
            repo = self.gh.get_repo(self.repo_name)
            contents = repo.get_contents(self.file_path)
            csv_str = contents.decoded_content.decode("utf-8")
            df = pd.read_csv(StringIO(csv_str))
            for col in REQUIRED_COLUMNS:
                if col not in df.columns: df[col] = None
            return df
        except Exception as e:
            st.warning(f"GitHub 读取失败 (可能是新仓库): {e}")
            return pd.DataFrame(columns=REQUIRED_COLUMNS)

    def push_data(self, df):
        try:
            repo = self.gh.get_repo(self.repo_name)
            csv_content = df.to_csv(index=False)
            try:
                contents = repo.get_contents(self.file_path)
                repo.update_file(contents.path, f"Update: {date.today()}", csv_content, contents.sha)
                return True, "更新成功"
            except GithubException:
                repo.create_file(self.file_path, "Initial commit", csv_content)
                return True, "创建成功"
        except Exception as e:
            return False, str(e)

# ==============================================================================
# 3. LLM 服务集成 (完全从配置读取)
# ==============================================================================

def get_llm_explanation(api_key, base_url, model_name, term, context, mode):
    if not api_key:
        return "⚠️ 请配置 API Key"
    
    client = OpenAI(api_key=api_key, base_url=base_url)
    
    prompts = {
        "explain": f"""
        请简要解释术语 "{term}"。
        背景上下文：{context}
        要求：
        1. 用一句话定义。
        2. 给出一个生活中的类比。
        3. 列出3个关键特征。
        使用 Markdown 格式。
        """,
        "examples": f"""
        请为 "{term}" 生成3个例句。
        要求：难度递增（初级、中级、高级/易错场景）。
        包含中文翻译。
        """,
        "quiz": f"""
        请基于 "{term}" 出一道填空题。
        不要直接显示答案，将答案用 || 符号包裹，例如 ||Answer||。
        提供一个微小的提示。
        """
    }
    
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "你是一位专业的学习辅导老师。"},
                {"role": "user", "content": prompts[mode]}
            ],
            temperature=0.7
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"API Error: {str(e)}"

# ==============================================================================
# 4. Streamlit UI 布局
# ==============================================================================

with st.sidebar:
    st.header("⚙️ 设置")
    
    # --- 关键修改：从 Secrets 读取默认值 ---
    # 使用 st.secrets.get 安全读取，防止 key 不存在报错
    sec_gh_token = st.secrets.get("GITHUB_TOKEN", "")
    sec_api_key = st.secrets.get("LLM_API_KEY", "")
    sec_base_url = st.secrets.get("LLM_BASE_URL", "https://models.sjtu.edu.cn/api/v1")
    sec_model = st.secrets.get("LLM_MODEL", "DeepSeek-V3-685B")
    
    with st.expander("API 配置", expanded=True):
        # 即使有 Secrets，也允许用户在 UI 上临时覆盖（方便调试）
        gh_token = st.text_input("GitHub Token", value=sec_gh_token, type="password")
        repo_name = st.text_input("Repo Name", value="yourname/memo-app")
        
        st.divider()
        st.caption("LLM 服务配置")
        api_key = st.text_input("API Key", value=sec_api_key, type="password")
        base_url = st.text_input("Base URL", value=sec_base_url)
        model_name = st.text_input("Model Name", value=sec_model)

    st.divider()
    
    st.subheader("📚 数据操作")
    if st.button("🔄 同步云端数据"):
        if gh_token and repo_name:
            with st.spinner("同步中..."):
                syncer = GitHubSync(gh_token, repo_name)
                st.session_state.data = syncer.pull_data()
            st.success(f"已加载 {len(st.session_state.data)} 条数据")
        else:
            st.error("请完善 GitHub 配置")

    uploaded_file = st.file_uploader("导入 CSV", type=['csv'])
    if uploaded_file:
        new_df = pd.read_csv(uploaded_file)
        if {'term', 'definition'}.issubset(new_df.columns):
            new_df['id'] = [str(uuid.uuid4()) for _ in range(len(new_df))]
            new_df['last_review'] = ""
            new_df['next_review'] = date.today().strftime('%Y-%m-%d')
            new_df['interval'] = 0
            new_df['repetitions'] = 0
            new_df['ease_factor'] = 2.5
            new_df['status'] = 'new'
            for col in REQUIRED_COLUMNS:
                if col not in new_df.columns: new_df[col] = ""
            st.session_state.data = pd.concat([st.session_state.data, new_df[REQUIRED_COLUMNS]], ignore_index=True)
            st.success(f"已导入 {len(new_df)} 条新词")

    if st.button("☁️ 保存进度"):
        if gh_token and repo_name:
            with st.spinner("保存中..."):
                syncer = GitHubSync(gh_token, repo_name)
                success, msg = syncer.push_data(st.session_state.data)
                if success: st.toast(msg, icon="✅")
                else: st.error(msg)

# --- 主界面 ---

st.title("🧠 记忆训练场")

today_str = date.today().strftime('%Y-%m-%d')
df = st.session_state.data
valid_date_mask = df['next_review'].notna() & (df['next_review'] != "")
due_mask = valid_date_mask & (df['next_review'] <= today_str)
new_mask = df['status'] == 'new'
review_queue = df[due_mask | new_mask]
count_due = len(review_queue)

col1, col2, col3 = st.columns(3)
col1.metric("今日待复习", f"{count_due}")
col2.metric("总词条数", len(df))
status_text = "在线" if api_key else "未配置"
col3.metric("API状态", status_text)

st.divider()

if count_due > 0:
    current_index = review_queue.index[0]
    card = df.loc[current_index]
    
    with st.container(border=True):
        st.markdown(f"### 📇 {card['term']}")
        st.caption(f"状态: {card['status']} | 间隔: {card['interval']}天")
        
        with st.expander("🤖 智能助教"):
            t1, t2, t3 = st.tabs(["💡 深度解释", "📝 场景例句", "❓ 模拟测试"])
            
            # 使用 lambda 简化参数传递
            call_llm = lambda mode: get_llm_explanation(api_key, base_url, model_name, card['term'], card['context'], mode)
            
            with t1:
                if st.button("生成解释"):
                    with st.spinner("分析中..."):
                        st.markdown(call_llm("explain"))
            with t2:
                if st.button("生成例句"):
                    with st.spinner("撰写中..."):
                        st.markdown(call_llm("examples"))
            with t3:
                if st.button("生成测试"):
                    with st.spinner("出题中..."):
                        st.markdown(call_llm("quiz"))

        st.write("---")

        if not st.session_state.show_answer:
            st.button("👁️ 查看背面", on_click=lambda: st.session_state.update(show_answer=True), use_container_width=True)
        else:
            st.markdown("#### 💡 答案")
            st.info(card['definition'])
            if card['context']: st.markdown(f"**备注**: {card['context']}")
            
            st.write("---")
            c1, c2, c3 = st.columns(3)
            
            def submit_review(quality):
                new_state = SRSManager.calculate_next_review(card, quality)
                for k, v in new_state.items():
                    st.session_state.data.at[current_index, k] = v
                st.session_state.show_answer = False
                st.toast("已更新记忆曲线")
            
            with c1: st.button("🔴 忘记", on_click=submit_review, args=(0,), use_container_width=True)
            with c2: st.button("🟡 模糊", on_click=submit_review, args=(3,), use_container_width=True)
            with c3: st.button("🟢 掌握", on_click=submit_review, args=(5,), use_container_width=True)
else:
    st.balloons()
    st.success("🎉 今天的学习计划已完成！")
    with st.expander("📊 查看所有词条"):
        st.dataframe(st.session_state.data)