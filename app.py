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
    page_title="MemoFlow - AI 间隔重复训练",
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
        """
        SM-2 算法实现
        Quality: 0 (Again), 3 (Hard), 5 (Good) - 简化版评分
        """
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
            # EF 更新公式
            ef = ef + (0.1 - (5 - quality) * (0.08 + (5 - quality) * 0.02))
        else:
            reps = 0
            interval = 1
            # 忘记时不减少 EF，避免陷阱
        
        if ef < 1.3:
            ef = 1.3

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
            # 确保列完整
            for col in REQUIRED_COLUMNS:
                if col not in df.columns:
                    df[col] = None
            return df
        except Exception as e:
            st.warning(f"无法从 GitHub 拉取数据 (可能是初次运行): {e}")
            return pd.DataFrame(columns=REQUIRED_COLUMNS)

    def push_data(self, df):
        try:
            repo = self.gh.get_repo(self.repo_name)
            csv_content = df.to_csv(index=False)
            
            try:
                # 尝试获取文件以更新
                contents = repo.get_contents(self.file_path)
                repo.update_file(
                    contents.path, 
                    f"Update vocab: {date.today()}", 
                    csv_content, 
                    contents.sha
                )
                return True, "更新成功"
            except GithubException:
                # 文件不存在，创建新文件
                repo.create_file(
                    self.file_path, 
                    "Initial commit via MemoFlow", 
                    csv_content
                )
                return True, "创建并保存成功"
        except Exception as e:
            return False, str(e)

# ==============================================================================
# 3. LLM 服务集成
# ==============================================================================

def get_llm_explanation(api_key, term, context, mode):
    if not api_key:
        return "⚠️ 请先在侧边栏设置 OpenAI API Key"
    
    client = OpenAI(api_key=api_key)
    
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
            model="gpt-4o-mini", # 或 gpt-3.5-turbo
            messages=[
                {"role": "system", "content": "你是一位专业的学习辅导老师。"},
                {"role": "user", "content": prompts[mode]}
            ],
            temperature=0.7
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error: {str(e)}"

# ==============================================================================
# 4. Streamlit UI 布局
# ==============================================================================

# --- Sidebar: 设置与同步 ---
with st.sidebar:
    st.title("⚙️ 控制台")
    
    # 环境变量获取（优先从 secrets 获取，否则手动输入）
    default_gh_token = st.secrets.get("GITHUB_TOKEN", "")
    default_openai_key = st.secrets.get("OPENAI_API_KEY", "")
    
    with st.expander("API 设置", expanded=not default_gh_token):
        gh_token = st.text_input("GitHub Token", value=default_gh_token, type="password")
        repo_name = st.text_input("Repo Name (user/repo)", value="yourname/memo-app")
        openai_key = st.text_input("OpenAI API Key", value=default_openai_key, type="password")

    st.divider()
    
    # 数据管理
    st.subheader("📚 数据管理")
    
    # 1. 同步按钮
    if st.button("🔄 从 GitHub 拉取数据"):
        if gh_token and repo_name:
            with st.spinner("正在拉取..."):
                syncer = GitHubSync(gh_token, repo_name)
                st.session_state.data = syncer.pull_data()
            st.success(f"已加载 {len(st.session_state.data)} 条数据")
        else:
            st.error("请配置 GitHub Token 和 Repo Name")

    # 2. 导入 CSV
    uploaded_file = st.file_uploader("追加 CSV 数据", type=['csv'])
    if uploaded_file:
        new_df = pd.read_csv(uploaded_file)
        if {'term', 'definition'}.issubset(new_df.columns):
            # 数据清洗与合并
            new_df['id'] = [str(uuid.uuid4()) for _ in range(len(new_df))]
            new_df['last_review'] = ""
            new_df['next_review'] = date.today().strftime('%Y-%m-%d')
            new_df['interval'] = 0
            new_df['repetitions'] = 0
            new_df['ease_factor'] = 2.5
            new_df['status'] = 'new'
            
            # 补齐其他列
            for col in REQUIRED_COLUMNS:
                if col not in new_df.columns:
                    new_df[col] = ""
            
            # 合并到 session state (暂时不存云端)
            st.session_state.data = pd.concat([st.session_state.data, new_df[REQUIRED_COLUMNS]], ignore_index=True)
            st.success(f"已添加 {len(new_df)} 条新词，请点击下方保存同步到云端。")
        else:
            st.error("CSV 必须包含 term 和 definition 列")

    # 3. 保存按钮
    if st.button("☁️ 保存并推送到 GitHub", type="primary"):
        if gh_token and repo_name:
            with st.spinner("正在推送..."):
                syncer = GitHubSync(gh_token, repo_name)
                success, msg = syncer.push_data(st.session_state.data)
                if success:
                    st.toast(msg, icon="✅")
                else:
                    st.error(msg)
        else:
            st.error("配置缺失")

# --- Main Area: 学习界面 ---

st.title("🧠 记忆训练场")

# 筛选今日任务
today_str = date.today().strftime('%Y-%m-%d')
df = st.session_state.data

# 逻辑：next_review <= today OR status == 'new'
# 确保 next_review 是字符串且不为空
valid_date_mask = df['next_review'].notna() & (df['next_review'] != "")
due_mask = valid_date_mask & (df['next_review'] <= today_str)
new_mask = df['status'] == 'new'

# 待复习列表
review_queue = df[due_mask | new_mask]
count_due = len(review_queue)

col_metric1, col_metric2, col_metric3 = st.columns(3)
col_metric1.metric("今日待复习", f"{count_due}", delta_color="inverse")
col_metric2.metric("总词条数", len(df))
col_metric3.metric("已掌握 (Rep>3)", len(df[df['repetitions'] > 3]))

st.divider()

if count_due > 0:
    # 取出第一张卡片
    # 注意：我们操作的是 session_state 里的 df，通过 index 定位
    current_index = review_queue.index[0]
    card = df.loc[current_index]
    
    # 学习卡片容器
    with st.container(border=True):
        # 1. 正面 (Term)
        st.markdown(f"### 📇 {card['term']}")
        st.caption(f"当前状态: {card['status']} | 连续正确: {card['repetitions']} | 下次: {card['next_review']}")
        
        # LLM 辅助工具栏
        with st.expander("🤖 AI 助教 (点击展开)"):
            tab1, tab2, tab3 = st.tabs(["深度解释", "场景例句", "主动测试"])
            
            with tab1:
                if st.button("生成解释", key="btn_expl"):
                    with st.spinner("Thinking..."):
                        st.markdown(get_llm_explanation(openai_key, card['term'], card['context'], "explain"))
            with tab2:
                if st.button("生成例句", key="btn_exmp"):
                    with st.spinner("Thinking..."):
                        st.markdown(get_llm_explanation(openai_key, card['term'], card['context'], "examples"))
            with tab3:
                if st.button("生成测试", key="btn_quiz"):
                    with st.spinner("Thinking..."):
                        st.markdown(get_llm_explanation(openai_key, card['term'], card['context'], "quiz"))

        st.write("---")

        # 2. 背面 (Definition) - 交互区
        if not st.session_state.show_answer:
            st.button("👁️ 显示答案", on_click=lambda: st.session_state.update(show_answer=True), use_container_width=True)
        else:
            st.markdown("#### 💡 答案/定义")
            st.info(card['definition'])
            if card['context']:
                st.markdown(f"**备注/上下文**: {card['context']}")
            
            st.write("")
            st.markdown("##### 请根据回忆情况评分：")
            
            # 评分按钮布局
            col_b1, col_b2, col_b3 = st.columns(3)
            
            def submit_review(quality):
                # 计算新状态
                new_state = SRSManager.calculate_next_review(card, quality)
                # 更新 DataFrame
                for k, v in new_state.items():
                    st.session_state.data.at[current_index, k] = v
                # 重置 UI 状态
                st.session_state.show_answer = False
                # 提示
                st.toast("复习记录已更新", icon="🎉")
                # 自动保存到本地 session (页面刷新不丢失，但刷新 tab 会丢失)
            
            with col_b1:
                st.button("🔴 忘记了 (Again)", on_click=submit_review, args=(0,), use_container_width=True)
            with col_b2:
                st.button("🟡 有点模糊 (Hard)", on_click=submit_review, args=(3,), use_container_width=True)
            with col_b3:
                st.button("🟢 完全掌握 (Good)", on_click=submit_review, args=(5,), use_container_width=True)

else:
    st.balloons()
    st.success("🎉 太棒了！今天的复习任务已全部完成。")
    
    with st.expander("查看所有词条数据"):
        st.dataframe(st.session_state.data)