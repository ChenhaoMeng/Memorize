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
    page_title="MemoFlow - 多词书版",
    page_icon="📚",
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
if 'current_book' not in st.session_state:
    st.session_state.current_book = None # 当前选中的词书文件名
if 'book_list' not in st.session_state:
    st.session_state.book_list = []      # 词书列表

# ==============================================================================
# 2. 核心逻辑：SM-2 算法 & GitHub 同步 (升级版)
# ==============================================================================

class SRSManager:
    @staticmethod
    def calculate_next_review(row, quality):
        """SM-2 算法"""
        reps = int(row['repetitions'])
        ef = float(row['ease_factor'])
        interval = int(row['interval'])

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
        self.data_dir = "data" # 统一存放在 data 目录下

    def get_repo(self):
        return self.gh.get_repo(self.repo_name)

    def list_books(self):
        """列出 data 目录下的所有 csv 文件"""
        try:
            repo = self.get_repo()
            contents = repo.get_contents(self.data_dir)
            books = [f.name for f in contents if f.name.endswith(".csv")]
            return books
        except Exception:
            return []

    def pull_data(self, filename):
        """读取指定词书"""
        try:
            repo = self.get_repo()
            path = f"{self.data_dir}/{filename}"
            contents = repo.get_contents(path)
            csv_str = contents.decoded_content.decode("utf-8")
            df = pd.read_csv(StringIO(csv_str))
            # 补全列
            for col in REQUIRED_COLUMNS:
                if col not in df.columns: df[col] = None
            return df
        except Exception as e:
            st.error(f"读取失败: {e}")
            return pd.DataFrame(columns=REQUIRED_COLUMNS)

    def push_data(self, df, filename):
        """保存指定词书"""
        try:
            repo = self.get_repo()
            path = f"{self.data_dir}/{filename}"
            csv_content = df.to_csv(index=False)
            
            try:
                contents = repo.get_contents(path)
                repo.update_file(contents.path, f"Update {filename}", csv_content, contents.sha)
                return True, "更新成功"
            except GithubException:
                # 如果文件不存在（新建词书情况）
                repo.create_file(path, f"Create {filename}", csv_content)
                return True, "创建成功"
        except Exception as e:
            return False, str(e)

# ==============================================================================
# 3. LLM 服务集成
# ==============================================================================

def get_llm_explanation(api_key, base_url, model_name, term, context, mode):
    if not api_key: return "⚠️ 请配置 API Key"
    
    client = OpenAI(api_key=api_key, base_url=base_url)
    
    prompts = {
        "explain": f"请简要解释术语 '{term}'。背景：{context}。要求：1.一句话定义。2.生活类比。3.三个关键点。Markdown格式。",
        "examples": f"请为 '{term}' 生成3个例句（初级/中级/高级），包含中文翻译。",
        "quiz": f"基于 '{term}' 出一道填空题，答案用 || 包裹。"
    }
    
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompts[mode]}],
            temperature=0.7
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"API Error: {str(e)}"

# ==============================================================================
# 4. Streamlit UI 布局
# ==============================================================================

# --- 初始化同步器 ---
sec_gh_token = st.secrets.get("GITHUB_TOKEN", "")
sec_repo = "yourname/memo-app" # 可以改为 secrets 读取
# 尝试初始化 syncer
syncer = None
if sec_gh_token:
    # 这里假设用户会在界面输入 repo_name，或者写死在代码里
    # 为了灵活，我们在 sidebar 获取 repo_name
    pass 

with st.sidebar:
    st.header("🗂️ 词书管理")
    
    # --- 1. 基础配置 ---
    with st.expander("🔐 仓库配置", expanded=False):
        gh_token = st.text_input("GitHub Token", value=sec_gh_token, type="password")
        repo_name = st.text_input("Repo Name", value="yourname/memo-app") # 替换为你的默认值
    
    # 实例化 Syncer
    if gh_token and repo_name:
        syncer = GitHubSync(gh_token, repo_name)
    
    st.divider()

    if syncer:
        # --- 2. 词书选择 ---
        if st.button("🔄 刷新词书列表"):
            with st.spinner("扫描中..."):
                books = syncer.list_books()
                st.session_state.book_list = books
                if not books: st.warning("未找到词书，请先新建")
        
        # 下拉菜单选择词书
        book_options = st.session_state.book_list
        selected_book = st.selectbox(
            "选择当前词书", 
            options=book_options,
            index=0 if book_options else None
        )

        # 加载按钮
        if st.button("📥 加载选中词书", type="primary"):
            if selected_book:
                with st.spinner(f"正在读取 {selected_book}..."):
                    st.session_state.data = syncer.pull_data(selected_book)
                    st.session_state.current_book = selected_book
                st.success(f"已加载: {selected_book}")
            else:
                st.error("请先刷新并选择词书")

        st.divider()

        # --- 3. 新建/上传 ---
        st.subheader("➕ 新增内容")
        tab_add1, tab_add2 = st.tabs(["📄 上传追加", "🆕 新建词书"])
        
        with tab_add1:
            # 追加到当前词书
            uploaded_file = st.file_uploader("导入CSV到当前词书", type=['csv'])
            if uploaded_file and st.session_state.current_book:
                if st.button("确认追加"):
                    new_df = pd.read_csv(uploaded_file)
                    # 初始化新数据
                    new_df['id'] = [str(uuid.uuid4()) for _ in range(len(new_df))]
                    new_df['next_review'] = date.today().strftime('%Y-%m-%d')
                    new_df['status'] = 'new'
                    for col in REQUIRED_COLUMNS:
                        if col not in new_df.columns: new_df[col] = "" # 默认值
                        if col in ['interval', 'repetitions']: new_df[col] = 0
                        if col == 'ease_factor': new_df[col] = 2.5
                    
                    st.session_state.data = pd.concat([st.session_state.data, new_df[REQUIRED_COLUMNS]], ignore_index=True)
                    st.success(f"已添加 {len(new_df)} 条，请记得保存！")

        with tab_add2:
            # 创建全新的文件
            new_book_name = st.text_input("新词书文件名 (如: python.csv)")
            new_book_file = st.file_uploader("上传初始CSV (可选)", type=['csv'], key="new_book_upl")
            if st.button("创建新词书"):
                if not new_book_name.endswith(".csv"):
                    new_book_name += ".csv"
                
                # 准备初始化数据
                if new_book_file:
                    init_df = pd.read_csv(new_book_file)
                    # ...同样的初始化逻辑...
                    init_df['id'] = [str(uuid.uuid4()) for _ in range(len(init_df))]
                    init_df['next_review'] = date.today().strftime('%Y-%m-%d')
                    init_df['status'] = 'new'
                    for col in REQUIRED_COLUMNS:
                        if col not in init_df.columns: init_df[col] = ""
                        if col in ['interval', 'repetitions']: init_df[col] = 0
                        if col == 'ease_factor': init_df[col] = 2.5
                    final_init_df = init_df[REQUIRED_COLUMNS]
                else:
                    final_init_df = pd.DataFrame(columns=REQUIRED_COLUMNS)

                # 直接推送到 GitHub
                success, msg = syncer.push_data(final_init_df, new_book_name)
                if success:
                    st.success(f"词书 {new_book_name} 创建成功！请刷新列表。")
                else:
                    st.error(msg)

        st.divider()
        # --- 4. 保存当前进度 ---
        if st.button("☁️ 保存当前词书进度"):
            if st.session_state.current_book:
                with st.spinner("同步中..."):
                    success, msg = syncer.push_data(st.session_state.data, st.session_state.current_book)
                    if success: st.toast(f"{st.session_state.current_book} 保存成功", icon="✅")
                    else: st.error(msg)
            else:
                st.error("未加载任何词书")

# --- 主界面 ---

st.title(f"🧠 记忆训练场")
if st.session_state.current_book:
    st.caption(f"当前词书: {st.session_state.current_book}")
else:
    st.info("👈 请在左侧选择或新建一个词书开始学习")
    st.stop() # 如果没选书，停止渲染下方内容

# 下面是复习逻辑，与之前一致，但基于当前 session_state.data
sec_api_key = st.secrets.get("LLM_API_KEY", "")
sec_base_url = st.secrets.get("LLM_BASE_URL", "https://models.sjtu.edu.cn/api/v1")
sec_model = st.secrets.get("LLM_MODEL", "DeepSeek-V3-685B")

today_str = date.today().strftime('%Y-%m-%d')
df = st.session_state.data
valid_date_mask = df['next_review'].notna() & (df['next_review'] != "")
due_mask = valid_date_mask & (df['next_review'] <= today_str)
new_mask = df['status'] == 'new'
review_queue = df[due_mask | new_mask]
count_due = len(review_queue)

col1, col2, col3 = st.columns(3)
col1.metric("今日待复习", f"{count_due}")
col2.metric("当前本总词数", len(df))
col3.metric("LLM状态", "Ready" if sec_api_key else "Missing Key")

st.divider()

if count_due > 0:
    current_index = review_queue.index[0]
    card = df.loc[current_index]
    
    with st.container(border=True):
        st.markdown(f"### 📇 {card['term']}")
        
        with st.expander("🤖 智能助教"):
            t1, t2, t3 = st.tabs(["解释", "例句", "测试"])
            call_llm = lambda mode: get_llm_explanation(sec_api_key, sec_base_url, sec_model, card['term'], card['context'], mode)
            with t1:
                if st.button("解释"): st.markdown(call_llm("explain"))
            with t2:
                if st.button("例句"): st.markdown(call_llm("examples"))
            with t3:
                if st.button("测试"): st.markdown(call_llm("quiz"))

        st.write("---")

        if not st.session_state.show_answer:
            st.button("👁️ 显示答案", on_click=lambda: st.session_state.update(show_answer=True), use_container_width=True)
        else:
            st.info(card['definition'])
            if card['context']: st.caption(f"备注: {card['context']}")
            
            c1, c2, c3 = st.columns(3)
            def submit_review(quality):
                new_state = SRSManager.calculate_next_review(card, quality)
                for k, v in new_state.items():
                    st.session_state.data.at[current_index, k] = v
                st.session_state.show_answer = False
                st.toast("已更新")
            
            with c1: st.button("🔴 忘记", on_click=submit_review, args=(0,), use_container_width=True)
            with c2: st.button("🟡 模糊", on_click=submit_review, args=(3,), use_container_width=True)
            with c3: st.button("🟢 掌握", on_click=submit_review, args=(5,), use_container_width=True)
else:
    st.balloons()
    st.success("🎉 当前词书任务已完成！")
    with st.expander("查看数据表"):
        st.dataframe(st.session_state.data)