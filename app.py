import streamlit as st
import pandas as pd
import uuid
import json
import re
from datetime import date, timedelta, datetime
import os
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
    st.session_state.current_book = None 
if 'book_list' not in st.session_state:
    st.session_state.book_list = []

# ==============================================================================
# 2. 核心逻辑：SM-2 算法 & GitHub 同步
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
            df = pd.read_csv(StringIO(csv_str))
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
            csv_content = df.to_csv(index=False)
            try:
                contents = repo.get_contents(path)
                repo.update_file(contents.path, f"Update {filename}", csv_content, contents.sha)
                return True, "更新成功"
            except GithubException:
                repo.create_file(path, f"Create {filename}", csv_content)
                return True, "创建成功"
        except Exception as e:
            return False, str(e)

# ==============================================================================
# 3. LLM 服务集成 (新增：结构化生成)
# ==============================================================================

def get_llm_client(api_key, base_url):
    return OpenAI(api_key=api_key, base_url=base_url)

# 原有的解释/出题功能
def get_llm_explanation(api_key, base_url, model_name, term, context, mode):
    if not api_key: return "⚠️ 请配置 API Key"
    client = get_llm_client(api_key, base_url)
    
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

# [新增功能] 生成结构化制卡数据
def generate_ai_card(api_key, base_url, model_name, term):
    if not api_key: return None, "⚠️ API Key 未配置"
    client = get_llm_client(api_key, base_url)

    # 强制要求 JSON 格式的 Prompt
    system_prompt = "你是一个专业的数据生成助手。请根据用户输入的术语，生成用于记忆卡片的定义和上下文。"
    user_prompt = f"""
    术语："{term}"
    
    请输出且仅输出一个标准的 JSON 对象，不要包含 ```json 标记或其他废话。格式如下：
    {{
        "definition": "这里是核心定义，简明扼要，适合背诵。",
        "context": "这里是语境、助记提示或一个经典例句（包含中文翻译）。"
    }}
    如果术语是中文，定义用中文；如果是英文，定义用中文，Context提供英文例句。
    """

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.7,
            response_format={"type": "json_object"} # 尝试强制 JSON 模式（如果模型支持）
        )
        content = response.choices[0].message.content
        
        # 清洗可能存在的 markdown 标记
        content = re.sub(r'```json\s*', '', content)
        content = re.sub(r'```', '', content)
        
        data = json.loads(content)
        return data, None
    except Exception as e:
        return None, f"生成失败: {str(e)}"

# ==============================================================================
# 4. Streamlit UI 布局
# ==============================================================================

# ... (前面的代码保持不变) ...

# ==============================================================================
# 4. Streamlit UI 布局
# ==============================================================================

# --- [修改点] 从 Secrets 读取默认配置 ---
sec_gh_token = st.secrets.get("GITHUB_TOKEN", "")
sec_repo_name = st.secrets.get("GITHUB_REPO", "") # 新增：读取仓库名

syncer = None

with st.sidebar:
    st.header("🗂️ 词书管理")
    
    # --- 1. 基础配置 ---
    # expanded=False 收起配置，因为配置好了就不用老看了
    with st.expander("🔐 仓库配置", expanded=not sec_repo_name): 
        gh_token = st.text_input("GitHub Token", value=sec_gh_token, type="password")
        
        # [修改点] value 使用 secrets 里的值
        repo_name = st.text_input("Repo Name", value=sec_repo_name, placeholder="username/repo")
    
    # 实例化 Syncer
    if gh_token and repo_name:
        # 简单的格式校验，防止 404
        if "/" not in repo_name:
            st.error("仓库名格式错误！应为：用户名/仓库名")
        else:
            syncer = GitHubSync(gh_token, repo_name)

    st.divider()

    if syncer:
        # ... (后续代码完全不用动) ...
        if st.button("🔄 刷新词书列表"):
            with st.spinner("扫描中..."):
                st.session_state.book_list = syncer.list_books()
        
        book_options = st.session_state.book_list
        selected_book = st.selectbox("选择当前词书", options=book_options, index=0 if book_options else None)

        if st.button("📥 加载选中词书", type="primary"):
            if selected_book:
                with st.spinner(f"正在读取 {selected_book}..."):
                    st.session_state.data = syncer.pull_data(selected_book)
                    st.session_state.current_book = selected_book
                st.success(f"已加载: {selected_book}")

        st.divider()

        # --- 新增/上传区域 ---
        st.subheader("➕ 生产力工具")
        
        # Tab 分组
        tab_ai, tab_csv, tab_new = st.tabs(["✨ AI制卡", "📄 CSV追加", "🆕 建新书"])
        
        # [修改点] AI 制卡功能
        with tab_ai:
            if st.session_state.current_book:
                st.caption(f"追加到: {st.session_state.current_book}")
                ai_term = st.text_input("输入要背的词/概念", placeholder="例如: RAG / 相对论 / Serendipity")
                
                if st.button("🪄 生成并添加"):
                    if not ai_term:
                        st.warning("请输入内容")
                    else:
                        # 读取配置
                        k = st.secrets.get("LLM_API_KEY", "")
                        b = st.secrets.get("LLM_BASE_URL", "https://models.sjtu.edu.cn/api/v1")
                        m = st.secrets.get("LLM_MODEL", "deepseek-v3")

                        with st.spinner("DeepSeek 正在思考并制作卡片..."):
                            result_data, err = generate_ai_card(k, b, m, ai_term)
                            
                        if result_data:
                            # 构造新行
                            new_row = {
                                'id': str(uuid.uuid4()),
                                'term': ai_term,
                                'definition': result_data.get('definition', ''),
                                'context': result_data.get('context', ''),
                                'last_review': '',
                                'next_review': date.today().strftime('%Y-%m-%d'),
                                'interval': 0, 
                                'repetitions': 0, 
                                'ease_factor': 2.5, 
                                'status': 'new'
                            }
                            # 追加到 DataFrame
                            st.session_state.data = pd.concat([
                                st.session_state.data, 
                                pd.DataFrame([new_row])
                            ], ignore_index=True)
                            
                            st.success(f"✅ 已添加：{ai_term}")
                            with st.expander("查看生成详情", expanded=True):
                                st.write(f"**定义**: {new_row['definition']}")
                                st.write(f"**备注**: {new_row['context']}")
                            st.info("💡 记得点击下方保存按钮同步到云端！")
                        else:
                            st.error(err)
            else:
                st.warning("请先在上方加载一个词书")

        with tab_csv:
            uploaded_file = st.file_uploader("导入CSV到当前词书", type=['csv'])
            if uploaded_file and st.session_state.current_book:
                if st.button("确认CSV追加"):
                    new_df = pd.read_csv(uploaded_file)
                    new_df['id'] = [str(uuid.uuid4()) for _ in range(len(new_df))]
                    new_df['next_review'] = date.today().strftime('%Y-%m-%d')
                    new_df['status'] = 'new'
                    for col in REQUIRED_COLUMNS:
                        if col not in new_df.columns: new_df[col] = ""
                        if col in ['interval', 'repetitions']: new_df[col] = 0
                        if col == 'ease_factor': new_df[col] = 2.5
                    st.session_state.data = pd.concat([st.session_state.data, new_df[REQUIRED_COLUMNS]], ignore_index=True)
                    st.success("CSV 追加成功")

        with tab_new:
            new_book_name = st.text_input("新文件名 (如: java.csv)")
            if st.button("创建空词书"):
                if not new_book_name.endswith(".csv"): new_book_name += ".csv"
                empty_df = pd.DataFrame(columns=REQUIRED_COLUMNS)
                success, msg = syncer.push_data(empty_df, new_book_name)
                if success: st.success("创建成功，请刷新列表")
                else: st.error(msg)

        st.divider()
        
        if st.button("☁️ 保存当前词书进度", type="primary"):
            if st.session_state.current_book:
                with st.spinner("同步中..."):
                    success, msg = syncer.push_data(st.session_state.data, st.session_state.current_book)
                    if success: st.toast("保存成功", icon="✅")
                    else: st.error(msg)
            else:
                st.error("未加载任何词书")

# --- 主界面 ---

st.title(f"🧠 记忆训练场")

# LLM 配置 (复习界面用)
sec_api_key = st.secrets.get("LLM_API_KEY", "")
sec_base_url = st.secrets.get("LLM_BASE_URL", "https://models.sjtu.edu.cn/api/v1")
sec_model = st.secrets.get("LLM_MODEL", "DeepSeek-V3-685B")

if not st.session_state.current_book:
    st.info("👈 请在左侧选择或新建一个词书开始学习")
    st.stop()

# 数据统计
df = st.session_state.data
today_str = date.today().strftime('%Y-%m-%d')
# 修复空值问题，确保 next_review 是字符串
df['next_review'] = df['next_review'].fillna('')
valid_date_mask = df['next_review'] != ""
due_mask = valid_date_mask & (df['next_review'] <= today_str)
new_mask = df['status'] == 'new'
review_queue = df[due_mask | new_mask]

col1, col2, col3 = st.columns(3)
col1.metric("今日待复习", len(review_queue))
col2.metric("当前词书", st.session_state.current_book)
col3.metric("总词条", len(df))

st.divider()

if len(review_queue) > 0:
    current_index = review_queue.index[0]
    card = df.loc[current_index]
    
    with st.container(border=True):
        st.markdown(f"### 📇 {card['term']}")
        
        # 助教功能
        with st.expander("🤖 助教面板"):
            t1, t2, t3 = st.tabs(["解释", "例句", "测试"])
            call_llm = lambda mode: get_llm_explanation(sec_api_key, sec_base_url, sec_model, card['term'], card['context'], mode)
            with t1:
                if st.button("💡 解释"): st.markdown(call_llm("explain"))
            with t2:
                if st.button("📝 例句"): st.markdown(call_llm("examples"))
            with t3:
                if st.button("❓ 测试"): st.markdown(call_llm("quiz"))

        st.write("---")

        if not st.session_state.show_answer:
            st.button("👁️ 显示答案", on_click=lambda: st.session_state.update(show_answer=True), use_container_width=True)
        else:
            st.success(f"定义：{card['definition']}")
            if card['context']: st.caption(f"备注：{card['context']}")
            
            c1, c2, c3 = st.columns(3)
            def submit_review(quality):
                new_state = SRSManager.calculate_next_review(card, quality)
                for k, v in new_state.items():
                    st.session_state.data.at[current_index, k] = v
                st.session_state.show_answer = False
                st.toast("已更新进度")
            
            with c1: st.button("🔴 忘记", on_click=submit_review, args=(0,), use_container_width=True)
            with c2: st.button("🟡 模糊", on_click=submit_review, args=(3,), use_container_width=True)
            with c3: st.button("🟢 掌握", on_click=submit_review, args=(5,), use_container_width=True)
else:
    st.balloons()
    st.success("🎉 当前词书任务已完成！")
    with st.expander("查看数据表"):
        st.dataframe(st.session_state.data)