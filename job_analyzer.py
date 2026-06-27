import streamlit as st
import requests
import ollama
import pandas as pd
import json
import concurrent.futures

# --- CONFIGURATION ---
st.set_page_config(page_title="HN Job Market Analyzer", layout="wide")

# --- SIDEBAR: USER SETTINGS ---
with st.sidebar:
    st.header("Configuration")
    
    # User Profile Input
    st.subheader("Your Profile")
    user_skills = st.text_area(
        "Enter your skills (comma separated):",
        value="Python, SQL, Data Analysis, Junior Developer, Internship",
        height=150,
        help="The AI will compare these skills against job descriptions to generate a Match Score."
    )
    
    st.markdown("---")
    
    # App Settings
    st.subheader("Scan Settings")
    num_posts = st.slider("Number of jobs to analyze", min_value=5, max_value=50, value=10)
    st.caption("Higher numbers take longer to process.")

# --- FUNCTIONS ---

@st.cache_data(ttl=3600)
def get_latest_hiring_thread_id():
    """
    Finds the ID of the latest 'Who is hiring?' thread by scanning
    submissions from the user 'whoishiring'.
    """
    try:
        user_url = 'https://hacker-news.firebaseio.com/v0/user/whoishiring.json'
        user_data = requests.get(user_url).json()
        submitted_ids = user_data.get('submitted', [])[:30]

        for story_id in submitted_ids:
            story_url = f'https://hacker-news.firebaseio.com/v0/item/{story_id}.json'
            story = requests.get(story_url).json()
            
            # Search for the standard thread title format
            if story and "Who is hiring?" in story.get('title', ""):
                return story['id'], story['title']
        return None, None
    except Exception as e:
        st.error(f"Connection Error: {e}")
        return None, None

def fetch_comment_text(comment_id):
    """
    Fetches the raw text of a single comment (job post).
    """
    try:
        url = f"https://hacker-news.firebaseio.com/v0/item/{comment_id}.json"
        data = requests.get(url, timeout=3).json()
        
        # specific cleanup for HN HTML formatting
        if data and 'text' in data and not data.get('deleted'):
            text = data['text']
            text = text.replace('<p>', '\n').replace('&#x2F;', '/').replace('&quot;', '"')
            return text
        return None
    except:
        return None

def extract_job_data(job_text, skills_profile):
    """
    Uses Llama 3.2 to extract structured data and calculate a match score.
    """
    prompt = f"""
    Analyze this job description against the candidate's skills.

    CANDIDATE SKILLS: "{skills_profile}"
    JOB TEXT (Snippet): "{job_text[:1200]}"

    Return a JSON object with exactly these keys:
    - "role": Job title (short string).
    - "company": Company name (or "Unknown").
    - "tech_stack": List of key technologies (max 5 items, e.g. ["Python", "AWS"]).
    - "remote": "Yes", "No", or "Hybrid".
    - "match_score": Integer 0-100 (How well candidate skills match requirements).
    - "match_reason": A short sentence explaining the score.

    Respond ONLY with valid JSON.
    """
    
    try:
        response = ollama.chat(
            model='llama3.2',
            format='json',
            messages=[{'role': 'user', 'content': prompt}]
        )
        return json.loads(response['message']['content'])
    except Exception:
        # Return a placeholder on failure to avoid crashing the app
        return {
            "role": "Error Parsing",
            "company": "N/A",
            "tech_stack": [],
            "remote": "N/A",
            "match_score": 0,
            "match_reason": "AI processing failed"
        }

# --- MAIN APPLICATION ---

st.title("Hacker News Job Market Analyzer")
st.markdown("Automated tool to scan 'Who is Hiring' threads and match jobs to your profile.")

# 1. Locate Thread
if 'thread_id' not in st.session_state:
    with st.spinner("Locating latest hiring thread..."):
        tid, ttitle = get_latest_hiring_thread_id()
        if tid:
            st.session_state['thread_id'] = tid
            st.session_state['thread_title'] = ttitle
            st.success(f"Target Thread: {ttitle}")
        else:
            st.error("Could not automatically find the latest 'Who is hiring' thread.")

# 2. Analysis Logic
if 'thread_id' in st.session_state:
    
    if st.button("Start Analysis"):
        
        # Status Containers
        status_text = st.empty()
        progress_bar = st.progress(0)
        
        # Step A: Get all comment IDs
        status_text.text("Fetching job IDs from thread...")
        thread_url = f"https://hacker-news.firebaseio.com/v0/item/{st.session_state['thread_id']}.json"
        thread_info = requests.get(thread_url).json()
        all_ids = thread_info.get('kids', [])[:num_posts] # Limit based on slider
        
        # Step B: Download Text (Parallel)
        status_text.text(f"Downloading {len(all_ids)} job descriptions...")
        raw_texts = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            results = executor.map(fetch_comment_text, all_ids)
            for text in results:
                if text:
                    raw_texts.append(text)
        
        # Step C: Analyze with AI (Sequential)
        analyzed_jobs = []
        total_jobs = len(raw_texts)
        
        for i, text in enumerate(raw_texts):
            status_text.text(f"Analyzing Job {i+1}/{total_jobs}...")
            
            # Pass user_skills to the AI function
            job_data = extract_job_data(text, user_skills)
            if job_data:
                analyzed_jobs.append(job_data)
            
            progress_bar.progress((i + 1) / total_jobs)
            
        progress_bar.empty()
        status_text.success("Analysis Complete")
        
        # Step D: Display Results
        if analyzed_jobs:
            df = pd.DataFrame(analyzed_jobs)
            
            # Sort by Match Score (Highest first)
            df = df.sort_values(by="match_score", ascending=False)
            
            # 1. High Level Metrics
            st.subheader("Market Overview")
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Top Technologies Requested**")
                # Flatten the list of lists into a single series
                all_tech = [t for stack in df['tech_stack'] for t in stack]
                if all_tech:
                    st.bar_chart(pd.Series(all_tech).value_counts().head(10))
                else:
                    st.write("No tech stack data found.")
            
            with col2:
                st.write("**Remote Work Availability**")
                st.bar_chart(df['remote'].value_counts())

            # 2. Detailed Table
            st.subheader(f"Top Matches for Your Profile")
            st.dataframe(
                df,
                column_config={
                    "match_score": st.column_config.ProgressColumn(
                        "Match Score",
                        help="Based on your provided skills",
                        format="%d%%",
                        min_value=0,
                        max_value=100,
                    ),
                    "tech_stack": "Tech Stack",
                    "match_reason": "AI Reasoning",
                    "role": "Role",
                    "company": "Company",
                    "remote": "Remote"
                },
                use_container_width=True,
                hide_index=True
            )