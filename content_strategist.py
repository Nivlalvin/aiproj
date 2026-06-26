import streamlit as st
import pandas as pd
import ollama
from googleapiclient.discovery import build
from youtube_transcript_api import YouTubeTranscriptApi
import json
import re
import concurrent.futures
from datetime import datetime
import isodate 

# --- CONFIGURATION ---
st.set_page_config(page_title="YouTube Strategist Pro", layout="wide")

# Custom CSS 
st.markdown("""
    <style>
        .block-container {padding-top: 2rem; padding-bottom: 2rem;}
        div[data-testid="stMetricValue"] {font-size: 1.4rem;}
    </style>
""", unsafe_allow_html=True)

# --- SESSION STATE ---
if 'api_key_memory' not in st.session_state:
    st.session_state['api_key_memory'] = ""

# --- SIDEBAR ---
with st.sidebar:
    st.header("Settings")
    
    api_input = st.text_input(
        "YouTube API Key", 
        type="password", 
        value=st.session_state['api_key_memory'],
        help="Enter your Google Cloud API Key."
    )
    if api_input:
        st.session_state['api_key_memory'] = api_input
        
    st.divider()
    
    mode = st.radio("Analysis Mode", ["Search Trends", "Analyze Single Video"])
    
    if mode == "Search Trends":
        target_topic = st.text_input("Topic to Analyze", value="AI Agents")
        num_videos = st.slider("Videos to Analyze", 3, 10, 5)
    else:
        video_url = st.text_input("Paste YouTube URL")

# --- HELPER FUNCTIONS ---

def parse_duration(pt_string):
    """Converts ISO 8601 duration (PT5M33S) to readable string."""
    try:
        dur = isodate.parse_duration(pt_string)
        total_seconds = dur.total_seconds()
        minutes = int(total_seconds // 60)
        seconds = int(total_seconds % 60)
        return f"{minutes}:{seconds:02d}", total_seconds
    except:
        return "N/A", 0

def calculate_velocity(published_at, view_count):
    """Calculates Views Per Day since publication."""
    try:
        pub_date = datetime.strptime(published_at, "%Y-%m-%dT%H:%M:%SZ")
        days_active = (datetime.now() - pub_date).days
        if days_active < 1: days_active = 1
        return round(view_count / days_active)
    except:
        return 0

def extract_video_id(url):
    if not url: return None
    regex = r"(?:v=|\/)([0-9A-Za-z_-]{11}).*"
    match = re.search(regex, url)
    if match: return match.group(1)
    return None

def fetch_transcript_text(video_id):
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        # Limit context for AI
        text = " ".join([t['text'] for t in transcript])
        return text[:3000], "Transcript"
    except:
        return None, "None"

def fetch_video_data_parallel(youtube, video_id):
    try:
        request = youtube.videos().list(
            part='statistics,snippet,contentDetails',
            id=video_id
        )
        response = request.execute()
        
        if not response['items']: return None
        
        item = response['items'][0]
        stats = item['statistics']
        snippet = item['snippet']
        
        views = int(stats.get('viewCount', 1))
        likes = int(stats.get('likeCount', 0))
        engagement = (likes / views) * 100 if views > 0 else 0
        
        duration_str, duration_sec = parse_duration(item['contentDetails']['duration'])
        velocity = calculate_velocity(snippet['publishedAt'], views)
        
        tags = snippet.get('tags', [])[:10]
        
        content_text, source_type = fetch_transcript_text(video_id)
        
        if source_type == "None":
            content_text = snippet['description'][:2500]
            source_type = "Description"
            
        return {
            'id': item['id'],
            'title': snippet['title'],
            'channel': snippet['channelTitle'],
            'published': snippet['publishedAt'][:10],
            'views': views,
            'velocity': velocity,
            'likes': likes,
            'engagement': round(engagement, 2),
            'duration_str': duration_str,
            'tags': tags,
            'thumbnail': snippet['thumbnails']['high']['url'],
            'content_text': content_text,
            'source_type': source_type
        }
    except Exception:
        return None

def analyze_video_with_ai(data):
    pacing_context = f"Video is {data['duration_str']} long."
    
    prompt = f"""
    Analyze this YouTube video.
    
    Title: "{data['title']}"
    Tags Used: {', '.join(data['tags'])}
    Content: "{data['content_text']}"
    Context: {pacing_context}
    
    Task: Return JSON with these exact keys.
    - "summary": 2-sentence summary.
    - "hook_strategy": Specific trigger used (e.g. Urgency, Controversy).
    - "tone": The vibe (e.g. Fast-paced, Relaxed, Educational).
    - "target_audience": Who is this for?
    - "key_insight": Why is this video successful?
    
    Respond ONLY with JSON.
    """
    
    try:
        response = ollama.chat(
            model='llama3.2',
            format='json',
            options={'num_predict': 200, 'temperature': 0.3},
            messages=[{'role': 'user', 'content': prompt}]
        )
        result = json.loads(response['message']['content'])
        
        for key in ["summary", "hook_strategy", "tone", "target_audience", "key_insight"]:
            if key not in result: result[key] = "N/A"
        return result
        
    except Exception as e:
        return {
            "summary": "Analysis Failed",
            "hook_strategy": "Unknown",
            "tone": "Unknown",
            "target_audience": "Unknown",
            "key_insight": f"Error: {str(e)}"
        }

# --- MAIN UI ---

st.title("YouTube Strategist Pro")
st.markdown("Automated Analysis & Metadata Extraction")

current_key = st.session_state['api_key_memory']
if not current_key:
    st.warning("Please enter your YouTube API Key in the settings sidebar.")
    st.stop()

if st.button("Run Deep Analysis", type="primary"):
    
    youtube = build('youtube', 'v3', developerKey=current_key)
    video_ids = []
    
    # 1. Search Phase
    with st.spinner("Locating videos..."):
        if mode == "Search Trends":
            try:
                req = youtube.search().list(
                    q=target_topic, part='snippet', type='video',
                    order='viewCount', maxResults=num_videos, regionCode='US'
                )
                res = req.execute()
                video_ids = [item['id']['videoId'] for item in res['items']]
            except Exception as e:
                st.error(f"Search Error: {e}")
                
        elif mode == "Analyze Single Video":
            vid_id = extract_video_id(video_url)
            if vid_id: video_ids = [vid_id]
            else: st.error("Invalid URL")

    # 2. Parallel Fetch & Streaming
    if video_ids:
        st.subheader("Analysis Results")
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        csv_data = []
        completed = 0
        total = len(video_ids)
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            future_to_vid = {executor.submit(fetch_video_data_parallel, youtube, vid): vid for vid in video_ids}
            
            for future in concurrent.futures.as_completed(future_to_vid):
                data = future.result()
                
                if data:
                    status_text.caption(f"Processing: {data['title']}")
                    
                    # Run AI
                    ai = analyze_video_with_ai(data)
                    
                    # Store Data
                    row = data.copy()
                    row.update(ai)
                    del row['content_text'] 
                    csv_data.append(row)
                    
                    # --- NEW PROFESSIONAL CARD LAYOUT ---
                    with st.container(border=True):
                        # Header
                        st.subheader(data['title'])
                        st.caption(f"Channel: {data['channel']} | Published: {data['published']}")
                        
                        # Top Metrics Row
                        m1, m2, m3, m4 = st.columns(4)
                        m1.metric("Total Views", f"{data['views']:,}")
                        m2.metric("Velocity", f"{data['velocity']:,}/day", help="Average views per day since upload")
                        m3.metric("Engagement", f"{data['engagement']}%", help="Likes divided by Views")
                        m4.metric("Duration", data['duration_str'])
                        
                        st.divider()
                        
                        # Content Split
                        col_img, col_text = st.columns([1, 2])
                        
                        with col_img:
                            st.image(data['thumbnail'], use_container_width=True)
                            if data['tags']:
                                st.markdown("**SEO Tags**")
                                # Display tags as simple text chips
                                st.caption(" • ".join(data['tags'][:8]))

                        with col_text:
                            st.markdown("**Executive Summary**")
                            st.write(ai.get('summary'))
                            
                            st.markdown("**Strategy Breakdown**")
                            # Colored info boxes for key insights
                            c1, c2 = st.columns(2)
                            c1.info(f"Hook: {ai.get('hook_strategy')}")
                            c2.info(f"Tone: {ai.get('tone')}")
                            
                            # Success Insight
                            st.success(f"Key Insight: {ai.get('key_insight')}")
                    
                    # ------------------------------------
                
                completed += 1
                progress_bar.progress(completed / total)

        status_text.empty()
        st.success("Processing Complete")
        
        if csv_data:
            df = pd.DataFrame(csv_data)
            st.download_button(
                label="Download Report (CSV)",
                data=df.to_csv(index=False).encode('utf-8'),
                file_name='youtube_analysis.csv',
                mime='text/csv'
            )
