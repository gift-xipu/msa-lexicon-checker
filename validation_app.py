import streamlit as st
import pandas as pd
import os
import datetime
import random
import json
import ast
import base64
import io

# --- Constants ---
DATA_DIR = "data"
PARTICIPANT_FILE = os.path.join(DATA_DIR, "participants.csv")
ANSWERS_FILE_TEMPLATE = os.path.join(DATA_DIR, "answers_{language}.csv")
AVAILABLE_LANGUAGES = ["sotho", "sepedi", "setswana"]

# --- Helper Functions ---
def safe_literal_eval(val):
    """Extract list/dict values from strings."""
    if isinstance(val, (list, dict)):
        return val
    if not isinstance(val, str) or not val.strip():
        return val
    
    try:
        val = val.replace('\\"', '"')
        if val.startswith('[') and val.endswith(']'):
            try:
                return json.loads(val)
            except:
                try:
                    return ast.literal_eval(val)
                except:
                    return val
        return val
    except:
        return val

def initialize_state():
    defaults = {
        'app_stage': 'welcome',
        'user_language': None,
        'user_name': "",
        'participant_id': None,
        'word_df': pd.DataFrame(),
        'word_indices': [],
        'current_word_idx_position': 0,
        'user_answers': [],
        'all_participants': pd.DataFrame(),
        'all_answers': pd.DataFrame()
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

@st.cache_data
def load_lexicon(language):
    """Load word list from CSV file"""
    if not language:
        return pd.DataFrame(), "No language selected."
    
    filename = f"{language}.csv"
    filepath = os.path.join(DATA_DIR, filename)
    
    try:
        if not os.path.exists(filepath):
            return pd.DataFrame(), f"Cannot find {filename}"
        
        df = pd.read_csv(filepath)
        
        if 'word' not in df.columns:
            return pd.DataFrame(), "File missing word column"
        
        # Clean up data
        df['word'] = df['word'].fillna('').astype(str)
        df['meaning'] = df['meaning'].fillna('').astype(str)
        df['sentiment'] = df['sentiment'].fillna('').astype(str)
        df['explanation'] = df['explanation'].fillna('').astype(str)
        
        if 'rating' in df.columns and 'intensity' not in df.columns:
            df['intensity'] = df['rating']
        
        df['intensity'] = pd.to_numeric(df.get('intensity', 0), errors='coerce').fillna(0).astype(int)
        
        return df.copy(), None
    except Exception as e:
        return pd.DataFrame(), f"Error: {str(e)}"

def save_participant_info(pid, name, lang):
    """Save who is taking the test"""
    data = {
        'participant_id': [pid],
        'name': [name],
        'language': [lang],
        'start_time': [datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")]
    }
    df = pd.DataFrame(data)
    
    # Save locally
    try:
        os.makedirs(DATA_DIR, exist_ok=True)
        exists = os.path.exists(PARTICIPANT_FILE)
        df.to_csv(PARTICIPANT_FILE, mode='a', header=not exists, index=False)
        
        # Also append to our in-memory collection
        st.session_state.all_participants = pd.concat([st.session_state.all_participants, df], ignore_index=True)
        
        return True
    except Exception as e:
        st.error(f"Error saving participant: {e}")
        return False

def save_answers(answers, lang, pid):
    """Save user responses"""
    if not answers:
        return False
    
    try:
        # Create a DataFrame from answers
        df = pd.DataFrame(answers)
        
        # Add participant ID and language
        df['participant_id'] = pid
        df['language'] = lang
        
        # Make sure data directory exists
        os.makedirs(DATA_DIR, exist_ok=True)
        
        # Get the file path
        path = ANSWERS_FILE_TEMPLATE.format(language=lang)
        
        # Check if file exists to determine if we need headers
        exists = os.path.exists(path)
        
        # Save to CSV
        df.to_csv(path, mode='a', header=not exists, index=False)
        
        # Also append to our in-memory collection
        st.session_state.all_answers = pd.concat([st.session_state.all_answers, df], ignore_index=True)
        
        return True
    except Exception as e:
        st.error(f"Error saving answers: {e}")
        return False

def get_csv_download_link(df, filename, link_text):
    """Generate a link to download the dataframe as a CSV file."""
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">📥 {link_text}</a>'
    return href

# --- App Interface ---
st.set_page_config(page_title="Word Checker", layout="centered")

# Clean custom CSS
st.markdown("""
<style>
    .word-display {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        margin: 20px 0;
    }
    .meaning-display {
        font-size: 1.2rem;
        text-align: center;
        margin-bottom: 20px;
    }
    .system-info {
        padding: 15px;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    .stButton > button {
        width: 100%;
        border-radius: 20px;
    }
    .download-link {
        text-align: center;
        margin: 20px 0;
    }
</style>
""", unsafe_allow_html=True)

initialize_state()

# Admin panel (accessible via query parameter)
if st.query_params.get("admin") == "true":
    st.title("Word Checker Admin Panel")
    
    # Load all data from CSV files
    try:
        # Load participants
        if os.path.exists(PARTICIPANT_FILE):
            participants_df = pd.read_csv(PARTICIPANT_FILE)
            st.session_state.all_participants = participants_df
        else:
            st.warning("No participant data found.")
            participants_df = pd.DataFrame()
        
        # Load answers for each language
        all_answers = []
        for lang in AVAILABLE_LANGUAGES:
            answer_path = ANSWERS_FILE_TEMPLATE.format(language=lang)
            if os.path.exists(answer_path):
                lang_df = pd.read_csv(answer_path)
                all_answers.append(lang_df)
        
        if all_answers:
            answers_df = pd.concat(all_answers, ignore_index=True)
            st.session_state.all_answers = answers_df
        else:
            st.warning("No answer data found.")
            answers_df = pd.DataFrame()
        
        # Display data
        st.header("Participants")
        if not participants_df.empty:
            st.write(f"Total participants: {len(participants_df)}")
            st.dataframe(participants_df)
            st.markdown(get_csv_download_link(participants_df, "participants.csv", "Download Participants CSV"), unsafe_allow_html=True)
        
        st.header("All Answers")
        if not answers_df.empty:
            st.write(f"Total answers: {len(answers_df)}")
            st.dataframe(answers_df)
            st.markdown(get_csv_download_link(answers_df, "all_answers.csv", "Download All Answers CSV"), unsafe_allow_html=True)
            
            # Individual language downloads
            st.header("Download by Language")
            for lang in AVAILABLE_LANGUAGES:
                lang_answers = answers_df[answers_df['language'] == lang]
                if not lang_answers.empty:
                    st.markdown(f"**{lang.capitalize()}** ({len(lang_answers)} answers)")
                    st.markdown(get_csv_download_link(lang_answers, f"answers_{lang}.csv", f"Download {lang.capitalize()} CSV"), unsafe_allow_html=True)
    
    except Exception as e:
        st.error(f"Error loading data: {e}")
    
    st.header("Return to App")
    if st.button("Back to App"):
        # Remove admin parameter
        params = st.query_params.to_dict()
        if "admin" in params:
            del params["admin"]
        st.query_params.update(params)
        st.rerun()
    
    # Stop execution here so the main app doesn't run
    st.stop()

# Main app
st.title("Word Checker")

# Step 1: Choose Language
if st.session_state.app_stage == 'welcome':
    st.write("Select a language to begin")
    
    for lang in AVAILABLE_LANGUAGES:
        if st.button(lang.capitalize(), use_container_width=True):
            st.session_state.user_language = lang
            st.session_state.app_stage = 'user_info'
            st.rerun()

# Step 2: Enter Name
elif st.session_state.app_stage == 'user_info':
    st.subheader(f"Check {st.session_state.user_language.capitalize()} words")
    
    name = st.text_input("Your name")
    
    if st.button("Start"):
        if name.strip():
            st.session_state.user_name = name.strip()
            st.session_state.participant_id = f"{st.session_state.user_language}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            if save_participant_info(st.session_state.participant_id, st.session_state.user_name, st.session_state.user_language):
                df, error_msg = load_lexicon(st.session_state.user_language)
                
                if error_msg:
                    st.error(error_msg)
                elif df.empty:
                    st.error(f"No words found for {st.session_state.user_language}")
                else:
                    st.session_state.word_df = df
                    indices = list(df.index)
                    random.shuffle(indices)
                    st.session_state.word_indices = indices
                    st.session_state.current_word_idx_position = 0
                    st.session_state.user_answers = []
                    st.session_state.app_stage = 'validation'
                    st.rerun()
        else:
            st.error("Please enter your name")

# Step 3: Check Words
elif st.session_state.app_stage == 'validation':
    total = len(st.session_state.word_indices)
    current = st.session_state.current_word_idx_position
    
    if current >= total:
        st.session_state.app_stage = 'complete'
        st.rerun()
    
    idx = st.session_state.word_indices[current]
    word_entry = st.session_state.word_df.loc[idx]
    
    # Progress
    st.progress((current + 1) / total)
    st.caption(f"Word {current + 1} of {total}")
    
    # Display word prominently
    st.markdown(f'<div class="word-display">{word_entry["word"]}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="meaning-display">Meaning: {word_entry.get("meaning", "-")}</div>', unsafe_allow_html=True)
    
    # System analysis with bootstrapped sentence
    if 'source_sentences' in word_entry and word_entry['source_sentences']:
        parsed_sentences = safe_literal_eval(word_entry['source_sentences'])
        example_sentence = parsed_sentences[0] if isinstance(parsed_sentences, list) and parsed_sentences else ""
    else:
        example_sentence = ""
    
    st.markdown(f"""
    <div class="system-info">
        <b>System says:</b> {word_entry.get('sentiment', 'Unknown')}<br>
        {f'<b>Example:</b> "{example_sentence}"' if example_sentence else ''}
    </div>
    """, unsafe_allow_html=True)
    
    # Questions form
    with st.form(key=f"form_{idx}"):
        # Q1: Meaning check
        q1 = st.radio(
            "Is the meaning correct?",
            ["Yes", "Partly", "No"],
            horizontal=True
        )
        
        q2 = st.text_input("Suggest correction (optional)")
        
        st.markdown("---")
        
        # Q2: Word sentiment
        q3 = st.radio(
            "What feeling does the word express?",
            ["Positive", "Neutral", "Negative"],
            horizontal=True
        )
        
        
        st.markdown("---")
        
        # Q3: System understanding check (based on example sentence)
        if example_sentence:
            q5 = st.radio(
                f"Based on the example, does the system understand this word correctly?",
                ["Yes", "Partly", "No"],
                horizontal=True
            )
            
            q6 = ""
            if q5 != "Yes":
                q6 = st.text_input("What needs correction?")
        else:
            q5 = "No example available"
            q6 = ""
        
        st.markdown("---")
        
        # Q4: Different context question
        q7 = st.radio(
            "Can this word be used differently in other contexts?",
            ["Yes", "No"],
            horizontal=True
        )
        
        q8 = ""
        if q7 == "Yes":
            q8 = st.text_area(
                "How might it be used differently?",
                placeholder="Example: 'It can mean something negative when...' or 'In formal settings, it means...'"
            )
        
        submit = st.form_submit_button("Next", use_container_width=True)
        
        if submit:
            if q1 and q3:
                # Make sure all fields are in a normal Python format (not Pandas/NumPy types)
                answer = {
                    "word": str(word_entry['word']),
                    "meaning_correct": str(q1),
                    "meaning_fix": str(q2) if q2 else None,
                    "word_sentiment": str(q3),
                    "system_understands": str(q5),
                    "understanding_correction": str(q6) if q6 else None,
                    "different_context": str(q7),
                    "context_explanation": str(q8) if q8 else None,
                    "system_sentiment": str(word_entry.get('sentiment', 'Unknown')),
                    "system_intensity": int(word_entry.get('intensity', 0)),
                    "prompt_type": str(word_entry.get('prompt_type', 'Unknown')),
                    "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
                
                # Add example sentence if it exists
                if example_sentence:
                    answer["example_sentence"] = str(example_sentence)
                
                # Save each answer to both session state and CSV
                st.session_state.user_answers.append(answer)
                success = save_answers([answer], st.session_state.user_language, st.session_state.participant_id)
                
                if success:
                    st.session_state.current_word_idx_position += 1
                    st.rerun()
                else:
                    st.error("Error saving your answer. Please try again.")
            else:
                st.error("Please answer all required questions")

# Step 4: Done
elif st.session_state.app_stage == 'complete':
    st.success("All done! Thank you for your help.")
    
    # Save remaining answers
    if st.session_state.user_answers:
        save_answers(st.session_state.user_answers, st.session_state.user_language, st.session_state.participant_id)
        st.session_state.user_answers = []
    
    if st.button("Start Over"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

else:
    st.error("Something went wrong.")
    if st.button("Restart"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()
