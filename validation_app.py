import streamlit as st
import pandas as pd
import os
import datetime
import random
import json
import ast
import base64
import io
import csv

# --- Constants ---
DATA_DIR = "data"
PARTICIPANT_FILE = os.path.join(DATA_DIR, "participants.csv")
ANSWERS_FILE_TEMPLATE = os.path.join(DATA_DIR, "answers_{language}.csv")
AVAILABLE_LANGUAGES = ["sotho", "sepedi", "setswana"]
WORDS_TO_SHOW = 18  # Number of words to randomly select from the full CSV

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
        'all_answers': pd.DataFrame(),
        'form_key': 0  # For forcing form re-creation and scroll to top
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

def read_csv_manually(filepath):
    """Read a CSV file manually, handling rows with inconsistent field counts"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            # Read the entire file content
            content = f.read()
            
        # Split into lines
        lines = content.splitlines()
        if not lines:
            return pd.DataFrame()
            
        # Parse header
        header = list(csv.reader([lines[0]]))[0]
        expected_fields = len(header)
        
        # Parse rows
        data = []
        inconsistent_rows = 0
        
        for i, line in enumerate(lines[1:], 2):  # Start from line 2 (1-indexed)
            # Skip empty lines
            if not line.strip():
                continue
                
            # Try to parse the line with CSV reader
            try:
                row = list(csv.reader([line]))[0]
                
                # Handle inconsistent field count
                if len(row) < expected_fields:
                    # Add empty values for missing fields
                    row.extend([''] * (expected_fields - len(row)))
                    inconsistent_rows += 1
                elif len(row) > expected_fields:
                    # Truncate extra fields
                    row = row[:expected_fields]
                    inconsistent_rows += 1
                
                data.append(row)
            except Exception as e:
                st.warning(f"Skipping line {i} due to parsing error: {str(e)}")
        
        # Only show one summary warning instead of per-line warnings
        if inconsistent_rows > 0:
            st.info(f"Fixed {inconsistent_rows} rows with incorrect field counts (expected {expected_fields} fields).")
                
        # Create DataFrame
        df = pd.DataFrame(data, columns=header)
        return df
        
    except Exception as e:
        st.error(f"Error manually reading CSV: {str(e)}")
        return pd.DataFrame()

def normalize_column_names(df):
    """Normalize column names to handle case insensitivity and whitespace"""
    # Create a mapping of normalized names to original names
    col_map = {}
    for col in df.columns:
        normalized = col.lower().strip()
        col_map[normalized] = col
    
    # Check for 'word' column in case-insensitive way
    word_col = None
    for norm_name, orig_name in col_map.items():
        if norm_name == 'word' or norm_name == 'words' or 'word' in norm_name:
            word_col = orig_name
            break
    
    # If found, rename to standard 'word'
    if word_col and word_col != 'word':
        df = df.rename(columns={word_col: 'word'})
        st.info(f"Renamed column '{word_col}' to 'word'")
    
    return df

def load_lexicon(language, num_words=WORDS_TO_SHOW):
    """Load word list from CSV file with extra-robust error handling"""
    if not language:
        return pd.DataFrame(), "No language selected."
    
    filename = f"{language}.csv"
    filepath = os.path.join(DATA_DIR, filename)
    
    try:
        if not os.path.exists(filepath):
            return pd.DataFrame(), f"Cannot find {filename}"
        
        # Try our completely manual CSV parser first (most robust)
        full_df = read_csv_manually(filepath)
        
        # Normalize column names to handle case sensitivity
        full_df = normalize_column_names(full_df)
        
        # Fallbacks if needed (should be unnecessary with manual parser)
        if full_df.empty:
            try:
                # Attempt with pandas and flexible quoting
                full_df = pd.read_csv(filepath, quoting=csv.QUOTE_NONE, escapechar='\\')
                full_df = normalize_column_names(full_df)
            except:
                try:
                    # Try with more permissive settings
                    full_df = pd.read_csv(filepath, on_bad_lines='skip')
                    full_df = normalize_column_names(full_df)
                except:
                    # Last resort
                    try:
                        full_df = pd.read_csv(filepath, error_bad_lines=False)
                        full_df = normalize_column_names(full_df)
                    except:
                        return pd.DataFrame(), "Could not parse CSV file with any method"
        
        if full_df.empty:
            return pd.DataFrame(), "CSV file is empty or could not be parsed"
            
        if 'word' not in full_df.columns:
            # Try to identify a suitable column to use as 'word'
            possible_word_cols = [col for col in full_df.columns if 'word' in col.lower()]
            if possible_word_cols:
                word_col = possible_word_cols[0]
                full_df = full_df.rename(columns={word_col: 'word'})
                st.info(f"Using column '{word_col}' as the word column")
            else:
                # If no suitable column found, use the first column
                first_col = full_df.columns[0]
                full_df = full_df.rename(columns={first_col: 'word'})
                st.warning(f"No 'word' column found. Using first column '{first_col}' instead.")
        
        # Clean up data and ensure all required columns exist
        columns_to_check = ['word', 'meaning', 'sentiment', 'explanation']
        for col in columns_to_check:
            if col not in full_df.columns:
                full_df[col] = ''
        
        full_df['word'] = full_df['word'].fillna('').astype(str)
        full_df['meaning'] = full_df['meaning'].fillna('').astype(str)
        full_df['sentiment'] = full_df['sentiment'].fillna('').astype(str)
        full_df['explanation'] = full_df['explanation'].fillna('').astype(str)
        
        if 'rating' in full_df.columns and 'intensity' not in full_df.columns:
            full_df['intensity'] = full_df['rating']
        
        if 'intensity' not in full_df.columns:
            full_df['intensity'] = 0
            
        full_df['intensity'] = pd.to_numeric(full_df['intensity'], errors='coerce').fillna(0).astype(int)
        
        # Display the column names for debugging
        st.info(f"CSV columns found: {', '.join(full_df.columns.tolist())}")
        
        # Remove rows with empty or missing words
        full_df = full_df[full_df['word'].notna() & (full_df['word'] != '')]
        
        # Randomly select the specified number of words
        if len(full_df) > num_words:
            # Get random sample without replacement
            selected_indices = random.sample(range(len(full_df)), num_words)
            df = full_df.iloc[selected_indices].copy().reset_index(drop=True)
        else:
            df = full_df.copy()
            # If we have fewer words than requested, use all of them
            if len(df) < num_words:
                st.warning(f"CSV contains only {len(df)} words, showing all available.")
            
        return df, None
    except Exception as e:
        return pd.DataFrame(), f"Error loading lexicon: {str(e)}"

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
            participants_df = read_csv_manually(PARTICIPANT_FILE)
            st.session_state.all_participants = participants_df
        else:
            st.warning("No participant data found.")
            participants_df = pd.DataFrame()
        
        # Load answers for each language
        all_answers = []
        for lang in AVAILABLE_LANGUAGES:
            answer_path = ANSWERS_FILE_TEMPLATE.format(language=lang)
            if os.path.exists(answer_path):
                lang_df = read_csv_manually(answer_path)
                if not lang_df.empty:
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
        st.error(f"Error loading data: {str(e)}")
    
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
# Always start with page title at top for anchoring
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
                # Remove caching to ensure fresh selection each time
                if hasattr(st, 'cache_data'):
                    st.cache_data.clear()
                    
                df, error_msg = load_lexicon(st.session_state.user_language)
                
                if error_msg:
                    st.error(error_msg)
                elif df.empty:
                    st.error(f"No words found for {st.session_state.user_language}")
                else:
                    # Double check we have exactly 18 words
                    if len(df) != WORDS_TO_SHOW and len(df) > WORDS_TO_SHOW:
                        # Force exactly WORDS_TO_SHOW if needed
                        selected_indices = random.sample(range(len(df)), WORDS_TO_SHOW)
                        df = df.iloc[selected_indices].copy().reset_index(drop=True)
                    
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
    
    # Questions form - use unique form key for each submission to force rebuild
    form_key = f"form_{idx}_{st.session_state.form_key}"
    with st.form(key=form_key):
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
                    # Update form key to force the form to rebuild and scroll to top
                    st.session_state.form_key += 1
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
