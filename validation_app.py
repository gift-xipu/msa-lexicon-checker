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
import traceback # Added for better error reporting

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
        return val # Return non-strings or empty strings as is

    # Attempt JSON first for stricter parsing (handles escaped quotes well)
    try:
        # Normalize common escape issues before JSON parsing
        processed_val = val.replace('\\"', '"') # Handle escaped double quotes
        if processed_val.startswith('[') and processed_val.endswith(']'):
             # Check if it looks like a list that might contain dicts
            if '{' in processed_val and '}' in processed_val:
                # Try replacing single quotes around keys/values if needed for JSON
                # Be careful not to replace single quotes within strings
                try:
                    # This is a simplified attempt, might need refinement for complex cases
                    processed_val = processed_val.replace("'s ", "___TEMP_APOSTROPHE___") # Preserve apostrophes
                    processed_val = processed_val.replace("': '", '": "')
                    processed_val = processed_val.replace("{'", '{"')
                    processed_val = processed_val.replace("',", '",')
                    processed_val = processed_val.replace("'}", '"}')
                    processed_val = processed_val.replace("___TEMP_APOSTROPHE___", "'s ")
                except Exception:
                    pass # If replacement fails, continue to ast
            return json.loads(processed_val)
        # Could add similar logic for dicts starting with '{'
    except json.JSONDecodeError:
        pass # If JSON fails, try ast.literal_eval

    # Fallback to ast.literal_eval for simpler list/dict structures
    try:
        # Check if it looks like a literal structure before evaluating
        if (val.startswith('[') and val.endswith(']')) or \
           (val.startswith('{') and val.endswith('}')):
            return ast.literal_eval(val)
        return val # Return original string if not list/dict like
    except (ValueError, SyntaxError, MemoryError, TypeError):
         # Handle cases where literal_eval fails
        return val # Return original string if evaluation fails

def initialize_state():
    """Initialize Streamlit session state variables"""
    defaults = {
        'app_stage': 'welcome',
        'user_language': None,
        'user_name': "",
        'participant_id': None,
        'word_df': pd.DataFrame(),
        'word_indices': [],
        'current_word_idx_position': 0,
        'user_answers': [], # Stores answers for the current session *before* final save
        'all_participants': pd.DataFrame(), # In-memory cache for admin panel
        'all_answers': pd.DataFrame(), # In-memory cache for admin panel
        'form_key': 0,  # For forcing form re-creation and scroll to top
        'lexicon_error': None # To store lexicon loading errors
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

def read_csv_manually(filepath):
    """Read a CSV file manually, handling rows with inconsistent field counts more robustly"""
    data = []
    header = []
    expected_fields = 0
    inconsistent_rows = 0
    skipped_lines = []

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            # Use csv.reader for robust handling of quotes and delimiters
            reader = csv.reader(f)
            try:
                header = next(reader)
                expected_fields = len(header)
                # st.write(f"DEBUG: Header read: {header}, Expected fields: {expected_fields}") # Debug
            except StopIteration:
                st.warning(f"CSV file '{os.path.basename(filepath)}' appears to be empty.")
                return pd.DataFrame() # File is empty
            except Exception as e:
                st.error(f"Error reading header from '{os.path.basename(filepath)}': {e}")
                return pd.DataFrame()

            for i, row in enumerate(reader, 2): # Start line count from 2
                # st.write(f"DEBUG: Reading row {i}: {row}") # Debug
                if not any(field.strip() for field in row): # Skip completely empty rows
                    # st.write(f"DEBUG: Skipping empty row {i}") # Debug
                    continue

                current_fields = len(row)
                if current_fields == expected_fields:
                    data.append(row)
                elif current_fields < expected_fields:
                    # Pad missing fields with empty strings
                    row.extend([''] * (expected_fields - current_fields))
                    data.append(row)
                    inconsistent_rows += 1
                    # st.write(f"DEBUG: Padded row {i}") # Debug
                else: # current_fields > expected_fields
                    # Truncate extra fields
                    data.append(row[:expected_fields])
                    inconsistent_rows += 1
                    # st.write(f"DEBUG: Truncated row {i}") # Debug

    except FileNotFoundError:
        st.error(f"File not found: {filepath}")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Critical error reading CSV '{os.path.basename(filepath)}': {str(e)}")
        st.code(traceback.format_exc()) # Show full traceback for debugging
        return pd.DataFrame()

    if inconsistent_rows > 0:
        st.info(f"File '{os.path.basename(filepath)}': Corrected {inconsistent_rows} rows with inconsistent field counts (expected {expected_fields}).")

    if not data:
       # st.warning(f"No data rows found in '{os.path.basename(filepath)}' after reading header.") # Debug
       # Return empty DataFrame with correct columns if only header existed
       if header:
           return pd.DataFrame(columns=header)
       else:
           return pd.DataFrame()

    try:
        df = pd.DataFrame(data, columns=header)
        # st.write(f"DEBUG: Created DataFrame with shape {df.shape}") # Debug
        return df
    except Exception as e:
        st.error(f"Error creating DataFrame from '{os.path.basename(filepath)}' data: {e}")
        # Try to provide more context if possible
        if header and data:
            st.write("Header:", header)
            st.write("First 5 rows of data:", data[:5])
        return pd.DataFrame()

def normalize_column_names(df):
    """Normalize column names to lower case and strip whitespace."""
    original_columns = list(df.columns)
    df.columns = [str(col).lower().strip() for col in df.columns]
    new_columns = list(df.columns)
    # Report changes if any occurred
    changed_cols = {orig: new for orig, new in zip(original_columns, new_columns) if orig != new}
    if changed_cols:
        st.info(f"Normalized column names: {changed_cols}")
    return df

def load_lexicon(language, num_words=WORDS_TO_SHOW):
    """Load word list from CSV file, normalize columns, select words."""
    st.session_state.lexicon_error = None # Reset error state
    if not language:
        st.session_state.lexicon_error = "No language selected."
        return pd.DataFrame()

    filename = f"{language}.csv"
    filepath = os.path.join(DATA_DIR, filename)

    if not os.path.exists(filepath):
        st.session_state.lexicon_error = f"Lexicon file not found: {filepath}"
        return pd.DataFrame()

    try:
        # Use the robust manual CSV reader
        full_df = read_csv_manually(filepath)

        if full_df.empty:
            # Check if the file was just empty or if read_csv_manually reported an error
            if os.path.getsize(filepath) > 0 :
                 st.warning(f"Could not parse any data from '{filename}', though the file is not empty.")
                 # Fallback attempt with standard pandas read_csv if manual failed unexpectedly
                 try:
                     st.info("Attempting fallback CSV read with pandas...")
                     full_df = pd.read_csv(filepath, on_bad_lines='warn') # 'warn' is better than 'skip' for diagnostics
                     if full_df.empty:
                         st.session_state.lexicon_error = f"Fallback pandas read also resulted in empty DataFrame for '{filename}'."
                         return pd.DataFrame()
                     st.success("Fallback pandas read successful.")
                 except Exception as pd_e:
                     st.session_state.lexicon_error = f"Could not parse '{filename}' with manual or pandas reader. Error: {pd_e}"
                     return pd.DataFrame()
            else:
                 st.session_state.lexicon_error = f"Lexicon file '{filename}' is empty."
                 return pd.DataFrame()

        # --- Column Normalization and Selection ---
        full_df = normalize_column_names(full_df)

        # Identify the 'word' column (case-insensitive)
        word_col_found = None
        possible_word_cols = ['word', 'words', 'term', 'lemma'] # Add other likely names
        for col in possible_word_cols:
            if col in full_df.columns:
                word_col_found = col
                break

        if not word_col_found:
             # If no standard name, try finding one containing 'word'
            cols_with_word = [c for c in full_df.columns if 'word' in c]
            if cols_with_word:
                 word_col_found = cols_with_word[0]
                 st.info(f"Found potential word column by substring: '{word_col_found}'")
            # If still not found, use the first column as a last resort
            elif len(full_df.columns) > 0:
                 word_col_found = full_df.columns[0]
                 st.warning(f"No standard 'word' column found. Using first column '{word_col_found}' as the word source.")
            else:
                 st.session_state.lexicon_error = f"CSV file '{filename}' has no columns."
                 return pd.DataFrame()

        # Rename the identified column to 'word' if it's different
        if word_col_found != 'word':
            st.info(f"Renaming column '{word_col_found}' to 'word'.")
            full_df = full_df.rename(columns={word_col_found: 'word'})

        if 'word' not in full_df.columns:
             st.session_state.lexicon_error = "Failed to identify or rename a 'word' column."
             return pd.DataFrame()
        # --- End Column Normalization ---

        # --- Ensure Standard Columns Exist ---
        # Define standard columns expected for the validation task
        standard_cols = ['word', 'meaning', 'sentiment', 'explanation', 'intensity', 'source_sentences', 'prompt_type']
        for col in standard_cols:
            if col not in full_df.columns:
                # Check for alternatives (e.g., 'rating' for 'intensity')
                if col == 'intensity' and 'rating' in full_df.columns:
                    full_df['intensity'] = pd.to_numeric(full_df['rating'], errors='coerce').fillna(0).astype(int)
                    st.info("Used 'rating' column for 'intensity'.")
                elif col == 'source_sentences' and 'example' in full_df.columns:
                     full_df['source_sentences'] = full_df['example']
                     st.info("Used 'example' column for 'source_sentences'.")
                else:
                    full_df[col] = '' # Add missing standard columns as empty
                    st.info(f"Added missing standard column: '{col}'")

        # Ensure correct types and handle NaNs
        full_df['word'] = full_df['word'].fillna('').astype(str)
        full_df['meaning'] = full_df['meaning'].fillna('').astype(str)
        full_df['sentiment'] = full_df['sentiment'].fillna('').astype(str)
        full_df['explanation'] = full_df['explanation'].fillna('').astype(str)
        full_df['intensity'] = pd.to_numeric(full_df['intensity'], errors='coerce').fillna(0).astype(int)
        full_df['source_sentences'] = full_df['source_sentences'].fillna('').astype(str)
        full_df['prompt_type'] = full_df['prompt_type'].fillna('Unknown').astype(str)
        # --- End Standard Columns ---

        # Remove rows where the 'word' column is empty after processing
        original_count = len(full_df)
        full_df = full_df[full_df['word'].str.strip() != '']
        if len(full_df) < original_count:
            st.info(f"Removed {original_count - len(full_df)} rows with empty 'word' values.")

        if full_df.empty:
            st.session_state.lexicon_error = f"No valid words found in '{filename}' after cleaning."
            return pd.DataFrame()

        # --- Word Selection ---
        st.write(f"Total valid words found in {filename}: {len(full_df)}")
        # st.write("Sample data after processing:") # Optional: Show sample
        # st.dataframe(full_df.head(3))

        if len(full_df) >= num_words:
            # Get random sample without replacement
            selected_indices = random.sample(range(len(full_df)), num_words)
            df = full_df.iloc[selected_indices].copy().reset_index(drop=True)
            st.write(f"Randomly selected {num_words} words.")
        else:
            df = full_df.copy()
            st.warning(f"Lexicon contains only {len(df)} words (less than requested {num_words}). Using all available words.")
        # --- End Word Selection ---

        return df

    except Exception as e:
        st.error(f"An unexpected error occurred while loading lexicon '{filename}': {str(e)}")
        st.code(traceback.format_exc())
        st.session_state.lexicon_error = f"Error loading lexicon: {str(e)}"
        return pd.DataFrame()

def save_participant_info(pid, name, lang):
    """Save participant details to the CSV file."""
    data = {
        'participant_id': [pid],
        'name': [name],
        'language': [lang],
        'start_time': [datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")]
    }
    df_new = pd.DataFrame(data)

    try:
        os.makedirs(DATA_DIR, exist_ok=True)
        # Use 'a' mode and check existence for header writing
        file_exists = os.path.exists(PARTICIPANT_FILE)
        df_new.to_csv(PARTICIPANT_FILE, mode='a', header=not file_exists, index=False, lineterminator='\n') # Use lineterminator for consistency

        # Optionally update in-memory admin cache (if needed immediately)
        # st.session_state.all_participants = pd.concat([st.session_state.all_participants, df_new], ignore_index=True)

        return True
    except Exception as e:
        st.error(f"Error saving participant info: {e}")
        st.code(traceback.format_exc())
        return False

def save_answers(answers_list, lang, pid):
    """Save a list of user responses to the language-specific CSV."""
    if not answers_list:
        st.warning("No answers provided to save.")
        return False # Nothing to save

    try:
        # Create a DataFrame from the list of answer dictionaries
        df_new = pd.DataFrame(answers_list)

        # Ensure participant_id and language are present
        df_new['participant_id'] = pid
        df_new['language'] = lang

        # Reorder columns for consistency (optional but good practice)
        cols_order = ['participant_id', 'language', 'word', 'meaning_correct', 'meaning_fix',
                      'word_sentiment', 'system_understands', 'understanding_correction',
                      'different_context', 'context_explanation', 'system_sentiment',
                      'system_intensity', 'prompt_type', 'example_sentence', 'timestamp']
        # Ensure all columns exist, adding missing ones as None/empty
        for col in cols_order:
             if col not in df_new.columns:
                 df_new[col] = None # Or appropriate default like '' or 0

        df_new = df_new[cols_order] # Apply the order

        # --- Save to CSV ---
        os.makedirs(DATA_DIR, exist_ok=True)
        path = ANSWERS_FILE_TEMPLATE.format(language=lang)
        file_exists = os.path.exists(path)

        # Append data; write header only if file doesn't exist
        df_new.to_csv(path, mode='a', header=not file_exists, index=False, quoting=csv.QUOTE_MINIMAL, lineterminator='\n') # Use QUOTE_MINIMAL

        # Optionally update in-memory admin cache
        # st.session_state.all_answers = pd.concat([st.session_state.all_answers, df_new], ignore_index=True)

        return True
    except Exception as e:
        st.error(f"Error saving answers for language '{lang}': {e}")
        st.code(traceback.format_exc())
        return False

def get_csv_download_link(df, filename, link_text):
    """Generate a link to download the dataframe as a CSV file."""
    if df.empty:
        return f"<i>No data available to download for {filename}.</i>"
    try:
        # Use StringIO to handle CSV creation in memory
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False, quoting=csv.QUOTE_MINIMAL) # Match saving style
        csv_buffer.seek(0)
        csv_data = csv_buffer.getvalue()
        b64 = base64.b64encode(csv_data.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="{filename}" style="text-decoration: none; padding: 5px 10px; background-color: #4CAF50; color: white; border-radius: 5px; margin: 5px;">📥 {link_text}</a>'
        return href
    except Exception as e:
        st.error(f"Error generating download link for {filename}: {e}")
        return f"<i>Could not generate download link for {filename}.</i>"

# --- App Interface ---
st.set_page_config(page_title="Word Checker", layout="centered")

# Clean custom CSS
st.markdown("""
<style>
    /* General body styling (optional) */
    body {
        font-family: sans-serif;
    }
    /* Word display styling */
    .word-display {
        font-size: 2.8rem; /* Slightly larger */
        font-weight: bold;
        color: #2c3e50; /* Darker color */
        text-align: center;
        margin: 25px 0 10px 0; /* Adjust margins */
        padding: 10px;
        background-color: #f8f9fa; /* Light background */
        border-radius: 8px;
        border: 1px solid #dee2e6; /* Subtle border */
    }
    /* Meaning display styling */
    .meaning-display {
        font-size: 1.3rem; /* Slightly larger */
        color: #555; /* Medium gray */
        text-align: center;
        margin-bottom: 25px;
    }
    /* System info box */
    .system-info {
        background-color: #e9ecef; /* Lighter gray background */
        color: #495057; /* Text color */
        padding: 15px;
        border-radius: 8px;
        margin-bottom: 25px;
        border-left: 5px solid #17a2b8; /* Info blue accent */
        font-size: 0.95rem;
    }
    .system-info b { /* Style bold text within the box */
        color: #0056b3; /* Darker blue */
    }
    /* Make Streamlit buttons more prominent */
    .stButton > button {
        width: 100%; /* Full width */
        border-radius: 25px; /* More rounded */
        padding: 10px 20px; /* More padding */
        font-size: 1.1rem; /* Larger text */
        font-weight: 600;
        margin-top: 10px; /* Add space above buttons */
        border: none; /* Remove default border */
        background-image: linear-gradient(to right, #6a11cb 0%, #2575fc 100%); /* Gradient background */
        color: white;
        transition: all 0.3s ease; /* Smooth transition */
        box-shadow: 0 2px 5px rgba(0,0,0,0.2);
    }
    .stButton > button:hover {
        opacity: 0.9;
        box-shadow: 0 4px 8px rgba(0,0,0,0.3);
    }
    .stButton > button:active {
         transform: translateY(1px); /* Click effect */
    }
    /* Styling for download links */
    .download-link-container { /* Add a container for spacing */
        text-align: center;
        margin: 20px 0;
        padding: 10px;
    }
    /* Radio button consistency */
    div[role="radiogroup"] > label {
        margin-right: 15px; /* Space out radio options */
    }
    /* Progress bar styling */
    .stProgress > div > div {
        background-image: linear-gradient(to right, #6a11cb 0%, #2575fc 100%); /* Match button gradient */
    }
</style>
""", unsafe_allow_html=True)

initialize_state() # Make sure state is initialized early

# --- Admin Panel ---
# Access via ?admin=true in the URL
if st.query_params.get("admin") == "true":
    st.title("📊 Word Checker Admin Panel")

    # --- Load Data ---
    st.subheader("Data Loading Status")
    # Load Participants
    participants_df = pd.DataFrame()
    try:
        if os.path.exists(PARTICIPANT_FILE):
            participants_df = read_csv_manually(PARTICIPANT_FILE)
            if participants_df.empty and os.path.getsize(PARTICIPANT_FILE) > 0:
                 st.warning(f"Participants file ({PARTICIPANT_FILE}) exists but could not be parsed.")
            else:
                 st.success(f"Loaded {len(participants_df)} participants from {PARTICIPANT_FILE}")
        else:
            st.info("Participant file not found.")
    except Exception as e:
        st.error(f"Error loading participants: {str(e)}")

    # Load Answers (more robustly)
    all_answers_list = []
    combined_answers_df = pd.DataFrame()
    st.markdown("---")
    st.write("Attempting to load answers for each language:")
    load_errors = False

    for lang in AVAILABLE_LANGUAGES:
        answer_path = ANSWERS_FILE_TEMPLATE.format(language=lang)
        st.write(f"* **{lang.capitalize()}:**")
        if os.path.exists(answer_path):
            st.write(f"    - File found: `{answer_path}`")
            try:
                lang_df = read_csv_manually(answer_path)
                if not lang_df.empty:
                    # **CRITICAL CHECK:** Ensure 'language' column consistency
                    if 'language' not in lang_df.columns:
                        st.warning(f"    - **Warning:** File for '{lang}' is missing the 'language' column! Attempting to add it.")
                        lang_df['language'] = lang # Add language based on filename as fallback
                    elif not lang_df['language'].iloc[0].strip(): # Check if language column exists but is empty
                        st.warning(f"    - **Warning:** File for '{lang}' has an empty 'language' column! Attempting to fill.")
                        lang_df['language'] = lang

                    # Ensure participant_id exists
                    if 'participant_id' not in lang_df.columns:
                         st.warning(f"    - **Warning:** File for '{lang}' is missing 'participant_id' column.")
                         lang_df['participant_id'] = 'Unknown'


                    st.success(f"    - Successfully loaded {len(lang_df)} rows.")
                    all_answers_list.append(lang_df)
                elif os.path.getsize(answer_path) > 0:
                     st.warning(f"    - File for '{lang}' exists but was parsed as empty.")
                     load_errors = True
                # else: # File exists but is genuinely empty
                #     st.info(f"    - File for '{lang}' exists but is empty.")

            except Exception as read_err:
                st.error(f"    - Failed to read or process file for '{lang}': {read_err}")
                st.code(traceback.format_exc()) # Show details
                load_errors = True
        else:
            st.info(f"    - No answers file found for '{lang}'.")

    # Combine loaded DataFrames
    st.markdown("---")
    if all_answers_list:
        try:
            combined_answers_df = pd.concat(all_answers_list, ignore_index=True)
            st.success(f"Successfully combined answers from all found files. Total rows: {len(combined_answers_df)}")

            # **Diagnostic**: Check languages present after concat
            if 'language' in combined_answers_df.columns:
                 present_langs = combined_answers_df['language'].unique()
                 st.write(f"Languages found in combined data: `{', '.join(map(str, present_langs))}`")
                 # Check for missing/empty language strings
                 if combined_answers_df['language'].isnull().any() or (combined_answers_df['language'] == '').any():
                     st.warning("Some rows in the combined data have missing or empty language values!")
            else:
                 st.error("**Critical Error:** 'language' column is missing after concatenating answer DataFrames!")
                 load_errors = True

        except Exception as concat_err:
            st.error(f"Error during concatenation of answer data: {concat_err}")
            load_errors = True
    elif not load_errors:
        st.info("No answer data files were found or contained data.")
    else:
         st.warning("Could not load any answer data due to previous errors.")

    # --- Display Data ---
    st.markdown("---")
    st.header("View & Download Data")

    # Participants Display
    st.subheader("Participants")
    if not participants_df.empty:
        st.write(f"Total registered participants: **{len(participants_df)}**")
        st.dataframe(participants_df, use_container_width=True)
        st.markdown(get_csv_download_link(participants_df, "participants_data.csv", "Download Participants CSV"), unsafe_allow_html=True)
    else:
        st.info("No participant data loaded.")

    # Answers Display
    st.subheader("All Answers")
    if not combined_answers_df.empty:
        st.write(f"Total collected answers: **{len(combined_answers_df)}**")
        st.dataframe(combined_answers_df, use_container_width=True)
        st.markdown(get_csv_download_link(combined_answers_df, "all_answers_data.csv", "Download All Answers CSV"), unsafe_allow_html=True)

        # Individual Language Downloads
        st.subheader("Download Answers by Language")
        if 'language' in combined_answers_df.columns:
            lang_summary = {}
            cols_to_display = ['language', 'word', 'meaning_correct', 'word_sentiment', 'participant_id', 'timestamp'] # Sample columns
            missing_cols = [c for c in cols_to_display if c not in combined_answers_df.columns]
            if missing_cols:
                 st.warning(f"Note: Columns missing for summary view: {missing_cols}")
                 cols_to_display = [c for c in cols_to_display if c in combined_answers_df.columns]


            for lang in AVAILABLE_LANGUAGES:
                 lang_answers = combined_answers_df[combined_answers_df['language'] == lang]
                 count = len(lang_answers)
                 lang_summary[lang] = count
                 if count > 0:
                    with st.expander(f"{lang.capitalize()} ({count} answers)"):
                         st.dataframe(lang_answers[cols_to_display].head(), use_container_width=True) # Show head
                         st.markdown(get_csv_download_link(lang_answers, f"answers_{lang}.csv", f"Download {lang.capitalize()} CSV"), unsafe_allow_html=True)
                 # else: # Optionally show languages with 0 answers found
                 #    st.write(f"{lang.capitalize()}: 0 answers found in loaded data.")

            st.subheader("Answers Summary by Language")
            st.bar_chart(pd.Series(lang_summary))

        else:
            st.error("Cannot provide per-language downloads because the 'language' column was not found in the combined data.")

    elif not load_errors:
        st.info("No answer data loaded.")
    else:
        st.warning("Answer data could not be displayed due to loading/processing errors.")


    # --- Back Button ---
    st.markdown("---")
    st.header("Return to App")
    if st.button("⬅️ Back to Word Checker App"):
        # Remove admin parameter using st.query_params
        current_params = st.query_params.to_dict()
        if "admin" in current_params:
            del current_params["admin"]
        st.query_params.clear()
        st.query_params.update(**current_params) # Update with modified params
        st.rerun() # Rerun the script without the admin param

    st.stop() # Stop execution here for admin panel

# ==========================
# --- Main Application Logic ---
# ==========================

st.title("📖 Word Checker")

# Stage 1: Welcome & Language Selection
if st.session_state.app_stage == 'welcome':
    st.header("Welcome!")
    st.write("Please select the language you want to check words for:")

    cols = st.columns(len(AVAILABLE_LANGUAGES))
    for i, lang in enumerate(AVAILABLE_LANGUAGES):
        if cols[i].button(lang.capitalize(), key=f"lang_select_{lang}", use_container_width=True):
            st.session_state.user_language = lang
            st.session_state.app_stage = 'user_info'
            # Clear previous lexicon errors before proceeding
            st.session_state.lexicon_error = None
            st.rerun()

# Stage 2: User Info & Lexicon Loading
elif st.session_state.app_stage == 'user_info':
    st.header(f"Check {st.session_state.user_language.capitalize()} Words")
    st.write("Please enter your name to begin.")

    name = st.text_input("Your Name", value=st.session_state.user_name, key="user_name_input")

    if st.button("🚀 Start Checking Words", key="start_button"):
        user_name_stripped = name.strip()
        if user_name_stripped:
            st.session_state.user_name = user_name_stripped
            # Generate a more unique participant ID
            timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f') # Added microseconds
            st.session_state.participant_id = f"{st.session_state.user_language}_{user_name_stripped.replace(' ','_')}_{timestamp}"

            # Save participant info
            if save_participant_info(st.session_state.participant_id, st.session_state.user_name, st.session_state.user_language):
                st.success(f"Welcome, {st.session_state.user_name}! Loading words...")

                # Load the lexicon for the selected language
                # Cache clearing (consider if really needed, might slow down if file is large)
                # if hasattr(st, 'cache_data'):
                #     st.cache_data.clear()
                # if hasattr(st, 'cache_resource'):
                #      st.cache_resource.clear()

                with st.spinner(f"Loading {st.session_state.user_language} lexicon..."):
                     df = load_lexicon(st.session_state.user_language, WORDS_TO_SHOW)

                # Check if loading was successful (check session state error)
                if st.session_state.lexicon_error:
                     st.error(f"Failed to load lexicon: {st.session_state.lexicon_error}")
                     # Offer to go back or retry?
                     if st.button("Go Back"):
                         st.session_state.app_stage = 'welcome'
                         st.rerun()
                elif df.empty:
                     st.error(f"No words were loaded for {st.session_state.user_language}. The file might be empty or incorrectly formatted.")
                     if st.button("Go Back"):
                         st.session_state.app_stage = 'welcome'
                         st.rerun()
                else:
                     st.success(f"Loaded {len(df)} words. Let's start!")
                     st.session_state.word_df = df
                     indices = list(df.index)
                     random.shuffle(indices)
                     st.session_state.word_indices = indices
                     st.session_state.current_word_idx_position = 0
                     st.session_state.user_answers = [] # Clear previous answers if any
                     st.session_state.form_key = 0 # Reset form key
                     st.session_state.app_stage = 'validation'
                     st.rerun() # Move to validation stage
            else:
                st.error("Could not save participant information. Please check file permissions or contact support.")
                # Avoid proceeding if participant info fails to save
        else:
            st.warning("❗ Please enter your name.")

    if st.button("Back to Language Selection"):
         st.session_state.app_stage = 'welcome'
         st.rerun()


# Stage 3: Word Validation Loop
elif st.session_state.app_stage == 'validation':
    total_words = len(st.session_state.word_indices)
    current_index_pos = st.session_state.current_word_idx_position

    # Check if we've finished all words
    if current_index_pos >= total_words:
        st.session_state.app_stage = 'complete'
        st.rerun() # Go to completion stage

    # Get the current word's actual index in the DataFrame
    current_df_idx = st.session_state.word_indices[current_index_pos]
    word_entry = st.session_state.word_df.loc[current_df_idx]

    # --- Display Progress and Word ---
    st.progress((current_index_pos + 1) / total_words)
    st.caption(f"Word {current_index_pos + 1} of {total_words} ({st.session_state.user_language.capitalize()})")

    st.markdown(f'<div class="word-display">{word_entry["word"]}</div>', unsafe_allow_html=True)
    meaning = word_entry.get("meaning", "").strip()
    st.markdown(f'<div class="meaning-display"><b>Provided Meaning:</b> {meaning if meaning else "<i>(No meaning provided)</i>"}</div>', unsafe_allow_html=True)

    # --- Display System Analysis ---
    system_sentiment = word_entry.get('sentiment', 'Unknown').strip()
    system_intensity = word_entry.get('intensity', 0) # Already cast to int in load_lexicon
    example_sentence = ""
    # Use safe_literal_eval to handle potential list stored as string
    raw_sentences = word_entry.get('source_sentences', '')
    parsed_sentences = safe_literal_eval(raw_sentences)
    if isinstance(parsed_sentences, list) and parsed_sentences:
        example_sentence = str(parsed_sentences[0]).strip() # Take the first one
    elif isinstance(parsed_sentences, str) and parsed_sentences:
         example_sentence = parsed_sentences # Use as is if it wasn't a list string

    system_info_html = f"""
    <div class="system-info">
        <b>System Analysis:</b><br>
        - Sentiment: <b>{system_sentiment if system_sentiment else 'N/A'}</b><br>
        - Intensity: <b>{system_intensity if system_sentiment else 'N/A'}</b><br>
        {f'- Example Usage: <i>"{example_sentence}"</i>' if example_sentence else '- <i>(No example sentence provided)</i>'}
    </div>
    """
    st.markdown(system_info_html, unsafe_allow_html=True)

    # --- Questions Form ---
    # Use a unique key combined with word index and form key increment
    form_key = f"form_{current_df_idx}_{st.session_state.form_key}"
    with st.form(key=form_key):
        st.subheader("Your Feedback")

        # Q1: Meaning Correctness
        st.markdown("**1. Meaning Check**")
        q1_meaning_correct = st.radio(
            "Is the provided meaning accurate for this word?",
            ["Yes", "Partly", "No"],
            horizontal=True,
            key=f"q1_{current_df_idx}"
        )
        q2_meaning_fix = st.text_input(
             "If 'Partly' or 'No', please suggest a correction or clarification:",
             key=f"q2_{current_df_idx}"
             )

        st.markdown("---")

        # Q2: User Sentiment Assessment
        st.markdown("**2. Word Feeling**")
        q3_word_sentiment = st.radio(
            "How would you describe the typical feeling this word expresses?",
            ["Positive", "Neutral", "Negative"],
            horizontal=True,
            key=f"q3_{current_df_idx}"
        )

        st.markdown("---")

        # Q3: System Understanding (based on example)
        st.markdown("**3. System Understanding (based on example)**")
        if example_sentence:
            q5_system_understands = st.radio(
                f"Based on the example sentence, does the system seem to understand this word correctly in context?",
                ["Yes", "Partly", "No"],
                horizontal=True,
                key=f"q5_{current_df_idx}"
            )
            q6_understanding_correction = ""
            # Show correction box only if understanding is not perfect
            if q5_system_understands != "Yes":
                q6_understanding_correction = st.text_input(
                    "What seems misunderstood or needs correction?",
                    key=f"q6_{current_df_idx}"
                    )
        else:
            st.info("No example sentence provided for context check.")
            q5_system_understands = "N/A - No Example"
            q6_understanding_correction = "" # Ensure it's defined

        st.markdown("---")

        # Q4: Alternative Contexts
        st.markdown("**4. Other Uses**")
        q7_different_context = st.radio(
            "Can this word be used differently in other situations or contexts (e.g., different meaning, formality, or feeling)?",
            ["Yes", "No"],
            horizontal=True,
            key=f"q7_{current_df_idx}"
        )
        q8_context_explanation = ""
        # Show explanation box only if 'Yes'
        if q7_different_context == "Yes":
            q8_context_explanation = st.text_area(
                "Briefly describe how it might be used differently:",
                placeholder="Example: 'Can mean X in slang', 'More formal way to say Y', 'Negative if used sarcastically'",
                key=f"q8_{current_df_idx}"
            )

        st.markdown("---") # Separator before submit button

        # Submit Button
        submitted = st.form_submit_button("➡️ Next Word", use_container_width=True)

        if submitted:
             # Validate required fields (radio buttons always have a value)
            # No explicit validation needed unless we add more complex inputs

            # Collect answers into a dictionary
            answer_data = {
                "word": str(word_entry['word']), # Ensure string
                "meaning_correct": str(q1_meaning_correct),
                "meaning_fix": str(q2_meaning_fix).strip() if q2_meaning_fix else None,
                "word_sentiment": str(q3_word_sentiment),
                "system_understands": str(q5_system_understands),
                "understanding_correction": str(q6_understanding_correction).strip() if q6_understanding_correction else None,
                "different_context": str(q7_different_context),
                "context_explanation": str(q8_context_explanation).strip() if q8_context_explanation else None,
                "system_sentiment": str(system_sentiment), # Record what the system showed
                "system_intensity": int(system_intensity), # Record what the system showed
                "prompt_type": str(word_entry.get('prompt_type', 'Unknown')), # Record original prompt type
                "example_sentence": str(example_sentence) if example_sentence else None, # Record the sentence shown
                "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }

            # Append to session state list (primarily for potential batch save later)
            st.session_state.user_answers.append(answer_data)

            # Save this single answer immediately for robustness
            save_success = save_answers(
                [answer_data], # Save function expects a list
                st.session_state.user_language,
                st.session_state.participant_id
                )

            if save_success:
                # Increment position and form key, then rerun to display next word
                st.session_state.current_word_idx_position += 1
                st.session_state.form_key += 1 # Crucial: forces form redraw
                st.rerun()
            else:
                # Keep user on the same word, show error
                st.error("❌ Error saving your answer for this word. Please try submitting again.")
                # Do NOT increment position or form key, allow retry

# Stage 4: Completion
elif st.session_state.app_stage == 'complete':
    st.balloons()
    st.success("🎉 All done! Thank you for your valuable feedback!")
    st.markdown(f"You have successfully checked **{len(st.session_state.word_indices)}** words in **{st.session_state.user_language.capitalize()}**.")
    st.info(f"Participant ID: `{st.session_state.participant_id}`") # Show ID for reference

    # Final check: Ensure all answers collected in session state were attempted to be saved
    # (save_answers already tries individually, so this is mostly cleanup)
    if st.session_state.user_answers:
         st.warning("Attempting to clear session answer buffer (should be empty if saves were successful)...")
         # Data should already be saved individually, just clear the buffer
         st.session_state.user_answers = []


    st.markdown("---")
    if st.button("🏁 Start Over (New Language or User)", key="start_over_button"):
        # Clear *most* session state keys to reset the app
        # Keep necessary ones like admin state if implemented differently
        keys_to_keep = [] # Add any keys you want to persist across restarts
        for key in list(st.session_state.keys()):
             if key not in keys_to_keep:
                 del st.session_state[key]
        # Re-initialize default state
        initialize_state()
        st.rerun()

# Fallback Stage (Error)
else:
    st.error("🚨 An unexpected application state occurred.")
    st.warning("Please try restarting the process.")
    if st.button("🔄 Restart Application"):
        # Full reset
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        initialize_state()
        st.rerun()
