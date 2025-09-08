import streamlit as st
from pathlib import Path
from sentence_transformers import SentenceTransformer, util
import re
import string
import pandas as pd
from io import BytesIO
from datetime import datetime

# ------------------------
# Load model (local or online)
# ------------------------
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")  # <-- change if model is not local

model = load_model()

# ------------------------
# Seed categories
# ------------------------
categories = {
    # ... your full categories dict here (unchanged)
}

# Build initial embeddings (case-insensitive)
category_embeddings = {
    cat: model.encode([s.lower() for s in sentences], convert_to_tensor=True).mean(dim=0)
    for cat, sentences in categories.items()
}

# Threshold
SIMILARITY_THRESHOLD = 0.53
new_groups = {}
group_counter = 1

# ------------------------
# Regex for ticket/order/ID
# ------------------------
ID_PATTERN = re.compile(r"(?<!\w)(?:1-[A-Za-z0-9]+|250\d+|Q\d+|TM\d+)(?!\w)", re.IGNORECASE)

def has_valid_id(msg: str) -> bool:
    return bool(ID_PATTERN.search(msg))

def extract_ids(msg: str):
    return ID_PATTERN.findall(msg)

# ------------------------
# Categorization helpers
# ------------------------
def clean_message(msg: str) -> str:
    msg = ID_PATTERN.sub("", msg)  # remove IDs
    msg = msg.lower()
    msg = re.sub(rf"[{re.escape(string.punctuation)}]", " ", msg)
    msg = re.sub(r"\s+", " ", msg)
    return msg.strip()

def categorize_message(msg):
    global group_counter
    clean_msg = clean_message(msg)
    emb = model.encode(clean_msg, convert_to_tensor=True)
    scores = {cat: util.cos_sim(emb, emb_cat).item() for cat, emb_cat in category_embeddings.items()}
    best_cat, best_score = max(scores.items(), key=lambda x: x[1])
    if best_score >= SIMILARITY_THRESHOLD:
        return best_cat, best_score
    else:
        new_cat = f"auto_group_{group_counter}"
        new_groups[new_cat] = emb
        category_embeddings[new_cat] = emb
        group_counter += 1
        return new_cat, best_score

# ------------------------
# Streamlit UI
# ------------------------
st.set_page_config(layout="centered")
st.title("📂 TMF Reporter 3")
st.write("Enhanced report categorizer.")

tab1, tab2, tab3 = st.tabs(["Text Cleansing", "Categorizer", "Categorize Single Message"])

# ------------------------
# Tab 1: File categorizer
# ------------------------
with tab1:
    uploaded_file = st.file_uploader("Upload cleansed_output.txt", type=["txt"])
    if uploaded_file:
        lines = uploaded_file.read().decode("utf-8").splitlines()
        messages = []
        for line in lines:
            if "]" in line and ":" in line:
                msg = line.split(":", 2)[-1].strip()
                if msg and has_valid_id(msg):
                    messages.append(msg)
        st.success(f"Loaded {len(messages)} messages (filtered unnecessary text)")

        results = []
        for msg in messages:
            cat, score = categorize_message(msg)
            results.append((msg, cat, round(score, 2)))

        view_mode = st.radio("Select View Mode", ["Developer View", "User View"])
        st.markdown("---")

        if view_mode == "Developer View":
            st.subheader("Categorized Messages")
            dev_output_lines = [f"[{cat}] ({score}) → {msg}" for msg, cat, score in results]
            dev_output_text = "\n\n".join(dev_output_lines)
            st.markdown(
                """<style> textarea[disabled] { cursor: default !important; } </style>""",
                unsafe_allow_html=True,
            )
            st.text_area("Results", dev_output_text, height=500, disabled=True)
            st.markdown("---")
            st.subheader("📊 Category Summary")
            summary = {}
            for _, cat, _ in results:
                summary[cat] = summary.get(cat, 0) + 1
            st.table([{"Category": k, "Count": v} for k, v in summary.items()])

        else:
            st.subheader("📋 Grouped by Category")
            grouped = {}
            seen_ids = set()
            for msg, cat, _ in results:
                ids = extract_ids(msg)
                if not ids:
                    continue
                for tid in ids:
                    if tid in seen_ids:
                        continue
                    seen_ids.add(tid)
                    if cat not in grouped:
                        grouped[cat] = set()
                    grouped[cat].add(tid)

            output_lines = []
            for cat, ids in grouped.items():
                output_lines.append(f"{cat}:")
                for tid in sorted(ids):
                    output_lines.append(f" {tid}")
                output_lines.append("")
            output_text = "\n".join(output_lines)
            st.markdown(
                """<style> textarea[disabled] { cursor: default !important; } </style>""",
                unsafe_allow_html=True,
            )
            st.text_area("Grouped Results", output_text, height=500, disabled=True)

            # Export
            export_data = []
            for cat, ids in grouped.items():
                for tid in sorted(ids):
                    export_data.append({"Ticket/ID": tid, "Category": cat})
            df_export = pd.DataFrame(export_data)
            today_str = datetime.today().strftime("%d.%m.%Y")
            filename = f"FF TT Report {today_str}.xlsx"
            output = BytesIO()
            with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
                df_export.to_excel(writer, index=False, sheet_name="Report")
            output.seek(0)
            st.subheader("📥 Export Report to Excel")
            st.download_button(
                label="Download Excel Report",
                data=output,
                file_name=filename,
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )

# ------------------------
# Tab 2: Single message tester
# ------------------------
with tab2:
    st.subheader("🔎 Categorize Single Message")
    test_msg = st.text_input("Enter a message:")
    if test_msg:
        cat, score = categorize_message(test_msg)
        st.write(f"Prediction: **{cat}** (score={score:.2f})")

# ------------------------
# Tab 3: Text Cleansing (your function)
# ------------------------
def filter_messages(file_contents, base_names):
    timestamp_pattern = re.compile(
        r'\[\d{1,2}/\d{1,2}/\d{4} \d{1,2}:\d{2} (?:am|pm)\]'
        r'|\[\d{1,2}:\d{2} (?:am|pm), \d{1,2}/\d{1,2}/\d{4}\]'
        r'|\[\d{1,2}:\d{2}, \d{1,2}/\d{1,2}/\d{4}\]'
        r'|^\[\d{1,2}/\d{1,2}/\d{4} \d{1,2}:\d{2} [APM]{2}]'
    )
    name_patterns = [
        re.compile(rf'\b{re.escape(name)}\b', re.IGNORECASE)
        if re.match(r'\w+', name)
        else re.compile(rf'{re.escape(name)}', re.IGNORECASE)
        for name in base_names
    ]

    filtered_lines, current_message = [], []
    skip_block = False

    for line in file_contents.splitlines():
        if timestamp_pattern.match(line):
            if current_message:
                filtered_lines.append(" ".join(current_message).strip().lower())
                current_message = []
            skip_block = any(pattern.search(line) for pattern in name_patterns)
        if not skip_block:
            current_message.append(line.strip().lower())

    if not skip_block and current_message:
        filtered_lines.append(" ".join(current_message).strip().lower())

    return "\n\n".join(filtered_lines)

def process_uploaded_files_filtering(uploaded_files, base_names):
    all_output = []
    for uploaded_file in uploaded_files:
        file_contents = uploaded_file.read().decode("utf-8")
        filtered_text = filter_messages(file_contents, base_names)
        all_output.append(f"===Cleansed content from {uploaded_file.name}:===\n{filtered_text}")
    return "\n\n".join(all_output)

with tab3:
    st.header("Text Cleansing")
    base_names_input = st.text_area(
        "Enter names (to be removed when cleansing text file)",
        "Hartina, Tina, Normah, Pom, Afizan, Pijan, Ariff, Arep, Arip, Dheffirdaus, "
        "Dhef, Dheff, Dheft, Hazrina, Rina, Nurul, Huda, Zazarida, Zaza, Eliasaph, "
        "Wan, ] : , ] :"
    )
    base_names = [name.strip() for name in base_names_input.split(",")]
    uploaded_files_filter = st.file_uploader(
        "Upload text file for cleansing (max 2)", type="txt", accept_multiple_files=True
    )
    if uploaded_files_filter and len(uploaded_files_filter) > 2:
        st.error("You can only upload up to 2 files.")
    else:
        if uploaded_files_filter and st.button("Cleanse file"):
            filtered_output = process_uploaded_files_filtering(uploaded_files_filter, base_names)
            st.markdown(
                """<style> .stTextArea textarea[disabled] { cursor: default; } </style>""",
                unsafe_allow_html=True,
            )
            st.text_area("Cleansed Output", value=filtered_output, height=400, disabled=True)
            download_data = BytesIO(filtered_output.encode("utf-8"))
            st.download_button(
                label="Download cleansed text",
                data=download_data,
                file_name="cleansed_output.txt",
                mime="text/plain",
            )
