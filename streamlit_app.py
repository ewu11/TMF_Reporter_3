import streamlit as st
from pathlib import Path
from sentence_transformers import SentenceTransformer, util
import re

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
    "Next Order Activity Not Appear": [
        "not in rol",
        "tak ada next owner",
        "tiada dalam rol",
        "order tiada/ xda owner",
        "mohon bantuan next activity tak appear"
    ],
    "Update Order Equipment Details": [
        "order modify",
        "error pink"
    ],
    "RG6-RG7 Equipment Update": [
        "wifi6 2.5ghz",
        "rg7 ke rg6 combo",
        "rg7 ke rg6 ax3000",
        "order schoolnet",
        "update cpe wifi 7 ke wifi 6 ax3000 combo"
    ],
    "Update Order New/ Existing Equipment Info": [
        "ubah equipment ke rg6",
        "nak router baru",
        "nak rg baru",
        "tiada button replace cpe",
        "tukar new vm ke onu combo"
    ],
    "Bypass ExtraPort": [
        "tolong bypass extraport",
        "iris access denied",
        "iris tak dapat masuk",
        "dp out"
    ],
    "Unable to Swap Number": [
        "order whp/ v1p",
        "tak dapat swap number",
        "call/ contact ftc/ tva",
    ],
    "Invalid ICBRN Number": [
        "verify to proceed failed",
        "v2p failed",
        "icbrn num invalid",
        "brn number invalid"
    ]
}

# Build initial embeddings
# Now is case-insensitive -- simpler categorization
category_embeddings = {
    cat: model.encode([s.lower() for s in sentences], convert_to_tensor=True).mean(dim=0)
    for cat, sentences in categories.items()
}

# Threshold
SIMILARITY_THRESHOLD = 0.45
new_groups = {}
group_counter = 1

# ------------------------
# Regex for ticket/order/ID
# ------------------------
# ID_PATTERN = re.compile(r"(1-[A-Za-z0-9]+|250\d+|Q\d+|TM\d+)", re.IGNORECASE)

# only match IDs that are NOT part of a larger alphanumeric token
ID_PATTERN = re.compile(r"(?<!\w)(?:1-[A-Za-z0-9]+|250\d+|Q\d+|TM\d+)(?!\w)", re.IGNORECASE)


def has_valid_id(msg: str) -> bool:
    return bool(ID_PATTERN.search(msg))

def extract_ids(msg: str):
    return ID_PATTERN.findall(msg)

# ------------------------
# Categorization function
# Now case-insensitive
# ------------------------
def clean_message(msg: str) -> str:
    # Remove ticket/order IDs
    msg = ID_PATTERN.sub("", msg)
    # Lowercase + strip
    msg = msg.lower().strip()
    return msg
    
def categorize_message(msg):
    global group_counter
    clean_msg = clean_message(msg)  # normalize before encoding
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
st.title("📂 TMF Report 3")
st.write("Enhanced report categorizer.")

tab1, tab2 = st.tabs(["Categorizer", "Categorize Single Message"])

# ------------------------
# Tab 1: File categorizer
# ------------------------
with tab1:
    uploaded_file = st.file_uploader("Upload cleansed_output.txt", type=["txt"])

    if uploaded_file:
        # Read messages
        lines = uploaded_file.read().decode("utf-8").splitlines()
        messages = []
        for line in lines:
            if "]" in line and ":" in line:
                msg = line.split(":", 2)[-1].strip()
                if msg and has_valid_id(msg):   # filter only with valid IDs
                    messages.append(msg)

        st.success(f"Loaded {len(messages)} messages (filtered unnecessary text)")

        # Run categorization
        results = []
        for msg in messages:
            cat, score = categorize_message(msg)
            results.append((msg, cat, round(score, 2)))

        # Choose View Mode
        view_mode = st.radio("Select View Mode", ["Developer View", "User View"])
        st.markdown("---")  # horizontal line

        if view_mode == "Developer View":
            # Display results
            st.subheader("Categorized Messages")
            for msg, cat, score in results[:100]:  # preview first 100
                st.markdown(f"**[{cat}]** ({score}) → {msg}")

            # Summary
            st.subheader("📊 Category Summary")
            summary = {}
            for _, cat, _ in results:
                summary[cat] = summary.get(cat, 0) + 1
            st.table([{"Category": k, "Count": v} for k, v in summary.items()])

        else:
            # User-friendly grouping
            st.subheader("📋 Grouped by Category")
            grouped = {}
            for msg, cat, _ in results:
                ids = extract_ids(msg)
                if not ids:
                    continue
                if cat not in grouped:
                    grouped[cat] = set()
                grouped[cat].update(ids)

            # Build formatted string
            output_lines = []
            for cat, ids in grouped.items():
                output_lines.append(f"{cat}:")
                for tid in sorted(ids):
                    output_lines.append(f"  {tid}")
                output_lines.append("")  # spacing

            output_text = "\n".join(output_lines)

            # Display inside scrollable text area
            # Custom CSS: make text_area cursor default (not text cursor)
            st.markdown(
                """
                <style>
                textarea[disabled] {
                    cursor: default !important;
                }
                </style>
                """,
                unsafe_allow_html=True
            )
            
            # Custom CSS: make text_area cursor default (not text cursor)
            st.markdown(
                """
                <style>
                textarea[disabled] {
                    cursor: default !important;
                }
                </style>
                """,
                unsafe_allow_html=True
            )
            
            # Display inside scrollable, taller text area
            st.text_area("Grouped Results", output_text, height=500, disabled=True)
            
            # ------------------------
            # Export to Excel
            # ------------------------
            import pandas as pd
            from io import BytesIO
            from datetime import datetime
            
            st.markdown("---")  # horizontal line
            
            # Build dataframe for export
            export_data = []
            for cat, ids in grouped.items():
                for tid in sorted(ids):
                    export_data.append({"Category": cat, "Ticket/ID": tid})
            
            df_export = pd.DataFrame(export_data)
            
            # Generate filename with today's date
            today_str = datetime.today().strftime("%d.%m.%Y")
            filename = f"FF TT Report {today_str}.xlsx"
            
            # Convert to Excel in memory
            output = BytesIO()
            with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
                df_export.to_excel(writer, index=False, sheet_name="Report")
            output.seek(0)

            st.subheader("📥 Export Report to Excel")
            
            # Download button
            st.download_button(
                label="Download Excel Report",
                data=output,
                file_name=filename,
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )


# ------------------------
# Tab 2: Test single message
# ------------------------
with tab2:
    st.subheader("🔎 Categorize Single Message")
    test_msg = st.text_input("Enter a message:")
    if test_msg:
        cat, score = categorize_message(test_msg)
        st.write(f"Prediction: **{cat}** (score={score:.2f})")
