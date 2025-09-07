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
    "Missing Order": [
        "order missing",
        "masih tiada dalam tmf",
        "order tiada di page schedule"
    ],
    "Next Order Activity Not Appear": [
        "not in rol",
        "tak ada next owner",
        "tiada dalam rol",
        "order tiada/ xda owner",
        "mohon next activity tak appear",
        "bantuan next act x appear",
        "order return tiada next owner",
        "mir in progress",
        "mir ra ip",
        "mohon bantuan order tiada dlm rol",
        "mohon bantuan next act x appear"
    ],
    "Revert SWIFT-TMF Order": [
        "tiada di subscriber activity list"
    ],
    "Missing Order Manual Assign Button": [
        "tidak boleh manual assign",
        "x boleh ma",
        "ma button missing",
        "ma button tak appear",
        "tiada aktiviti manual assign",
        "fail to assign"
    ],
    "Reopen Order Proposed Cancel": [
        "tarik balik order",
        "proposed cancel",
        "propose cancelled ke rol",
        "movekan order ke return order list semula",
        "masuk dalam rol semula",
        "masuk dalam basket rol",
        "masuk ke dalam bakul rol"
    ],
    "Update Order Equipment Details": [
        "order modify",
        "error pink",
        "bantu order tambah mesh",
        "tak boleh done",
        "error equipment not same",
        "update cpe rg5 request update ke rg6 tukar wifi 5 kepada wifi 6"
    ],
    "RG6-RG7 Equipment Update": [
        "wifi6 2.5ghz",
        "rg7 ke rg6 combo",
        "rg7 ke rg6 ax3000",
        "order schoolnet",
        "update cpe wifi 7 ke wifi 6 ax3000 combo",
        "minta tukar rg7 ke rg6 combo",
        "tolong tukar eq ke combo biasa order schoolnet",
        "mohon bantu tukar equipment dari rg7 ke rg6 combo ax3000",
        "minta bantuan tukar wifi 7 kepada wifi 6"
    ],
    "Update Order New/ Existing Equipment Info": [
        "ubah equipment ke rg6",
        "nak router baru",
        "nak rg baru",
        "tiada button replace cpe",
        "tukar new vm ke onu combo",
        "tukar new service point ke new combo box",
        "add new equipment combo vdsl ke ftth",
        "order modify tukar wifi ax3000"
    ],
    "Bypass ExtraPort": [
        "tolong bypass extraport",
        "iris access denied",
        "iris tak dapat masuk",
        "dp out",
        "bypass order extra port flag y"
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
    ],
    "Update Contact Number": [
        "tukar hp contact number",
        "update contact nombor",
        "change ctc no",
        "tukar hp contact number"
    ],
    "TT CPE List not Same as Physical": [
        "dekat equipment takda modem",
        "site dah tukar baru"
    ],
    "TT RG6/ Combo Update": [
        "xlepas nak replace combo",
        "sn lama sn baru",
        "combo ctt old rg sn new rg sn",
        "sn lama sn baru customer setuju renew contract",
        "dijanjikan oleh tmpoint",
        "tukar flag ke combobox ctt old rg new",
        "enable flag ctt old rg new rg",
        "mintak tukar combo old combo new combo",
        "mohon enable flag ctt old rg new rg",
        "old rg new rg ctt truckroll",
        "bantuan tukar flag ke combo box mohon update acs bind"
    ],
    "TT HSBA Reappointment": [
        "bantu patch appt am",
        "bantu patch appointment pm",
        "bantuan patchkan appt tmf ctt am pm"
    ],
    "TT Error 400": [
        "ctt error 400",
        "semakan eror 400"
    ],
    "TT Missing": [
        "ctt missing"
    ]
}

# Build initial embeddings
# Now is case-insensitive -- simpler categorization
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
