import streamlit as st
from pathlib import Path
from sentence_transformers import SentenceTransformer, util
import re
import string

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
        "order tiada di page schedule",
        "order pending processing dalam tmf",
        "order tiada dalam oal",
        "x ade dalam oal",
        "mohon bantuan order pending processing",
        "order status pending processing customer complained late installation"
    ],
    "Next Order Activity Not Appear": [
        "not in rol",
        "tak ada next owner",
        "xda owner"
        "tiada dalam rol",
        "order tiada/ xda owner",
        "next activity tak appear",
        "bantuan next act x appear",
        "order return tiada next owner",
        "mir in progress",
        "mir ra ip",
        "bantuan order tiada dlm rol",
        "bantuan next act x appear",
        "order return tapi missing owner",
        "order xde dlm rol",
        "boleh bantu done mir"
    ],
    "Order D&A": [
        "task d&a ip",
        "order dna ip"
    ],
    "Order Returned but Unscheduled": [
        "order return status unschedule"
    ],
    "Revert BAU SWIFT-TMF Order": [
        "tiada di subscriber activity list",
        "revert order id dibawah ke tmf system urgent"
    ],
    "Missing Order Manual Assign Button": [
        "tidak boleh manual assign",
        "x boleh ma",
        "ma button missing",
        "ma button tak appear",
        "tiada aktiviti manual assign",
        "fail to assign",
        "bantu manual assign not appear tqvm",
        "bantuan manual slot x appear"
    ],
    "Reopen Order Proposed Cancel": [
        "tarik balik order",
        "proposed cancel",
        "propose cancelled ke rol",
        "movekan order ke return order list semula",
        "masuk dalam rol semula",
        "masuk dalam basket rol",
        "masuk ke dalam bakul rol",
        "order status propose cancel bantu utk ra semula"
    ],
    "Order Incomplete but Auto Done": [
        "minta tolong check order ttiba done sendiri",
        "bantu ui nk done order terus ke saf",
        "x sempat scan cpe lg"
    ],
    "Order Unsync": [
        "bantuan ui maklum dah done tapi nampak in progress dalam tmf"
    ],
    "Update Order Equipment Details": [
        "order modify",
        "error pink",
        "bantu order tambah mesh",
        "tak boleh done",
        "error equipment not same",
        "update cpe rg5 request update ke rg6 tukar wifi 5 kepada wifi 6",
        "ui nak done order tak lepas keluar error macam ni",
        "tukar new cpe ke existing order force done cancel",
        "upgrade 1gbps tukar modem router tak support 1gbps mohon tukar equipment ke combo wifi 6",
        "ui x blh nk done order equipment tmf existing cpe berlainan",
        "format apa2 masalah tukar new equiment kpd rg6 combo ax3000"
    ],
    "RG6-RG7 Equipment Update": [
        "wifi6 2.5ghz",
        "rg7 ke rg6 combo",
        "rg7 ke rg6 ax3000",
        "order schoolnet",
        "update cpe wifi 7 ke wifi 6 ax3000 combo",
        "tukar rg7 ke rg6 combo",
        "tolong tukar eq ke combo biasa order schoolnet",
        "bantu tukar equipment dari rg7 ke rg6 combo ax3000",
        "bantuan tukar wifi 7 kepada wifi 6",
        "1gb tukarkan combo dengan mesh combo rg7 ke combo ax3000",
        "tukar equipment be72000 ke ax3000 1gbps",
        "tukar equipment rg dan mesh dari rg7 ke rg6 combo 2.5g",
        "flag masih rg7",
        "ru pemasangan pakai rg6",
        "order dh siap nk done kuar error guna wifi rg6 ax3000"
    ],
    "Update Order New/ Existing Equipment Info": [
        "ubah equipment ke rg6",
        "nak router baru",
        "nak rg baru",
        "tiada button replace cpe",
        "tukar new vm ke onu combo",
        "tukar new service point ke new combo box",
        "add new equipment combo vdsl ke ftth",
        "order modify tukar wifi ax3000",
        "sp masih existing",
        "service point masih existing",
        "tukar sp kepada existing add new rg6 ax3000",
        "order upgrade 1g add new combo tukar btu ke existing",
        "add new uonu",
        "tukarkan existing rg ke new combo order modify"
    ],
    "Update Network Info": [
        "dekat granite dp type street cabinet tapi di network on pole",
        "yang betul adalah street cabinet",
        "bantuan dah tukar dp snopbot sama ada error"
    ],
    "Bypass ExtraPort": [
        "tolong bypass extraport",
        "iris access denied",
        "iris tak dapat masuk",
        "dp out",
        "bypass order extra port flag y",
        "bantuan bypass xp",
        "bypass fdp cleansing dp ada reading tracebite tkde reading"
    ],
    "Bypass HSI": [
        "bypass speedtest dan wifi analyzer"
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
        "brn number invalid",
        "no ssm/ic cust dlm tmf tiada",
        "minta ic no untuk order ni",
        "bantuan ic dan ssm no invalid",
        "fail utk update ic br no"
    ],
    "Update Contact Number": [
        "tukar hp contact number",
        "update contact nombor",
        "change ctc no",
        "tukar hp contact number"
    ],
    "TT CPE List not Same as Physical": [
        "dekat equipment takda modem",
        "site dah tukar baru",
        "update cpe di fizikal tidak sama dengan sistem",
        "add onu lama sbb perlu tukar onu sahaja",
        "sn onu fizikal sn onu dalam list bantu update list equipment",
        "update sn onu lama cust",
        "sn rg fizikal tak sama dengan tmf sn fizikal sn tmf",
        "tolong update service point sn dalam tmf dengan fizikal tak sama",
        "add kan onu dalam tmf sn no"
    ],
    "TT RG6/ Combo Update": [
        "xlepas nak replace combo",
        "sn lama sn baru",
        "combo ctt old rg sn new rg sn",
        "sn lama sn baru customer setuju renew contract",
        "dijanjikan oleh tmpoint",
        "tukar flag ke combobox ctt old rg new",
        "enable flag ctt old rg new rg",
        "tukar combo old combo new combo",
        "enable flag ctt old rg new rg",
        "old rg new rg ctt truckroll",
        "tukar flag ke combo box mohon update acs bind",
        "sn lama: rgx sn baru: unc30val",
        "upgrade pakej",
        "no eligible or suitable cpe in handlist for this equipment",
        "xda pertukaran cpe sebelum ini",
        "team tak boleh done ctt buat penukaran rg pakej old on new sn",
        "team nak close ctt x lepas sn lama sn baru",
        "old sn rgx new sn unc",
        "berikan detail spt dibwh ctt no old rg sn new rg sn remark serial number does not exist in your cpe",
        "truckroll cpe replacement old rg newrg",
        "enable flag ctt no old rg new rg",
        "tukar flag tt no new sn old sn tt truckroll",
        "enable flag rg6 di dalam tmf",
        "mengisi google form",
        "ctt no old rg sn new rg sn"
    ],
    "TT RG5 Equipment Update": [
        "ctt sn lama rgx sn baru rgx pakej 100mbps"
    ],
    "TT HSBA Reappointment": [
        "patch appt am",
        "patch appointment pm",
        "patchkan appt tmf ctt am pm"
    ],
    "TT Error 400": [
        "ctt error 400",
        "semakan eror 400",
        "ctt xleh slot appt",
        "tt unable to slot appointment",
        "betulkan cabinet id",
        "bantuan delete olt cab id",
        "betulkan cab id yg betul",
        "tiada detail cab",
        "mohon bantu ctt tiada detail kabinet"
    ],
    "TT with No Activity Worktype": [
        "ctt tiada activity work type"
    ],
    "TT V1P": [
        "utk slot am pm",
        "tt v1p",
        "utk appt ctt hari ini pukul :",
        "utk slot 1-2 @ am pm",
        "appt v1p @ am pm",
        "1-2 appmt am pm tq",
        "1-2 bantuan ra pm tq",
        "ra tt v1p 1-2 am pm",
        "appt ctt v1p @ 1-2"
    ],
    "TT-LR Linkage": [
        "ctt unlink tiada dalam tmf",
        "clear tmf ctt link lr"
    ],
    "TT Missing": [
        "ctt missing",
        "ctt x appear dalam tmf",
        "ctt missing slps return fs troubleshooting",
        "ctt missing. mohon bantuan 1-1",
        "cek kan ctt no tersebut x appear dlm tmf"
    ],
    "TT Duplicate Activity": [
        "cancel 1 activity id a-",
        "cancel act duplicate"
    ]
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
    # Remove ticket/order IDs
    msg = ID_PATTERN.sub("", msg)
    # Lowercase
    msg = msg.lower()
    # Remove punctuation
    msg = re.sub(rf"[{re.escape(string.punctuation)}]", " ", msg)
    # Collapse multiple spaces
    msg = re.sub(r"\s+", " ", msg)
    return msg.strip()
    
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
st.title("📂 TMF Reporter 3")
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
            # Display results in scrollable text area
            st.subheader("Categorized Messages")
        
            dev_output_lines = []
            for msg, cat, score in results:
                dev_output_lines.append(f"[{cat}] ({score}) → {msg}")
            dev_output_text = "\n\n".join(dev_output_lines)
        
            # Custom CSS: default cursor in disabled textarea
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
        
            st.text_area("Results", dev_output_text, height=500, disabled=True)

            st.markdown("---")  # horizontal line
        
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
            seen_ids = set()  # ensure unique IDs across all categories

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

            # Build formatted string
            output_lines = []
            for cat, ids in grouped.items():
                output_lines.append(f"{cat}:")
                for tid in sorted(ids):
                    output_lines.append(f"  {tid}")
                output_lines.append("")  # spacing

            output_text = "\n".join(output_lines)

            # Custom CSS: default cursor in disabled textarea
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
                    export_data.append({"Ticket/ID": tid, "Category": cat})
            
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
