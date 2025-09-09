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
# Categories (to be filled later)
# ------------------------
categories = {
    "Missing Order": [
        "missing",
        "masih tiada dalam tmf",
        "tiada di page schedule",
        "pending processing dalam tmf",
        "tiada dalam oal",
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
        "boleh bantu done mir",
        "order return tak masuk dalam bakul lobs"
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
        "propose cancel masuk dalam rol semula",
        "propose cancel masuk dalam basket rol",
        "propose cancel masuk ke dalam bakul rol",
        "order status propose cancel"
    ],
    "Order Incomplete but Auto Done": [
        "ttiba done sendiri",
        "bantu ui nk done terus ke saf",
        "x sempat scan cpe lg",
        "tiba done sendiri masa tengah nak masuk attachment",
        "mhn bantuan order ni nak  done tetapi tmf auto refresh dan cpe belum pe tak update lg",
        "dalam portal dh completed"
    ],
    "Order Unsync": [
        "ui maklum dah done tapi nampak in progress dalam tmf"
    ],
    "Invalid Order Segment": [
        "tukarkan segment kepada sme order s10",
        "tukar segment kepada sme",
        "segment yang betul adalah sme"
    ],
    "Update Order Equipment Details": [
        "error pink",
        "order tambah mesh",
        # "tak boleh done",
        "error equipment not same",
        "update cpe rg5 request update ke rg6 tukar wifi 5 kepada wifi 6",
        "ui nak done order tak lepas keluar error macam ni",
        "tukar new cpe ke existing order force done cancel",
        "upgrade 1gbps tukar modem router tak support 1gbps mohon tukar equipment ke combo wifi 6",
        "ui x blh nk done order equipment tmf existing cpe berlainan"
        "new equipment kpd rg6 combo ax3000"
        "mhn bantuan order nak done tap error SN baru"
    ],
    "RG6-RG7 Equipment Update": [
        "wifi6 2.5ghz",
        #"rg7 ke rg6 combo",
        #"rg7 ke rg6 ax3000",
        "order schoolnet",
        "order update cpe wifi 7 ke wifi 6 ax3000 combo",
        "order tukar rg7 ke rg6 combo",
        "order tolong tukar eq ke combo biasa order schoolnet",
        "order bantu tukar equipment dari rg7 ke rg6 combo ax3000",
        "order bantuan tukar wifi 7 kepada wifi 6",
        "order 1gb tukarkan combo dengan mesh combo rg7 ke combo ax3000",
        "order tukar equipment be72000 ke ax3000 1gbps",
        "order tukar equipment rg dan mesh dari rg7 ke rg6 combo 2.5g",
        "order mohon bantu flag masih rg7",
        "ru pemasangan pakai rg6",
        "order dh siap nk done kuar error guna wifi rg6 ax3000",
        "order tukar existing equipment combo box ke be7200 wifi 7 ke combo box ax3000",
        "order tukar new equipment dari rg7 ke combo rg6",
        "2508000079245568 bantuan blh tukar kan equipment wifi 7 tukar kepada wifi 6 sebab cust pakai speed 500mbps"
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
        "tukarkan existing rg ke new combo order modify",
        "wifi rg jadi new equipment atau existing",
        "ui tak boleh install sebab status rg delete",
        "tukarkan equipment serv point",
        "service point ke rg6 ax3000",
        "mohon bantuan untuk tukarkan equipment serv point vm ke  rg6 ax3000 order modify 1gb"
    ],
    "Update Network Info": [
        "dekat granite dp type street cabinet tapi di network on pole",
        "yang betul adalah street cabinet",
        "bantuan dah tukar dp snopbot sama ada error",
        "minta bantuan adv ctt 1-115970783375 tak boleh tukar fdp dalam granite di tmf",
        
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
        "mohon bantuan untuk error ni. id :q103945 tt number:1-26818262517 old sn:whps20al2405000239 new sn:whp910dl2501001033"
    ],
    "Invalid ICBRN Number": [
        "verify to proceed failed",
        "v2p failed",
        "icbrn num invalid",
        "brn number invalid",
        "no ssm/ic cust dlm tmf tiada",
        "minta ic no untuk order ni",
        "bantuan ic dan ssm no invalid",
        "fail utk update ic br no",
        "minta ic no detail customer",
        "ic dan ssm no",
        "masuk tapi invalid"
    ],
    "Update Contact Number": [
        "tukar hp contact number",
        "update contact nombor",
        "change ctc no",
        "tukar hp contact number",
        "kindly assist to update ctc"
    ],
    "TT CPE List not Same as Physical": [
        "dekat takda modem",
        "site dah tukar baru",
        "cpe di fizikal tidak sama dengan sistem",
        "add onu lama sbb perlu tukar onu sahaja",
        "onu fizikal onu dalam list bantu list",
        "onu lama cust",
        "fizikal tak sama",
        "service point dengan fizikal tak sama",
        "add kan onu"
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
        "ctt no old rg sn new rg sn",
        "enable flag rg6 di dalam tmf",
        "ctt no old rg sn new rg sn",
        "update old rg new rg serial number",
        "replace rg7 with rg6 in tmf",
        "provide old rg sn and new rg sn",
        "fill google form for rg6 combo update",
        "assalamualaikum & salam sejahtera,  remark : mohon bantuan tukarkan flag ke combo box dan mohon update acs bind.  ctt: 1-116311140378 s/n lama: rgxaztwr2112008593 s/n baru: unc30val2411098930 login id : norsitikamsiah@unifi  terima kasih 🙂 @~taufik z."
        "remark : mohon bantuan tukarkan flag ke combo box dan mohon update acs bind.*  ctt: 1-116270632518 s/n cpe lama: rgxtplec2205052950 s/n cpe baru: unc30val2411098979 login id : tey6686@unifi  terima kasih 🙂",
        "remark : mohon tukarkan flag ke combo box ctt:  1-115865956193  s/n lama: rgxtplc11812010132 s/n baru: uncfh5f32502024708 login id: zaynvivian95@unifi terima kasih",
        "remark : mohon bantuan tukarkan flag ke combobox dan mohon update acs bind.  ctt: 1-116312296513 s/n lama: rgx825dl2012009971 s/n baru : unc30val2501010999 s/n baru (gpon) : cwtcb81ad76c login id : suhairina08@unifi",
        "remark : mohon tukarkan flag ke combo box ctt:  1-115859379625  s/n lama: rgwtull71206003141 s/n baru: unc30val2412061346 login id: virtualmonn78@unifi terima kasih",
        "assalamualaikum & salam sejahtera,  remark : mohon bantuan tukarkan flag ke combo box dan mohon update acs bind.  ctt: 1-116302969812 s/n lama: rgxperdl1707013468 s/n baru: unc30val2411098814 login id : soontat64@unifi  terima kasih 🙂 @~taufik z.",
        "assalamualaikum  remark : mohon tukarkan flag ke combo box  ctt: 1-116311031131 s/n lama: rgx830dl2109006052 s/n baru: unc30val2411083924 logn id : nesznazeir123@unifi  terima kasih 🙂",
        "remark : mohon tukarkan flag ke combo box ctt:  1-115976383308  s/n lama: rg6fhax32208019428 s/n baru: unc30val2411014648 login id: honghai@unifi terima kasih",
        "remark : mohon bantuan tukarkan flag ke combobox dan mohon update acs bind.  ctt: 1-116311067865 s/n lama: rgx842dl1806014106 s/n baru : unc30val2410031830 s/n baru (gpon) : cwtcb8124379 login id : yenthing1638@unifi",
        "remark : mohon bantuan tukarkan flag ke combobox dan mohon update acs bind  ctt no: 1-116152914651 old s/n: rgxdvgdl2110035485 new s/n: unc30val2410074284 login id: eyracik86@unifi  terima kasih",
        "1-115970926445 minta update flag tukar ke combo box. customer sudah tukar pakej ke 300m.",
        "mohon tukar tagging ke combo/rg6 no ctt:  1-115969916464 sn lama:  rgxdvgdl2109025080 sn baru:  unc30val2412069248",
        "enable combo flag 1-116316769444 old sn : rgxaztwr2105049908 new sn : unc30val2412061412",
        "1-115972600583 old sn : rgxperdl1705008311 new sn : unc30val2501002986",
        "assalamualaikum & salam sejahtera, tuan/puan zone ctt coordinator/sp/exec untuk setiap permohonan untuk enable flag rg6 di dalam tmf, mohon berikan maklumat seperti di bawah & mengisi google form yang disertakan (link terkini) & lampirkan screenshot form submission di sini.  info: ctt no: 1-115797194828 old rg sn: pvby2c3015921 new rg sn: unc30val2501057440  https://forms.gle/irzc9fmbnz8yzgtva  team support, mohon bantuan untuk enable flag hanya selepas bukti form submission disertakan.  terima kasih atas kerjasama semua.",
        "truckroll cpe replacement  1-116315005124 old rg rgxdvgdl2109013485 newrg unc30val2501056851",
        "nnt berikan detail spt dibwh: ctt no: nnt berikan detail spt dibwh:  ctt no: 1-116000429645 old rg sn: rgx830dl2109018545 new rg sn: unc30val2501034470 remark : serial number does not exist in your cpe ( truckroll )",
        "mohon enable flag ctt no : 1-116323556555 old rg :aa0c724e4c8bb9cc0 new rg : unc30val2412064290",
        "info: ctt no:  1-116312914201   old rg : rg6dl60z2206082971  new rg sn: uncfh5f32411022966    https://forms.gle/irzc9fmbnz8yzgtva  team support, mohon bantuan untuk enable flag hanya selepas bukti form submission disertakan.  terima kasih atas kerjasama semua."
    ],
    "TT RG5 Equipment Update": [
        "ctt sn lama rgx sn baru rgx pakej 100",
        "ctt :1-116313934089 sn lama:rgxtplc11903028849 sn baru:rgx835dl2307001679 pakej:100mbps",
        ""
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
        "appt ctt v1p @ 1-2",
        "team mohon bantu utk appt ctt ni hari ini pukul",
        "team minta buat appt pukul hari ini"
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

    scores = {
        cat: util.cos_sim(emb, emb_cat).item()
        for cat, emb_cat in category_embeddings.items()
    }
    if scores:
        best_cat, best_score = max(scores.items(), key=lambda x: x[1])
    else:
        best_cat, best_score = "Uncategorized", 0

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

# Function to filter messages based on base names
def filter_messages(file_contents, base_names):
    # timestamp_pattern = re.compile(r'\[\d{2}:\d{2}, \d{1,2}/\d{1,2}/\d{4}\]|^\[\d{1,2}/\d{1,2}/\d{4} \d{1,2}:\d{2} [APM]{2}]')
    timestamp_pattern = re.compile(r'\[\d{1,2}/\d{1,2}/\d{4} \d{1,2}:\d{2} (?:am|pm)\]|\[\d{1,2}:\d{2} (?:am|pm), \d{1,2}/\d{1,2}/\d{4}\]|\[\d{1,2}:\d{2}, \d{1,2}/\d{1,2}/\d{4}\]|^\[\d{1,2}/\d{1,2}/\d{4} \d{1,2}:\d{2} [APM]{2}]')
    name_patterns = [
        re.compile(rf'\b{re.escape(name)}\b', re.IGNORECASE) if re.match(r'\w+', name)
        else re.compile(rf'{re.escape(name)}', re.IGNORECASE)  # No word boundary for non-word characters
        for name in base_names
    ]

    filtered_lines = []
    skip_block = False
    current_message = []

    for line in file_contents.splitlines():
        if timestamp_pattern.match(line):
            if current_message:
                filtered_lines.append(' '.join(current_message).strip().lower())
                current_message = []

            if any(pattern.search(line) for pattern in name_patterns):
                skip_block = True
            else:
                skip_block = False

        if not skip_block:
            current_message.append(line.strip().lower())

    if not skip_block and current_message:
        filtered_lines.append(' '.join(current_message).strip().lower())

    return '\n\n'.join(filtered_lines)

# Function to process all files for Process 1
def process_uploaded_files_filtering(uploaded_files, base_names):
    all_output = []

    for uploaded_file in uploaded_files:
        file_contents = uploaded_file.read().decode("utf-8")
        filtered_text = filter_messages(file_contents, base_names)
        all_output.append(f"===Cleansed content from {uploaded_file.name}:===\n{filtered_text}")
    
    combined_output = "\n\n".join(all_output)
    return combined_output

tab1, tab2, tab3 = st.tabs(["Text Cleansing", "Categorizer", "Categorize Single Message"])

# ------------------------
# Tab 1: File categorizer
# ------------------------
with tab1:
    st.subheader("🧽 Text Cleansing")

    base_names_input = st.text_area(
        "Enter names (to be removed when cleansing text file)",
        "Hartina, Tina, Normah, Pom, Afizan, Pijan, Ariff, Arep, Arip, Dheffirdaus, Dhef, Dheff, Dheft, Hazrina, Rina, Nurul, Huda, Zazarida, Zaza, Eliasaph, Wan, ] : , ] :"
    )
    base_names = [name.strip() for name in base_names_input.split(",")]

    uploaded_files_filter = st.file_uploader(
        "Upload text file for cleansing (max 2)", type="txt", accept_multiple_files=True
    )

    if uploaded_files_filter and len(uploaded_files_filter) > 2:
        st.error("You can only upload up to 2 files.")
    else:
        if uploaded_files_filter and st.button('Cleanse file'):
            filtered_output = process_uploaded_files_filtering(uploaded_files_filter, base_names)

            # CSS to disable cursor change
            st.markdown(
                """
                <style>
                .stTextArea textarea[disabled] { cursor: default; }
                </style>
                """,
                unsafe_allow_html=True
            )

            # Display output
            st.text_area("Cleansed Output", value=filtered_output, height=400, disabled=True)

            # Download
            download_data = BytesIO(filtered_output.encode("utf-8"))
            st.download_button(
                label="Download cleansed text",
                data=download_data,
                file_name="cleansed_output.txt",
                mime="text/plain"
            )
with tab2:
    st.subheader("🛠 Categorizer")
    
    uploaded_file = st.file_uploader("Upload cleansed_output.txt", type=["txt"])

    if uploaded_file:
        # Read messages
        lines = uploaded_file.read().decode("utf-8").splitlines()
        messages = []

        for line in lines:
            if "]" in line and ":" in line:
                msg = line.split(":", 2)[-1].strip()
                if msg and has_valid_id(msg):
                    # filter only with valid IDs
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
            st.subheader("Categorized Messages (by ID)")
        
            dev_output_lines = []
            expanded_results = []  # store per-ID results
        
            for msg, cat, score in results:
                ids = extract_ids(msg)
                if ids:
                    for tid in ids:
                        dev_output_lines.append(f"[{cat}] ({score:.2f}) → {tid} | {msg}")
                        expanded_results.append((tid, cat, score, msg))
                else:
                    # if no ID, still include the message
                    dev_output_lines.append(f"[{cat}] ({score:.2f}) → {msg}")
                    expanded_results.append(("N/A", cat, score, msg))
        
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
            st.markdown("---")
        
            # Summary (count by IDs instead of messages)
            st.subheader("📊 Category Summary")
            summary = {}
            for tid, cat, _, _ in expanded_results:
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
            st.markdown("---")
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
            st.download_button(
                label="Download Excel Report",
                data=output,
                file_name=filename,
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

# ------------------------
# Tab 2: Test single message
# ------------------------
with tab3:
    st.subheader("🔎 Categorize Single Message")
    test_msg = st.text_input("Enter a message:")

    if test_msg:
        cat, score = categorize_message(test_msg)
        st.write(f"Prediction: **{cat}** (score={score:.2f})")
