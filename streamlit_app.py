import streamlit as st
import re
import pandas as pd
from io import BytesIO
from datetime import datetime

# ------------------------
# Regex for ID extraction
# ------------------------
ID_PATTERN = re.compile(
    r"\b(1-[A-Za-z0-9]+|25\d{6,}|Q\d{5,}|TM\d{5,})\b", re.IGNORECASE
)

# ------------------------
# Categories (sample only, update as needed)
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

# ------------------------
# Helper: Categorize message
# ------------------------
def categorize_message(message: str, threshold: float = 0.53):
    msg_lower = message.lower()
    for category, keywords in categories.items():
        for kw in keywords:
            if kw.lower() in msg_lower:
                return category, 1.0  # matched keyword
    return "auto_group", threshold  # fallback


# ------------------------
# Helper: extract IDs
# ------------------------
def extract_ids(message: str):
    return ID_PATTERN.findall(message)


# ------------------------
# Helper: read-only text area
# ------------------------
def readonly_text_area(label, value, height=400, key=None):
    st.markdown(
        """
        <style>
        textarea[readonly] {
            background-color: #f8f9fa !important;
            cursor: text !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    return st.text_area(
        label,
        value=value,
        height=height,
        key=key,
    )


# ------------------------
# Streamlit App
# ------------------------
st.set_page_config(layout="wide")

st.title("📂 Trouble Ticket Categorizer")

tab1, tab2 = st.tabs(["👥 User View", "👨‍💻 Developer View"])

# ========================
# USER VIEW
# ========================
with tab1:
    st.header("User View")

    uploaded_file = st.file_uploader("Upload a text file", type=["txt"])

    if uploaded_file:
        text = uploaded_file.read().decode("utf-8")
        messages = text.splitlines()

        grouped = {}
        for msg in messages:
            ids = extract_ids(msg)
            if not ids:
                continue  # skip messages without valid IDs
            cat, score = categorize_message(msg)
            grouped.setdefault(cat, []).extend(ids)

        # Show grouped output
        user_output_lines = []
        for cat, ids in grouped.items():
            user_output_lines.append(f"\n{cat}:")
            for i in ids:
                user_output_lines.append(i)

        user_output = "\n".join(user_output_lines)

        st.markdown("---")
        readonly_text_area("Grouped Results", user_output, height=300, key="user_view")

        # Export to Excel
        st.markdown("---")
        if st.button("📤 Export to Excel"):
            df = []
            for cat, ids in grouped.items():
                for i in ids:
                    df.append({"Category": cat, "ID": i})

            df = pd.DataFrame(df)
            output = BytesIO()
            with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
                df.to_excel(writer, index=False, sheet_name="Report")

            today = datetime.today().strftime("%d.%m.%Y")
            filename = f"FF TT Report {today}.xlsx"

            st.download_button(
                label="⬇️ Download Excel",
                data=output.getvalue(),
                file_name=filename,
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )

# ========================
# DEVELOPER VIEW
# ========================
with tab2:
    st.header("Developer View")

    sample_message = st.text_area("Test single message:", "", height=100)

    if st.button("Categorize Sample"):
        ids = extract_ids(sample_message)
        if ids:
            cat, score = categorize_message(sample_message)
            st.write(f"Category: {cat}, Score: {score}, IDs: {ids}")
        else:
            st.write("⚠️ No valid IDs found in message.")

    uploaded_file = st.file_uploader("Upload a text file for analysis", type=["txt"], key="dev")

    if uploaded_file:
        text = uploaded_file.read().decode("utf-8")
        messages = text.splitlines()

        results = []
        for msg in messages:
            ids = extract_ids(msg)
            if not ids:
                continue
            cat, score = categorize_message(msg)
            results.append((msg, cat, score))

        dev_output_lines = []
        for msg, cat, score in results:
            dev_output_lines.append(f"[{cat}] ({score}) → {msg}")

        dev_output_text = "\n".join(dev_output_lines)

        st.subheader("Categorized Messages")
        readonly_text_area("Results", dev_output_text, height=400, key="dev_output")

        # Summary
        st.subheader("📊 Category Summary")
        summary = {}
        for _, cat, _ in results:
            summary[cat] = summary.get(cat, 0) + 1
        st.table([{"Category": k, "Count": v} for k, v in summary.items()])
