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
# Biasing rules
# ------------------------
def apply_bias(msg: str, scores: dict) -> dict:
    """
    Adjust similarity scores based on keyword rules.
    """
    text = msg.lower()

    if "tt" in text or "ctt" in text:
        if "tiada slot" in text or "no slot" in text:
            scores["TT Error 400"] = scores.get("TT Error 400", 0) + 0.3
            scores["Next Order Activity Not Appear"] = scores.get("Next Order Activity Not Appear", 0) - 0.1
        if "boleh tukar combo ke" in text:
            scores["TT RG6/ Combo Update"] = scores.get("TT RG6/ Combo Update", 0) + 0.1
            scores["New/ Existing/ Delete Equipment Info Update"] = scores.get("New/ Existing/ Delete Equipment Info Update", 0) - 0.1
        if "error 400" in text or "err 400" in text:
            scores["TT Error 400"] = scores.get("TT Error 400", 0) + 0.3
        if ("100 mbps" not in text and "300 mbps" in text) and ("old rg" in text or "new rg"):
            scores["TT RG6/ Combo Update"] = scores.get("TT RG6/ Combo Update]", 0) + 0.3
            scores["TT RG5 Equipment Update"] = scores.get("TT RG5 Equipment Update", 0) - 0.3
        if ("tukar" in text and "combo" in text) or "flag" in text:
            scores["TT RG6/ Combo Update"] = scores.get("TT RG6/ Combo Update", 0) + 0.2
            scores["RG6 - RG7 Equipment Info Update"] = scores.get("RG6 - RG7 Equipment Info Update", 0) - 0.1
        if "cab" in text or "cabinet" in text or "fdp" in text or "dp" in text or "fdc" in text:
            scores["TT Error 400"] = scores.get("TT Error 400", 0) + 0.2
            scores["Manual Assign Button not Appear"] = scores.get("Manual Assign Button not Appear", 0) - 0.1
        if "combo" in text:
            scores["TT RG6/ Combo Update"] = scores.get("TT RG6/ Combo Update", 0) + 0.2
            scores["Release Assign to Me"] = scores.get("Release Assign to Me", 0) - 0.1
        if "missing" in text:
            scores["TT Missing"] = scores.get("TT Missing", 0) + 0.2
            if ("dalam" in text or "dlm" in text) and ("tmforce" in text or "tmf" in text):
                scores["TT Missing"] = scores.get("TT Missing", 0) + 0.3
        if ("dalam" in text or "dlm" in text) and ("tmforce" in text or "tmf" in text):
            scores["TT Missing"] = scores.get("TT Missing", 0) + 0.3
        if "trig" in text:
            scores["TT Missing"] = scores.get("TT Missing", 0) + 0.2
        if "link" in text or "lr" in text:
            scores["TT - LR Linkage"] = scores.get("TT - LR Linkage", 0) + 0.2
            scores["Manual Assign Button not Appear"] = scores.get("Manual Assign Button not Appear", 0) - 0.1
        if "v1p" in text and "slot" in text:
            scores["TT V1P"] = scores.get("TT V1P", 0) + 0.2
            scores["Order Capping Issue"] = scores.get("Order Capping Issue", 0) - 0.1
        if (re.search(r"slot", text)) and (re.search(r"ap(.)?(.)?(.)?(.)?(.)?(.)?", text)):
            scores["TT Error 400"] = scores.get("TT Error 400", 0) + 0.3
            scores["Release Assign to Me"] = scores.get("Release Assign to Me", 0) - 0.3
        if (re.search(r"cpe", text)) and (re.search(r"sn.*exist(.)?(.)?(.)?", text)):
            scores["TT RG6/ Combo Update"] = scores.get("TT RG6/ Combo Update", 0) + 0.2
            scores["Update Order Equipment Details"] = scores.get("Update Order Equipment Details", 0) - 0.1
        if (re.search(r"en(.)?m(.)?d(.)?r(.)?d", text)):
            scores["TT RG6/ Combo Update"] = scores.get("TT RG6/ Combo Update", 0) + 0.2
            scores["Update Order Equipment Details"] = scores.get("Update Order Equipment Details", 0) - 0.1
        if (re.search(r"cpe.*swap.*serial number.*exist", text)):
            scores["TT RG6/ Combo Update"] = scores.get("TT RG6/ Combo Update", 0) + 0.2
            scores["TT Error 400"] = scores.get("TT Error 400", 0) - 0.1
        if (re.search(r"del(.)?(.)?(.)?", text) and re.search(r"(.)?d(c|p)( id)?", text)):
            scores["Update Granite Network Details"] = scores.get("Update Granite Network Details", 0) + 0.2
            scores["Invalid ICBRN Number"] = scores.get("Invalid ICBRN Number", 0) - 0.1
        if (re.search(r"(f)?d(c|p)?", text) or re.search(r"cab(inet)?", text)) and re.search(r"map(ping)?", text):
            scores["TT Error 400"] = scores.get("TT Error 400", 0) + 0.2
            scores["Update Granite Network Details"] = scores.get("Update Granite Network Details", 0) - 0.2
        if ("no slot" in text):
            scores["TT Error 400"] = scores.get("TT Error 400", 0) + 0.3
            scores["Update Order Equipment Details"] = scores.get("Update Order Equipment Details", 0) - 0.1
            scores["Order Capping Issue"] = scores.get("Order Capping Issue", 0) - 0.1
            scores["Release Assign to Me"] = scores.get("Release Assign to Me", 0) - 0.1
        if (re.search(r"stat(.)?(.)?(.)?", text) and re.search(r"m(.)?s(.)?(.)?", text)) and re.search(r"new", text):
            scores["TT Missing"] = scores.get("TT Missing", 0) + 0.2
        if (re.search(r"no slot", text)):
            scores["TT Error 400"] = scores.get("TT Error 400", 0) + 0.1
        if (re.search(r"cancel(.)?(.)?(.)? s(.)?l(.)?(.)?.*(act(.)?(.)?(.)?(.)?(.)?|ak(.)?(.)?(.)?(.)?(.)?(.)?|xtvt)", text)):
            scores["TT Duplicate Activity"] = scores.get("TT Duplicate Activity", 0) + 0.1
        if re.search(r"patch.*(am|pm)", text):
            scores["TT HSBA Reappointment"] = scores.get("TT HSBA Reappointment", 0) + 0.1 
            scores["TT Missing"] = scores.get("TT Missing", 0) - 0.1
        if (re.search(r"(whp|v1p)", text) and re.search(r"slot(.)?(.)?(.)?", text)):
            scores["TT V1P"] = scores.get("TT V1P", 0) + 0.2
    
    if "order" in text or "oder" in text:
        if "tukar equipment ke combo ax3000" in text or "customer package" in text:
            scores["RG6 - RG7 Equipment Info Update"] = scores.get("RG6 - RG7 Equipment Info Update", 0) + 0.1
            scores["New/ Existing/ Delete Equipment Info Update"] = scores.get("New/ Existing/ Delete Equipment Info Update", 0) - 0.1
        if "tukar eqpmnt jadi combo" in text or "mahukan combo" in text:
            scores["New/ Existing/ Delete Equipment Info Update"] = scores.get("New/ Existing/ Delete Equipment Info Update", 0) + 0.1
            scores["Order Capping Issue"] = scores.get("Order Capping Issue]", 0) - 0.1
        if "ra" in text and ("dalam" in text or "dlm" in text) and ("rtn" in text or "return" in text or "returned" in text):
            scores["Next Order Activity Not Appear"] = scores.get("Next Order Activity Not Appear", 0) + 0.1
            scores["Order Missing/ Pending Processing"] = scores.get("Order Missing/ Pending Processing", 0) - 0.1
        if ("sn:" in text or "sn" in text or "s/n" in text or r"s\/n" in text or "s/n:" in text or r"s\/n:" in text) and ("ctc" not in text or "contact" not in text):
            scores["RG6 - RG7 Equipment Info Update"] = scores.get("RG6 - RG7 Equipment Info Update", 0) + 0.2
            scores["Update Contact Number"] = scores.get("Update Contact Number", 0) - 0.3
        if ("equipment" in text or "eqp" in text or "eqmnt" in text) and "vendor" in text:
            scores["Update Order Equipment Details"] = scores.get("Update Order Equipment Details", 0) + 0.2
            scores["New/ Existing/ Delete Equipment Info Update"] = scores.get("New/ Existing/ Delete Equipment Info Update", 0) - 0.1
        if "vdsl" in text or "ftth" in text:
            scores["New/ Existing/ Delete Equipment Info Update"] = scores.get("New/ Existing/ Delete Equipment Info Update", 0) + 0.2
            scores["TT RG6/ Combo Update"] = scores.get("TT RG6/ Combo Update", 0) - 0.1
        if ("ma" in text or "manual assign" in text):
            scores["Manual Assign Button not Appear"] = scores.get("Manual Assign Button not Appear", 0) + 0.1
            if ("btn" in text or "butang" in text or "button" in text):
                scores["Manual Assign Button not Appear"] = scores.get("Manual Assign Button not Appear", 0) + 0.3
        if "tukar" in text and "rg7" in text and "rg6" in text:
            scores["RG6 - RG7 Equipment Info Update"] = scores.get("RG6 - RG7 Equipment Info Update", 0) + 0.2
            scores["Order Capping Issue"] = scores.get("Order Capping Issue", 0) - 0.1
        if "oal" in text:
            scores["Order Missing/ Pending Processing"] = scores.get("Order Missing/ Pending Processing", 0) + 0.2
            scores["Manual Assign Button not Appear"] = scores.get("Manual Assign Button not Appear", 0) - 0.1
        if "rol" in text and ("tiada" in text or "xda" in text) and ("dalam" in text or "dlm" in text):
            scores["Next Order Activity Not Appear"] = scores.get("Next Order Activity Not Appear", 0) + 0.2
            scores["Release Assign to Me"] = scores.get("Release Assign to Me", 0) - 0.1
        if ("mesh" in text and "tambah" in text) or "tick" in text:
            scores["Update Order Equipment Details"] = scores.get("Update Order Equipment Details", 0) + 0.2
            scores["Manual Assign Button not Appear"] = scores.get("Manual Assign Button not Appear", 0) - 0.1
        if (re.search(r"inst(.)?(.)?(.)?(.)?(.)?(.)?(.)?(.)?", text) and re.search(r"d(.)?n(.)?", text) and re.search(r"m(.)?s(.)?(.)?", text)):
            scores["Unsync Order"] = scores.get("Unsync Order", 0) + 0.2
            scores["Next Order Activity Not Appear"] = scores.get("Next Order Activity Not Appear", 0) - 0.1
        if (re.search(r"ru|ui", text) and re.search(r"done(.)?(.)?(.)? o(.)?der", text) and re.search(r"err(.)?(.)?", text)):
            scores["Update Order Equipment Details"] = scores.get("Update Order Equipment Details", 0) + 0.2
            scores["Release Assign to Me"] = scores.get("Release Assign to Me", 0) - 0.1
        if (re.search(r"(butang|b(.)?t(.)?(.)?(.)?)", text) and re.search(r"cc", text)):
            scores["CC Not Appear"] = scores.get("CC Not Appear", 0) + 0.2
            scores["Release Assign to Me"] = scores.get("Release Assign to Me", 0) - 0.1
        if (re.search(r"new(k(.)?n)?", text) and re.search(r"(vm|btu|sp|rg)", text)):
            scores["New/ Existing/ Delete Equipment Info Update"] = scores.get("New/ Existing/ Delete Equipment Info Update", 0) + 0.2
            scores["Force Done Order"] = scores.get("Force Done Order", 0) - 0.1
        if (re.search(r"revert", text) and re.search(r"ke tmf system", text)):
            scores["Revert BAU SWIFT-TMF Order"] = scores.get("Revert BAU SWIFT-TMF Order", 0) + 0.2
            scores["Reopen Proposed Cancel Order"] = scores.get("Reopen Proposed Cancel Order", 0) - 0.1
        if (re.search(r"(nova|tm(.)?f(orce)?(.*siap.*p(.)?s(.)?(.)?(.)?)?)", text) and re.search(r"(m(.)?s(.)?(.)?.*process(.)?(.)?(.)?)", text)):
            scores["Unsync Order"] = scores.get("Unsync Order", 0) + 0.2
            scores["TT TMF-Physical CPE Unsync"] = scores.get("TT TMF-Physical CPE Unsync", 0) - 0.1
    
    if "tukar kan equipment ke existing" in text or "hanya tambah fixed ip bukan tukar brg" in text:
        scores["New/ Existing/ Delete Equipment Info Update"] = scores.get("New/ Existing/ Delete Equipment Info Update", 0) + 0.1
        scores["RG6 - RG7 Equipment Info Update"] = scores.get("RG6 - RG7 Equipment Info Update", 0) - 0.1

    if ("order" in text or "oder" in text) and ("blh" in text or "boleh" in text) and "done" in text and "equipment semua existing" in text:
        scores["Update Order Equipment Details"] = scores.get("Update Order Equipment Details]", 0) + 0.1
        scores["New/ Existing/ Delete Equipment Info Update"] = scores.get("New/ Existing/ Delete Equipment Info Update", 0) - 0.1

    if ("slot" in text or "slotkan" in text or "set" in text) and ("appointmnet" in text or "appmnt" in text or "apt" in text or "appmnt" in text or "appt" in text) and ("v1p" in text or "whp" in text):
        scores["TT V1P"] = scores.get("TT V1P", 0) + 0.2
        scores["TT HSBA Reappointment"] = scores.get("TT HSBA Reappointment", 0) - 0.1

    if "fdp" in text or "cab" in text:
        scores["Update Granite Network Details"] = scores.get("Update Granite Network Details", 0) + 0.2
        scores["New/ Existing/ Delete Equipment Info Update"] = scores.get("New/ Existing/ Delete Equipment Info Update", 0) - 0.1

    if (re.search(r"add new (rg|sp|btu|service point)", text) and "missing" in text):
        scores["New/ Existing/ Delete Equipment Info Update"] = scores.get("New/ Existing/ Delete Equipment Info Update", 0) + 0.2
        scores["RG6 - RG7 Equipment Info Update"] = scores.get("RG6 - RG7 Equipment Info Update", 0) - 0.2

    if ("appointment" in text or "appmnt" in text or "appt" in text or "apt" in text or "appment" in text) and "set" in text and ("pukul" in text or "jam" in text):
        scores["TT V1P"] = scores.get("TT V1P]", 0) + 0.2
        scores["Order Capping Issue"] = scores.get("Order Capping Issue", 0) - 0.1
        if re.search(r"\b1-2\d{10,11}\b", text) or re.search(r"slot.*id", text):
            scores["TT V1P"] = scores.get("TT V1P]", 0) + 0.4
            scores["TT HSBA Reappointment"] = scores.get("TT HSBA Reappointment", 0) - 0.1
            scores["TT Unsync"] = scores.get("TT Unsync", 0) - 0.1
            scores["Release Assign to Me"] = scores.get("Release Assign to Me", 0) - 0.1
            scores["Invalid ICBRN Number"] = scores.get("Invalid ICBRN Number", 0) - 0.1
    
    # if "cpe" in text:
    #     scores["TT RG6/ Combo Update"] = scores.get("TT RG6/ Combo Update", 0) + 0.2
    #     scores["Update Contact Number"] = scores.get("Update Contact Number", 0) - 0.1

    if ("ctt" in text or "tt" in text) and ("view slot" in text or "skillset" in text or "slot" in text or "mapping" in text or "cab" in text or "cabinet" in text or "dp" in text):
        scores["TT Error 400"] = scores.get("TT Error 400]", 0) + 0.2
        scores["TT V1P"] = scores.get("TT V1P", 0) - 0.1
        scores["TT TMF-Physical CPE Unsync"] = scores.get("TT TMF-Physical CPE Unsync", 0) - 0.1
        scores["New/ Existing/ Delete Equipment Info Update"] = scores.get("New/ Existing/ Delete Equipment Info Update", 0) - 0.1

    if ("valid rg" in text) and ("tukar" in text or "tkr" in text) and ("cpe" in text):
        scores["TT RG6/ Combo Update"] = scores.get("TT RG6/ Combo Update]", 0) + 0.3
        scores["RG6 - RG7 Equipment Info Update"] = scores.get("RG6 - RG7 Equipment Info Update", 0) - 0.1
        scores["Update Order Equipment Details"] = scores.get("Update Order Equipment Details", 0) - 0.1 
        scores["New/ Existing/ Delete Equipment Info Update"] = scores.get("New/ Existing/ Delete Equipment Info Update", 0) - 0.1 
        scores["TT TMF-Physical CPE Unsync"] = scores.get("TT TMF-Physical CPE Unsync", 0) - 0.1
        scores["Release Assign to Me"] = scores.get("Release Assign to Me", 0) - 0.1
    
    if "order dah refresh network tp failed" in text:
        scores["Update Granite Network Details"] = scores.get("Update Granite Network Details", 0) + 0.3

    if "tiada next owner" in text or re.search(r"next act not appear", text):
        scores["Next Order Activity Not Appear"] = scores.get("Next Order Activity Not Appear", 0) + 0.2

    if "source skill blank" in text:
        scores["TT - Activity Work Type Blank"] = scores.get("TT - Activity Work Type Blank", 0) + 0.2

    if re.search(r"(.)?(.)? fail(.)?(.)? verify (.)?(.)?br(.)? cust(.)?(.)?(.)?(.)?", text):
        scores["Invalid ICBRN Number"] = scores.get("Invalid ICBRN Number", 0) + 0.2
    
    if "pending processing" in text:
        scores["Order Missing/ Pending Processing"] = scores.get("Order Missing/ Pending Processing", 0) + 0.1

    if ("ma" in text or "manual assign" in text) and "appear" in text:
        scores["Manual Assign Button not Appear"] = scores.get("Manual Assign Button not Appear", 0) + 0.1

    if "granite" in text or ("refresh" in text and "fail" in text):
        scores["Update Granite Network Details"] = scores.get("Update Granite Network Details", 0) + 0.1

    if (("err" in text and "400" in text) or re.search(r"ctt not slot", text) or re.search(r"(.)?tt.*tiada slot appt", text)):
        scores["TT Error 400"] = scores.get("TT Error 400", 0) + 0.2

    if "add" in text and "new" in text and ("sp" in text or "service point" in text):
        scores["New/ Existing/ Delete Equipment Info Update"] = scores.get("New/ Existing/ Delete Equipment Info Update", 0) + 0.2

    if ("remove" in text or "release" in text) and ("assign" in text or "me" in text) and "tag" in text:
        scores["Release Assign to Me"] = scores.get("Release Assign to Me", 0) + 0.2

    if (re.search(r"m(.)?as(.)?k", text) and re.search(r"rol", text)):
        scores["Next Order Activity Not Appear"] = scores.get("Next Order Activity Not Appear", 0) + 0.2

    if (re.search(r"ada", text)) and (re.search(r"tm(.)?f(orce)?", text)) and (re.search(r"h(.)?n(.)?(.)?", text)):
        scores["TT TMF-Physical CPE Unsync"] = scores.get("TT TMF-Physical CPE Unsync", 0) + 0.1
        scores["TT V1P"] = scores.get("TT V1P", 0) - 0.1

    if (re.search(r"(tiada|xd(.)?)", text) and re.search(r"(.)?(.)?l", text)):
        scores["Order Missing/ Pending Processing"] = scores.get("Order Missing/ Pending Processing", 0) + 0.2

    if (re.search(r"ra", text) or re.search(r"ap(.)?(.)?(.)?(.)?(.)?(.)?(.)?(.)?(.)?", text)) and re.search(r"v1p", text) or re.search(r"1-2\d{9,11}", text):
        scores["TT V1P"] = scores.get("TT V1P", 0) + 0.2

    if re.search(r"25\d{12,14}", text) and re.search(r"slot", text):
        scores["Order Unable to Slot"] = scores.get("Order Unable to Slot") + 0.1
        scores["TT Error 400"] = scores.get("TT Error 400", 0) - 0.1

    if (re.search(r"appear", text) and re.search(r"cc", text)):
        scores["CC Not Appear"] = scores.get("CC Not Appear", 0) + 0.2

    if (re.search(r"t(.)?mb(.)?h", text) and re.search(r"(btu|sp|service point)", text) and re.search(r"d(.)?t(.)?(.)?l", text)):
        scores["TT TMF-Physical CPE Unsync"] = scores.get("TT TMF-Physical CPE Unsync", 0) + 0.2
        scores["Order Missing/ Pending Processing"] = scores.get("Order Missing/ Pending Processing", 0) - 0.1

    if (re.search(r"d(.)?a", text) and re.search(r"(m(.)?s(.)?(.)?.*)?(ip|in(.)?progre(.)?(.)?)", text)):
        scores["Order D&A In-Progress"] = scores.get("Order D&A In-Progress", 0) + 0.2
        scores["Release Assign to Me"] = scores.get("Release Assign to Me", 0) - 0.1

    if (re.search(r"upd(.)?(.)?(.)?", text) and re.search(r"stat(.)?(.)?", text) and re.search(r"r(.)?t(.)?(.)?n(.)?(.)?", text)):
        scores["Order Returned but Unscheduled"] = scores.get("Order Returned but Unscheduled", 0) + 0.2
        scores["TT TMF-Physical CPE Unsync"] = scores.get("TT TMF-Physical CPE Unsync", 0) - 0.1

    if re.search(r"100mb(.)?(.)?.*t(.)?k(.)?(.)?.*combo", text):
        scores["TT RG6/ Combo Update"] = scores.get("TT RG6/ Combo Update", 0) + 0.2
        scores["Order In-Progress but Auto Done"] = scores.get("Order In-Progress but Auto Done", 0) - 0.1

    # Cap scores between 0.0 and 1.0
    scores = {k: max(0.0, min(v, 1.0)) for k, v in scores.items()}

    return scores

# ------------------------
# Categories (to be filled later)
# ------------------------
categories = {
    "Order Missing/ Pending Processing": [
        "order pending processing  1-115106826851",
        "order status pending processing in tm force since 25/8/25. iris reprot sd4753885 lodge on 28/8/25 until now still remain pend processing. pls assist asap. customer complained on late installation.",
        "1-cbeor26 - order ni masih belum ada dlm tmf",
        "2509000080460280 - mohon bantu ru inform tak nampak order dalam dalam tmf tapi order dah assign atas id dia  id : q104758 ru : mohd azizi bin mahzun annuar",
        "tak nampak order dalam dalam tmf",
        "tapi order dah assign atas id dia",
        "pending processing",
        "mir dah done tapi ra masih ip. tapi dekat order activity list tulis record not found",
        "order tak appear dalam oal tmf",
        "order ra xappear di scheduled page",
        "tiada dalam oal",
        "team order tk nampak dlm id nx2502095"
    ],
    "Next Order Activity Not Appear": [
        "tiada dlm rol..tq",
        "tiada dlm rol",
        "return tak masuk dalam bakul lobs/cxm",
        "done mir?",
        "order return, next activity tak appear",
        "salam team,  trigger semula order ini di page rrol. activity mir di siebel masih in progress 1-114085556792",
        "not in rol | hi team  order returned due to customer tapi tak ada next owner acitivty. tqvm",
        "salam, order tiada dalam rol utk ra. 1-115950242200. tq",
        "next activity x appear",
        "....2508000079730082 order ni tiada di scheduled page tp ada di page order search",
        "no pending user, tq  2509000080365841",
        "00378065688 - order 1-cbi3c7w 00378035878 - order : 1-cbhvjqo  masukkan semula dalam oal...ui tidak nampak order sebab mdf sudah done activity remove mdf jumpering...tq",
        "masukkan semula dalam oal",
        "npua",
        "order npua",
        "tiada dalam oal",
        "tiada dalam rol ..mohon bantu.",
        "slot 10/9 tapi mir masih ip",
        "salam 2509000080381065 masukan order dalam basket jcc. tq",
        "order tiada di schedule page 1-115878860659 tq",
        "order ra, next act (act cc) not appear",
        "next act (act cc) not appear",
        "(act cc) not appear",
        "order mir ip",
        "mir ip",
        "order roc no next owner",
        "order roc",
        "no next owner",
        "order return tiada next owner  1-112726082278",
        "order ini   1-112532187256 ,  belum ra tp dalam nova . aktiviti ra sudah done . dalam tmf , order tiada masuk dalam returned order . iris : sd4790757 ./ tk",
        "order rtn x masuk basket jcc",
        "order sip, act not appear",
        "order ni xda ble update any activity",
        "update dlm rol"
    ],
    "CC Not Appear": [
        "cc not appear",
        "bantu cc x kuar",
        "cc tak appear 1-114576765037",
        "1-116383637429 mohon bantu cc not appear",
        "order tiada butang cc di nova"
    ],
    "Order Returned but Unscheduled": [
        "2509000080687620 return jadi unscheduled",
        "return jd unscheduled",
        "ra date hari ini jam4 pm..jadi unscheduled 2509000081041721",
        "1-116327524637 - bantuan order return jd unscheduled"
        "return jd unscheduled",
        "order return jadi unschedule",
        "update p status rtn"
    ],
    "Order Capping Issue": [
        "bukakan capping utk id q003975 nak slot whp 14 line 1 lokasi (cust minta buat semua harini)",
        "gomen kementerian sumber manusia  dlm portal tmf ui q003975 ada dpt 13 order. tp ui hanya ada 10 order shj 1-cbjdpgv 1-cbjzqxz 1-cbjx85i 1-cbjy128 1-cbjz0vp 1-cbjz165 1-cbjx8k4 1-cbjy0o0 1-cbjx22i 1-cbizjkr"
    ],
    "Order Address Unsync": [
        "2509000082201564, address x sama dengan taas.  dalam tmf(alamat lama) : no 5a level 1 bazar mara jalan pantai batu 4 nil 71050 port dickson, negeri sembilan malaysia  dalam taas(yg betul) : 15-g jalan prima 4 g  lukut prima lukut negeri sembilan malaysia 71010",
        "address unsync iris:  sd4815419 order no: 2509000081940810 taas : view tmf order status :  processing tmf service item status :returned issue: unsync address  at tmf/ unable to reschedule correct add: 697-28-04 blok 697 kondominium desa kiara ss21 pj jalan damansara 28 ftth blok 697 kondominium desa kiara ss21  dlm tmf: no b-13a-11 level 13a blok b empire city jalan damansara damansara perdana pju 8 47820 petaling jaya, selangor malaysia"
    ],
    "Order Unable to Slot": [
        "calendar no slot orders num :2509000082543130",
        "order status unscheduled xboleh slot",
        "order tidak boleh book appointment."
    ],
    "Force Done Order": [
        "order fd - mhn bypass - tq",
        "tukarkan cpe kepada existing, order force done",
        "order forcedone for cancellation",
        "order forcedone for cancellation. hsi failed to done. mohon bypass hsi"
    ],
    "RG6 - RG7 Equipment Info Update": [
        "equipment rg7 ke rg6 ax3000 (router dan juga mesh)",
        "tukar equipment ke rg6 2.5",
        "(1gbps) ... mintak tukarkan tagging wifi 7 kepada wifi 6",
        "tukarkan ax3000 kepada rg7",
        "mintak tukarkan equipmnt rg7 ke combo 2.5g skli mesh. tq",
        "combo rg7 ke rg6 order 1gbps unc30val2412065163 unc30val2412065169",
        "salam kak mintak tukar dari wifi7 ke wifi6 unc30val2412061878",
        "unifi home 2gbps with netflix bantuan tukar rg7 ke rg6",
        "mohon bantu tukarkan rg vdsl single box kepada rg 7..tq",
        "1.)order schoolnet- mohon bantu tukar rg7 ke existing rg6 - combo ax3000",
        "order schoolnet- mohon bantu tukar rg7 ke rg6 - combo ax3000",
        "equipment rg dan mesh... dari rg7 ke rg6 combo 2.5g",
        "brg dari rg7 ke combo ax3000 4.zone 5 : bw",
        "flag masih rg7 1-115972738401",
        "flag masih rg7, ru pemasangan pakai rg6  1-115559812996",
        "1-115109708172    mohon bantu tukar cpe kepada rg6",
        "2509000080219214 | isu fail update equipment  mohon bantuan untuk update cpe wifi 7 ke wifi6 ax3000 combo.  no order: 2509000080219214 s/n no 1: unc25val2412025438 s/n no 2 mesh: unc25val2412025439",
        "tma q104706 1-113191017898 mntk tlg tukar eq ke combo biasa, order schoolnet. tq",
        "mohon bantuan tukarkan equipment rg7 ke rg6 ax3000 router & mesh....order modify 1gb  2509000080869301",
        "mohon bantuan flag masih rg7 ru x lepas nak donekan order..cust sudah nak keluar rumah  1-115865565435",
        "1-115874070879 mohon bantu. masih error utk done. mohon bantu trigger wifi6",
        "trigger wifi6",
        "updated equipment pda rg6 ax3000..order 1gb",
        "ubah ke wifi 7 .customer behaviour. order downgrade ke 1gb. customer tetap nak wifi 7 rujukan dr cff",
        "dear all mohon bantu 1-116411918915 consumer minta tukar equipment service point ke rg7 sebab flag detail dlm tmf be7000 , tq",
        "tukar equipment service point ke rg7 sebab flag detail dlm tmf be7000",
        "2509000080972753 order modify mohon bantu tukar rg7 kepada rg6..tq",
        "minta tukar cpe rg6, sebab rg7 tiada stok.. customer setuju",
        "order 1gb tp dlm tmf guna rg7. minta ubah equipment dr rg7 ke rg6 combo ax3000. tq",
        "updated mesh dan wifi pada rg6 ax3000..order 1gb",
        "mohon bantuan untuk tukar cpe rg7 ke ax3000 (1000006013) combo ax3000 (rg6) 2509000081963621",
        "update detail dibawah:  no order: 2509000081969792 no s/n: unc30val2410035159",
        "tukarkan rg7 ke rg6 kerana rg7 tiada stock 4.zone 5 : bw",
        "order id: 2509000080433971 2.ru id: q106095 3.masalah/req : tukarkan rg7 ke rg6 kerana rg7 tiada stock 4.zone 5 : bukit mertajam",
        "wifi 7 ke wifi 6",
        "order modify 1gb",
        "2509000083144439-update wifi 7 ke wifi 6,order modify 1gb"
    ],
    "New/ Existing/ Delete Equipment Info Update": [
        "semakan order upgrade pakej 2gb tp dlm tmf equipment yg new vm shj",
        "tukarkan vm kepada rg6 tq",
        "cpe new ke existing, order force done cancel - iris error",
        "tukar wifi(rg) jdi new equipment atau existing.order hanya tambah mesh tpi rg dh delete .ui tak boleh install sebab status rg delete sd4773474",
        "tukar servis poin ke combo box",
        "order tmforce team id: order date: 8/9/2025 order no: remark: cust apply upgrade 1gbps, tapi dalam tmf hanya tukar modem. router sedia ada tak support 1gbps. mohon tukar equipment ke combo wifi 6"
        "relocate bantuan ui x blh nk done order, equipment dlm tmf adalah existing  tp cpe berlainan.. mesh rg6 dlink, uonu ax3000 skyworth",
        "tukarkan  ata jd existing sbb kat site customer dah ada existing ata 4 port"
        "sp ke combo rg6 ax3000",
        "bleh check dk oder ni.dkleh nk complete..nk scan combo xleh",
        "ru xleh scan cpe combo..dh try log out login pun masih sama",
        "bg isu order ini  1-115954384256 order ni cust cadari kap baru tukar rg6 combo minggu lepas ..harinhi keluar oder modify tukar onu pulak..blh check blik tk order ni  tq"
        "order ni boleh tukar equipment dari rg4 ke rg6 ax3000 x... upgrade 500mbps",
        "existing equipment combo box be7200 wifi 7 ke combo box ax3000 2.5ghz 4.zone : pgc",
        "tukar mesh ke existing ye, order schoolnet ( sekolah dah ada existing mesh )",
        "tukarkan equipment serv point vm ke new combo rg7 order modify 1gb  1-115960101049",
        "tukar equipment new mesh wifi ke combo rg6, modify 500mbps  2508000079549409",
        "tukarkan existing service point ke new . ru maklum model ni tak  support universal..order relocate",
        "1-115865470728 modify 09.30  mhn bantuan tukarkan new sp ke new combo (rg6) order modify 1g...tq",
        "1-115791690115 modify upgrade 1gb - mohon bantuan tukar rg7 ke rg6 ax3000 - tukar existing rg4 ke new rg6 ax3000 - delete new service point",
        "1-115803658502 mohon bantu.. service point masih existing",
        "mohon bantu.service point masih new.mohon tukar existing & add new rg combo box 1-115965608879",
        "mohon bantuan add new combo box. order modify infra dari vdsl ke ftth tq  1-115625268896",
        "mohon bantu jadikan cpe existing. order relocate guna cpe asal: 1-115392918697",
        "jadikan cpe existing",
        "order relocate guna cpe asal",
        "1-115874174567 modify upgrade 1gb - delete new service point",
        "delete new",
        "delete new service point",
        "modify upgrade",
        "tukarkan  8 dect (new cpe) kpd existing krn cstmer guna pabx tq",
        "minta tukar btu ke new..order x share btu",
        "boleh check ini equipment sdh add 1 ata pun masihh sama",
        "oder. ni 1gb tp tukar modem sj btolke",
        "jdkan new btu dekat equipment..order relocate sebelum ini cust pakai copper (sbvm)",
        "add new sp, dekat dlm order tak de sp",
        "add modem...order modify infra.",
        "updated order 2gb guna combo wifi 7.dlam tmf ada modem shj,.",
        "tukar btu ke new..order x share btu -1-116438099969",
        "tukar sp kepada existing, order relocate btu masih baru",
        "mintak existing cpe onu order relocate",
        "existing cpe onu",
        "order modify tolong tukar eqiupment new ke existing",
        "eqiupment new ke existing",
        "order relocate. dekat site customer tiada combo box tetapi dalam tmf cpe info ada existing combo box(uonu).  mohon add existing onu & existing rg5. kerana di site x ada uonu",
        "sistem ada modem sahaja. tapi cust request router",
        "delete 1 unifi plusbox dlm tm force di dalam nova pakej hnya 1 unifi plusbox 1-116377604103",
        "1-115786010445 req tukar ke existing equipment ,customer nak kekal router lama dia( router beli sendiri)",
        "tukar ke existing equipment",
        "customer nak kekal router lama dia( router beli sendiri",
        "mohon bantu tukarkan new serv point kepada combo rg6 ax3000 order modify 1gb  1-116442585197",
        "mohon bantu tukar rg4 kpd rg6 combo (ax3000) & jadikan sp kpd existing. | 1-116490051611",
        "rg4 kpd rg6 combo (ax3000) & jadikan sp kpd existing",
        "tukar eqpmnt jadi combo x? order modi tapi tukar onu shj. pelanggan mahukan combo.",
        "add new btu order modify infra",
        "minta bantu tukar new service point ke new combo box order modify infra dari vdsl ke ftth  1-116379809716",
        "add new service point 1-106717512390",
        "ubah equipment p existing",
        "add new rg combo kerana dalam tmf missing cpe rg.order modify add mesh  sn rg combo : unc30val2411021166",
        "add on rg - action code new | 2509000082740713",
        "order modify,mhn newkan vm"
    ],
    "Update Order Equipment Details": [
        "nk done keluar error seperti berikut..tq router-rg6dlax32311015973 mesh rg6-rg6dlax32311015977",
        "order nak done tap error sn baru 👉 unc30val2501038792 ....tq",
        "order modify add no shj x dapat nak done order,ade error pink.  order:2509000080361602  del:cpogelgc2501002409   onu:cw7220al2508010552 cw7220al2508010553  ata:atalc4nr2501008265",
        "ru tak boleh done, order add new ata sahaja, tak tukar cpe lain, tak boleh done, sn ata new: atanov042505000248",
        "intelligence netcare sdn bhd mohon bantu ui tak dapat complete order,order delete fix ip,cust pakai cpe sendiri tak guna cpe tm.tq",
        "bypaskn rg dgn mesh,main tak sama nak done tak lepas",
        "update rg kpd mode combo.  2509000081013138",
        "-116572297563 sn lama-rg6dlax32401051346 snbaru-uon30val2501028982 id:q105041",
        "2509000081081426- ru dpt error seperti di atas semasa nak scan cpe",
        "ui maklum order tak boleh done, dapat error, equipment semua existing order no : 2509000081327334",
        "done order sebab equipment vendor tak sama . main ( skyworth ) , mesh (dlink) , boleh minta eqp vendor mesh tukar ke skyworth",
        "order cust tambah mesh tp existing kat tmf rg6, kat rumah cust rg4.. boleh ke ui nak tick tambah dua2 baru skyworth",
        "2509000080337336 failed to scan barcode. getting error. mohon bantuan  uncfh5f32502005906 mesh uncfh5f32502009890",
        "2504000064702662 blh buang equipment ata, cust nk pakai unifi sahaja",
        "ru tak boleh done order keluar error ni",
        "xlepas nak bind cpe",
        "order relocate tak boleh close equipment tak same.."
    ],
    "Unable to Swap Number": [
        "1-c1z5awa order whp tidak dapat swap number ...sudah call ftc dia suruh refer jcom",
        "order whp ui nak done kan order tapi tak boleh",
        "tak lepas perform swap number",
        "order whp ui nak done kan order tapi tak boleh"
    ],
    "Bypass Extraport": [
        "bypass xp",
        "bypass fdp clensing ..dekat dp ada reading .. dalam tracebite tkde reading ..",
        "bypass extraport. iris access denied",
        "1-115636242567 tolong bypass extraport lhd_c001_dp0071 09out",
        "iris error.. xboleh log iris  2509000081023987 mohon bantu bypass ep.no order tiada dlm apps/troika.ep flag y",
        "bypass extraport 4.zone : bw  gambar failed dah attach",
        "bypass extraport verification failed",
        "bypass xp",
        "by pass extraport. tiada dalam portal  2509000081355996 | sd4795360",
        "bypass extraport order tiada dlm troika iris tak boleh nak masuk.. ✅",
        "q003354 2509000081098168 tlg bypass extraport"
    ],
    "Bypass HSI": [
        "minta bypass speedtest dan wifi analyzer sbb oder force done",
        "1-112885939768 hsi failed - order fdc .. ble bantu bypass iris nak balik awal kali",
        "ui maklum dah up kat site tapi xleh nak done hsi xlepas. plgn dah boleh online. boleh bypass hsi ke?",
        "nak done hsi xlepas",
        "plgn dah boleh online",
        "boleh bypass hsi ke",
        "order force done. verify hsi failed",
        "verify hsi failed",
        "minta bypass hsi,order dh psg tadi,cust terus off equipement dan dh bercuti 4.zone 5 : sja",
        "ru dapat error nak donekan hsi 1-118001286324, interner dah up"
    ],
    "Bypass IPTV": [
        "tlg bypass iptv,order modify"
    ],
    "Manual Assign Button not Appear": [
        "tiada button manual assign",
        "hsba manual slot x appear",
        "manual assign x appear",
        "salam, order tiada button manual assign",
        "manual assign tak appear",
        "1-c1z5awa button manual assign tidak appear tqvm",
        "team tiada button ma. cxm info sdh done cc 1-114890055872",
        "1-114089879333 mohon bantuan order ra tidak boleh manual assign",
        "dear all 1-112893674327///jabatan sukarelawan malaysia rela negeri sarawak order pada 04092025. tiada aktivi manual assign. failed to assign. sd4769621",
        "1-cbc64pd ma not appear",
        "ma not appear",
        "team ..tiada button ma. cc sdh done",
        "1-cbjz7uk ma not appear",
        "ma",
        "selamat pagi, order tiada butang ma di tmf 1-116446720439 tq",
        "order tiada butang ma di tmf",
        "tiada butang ma",
        "order tiada butang ma di tmf",
        "tak keluar ma",
        "bantuan order tak keluar ma tq 1-116382004540",
        "1-114656007104 / ma not appear",
        "1-cbkn1lw manual assign not appear",
        "takdak m.assign",
        "ra,manual assign not appear",
        "manual assign xappear"
    ],
    "Order Error 500": [
        "Tak boleh nk return",
        "bantuan id Q102829 x blh nk return order,ui dh try relogin pun sama"
    ],
    "Invalid ICBRN Number": [
        "no ssm/ic cust, dlm tmf tiada",
        "minta ic no untuk order ni.. xd dekat detail customer",
        "ui fail utk update ic/br no",
        "ic dan ssm no. dah masuk tp invalid, tak lepas done.",
        "order x bole done ic cust invalid.",
        "team sdh masukkan no ic sperti penama didalam order tetapi tidak dapat",
        "ui inform, masukkan ic cust tp tak boleh atau kne masuk br no? tp takde info br number",
        "2509000081253533 | ,format apa2 masalah / bypass extraport/btu/s.test/dll 1.order .2509000081253533 2masalah/req: ic  no tak betul 3.q001808 4.zone 5 : byb",
        "req: ic  no tak betul",
        "perlu masukkn no ic",
        "br no. spt semakan dgn cust: 519537-x  minta bantuan utk error diatas. ru sudah masukkan br no. yg betul spt semakan dgn cust",
        "dah masukan no ic betul tapi dapat error dalam tmf:",
        "ssm number in taas 1337374-m  2509000082430600 mohon bantu. team dah masukan no ssm yg betul tapi error nk done dlm tmf",
        "dah try ic dan ssm pun still tak boleh",
        "ru failed verify br cust"
    ],
    "Invalid Order Segment": [
        "segment  kepada sme. order s10. tq 2509000080543072",
        "2509000080425247 mohon bantuan segment kepada sme. segment yang betul adalah sme",
        "order ni kepada segment sme, order biz",
        "order number: 2509000080880931 account: tecstar auto sdn.bhd.  mohon bantuan, customer sme, pakej business, tapi segment group consumer.",
        "mohon pembetulan ub unifi unsync segment 2509000080460927 93 coffee  & tea industries sdn. bhd. 1593513-a  taas sme tmf consumer (to change sme)"
    ],
    "SLIMS CPE Issue": [
        "1-116094469447 mintak bantuan,cpe lock.. sn- uonzteh92502027553 mac id : 648505dced20"
    ],
    "Update Contact Number": [
        "ctc no",
        "team, mohon bantu 1-116379701877 ctc no 0166855599 di tmf. thanks"
        "team, kindly assist to ctc 0139433051  for order id :2508000078626166 . tq.",
        "team, mohon bantu utk tukar hp contact number 0129898252 utk order 2508000079145394, tq",
        "2509000080420382 – pls change contact no. to 012-8461870. tq.",
        "tukarkan contact number kepada 01139851691",
        "contact number.....tq order:  2508000078206670 / sw food & bar concepts sdn bhd ctc no: 0166812312",
        "contact number. ....tq order:  2508000079407460 / yap mun kit contact no mobile: 01112240477 contact no home : 0177060128",
        "team, mohon bantu 2509000080776773,  upadte ctc no 0172678851 di tmf. tq",
        "upadte ctc no 0172678851 di tmf",
        "update contact number 01159893540. tq",
        "update contact number : 012-8188024"
    ],
    "Release Assign to Me": [
        "dapat error nak ra order",
        "x boleh book appt, keluar error seperti di atas...tq",
        "release order ""asssign to me""...team maklum nk release tp hilang dlm tmf...tq",
        "tak boleh nk slot ke tarikh yg available.  tmf keluar pop up ""fail..slot not available. kindly refresh the page""",
        "ra kan order ni harini. ui dah siap pemasangan",
        "ra kan order ni harini",
        "remove tag assign to me dr team ui yang sdh lock"
    ],
    "Order D&A In-Progress": [
        "task d&a ip",
        "d&a masih in progress",
        "d&a ip",
        "dna ip",
        "d&a masih ip",
        "assalamualaikum 1-116382418268 - d&a masih in progress, mohon bantu",
        "d&a inprogress",
        "order 1-116064693500 d&a inprogress (sd4785299 )",
        "dna & osm ip",
        "dna & osm ip",
        "bantuan dna & osm ip,,,cust urgent req install harini jgk  tq",
        "d&a in progress",
        "task d&a ip",
        "urgent. cust nk slot harini. tp d&a osm masih ip."
    ],
    "Unsync Order": [
        "ui maklum dah done tapi nampak in progress lagi dlm tmf. tk",
        "2508000079434474 nak tanya knp takde button book appt. (nak retime...)",
        "ui kata xde button utk on site. tk",
        "mntk tlg tngk order dah done tpi masih in progress",
        "jadikan order kepada status completed sebab pihak vasp daripada digi confirm order telah completed di fizikal...dlm tmf ui tidak boleh done sebab semua activity sudah completed",
        "1-115390711719-mohon bantuan check order ni minta tlg cek dh completed tp masih inprogress",
        "tlg cek dh completed tp masih inprogress",
        "ui tak boleh donekan order",
        "dah done tapi still in progress",
        "order ni dah onsite.. tp tetibe back default",
        "installation status done ,order masih status returned/processing",
        "order ni kuning dari pagi tadi..sya tiada usik apa2..masalah tidak dpt task",
        "xboleh done order",
        "order siap pasang/completed pada 14/10/2025. tetapi dalam nova & tmf , order ini apt scheduled telah berubah tarikh kepada 17/10/2025. dalam nova/tmf order masih status processing. mohon bantu utk trigger status completed",
        "oder ni sy x active.. tapi dekat saya dah hijau"
    ],
    "Revert BAU SWIFT-TMF Order": [
        "1-cbga8gd | assalam team, mohon revert id dibawah ke tmf system. urgent!!! tq 1-cbga8gd 1-cbgnu4p",
        "1-cbeoqy6 - status completed done in tmf 3/9/2025 tq 1-cbeub0k - status completed - swift tq 1-cbeor26 - masih tiada di subscriber activity list swift @ tmf"
    ],
    "Reopen Proposed Cancel Order": [
        "order status propose cancel - utk ra semula - tq",
        "status order propose reappt......order ada di page order search tapi tiada di rrol",
        "2508000079953130 | hi team , tarik balik order ini from propose cancel pool, order can be installed update by ru team. tqvm team",
        "order proposed cancel 2508000079156490 bantuan... order proceed pemasangan...",
        "revert ke rol",
        "keluar error ni untuk onsite. order force done cancel",
        "revert ke basket jcc, plgn nk proceed pasang 1-106182430364",
        "untuk buka order rcl ke rrol  2508000078490589 1-115407078949 2509000080848720 2509000080183389 2509000082054485",
        "masukkan semula order dlm resolved..order nak proceed"
    ],
    "Order In-Progress but Auto Done": [
        "tiba sendiri masa tengah nak masuk attachment . detail speedtest semua takmasuk pun lgi sbb nk bypass btu",
        "terus ke saf..x sempat scan cpe lg, tq",
        "ui dah complete order tapi masih x hilang. ui xleh nak buat next order. tk",
        "2509000081788103 bantuan ui blm sempat scan cpe ..order sudah status completed",
        "team baru buat hsi .. terus auto done order..",
        "order xdan scan barcode sn dah skip pi ke saf ..",
        "ui blum complete tp tetiba dh status hijau dlm tmf."
    ],
    "TT RG5 Equipment Update": [
        "cpe rg5 & onu ke new router combo",
        "ada error untuk done  1-116311485233  sn router : vdlf24dl2111011465",
        "rg5 minta bantuan  ctt no :1-116329668255 package : 100mbps sn lama :vdlinnne2007051543 sn baru : vdlf24dl2111009558",
        "rg 5..equitment combo.. tapi cust pakai rg 5"
    ],
    "TT RG6/ Combo Update": [
        "salam team x dapt close ctt untuk pertukaran router minta bantuan  ctt no : 1-116315464203 package : 100mbps sn lama : rg6fhax32208032145 sn baru :unc30val2410080579",
        "sn rg lama:rgxaztwr2205016652 sn combo baru : unc30val2410081417 no ctt:1-116314999694 package :100mbps mohon bantu tm point telah janjikan set combo tq ya",
        "assalammualaikum, mohon bantuan untuk error ni.   id :q103945 tt number:1-26818262517 old sn:whps20al2405000239 new sn:whp910dl2501001033",
        "salam,mohon tukar tagging,cust pekej 300mb  info: ctt no: 1-116182596740 old rg sn: rgxtplc12102000715 new rg sn: uncfh5f32411019303",
        "info:tukar rg5 ke combo 100 mbps ctt no:1-115950652482 old rg sn:rgxaztwr2207005074 new rg sn:uncfh5f32411032691",
        "remark : mohon bantuan tukarkan flag ke combo box dan mohon update acs bind.*  ctt: 1-116270632518 s/n cpe lama: rgxtplec2205052950 s/n cpe baru: unc30val2411098979 login id : tey6686@unifi  terima kasih 🙂",
        "assalamualaikum & salam sejahtera,  remark : mohon bantuan tukarkan flag ke combo box dan mohon update acs bind.  ctt: 1-116311140378 s/n lama: rgxaztwr2112008593 s/n baru: unc30val2411098930 login id : norsitikamsiah@unifi  terima kasih 🙂 @~taufik z.",
        "remark : mohon bantuan tukarkan flag ke combo box dan mohon update acs bind.*  ctt: 1-115722470135 s/n cpe lama: rgxperdl1709003741 s/n cpe baru: unc30val2411098813 login id : isminorhanbinismai@unifi  terima kasih 🙂 @~taufik z.",
        "remark : mohon tukarkan flag ke combo box ctt:  1-115865956193  s/n lama: rgxtplc11812010132 s/n baru: uncfh5f32502024708 login id: zaynvivian95@unifi terima kasih",
        "remark : mohon bantuan tukarkan flag ke combobox dan mohon update acs bind.  ctt: 1-116312296513 s/n lama: rgx825dl2012009971 s/n baru : unc30val2501010999 s/n baru (gpon) : cwtcb81ad76c login id : suhairina08@unifi",
        "remark : mohon tukarkan flag ke combo box ctt:  1-115976383308  s/n lama: rg6fhax32208019428 s/n baru: unc30val2411014648 login id: honghai@unifi terima kasih",
        "1-115970926445 minta update flag tukar ke combo box. customer sudah tukar pakej ke 300m.",
        "mohon tukar tagging ke combo/rg6 no ctt:  1-115969916464 sn lama:  rgxdvgdl2109025080 sn baru:  unc30val2412069248",
        "ctt :1-116313934089 sn lama:rgxtplc11903028849 sn baru:rgx835dl2307001679 pakej:100mbps",
        "enable combo flag 1-116316769444 old sn : rgxaztwr2105049908 new sn : unc30val2412061412",
        "1-115972600583 old sn : rgxperdl1705008311 new sn : unc30val2501002986",
        "truckroll cpe replacement  1-116315005124 old rg rgxdvgdl2109013485 newrg unc30val2501056851",
        "nnt berikan detail spt dibwh: ctt no: nnt berikan detail spt dibwh:  ctt no: 1-116000429645 old rg sn: rgx830dl2109018545 new rg sn: unc30val2501034470 remark : serial number does not exist in your cpe ( truckroll )",
        "ctt no :  1-116323270125 old rg :  vdlaztds2008003140 new rg : unc30val2412066279  mohon enable flag",
        "mintk tukar flag tt no:1-116321092138 new sn:uncfh5f32411001561 old sn:rgxaztwr2207027329 tt truckroll",
        "info: ctt no:  1-116312914201   old rg : rg6dl60z2206082971  new rg sn: uncfh5f32411022966    https://forms.gle/irzc9fmbnz8yzgtva  team support, mohon bantuan untuk enable flag hanya selepas bukti form submission disertakan.  terima kasih atas kerjasama semua.",
        "mohon bantu kembali, ui update cpe bertukar ke rg5, req update ke rg6  order no:1-112022311713 minta bantuan utk tukar wifi 5 kepada wifi 6",
        "1-115880410969..xlepas nk replace combo",
        "ctt - 1-11588495833 sn lama - rgx835dl2307000024 sn baru - unc30val2501040315  bantuan - customer setuju renew contract... dh dijanji kan oleh tm point..  tq",
        "no ctt:1-115787315701 old ce:com60vdl2304001102 new cpe:cw7220al2508011852",
        "1-116380461845 | salam, mohon tukar tagging,cust pakej 100mbps. sudah tiada stock rg5 dan sbvm.  info: ctt no: 1-116380461845 old rg sn: rgx842dl2107015642 new rg sn: uncfh5f32412032170",
        "sudah tiada stock rg5 dan sbvm",
        "mohon bantuan..1-115961396302..tukar onu,xboleh nk done tt..sn onu lama onunokia1803010458..sn baru uonwfh2a2505004787",
        "mohon bantuan @en md rad mohon bantuan ctt no :1-116321367562 no siri lama :vdlinnne1911037019 no siri baru  :unc30val250134169",
        "salam bantuan team tak boleh tutup ctt tk boleh swap cpe  team tukar combo box  ctt no 1-116375717449",
        "ctt tk boleh swap cpe  team tukar combo box",
        "ada error untuk done  1-116336590617  sn baru : uncfh5f32502020536 sn lama : rg6wfh1y220701631",
        "ada error untuk done",
        "1-116325374545 - tt ni sblm ni pd 13/8/25 tukar rg6 & mesh wifi kepada combo box & mesh wifi.. customer pilih option renew contract..dlm rekod tt yg keluar balik masih cpe lama rg6..mohon bantuan untuk update combo box baru dlm equipment customer",
        " salam team x dapt close ctt untuk pertukaran router combo minta bantuan  ctt no :1-115980922928 package : 300mbps sn lama :vdlinnne1911015109 sn baru : unc30val2412063318",
        "info:100 mbp combo ctt no: 1-116384133328 old rg sn:vdlaztds2109009572 new rg sn:uncfh5f32411042436 ctt save program dr telemarketing request renew contract tukar router wifi 6",
        "ctt no : 1-116444069593 rg old : rgxtplc11901017459 rg new : uncfh5f32411020443",
        "no tt:1-116480219495 remark: ctt truck roll cust tukar ke combo bantuan tukar flag",
        "remark: ctt truck roll cust tukar ke combo bantuan tukar flag",
        "assalamualaikum & salam sejahtera, tuan/puan zone ctt coordinator/sp/exec untuk setiap permohonan untuk enable flag rg6 di dalam tmf, mohon berikan maklumat seperti di bawah & mengisi google form yang disertakan (link terkini) & lampirkan screenshot form submission di sini.  info: ctt no: 1-116443193056  old rg sn: rgx835dl2307011663  new rg sn: unc30val2501028768  https://forms.gle/irzc9fmbnz8yzgtva  team support, mohon bantuan untuk enable flag hanya selepas bukti form submission disertakan.  terima kasih atas kerjasama semua.",
        "minta update  no ctt : 1-116483225947 sn lama : rg6gleex2204026011 sn baru : unc30val2501033247  mesh lama : rg6gleex2204025382 mesh baru : unc30val2501033248",
        "ctt no :1-116446218772 package : 300mbps sn lama :rgxperdl1707017730 sn baru : uncfh5f32412028927 tm janji dengan customer untuk tukar router yang terbaru",
        "tm janji",
        "tm janji dengan customer",
        "tm janji dengan customer untuk tukar router yg terbaru",
        "x dapt close ctt untuk pertukaran router combo",
        "x dapt close ctt",
        "pertukaran router combo",
        "ctt no :1-116448629262 package : 500mbps sn lama :rgx830dl2108032644 sn baru : uncfh5f32412028466",
        "package : 500mbps sn lama :rgx830dl2108032644 sn baru : uncfh5f32412028466",
        "old s/n: uncfh5f32406024342 new s/n: uncfh5f32412023471",
        "old s/n: uncfh5f32406024342 new s/n: uncfh5f32412023471 xboleh swap mesh",
        "nk closed tak blh..tukar combo box",
        "ctt: 1-116478717255  mesh  old sn: com60vdl2305001957 new sn: unc25val2412025598 fill up ....☝...tq.......zita..",
        "team tak boleh tutup ctt tk boleh swap cpe  team tukar combo box  ctt no 1-116320070051",
        "ctt tk boleh swap cpe  team tukar combo box",
        "salam team x dapt close ctt untuk pertukaran router combo minta bantuan  ctt no :1-116618160695 package : 300mbps sn lama :rgx8825dl2101000969 sn baru : uncfh5f32412028562",
        "1-116573977145 old cpe. rg6gleex2205030063 new cpe. unc30val2501035091",
        "info: ctt no: 1-116534218062  old rg sn : com6ovdl2306000083 new rg sn:unc30val2501032667",
        "mohon tukar tagging,cust dah pakai rg6.  info: ctt no: 1-1156539355602 old rg sn: rg6dl60z2208039147 new rg sn: uncfh5f32411019291",
        "tuan/puan zone ctt coordinator/sp/exec untuk setiap permohonan untuk enable flag rg6 di dalam tmf, mohon berikan maklumat seperti di bawah & mengisi google form yang disertakan (link terkini) & lampirkan screenshot form submission di sini.  info: ctt no: 1-116539355602 old rg sn: rg6dl60z2208039147 new rg sn: uncfh5f32411019291  https://forms.gle/irzc9fmbnz8yzgtva  team support, mohon bantuan untuk enable flag hanya selepas bukti form submission disertakan.  terima kasih atas kerjasama semua.",
        "enable flag hanya selepas bukti form submission disertakan",
        "ctt no: 1-116539355602 old rg sn: rg6dl60z2208039147 new rg sn: uncfh5f32411019291",
        "double confirm ctt ni boleh tukar combo ke? sbb dia 100mbps. tapi kalau tgk notes sini dia mintak tukar tanpa renew contract. boleh proceed ke ?",
        "ctt no:1-116619626412 old rg sn: uncfh5f32412032743 new rg sn: unc30val2410077802 300 mbps",
        "ctt no:1-116619626412 old rg sn: uncfh5f32412032743 new rg sn: unc30val2410077802",
        "old rg sn: unc",
        "new rg sn: unc",
        "error cpe swap  ctt :1-116670617245 svc no :6049767605 rg old sn:rg6wfh1y2208006277 nnew:uncfh5f32412014687",
        "tkr combo script: speed 500mbps  info: ctt no: 1-116680278435 old rg sn:rgx842dl2101014587 new rg sn: unc25val2412004704",
        "ctt number :- 1-116573718745 old rg - rgxaztwr2208044963 new rg - unc30val2410053158 remark cff :cust upgrade pakej mohon tukar cpe combo",
        "ctt choppy =1-116671146138 ,,,team perlu tukar rg5 kpd combo mohon bantu",
        "ctt ni pakai 100mbps. rg5 dah takda dekat lo cheras. so mcm mana ya boleh ke kalau nak pakai combo?",
        "update cpe 1-116434798675  s/n on site:uonzteh92109051552 s/n on tmf:uonzteh92109052452",
        "error nak close ctt no : 1-117012414029 remark : cpe swap serial number does not exist in you cpe list",
        "1-117237507045 old vdlinnne170700622 new unc30val2501032873  500mbps",
        "100mbps mau tukar combo tuan atur dulu di tmforce"
    ],
    "TT V1P": [
        "assalamualaikum..tt v1p 1-26814783348 appt semula hari ni 10.30 am..tq",
        "appt v1p @ 4.30pm 1-26818203485",
        "00506916920 1-26817775658 team mohon bantu utk appt ctt ni hari ini pukul 14:30 tq",
        "team minta buat appt pukul 16:30 hari ini 00506724003 1-26818262517",
        "slot 1-26817279018 @ 4.30pm",
        "1-26815105984 appt 3.30pm  tq",
        "tt v1p 1-26814783348 appt semula hari ni 10.30 am..tq",
        "appt v1p @ 4.30pm 1-26818203485",
        "00506916920 1-26817775658 team mohon bantu utk appt ctt ni hari ini pukul 14:30 tq",
        "1-26819223052 bantuan ra 2.30pm, tq",
        "ra tt v1p 1-26819345735  3:30 pm",
        "dear team apptkan v1p 1-26818264514|appt 330pm 🌾terima kasih🌾",
        "appt v1p 1-26818331322 430pm",
        "1-26819047788 bantu ra 11:45 am",
        "1-2",
        "bantu ra",
        "bantu ra am pm",
        "1-26820223665 bantuan v1p slot 11am",
        "mohon bantuan ra appt ctt v1p @230pm. tq 1-26819047743",
        "1-26813027589 ra 2:30pm",
        "ra ctt v1p 4:30 1-26821452755",
        "ctt 1-26821375701 / v1p mohon bantu slot appt tt v1p 4.30pm...tqvm",
        "slot",
        "bantu slot",
        "bantu slot appt",
        "bantu slot appt tt v1p",
        "team minta appt hari ini pukul 11:30 utk 00506776246 1-26822579516",
        "appt hari ini pukul 11:30 utk 00506776246 1-26822579516",
        "1-26822622537 mohon bantuan untuk book kan v1p jam 1130 am. tqtqt",
        "book kan v1p jam 1130 am",
        "1-26822622537 book kan v1p jam 1130 am",
        "slot appt appt 4:30 ctt whp .tq 1-26821969962",
        "slot appt appt 4:30 ctt whp",
        "slot appt 1-26821969962",
        "1-26822843515 appt 3.30pm  tq",
        "1-26822063055 - whp mohon appt 230pm tq",
        "mohon appt today at 230pm ctt v1p.ctt no : 1-26820229835..tq",
        "1-26823566715 bantuan slotkan tt whp 4.30pm..tq",
        "ctt : 1-26825685185 tolong set up appment hari ini jam 4pm",
        "1-26710708621 tolong set appment pukul 4.30pm hari ini. slot kat id tm31638",
        "team sao ctt : 1-26710708621 tolong set appment pukul 4.30pm hari ini. slot kat id tm31638",
        "mohon ra terdekat ctt v1p 3.30 n slot semula b17296...tq",
        "v1p_nodialtone || 1-26826779718 , tlg buat appt 11:30am ..tq",
        "1-26828417098 assgn app pukul 330"
    ],
    "TT Error 400": [
        "slot appt",
        "1-115955192035 ada eror 400. tq",
        "1-115961233664, xleh slot appt",
        "error 400 tq",
        "ctt 1-116394965125. error 400 tq",
        "no slot 1-116330643385, tq",
        "appointment error (400). tq  1-116374056595",
        "salam, ctt 1-116345228405 x keluar slot book appt",
        "tdk bole slot appt, error 400",
        "semakan 1-116487978265, ada eror 400. tq",
        "error 400",
        "err 400",
        "eror 400",
        "1-116493162085, tjb_v1011_0001, ada error 400 , tq",
        "ada error 400",
        "ada eror 400",
        "failed to manual slot id",
        "xboleh book appt utk ctt ini.1-116477712816",
        "mohon bantu, ctt error 400 1-116497030791",
        "ctt error 400",
        "1-116315635487 mohon bantu error 400",
        "salam team, mohon bantuan semak ctt 1-116310853735, error 404 tq",
        "semak ctt 1-116310853735, error 404 tq",
        "error 404",
        "ctt err 404",
        "mohon bantu tt 1-116581090012/ass_c888_dp0001 error 400",
        "tt 1-116581090012/ass_c888_dp0001 error 400",
        "assalamualaikum & selamat pagi....team.....mohon bantu tt 1-116623676624.......coz tiada slot/no slot.....tq........zita..",
        "tt 1-116623676624.......coz tiada slot/no slot",
        "tiada slot/no slot",
        "tiada slot"
        "no slot",
        "tidak dapat view slot, team dah open skillset ctt: 1-116376826125 created date: 09/09/2025 15:00:27 dp id: tdi_c025_dp0029 customer : astro  zone tdi",
        "tidak dapat view slot  , minta bantu mapping cab id & hsba skill set.  ctt: 1-116648418155 created date: 15/09/2025 11:21:08 am dp id: kin_c046_dp0007 customer : measat broadcast network systems sdn bhd  zone puchong",
        "tt tiada cab/id --1-26823601984",
        "ctt tak boleh slot appt 1-107063025547",
        "ctt no slot 1-118571148885 1-118589296285 1-118575450265 1-118555692365",
        "tt 1-117806195075 tiada slot appt..",
        "bantu ctt x blh slot appt : 1-26829977648"
    ],
    "TT HSBA Reappointment": [
        "untuk patchkan ctt tq 1-116312148935 - appt on 10/9/25 at 11.30am",
        "patch ctt 1-116317835775 pada 11/09/2025  @ 11:30 am",
        "patch ctt dibawah :  1-116320728665 - 09/09/2025 @ 02:30:00 pm",
        "1-115504698155 - team patch appt on 09/09/2025 10:30:00 am. tq",
        "salam team, 1-116326647785 -patch slot 12/09/2025 09:30:00 am.tq",
        "utk patch ctt dibawah :  1-115973031505 - 09/09/2025 @ 04:30:00 pm  1-115804178865 - 09/09/2025 @ 02:30:00 pm  1-115789688125 - 09/09/2025 @ 09:30:00 am",
        "pacth ctt 1-116385142175 12/09/2025 11:30:00 am tq",
        "tukarkan status order dari unscheduled kepada scheduled.. 1-116431932204 13/9/2025 9:30",
        "tukarkan status order dari unscheduled kepada scheduled",
        "status order dari unscheduled kepada scheduled 13/9/2025 9:30"
    ],
    "TT Missing": [
        "tt 1-115443089895 tiada dalam tmf",
        "ctt missing slps return fs troubleshooting tq 1-116065619936",
        "ctt missing. 1-116311028092",
        "tt 1-115443089895 tiada dalam tmf",
        "ctt missing 1-116139913645",
        "ctt missing",
        "team.....trigger tt 1-116324250905....coz telah missing dlm activity list tmf......tq......zita.",
        "tt missing. 1-116132927093",
        "trigger ctt ini missing",
        "missing di tmf",
        "ctt tiada di tmf 1-116534984525..tq",
        "tt xda di tmf",
        "ctt 1-116531652175 missing di tm force, nova unscheduled . mohon bantu ..tqsm",
        "ctt tidak masuk ke tmf. tq 1-116446350874",
        "trigger ctt tiada dlm tmf  1-116582031641",
        "trigger ctt tiada dlm tmf",
        "trigger tt dalam tmf",
        "ctt x apear 1-116870112592"
    ],
    "TT Unsync": [
        "ctt status cancel dlm nova masih appear dlm tmf  1-116258282215",
        "tt cancel tapi masih appear",
        "team nak close ctt x boleh",
        "tak boleh colse ctt",
        "tt 1-112719548255 not sync tmf resolve, nova ip",
        "bntu clearkn tmf nova cancelled",
        "ctt sudah closed di nova",
        "1-116503447165 mohon bantuan team nak close tt x boleh",
        "team nak close tt x boleh",
        "status cancel dalam nova masih appier  dalam tmf",
        "id ni team ui  hari ni bantu  buat tt restoration  issue tiang bengkok , nak return ke nff tak boleh ..tt nova hsba digi 1-116633008355 tt status on site",
        "tt 1-118250210770 sudah cancel tp portal x hilang lg",
        "ctt ni dah resolved tp dlm system belum closed.. aging 0106d 14:41:15.. mohon bantu boleh closed kan ke dlm system.."
    ],
    "Update Granite Network Details": [
        "1-116311380548 - betulkan dp id tin-1-d6qy8p3-001 kepada tin_c006_db0034",
        "1-115958165689 tiada detail cab",
        "adv ctt 1-115970783375 tak boleh tukar fdp dalam granite di tmf",
        "ctt tiada detail kabinet 1-116322359307",
        "1-116322359307 | mohon bantu ctt tiada detail kabinet 1-116322359307",
        "tt 1-116383768545 granite tiada data",
        "update cabinet/fdp dalam tmf,fizikal plgn berada di kgu_c002_db0012..tq  1-115861232625 kgu_c002_dp0009a - tmforce kgu_c002_db0012 -granite",
        "detail granite x appear  2509000080355497",
        "order dah refresh network tp failed",
        "provide dp id 1-116696621386",
        "1-116571996775 update detail fdp & cab id, kerosakan di melaka, building btm..",
        "ftc dah snop.dalam tmf belum update.dah refresh fail  1-116379809716 old dp - dp slb_999_014_0001 new dp -  db slb_c054_db0021",
        "2509000082078987 granite kosong. dah refresh pun fail. ",
        "del dp id ctt berikut : 1-26828131949",
        "detail granite not appear",
        "tambah detail btu dalam tmf. site tukar btu tapi detail btu lama tiada dalam tmf"
    ],
    "Update DP-Cabinet Location": [
        "salam team,  mohon bantuan update dp location order digi ni 1-115819132630 daripada on pole kepada building floor.",
        "daripada on pole kepada building floor",
        "dekat granite dp type street cabinet tapi di network on pole. dekat site dp type yang betul adalah street cabinet",
        "tukarkan network tagging dari on pole ke building floor...dekat site order bf"
    ],
    "TT Error 400": [
        "1-115733638005 | salam team,  mohon bantuan tidak dapat view slot ctt: 1-115733638005 dp id:  mti_c102_dp0001",
        "salam tmf team, @normah mohd salleh @kak nurul mohon bantuan untuk issue ctt ni team: 1-114491447905 missing cab di granite di tm force. ctt : 1-114491447905 cab : swy_c019 dp : swy_c019_dp0017",
        "missing cab di granite di tm force",
        "1-116379866205 ult_c010_dp0050 mohon bantuan no slot",
        "bantuan tt slot",
        "bantu ctt no slot",
        "tidak dapat view slot, ada error 400, advice tmf team, mapping cab id & hsba skill set ke_c999_056.  ctt number :1-116535146015 created date : 12/09/2025 09:59:30 am dp id : fsn_c027a_dp0011 customer : measat broadcast network systems sdn bhd"
    ],
    "TT Duplicate Activity": [
        "cancel 1 activity id 1-116247023935  a-0009903201 a-0009903196",
        "cancel act duplicate 1-116315708835 a-0009908381",
        "ctt 1-116586656460 mintak delete activiti a-0009962626 duplicate atas ctt yg sama",
        "a-0009982910 - 1-116679412161 bntuan team delete xtvt ni. duplicate.tq",
        "tt 1-118615976875 cancel slh 1 activity"
    ],
    "TT TMF-Physical CPE Unsync": [
        "sn rg fizikal dengan tmf tak sama  ctt : 1-116245831767 fizikal : 159fe54e0057211b0 tmf : 0405202300000",
        "1-115885719055 sn onu fizikal - uonwfh2a2312002473 sn onu dalam list - uonwfh2a2312000612 bantu update list equipment.",
        "sn rg fizikal tak sama dengan tmf no ctt : 1-116153828615 rg sn fizikal : rgx842dl1903020843 sn tmf : rgwtull71309006364",
        "tolong update service point punya sn. sebab dalam tmf dengan fizikal tak sama. tq  no ctt : 1-116248891495 sn : onualual1706002509 mac : 9c50eee65894",
        "add onu dalam equipment di tmf, team inform customer ada onu di fizikal  1-116313069961  sn : uonzteh92505209321",
        "1-116313069961 add kan onu dalam tmf.  sn no : uonzteh92505209321",
        "1-116316069495. minta bantuan git add onu lama dlm order sbb perlu tukar onu sahaja.  onu lama : onualual1603007277 onu baru : uonzteh92505201369",
        "ctt no: 1-116045080136 fizikal onu : 4857443d6927a9a update sn onu lama cust",
        "1-116330386115 tt ni cust takde mesh bila nak update keluar mcm ni",
        "tmf , tt ni nak tukar stb, tapi tak ada dalam equipment info. kat nova masih active hypptv",
        "dalam equipment tmf tiada onu tapi di premis existed.  1-116376046915 uonwfh2a2506008576",
        "customer nie fizikal ada 2 sahaje combo. unc25val2412013327 yang nie tak wujud fizikal. minta remove dri list equitment customer",
        "xboleh swap combo, combo asal skyworth, saya tukar dgn skyworth same brand  1-116321727895",
        "xboleh swap combo, combo asal skyworth",
        "verify phone dalam tmf tiada tapi customer inform pernah dapat phone dari tm  1-116327647495",
        "cust no tel isu ata xde kat site dlm tmf ade cmne nk wat",
        "team xboleh update cpe..",
        "x dapt close ctt untuk pertukaran  cpe..dlm tm force cpe list btu tiada..team nak scan btu tak boleh",
        "verify, customer ada langgan mesh sekali ke ? sebab team inform di fizikal ada mesh tetapi system tiada",
        "delete equipment mesh, pelanggan maklum tidak pakai mesh.. fizikal pon tiada mesh hanya ada 1 router sahaja",
        "1-116541338489 mohn bantuan verify ctt ni vdsl ke atau fiber sebab dekat equipmnt tmf ada modem dekat site tiada modem 🙏🏻",
        "add equipment service point 1-116652759125",
        "tukar cpe, yang valid rg sahaja, mesh tidak valid untuk tukar",
        "1-116694278247 old btu onualual1508010393 new btu uonzteh92412010786 minta add onu dalam tmforce..",
        "add onu dalam tmforce",
        "old btu new btu",
        "update cpe dlm system  :  kerana error : sn not exist  ctt:1-116997958275  num siri cpe yg hendak ditukar: unc30val2412064577",
        "bru pasang bulan 7 pakai fiber..dekat sub ada modem dan single box …dlm tmf hanya ada singlebox je..nk tukar modem",
        "boleh tukar combo ke customer nie. customer dah renew kontrak dekat tm point  1-117335657805"
    ],
    "TT - LR Linkage": [
        "salam team bantuan clear tmf ctt link lr20250312-39719 1-106191315217 1-106194182478 1-106312056435 1-106312232721 1-106346917798 1-106471083514 1-106678032670 1-109407463175  tq",
        "ctt unlink 1-115388049295 tiada dalam tmf..",
        "tt unlink 1-115788745277 tiada dalam tmf..",
        "ctt unlink from ntt,no slot",
        "ctt unlink from ntt"
    ],
    "TT - Activity Work Type Blank": [
        "mohon bantuan...ctt tiada activity work type...ctt no : 1-115880486755..iris sudah closed minta retry...tq",
        "xtvt worktype xd. zon nsj. tq"
        "source skill blank"
    ],
    "User Management Issues": [
        "ru inform id problem tak dapat masuk tmf q004560",
        "team unbale to log id id baru"
    ]
}

# Build initial embeddings (case-insensitive)
category_embeddings = {
    cat: model.encode([s.lower() for s in sentences], convert_to_tensor=True).mean(dim=0)
    for cat, sentences in categories.items()
}

# Threshold
SIMILARITY_THRESHOLD = 0.52
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

# added biasing implementation
def categorize_message(msg):
    global group_counter

    clean_msg = clean_message(msg)  # normalize before encoding
    emb = model.encode(clean_msg, convert_to_tensor=True)

    scores = {
        cat: util.cos_sim(emb, emb_cat).item()
        for cat, emb_cat in category_embeddings.items()
    }

    # Apply biasing rules
    scores = apply_bias(clean_msg, scores)

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
