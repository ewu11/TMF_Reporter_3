import streamlit as st
from pathlib import Path
from sentence_transformers import SentenceTransformer, util

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
    "equipment_change": [
        "tukar rg7 ke rg6",
        "update wifi7 ke wifi6 ax3000",
        "ubah equipment kepada combo"
    ],
    "bypass_issue": [
        "bypass extraport",
        "iris access denied",
        "tak boleh masuk bypass"
    ],
    "order_issue": [
        "order returned",
        "tiada dalam rol",
        "order tak boleh done keluar error"
    ],
    "contact_update": [
        "tolong update contact number",
        "change phone number for order"
    ]
}

# Build initial embeddings
category_embeddings = {
    cat: model.encode(sentences, convert_to_tensor=True).mean(dim=0)
    for cat, sentences in categories.items()
}

# Threshold
SIMILARITY_THRESHOLD = 0.70
new_groups = {}
group_counter = 1


# ------------------------
# Categorization function
# ------------------------
def categorize_message(msg):
    global group_counter
    emb = model.encode(msg, convert_to_tensor=True)
    scores = {cat: util.cos_sim(emb, emb_cat).item() for cat, emb_cat in category_embeddings.items()}
    best_cat, best_score = max(scores.items(), key=lambda x: x[1])

    if best_score >= SIMILARITY_THRESHOLD:
        return best_cat, best_score
    else:
        # create new category dynamically
        new_cat = f"auto_group_{group_counter}"
        new_groups[new_cat] = emb
        category_embeddings[new_cat] = emb
        group_counter += 1
        return new_cat, best_score


# ------------------------
# Streamlit UI
# ------------------------
st.title("📂 Chat Log Categorizer")
st.write("Automatically categorize messages based on semantic similarity.")

# Upload file
uploaded_file = st.file_uploader("Upload cleansed_output.txt", type=["txt"])

if uploaded_file:
    # Read messages
    lines = uploaded_file.read().decode("utf-8").splitlines()
    messages = []
    for line in lines:
        if "]" in line and ":" in line:
            msg = line.split(":", 2)[-1].strip()
            if msg:
                messages.append(msg)

    st.success(f"Loaded {len(messages)} messages")

    # Run categorization
    results = []
    for msg in messages:
        cat, score = categorize_message(msg)
        results.append((msg, cat, round(score, 2)))

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

# Textbox for single message test
st.subheader("🔎 Test Single Message")
test_msg = st.text_input("Enter a message:")
if test_msg:
    cat, score = categorize_message(test_msg)
    st.write(f"Prediction: **{cat}** (score={score:.2f})")
