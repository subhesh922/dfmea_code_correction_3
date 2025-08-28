import streamlit as st
import requests
#from utils.file_parser import parse_excel_or_csv
import pandas as pd 
import os 
from pathlib import Path
from dotenv import load_dotenv
import io 
load_dotenv()
# Make text and widgets larger + center content wider
st.markdown(
    """
    <style>
    /* Make main container wider but centered */
    .block-container {
        max-width: 90% !important;
        padding-left: 5rem !important;
        padding-right: 5rem !important;
        margin: 0 auto !important;
    }

    /* Bigger font for text areas */
    .stTextArea textarea {
        font-size: 16px !important;
    }

    /* Bigger font for dropdowns */
    .stSelectbox div, .stMultiSelect div {
        font-size: 16px !important;
    }

    /* --- Make tabs larger and bolder --- */
    .stTabs [role="tab"] {
        font-size: 18px !important;
        font-weight: 600 !important;
        padding: 12px 24px !important;
        border-radius: 8px !important;
    }

    .stTabs [role="tab"][aria-selected="true"] {
        background-color: #f0f2f6 !important;
        color: black !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)


API_BASE = "http://127.0.0.1:8000"

st.set_page_config(page_title="🛡️ DFMEA GenAI", layout="centered")
st.title("📊 DFMEA Generator", width="stretch")

# --- Tabs ---
tab1, tab2 = st.tabs(["📂 Upload & Preview", "📑 DFMEA Output"])

    # --- Section 1: PRDs Upload ---
with tab1:
    st.subheader("1) Upload DOCs/Specs (.docx / .pptx / .pdf / .csv / .xlsx)")
    st.caption("Upload one or more PRD files")
    uploaded_prds = st.file_uploader(
        "Drag and drop or Browse files",
        type=["docx", "doc", "pptx", "ppt", "pdf", "csv", "xls", "xlsx"],
        accept_multiple_files=True,
        key="prd_upload"
    )

    # --- Section 2: Knowledge Base Upload ---
    st.subheader("2) Upload Knowledge Base (.csv / .xlsx)")
    st.caption("Upload one or more Knowledge Bank files (.csv / .xlsx)")
    uploaded_kb = st.file_uploader(
        "Drag and drop or Browse files",
        type=["csv", "xls", "xlsx"],
        accept_multiple_files=True,
        key="kb_upload"
    )

    # --- Section 3: Field Issues Upload ---
    st.subheader("3) Upload Field Reported Issues (.csv / .xlsx)")
    st.caption("Upload one or more Field Reported Issues files (.csv / .xlsx)")
    uploaded_fi = st.file_uploader(
        "Drag and drop or Browse files",
        type=["csv", "xls", "xlsx"],
        accept_multiple_files=True,
        key="fi_upload"
    )
# --- Sidebar: Subproduct & Product Mapping ---

st.sidebar.markdown("### 📦 Subproduct Selection")
st.sidebar.info("📌 Products will be picked automatically based on your subproduct selection.")

# Project root = one level above current file
BASE_DIR = Path(__file__).resolve().parent.parent  

# --- Sidebar: Subproduct & Product Mapping ---

# Build mapping file path
MAPPING_FILE_PATH = BASE_DIR / "server" / "Runner_files" / "Subproduct-Product- Mapping" / "Mapping.csv"
df_map = pd.read_csv(MAPPING_FILE_PATH)

# Build mapping: {subproduct_combo: [Product, Product...]}
SUBPRODUCT_MAPPING = {}
for _, row in df_map.iterrows():
    sub = str(row["Sub_product"]).strip()
    prod = str(row["Product"]).strip()
    SUBPRODUCT_MAPPING.setdefault(sub, []).append(prod)

# Extract subproduct combos directly from the CSV
SUBPRODUCT_OPTIONS = sorted(SUBPRODUCT_MAPPING.keys())

# Initialize subproducts in session_state if empty
if "subproducts_selected" not in st.session_state or not st.session_state.subproducts_selected:
    st.session_state.subproducts_selected = [SUBPRODUCT_OPTIONS[0]]

# --- Sidebar multiselect (user chooses subproducts) ---
st.sidebar.markdown("### 📦 Subproduct Selection")
st.session_state.subproducts_selected = st.sidebar.multiselect(
    "Choose subproduct combination(s):",
    options=SUBPRODUCT_OPTIONS,
    default=st.session_state.subproducts_selected,
    help="Select one or more subproduct combinations defined in mapping CSV"
)

# Derive available products dynamically
available_products = []
for sub in st.session_state.subproducts_selected:
    available_products.extend(SUBPRODUCT_MAPPING.get(sub, []))
available_products = sorted(set(available_products))

# Store in session_state for backend use
st.session_state["available_products"] = available_products

# --- Sidebar: Show mapped products ---
if available_products:
    st.sidebar.markdown("### 🏷️ Products automatically mapped")
    st.sidebar.success(", ".join(available_products))
else:
    st.sidebar.warning("⚠️ No product mapping found for the selected subproducts.")


# --- Sidebar Admin Section ---
st.sidebar.markdown("### 🔑 Admin Login")
st.sidebar.caption("This area is for admin users only. Please login to unlock advanced options.")

# Admin login field
admin_pass = st.sidebar.text_input("Enter admin password", type="password")
if st.sidebar.button("Login as Admin"):
    if admin_pass == os.getenv("ADMIN_PASSWORD"):
        st.session_state.is_admin = True
        st.sidebar.success("✅ Logged in as Admin")
    else:
        st.sidebar.error("❌ Invalid password")

# Show admin-only controls *after* login
if st.session_state.get("is_admin", False):
    # --- Admin-only Focus Prompt Section ---
    with st.sidebar:
        st.markdown("### 🔒 Focus Prompt\nFocus (admin only)")
        st.caption("This section is for **admin users only**. "
                "Use focus prompts to guide the DFMEA output. "
                "These help refine results for new subproducts or critical scenarios.")
        st.sidebar.info(
        "💡 **Tip for Admins:**\n"
        "- Use this prompt to fine-tune results.\n"
        "- Be specific about what to **prioritize** (e.g., critical defects, safety issues).\n"
        "- Clearly state what to **exclude** (e.g., duplicate issues, resolved items).\n"
        "- When trying a **new subproduct**, guide the system with context (e.g., 'Focus on thermal reliability tests for DISPLAY+TP')."
        )
        # Suggestions
        suggestions = [
            "Prioritize failure modes for {PRODUCT} / {SUBPRODUCT} that directly impact critical functionality...",
            "Emphasize thermal-related failures in {PRODUCT} / {SUBPRODUCT}, such as overheating...",
            "Focus on new field issues not previously captured in the Knowledge Bank for {PRODUCT} / {SUBPRODUCT}...",
            "Highlight recurring failures across multiple reports for {PRODUCT} / {SUBPRODUCT}...",
            "For {PRODUCT} / {SUBPRODUCT}, generate DFMEA entries where Severity ≥ 8...",
            "Prioritize display visibility issues under sunlight or rugged conditions...",
            "Focus on connector and solder joint failures within {PRODUCT} / {SUBPRODUCT}...",
            "Generate entries emphasizing preventive controls not already seen in the Knowledge Bank...",
            "Analyze failures that lead to safety hazards (e.g., electrical shock, overheating)...",
            "Prioritize sensor-related issues within {SUBPRODUCT} for {PRODUCT}...",
            "Focus on recent field issues from the last 2 years...",
            "Emphasize mechanical stress failures (drops, shocks, vibration cracks)...",
            "Ensure at least 50% of DFMEA entries are inferred intelligently...",
            "Generate entries focusing on power-related failures (charging, discharging, regulator failures)...",
            "Prioritize hard-to-detect issues in {PRODUCT} / {SUBPRODUCT} where Detection ≥ 7..."
        ]

        selected_prompt = st.selectbox("📌 Suggested Prompts", options=["-- Select a focus prompt --"] + suggestions)

        # Editable text area for final prompt
        if "focus_prompt" not in st.session_state:
            st.session_state["focus_prompt"] = ""

        if st.button("➕ Insert Prompt"):
            if selected_prompt != "-- Select a focus prompt --":
                st.session_state["focus_prompt"] = selected_prompt

        focus_prompt = st.text_area(
            "✍️ Custom Focus Prompt",
            value=st.session_state["focus_prompt"],
            height=150,
            key="focus_prompt_textarea"
        )


with tab1:
    if st.button("✨ Create DFMEA", type="primary", use_container_width=True):
        with st.spinner("⏳ Generating DFMEA... please wait while the backend processes your request."):
            try:
                # ✅ Ensure subproducts always a list
                subproducts = st.session_state.get("subproducts_selected", [])
                if isinstance(subproducts, str):
                    subproducts = [subproducts]
                elif not isinstance(subproducts, list):
                    subproducts = list(subproducts)

                # ✅ Ensure products always a list
                products = st.session_state.get("available_products", [])
                if isinstance(products, str):
                    products = [products]
                elif not isinstance(products, list):
                    products = list(products)

                # ✅ Prepare files for requests.post
                files = []
                if uploaded_prds:
                    for f in uploaded_prds:
                        files.append(("prds", (f.name, f.getvalue(), "application/octet-stream")))
                if uploaded_kb:
                    for f in uploaded_kb:
                        files.append(("knowledge_base", (f.name, f.getvalue(), "application/octet-stream")))
                if uploaded_fi:
                    for f in uploaded_fi:
                        files.append(("field_issues", (f.name, f.getvalue(), "application/octet-stream")))

                # Call backend API
                response = requests.post(
                    f"{API_BASE}/dfmea/generate",
                    data={
                        "products": products,
                        "subproducts": subproducts,
                        "focus": st.session_state.get("focus_prompt", "")
                    },
                    files=files
                )

                if response.status_code == 200:
                    result = response.json()

                    # 🔹 Save backend response JSON into session state
                    st.session_state["dfmea_entries"] = result.get("dfmea_entries", [])

                    # ✅ Replace spinner with success alert in SAME place
                    st.success("🎉 Pipeline run complete! Go to **DFMEA Output TAB** to view results and download.")
                else:
                    st.error(f"❌ Backend error: {response.status_code}")

            except Exception as e:
                st.error(f"⚠️ Request failed: {e}")

        # with st.spinner("⏳ Generating DFMEA... please wait while the backend processes your request."):
        #     try:
        #         # ✅ Ensure subproducts always a list
        #         subproducts = st.session_state.get("subproducts_selected", [])
        #         if isinstance(subproducts, str):
        #             subproducts = [subproducts]
        #         elif not isinstance(subproducts, list):
        #             subproducts = list(subproducts)

        #         # ✅ Ensure products always a list
        #         products = st.session_state.get("available_products", [])
        #         if isinstance(products, str):
        #             products = [products]
        #         elif not isinstance(products, list):
        #             products = list(products)

        #         # ✅ Prepare files for requests.post
        #         files = []
        #         if uploaded_prds:
        #             for f in uploaded_prds:
        #                 files.append(("prds", (f.name, f.getvalue(), "application/octet-stream")))
        #         if uploaded_kb:
        #             for f in uploaded_kb:
        #                 files.append(("knowledge_base", (f.name, f.getvalue(), "application/octet-stream")))
        #         if uploaded_fi:
        #             for f in uploaded_fi:
        #                 files.append(("field_issues", (f.name, f.getvalue(), "application/octet-stream")))

        #         # Call backend API
        #         response = requests.post(
        #             f"{API_BASE}/dfmea/generate",
        #             data={
        #                 "products": products,
        #                 "subproducts": subproducts,
        #                 "focus": st.session_state.get("focus_prompt", "")
        #             },
        #             files=files
        #         )

        #         if response.status_code == 200:
        #             result = response.json()

        #             # 🔹 Save backend response JSON into session state
        #             st.session_state["dfmea_entries"] = result.get("dfmea_entries", [])
        #             # ✅ NEW: Show success alert in Tab 1
        #             st.success("🎉 Pipeline run complete! Please go to **DFMEA Output Tab** to view the results and download if needed.")
        #         else:
        #             st.error(f"❌ Backend error: {response.status_code}")
        #     except Exception as e:
        #         st.error(f"⚠️ Request failed: {e}")

with tab2:
    st.header("📊 DFMEA Analysis Output")

    if "dfmea_entries" in st.session_state:
        dfmea_entries = st.session_state["dfmea_entries"]

        if dfmea_entries:
            st.success("✅ DFMEA entries ready!")

            # Show in table form (if dict-like)
            try:
                df = pd.DataFrame(dfmea_entries)
                st.subheader("Tabular View")
                st.dataframe(df)
            except Exception:
                st.warning("⚠️ Could not render DFMEA entries as table.")

            # Show JSON
            st.subheader("Raw JSON Response")
            st.json(dfmea_entries)
        else:
            st.warning("⚠️ Empty DFMEA response.")
    else:
        #st.info("ℹ️ Run DFMEA analysis first in Tab 1.")

        st.info("⚙️ Run the analysis to generate and download the DFMEA file.")

#---------Preview Uploaded data----------------------------            

def _preview_df(upload_file):
    """Robust preview: reset pointer; try csv then excel, then reverse."""
    try:
        upload_file.seek(0)
    except Exception:
        pass
    name = upload_file.name.lower()
    try:
        if name.endswith(".csv"):
            upload_file.seek(0)
            return pd.read_csv(upload_file).head(20)
        elif name.endswith(".xlsx"):
            upload_file.seek(0)
            return pd.read_excel(upload_file).head(20)
        else:
            return None
    except Exception:
        # fallback (in case extension mismatches content)
        try:
            upload_file.seek(0)
            return pd.read_excel(upload_file).head(20)
        except Exception:
            try:
                upload_file.seek(0)
                return pd.read_csv(upload_file).head(20)
            except Exception:
                return None

# ===== Optional previews (robust) =====
with tab1:
    st.divider()
    c1, c2 = st.columns(2)
    with c1:
        if uploaded_kb:
            st.caption("Knowledge Base (first file preview)")
            df = _preview_df(uploaded_kb[0])
            if df is not None:
                st.dataframe(df)
            else:
                st.warning("KB preview failed (format not recognized).")

    with c2:
        if uploaded_fi:
            st.caption("Field Issues (first file preview)")
            df = _preview_df(uploaded_fi[0])
            if df is not None:
                st.dataframe(df)
            else:
                st.warning("Field preview failed (format not recognized).")

    if uploaded_prds:
        st.caption("PRD/Spec files uploaded:")
        st.write([f.name for f in uploaded_prds])
        df = _preview_df(uploaded_prds[0])
        if df is not None:
            st.caption("PRD/Spec preview (first file)")
            st.dataframe(df)
