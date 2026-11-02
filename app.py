
import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Page settings
st.set_page_config(
    page_title="SiteNote AI",
    page_icon="🏗️",
    layout="centered"
)

# Load the Hugging Face model
@st.cache_resource
def load_model():
    model_name = "google/flan-t5-small"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    return tokenizer, model


# App heading
st.title("🏗️ SiteNote AI")
st.subheader("Construction Site Inspection Report Generator")

st.write(
    "Enter your construction site observations and let AI "
    "turn them into a professional inspection report."
)

st.divider()

# User inputs
category = st.selectbox(
    "Select Inspection Category",
    [
        "Structural",
        "Finishing",
        "Electrical",
        "Plumbing",
        "Safety"
    ]
)

language = st.selectbox(
    "Select Report Language",
    ["English", "Urdu"]
)

observation = st.text_area(
    "Describe your site observation",
    placeholder="Example: Cracks were observed in the plaster near the ground-floor staircase.",
    height=150
)

# Generate report
if st.button("Generate Inspection Report", type="primary"):

    if not observation.strip():
        st.warning("Please enter a site observation first.")

    else:
        try:
            with st.spinner("Loading AI model and generating your report..."):

                tokenizer, model = load_model()

                prompt = f"""
                Write a professional construction site inspection report
                in {language}.

                Inspection category: {category}

                Site observation: {observation}

                Organize the report under these three headings:
                1. Site Observation
                2. Possible Concern
                3. Recommended Action

                Use clear and professional language.
                Do not invent measurements, facts, or site conditions.
                Base the report only on the observation provided.
                """

                inputs = tokenizer(
                    prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=512
                )

                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=180,
                        num_beams=2
                    )

                report = tokenizer.decode(
                    outputs[0],
                    skip_special_tokens=True
                )

            st.success("Report generated successfully!")

            st.subheader("Generated Inspection Report")
            st.write(report)

            st.download_button(
                label="Download Report",
                data=report,
                file_name="Site_Inspection_Report.txt",
                mime="text/plain"
            )

        except Exception as e:
            st.error("Something went wrong while generating the report.")
            st.write(str(e))
