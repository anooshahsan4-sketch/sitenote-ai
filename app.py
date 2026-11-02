
import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

st.set_page_config(
    page_title="SiteNote AI",
    page_icon="🏗️",
    layout="centered"
)

@st.cache_resource
def load_model():
    model_name = "google/flan-t5-small"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return tokenizer, model

st.title("🏗️ SiteNote AI")
st.subheader("Construction Site Inspection Report Generator")

st.write("Turn rough construction observations into professional inspection reports.")

st.divider()

category = st.selectbox(
    "Select Inspection Category",
    ["Structural", "Finishing", "Electrical", "Plumbing", "Safety"]
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

if st.button("Generate Inspection Report", type="primary"):

    if not observation.strip():
        st.warning("Please enter a site observation first.")

    else:
        try:
            with st.spinner("Generating your report..."):

                tokenizer, model = load_model()

                prompt = f"""
                Write a short professional construction inspection report.
                The report must be in {language}.

                Category: {category}
                Observation: {observation}

                Use exactly these headings:
                Site Observation:
                Possible Concern:
                Recommended Action:

                Write the actual report under each heading.
                Do not repeat the instructions.
                Do not invent facts or measurements.
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
                        num_beams=4,
                        repetition_penalty=1.5,
                        no_repeat_ngram_size=3
                    )

                report = tokenizer.decode(
                    outputs[0],
                    skip_special_tokens=True
                )

            st.success("Report generated!")
            st.subheader("Generated Inspection Report")
            st.write(report)

            st.download_button(
                "Download Report",
                data=report,
                file_name="Site_Inspection_Report.txt",
                mime="text/plain"
            )

        except Exception as e:
            st.error("An error occurred:")
            st.write(str(e))
