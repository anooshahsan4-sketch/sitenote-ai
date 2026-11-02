
import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Page configuration
st.set_page_config(
    page_title="SiteNote AI",
    page_icon="🏗️",
    layout="centered"
)

# Load Hugging Face model
@st.cache_resource
def load_model():

    model_name = "Qwen/Qwen2.5-0.5B-Instruct"

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        low_cpu_mem_usage=True
    )

    model.eval()

    return tokenizer, model


# App interface
st.title("🏗️ SiteNote AI")
st.subheader("Construction Site Inspection Report Generator")

st.write(
    "Convert rough construction site observations "
    "into professional inspection reports using AI."
)

st.divider()

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

        st.warning("Please enter a site observation.")

    else:

        try:
            with st.spinner("Loading AI model and generating report..."):

                tokenizer, model = load_model()

                messages = [
                    {
                        "role": "system",
                        "content": (
                            "You are a professional construction "
                            "site inspection report assistant. "
                            "Only use facts provided by the user. "
                            "Never invent measurements or site conditions."
                        )
                    },
                    {
                        "role": "user",
                        "content": f"""
                        Create a professional construction site
                        inspection report in {language}.

                        Inspection Category: {category}

                        Site Observation: {observation}

                        Use exactly these three sections:

                        1. Site Observation
                        Describe what was observed.

                        2. Possible Concern
                        Explain what the observation might indicate.
                        Do not state an unverified cause as fact.

                        3. Recommended Action
                        Suggest suitable inspection or corrective steps.

                        Write a complete report, not instructions.
                        """
                    }
                ]

                prompt = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )

                inputs = tokenizer(
                    prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=1024
                )

                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=250,
                        do_sample=False,
                        repetition_penalty=1.15,
                        no_repeat_ngram_size=3
                    )

                new_tokens = outputs[0][inputs["input_ids"].shape[1]:]

                report = tokenizer.decode(
                    new_tokens,
                    skip_special_tokens=True
                )

            st.success("Inspection report generated!")

            st.subheader("Generated Inspection Report")
            st.write(report)

            st.download_button(
                label="Download Report",
                data=report,
                file_name="Site_Inspection_Report.txt",
                mime="text/plain"
            )

        except Exception as e:

            st.error("Unable to generate the report.")
            st.write(str(e))
