
import streamlit as st
from huggingface_hub import InferenceClient

# Page configuration
st.set_page_config(
    page_title="SiteNote AI",
    page_icon="🏗️",
    layout="centered"
)

# Application heading
st.title("🏗️ SiteNote AI")
st.subheader("Smart Construction Site Report Generator")

st.write(
    "Convert simple construction site observations "
    "into professional inspection reports using AI."
)

st.divider()

# User input
st.write("### Enter Site Observation")

observation = st.text_area(
    "Describe the issue you observed:",
    placeholder="Example: Cracks appeared on the wall near the window..."
)

# Category selection
category = st.selectbox(
    "Select Issue Category",
    ["Structural", "Finishing", "Electrical", "Plumbing", "Safety"]
)

# Language selection
language = st.selectbox(
    "Report Language",
    ["English", "Urdu"]
)

# Generate AI report
if st.button("Generate AI Report", type="primary"):

    if not observation.strip():
        st.warning("Please enter a site observation.")

    else:
        try:
            client = InferenceClient(
                model="google/gemma-2-2b-it:featherless-ai",
                token=st.secrets["HF_TOKEN"]
            )

            prompt = f"""
            You are an assistant helping construction site engineers.

            Issue category: {category}
            Site observation: {observation}

            Write a professional inspection report in {language}.

            Include:
            1. Site Observation
            2. Possible Concern
            3. Recommended Action

            Use clear professional language.
            Do not invent measurements or facts.
            Do not claim a defect is confirmed without inspection.
            """

            with st.spinner("AI is preparing your report..."):

                response = client.chat_completion(
                    messages=[
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    max_tokens=500,
                    temperature=0.4
                )

                report = response.choices[0].message.content

            st.success("Report generated successfully!")

            st.write("### 📋 Generated Inspection Report")
            st.markdown(report)

            st.download_button(
                label="Download Report",
                data=report,
                file_name="Site_Inspection_Report.txt",
                mime="text/plain"
            )

        except Exception as e:
            st.error("Unable to generate the report.")
            st.error(str(e))
