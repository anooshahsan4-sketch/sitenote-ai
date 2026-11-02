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

# AI report generation
if st.button("Generate AI Report", type="primary"):

    if not observation.strip():
        st.warning("Please enter a site observation first.")

    else:
        try:
            # Connect to Hugging Face
            client = InferenceClient(
                model="Qwen/Qwen2.5-7B-Instruct",
                token=st.secrets["HF_TOKEN"]
            )

            # Create instructions for the AI
            prompt = f"""
            You are an assistant helping construction site engineers
            prepare professional inspection reports.

            Issue category: {category}
            Site observation: {observation}

            Generate a professional report in {language}.

            Include these headings:
            1. Site Observation
            2. Possible Concern
            3. Recommended Action

            Use clear, professional language.
            Do not invent measurements or facts.
            Do not claim a structural defect is confirmed
            without proper professional inspection.
            """

            # Generate the report
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

            # Display the result
            st.success("Your AI report has been generated!")

            st.write("### 📋 Generated Inspection Report")
            st.markdown(report)

            # Download report
            st.download_button(
                label="Download Report",
                data=report,
                file_name="Site_Inspection_Report.txt",
                mime="text/plain"
            )

        except Exception as e:
            st.error("Something went wrong while generating the report.")
            st.write("Error details:", str(e))
