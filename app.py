
import streamlit as st

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
    "Convert your simple construction site observations "
    "into professional inspection reports using AI."
)

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

# Generate button
if st.button("Generate Report"):
    if observation.strip():
        st.success("Your observation has been submitted!")
        st.write("### Your Observation")
        st.write(observation)

        st.write("### Selected Category")
        st.write(category)
    else:
        st.warning("Please enter a site observation first.")
