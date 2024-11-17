from langchain_google_genai import GoogleGenerativeAI
from dotenv import load_dotenv
import os
import streamlit as st

load_dotenv(dotenv_path="../../../.env", override=True)


@st.cache_resource
#def load_models() -> tuple[GoogleGenerativeAI, GoogleGenerativeAI]:
#    """Load Gemini 1.5 Flash and Pro models."""
#    return GoogleGenerativeAI("gemini-1.5-flash"), GoogleGenerativeAI("gemini-1.5-pro")

def load_model(model_name: str) -> GoogleGenerativeAI:
    """Loads a Gemini model."""
    return GoogleGenerativeAI(model=model_name, google_api_key=os.getenv("GOOGLE_API_KEY"))

gemini_15_flash = load_model("gemini-1.5-flash")
gemini_15_pro = load_model("gemini-1.5-pro")

def get_model_name(model: GoogleGenerativeAI) -> str:
  """Get Gemini Model Name"""
  # Access the model name attribute directly
  model_name = model.name.replace
  return f"`{model_name}`"  # Return the model name wrapped in backticks


tab1, tab2, tab3, tab4 = st.tabs(
    ["Generate story", "Marketing campaign", "Image Playground", "Video Playground"]
)

with tab1:
    st.subheader("Generate a story")
    selected_model = st.radio(
        "Select Gemini Model:",
        [gemini_15_pro, gemini_15_flash],
        #format_func=get_model_name,
        key="selected_model_story",
        horizontal=True,
    )


with tab2:
    st.subheader("Generate your marketing campaign")

with tab3:
    st.subheader("Image Playground")

with tab4:
    st.subheader("Video Playground")

def main_logic():
    st.header("Generative AI Gemini 1.5 API", divider="rainbow")
    st.sidebar.title("Options")

if __name__ == "__main__":
   main_logic()