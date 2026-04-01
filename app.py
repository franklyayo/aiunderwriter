import json
import logging
import os
import pandas as pd
import streamlit as st
from openai import OpenAI
from transformers import pipeline
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter, SentenceTransformersTokenTextSplitter
import asyncio
import threading

st.set_page_config(
    page_title="AI Underwriter Pro",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)

def analyze_risk(parsed_guidlines, parsed_application_form):
    prompt = f"""<guidelines>{parsed_guidlines}</guidelines> <application_form>{parsed_application_form}</application_form>."""

    client = OpenAI()
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": "You will be provided an insurance underwriting guidelines (delimited with XML tags) <guidelines></guidelines> and an insurance application form (delimited with XML tags) <application_form></application_form>. Can you provide some information about the relevant risks on the application following the guidelines ?."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0,
        max_tokens=1024,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )

    relevant_risks = []
    for choice in response.choices:
        relevant_risks.append(choice.message.content)

    return relevant_risks


@st.cache_data
@st.cache_resource
def extract_data(file, file_name):
    # 1. Extract bytes from the Streamlit UploadedFile object
    file_bytes = file.getvalue()
    
    # 2. Create a unique temporary file path on the server's disk
    file_path = f"temp_{file_name}.pdf"
    
    # 3. Write the bytes to this physical file
    with open(file_path, "wb") as temp_file:
        temp_file.write(file_bytes)

    # 4. Now LangChain can read it from the disk
    loader = PyPDFLoader(file_path)
    pages = loader.load()

    if file_name == "underwriting_guidlines":
        # 1. Use standard splitter instead of the heavy BAAI model
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=50, length_function=len)
        guidelines_docs = text_splitter.split_documents(pages)
        guidelines = "\n\n".join([doc.page_content for doc in guidelines_docs])
        
        chunk_size = 2000
        chunks = [guidelines[i:i+chunk_size] for i in range(0, len(guidelines), chunk_size)]
        
        # 2. Use OpenAI for summarization instead of local BART
        client = OpenAI()
        summarized_texts = []
        for chunk in chunks:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant. Summarize the following insurance underwriting guidelines concisely, retaining all critical risk factors."},
                    {"role": "user", "content": chunk}
                ],
                temperature=0
            )
            summarized_texts.append(response.choices[0].message.content)
            
        text_content = "\n\n".join(summarized_texts)
       
    else:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=50, length_function=len)
        docs = text_splitter.split_documents(pages)
        text_content = "\n\n".join([doc.page_content for doc in docs])

    # 5. Clean up the physical file so we don't clog the server
    if os.path.exists(file_path):
        os.remove(file_path)

    return text_content


async def extract_data_async(file, file_name):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, extract_data, file, file_name)




async def main():
    # --- Custom Header ---
    st.markdown("<h1 style='text-align: center;'>Insurance Risk Analysis Engine</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #6B7280;'>Upload the underwriting guidelines and application form to instantly generate a risk profile.</p>", unsafe_allow_html=True)
    st.divider()

    # --- Side-by-Side Uploaders ---
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📄 Guidelines")
        uploaded_underwriting_guidelines = st.file_uploader('Upload Underwriting Guidelines', type="pdf", help='Must be a standard PDF document')
        
    with col2:
        st.subheader("📝 Application")
        uploaded_application_form = st.file_uploader('Upload Application Form', type="pdf", help='Must be a standard PDF document')
        
    st.write("") # Add a little spacing

    # --- Centered Call-to-Action Button ---
    _, col_btn, _ = st.columns([1, 1, 1])
    with col_btn:
        # The type="primary" makes the button pop with the theme's main color
        analyze_btn = st.button('Analyze Risk Profile', use_container_width=True, type="primary")

    st.divider()

    # --- Execution & Results ---
    if analyze_btn:
        if uploaded_underwriting_guidelines and uploaded_application_form:
            
            with st.status("Performing Risk Analysis...", expanded=True) as status:
                st.write("Extracting data from documents...")
                parsed_guidelines = extract_data(uploaded_underwriting_guidelines, "underwriting_guidlines")
                parsed_application_form = extract_data(uploaded_application_form, "application_form")
                
                st.write("Cross-referencing risks with OpenAI...")
                relevant_risks = analyze_risk(parsed_guidelines, parsed_application_form)
                
                status.update(label="Analysis Complete!", state="complete", expanded=False)
            
            # --- Better Results Display ---
            st.success("Risk factors identified successfully.")
            st.subheader("Key Findings")
            
            for i, risk in enumerate(relevant_risks):
                # Use expanders to keep the screen from getting too cluttered
                with st.expander(f"Risk Factor {i+1}", expanded=True):
                    st.info(risk)
                    
        else:
            st.warning("⚠️ Please upload both the guidelines and the application form to proceed.")

            
if __name__ == '__main__':
    asyncio.run(main())
