import streamlit as st
import requests

ngrok_url = "https://1ab5-2401-4900-1c97-194-fc5e-448e-fb6-acba.ngrok-free.app"

st.title("NCERT Curriculum Assistant Chatbot")
st.write("Upload a PDF and interact with the Llama3 chatbot to get answers based on the document content.")

uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

if uploaded_file is not None:
    # Save the uploaded file
    with open(f"pdf/{uploaded_file.name}", "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # Notify the user about the file upload
    st.success(f"Successfully uploaded {uploaded_file.name}")
    
    # Send the file to the backend
    files = {"file": open(f"pdf/{uploaded_file.name}", "rb")}
    response = requests.post(f"{ngrok_url}/pdf", files=files)
    
    if response.status_code == 200:
        st.write("PDF processed successfully!")
    else:
        st.write("Error processing PDF.")

# Input for user query
user_query = st.text_input("Enter your query")

# Button to submit the query
if st.button("Get Answer"):
    if user_query:
        # Send the query to the backend
        response = requests.post(f"{ngrok_url}/ask_pdf", json={"query": user_query})
        
        if response.status_code == 200:
            result = response.json()
            st.write(f"Answer: {result['answer']}")
            st.write("Sources:")
            for source in result["sources"]:
                st.write(f"- {source['source']}: {source['page_content']}")
        else:
            st.write("Error retrieving answer.")
    else:
        st.write("Please enter a query.")


# import streamlit as st
# import requests

# # Replace with your Ngrok public URL
# ngrok_url = "https://1ab5-2401-4900-1c97-194-fc5e-448e-fb6-acba.ngrok-free.app"

# # Title and description
# st.title("NCERT Curriculum Assistant Chatbot")
# st.write("Upload a PDF and interact with the Llama3 chatbot to get answers based on the document content.")

# # File uploader for PDFs
# uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

# if uploaded_file is not None:
#     # Save the uploaded file
#     with open(f"pdf/{uploaded_file.name}", "wb") as f:
#         f.write(uploaded_file.getbuffer())
    
#     # Notify the user about the file upload
#     st.success(f"Successfully uploaded {uploaded_file.name}")
    
#     # Send the file to the backend
#     files = {"file": open(f"pdf/{uploaded_file.name}", "rb")}
#     response = requests.post(f"{ngrok_url}/pdf", files=files)
    
#     if response.status_code == 200:
#         st.write("PDF processed successfully!")
#     else:
#         st.write("Error processing PDF.")

# # Input for user query
# user_query = st.text_input("Enter your query")

# # Button to submit the query
# if st.button("Get Answer"):
#     if user_query:
#         # Send the query to the backend
#         response = requests.post(f"{ngrok_url}/ask_pdf", json={"query": user_query})
        
#         if response.status_code == 200:
#             result = response.json()
#             st.write(f"Answer: {result['answer']}")
#             st.write("Sources:")
#             for source in result["sources"]:
#                 st.write(f"- {source['source']}: {source['page_content']}")
#         else:
#             st.write("Error retrieving answer.")
#     else:
#         st.write("Please enter a query.")
