import streamlit as st
import requests

st.set_page_config(page_title="Multi-Agent Game Studio", layout="wide")

st.title("👾 Multi-Agent Retro Game Studio")
st.write("Type a game concept below to send to your FastAPI LangGraph backend server!")

# Define backend destination URL
BACKEND_URL = "https://langraph-tutorials.onrender.com"

user_prompt = st.text_input("What kind of game do you want to build?", placeholder="Build a retro Space Invaders game...")

if st.button("Launch Agent Studio Loop"):
    if user_prompt:
        with st.spinner("Streaming task to FastAPI Backend... Please watch your server logs."):
            try:
                # Send the data over the network to your server script
                response = requests.post(BACKEND_URL, json={"prompt": user_prompt})
                
                if response.status_code == 200:
                    game_html = response.json()["html_code"]
                    st.success("🎉 Game built successfully!")
                    
                    st.download_button(label="💾 Download File", data=game_html, file_name="game.html", mime="text/html")
                    st.components.v1.html(game_html, height=500, scrolling=True)
                else:
                    st.error(f"Error: {response.json().get('detail')}")
            except Exception as e:
                st.error(f"Could not connect to the API server: {str(e)}")
