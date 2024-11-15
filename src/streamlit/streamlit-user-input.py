import streamlit as st

#  Your existing code logic here 


def main_logic():
    # Your existing code logic here
    st.title("My Streamlit App")
    user_input = st.text_input("Enter your name: ")
    if user_input:
        st.write(f"Hello, {user_input}! Welcome to my Streamlit app!")
   

if __name__ == "__main__":
   main_logic()
