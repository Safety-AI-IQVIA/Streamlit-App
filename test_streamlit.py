import streamlit as st
import pandas as pd
import numpy as np

# Title
st.title("Basic Streamlit App")

# Header
st.header("This is a header")

# Subheader
st.subheader("This is a subheader")

# Text
st.write("This is some text.")

# Markdown
st.markdown("### This is Markdown")

# Display data
st.write("## Display Data")

# Create a sample DataFrame
data = pd.DataFrame({
    'Name': ['Alice', 'Bob', 'Charlie', 'David', 'Eva'],
    'Age': [25, 30, 35, 40, 45]
})

# Display the DataFrame
st.dataframe(data)

# Display a line chart
st.line_chart(data['Age'])

# Display a bar chart
st.bar_chart(data.set_index('Name'))

# Display a scatter plot
st.write("### Scatter Plot")
st.scatter_chart(data)

# Interactive widgets
st.write("## Interactive Widgets")

# Slider
age_slider = st.slider("Select an age:", min_value=25, max_value=45, value=30)
st.write(f"Selected age: {age_slider}")

# Checkbox
if st.checkbox("Show Data"):
    st.write(data)

# Selectbox
name_select = st.selectbox("Select a name:", data['Name'])
st.write(f"Selected name: {name_select}")

# Radio button
color = st.radio("Select a color:", ["Red", "Green", "Blue"])
st.write(f"Selected color: {color}")

# Button
if st.button("Click Me"):
    st.write("Button clicked!")

# Sidebar
st.sidebar.header("Sidebar")
st.sidebar.text("This is the sidebar.")

# File uploader
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("Uploaded file data:")
    st.dataframe(df)

# Show a map
st.map(np.random.randn(1000, 2) / [50, 50
