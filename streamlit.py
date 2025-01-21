import streamlit as st
from phi.agent import Agent
from phi.tools.yfinance import YFinanceTools
from phi.tools.duckduckgo import DuckDuckGo
from phi.model.groq import Groq
from dotenv import load_dotenv
import os
import phi

# Load environment variables
load_dotenv()

# Set PHI API key
phi.api = os.getenv("PHI_API_KEY")

# Define agents
web_search_agent = Agent(
    name="Web Search Agent",
    role="Search the web for information",
    model=Groq(id="llama-3.3-70b-versatile"),
    tools=[DuckDuckGo()],
    instructions=["Always include sources"],
    show_tools_calls=True,
    markdown=True,
)

finance_agent = Agent(
    name="Finance AI Agent",
    model=Groq(id="llama-3.3-70b-versatile"),
    tools=[
        YFinanceTools(
            stock_price=True,
            analyst_recommendations=True,
            stock_fundamentals=True,
            company_news=True,
        ),
    ],
    instructions=["Use tables to display the data"],
    show_tool_calls=True,
    markdown=True,
)

# Streamlit app logic
def main():
    st.title("AI Agent App")

    # Tabs for different agents
    tab = st.sidebar.selectbox("Choose an Agent", ["Web Search Agent", "Finance Agent"])

    if tab == "Web Search Agent":
        st.header("Web Search Agent")
        query = st.text_input("Enter your search query:")
        if st.button("Search"):
            with st.spinner("Searching the web..."):
                result = web_search_agent.run(query)
                st.markdown(result)

    elif tab == "Finance Agent":
        st.header("Finance Agent")
        stock_ticker = st.text_input("Enter stock ticker (e.g., AAPL):")
        if st.button("Get Financial Data"):
            with st.spinner("Fetching financial data..."):
                result = finance_agent.run(f"Get data for {stock_ticker}")
                st.markdown(result)


# Run the Streamlit app
if __name__ == "__main__":
    main()
