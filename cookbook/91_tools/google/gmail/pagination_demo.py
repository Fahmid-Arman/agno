"""
Gmail Pagination Demo

Demonstrates the max_results capping and page_token pagination features.
The agent will fetch emails in pages, showing that pagination works correctly.
"""

from agno.agent import Agent
from agno.models.openai import OpenAIResponses
from agno.tools.google.gmail import GmailTools

# Create Gmail tools with a small max_results to demonstrate pagination
gmail_tools = GmailTools(max_results=5)

agent = Agent(
    model=OpenAIResponses(id="gpt-5.5"),
    tools=[gmail_tools],
    instructions="You are a Gmail assistant. When asked to fetch emails, use the Gmail tools.",
    markdown=True,
)

if __name__ == "__main__":
    # Test 1: Agent requests many emails but gets capped
    agent.print_response(
        "Get my latest 50 emails. Just show me the count and first 2 subjects.",
        stream=True,
    )

    # Test 2: Agent uses search_threads which returns nextPageToken
    agent.print_response(
        "Search for threads from the last 7 days. Return the thread IDs and let me know if there are more pages available.",
        stream=True,
    )
