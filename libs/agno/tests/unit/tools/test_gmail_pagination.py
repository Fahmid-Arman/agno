"""Tests for Gmail pagination and max_results configuration."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from agno.tools.google.auth import AuthConfig
from agno.tools.google.gmail import GmailTools


class TestMaxResultsConfig:
    """Test max_results configuration via AuthConfig."""

    def test_default_max_results(self):
        """AuthConfig defaults to 20 max results per request."""
        auth = AuthConfig()
        assert auth.max_results == 20

    def test_custom_max_results(self):
        """AuthConfig accepts custom max_results."""
        auth = AuthConfig(max_results=25)
        assert auth.max_results == 25

    def test_max_results_limit_uses_config(self):
        """_max_results_limit returns config value."""
        auth = AuthConfig(max_results=25)
        toolkit = GmailTools.__new__(GmailTools)
        toolkit._auth = auth
        assert toolkit._max_results_limit() == 25

    def test_max_results_limit_default(self):
        """_max_results_limit returns 20 when no auth."""
        toolkit = GmailTools.__new__(GmailTools)
        toolkit._auth = None
        assert toolkit._max_results_limit() == 20


class TestGmailPagination:
    """Test pagination in Gmail list methods."""

    @pytest.fixture
    def mock_gmail_service(self):
        """Create a mock Gmail service."""
        service = MagicMock()
        return service

    @pytest.fixture
    def gmail_tools(self, mock_gmail_service):
        """Create GmailTools with mocked service."""
        with patch.object(GmailTools, "_resolve_creds", return_value=MagicMock(valid=True)):
            with patch.object(GmailTools, "_build_service", return_value=mock_gmail_service):
                auth = AuthConfig(max_results=10)
                tools = GmailTools(auth=auth)
                tools._service = mock_gmail_service
                tools.creds = MagicMock(valid=True)
                return tools

    def test_get_latest_emails_caps_count(self, gmail_tools, mock_gmail_service):
        """get_latest_emails caps count to max_results."""
        mock_gmail_service.users.return_value.messages.return_value.list.return_value.execute.return_value = {
            "messages": []
        }

        gmail_tools.get_latest_emails(count=100)

        call_kwargs = mock_gmail_service.users().messages().list.call_args[1]
        assert call_kwargs["maxResults"] == 10  # Capped to auth config

    def test_get_latest_emails_passes_page_token(self, gmail_tools, mock_gmail_service):
        """get_latest_emails passes page_token to API."""
        mock_gmail_service.users.return_value.messages.return_value.list.return_value.execute.return_value = {
            "messages": []
        }

        gmail_tools.get_latest_emails(count=5, page_token="test_token_123")

        call_kwargs = mock_gmail_service.users().messages().list.call_args[1]
        assert call_kwargs["pageToken"] == "test_token_123"

    def test_get_latest_emails_returns_next_page_token(self, gmail_tools, mock_gmail_service):
        """get_latest_emails includes nextPageToken in response when available."""
        mock_gmail_service.users.return_value.messages.return_value.list.return_value.execute.return_value = {
            "messages": [{"id": "msg1"}],
            "nextPageToken": "next_page_abc",
        }
        mock_gmail_service.users.return_value.messages.return_value.get.return_value.execute.return_value = {
            "id": "msg1",
            "payload": {"headers": []},
        }

        result = gmail_tools.get_latest_emails(count=5)

        assert "next_page_abc" in result
        assert "More results available" in result

    def test_search_threads_returns_next_page_token_in_json(self, gmail_tools, mock_gmail_service):
        """search_threads includes nextPageToken in JSON response."""
        import json

        mock_gmail_service.users.return_value.threads.return_value.list.return_value.execute.return_value = {
            "threads": [{"id": "thread1", "snippet": "test"}],
            "nextPageToken": "thread_page_xyz",
            "resultSizeEstimate": 100,
        }

        result = gmail_tools.search_threads(query="is:unread", count=5)
        parsed = json.loads(result)

        assert parsed["nextPageToken"] == "thread_page_xyz"
        assert "threads" in parsed

    def test_list_drafts_returns_next_page_token_in_json(self, gmail_tools, mock_gmail_service):
        """list_drafts includes nextPageToken in JSON response."""
        import json

        mock_gmail_service.users.return_value.drafts.return_value.list.return_value.execute.return_value = {
            "drafts": [{"id": "draft1"}],
            "nextPageToken": "draft_page_123",
            "resultSizeEstimate": 50,
        }

        result = gmail_tools.list_drafts(count=5)
        parsed = json.loads(result)

        assert parsed["nextPageToken"] == "draft_page_123"

    def test_backward_compatibility_no_page_token(self, gmail_tools, mock_gmail_service):
        """Methods work without page_token (backward compatible)."""
        mock_gmail_service.users.return_value.messages.return_value.list.return_value.execute.return_value = {
            "messages": []
        }

        # Should not raise - page_token defaults to None
        gmail_tools.get_latest_emails(count=5)
        gmail_tools.get_unread_emails(count=5)
        gmail_tools.search_emails(query="test", count=5)

    def test_no_next_page_token_when_exhausted(self, gmail_tools, mock_gmail_service):
        """No pagination message when all results returned."""
        mock_gmail_service.users.return_value.messages.return_value.list.return_value.execute.return_value = {
            "messages": []
            # No nextPageToken = no more results
        }

        result = gmail_tools.get_latest_emails(count=5)

        assert "More results available" not in result
        assert "page_token" not in result
