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

    def test_get_latest_emails_returns_formatted_emails(self, gmail_tools, mock_gmail_service):
        """get_latest_emails returns formatted email content."""
        mock_gmail_service.users.return_value.messages.return_value.list.return_value.execute.return_value = {
            "messages": [{"id": "msg1"}],
        }
        mock_gmail_service.users.return_value.messages.return_value.get.return_value.execute.return_value = {
            "id": "msg1",
            "payload": {"headers": []},
        }

        result = gmail_tools.get_latest_emails(count=5)

        assert "Message ID: msg1" in result

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

    def test_empty_results(self, gmail_tools, mock_gmail_service):
        """Empty results returns no emails found message."""
        mock_gmail_service.users.return_value.messages.return_value.list.return_value.execute.return_value = {
            "messages": []
        }

        result = gmail_tools.get_latest_emails(count=5)

        assert "No emails found" in result


class TestBatchOperations:
    """Test batch operations using batchModify endpoint."""

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

    def test_single_message_uses_modify(self, gmail_tools, mock_gmail_service):
        """Single message ID uses messages.modify (not batchModify)."""
        mock_gmail_service.users.return_value.messages.return_value.modify.return_value.execute.return_value = {}

        gmail_tools.mark_email_as_read("msg1")

        mock_gmail_service.users().messages().modify.assert_called()
        mock_gmail_service.users().messages().batchModify.assert_not_called()

    def test_multiple_messages_uses_batch_modify(self, gmail_tools, mock_gmail_service):
        """Multiple message IDs use messages.batchModify."""
        mock_gmail_service.users.return_value.messages.return_value.batchModify.return_value.execute.return_value = {}

        gmail_tools.mark_email_as_read("msg1,msg2,msg3")

        mock_gmail_service.users().messages().batchModify.assert_called()
        call_kwargs = mock_gmail_service.users().messages().batchModify.call_args[1]
        assert call_kwargs["body"]["ids"] == ["msg1", "msg2", "msg3"]
        assert call_kwargs["body"]["removeLabelIds"] == ["UNREAD"]

    def test_star_multiple_emails(self, gmail_tools, mock_gmail_service):
        """star_email works with multiple IDs."""
        mock_gmail_service.users.return_value.messages.return_value.batchModify.return_value.execute.return_value = {}

        result = gmail_tools.star_email("msg1,msg2")

        assert "2 email(s)" in result
        call_kwargs = mock_gmail_service.users().messages().batchModify.call_args[1]
        assert call_kwargs["body"]["addLabelIds"] == ["STARRED"]

    def test_archive_multiple_emails(self, gmail_tools, mock_gmail_service):
        """archive_email works with multiple IDs."""
        mock_gmail_service.users.return_value.messages.return_value.batchModify.return_value.execute.return_value = {}

        result = gmail_tools.archive_email("msg1,msg2,msg3")

        assert "3 email(s)" in result
        call_kwargs = mock_gmail_service.users().messages().batchModify.call_args[1]
        assert call_kwargs["body"]["removeLabelIds"] == ["INBOX"]

    def test_modify_message_labels_batch(self, gmail_tools, mock_gmail_service):
        """modify_message_labels uses batchModify for multiple IDs."""
        import json

        mock_gmail_service.users.return_value.messages.return_value.batchModify.return_value.execute.return_value = {}
        mock_gmail_service.users.return_value.labels.return_value.list.return_value.execute.return_value = {
            "labels": [{"id": "Label_1", "name": "Work", "type": "user"}]
        }

        result = gmail_tools.modify_message_labels(message_id="msg1,msg2", add_labels="STARRED", remove_labels="UNREAD")
        parsed = json.loads(result)

        assert parsed["modified"] == 2
        assert parsed["message_ids"] == ["msg1", "msg2"]

    def test_modify_message_labels_single(self, gmail_tools, mock_gmail_service):
        """modify_message_labels uses modify for single ID."""
        import json

        mock_gmail_service.users.return_value.messages.return_value.modify.return_value.execute.return_value = {
            "id": "msg1",
            "labelIds": ["STARRED"],
        }
        mock_gmail_service.users.return_value.labels.return_value.list.return_value.execute.return_value = {
            "labels": []
        }

        result = gmail_tools.modify_message_labels(message_id="msg1", add_labels="STARRED")
        parsed = json.loads(result)

        assert parsed["id"] == "msg1"
        assert "STARRED" in parsed["labelIds"]

    def test_accepts_list_input(self, gmail_tools, mock_gmail_service):
        """Batch methods accept list input in addition to comma-separated string."""
        mock_gmail_service.users.return_value.messages.return_value.batchModify.return_value.execute.return_value = {}

        result = gmail_tools.mark_email_as_read(["msg1", "msg2", "msg3"])

        assert "3 email(s)" in result
        call_kwargs = mock_gmail_service.users().messages().batchModify.call_args[1]
        assert call_kwargs["body"]["ids"] == ["msg1", "msg2", "msg3"]
