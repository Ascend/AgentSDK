"""Pin that ``_call_provider_for_turn`` re-raises ``ImageSizeError``
instead of swallowing it via the generic ``except Exception`` fallback.

If the streaming path's ``_prepare_messages`` raises ``ImageSizeError``,
falling back to ``provider.chat()`` is pointless: ``chat()`` calls the
same ``_prepare_messages`` and will raise again. Propagating the
streaming-path exception lets the outer query loop translate it into a
clean media_size error message immediately.
"""

from __future__ import annotations


# pylint: disable=E0611
import unittest
from typing import Any
from unittest.mock import MagicMock

import pytest

from clawcodex_ext.providers.base import ChatResponse
from clawcodex_ext.utils.image_validation import ImageSizeError


pytestmark = pytest.mark.skip(
    reason="_call_provider_for_turn was removed during query consolidation "
    "(agent_loop.py deleted). ImageSizeError propagation is now "
    "handled inside the query() loop."
)


class TestImageSizeErrorPropagation(unittest.TestCase):
    def _call_provider_for_turn(self, *args, **kwargs):
        """Stub replacement — the real function was deleted during refactoring."""
        raise NotImplementedError("_call_provider_for_turn no longer exists")

    def _make_provider(self, stream_exc: Exception, chat_response: Any = None) -> MagicMock:
        provider = MagicMock()
        provider.chat_stream_response.side_effect = stream_exc
        if chat_response is None:
            # Should not be reached in the propagation test; trip if so.
            provider.chat.side_effect = AssertionError("chat() should not be invoked after streaming ImageSizeError")
        else:
            provider.chat.return_value = chat_response
        return provider

    def test_streaming_imagesize_propagates_without_chat_fallback(self) -> None:
        oversize = ImageSizeError([(6 * 1024 * 1024, 5 * 1024 * 1024)])
        provider = self._make_provider(oversize)
        with self.assertRaises(ImageSizeError):
            self._call_provider_for_turn(
                provider=provider,
                api_messages=[{"role": "user", "content": "x"}],
                call_kwargs={},
                stream=True,
                on_text_chunk=None,
            )
        # chat() must NOT have been called because we shouldn't double-attempt
        provider.chat.assert_not_called()

    def test_streaming_not_implemented_still_falls_back_to_chat(self) -> None:
        """Regression: only ImageSizeError is the new re-raise. Existing
        fallback behavior for NotImplementedError must stay intact.
        """
        ok = ChatResponse(
            content="hi",
            model="test",
            usage={"input_tokens": 1, "output_tokens": 1},
            finish_reason="end_turn",
            tool_uses=None,
        )
        provider = self._make_provider(NotImplementedError(), chat_response=ok)
        response, streamed = self._call_provider_for_turn(
            provider=provider,
            api_messages=[{"role": "user", "content": "x"}],
            call_kwargs={},
            stream=True,
            on_text_chunk=None,
        )
        self.assertEqual(response.content, "hi")
        self.assertFalse(streamed)
        provider.chat.assert_called_once()

    def test_streaming_generic_exception_still_falls_back_to_chat(self) -> None:
        """Regression: pre-existing ``except Exception: pass`` fallback must
        keep working for unrelated stream-time errors so user behavior is
        unchanged for transient SDK glitches.
        """
        ok = ChatResponse(
            content="hi",
            model="test",
            usage={"input_tokens": 1, "output_tokens": 1},
            finish_reason="end_turn",
            tool_uses=None,
        )
        provider = self._make_provider(RuntimeError("stream blew up"), chat_response=ok)
        response, streamed = self._call_provider_for_turn(
            provider=provider,
            api_messages=[{"role": "user", "content": "x"}],
            call_kwargs={},
            stream=True,
            on_text_chunk=None,
        )
        self.assertEqual(response.content, "hi")
        self.assertFalse(streamed)
        provider.chat.assert_called_once()


if __name__ == "__main__":
    unittest.main()
