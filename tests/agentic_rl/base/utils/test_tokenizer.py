#!/usr/bin/env python3
# coding=utf-8
# -------------------------------------------------------------------------# This file is part of the AgentSDK project.
# Copyright (c) 2026 Huawei Co.,Ltd.
#
# AgentSDK is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#
#          http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# -------------------------------------------------------------------------
import pytest
import unittest.mock as mock
import numpy as np

from agentic_rl.base.utils.tokenizer import (
    get_tokenizer,
    BaseTokenizer,
    _HuggingFaceTokenizer,
    replace_token_from_template,
    _add_or_replace_eos_token
)


class TestBaseTokenizer:
    """Test cases for BaseTokenizer abstract class."""
    
    def test_init(self):
        """Test BaseTokenizer initialization."""
        # Create a concrete implementation for testing
        class ConcreteTokenizer(BaseTokenizer):
            @property
            def vocab(self):
                return {}
            
            @property
            def inv_vocab(self):
                return {}
            
            @property
            def vocab_size(self):
                return 0
            
            def tokenize(self, text):
                return np.array([])
        
        tokenizer = ConcreteTokenizer("path/to/tokenizer", option1="value1", option2="value2")
        
        # Check unique_identifiers
        assert tokenizer.unique_identifiers["class"] == "ConcreteTokenizer"
        assert tokenizer.unique_identifiers["tokenizer_path"] == ["path/to/tokenizer"]
        assert tokenizer.unique_identifiers["option1"] == "value1"
        assert tokenizer.unique_identifiers["option2"] == "value2"
        
        # Check unique_description is a valid JSON string
        import json
        assert json.loads(tokenizer.unique_description) == tokenizer.unique_identifiers
    
    def test_optional_properties_raise(self):
        """Test that optional properties raise NotImplementedError."""
        # Create a minimal concrete implementation
        class MinimalTokenizer(BaseTokenizer):
            @property
            def vocab(self):
                return {}
            
            @property
            def inv_vocab(self):
                return {}
            
            @property
            def vocab_size(self):
                return 0
            
            def tokenize(self, text):
                return np.array([])
        
        tokenizer = MinimalTokenizer("path/to/tokenizer")
        
        # Test detokenize method
        with pytest.raises(NotImplementedError):
            tokenizer.detokenize(np.array([1, 2, 3]))
        
        # Test optional properties
        for prop in ["cls", "sep", "pad", "eod", "bos", "eos", "mask"]:
            with pytest.raises(NotImplementedError):
                getattr(tokenizer, prop)


class TestHuggingFaceTokenizer:
    """Test cases for _HuggingFaceTokenizer class."""
    
    def setup_method(self):
        """Set up mocks for each test."""
        # Mock transformers module
        self.mock_transformers = mock.MagicMock()
        self.mock_tokenizer_instance = mock.MagicMock()
        self.mock_transformers.AutoTokenizer.from_pretrained.return_value = self.mock_tokenizer_instance
        
        # Set up mock tokenizer properties
        self.mock_tokenizer_instance.get_vocab.return_value = {"hello": 1, "world": 2}
        self.mock_tokenizer_instance.__len__.return_value = 1000
        
        # Mock the __call__ method to return an object with input_ids
        self.mock_tokenizer_instance.return_value = mock.MagicMock()
        self.mock_tokenizer_instance.return_value.input_ids = [1, 2, 3]
        
        self.mock_tokenizer_instance.decode.return_value = "hello world"
        self.mock_tokenizer_instance.eos_token_id = 50256
        self.mock_tokenizer_instance.eos_token = "<|endoftext|>"
        self.mock_tokenizer_instance.pad_token_id = None
        self.mock_tokenizer_instance.pad_token = None
        
        # Patch transformers module
        self.mock_module = mock.patch.dict("sys.modules", {
            "transformers": self.mock_transformers
        })
        self.mock_module.start()
        
        # Create tokenizer instance
        self.tokenizer = _HuggingFaceTokenizer("path/to/tokenizer")
    
    def teardown_method(self):
        """Clean up mocks after each test."""
        self.mock_module.stop()
    
    def test_init(self):
        """Test _HuggingFaceTokenizer initialization."""
        self.mock_transformers.AutoTokenizer.from_pretrained.assert_called_once_with(
            pretrained_model_name_or_path="path/to/tokenizer"
        )
        
        assert self.tokenizer._vocab == {"hello": 1, "world": 2}
        assert self.tokenizer._inv_vocab == {1: "hello", 2: "world"}
    
    def test_vocab_size(self):
        """Test vocab_size property."""
        assert self.tokenizer.vocab_size == 1000
    
    def test_vocab(self):
        """Test vocab property."""
        assert self.tokenizer.vocab == {"hello": 1, "world": 2}
    
    def test_inv_vocab(self):
        """Test inv_vocab property."""
        assert self.tokenizer.inv_vocab == {1: "hello", 2: "world"}
        assert self.tokenizer.decoder == self.tokenizer.inv_vocab  # decoder is alias for inv_vocab
    
    def test_tokenize(self):
        """Test tokenize method."""
        result = self.tokenizer.tokenize("hello world")
        assert result == [1, 2, 3]
        self.mock_tokenizer_instance.assert_called_once_with("hello world")
    
    def test_detokenize(self):
        """Test detokenize method."""
        result = self.tokenizer.detokenize([1, 2, 3])
        assert result == "hello world"
        self.mock_tokenizer_instance.decode.assert_called_once_with([1, 2, 3])
    
    def test_eod_properties(self):
        """Test eod and eod_token properties."""
        assert self.tokenizer.eod == 50256
        assert self.tokenizer.eod_token == "<|endoftext|>"
        
        # Test setters
        self.tokenizer.eod = 1000
        assert self.mock_tokenizer_instance.eos_token_id == 1000
        
        self.tokenizer.eod_token = "<|end|>"
        assert self.mock_tokenizer_instance.eos_token == "<|end|>"
    
    def test_pad_properties(self):
        """Test pad and pad_token properties."""
        assert self.tokenizer.pad is None
        assert self.tokenizer.pad_token is None
        
        # Test setters
        self.tokenizer.pad = 2000
        assert self.mock_tokenizer_instance.pad_token_id == 2000
        
        self.tokenizer.pad_token = "<|pad|>"
        assert self.mock_tokenizer_instance.pad_token == "<|pad|>"
    
    def test_import_error(self):
        """Test that ImportError is raised when transformers is not available."""
        # Stop previous mock
        self.mock_module.stop()
        
        # Mock ModuleNotFoundError
        with mock.patch.dict("sys.modules", {"transformers": None}):
            with pytest.raises(ImportError):
                _HuggingFaceTokenizer("path/to/tokenizer")
        
        # Restart mock for other tests
        self.mock_module.start()


class TestGetTokenizer:
    """Test cases for get_tokenizer function."""
    
    def setup_method(self):
        """Set up mocks for each test."""
        # Mock _HuggingFaceTokenizer
        self.mock_hf_tokenizer = mock.MagicMock()
        self.mock_hf_tokenizer.eod_token = None
        self.mock_hf_tokenizer.eod = None
        self.mock_hf_tokenizer.pad_token = None
        self.mock_hf_tokenizer.pad = None
        
        # Get the actual mock class object
        self.mock_hf_class_obj = mock.MagicMock(return_value=self.mock_hf_tokenizer)
        self.mock_hf_class = mock.patch(
            "agentic_rl.base.utils.tokenizer._HuggingFaceTokenizer",
            self.mock_hf_class_obj
        )
        self.mock_hf_class.start()
        
        # Mock os.path.isdir
        self.mock_isdir = mock.patch("os.path.isdir", return_value=True)
        self.mock_isdir.start()
    
    def teardown_method(self):
        """Clean up mocks after each test."""
        self.mock_hf_class.stop()
        self.mock_isdir.stop()
    
    def test_get_huggingface_tokenizer(self):
        """Test getting HuggingFaceTokenizer."""
        # Don't set eod_token on mock_hf_tokenizer since we'll pass it in get_tokenizer
        
        result = get_tokenizer(
            tokenizer_model="path/to/tokenizer",
            tokenizer_type="HuggingFaceTokenizer",
            eos_token_id=50256,
            eos_token="<|endoftext|>"
        )
        
        assert result == self.mock_hf_tokenizer
        self.mock_hf_class_obj.assert_called_once_with("path/to/tokenizer")
        
        # Verify that eod_token and eod were set correctly
        assert self.mock_hf_tokenizer.eod_token == "<|endoftext|>"
        assert self.mock_hf_tokenizer.eod == 50256
    
    def test_invalid_tokenizer_type(self):
        """Test that ValueError is raised for invalid tokenizer_type."""
        with pytest.raises(NotImplementedError):
            get_tokenizer(
                tokenizer_model="path/to/tokenizer",
                tokenizer_type="InvalidTokenizer"
            )
    
    def test_invalid_tokenizer_model_path(self):
        """Test that ValueError is raised for invalid tokenizer_model path."""
        # Mock isdir to return False
        self.mock_isdir.stop()
        with mock.patch("os.path.isdir", return_value=False):
            with pytest.raises(ValueError):
                get_tokenizer(
                    tokenizer_model="invalid/path",
                    tokenizer_type="HuggingFaceTokenizer"
                )
        self.mock_isdir.start()
    
    def test_token_validation(self):
        """Test validation of token parameters."""
        # Test without matching token and id
        with pytest.raises(ValueError):
            get_tokenizer(
                tokenizer_model="path/to/tokenizer",
                eos_token_id=50256
            )
        
        with pytest.raises(ValueError):
            get_tokenizer(
                tokenizer_model="path/to/tokenizer",
                eos_token="<|endoftext|>"
            )
        
        with pytest.raises(ValueError):
            get_tokenizer(
                tokenizer_model="path/to/tokenizer",
                pad_token_id=0
            )
        
        with pytest.raises(ValueError):
            get_tokenizer(
                tokenizer_model="path/to/tokenizer",
                pad_token="<|pad|>"
            )
    
    def test_existing_token_validation(self):
        """Test validation when tokenizer already has tokens."""
        # Set up tokenizer with existing tokens
        self.mock_hf_tokenizer.eod_token = "<|existing|>"
        self.mock_hf_tokenizer.eod = 123
        self.mock_hf_tokenizer.pad_token = "<|pad|>"
        self.mock_hf_tokenizer.pad = 0
        
        # Test adding eos token when one already exists
        with pytest.raises(ValueError):
            get_tokenizer(
                tokenizer_model="path/to/tokenizer",
                eos_token_id=50256,
                eos_token="<|endoftext|>"
            )
        
        # Test adding pad token when one already exists
        with pytest.raises(ValueError):
            # Reset the mock to allow new calls, but keep existing tokens for this test
            self.mock_hf_class_obj.reset_mock()
            # Don't reset existing tokens since we want to test the validation
            get_tokenizer(
                tokenizer_model="path/to/tokenizer",
                eos_token_id=123,  # Use existing eod
                eos_token="<|existing|>",  # Use existing eod_token
                pad_token_id=1000,
                pad_token="<|new_pad|>"
            )
        
        # Reset tokens for other tests
        self.mock_hf_tokenizer.eod_token = None
        self.mock_hf_tokenizer.eod = None
        self.mock_hf_tokenizer.pad_token = None
        self.mock_hf_tokenizer.pad = None
    
    def test_eos_for_pad(self):
        """Test eos_for_pad functionality."""
        # Don't set eod_token on mock_hf_tokenizer since we'll pass it in get_tokenizer
        self.mock_hf_tokenizer.pad_token = None
        self.mock_hf_tokenizer.pad = None
        
        # Test with eos_for_pad=True (default)
        result = get_tokenizer(
            tokenizer_model="path/to/tokenizer",
            eos_token_id=50256,
            eos_token="<|endoftext|>"
        )
        
        assert result.pad_token == "<|endoftext|>"
        assert result.pad == 50256
        
        # Test with eos_for_pad=False
        # Reset the mock to allow new calls
        self.mock_hf_class_obj.reset_mock()
        self.mock_hf_tokenizer.eod_token = None  # Reset eod_token
        self.mock_hf_tokenizer.eod = None  # Reset eod
        self.mock_hf_tokenizer.pad_token = None
        self.mock_hf_tokenizer.pad = None
        
        result = get_tokenizer(
            tokenizer_model="path/to/tokenizer",
            eos_token_id=50256,
            eos_token="<|endoftext|>",
            eos_for_pad=False
        )
        
        assert result.pad_token is None
        assert result.pad is None
    
    def test_with_prompt_type(self):
        """Test get_tokenizer with prompt_type and prompt_type_path."""
        # Don't set eod_token on mock_hf_tokenizer since we'll pass it in get_tokenizer
        
        # Reset the mock to allow new calls
        self.mock_hf_class_obj.reset_mock()
        self.mock_hf_tokenizer.eod_token = None  # Reset eod_token
        self.mock_hf_tokenizer.eod = None  # Reset eod
        self.mock_hf_tokenizer.pad_token = None
        self.mock_hf_tokenizer.pad = None
        
        # Mock replace_token_from_template
        with mock.patch("agentic_rl.base.utils.tokenizer.replace_token_from_template") as mock_replace:
            result = get_tokenizer(
                tokenizer_model="path/to/tokenizer",
                eos_token_id=50256,
                eos_token="<|endoftext|>",
                prompt_type="test_template",
                prompt_type_path="path/to/templates.json"
            )
            
            mock_replace.assert_called_once_with(
                self.mock_hf_tokenizer.tokenizer,
                "test_template",
                "path/to/templates.json"
            )


class TestReplaceTokenFromTemplate:
    """Test cases for replace_token_from_template function."""
    
    def setup_method(self):
        """Set up mocks for each test."""
        # Mock tokenizer
        self.mock_tokenizer = mock.MagicMock()
        self.mock_tokenizer.eos_token_id = 50256
        self.mock_tokenizer.eos_token = "<|endoftext|>"
        self.mock_tokenizer.pad_token_id = None
        self.mock_tokenizer.pad_token = None
        self.mock_tokenizer.add_special_tokens.return_value = 0
        
        # Mock get_model_template
        self.mock_template = mock.MagicMock()
        self.mock_template.stop_words = ["<|stop1|>", "<|stop2|>"]
        self.mock_template.replace_eos = False
        
        self.mock_get_template = mock.patch(
            "agentic_rl.base.utils.tokenizer.get_model_template",
            return_value=self.mock_template
        )
        self.mock_get_template.start()
        
        # Mock _add_or_replace_eos_token
        self.mock_add_eos = mock.patch(
            "agentic_rl.base.utils.tokenizer._add_or_replace_eos_token"
        )
        self.mock_add_eos.start()
    
    def teardown_method(self):
        """Clean up mocks after each test."""
        self.mock_get_template.stop()
        self.mock_add_eos.stop()
    
    def test_replace_token_from_template_basic(self):
        """Test basic functionality of replace_token_from_template."""
        replace_token_from_template(self.mock_tokenizer, "test_template", "path/to/templates.json")
        
        # Check that pad token is set
        assert self.mock_tokenizer.pad_token == "<|endoftext|>"
        
        # Check that stop words are added
        self.mock_tokenizer.add_special_tokens.assert_called_once_with(
            dict(additional_special_tokens=["<|stop1|>", "<|stop2|>"]),
            replace_additional_special_tokens=False
        )
    
    def test_replace_token_from_template_with_replace_eos(self):
        """Test replace_token_from_template with replace_eos=True."""
        self.mock_template.replace_eos = True
        
        replace_token_from_template(self.mock_tokenizer, "test_template", "path/to/templates.json")
        
        # Check that _add_or_replace_eos_token is called with first stop word
        from agentic_rl.base.utils.tokenizer import _add_or_replace_eos_token
        _add_or_replace_eos_token.assert_called_once_with(self.mock_tokenizer, eos_token="<|stop1|>")
        
        # Check that remaining stop words are added
        self.mock_tokenizer.add_special_tokens.assert_called_once_with(
            dict(additional_special_tokens=["<|stop2|>"]),
            replace_additional_special_tokens=False
        )
    
    def test_replace_token_from_template_no_stop_words(self):
        """Test replace_token_from_template with replace_eos=True but no stop words."""
        self.mock_template.replace_eos = True
        self.mock_template.stop_words = []
        
        with pytest.raises(ValueError):
            replace_token_from_template(self.mock_tokenizer, "test_template", "path/to/templates.json")
    
    def test_replace_token_from_template_no_eos(self):
        """Test replace_token_from_template when tokenizer has no eos token."""
        self.mock_tokenizer.eos_token_id = None
        self.mock_tokenizer.eos_token = None
        
        replace_token_from_template(self.mock_tokenizer, "test_template", "path/to/templates.json")
        
        # Check that eos token is added
        from agentic_rl.base.utils.tokenizer import _add_or_replace_eos_token
        _add_or_replace_eos_token.assert_called_once_with(self.mock_tokenizer, eos_token="<|endoftext|>")


class TestAddOrReplaceEosToken:
    """Test cases for _add_or_replace_eos_token function."""
    
    def setup_method(self):
        """Set up mocks for each test."""
        self.mock_tokenizer = mock.MagicMock()
        self.mock_tokenizer.add_special_tokens.return_value = 0
    
    def test_add_eos_token(self):
        """Test adding eos token when none exists."""
        self.mock_tokenizer.eos_token_id = None
        
        _add_or_replace_eos_token(self.mock_tokenizer, "<|endoftext|>")
        
        self.mock_tokenizer.add_special_tokens.assert_called_once_with({
            "eos_token": "<|endoftext|>"
        })
    
    def test_replace_eos_token(self):
        """Test replacing existing eos token."""
        self.mock_tokenizer.eos_token_id = 50256
        
        _add_or_replace_eos_token(self.mock_tokenizer, "<|new_end|>")
        
        self.mock_tokenizer.add_special_tokens.assert_called_once_with({
            "eos_token": "<|new_end|>"
        })
    
    def test_add_new_eos_token(self):
        """Test adding eos token that requires new token."""
        self.mock_tokenizer.eos_token_id = None
        self.mock_tokenizer.add_special_tokens.return_value = 1  # Indicate new token added
        
        _add_or_replace_eos_token(self.mock_tokenizer, "<|new_end|>")
        
        self.mock_tokenizer.add_special_tokens.assert_called_once_with({
            "eos_token": "<|new_end|>"
        })