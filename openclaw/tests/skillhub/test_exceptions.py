#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -------------------------------------------------------------------------
#  This file is part of the AgentSDK project.
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
#
# AgentSDK is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#
#           http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# -------------------------------------------------------------------------


import pytest

from skillhub.exceptions.base import (
    SkillHubError,
    ValidationError,
    NetworkError,
    AuthenticationError,
    SourceError,
    ResolutionError,
    InstallError,
    SecurityError,
    NotFoundError,
)


class TestSkillHubError:
    """Tests for base SkillHubError."""

    def test_error_creation(self):
        """Test creating base error."""
        error = SkillHubError("Test error message")
        assert str(error) == "Test error message"
        assert error.code == "E_UNKNOWN"
        assert error.exit_code == 1

    def test_error_with_cause(self):
        """Test error with underlying cause."""
        cause = ValueError("Underlying error")
        error = SkillHubError("Test error", cause=cause)
        assert "Underlying error" in str(error)

    def test_error_with_context(self):
        """Test error with context data."""
        error = SkillHubError("Test error", context={"key": "value"})
        assert error.context == {"key": "value"}

    def test_error_with_cause_and_context(self):
        """Test error with both cause and context."""
        cause = ValueError("Cause")
        error = SkillHubError("Test", cause=cause, context={"k": "v"})
        assert error.cause == cause
        assert error.context == {"k": "v"}

    def test_error_str_without_cause(self):
        """Test string representation without cause."""
        error = SkillHubError("Simple error")
        assert str(error) == "Simple error"

    def test_error_str_with_cause(self):
        """Test string representation with cause."""
        cause = RuntimeError("Runtime cause")
        error = SkillHubError("Wrapper error", cause=cause)
        error_str = str(error)
        assert "Wrapper error" in error_str
        assert "Runtime cause" in error_str


class TestValidationError:
    """Tests for ValidationError."""

    def test_validation_error_creation(self):
        """Test creating validation error."""
        error = ValidationError("Invalid input")
        assert str(error) == "Invalid input"
        assert error.code == "E_VALIDATION"
        assert error.exit_code == 1

    def test_validation_error_inheritance(self):
        """Test that ValidationError inherits from SkillHubError."""
        error = ValidationError("Test")
        assert isinstance(error, SkillHubError)
        assert isinstance(error, Exception)


class TestNetworkError:
    """Tests for NetworkError."""

    def test_network_error_creation(self):
        """Test creating network error."""
        error = NetworkError("Connection failed")
        assert str(error) == "Connection failed"
        assert error.code == "E_NETWORK"
        assert error.exit_code == 2

    def test_network_error_with_cause(self):
        """Test network error with underlying cause."""
        cause = ConnectionError("Network unreachable")
        error = NetworkError("API call failed", cause=cause)
        assert error.cause == cause


class TestAuthenticationError:
    """Tests for AuthenticationError."""

    def test_auth_error_creation(self):
        """Test creating authentication error."""
        error = AuthenticationError("Invalid token")
        assert str(error) == "Invalid token"
        assert error.code == "E_AUTH"
        assert error.exit_code == 3

    def test_auth_error_with_context(self):
        """Test auth error with context."""
        error = AuthenticationError(
            "Auth failed",
            context={"platform": "github", "status": 401},
        )
        assert error.context["platform"] == "github"


class TestSourceError:
    """Tests for SourceError."""

    def test_source_error_creation(self):
        """Test creating source error."""
        error = SourceError("Source not found")
        assert str(error) == "Source not found"
        assert error.code == "E_SOURCE"
        assert error.exit_code == 4


class TestResolutionError:
    """Tests for ResolutionError."""

    def test_resolution_error_creation(self):
        """Test creating resolution error."""
        error = ResolutionError("Cannot resolve version")
        assert str(error) == "Cannot resolve version"
        assert error.code == "E_RESOLUTION"
        assert error.exit_code == 5


class TestInstallError:
    """Tests for InstallError."""

    def test_install_error_creation(self):
        """Test creating install error."""
        error = InstallError("Installation failed")
        assert str(error) == "Installation failed"
        assert error.code == "E_INSTALL"
        assert error.exit_code == 6

    def test_install_error_with_details(self):
        """Test install error with details."""
        error = InstallError(
            "Skill install failed",
            context={"skill": "test-skill", "version": "1.0.0"},
        )
        assert error.context["skill"] == "test-skill"


class TestSecurityError:
    """Tests for SecurityError."""

    def test_security_error_creation(self):
        """Test creating security error."""
        error = SecurityError("Checksum mismatch")
        assert str(error) == "Checksum mismatch"
        assert error.code == "E_SECURITY"
        assert error.exit_code == 7

    def test_security_error_for_signature(self):
        """Test security error for signature failure."""
        error = SecurityError(
            "Signature verification failed",
            context={"algorithm": "sha256"},
        )
        assert error.context["algorithm"] == "sha256"


class TestNotFoundError:
    """Tests for NotFoundError."""

    def test_not_found_error_creation(self):
        """Test creating not found error."""
        error = NotFoundError("Skill not found")
        assert str(error) == "Skill not found"
        assert error.code == "E_NOT_FOUND"
        assert error.exit_code == 8

    def test_not_found_error_with_resource(self):
        """Test not found error with resource info."""
        error = NotFoundError(
            "Skill 'test' not found",
            context={"skill": "test", "sources_searched": ["github", "gitee"]},
        )
        assert error.context["skill"] == "test"


class TestExceptionHierarchy:
    """Tests for exception inheritance hierarchy."""

    def test_all_errors_inherit_from_base(self):
        """Test that all errors inherit from SkillHubError."""
        errors = [
            ValidationError("test"),
            NetworkError("test"),
            AuthenticationError("test"),
            SourceError("test"),
            ResolutionError("test"),
            InstallError("test"),
            SecurityError("test"),
            NotFoundError("test"),
        ]
        for error in errors:
            assert isinstance(error, SkillHubError)
            assert isinstance(error, Exception)

    def test_error_codes_are_unique(self):
        """Test that error codes are unique."""
        codes = [
            SkillHubError.code,
            ValidationError.code,
            NetworkError.code,
            AuthenticationError.code,
            SourceError.code,
            ResolutionError.code,
            InstallError.code,
            SecurityError.code,
            NotFoundError.code,
        ]
        unique_codes = set(codes)
        assert len(unique_codes) == len(codes)

    def test_exit_codes_are_distinct(self):
        """Test that most exit codes are distinct."""
        errors = [
            NetworkError("test"),
            AuthenticationError("test"),
            SourceError("test"),
            ResolutionError("test"),
            InstallError("test"),
            SecurityError("test"),
            NotFoundError("test"),
        ]
        exit_codes = [e.exit_code for e in errors]
        unique_exit_codes = set(exit_codes)
        assert len(unique_exit_codes) == len(exit_codes)


class TestExceptionChaining:
    """Tests for exception chaining."""

    def test_exception_chain(self):
        """Test exception chain from inner to outer."""
        inner = ConnectionError("Inner connection issue")
        middle = NetworkError("Network problem", cause=inner)
        outer = InstallError("Install failed due to network", cause=middle)

        assert outer.cause == middle
        assert middle.cause == inner

    def test_exception_str_full_chain(self):
        """Test that str includes full chain."""
        inner = ValueError("root cause")
        outer = ValidationError("validation failed", cause=inner)
        error_str = str(outer)
        assert "validation failed" in error_str
        assert "root cause" in error_str


class TestExceptionRaising:
    """Tests for raising exceptions."""

    def test_raise_validation_error(self):
        """Test raising ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            raise ValidationError("Invalid skill name")
        assert exc_info.value.code == "E_VALIDATION"

    def test_raise_network_error(self):
        """Test raising NetworkError."""
        with pytest.raises(NetworkError) as exc_info:
            raise NetworkError("Connection timeout")
        assert exc_info.value.code == "E_NETWORK"

    def test_catch_as_base_error(self):
        """Test catching specific error as base error."""
        with pytest.raises(SkillHubError):
            raise NotFoundError("Not found")

    def test_try_catch_with_context(self):
        """Test try-catch with context preservation."""
        try:
            raise InstallError(
                "Install failed",
                context={"skill": "test", "path": "/skills/test"},
            )
        except SkillHubError as e:
            assert e.context["skill"] == "test"
            assert isinstance(e, InstallError)
