#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# -------------------------------------------------------------------------
# This file is part of the AgentSDK project.
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
# Copyright (c) 2026 Clawd Codex Team
#
# AgentSDK is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#
#          http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# -------------------------------------------------------------------------

"""Tests for P92-A document extractor.

Covers:
    - ``extract_search_document``: single Skill → SkillSearchDocument
    - ``extract_batch``: batch extraction with MCP/template tagging
    - ``SkillSearchDocument.text()`` and ``field_text()``
    - ``SkillSearchDocument.make_id()`` stability and uniqueness
    - Source type inference from ``loaded_from``
    - Hidden skill exclusion
    - Description priority (``when_to_use`` > ``description``)
    - Body priority (``markdown_content`` > ``content``)
    - Tag derivation from ``allowed_tools`` + namespace + source
"""
# pylint: disable=no-name-in-module

from __future__ import annotations

from clawcodex_ext.skills.model import Skill
from clawcodex_ext.services.skill_search.document import (
    SkillSearchDocument,
    extract_batch,
    extract_search_document,
)


class TestMakeId:
    def test_stability(self):
        id1 = SkillSearchDocument.make_id("local", "git_helper")
        id2 = SkillSearchDocument.make_id("local", "git_helper")
        assert id1 == id2

    def test_different_source(self):
        id1 = SkillSearchDocument.make_id("local", "deploy")
        id2 = SkillSearchDocument.make_id("project", "deploy")
        assert id1 != id2

    def test_different_name(self):
        id1 = SkillSearchDocument.make_id("local", "foo")
        id2 = SkillSearchDocument.make_id("local", "bar")
        assert id1 != id2

    def test_format(self):
        doc_id = SkillSearchDocument.make_id("local", "test")
        assert len(doc_id) == 16
        assert all(c in "0123456789abcdef" for c in doc_id)


class TestTextAndFieldText:
    def test_text_concatenation(self):
        doc = SkillSearchDocument(
            id="test",
            name="foo",
            title="Foo Title",
            description="does stuff",
            body="## Body",
            source="local",
        )
        text = doc.text()
        assert "foo" in text
        assert "Foo Title" in text
        assert "does stuff" in text
        assert "## Body" in text

    def test_text_skips_duplicate_title(self):
        doc = SkillSearchDocument(
            id="test",
            name="foo",
            title="foo",
            description="desc",
            body="body",
            source="local",
        )
        text = doc.text()
        assert text.count("foo") == 1

    def test_text_with_tags(self):
        doc = SkillSearchDocument(
            id="test",
            name="foo",
            title="Foo",
            description="d",
            body="b",
            source="local",
            tags=("bash", "python"),
        )
        text = doc.text()
        assert "bash python" in text

    def test_field_text_keys(self):
        doc = SkillSearchDocument(
            id="test",
            name="foo",
            title="Foo",
            description="desc",
            body="body",
            source="local",
        )
        fields = doc.field_text()
        assert fields["name"] == "foo"
        assert fields["title"] == "Foo"
        assert fields["description"] == "desc"
        assert fields["body"] == "body"
        assert fields["tags"] == ""


def _make_skill(
    name: str = "test_skill",
    description: str = "A test skill",
    content: str = "",
    markdown_content: str = "",
    loaded_from: str = "user",
    source: str = "userSettings",
    display_name: str | None = None,
    when_to_use: str | None = None,
    allowed_tools: list[str] | None = None,
    is_hidden: bool = False,
) -> Skill:
    return Skill(
        name=name,
        description=description,
        content=content,
        markdown_content=markdown_content,
        loaded_from=loaded_from,
        source=source,
        display_name=display_name,
        when_to_use=when_to_use,
        allowed_tools=allowed_tools or [],
        is_hidden=is_hidden,
    )


class TestExtractSearchDocument:
    def test_basic(self):
        skill = _make_skill(name="git_helper")
        doc = extract_search_document(skill)
        assert doc is not None
        assert doc.name == "git_helper"
        assert doc.title == "git_helper"
        assert doc.description == "A test skill"
        assert doc.source == "local"
        assert doc.weight == 1.1

    def test_display_name_as_title(self):
        skill = _make_skill(name="gh", display_name="Git Helper")
        doc = extract_search_document(skill)
        assert doc is not None
        assert doc.title == "Git Helper"

    def test_source_local(self):
        skill = _make_skill(loaded_from="user")
        doc = extract_search_document(skill)
        assert doc is not None
        assert doc.source == "local"
        assert doc.weight == 1.1

    def test_source_project(self):
        skill = _make_skill(loaded_from="project")
        doc = extract_search_document(skill)
        assert doc is not None
        assert doc.source == "project"
        assert doc.weight == 1.3

    def test_source_managed(self):
        skill = _make_skill(loaded_from="managed")
        doc = extract_search_document(skill)
        assert doc is not None
        assert doc.source == "local"

    def test_source_plugin(self):
        skill = _make_skill(loaded_from="plugin")
        doc = extract_search_document(skill)
        assert doc is not None
        assert doc.source == "local"

    def test_source_unknown_fallback(self):
        skill = _make_skill(loaded_from="unknown")
        doc = extract_search_document(skill)
        assert doc is not None
        assert doc.source == "local"

    def test_source_explicit_mcp(self):
        skill = _make_skill(loaded_from="user")
        doc = extract_search_document(skill, source_type="mcp")
        assert doc is not None
        assert doc.source == "mcp"
        assert doc.weight == 0.9

    def test_source_explicit_template(self):
        skill = _make_skill(loaded_from="user")
        doc = extract_search_document(skill, source_type="template")
        assert doc is not None
        assert doc.source == "template"
        assert doc.weight == 1.0

    def test_hidden_skill_excluded(self):
        skill = _make_skill(is_hidden=True)
        doc = extract_search_document(skill)
        assert doc is None

    def test_description_when_to_use_priority(self):
        skill = _make_skill(
            description="Generic desc",
            when_to_use="Use when committing code",
        )
        doc = extract_search_document(skill)
        assert doc is not None
        assert doc.description == "Use when committing code"

    def test_description_fallback(self):
        skill = _make_skill(description="Generic desc")
        doc = extract_search_document(skill)
        assert doc is not None
        assert doc.description == "Generic desc"

    def test_description_empty_fallback(self):
        skill = _make_skill(description="")
        doc = extract_search_document(skill)
        assert doc is not None
        assert doc.description == ""

    def test_body_markdown_content_priority(self):
        skill = _make_skill(
            content="raw content",
            markdown_content="## Markdown Body",
        )
        doc = extract_search_document(skill)
        assert doc is not None
        assert doc.body == "## Markdown Body"

    def test_body_content_fallback(self):
        skill = _make_skill(
            content="raw content",
            markdown_content="",
        )
        doc = extract_search_document(skill)
        assert doc is not None
        assert doc.body == "raw content"

    def test_body_empty(self):
        skill = _make_skill(content="")
        doc = extract_search_document(skill)
        assert doc is not None
        assert doc.body == ""

    def test_tags_from_allowed_tools(self):
        skill = _make_skill(allowed_tools=["Bash", "Read", "python"])
        doc = extract_search_document(skill)
        assert doc is not None
        assert "bash" in doc.tags
        assert "read" in doc.tags
        assert "python" in doc.tags

    def test_tags_from_namespace(self):
        skill = _make_skill(name="browser:playwright")
        doc = extract_search_document(skill)
        assert doc is not None
        assert "browser" in doc.tags
        assert "playwright" in doc.tags

    def test_tags_from_source_field(self):
        skill = _make_skill(source="userSettings")
        doc = extract_search_document(skill)
        assert doc is not None
        assert "usersettings" in doc.tags

    def test_tags_deduplication(self):
        skill = _make_skill(
            name="bash:bash",
            allowed_tools=["bash"],
            source="userSettings",
        )
        doc = extract_search_document(skill)
        assert doc is not None
        assert doc.tags.count("bash") == 1

    def test_tags_no_duplicates_from_namespace_and_tools(self):
        skill = _make_skill(
            name="bash",
            allowed_tools=["bash"],
        )
        doc = extract_search_document(skill)
        assert doc is not None
        assert doc.tags.count("bash") == 1

    def test_id_stability_across_calls(self):
        skill = _make_skill(name="git_helper")
        doc1 = extract_search_document(skill)
        doc2 = extract_search_document(skill)
        assert doc1 is not None and doc2 is not None
        assert doc1.id == doc2.id

    def test_custom_source_weights(self):
        skill = _make_skill(loaded_from="user")
        doc = extract_search_document(
            skill,
            source_weights={"local": 2.0, "project": 1.5},
        )
        assert doc is not None
        assert doc.weight == 2.0


class TestExtractBatch:
    def test_empty(self):
        docs = extract_batch([])
        assert docs == []

    def test_basic(self):
        skills = [
            _make_skill(name="a", loaded_from="user"),
            _make_skill(name="b", loaded_from="project"),
        ]
        docs = extract_batch(skills)
        assert len(docs) == 2
        assert {d.name for d in docs} == {"a", "b"}

    def test_mcp_tagging(self):
        skills = [
            _make_skill(name="browser_navigate", loaded_from="user"),
            _make_skill(name="git_helper", loaded_from="user"),
        ]
        docs = extract_batch(skills, mcp_skill_names={"browser_navigate"})
        mcp_doc = next(d for d in docs if d.name == "browser_navigate")
        local_doc = next(d for d in docs if d.name == "git_helper")
        assert mcp_doc.source == "mcp"
        assert mcp_doc.weight == 0.9
        assert local_doc.source == "local"
        assert local_doc.weight == 1.1

    def test_template_tagging(self):
        skills = [
            _make_skill(name="deploy_template", loaded_from="user"),
        ]
        docs = extract_batch(skills, template_skill_names={"deploy_template"})
        assert len(docs) == 1
        assert docs[0].source == "template"
        assert docs[0].weight == 1.0

    def test_mcp_overrides_loaded_from(self):
        skills = [_make_skill(name="foo", loaded_from="project")]
        docs = extract_batch(skills, mcp_skill_names={"foo"})
        assert docs[0].source == "mcp"

    def test_hidden_skills_skipped(self):
        skills = [
            _make_skill(name="visible", loaded_from="user"),
            _make_skill(name="hidden", loaded_from="user", is_hidden=True),
        ]
        docs = extract_batch(skills)
        assert len(docs) == 1
        assert docs[0].name == "visible"

    def test_combined_sources(self):
        skills = [
            _make_skill(name="local_a", loaded_from="user"),
            _make_skill(name="project_b", loaded_from="project"),
            _make_skill(name="mcp_c", loaded_from="user"),
            _make_skill(name="tpl_d", loaded_from="user"),
        ]
        docs = extract_batch(
            skills,
            mcp_skill_names={"mcp_c"},
            template_skill_names={"tpl_d"},
        )
        sources = {d.name: d.source for d in docs}
        assert sources["local_a"] == "local"
        assert sources["project_b"] == "project"
        assert sources["mcp_c"] == "mcp"
        assert sources["tpl_d"] == "template"
