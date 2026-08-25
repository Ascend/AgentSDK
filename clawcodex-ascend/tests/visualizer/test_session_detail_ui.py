#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# -------------------------------------------------------------------------
# This file is part of the AgentSDK project.
# Copyright (c) 2026 Clawd Codex Team
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
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


"""Session-detail timeline interaction and live-refresh contracts."""

from __future__ import annotations

import pytest


@pytest.fixture
def session_js(client) -> str:
    response = client.get("/static/js/session_view.js")
    assert response.status_code == 200
    return response.text


@pytest.fixture
def session_css(client) -> str:
    response = client.get("/static/css/session_view.css")
    assert response.status_code == 200
    return response.text


class TestSessionDetailUI:
    def test_event_summary_has_a_safe_fallback(self, session_js):
        assert "return event.label || '事件';" in session_js

    def test_session_view_drawer_uses_css_transition(self, session_js):
        assert "drawer.classList.add('open')" in session_js
        assert "drawer.classList.remove('open')" in session_js
        assert "style.setProperty('transition'" not in session_js
        assert "style.setProperty('left'" not in session_js

    def test_session_view_refreshes_selected_drawer_after_live_reload(self, session_js):
        fetch_start = session_js.index("async function fetchSession")
        render_start = session_js.index("render();", fetch_start)
        refresh_start = session_js.index("refreshSelectedDrawer();", render_start)
        helper_start = session_js.index("function refreshSelectedDrawer()")
        assert fetch_start < render_start < refresh_start < helper_start
        assert "const selected = state.eventMap.get(state.selectedId);" in session_js
        assert "if (selected) renderDrawer(selected, { focus: false });" in session_js
        assert "function renderDrawer(event, { focus = true } = {})" in session_js
        assert "if (focus) drawer.focus({ preventScroll: true });" in session_js

    def test_session_view_closes_drawer_when_filter_hides_selected_event(self, session_js):
        assert "const selected = state.eventMap.get(state.selectedId);" in session_js
        assert "if (selected && !state.active.has(categoryOf(selected))) closeDrawer();" in session_js
        toggle_start = session_js.index("const selected = state.eventMap.get(state.selectedId);")
        render_start = session_js.index("renderLegend();", toggle_start)
        assert toggle_start < render_start

    def test_session_view_sanitizes_badge_colors(self, session_js):
        assert "safeCssColor(agent.badgeColor)" in session_js
        assert "--badge-color:${escapeHtml(agent.badgeColor)}" not in session_js

    def test_session_view_clamps_timeline_event_width(self, session_js):
        assert "const leftPercent = clampPercent" in session_js
        assert "const innerPixels = Math.max(1, range.innerTrackPixels || range.trackPixels || 1)" in session_js
        assert "const maxWidthPx = Math.max(1, TRACK_INSET + innerPixels - leftPx)" in session_js
        assert "const hitWidthPx = Math.min(maxWidthPx, Math.max(widthPx, 8))" in session_js
        assert "width:${geometry.hitWidthPx}px" in session_js
        assert "--event-actual-width:${Math.min(geometry.widthPx, geometry.hitWidthPx)}px" in session_js
        assert "width:${rawWidth}%" not in session_js

    def test_session_view_keeps_dense_lane_events_inline(self, session_js, session_css):
        assert "function layoutLaneEvents(events, range)" in session_js
        assert "visualLeftPx" in session_js
        assert "cursor + EVENT_GAP_PX" in session_js
        assert "EVENT_COMPACT_HIT_WIDTH" in session_js
        assert "--event-top:${EVENT_TOP}px" in session_js
        assert "is-stacked" not in session_js
        assert "min-height: 46px" in session_css
        assert "height: 38px" in session_css
        assert "min-width: 0" in session_css
        assert ".timeline-event.is-stacked" not in session_css

    def test_session_view_shows_synthesized_subagents(self, session_js):
        assert "node.parent_id !== null && node.parent_id !== undefined" in session_js
        assert "node.metadata?.transcript_path" not in session_js
        assert "eventCenterPoint(subFirstEvent?.id" in session_js
        assert "eventCenterPoint(subLastEvent?.id" in session_js

    def test_session_view_highlights_agent_connectors_with_events(self, session_js, session_css):
        assert "connectorLinks: new Map()" in session_js
        assert "node.metadata?.spawn_bar_id" in session_js
        assert "appendConnectorId(eventId, connectorId)" in session_js
        assert "setLinkedConnectorsHighlighted(button, true)" in session_js
        assert "function bindConnectorHoverDelegation()" in session_js
        assert "canvas.addEventListener('pointermove'" in session_js
        assert "setHoveredConnectors(connectorIdsForTarget(pointer.target))" in session_js
        assert 'data-connector-id="${escapeHtml(connectorId)}"' in session_js
        assert "agent-connector-hit" in session_js
        assert 'tabindex="0" role="button"' not in session_js
        assert ".timeline-event.connector-highlighted" in session_css
        assert ".agent-connectors { position: absolute; top: 0; z-index: 2;" in session_css
        assert ".agent-connectors .agent-connector-hit" in session_css
        assert "outline: none" in session_css
        assert ".agent-connectors .connector-highlighted.agent-connector" in session_css

    def test_session_view_disconnects_live_tail_on_page_exit(self, session_js):
        assert "function disconnectLiveTail()" in session_js
        assert "window.clearTimeout(state.reloadTimer);" in session_js
        assert "window.vizWs.disconnect();" in session_js
        assert "state.liveTailConnected = false;" in session_js
        assert "window.addEventListener('pagehide', disconnectLiveTail);" in session_js
        assert "window.addEventListener('beforeunload', disconnectLiveTail);" in session_js

    def test_session_view_initialize_is_idempotent(self, session_js):
        assert "controlsBound: false" in session_js
        assert "liveTailConnected: false" in session_js
        assert "if (state.controlsBound) return;" in session_js
        assert "state.controlsBound = true;" in session_js
        assert "if (window.vizWs && !state.liveTailConnected)" in session_js
        assert "state.liveTailConnected = true;" in session_js
        catch_start = session_js.index("} catch (error) {")
        live_start = session_js.index("if (window.vizWs && !state.liveTailConnected)")
        assert "return;" in session_js[catch_start:live_start]

    def test_session_view_uses_clickable_hit_targets_for_short_events(self, session_css):
        assert "height: var(--event-height, 24px); min-width: 0; max-width: 100%;" in session_css
        assert "background: transparent" in session_css
        assert ".timeline-event::before" in session_css
        assert "width: max(var(--event-actual-width), 2px)" in session_css
        assert ".timeline-event.is-error::before" in session_css
        assert "min-width: 3px" not in session_css
        assert ".timeline-event.category-user, .timeline-event.category-system" not in session_css
        assert ".lane-track::before" not in session_css

    def test_session_view_labels_wide_timeline_events(self, session_js, session_css):
        assert "trackPixels" in session_js
        assert "const showLabel = geometry.widthPx >= 44" in session_js
        assert "timeline-event-label" in session_js
        assert 'title="${escapeHtml(label)}"' in session_js
        assert ".timeline-event.has-label" in session_css
        assert ".timeline-event-label" in session_css
