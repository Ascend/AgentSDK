(() => {
    const CATEGORY_ORDER = ['read', 'execute', 'write', 'orchestrate', 'llm', 'other', 'user', 'system'];
    const CATEGORY = {
        read: { label: '读取', color: '#8ae64f' },
        execute: { label: '执行', color: '#5c91ee' },
        write: { label: '写入', color: '#efb23c' },
        orchestrate: { label: '编排', color: '#ea4eaa' },
        llm: { label: 'LLM', color: '#9c78e8' },
        other: { label: '其他', color: '#77808d' },
        user: { label: '用户', color: '#4fc8d7' },
        system: { label: '系统', color: '#788694' },
    };
    const LABEL_WIDTH = 220;
    const TRACK_INSET = 10;
    const CONNECTOR_DOT_R = 3.5;
    const EVENT_GAP_PX = 3;
    const EVENT_TOP = 7;
    const EVENT_HEIGHT = 24;
    const EVENT_COMPACT_HIT_WIDTH = 3;
    const ZOOM_MIN = 1;
    const ZOOM_MAX = 40;

    const state = {
        data: null,
        zoom: 1,
        active: new Set(CATEGORY_ORDER),
        expanded: true,
        selectedId: null,
        eventMap: new Map(),
        reloadTimer: null,
        panning: null,
        lastFocus: null,
        lastFocusEventId: null,
        controlsBound: false,
        liveTailConnected: false,
        connectorLinks: new Map(),
        connectorHoverBound: false,
        hoveredConnectorIds: new Set(),
    };

    const el = (id) => document.getElementById(id);
    const escapeHtml = (value) => String(value ?? '')
        .replaceAll('&', '&amp;').replaceAll('<', '&lt;')
        .replaceAll('>', '&gt;').replaceAll('"', '&quot;').replaceAll("'", '&#39;');

    const safeCssColor = (value) => /^#[0-9a-fA-F]{3}([0-9a-fA-F]{3})?$/.test(String(value || '').trim())
        ? String(value || '').trim() : '';

    const clampPercent = (value) => {
        const number = Number(value);
        return Number.isFinite(number) ? Math.max(0, Math.min(100, number)) : 0;
    };

    function categoryOf(event) {
        if (event.type === 'user' || event.user_role === 'user') return 'user';
        if (event.type === 'system' || event.user_role === 'system') return 'system';
        if (event.type === 'llm_call') return 'llm';
        const value = event.category;
        if (value === 'llm_text') return 'llm';
        if (['read', 'execute', 'write', 'orchestrate'].includes(value)) return value;
        return 'other';
    }

    function formatClock(seconds) {
        if (!Number.isFinite(seconds) || seconds < 0) return '未记录';
        const rounded = Math.round(seconds), minutes = Math.floor(rounded / 60);
        return `${minutes}:${String(rounded % 60).padStart(2, '0')}`;
    }

    function formatDuration(ms, event) {
        if (event?.ts_unrecorded || event?.duration_unrecorded) return '未记录';
        const label = event?.duration_heuristic || event?.detail?.duration_source === 'estimated' ? '估算 ' : '';
        if (!Number.isFinite(ms)) return '未记录';
        if (ms < 1000) return `${label}${Math.round(ms)} ms`;
        if (ms < 60000) return `${label}${(ms / 1000).toFixed(ms < 10000 ? 2 : 1)} s`;
        return `${label}${Math.floor(ms / 60000)}m ${Math.round((ms % 60000) / 1000)}s`;
    }

    function formatAbsolute(seconds, explicit) {
        if (explicit) return explicit;
        if (!seconds) return '未记录';
        return new Intl.DateTimeFormat('zh-CN', {
            year: 'numeric', month: '2-digit', day: '2-digit',
            hour: '2-digit', minute: '2-digit', second: '2-digit', fractionalSecondDigits: 3,
        }).format(new Date(seconds * 1000));
    }

    function tokenLabel(value) {
        if (!value) return '—';
        return value >= 1000 ? `${Math.round(value / 1000)}K` : String(value);
    }

    function eventSummary(event) {
        if (event.type === 'user') return event.user_text || event.detail?.text || '用户消息';
        if (event.type === 'system') return event.system_text || event.detail?.text || event.label || '系统消息';
        if (event.type === 'llm_call') return event.detail?.text || event.detail?.text_preview || event.label || 'LLM 事件';
        if (event.type === 'tool_result') return event.detail?.tool_output || event.detail?.excerpt || event.label || '工具结果';
        const params = event.detail?.params;
        if (params && typeof params === 'object') {
            return params.file_path || params.path || params.command || params.description || event.label || '事件';
        }
        return event.label || '事件';
    }

    function bounds(data) {
        const realStarts = data.timeline.map((event) => event.start_time).filter((value) => value > 0),
            start = data.start_time || (realStarts.length ? Math.min(...realStarts) : 0);
        const ends = data.timeline.map((event) => Math.max(event.end_time || 0, event.start_time || 0)),
            end = Math.max(data.end_time || 0, ...(ends.length ? ends : [start + 1]));
        return { start, end: Math.max(end, start + 1), duration: Math.max(1, end - start) };
    }

    async function fetchSession({ preserveSelection = false } = {}) {
        const response = await fetch(`/api/viz/sessions/${encodeURIComponent(window.SESSION_ID)}`, { cache: 'no-store' });
        if (!response.ok) {
            let detail = `${response.status} ${response.statusText}`;
            try { detail = (await response.json()).detail || detail; } catch (_) { /* empty */ }
            const error = new Error(detail);
            error.status = response.status;
            throw error;
        }
        const incoming = await response.json();
        // Live tails grow monotonically: retain the last good render when a
        // transient parse/IO glitch returns an empty timeline, then retry normally.
        const incomingBars = (incoming.timeline || []).length;
        const currentBars = ((state.data || {}).timeline || []).length;
        if (incomingBars === 0 && currentBars > 0) {
            console.warn(
                'viz: ignoring empty session payload over non-empty state '
                + `(had ${currentBars} bars) — live-reload glitch`);
            return;
        }
        state.data = incoming;
        if (!preserveSelection || !state.data.timeline.some((event) => event.id === state.selectedId)) {
            closeDrawer();
        }
        render();
        refreshSelectedDrawer();
    }

    function refreshSelectedDrawer() {
        if (!state.selectedId) return; const selected = state.eventMap.get(state.selectedId);
        if (selected) renderDrawer(selected, { focus: false });
    }

    function render() {
        const data = state.data;
        el('timeline-shell').setAttribute('aria-busy', 'false');
        el('session-title').textContent = data.title || `Session ${data.session_id}`;
        const childNodes = realSubagentNodes(data);
        el('session-subtitle').textContent = [
            data.provider || 'provider 未记录',
            data.model || 'model 未记录',
            `${data.turn_count || 0} LLM`,
            `${data.tool_count || 0} 工具调用`,
            childNodes.length ? `${childNodes.length} 子 Agent` : '单 Agent',
            `上下文 ${tokenLabel(data.stats?.context_tokens)}`,
        ].join(' · ');

        if (data.parse_warnings?.length) {
            const warning = el('session-warning');
            warning.hidden = false;
            warning.innerHTML = `<strong>部分记录无法解析</strong><span>${escapeHtml(data.parse_warnings.join('；'))}</span>`;
        } else {
            el('session-warning').hidden = true;
        }
        renderLegend();
        renderTimeline();
    }

    function renderLegend() {
        const counts = Object.fromEntries(CATEGORY_ORDER.map((key) => [key, 0]));
        state.data.timeline.forEach((event) => { counts[categoryOf(event)] += 1; });
        el('category-legend').innerHTML = CATEGORY_ORDER.map((key) => `
            <button type="button" class="category-chip ${state.active.has(key) ? 'active' : ''}" data-category="${key}" aria-pressed="${state.active.has(key)}">
                <span style="--chip-color:${CATEGORY[key].color}"></span>${CATEGORY[key].label}<b>${counts[key]}</b>
            </button>`).join('');
        el('category-legend').querySelectorAll('[data-category]').forEach((button) => {
            button.addEventListener('click', () => {
                const key = button.dataset.category;
                if (state.active.has(key)) state.active.delete(key); else state.active.add(key);
                if (!state.active.size) state.active = new Set(CATEGORY_ORDER);
                const selected = state.eventMap.get(state.selectedId);
                if (selected && !state.active.has(categoryOf(selected))) closeDrawer();
                renderLegend();
                renderTimeline();
            });
        });
    }

    const realSubagentNodes = (data) => (data.agent_tree || [])
        .filter((node) => node && node.parent_id !== null && node.parent_id !== undefined);

    function renderTimeline() {
        const data = state.data;
        const timeline = data.timeline || [];
        const scroll = el('timeline-scroll');
        const status = el('timeline-state');
        const footer = el('timeline-footer');
        if (!timeline.length) {
            scroll.hidden = true;
            footer.hidden = true;
            status.hidden = false;
            const damaged = data.parse_warnings?.length;
            status.innerHTML = damaged
                ? '<strong>会话记录已损坏</strong><span>目录存在，但没有可展示的有效事件。解析警告已列在上方。</span>'
                : '<strong>空 Session</strong><span>这个 Session 还没有用户、LLM 或工具事件。</span>';
            return;
        }
        status.hidden = true;
        scroll.hidden = false;
        footer.hidden = false;

        const range = bounds(data);
        const viewport = Math.max(760, scroll.clientWidth || el('timeline-shell').clientWidth || 1000);
        const width = Math.max(viewport, Math.round(viewport * state.zoom));
        const cssLabelWidth = Number.parseFloat(getComputedStyle(document.documentElement).getPropertyValue('--lane-label-width')) || LABEL_WIDTH;
        const endColumnWidth = window.matchMedia('(max-width: 760px)').matches ? 70 : 92;
        const canvas = el('timeline-canvas');
        const canvasStyle = getComputedStyle(canvas);
        const canvasPaddingX = (Number.parseFloat(canvasStyle.paddingLeft) || 0) + (Number.parseFloat(canvasStyle.paddingRight) || 0);
        const renderRange = {
            ...range,
            trackPixels: Math.max(1, width - canvasPaddingX - cssLabelWidth - endColumnWidth),
        };
        renderRange.innerTrackPixels = Math.max(1, renderRange.trackPixels - TRACK_INSET * 2);
        canvas.style.width = `${width}px`;
        canvas.dataset.rangeStart = range.start;
        canvas.dataset.rangeDuration = range.duration;

        renderAxis(range);
        state.eventMap = new Map(timeline.map((event) => [event.id, event]));
        const nodes = realSubagentNodes(data);
        const childIds = new Set(nodes.map((node) => node.agent_id));
        const mainEvents = timeline.filter((event) => !event.agent_id || !childIds.has(event.agent_id));
        el('main-lane').innerHTML = laneHtml({
            id: 'main',
            name: data.model || data.agent_name || '主 Agent',
            badge: 'MAIN',
            events: mainEvents,
            contextTokens: data.stats?.context_tokens,
            range: renderRange,
            main: true,
        });

        const subSection = el('subagent-section');
        if (!nodes.length) {
            subSection.innerHTML = '';
        } else {
            const totalEvents = nodes.reduce((sum, node) => sum + timeline.filter((event) => event.agent_id === node.agent_id).length, 0);
            subSection.innerHTML = `
                <button type="button" class="subagent-toggle" id="subagent-toggle" aria-expanded="${state.expanded}">
                    <span class="subagent-toggle-label"><span class="subagent-toggle-arrow ${state.expanded ? 'expanded' : 'collapsed'}" aria-hidden="true"></span><span class="subagent-toggle-text">${formatClock(data.agent_layout_summary?.spawn_time || 0)} 派生 ${nodes.length} 子 Agent</span></span>
                    <small>${totalEvents} 个事件</small>
                </button>
                <div class="subagent-lanes" ${state.expanded ? '' : 'hidden'}>
                    ${nodes.map((node, index) => laneHtml({
                        id: node.agent_id,
                        name: node.name || `子 Agent ${index + 1}`,
                        badge: node.role || 'SUB',
                        badgeColor: node.role_color,
                        events: timeline.filter((event) => event.agent_id === node.agent_id),
                        contextTokens: node.stats?.context_tokens,
                        range: renderRange,
                        main: false,
                    })).join('')}
                </div>`;
            el('subagent-toggle').addEventListener('click', () => {
                state.expanded = !state.expanded;
                renderTimeline();
            });
        }
        footer.textContent = `主 Agent ${mainEvents.length} 个事件${nodes.length ? ` · 子 Agent ${nodes.length} 个 · 共 ${timeline.length} 个事件` : ''}`;
        bindEvents();
        requestAnimationFrame(() => renderConnectors(nodes, range));
    }

    function renderAxis(range) {
        const durationMinutes = range.duration / 60;
        let stepMinutes = 0.1;
        if (durationMinutes > 80) stepMinutes = 20;
        else if (durationMinutes > 40) stepMinutes = 10;
        else if (durationMinutes > 12) stepMinutes = 5;
        else if (durationMinutes > 6) stepMinutes = 2;
        else if (durationMinutes > 2) stepMinutes = 1;
        else if (durationMinutes > 0.8) stepMinutes = 0.5;
        else if (durationMinutes > 0.3) stepMinutes = 0.25;
        const ticks = [];
        for (let minute = 0; minute <= durationMinutes + stepMinutes * 0.01; minute += stepMinutes) {
            ticks.push(minute * 60);
        }
        el('time-axis').innerHTML = `<div class="axis-label">相对时间</div><div class="axis-track">${ticks.map((seconds) => `
            <span class="axis-tick ${seconds >= range.duration * 0.98 ? 'at-end' : ''}" style="left:${Math.min(100, seconds / range.duration * 100)}%"><i></i><b>${formatClock(seconds)}</b></span>`).join('')}</div>`;
    }

    function laneHtml(agent) {
        const visible = agent.events.filter((event) => state.active.has(categoryOf(event)));
        const layout = layoutLaneEvents(visible, agent.range);
        const stats = `${agent.events.filter((event) => event.type === 'tool_call').length} tools \u00b7 ${agent.events.filter((event) => event.type === 'llm_call').length} LLM`;
        return `<div class="lane-block ${agent.main ? 'main-agent' : 'sub-agent'}" data-agent-id="${escapeHtml(agent.id)}">
            <div class="agent-heading">
                <div class="agent-name"><span class="agent-badge" style="${safeCssColor(agent.badgeColor) ? `--badge-color:${safeCssColor(agent.badgeColor)}` : ''}">${escapeHtml(agent.badge)}</span><strong>${escapeHtml(agent.name)}</strong></div>
                <span>${stats}${agent.contextTokens ? ` \u00b7 ${tokenLabel(agent.contextTokens)}` : ''}</span>
            </div>
            <div class="lane-row">
                <div class="lane-label">${agent.main ? '\u4e3b Agent' : escapeHtml(agent.name)}</div>
                <div class="lane-track" data-lane-track="${escapeHtml(agent.id)}">
                    ${layout.items.map((item) => eventHtml(item.event, item.geometry)).join('')}
                    ${visible.length ? '' : '<span class="lane-empty">\u5f53\u524d\u7b5b\u9009\u4e0b\u65e0\u4e8b\u4ef6</span>'}
                </div>
                <div class="lane-end">${formatClock(agent.range.duration)}</div>
            </div>
        </div>`;
    }

    function eventGeometry(event, range) {
        const category = categoryOf(event);
        const rawStart = event.ts_unrecorded ? range.start : event.start_time;
        const leftPercent = clampPercent(((rawStart - range.start) / range.duration) * 100);
        const rawWidth = clampPercent(((event.duration_ms || 0) / 1000 / range.duration) * 100);
        const widthPercent = Math.max(0, Math.min(rawWidth, 100 - leftPercent));
        const innerPixels = Math.max(1, range.innerTrackPixels || range.trackPixels || 1);
        const leftPx = TRACK_INSET + (leftPercent / 100) * innerPixels;
        const maxWidthPx = Math.max(1, TRACK_INSET + innerPixels - leftPx);
        const widthPx = Math.min(maxWidthPx, (widthPercent / 100) * innerPixels);
        const hitWidthPx = Math.min(maxWidthPx, Math.max(widthPx, 8));
        return { category, leftPx, widthPx, hitWidthPx };
    }

    function layoutLaneEvents(events, range) {
        const innerPixels = Math.max(1, range.innerTrackPixels || range.trackPixels || 1);
        const leftBound = TRACK_INSET;
        const rightBound = TRACK_INSET + innerPixels;
        const items = events.map((event, index) => ({ event, geometry: eventGeometry(event, range), index }));
        if (!items.length) return { items };

        const requestedWidth = items.reduce((sum, item) => sum + item.geometry.hitWidthPx, 0);
        const requestedGap = EVENT_GAP_PX * Math.max(0, items.length - 1);
        const available = Math.max(1, rightBound - leftBound);
        if (requestedWidth + requestedGap > available) {
            const compactGap = items.length > 1 ? 1 : 0;
            const compactWidth = Math.max(1, (available - compactGap * (items.length - 1)) / items.length);
            items.forEach((item) => {
                item.geometry.hitWidthPx = Math.max(1, Math.min(item.geometry.hitWidthPx, Math.max(EVENT_COMPACT_HIT_WIDTH, compactWidth)));
            });
        }

        let cursor = leftBound - EVENT_GAP_PX;
        items.forEach((item) => {
            item.geometry.visualLeftPx = Math.max(item.geometry.leftPx, cursor + EVENT_GAP_PX);
            cursor = item.geometry.visualLeftPx + item.geometry.hitWidthPx;
        });

        const overflow = Math.max(0, cursor - rightBound);
        if (overflow) {
            const maxShiftLeft = Math.max(0, items[0].geometry.visualLeftPx - leftBound);
            const shift = Math.min(overflow, maxShiftLeft);
            items.forEach((item) => {
                item.geometry.visualLeftPx -= shift;
            });
        }

        return { items };
    }

    function eventHtml(event, geometry) {
        const label = event.label || event.type;
        const showLabel = geometry.widthPx >= 44 && geometry.hitWidthPx >= 44;
        return `<button type="button" class="timeline-event category-${geometry.category} ${showLabel ? 'has-label' : ''} ${event.status === 'error' ? 'is-error' : ''} ${state.selectedId === event.id ? 'selected' : ''}"
            style="left:${geometry.visualLeftPx ?? geometry.leftPx}px;width:${geometry.hitWidthPx}px;--event-actual-width:${Math.min(geometry.widthPx, geometry.hitWidthPx)}px;--event-color:${CATEGORY[geometry.category].color};--event-top:${EVENT_TOP}px;--event-height:${EVENT_HEIGHT}px"
            data-event-id="${escapeHtml(event.id)}" aria-label="${escapeHtml(label)}" title="${escapeHtml(label)}">${showLabel ? `<span class="timeline-event-label">${escapeHtml(label)}</span>` : ''}</button>`;
    }

    function bindEvents() {
        document.querySelectorAll('.timeline-event').forEach((button) => {
            const event = state.eventMap.get(button.dataset.eventId);
            button.addEventListener('mouseenter', (pointer) => {
                setLinkedConnectorsHighlighted(button, true);
                showTooltip(pointer, event);
            });
            button.addEventListener('mousemove', moveTooltip);
            button.addEventListener('mouseleave', () => {
                setLinkedConnectorsHighlighted(button, false);
                hideTooltip();
            });
            button.addEventListener('focus', () => setLinkedConnectorsHighlighted(button, true));
            button.addEventListener('blur', () => setLinkedConnectorsHighlighted(button, false));
            button.addEventListener('click', (pointer) => {
                pointer.stopPropagation();
                selectEvent(event);
            });
        });
    }

    function showTooltip(pointer, event) {
        const tooltip = el('event-tooltip');
        const category = categoryOf(event);
        tooltip.innerHTML = `<strong><span style="background:${CATEGORY[category].color}"></span>${escapeHtml(event.label)}</strong>
            <p>${escapeHtml(eventSummary(event)).slice(0, 260)}</p>
            <small>${formatClock(event.start_time - bounds(state.data).start)} · ${formatDuration(event.duration_ms, event)}</small>`;
        tooltip.hidden = false;
        moveTooltip(pointer);
    }

    function moveTooltip(pointer) {
        const tooltip = el('event-tooltip');
        if (tooltip.hidden) return;
        const x = Math.min(window.innerWidth - tooltip.offsetWidth - 12, pointer.clientX + 14);
        const y = Math.min(window.innerHeight - tooltip.offsetHeight - 12, pointer.clientY + 14);
        tooltip.style.transform = `translate(${Math.max(8, x)}px, ${Math.max(8, y)}px)`;
    }

    function hideTooltip() { el('event-tooltip').hidden = true; }

    const connectorEventIds = (connectorId) => state.connectorLinks.get(connectorId) || [];

    function setConnectorHighlighted(connectorId, highlighted) {
        if (!connectorId) return;
        const escaped = CSS.escape(connectorId);
        document.querySelectorAll(`[data-connector-id="${escaped}"]`).forEach((node) => {
            node.classList.toggle('connector-highlighted', highlighted);
        });
        connectorEventIds(connectorId).forEach((eventId) => {
            document.querySelector(`[data-event-id="${CSS.escape(eventId)}"]`)?.classList.toggle('connector-highlighted', highlighted);
        });
    }

    function setLinkedConnectorsHighlighted(eventButton, highlighted) {
        const connectorIds = (eventButton?.dataset.agentConnectorIds || '').split(/\s+/).filter(Boolean);
        connectorIds.forEach((connectorId) => setConnectorHighlighted(connectorId, highlighted));
    }

    function setHoveredConnectors(connectorIds) {
        const next = new Set(connectorIds.filter(Boolean));
        const current = state.hoveredConnectorIds || new Set();
        if (next.size === current.size && [...next].every((connectorId) => current.has(connectorId))) return;
        current.forEach((connectorId) => setConnectorHighlighted(connectorId, false));
        next.forEach((connectorId) => setConnectorHighlighted(connectorId, true));
        state.hoveredConnectorIds = next;
    }

    function connectorIdsForTarget(target) {
        const eventButton = target?.closest?.('.timeline-event');
        if (eventButton) return (eventButton.dataset.agentConnectorIds || '').split(/\s+/).filter(Boolean);
        const connectorNode = target?.closest?.('[data-connector-id]');
        return connectorNode?.dataset.connectorId ? [connectorNode.dataset.connectorId] : [];
    }

    function bindConnectorHoverDelegation() {
        if (state.connectorHoverBound) return;
        const canvas = el('timeline-canvas');
        if (!canvas) return;
        state.connectorHoverBound = true;
        canvas.addEventListener('pointermove', (pointer) => setHoveredConnectors(connectorIdsForTarget(pointer.target)));
        canvas.addEventListener('pointerleave', () => setHoveredConnectors([]));
    }

    function appendConnectorId(eventId, connectorId) {
        const button = document.querySelector(`[data-event-id="${CSS.escape(eventId)}"]`);
        if (!button) return;
        const ids = new Set((button.dataset.agentConnectorIds || '').split(/\s+/).filter(Boolean));
        ids.add(connectorId);
        button.dataset.agentConnectorIds = [...ids].join(' ');
    }

    function bindConnectorInteractions() {
        document.querySelectorAll('[data-connector-id]').forEach((node) => {
            const connectorId = node.dataset.connectorId;
            node.addEventListener('pointerenter', () => setConnectorHighlighted(connectorId, true));
            node.addEventListener('pointerleave', () => setConnectorHighlighted(connectorId, false));
            node.addEventListener('mouseenter', () => setConnectorHighlighted(connectorId, true));
            node.addEventListener('mouseleave', () => setConnectorHighlighted(connectorId, false));
        });
    }

    function selectEvent(event) {
        state.lastFocus = document.activeElement;
        state.lastFocusEventId = event.id;
        state.selectedId = event.id;
        document.querySelectorAll('.timeline-event').forEach((button) => button.classList.toggle('selected', button.dataset.eventId === event.id));
        renderDrawer(event);
    }

    function renderDrawer(event, { focus = true } = {}) {
        const category = categoryOf(event);
        el('drawer-dot').style.background = CATEGORY[category].color;
        el('drawer-title').textContent = event.label || '事件详情';
        const range = bounds(state.data);
        const detail = event.detail || {};
        const sections = [];
        const add = (title, value, open = false) => {
            if (value === undefined || value === null || value === '') return;
            let text;
            if (typeof value === 'string') text = value;
            else { try { text = JSON.stringify(value, null, 2); } catch (_) { text = String(value); } }
            const long = text.length > 280 || text.includes('\n');
            sections.push(`<section class="drawer-section"><div class="drawer-section-title"><h3>${escapeHtml(title)}</h3><button type="button" class="copy-button" data-copy-index="${drawerCopies.push(text) - 1}">复制</button></div>
                ${long ? `<details ${open ? 'open' : ''}><summary>${open ? '收起' : '展开完整内容'} · ${text.length} 字符</summary><pre>${escapeHtml(text)}</pre></details>` : `<pre>${escapeHtml(text)}</pre>`}</section>`);
        };
        drawerCopies = [];
        const durationSource = detail.duration_source === 'recorded' ? '真实记录'
            : detail.duration_source === 'record-gap' ? 'tool-use/result 时间差'
                : event.duration_heuristic ? '估算' : '未记录';
        const overview = `
            <dl class="drawer-overview">
                <div><dt>类别</dt><dd>${CATEGORY[category].label}</dd></div>
                <div><dt>相对时间</dt><dd>${event.ts_unrecorded ? '未记录' : formatClock(event.start_time - range.start)}</dd></div>
                <div><dt>绝对时间</dt><dd>${event.ts_unrecorded ? '未记录' : escapeHtml(formatAbsolute(event.start_time, event.absolute_time || detail.absolute_time))}</dd></div>
                <div><dt>持续时间</dt><dd>${formatDuration(event.duration_ms, event)}</dd></div>
                <div><dt>时长来源</dt><dd>${durationSource}</dd></div>
                <div><dt>状态</dt><dd>${escapeHtml(event.status)}</dd></div>
            </dl>`;
        add('摘要', eventSummary(event));
        if (event.type === 'llm_call') {
            add('LLM 输入', event.input_text || detail.input_text, true);
            add('LLM 输出', detail.text || detail.text_preview, true);
            add('Token', { input: event.token_in, output: event.token_out, usage: detail.usage });
            add('Model', event.model);
        } else if (event.type === 'tool_call') {
            add('工具输入', detail.tool_input ?? detail.params, true);
            add('工具输出', detail.tool_output, true);
            add('Tool Use ID', detail.tool_use_id);
        } else if (event.type === 'tool_result') {
            add('工具结果', detail.tool_output ?? detail.excerpt, true);
            add('Tool Use ID', detail.tool_use_id);
        } else {
            add(event.type === 'user' ? '用户消息' : '系统消息', event.user_text || event.system_text || detail.text, true);
        }
        el('drawer-body').innerHTML = overview + sections.join('');
        el('drawer-body').querySelectorAll('.copy-button').forEach((button) => button.addEventListener('click', async () => {
            try {
                const result = await copyText(drawerCopies[Number(button.dataset.copyIndex)] || '');
                const original = button.textContent;
                button.textContent = result === 'selected' ? '已选中' : '已复制';
                window.setTimeout(() => { button.textContent = original; }, 1200);
            } catch (_) { button.textContent = '复制不可用'; }
        }));
        const drawer = el('event-drawer');
        drawer.classList.add('open');
        drawer.setAttribute('aria-hidden', 'false');
        if (focus) drawer.focus({ preventScroll: true });
    }

    let drawerCopies = [];
    async function copyText(text) {
        if (navigator.clipboard?.writeText) {
            try {
                await navigator.clipboard.writeText(text);
                return 'copied';
            } catch (_) { /* use the selection-based fallback */ }
        }
        document.querySelectorAll('.copy-fallback-area').forEach((node) => node.remove());
        const area = document.createElement('textarea');
        area.value = text;
        area.setAttribute('readonly', '');
        area.className = 'copy-fallback-area';
        document.body.appendChild(area);
        area.focus();
        area.select();
        try {
            if (document.execCommand('copy')) {
                area.remove();
                return 'copied';
            }
        } catch (_) { /* leave selected text available to the user */ }
        return 'selected';
    }
    function closeDrawer() {
        state.selectedId = null;
        document.querySelectorAll('.timeline-event.selected').forEach((item) => item.classList.remove('selected'));
        const drawer = el('event-drawer');
        if (drawer) {
            drawer.classList.remove('open');
            drawer.setAttribute('aria-hidden', 'true');
        }
        document.querySelectorAll('.copy-fallback-area').forEach((node) => node.remove());
        const focusTarget = state.lastFocusEventId
            ? document.querySelector(`[data-event-id="${CSS.escape(state.lastFocusEventId)}"]`)
            : state.lastFocus;
        if (focusTarget?.isConnected && typeof focusTarget.focus === 'function') {
            focusTarget.focus({ preventScroll: true });
        }
        state.lastFocus = null;
        state.lastFocusEventId = null;
    }

    function renderConnectors(nodes, range) {
        const svg = el('agent-connectors');
        if (!nodes.length || !state.expanded) {
            svg.innerHTML = '';
            svg.setAttribute('width', '0');
            svg.setAttribute('height', '0');
            return;
        }
        const canvas = el('timeline-canvas');
        const mainTrack = canvas.querySelector('[data-lane-track="main"]');
        if (!mainTrack) return;
        const canvasRect = canvas.getBoundingClientRect();
        const mainRect = mainTrack.getBoundingClientRect();
        const trackLeft = mainRect.left - canvasRect.left;
        const width = mainRect.width;
        const height = canvas.scrollHeight;
        svg.style.left = `${trackLeft}px`;
        svg.setAttribute('width', String(width));
        svg.setAttribute('height', String(height));
        svg.setAttribute('viewBox', `0 0 ${width} ${height}`);
        const mainY = mainRect.top - canvasRect.top + mainRect.height / 2;
        const allEvents = state.data.timeline || [];
        const mainEvents = allEvents.filter((event) => !event.agent_id);
        const clampX = (value, inset = CONNECTOR_DOT_R) => {
            if (!Number.isFinite(value)) return inset;
            return Math.max(inset, Math.min(width - inset, value));
        };
        const nearestMainEvent = (seconds) => {
            if (!mainEvents.length) return null;
            const target = range.start + seconds;
            return mainEvents.reduce((best, event) => {
                const bestDistance = Math.abs((best.start_time || range.start) - target);
                const distance = Math.abs((event.start_time || range.start) - target);
                return distance < bestDistance ? event : best;
            }, mainEvents[0]);
        };
        const firstEvent = (events) => events.reduce((best, event) => {
            if (!best) return event;
            return (event.start_time || 0) < (best.start_time || 0) ? event : best;
        }, null);
        const lastEvent = (events) => events.reduce((best, event) => {
            if (!best) return event;
            return (event.end_time || event.start_time || 0) > (best.end_time || best.start_time || 0) ? event : best;
        }, null);
        const bezier = (fromX, fromY, toX, toY) => {
            const distance = Math.abs(toX - fromX);
            const control = Math.max(22, distance * 0.45);
            const direction = toX >= fromX ? 1 : -1;
            const c1 = clampX(fromX + control * direction, 0);
            const c2 = clampX(toX - control * direction, 0);
            return `M ${fromX} ${fromY} C ${c1} ${fromY}, ${c2} ${toY}, ${toX} ${toY}`;
        };
        const connectorPath = (connectorId, kind, d, eventIds) => {
            state.connectorLinks.set(connectorId, eventIds.filter(Boolean));
            eventIds.filter(Boolean).forEach((eventId) => appendConnectorId(eventId, connectorId));
            const label = kind === 'spawn' ? 'Sub-agent started' : 'Sub-agent returned';
            return `<path class="agent-connector ${kind}-connector" data-connector-id="${escapeHtml(connectorId)}" d="${d}"/>`
                + `<path class="agent-connector-hit ${kind}-connector-hit" data-connector-id="${escapeHtml(connectorId)}" aria-label="${label}" d="${d}"/>`;
        };
        state.connectorLinks = new Map();
        setHoveredConnectors([]);
        document.querySelectorAll('.timeline-event[data-agent-connector-ids]').forEach((button) => {
            delete button.dataset.agentConnectorIds;
            button.classList.remove('connector-highlighted');
        });
        const paths = [];
        nodes.forEach((node) => {
            const track = canvas.querySelector(`[data-lane-track="${CSS.escape(node.agent_id)}"]`);
            if (!track) return;
            const rect = track.getBoundingClientRect();
            const subY = rect.top - canvasRect.top + rect.height / 2;
            const events = allEvents.filter((event) => event.agent_id === node.agent_id);
            const spawnSeconds = Number.isFinite(node.spawn_x) ? node.spawn_x : 0;
            const joinSeconds = Number.isFinite(node.join_x) ? node.join_x : spawnSeconds;
            const first = events.length ? Math.min(...events.map((event) => event.start_time)) : range.start + spawnSeconds;
            const last = events.length ? Math.max(...events.map((event) => event.end_time || event.start_time)) : range.start + joinSeconds;
            const spawnEvent = node.metadata?.spawn_bar_id ? state.eventMap.get(node.metadata.spawn_bar_id) : nearestMainEvent(spawnSeconds);
            const joinEvent = nearestMainEvent(joinSeconds);
            const subFirstEvent = firstEvent(events);
            const subLastEvent = lastEvent(events);
            const eventCenterPoint = (eventId, fallbackX, fallbackY) => {
                if (!eventId) return { x: fallbackX, y: fallbackY };
                const eventButton = canvas.querySelector(`[data-event-id="${CSS.escape(eventId)}"]`);
                if (!eventButton) return { x: fallbackX, y: fallbackY };
                const eventRect = eventButton.getBoundingClientRect();
                return {
                    x: clampX(eventRect.left - mainRect.left + eventRect.width / 2),
                    y: eventRect.top - canvasRect.top + eventRect.height / 2,
                };
            };
            const spawnPoint = eventCenterPoint(spawnEvent?.id, clampX(spawnSeconds / range.duration * width), mainY);
            const joinPoint = eventCenterPoint(joinEvent?.id, clampX(joinSeconds / range.duration * width), mainY);
            const subStartPoint = eventCenterPoint(subFirstEvent?.id, events.length ? clampX((first - range.start) / range.duration * width) : spawnPoint.x, subY);
            const subEndPoint = eventCenterPoint(subLastEvent?.id, events.length ? clampX((last - range.start) / range.duration * width) : joinPoint.x, subY);
            const spawnConnectorId = `spawn-${node.agent_id}`;
            const mergeConnectorId = `merge-${node.agent_id}`;
            paths.push(connectorPath(spawnConnectorId, 'spawn', bezier(spawnPoint.x, spawnPoint.y, subStartPoint.x, subStartPoint.y), [spawnEvent?.id, subFirstEvent?.id]));
            paths.push(connectorPath(mergeConnectorId, 'merge', bezier(subEndPoint.x, subEndPoint.y, joinPoint.x, joinPoint.y), [subLastEvent?.id, joinEvent?.id]));
            paths.push(`<circle class="connector-dot" data-connector-id="${escapeHtml(spawnConnectorId)}" cx="${spawnPoint.x}" cy="${spawnPoint.y}" r="${CONNECTOR_DOT_R}"/>`);
            paths.push(`<circle class="connector-dot" data-connector-id="${escapeHtml(mergeConnectorId)}" cx="${joinPoint.x}" cy="${joinPoint.y}" r="${CONNECTOR_DOT_R}"/>`);
        });
        svg.innerHTML = paths.join('');
        bindConnectorInteractions();
        bindConnectorHoverDelegation();
    }

    function setZoom(next, anchorX = null) {
        const scroll = el('timeline-scroll');
        const oldWidth = el('timeline-canvas').offsetWidth || 1;
        const viewportX = anchorX ?? scroll.clientWidth / 2;
        const ratio = (scroll.scrollLeft + viewportX) / oldWidth;
        state.zoom = Math.max(ZOOM_MIN, Math.min(ZOOM_MAX, next));
        el('zoom-value').value = `${Math.round(state.zoom * 100)}%`;
        el('zoom-value').textContent = `${Math.round(state.zoom * 100)}%`;
        el('zoom-reset').hidden = Math.abs(state.zoom - 1) < 0.001;
        renderTimeline();
        requestAnimationFrame(() => {
            scroll.scrollLeft = Math.max(0, ratio * el('timeline-canvas').offsetWidth - viewportX);
        });
    }

    function bindControls() {
        if (state.controlsBound) return;
        state.controlsBound = true;
        el('zoom-in').addEventListener('click', () => setZoom(state.zoom * 1.25));
        el('zoom-out').addEventListener('click', () => setZoom(state.zoom / 1.25));
        el('zoom-reset').addEventListener('click', () => setZoom(1));
        el('drawer-close').addEventListener('click', closeDrawer);
        document.addEventListener('keydown', (event) => { if (event.key === 'Escape') closeDrawer(); });

        const scroll = el('timeline-scroll');
        scroll.addEventListener('wheel', (event) => {
            if (event.altKey) {
                event.preventDefault();
                scroll.scrollLeft += event.deltaY || event.deltaX;
                return;
            }
            event.preventDefault();
            const rect = scroll.getBoundingClientRect();
            const factor = Math.exp(-event.deltaY * 0.0012);
            setZoom(state.zoom * factor, event.clientX - rect.left);
        }, { passive: false });
        scroll.addEventListener('pointerdown', (event) => {
            if (event.button !== 0 || event.target.closest('button,a')) return;
            state.panning = { id: event.pointerId, x: event.clientX, left: scroll.scrollLeft, moved: false };
            scroll.classList.add('panning');
            scroll.setPointerCapture(event.pointerId);
        });
        scroll.addEventListener('pointermove', (event) => {
            if (!state.panning || state.panning.id !== event.pointerId) return;
            const delta = event.clientX - state.panning.x;
            if (Math.abs(delta) > 3) state.panning.moved = true;
            if (state.panning.moved) scroll.scrollLeft = state.panning.left - delta;
        });
        const stopPan = (event) => {
            if (!state.panning || state.panning.id !== event.pointerId) return;
            state.panning = null;
            scroll.classList.remove('panning');
        };
        scroll.addEventListener('pointerup', stopPan);
        scroll.addEventListener('pointercancel', stopPan);
        window.addEventListener('resize', () => { if (state.data) renderTimeline(); });
    }

    function scheduleReload() {
        window.clearTimeout(state.reloadTimer);
        state.reloadTimer = window.setTimeout(() => fetchSession({ preserveSelection: true }).catch(() => {}), 180);
    }

    function disconnectLiveTail() {
        window.clearTimeout(state.reloadTimer);
        if (window.vizWs && state.liveTailConnected) {
            window.vizWs.disconnect();
            state.liveTailConnected = false;
        }
    }

    async function initialize() {
        bindControls();
        try {
            await fetchSession();
        } catch (error) {
            el('timeline-shell').setAttribute('aria-busy', 'false');
            el('timeline-scroll').hidden = true;
            el('timeline-footer').hidden = true;
            const status = el('timeline-state');
            status.hidden = false;
            status.innerHTML = error.status === 404
                ? '<strong>Session not found</strong><span>Check the URL or session ID.</span><a class="btn" href="/">Back to sessions</a>'
                : `<strong>Load failed</strong><span>${escapeHtml(error.message)}</span><button class="btn" id="retry-session">Retry</button>`;
            el('retry-session')?.addEventListener('click', initialize);
            return;
        }
        if (window.vizWs && !state.liveTailConnected) {
            window.vizWs.on('transcript', scheduleReload);
            window.connectLiveTail(window.SESSION_ID);
            state.liveTailConnected = true;
        }
    }

    window.addEventListener('pagehide', disconnectLiveTail);
    window.addEventListener('beforeunload', disconnectLiveTail);
    document.addEventListener('DOMContentLoaded', initialize);
})();
