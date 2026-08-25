(() => {
    const VALID_STATUSES = new Set(['all', 'running', 'completed', 'success', 'failed', 'error', 'unknown', 'aborted', 'cancelled']);
    const VALID_LAYOUTS = new Set(['grid', 'list']);

    const readChoice = (key, allowed, fallback) => {
        try {
            const value = localStorage.getItem(key);
            return allowed.has(value) ? value : fallback;
        } catch (_) {
            return fallback;
        }
    };

    const writeChoice = (key, value) => {
        try { localStorage.setItem(key, value); } catch (_) { /* storage may be blocked */ }
    };

    const state = {
        workspace: 'default',
        workspaces: [],
        sessions: [],
        query: '',
        status: readChoice('viz.statusFilter', VALID_STATUSES, 'all'),
        layout: readChoice('viz.layoutMode', VALID_LAYOUTS, 'grid'),
        timer: null,
        sessionRequestSeq: 0,
    };

    const escapeHtml = (value) => String(value ?? '')
        .replaceAll('&', '&amp;').replaceAll('<', '&lt;')
        .replaceAll('>', '&gt;').replaceAll('"', '&quot;').replaceAll("'", '&#39;');

    const safeClassPart = (value) => String(value || 'unknown').replace(/[^a-zA-Z0-9_-]/g, '_') || 'unknown';

    const formatTime = (seconds) => {
        if (!seconds) return 'Not recorded';
        return new Intl.DateTimeFormat('zh-CN', {
            month: '2-digit', day: '2-digit', hour: '2-digit', minute: '2-digit', second: '2-digit',
        }).format(new Date(seconds * 1000));
    };

    const formatDuration = (ms) => {
        if (!ms) return 'Not recorded';
        const seconds = Math.round(ms / 1000);
        if (seconds < 60) return `${seconds}s`;
        const minutes = Math.floor(seconds / 60);
        return `${minutes}m ${seconds % 60}s`;
    };

    async function requestJson(url) {
        const response = await fetch(url, { cache: 'no-store' });
        if (!response.ok) throw new Error(`${response.status} ${response.statusText}`);
        return response.json();
    }

    async function loadWorkspaces() {
        state.workspaces = await requestJson('/api/viz/workspaces');
        if (!state.workspaces.some((item) => item.id === state.workspace)) {
            state.workspace = state.workspaces[0]?.id || 'default';
        }
        renderWorkspaces();
    }

    function sessionsUrl() {
        const params = new URLSearchParams();
        const query = state.query.trim();
        if (query) params.set('q', query);
        if (state.status !== 'all') params.set('status', state.status);
        const suffix = params.toString();
        return `/api/viz/workspaces/${encodeURIComponent(state.workspace)}/sessions${suffix ? `?${suffix}` : ''}`;
    }

    async function loadSessions({ quiet = false } = {}) {
        const grid = document.getElementById('sessions-grid');
        const requestSeq = ++state.sessionRequestSeq;
        if (!quiet) grid.innerHTML = '<div class="loading">Loading sessions...</div>';
        try {
            const sessions = await requestJson(sessionsUrl());
            if (requestSeq !== state.sessionRequestSeq) return;
            state.sessions = sessions;
            state.sessions.sort((a, b) => (b.end_time || b.start_time || 0) - (a.end_time || a.start_time || 0));
            renderSessions();
            setLiveState('live');
        } catch (error) {
            if (requestSeq !== state.sessionRequestSeq) return;
            if (!quiet) grid.innerHTML = `<div class="empty-state error-state"><strong>Load failed</strong><span>${escapeHtml(error.message)}</span></div>`;
            setLiveState('error');
        }
    }

    function renderWorkspaces() {
        const tabs = document.getElementById('workspace-tabs');
        tabs.innerHTML = state.workspaces.map((item) => `
            <button type="button" class="workspace-tab ${item.id === state.workspace ? 'active' : ''}" data-workspace="${escapeHtml(item.id)}">
                ${escapeHtml(item.name)} <span>${item.session_count ?? 0}</span>
            </button>`).join('');
        tabs.querySelectorAll('[data-workspace]').forEach((button) => button.addEventListener('click', () => {
            state.workspace = button.dataset.workspace;
            renderWorkspaces();
            loadSessions();
        }));
    }

    function filteredSessions() {
        const needle = state.query.trim().toLowerCase();
        return state.sessions.filter((session) => {
            if (state.status !== 'all' && session.status !== state.status) return false;
            if (!needle) return true;
            return [session.session_id, session.title, session.model, session.provider, session.workspace, session.agent_name]
                .some((value) => String(value || '').toLowerCase().includes(needle));
        });
    }

    function sessionCard(session) {
        const warning = session.parse_warnings?.length
            ? `<span class="warning-chip" title="${escapeHtml(session.parse_warnings.join('\n'))}">Warnings ${session.parse_warnings.length}</span>` : '';
        const href = `/session/${encodeURIComponent(session.session_id)}`;
        return `<a class="session-card" href="${href}" aria-label="Open session ${escapeHtml(session.title || session.session_id)}">
            <div class="session-card-top">
                <span class="status-badge status-${safeClassPart(session.status)}">${escapeHtml(session.status)}</span>${warning}
                <time>${formatTime(session.end_time || session.start_time)}</time>
            </div>
            <h3>${escapeHtml(session.title || session.session_id)}</h3>
            <code>${escapeHtml(session.session_id)}</code>
            <div class="session-card-meta">
                <span>${escapeHtml(session.provider || 'provider not recorded')}</span>
                <span>${escapeHtml(session.model || 'model not recorded')}</span>
            </div>
            <div class="session-card-stats">
                <span>${session.turn_count || 0} LLM</span><span>${session.tool_count || 0} tools</span><span>${formatDuration(session.duration_ms)}</span>
            </div>
        </a>`;
    }

    function sessionRow(session) {
        const href = `/session/${encodeURIComponent(session.session_id)}`;
        return `<a class="session-row" href="${href}" aria-label="Open session ${escapeHtml(session.title || session.session_id)}">
            <span class="cell">${formatTime(session.end_time || session.start_time)}</span>
            <span class="cell"><span class="status-badge status-${safeClassPart(session.status)}">${escapeHtml(session.status)}</span></span>
            <span class="cell cell-title">${escapeHtml(session.title || session.session_id)}</span>
            <span class="cell">${escapeHtml(session.provider || '-')}</span>
            <span class="cell">${escapeHtml(session.model || '-')}</span>
            <span class="cell">${formatDuration(session.duration_ms)}</span>
            <span class="cell">${session.turn_count || 0}</span>
            <span class="cell">${session.tool_count || 0}</span>
        </a>`;
    }

    function renderSessions() {
        const sessions = filteredSessions();
        const grid = document.getElementById('sessions-grid');
        const total = state.sessions.length;
        document.getElementById('session-count').textContent = sessions.length === total
            ? `${sessions.length} sessions`
            : `${sessions.length} / ${total} sessions`;
        grid.className = state.layout === 'list' ? 'sessions-list' : 'sessions-grid';
        if (!sessions.length) {
            const hasFilters = Boolean(state.query.trim()) || state.status !== 'all';
            grid.innerHTML = hasFilters
                ? '<div class="empty-state"><strong>No matching sessions</strong><span>Adjust search or status filters and try again.</span></div>'
                : '<div class="empty-state"><strong>No sessions yet</strong><span>Start a ClawCodex session and local records will appear here automatically.</span></div>';
            return;
        }
        if (state.layout === 'list') {
            grid.innerHTML = `<div class="list-header">
                <span class="cell">Updated</span><span class="cell">Status</span><span class="cell">Title</span>
                <span class="cell">Provider</span><span class="cell">Model</span><span class="cell">Duration</span>
                <span class="cell">LLM</span><span class="cell">Tools</span>
            </div>${sessions.map(sessionRow).join('')}`;
        } else {
            grid.innerHTML = sessions.map(sessionCard).join('');
        }
    }

    function setLayout(layout) {
        state.layout = layout;
        writeChoice('viz.layoutMode', layout);
        const gridButton = document.getElementById('layout-toggle-grid');
        const listButton = document.getElementById('layout-toggle-list');
        gridButton.classList.toggle('active', layout === 'grid');
        listButton.classList.toggle('active', layout === 'list');
        gridButton.setAttribute('aria-pressed', String(layout === 'grid'));
        listButton.setAttribute('aria-pressed', String(layout === 'list'));
        renderSessions();
    }

    function setLiveState(kind) {
        const indicator = document.getElementById('live-indicator');
        indicator.className = `live-indicator ${kind}`;
    }

    function stopPolling() {
        if (state.timer) {
            window.clearInterval(state.timer);
            state.timer = null;
        }
    }

    async function initialize() {
        document.getElementById('status-filter-select').value = state.status;
        document.getElementById('search-input').addEventListener('input', (event) => {
            state.query = event.target.value; loadSessions({ quiet: true });
        });
        document.getElementById('status-filter-select').addEventListener('change', (event) => {
            state.status = event.target.value;
            writeChoice('viz.statusFilter', state.status);
            loadSessions({ quiet: true });
        });
        document.getElementById('layout-toggle-grid').addEventListener('click', () => setLayout('grid'));
        document.getElementById('layout-toggle-list').addEventListener('click', () => setLayout('list'));
        document.getElementById('refresh-button').addEventListener('click', async () => { await loadWorkspaces(); await loadSessions(); });
        setLayout(state.layout);
        try { await loadWorkspaces(); await loadSessions(); }
        catch (error) {
            document.getElementById('sessions-grid').innerHTML = `<div class="empty-state error-state"><strong>Cannot scan sessions</strong><span>${escapeHtml(error.message)}</span></div>`;
            setLiveState('error');
        }
        state.timer = window.setInterval(() => {
            if (!document.hidden) { loadWorkspaces().then(() => loadSessions({ quiet: true })).catch(() => setLiveState('error')); }
        }, 5000);
    }

    window.addEventListener('pagehide', stopPolling);
    window.addEventListener('beforeunload', stopPolling);
    document.addEventListener('DOMContentLoaded', initialize);
})();
