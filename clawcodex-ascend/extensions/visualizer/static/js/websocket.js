/** Session WebSocket change notifications. */
class VizWebSocket {
    constructor() {
        this.ws = null;
        this.sessionId = null;
        this.reconnectTimer = null;
        this.handlers = {};
        this.generation = 0;
    }

    connect(sessionId) {
        const generation = ++this.generation;
        if (this.ws) {
            this.ws.onclose = null;
            this.ws.close();
        }
        window.clearTimeout(this.reconnectTimer);
        this.reconnectTimer = null;
        this.sessionId = sessionId;
        const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
        this.ws = new WebSocket(`${protocol}//${location.host}/api/viz/ws/sessions/${encodeURIComponent(sessionId)}`);
        this.ws.onopen = () => this._updateStatus('connected');
        this.ws.onmessage = (event) => {
            try {
                const data = JSON.parse(event.data);
                if (data.type === 'heartbeat') return;
                if (data.type === 'transcript_event') this._emit('transcript', data);
            } catch (error) {
                console.warn('Session WebSocket message parse error:', error);
            }
        };
        this.ws.onclose = () => {
            if (this.generation !== generation) return;
            this._updateStatus('idle');
            this.ws = null;
            window.clearTimeout(this.reconnectTimer);
            if (this.sessionId) {
                this.reconnectTimer = window.setTimeout(() => this.connect(this.sessionId), 3000);
            }
        };
        this.ws.onerror = () => this._updateStatus('idle');
    }

    disconnect() {
        this.generation += 1;
        this.sessionId = null;
        window.clearTimeout(this.reconnectTimer);
        if (this.ws) {
            this.ws.onclose = null;
            this.ws.close();
        }
        this.ws = null;
    }

    on(event, handler) {
        if (!this.handlers[event]) this.handlers[event] = [];
        if (!this.handlers[event].includes(handler)) this.handlers[event].push(handler);
    }

    _emit(event, data) {
        (this.handlers[event] || []).forEach((handler) => handler(data));
    }

    _updateStatus(status) {
        const statusElement = document.getElementById('ws-status');
        if (!statusElement) return;
        statusElement.textContent = status === 'connected' ? 'Live connected' : 'Live idle';
        statusElement.className = status === 'connected' ? 'connected' : 'idle';
    }
}

const vizWs = new VizWebSocket();

function connectLiveTail(sessionId) {
    vizWs.connect(sessionId);
}

window.connectLiveTail = connectLiveTail;
window.vizWs = vizWs;
