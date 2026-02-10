/**
 * Laneswarm Dashboard — WebSocket client + rendering
 *
 * Connects to ws://host:port/ws, receives snapshot + event stream,
 * and renders a live task grid, activity feed, and orchestrator status.
 */

// ── State ──────────────────────────────────────────────────
let ws = null;
let reconnectDelay = 1000;
const MAX_RECONNECT_DELAY = 30000;
let state = {
    tasks: [],
    progress: { total: 0, pending: 0, in_progress: 0, completed: 0, failed: 0, blocked: 0 },
    events: [],
    costs: { total_tokens: 0, total_wall_ms: 0 },
};
let feedPaused = false;
let currentFilter = 'all';
const MAX_FEED_ITEMS = 200;

// ── WebSocket ──────────────────────────────────────────────
function connect() {
    const proto = location.protocol === 'https:' ? 'wss' : 'ws';
    const url = `${proto}://${location.host}/ws`;
    setConnectionStatus('connecting');

    ws = new WebSocket(url);

    ws.onopen = () => {
        setConnectionStatus('connected');
        reconnectDelay = 1000;
    };

    ws.onmessage = (event) => {
        try {
            const msg = JSON.parse(event.data);
            handleMessage(msg);
        } catch (e) {
            console.error('Failed to parse WS message:', e);
        }
    };

    ws.onclose = () => {
        setConnectionStatus('disconnected');
        setTimeout(() => {
            reconnectDelay = Math.min(reconnectDelay * 1.5, MAX_RECONNECT_DELAY);
            connect();
        }, reconnectDelay);
    };

    ws.onerror = () => {
        ws.close();
    };
}

function handleMessage(msg) {
    switch (msg.type) {
        case 'snapshot':
            state = msg.data;
            renderAll();
            break;
        case 'event':
            state.events.unshift(msg.data);
            if (state.events.length > MAX_FEED_ITEMS) {
                state.events = state.events.slice(0, MAX_FEED_ITEMS);
            }
            updateFromEvent(msg.data);
            renderFeedItem(msg.data);
            renderProgress();
            renderOrchestrator();
            break;
        case 'task_detail':
            renderTaskDetailModal(msg.data);
            break;
    }
}

function updateFromEvent(event) {
    // Update task state from events
    const taskId = event.task_id;
    if (!taskId) return;

    const task = state.tasks.find(t => t.task_id === taskId);
    if (!task) return;

    // Track previous status to correctly adjust counters
    const prevStatus = task.status;

    switch (event.event_type) {
        case 'task_started':
            task.status = 'in_progress';
            task.current_phase = 'coding';
            state.progress.in_progress = (state.progress.in_progress || 0) + 1;
            if (prevStatus === 'pending') state.progress.pending = Math.max(0, (state.progress.pending || 0) - 1);
            renderTaskCard(task);
            break;
        case 'task_completed':
            task.status = 'completed';
            task.current_phase = 'completed';
            if (event.data.tokens) task.tokens_used += event.data.tokens;
            state.progress.completed = (state.progress.completed || 0) + 1;
            if (prevStatus === 'in_progress') state.progress.in_progress = Math.max(0, (state.progress.in_progress || 0) - 1);
            renderTaskCard(task);
            break;
        case 'task_failed':
            task.status = 'failed';
            task.error_message = event.data.error || 'Unknown error';
            state.progress.failed = (state.progress.failed || 0) + 1;
            if (prevStatus === 'in_progress') state.progress.in_progress = Math.max(0, (state.progress.in_progress || 0) - 1);
            renderTaskCard(task);
            break;
        case 'task_retrying':
            task.status = 'in_progress';
            task.retries = event.data.retry || task.retries;
            renderTaskCard(task);
            break;
        case 'agent_working':
            if (event.data.action) task.current_phase = event.data.action;
            renderTaskCard(task);
            break;
        case 'review_accepted':
            task.current_phase = 'promoting';
            renderTaskCard(task);
            break;
        case 'review_rejected':
            task.current_phase = 'reviewing';
            renderTaskCard(task);
            break;
        case 'progress_update':
            // Full progress update from orchestrator
            if (event.data.total !== undefined) {
                state.progress = event.data;
            }
            break;
        case 'cost_update':
            if (event.data.total_tokens !== undefined) {
                state.costs.total_tokens = event.data.total_tokens;
            }
            break;
    }
}

// ── Rendering ──────────────────────────────────────────────

function recalcProgress() {
    // Recount from actual task statuses — authoritative source of truth
    const p = { total: state.tasks.length, pending: 0, in_progress: 0, completed: 0, failed: 0, blocked: 0 };
    for (const t of state.tasks) {
        if (t.status === 'pending') p.pending++;
        else if (t.status === 'in_progress') p.in_progress++;
        else if (t.status === 'completed') p.completed++;
        else if (t.status === 'failed') p.failed++;
        else if (t.status === 'blocked') p.blocked++;
    }
    state.progress = p;
}

function renderAll() {
    recalcProgress();
    renderTaskGrid();
    renderActivityFeed();
    renderProgress();
    renderOrchestrator();
    renderDepGraph();
}

// ── Connection Status ──────────────────────────────────────

function setConnectionStatus(status) {
    const dot = document.getElementById('connection-status');
    dot.className = `status-dot ${status}`;
    dot.title = status.charAt(0).toUpperCase() + status.slice(1);
}

// ── Progress Bar ───────────────────────────────────────────

function renderProgress() {
    const { total, completed, failed } = state.progress;
    const done = (completed || 0) + (failed || 0);
    const pct = total > 0 ? (done / total) * 100 : 0;

    document.getElementById('progress-bar').style.width = `${pct}%`;
    document.getElementById('progress-text').textContent = `${done} / ${total} tasks`;

    // Update cost stats
    document.getElementById('total-tokens').textContent = formatNumber(state.costs.total_tokens);
    document.getElementById('total-time').textContent = formatTime(state.costs.total_wall_ms);
}

// ── Task Grid ──────────────────────────────────────────────

function renderTaskGrid() {
    const grid = document.getElementById('task-grid');

    if (state.tasks.length === 0) {
        grid.innerHTML = `
            <div class="empty-state" style="grid-column: 1 / -1;">
                <div class="empty-icon">&#9881;</div>
                <p>No tasks yet. Run <code>laneswarm serve --run</code> to start.</p>
            </div>
        `;
        return;
    }

    grid.innerHTML = '';
    for (const task of state.tasks) {
        grid.appendChild(createTaskCard(task));
    }
}

function createTaskCard(task) {
    const card = document.createElement('div');
    card.className = `task-card status-${task.status}`;
    card.id = `task-${task.task_id}`;
    card.onclick = () => requestTaskDetail(task.task_id);

    const phases = ['coding', 'reviewing', 'promoting'];
    const currentIdx = phases.indexOf(task.current_phase);

    card.innerHTML = `
        <div class="task-card-header">
            <span class="task-title">${escHtml(task.title)}</span>
            <span class="task-badge badge-${task.status}">${task.status.replace('_', ' ')}</span>
        </div>
        <div class="task-phase">
            ${phases.map((p, i) => {
                let cls = '';
                if (task.status === 'completed') cls = 'done';
                else if (task.status === 'failed') cls = i <= currentIdx ? 'failed' : '';
                else if (i < currentIdx) cls = 'done';
                else if (i === currentIdx && task.status === 'in_progress') cls = 'active';
                return `<span class="phase-dot ${cls}" title="${p}"></span>`;
            }).join('')}
            <span class="phase-label">${task.current_phase || 'waiting'}</span>
        </div>
        <div class="task-meta">
            <span title="Tokens">${formatNumber(task.tokens_used)} tok</span>
            <span title="Files">${(task.files_written || []).length} files</span>
            <span title="Time">${formatTime(task.wall_time_ms)}</span>
        </div>
    `;
    return card;
}

function renderTaskCard(task) {
    const existing = document.getElementById(`task-${task.task_id}`);
    if (existing) {
        const newCard = createTaskCard(task);
        existing.replaceWith(newCard);
    }
}

// ── Activity Feed ──────────────────────────────────────────

function renderActivityFeed() {
    const feed = document.getElementById('activity-feed');
    feed.innerHTML = '';
    const filtered = filterEvents(state.events);
    for (const event of filtered.slice(0, MAX_FEED_ITEMS)) {
        feed.appendChild(createFeedItem(event));
    }
}

function renderFeedItem(event) {
    if (!matchesFilter(event)) return;

    const feed = document.getElementById('activity-feed');
    const item = createFeedItem(event);
    feed.insertBefore(item, feed.firstChild);

    // Trim old items
    while (feed.children.length > MAX_FEED_ITEMS) {
        feed.removeChild(feed.lastChild);
    }

    // Auto-scroll to top (newest first)
    if (!feedPaused) {
        feed.scrollTop = 0;
    }
}

function createFeedItem(event) {
    const item = document.createElement('div');
    item.className = 'feed-item';

    const time = new Date(event.timestamp * 1000);
    const timeStr = time.toLocaleTimeString('en-US', { hour12: false, hour: '2-digit', minute: '2-digit', second: '2-digit' });
    const icon = getEventIcon(event.event_type);
    const msg = formatEventMessage(event);

    item.innerHTML = `
        <span class="feed-time">${timeStr}</span>
        <span class="feed-icon">${icon}</span>
        <span class="feed-msg">${msg}</span>
    `;
    return item;
}

// Pause auto-scroll on hover
document.addEventListener('DOMContentLoaded', () => {
    const feed = document.getElementById('activity-feed');
    feed.addEventListener('mouseenter', () => { feedPaused = true; });
    feed.addEventListener('mouseleave', () => { feedPaused = false; });

    document.getElementById('event-filter').addEventListener('change', (e) => {
        currentFilter = e.target.value;
        renderActivityFeed();
    });

    // Close modal on backdrop click
    const modal = document.getElementById('task-modal');
    modal.querySelector('.modal-backdrop').addEventListener('click', closeModal);
});

function filterEvents(events) {
    if (currentFilter === 'all') return events;
    return events.filter(e => matchesFilter(e));
}

function matchesFilter(event) {
    if (currentFilter === 'all') return true;
    return event.event_type.startsWith(currentFilter);
}

function getEventIcon(eventType) {
    const icons = {
        task_queued: '\u23F3',       // hourglass
        task_started: '\u25B6',      // play
        task_completed: '\u2705',    // green check
        task_failed: '\u274C',       // red x
        task_retrying: '\uD83D\uDD04',  // cycle
        agent_spawned: '\uD83E\uDD16',  // robot
        agent_working: '\u2699',     // gear
        agent_finished: '\uD83C\uDFC1', // flag
        review_started: '\uD83D\uDD0D', // magnifier
        review_accepted: '\uD83D\uDC4D', // thumbs up
        review_rejected: '\uD83D\uDC4E', // thumbs down
        promote_started: '\u2B06',   // up arrow
        promote_completed: '\uD83C\uDF89', // party
        promote_conflict: '\u26A0',  // warning
        run_started: '\uD83D\uDE80', // rocket
        run_completed: '\uD83C\uDFC6', // trophy
        progress_update: '\uD83D\uDCCA', // chart
        verification_passed: '\u2705',
        verification_failed: '\u26A0',
        cost_update: '\uD83D\uDCB0', // money
        budget_warning: '\u26A0',
        plan_created: '\uD83D\uDCCB', // clipboard
    };
    return icons[eventType] || '\u2022';
}

function formatEventMessage(event) {
    const taskRef = event.task_id
        ? `<span class="task-ref" onclick="requestTaskDetail('${escAttr(event.task_id)}')">${escHtml(event.task_id)}</span>`
        : '';

    switch (event.event_type) {
        case 'task_started':
            return `Task ${taskRef} started${event.data.model ? ` (${escHtml(event.data.model)})` : ''}`;
        case 'task_completed':
            return `Task ${taskRef} completed${event.data.tokens ? ` (${formatNumber(event.data.tokens)} tok)` : ''}`;
        case 'task_failed':
            return `Task ${taskRef} failed: ${escHtml((event.data.error || '').slice(0, 80))}`;
        case 'task_retrying':
            return `Task ${taskRef} retrying (attempt ${event.data.retry || '?'})`;
        case 'agent_spawned':
            return `Agent ${escHtml(event.agent_id || '')} spawned for ${taskRef}`;
        case 'agent_working':
            return `Agent working on ${taskRef}: ${escHtml(event.data.action || '')}`;
        case 'agent_finished':
            return `Agent finished ${taskRef} (${formatNumber(event.data.tokens_in || 0)}+${formatNumber(event.data.tokens_out || 0)} tok)`;
        case 'review_accepted':
            return `Review accepted for ${taskRef}`;
        case 'review_rejected':
            return `Review rejected for ${taskRef}`;
        case 'promote_completed':
            return `Task ${taskRef} promoted to main`;
        case 'promote_conflict':
            return `Promote conflict for ${taskRef}`;
        case 'run_started':
            return `Run started (${event.data.total_tasks || '?'} tasks)`;
        case 'run_completed':
            return `Run completed: ${event.data.completed || 0}/${event.data.total || 0} tasks, ${formatNumber(event.data.total_tokens || 0)} tokens`;
        case 'progress_update':
            return `Progress: ${event.data.completed || 0}/${event.data.total || 0} done`;
        default:
            return `${event.event_type} ${taskRef}`;
    }
}

// ── Orchestrator Section ───────────────────────────────────

function renderOrchestrator() {
    const p = state.progress;
    document.getElementById('orch-pending').textContent = p.pending || 0;
    document.getElementById('orch-running').textContent = p.in_progress || 0;
    document.getElementById('orch-completed').textContent = p.completed || 0;
    document.getElementById('orch-failed').textContent = p.failed || 0;
}

function renderDepGraph() {
    const container = document.getElementById('dep-graph');
    container.innerHTML = '';

    if (state.tasks.length === 0) return;

    // Simple linear visualization of tasks in order
    for (let i = 0; i < state.tasks.length; i++) {
        const task = state.tasks[i];
        if (i > 0) {
            const arrow = document.createElement('span');
            arrow.className = 'dep-arrow';
            arrow.textContent = '\u203A';
            container.appendChild(arrow);
        }

        const node = document.createElement('div');
        node.className = `dep-node ${task.status}`;
        node.textContent = task.task_id.length > 3 ? task.task_id.slice(0, 3) : task.task_id;
        node.title = `${task.title} (${task.status})`;
        node.onclick = () => requestTaskDetail(task.task_id);
        container.appendChild(node);
    }
}

// ── Task Detail Modal ──────────────────────────────────────

function requestTaskDetail(taskId) {
    if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({ type: 'get_task_detail', task_id: taskId }));
    }
}

function renderTaskDetailModal(detail) {
    const modal = document.getElementById('task-modal');
    const body = document.getElementById('modal-body');

    const badge = `<span class="task-badge badge-${detail.status}">${detail.status.replace('_', ' ')}</span>`;

    let html = `
        <div class="detail-header">
            <h3>${escHtml(detail.title)} ${badge}</h3>
            <div class="detail-id">${escHtml(detail.task_id)} \u2022 Lane: ${escHtml(detail.lane_name || 'N/A')}</div>
        </div>

        <div class="detail-section">
            <h4>Overview</h4>
            <div class="detail-grid">
                <div class="detail-item"><span class="label">Phase:</span> <span class="value">${escHtml(detail.current_phase || 'waiting')}</span></div>
                <div class="detail-item"><span class="label">Complexity:</span> <span class="value">${escHtml(detail.estimated_complexity)}</span></div>
                <div class="detail-item"><span class="label">Tokens:</span> <span class="value">${formatNumber(detail.tokens_used)}</span></div>
                <div class="detail-item"><span class="label">Time:</span> <span class="value">${formatTime(detail.wall_time_ms)}</span></div>
                <div class="detail-item"><span class="label">Retries:</span> <span class="value">${detail.retries} / ${detail.max_retries}</span></div>
                <div class="detail-item"><span class="label">Dependencies:</span> <span class="value">${detail.dependencies.length > 0 ? escHtml(detail.dependencies.join(', ')) : 'None'}</span></div>
            </div>
        </div>

        <div class="detail-section">
            <h4>Description</h4>
            <p style="font-size: 12px; color: var(--text-secondary); line-height: 1.5;">${escHtml(detail.description || 'No description')}</p>
        </div>
    `;

    // Agent Steps Timeline
    if (detail.agent_steps && detail.agent_steps.length > 0) {
        html += `<div class="detail-section"><h4>Agent Steps</h4><div class="timeline">`;
        for (const step of detail.agent_steps) {
            const t = new Date(step.timestamp * 1000);
            const timeStr = t.toLocaleTimeString('en-US', { hour12: false, hour: '2-digit', minute: '2-digit', second: '2-digit' });
            html += `
                <div class="timeline-step">
                    <span class="step-phase">${escHtml(step.phase)}</span>
                    <span class="step-time">${timeStr}</span>
                    <div class="step-summary">${escHtml(step.summary || '')}</div>
                </div>
            `;
        }
        html += `</div></div>`;
    }

    // Files Written
    if (detail.files_written && detail.files_written.length > 0) {
        html += `<div class="detail-section"><h4>Files Written</h4><ul class="file-list">`;
        for (const f of detail.files_written) {
            html += `<li>${escHtml(f)}</li>`;
        }
        html += `</ul></div>`;
    }

    // Verification Result
    if (detail.verification_result) {
        const vr = detail.verification_result;
        const cls = vr.passed ? 'passed' : 'failed';
        html += `
            <div class="detail-section">
                <h4>Verification</h4>
                <div class="verification-box ${cls}">
                    ${vr.passed ? 'PASSED' : 'FAILED'}${vr.summary ? ': ' + escHtml(vr.summary.slice(0, 300)) : ''}
                </div>
            </div>
        `;
    }

    // Review Summary
    if (detail.review_summary) {
        html += `
            <div class="detail-section">
                <h4>Review Summary</h4>
                <div class="review-text">${escHtml(detail.review_summary.slice(0, 500))}</div>
            </div>
        `;
    }

    // Error Message
    if (detail.error_message) {
        html += `
            <div class="detail-section">
                <h4>Error</h4>
                <div class="review-text" style="border-color: rgba(239,68,68,0.3); color: var(--red);">${escHtml(detail.error_message)}</div>
            </div>
        `;
    }

    body.innerHTML = html;
    modal.classList.remove('hidden');
}

function closeModal() {
    document.getElementById('task-modal').classList.add('hidden');
}

// Close on Escape
document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') closeModal();
});

// ── Helpers ────────────────────────────────────────────────

function formatNumber(n) {
    if (n == null || n === 0) return '0';
    if (n >= 1_000_000) return (n / 1_000_000).toFixed(1) + 'M';
    if (n >= 1_000) return (n / 1_000).toFixed(1) + 'k';
    return String(n);
}

function formatTime(ms) {
    if (ms == null || ms === 0) return '0s';
    if (ms < 1000) return Math.round(ms) + 'ms';
    const secs = ms / 1000;
    if (secs < 60) return secs.toFixed(1) + 's';
    const mins = Math.floor(secs / 60);
    const remSecs = Math.round(secs % 60);
    return `${mins}m ${remSecs}s`;
}

function escHtml(s) {
    if (!s) return '';
    const div = document.createElement('div');
    div.textContent = String(s);
    return div.innerHTML;
}

function escAttr(s) {
    return String(s).replace(/'/g, "\\'").replace(/"/g, '&quot;');
}

// ── Init ───────────────────────────────────────────────────

connect();
