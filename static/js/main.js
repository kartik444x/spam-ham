/* ── main.js : SpamShield frontend logic ───────────────────────── */

// ── Character counter ───────────────────────────────────────────
const msgInput  = document.getElementById('msgInput');
const charCount = document.getElementById('charCount');

if (msgInput) {
  msgInput.addEventListener('input', () => {
    const len = msgInput.value.length;
    charCount.textContent = `${len} / 500`;
    charCount.style.color = len > 450 ? '#f85149' : '';
    if (len > 500) msgInput.value = msgInput.value.slice(0, 500);
  });
}

// ── Clear button ────────────────────────────────────────────────
document.getElementById('clearBtn')?.addEventListener('click', () => {
  msgInput.value = '';
  charCount.textContent = '0 / 500';
  hide('result');
  hide('errorBox');
  hide('loadingBox');
  document.getElementById('result').className = 'result-box hidden';
});

// ── Sample chips ────────────────────────────────────────────────
document.querySelectorAll('.sample-chip').forEach(btn => {
  btn.addEventListener('click', () => {
    msgInput.value = btn.dataset.msg;
    charCount.textContent = `${btn.dataset.msg.length} / 500`;
    hide('result');
    hide('errorBox');
  });
});

// ── Single prediction ───────────────────────────────────────────
document.getElementById('predictBtn')?.addEventListener('click', async () => {
  const msg = msgInput.value.trim();
  if (!msg) {
    showError('Please enter a message to analyse.');
    return;
  }

  hide('result');
  hide('errorBox');
  show('loadingBox');

  try {
    const res  = await fetch('/predict', {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify({ message: msg }),
    });
    const data = await res.json();

    if (!res.ok || data.error) {
      showError(data.error || 'Server error. Please try again.');
      return;
    }

    renderResult(data);
  } catch (err) {
    showError('Could not connect to the server. Make sure app.py is running.');
  } finally {
    hide('loadingBox');
  }
});

// ── Batch prediction ─────────────────────────────────────────────
document.getElementById('batchBtn')?.addEventListener('click', async () => {
  const raw      = document.getElementById('batchInput').value.trim();
  const messages = raw.split('\n').map(l => l.trim()).filter(Boolean);

  if (!messages.length) {
    alert('Please enter at least one message.');
    return;
  }

  const batchEl = document.getElementById('batchResults');
  batchEl.innerHTML = '<div class="loading"><div class="spinner"></div> Processing …</div>';
  show('batchResults');

  try {
    const res  = await fetch('/batch', {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify({ messages }),
    });
    const data = await res.json();

    if (!res.ok || data.error) {
      batchEl.innerHTML = `<div class="alert alert-error">${data.error}</div>`;
      return;
    }

    renderBatch(data, batchEl);
  } catch (err) {
    batchEl.innerHTML = `<div class="alert alert-error">Connection error.</div>`;
  }
});

// ── Render helpers ───────────────────────────────────────────────

function renderResult({ label, confidence, processed }) {
  const isSpam   = label === 'spam';
  const resultEl = document.getElementById('result');

  document.getElementById('resultIcon').textContent  = isSpam ? '🚨' : '✅';
  document.getElementById('resultLabel').textContent = isSpam ? 'SPAM Detected' : 'Legitimate Message (Ham)';
  document.getElementById('resultSub').textContent   =
    isSpam
      ? 'This message shows strong spam indicators.'
      : 'This message appears to be genuine.';

  document.getElementById('confPct').textContent     = `${confidence}%`;
  const fillEl = document.getElementById('confFill');
  fillEl.style.width = '0%';

  resultEl.className = `result-box ${isSpam ? 'is-spam' : 'is-ham'}`;
  show('result');

  // animate bar after paint
  requestAnimationFrame(() => {
    requestAnimationFrame(() => {
      fillEl.style.width = `${confidence}%`;
    });
  });

  document.getElementById('processedText').textContent = processed || '(empty after cleaning)';
}

function renderBatch(items, container) {
  const spamCount = items.filter(i => i.label === 'spam').length;
  const hamCount  = items.length - spamCount;

  container.innerHTML = `
    <div class="batch-summary">
      <span>📊 ${items.length} messages analysed &nbsp;·&nbsp;
        <span style="color:var(--spam)">🚨 ${spamCount} spam</span> &nbsp;·&nbsp;
        <span style="color:var(--ham)">✅ ${hamCount} ham</span>
      </span>
    </div>
  `;

  items.forEach(({ message, label, confidence }, idx) => {
    const isSpam = label === 'spam';
    const item   = document.createElement('div');
    item.className = 'batch-item';
    item.style.animationDelay = `${idx * 50}ms`;
    item.innerHTML = `
      <span class="batch-msg">${escHtml(message)}</span>
      <span class="batch-badge ${isSpam ? 'badge-spam' : 'badge-ham'}">
        ${isSpam ? 'SPAM' : 'HAM'}
      </span>
      <span class="batch-conf">${confidence}%</span>
    `;
    container.appendChild(item);
  });
}

// ── Utility ──────────────────────────────────────────────────────

function show(id) { document.getElementById(id)?.classList.remove('hidden'); }
function hide(id) { document.getElementById(id)?.classList.add('hidden'); }

function showError(msg) {
  const el = document.getElementById('errorBox');
  el.textContent = msg;
  show('errorBox');
}

function escHtml(str) {
  return str.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}
