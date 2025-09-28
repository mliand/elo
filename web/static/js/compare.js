let curA = null;
let curB = null;
let currentPrompt = '';
let busy = false;

async function loadPair() {
  try {
    // show loading overlay
    const wrapA = document.getElementById('wrapA');
    const wrapB = document.getElementById('wrapB');
    wrapA.classList.add('loading');
    wrapB.classList.add('loading');
  const imgAEl = document.getElementById('imgA');
  const imgBEl = document.getElementById('imgB');
    // graceful swap-out
    imgAEl.classList.remove('appear');
    imgBEl.classList.remove('appear');
    imgAEl.classList.add('swap-out');
    imgBEl.classList.add('swap-out');
  const url = currentPrompt ? `/api/next_pair?prompt=${encodeURIComponent(currentPrompt)}` : '/api/next_pair';
  const { a, b, caption } = await apiGet(url);
  curA = a; curB = b;
    imgAEl.onload = () => { wrapA.classList.remove('loading'); imgAEl.classList.remove('swap-out'); imgAEl.classList.add('appear'); };
    imgBEl.onload = () => { wrapB.classList.remove('loading'); imgBEl.classList.remove('swap-out'); imgBEl.classList.add('appear'); };
    imgAEl.src = a.relpath;
    imgBEl.src = b.relpath;
    // hide model/filename to reduce bias
    const capEl = document.getElementById('currentPrompt');
    if (capEl) {
      const text = (currentPrompt || caption || '').trim();
      capEl.textContent = text ? `Prompt: ${text}` : '';
    }
    document.getElementById('msg').textContent = '';
  } catch (e) {
    document.getElementById('msg').textContent = `Failed to get pair: ${e}`;
  }
}

async function vote(winner, loser, tie=false) {
  try {
    if (busy) return; busy = true;
    const wrapA = document.getElementById('wrapA');
    const wrapB = document.getElementById('wrapB');
    // overlay spinners on both images while selecting opponent
    wrapA.classList.add('choosing');
    wrapB.classList.add('choosing');
    if (!tie) {
      const w = (winner.id === curA.id) ? wrapA : wrapB;
      const l = (loser.id === curA.id) ? wrapA : wrapB;
      w.classList.add('win-glow');
      l.classList.add('lose-dim');
      // winner badge with model name
      let model = winner.model || '';
      if (!model && winner.id) {
        if (winner.id.startsWith('model:')) model = winner.id.slice(6);
        else if (winner.id.includes(':')) model = winner.id.split(':').pop();
      }
      const badge = document.createElement('div');
      badge.className = 'win-badge';
      badge.textContent = model ? `Winner: ${model}` : 'Winner';
      w.appendChild(badge);
    } else {
      wrapA.classList.add('win-glow');
      wrapB.classList.add('win-glow');
    }
    // send model ids if available
    const payload = (winner.model && loser.model) ? {
      winner_model: winner.model, loser_model: loser.model, tie
    } : { winner_id: winner.id, loser_id: loser.id, tie };
    await apiPost('/api/compare', payload);
    setTimeout(async () => {
      wrapA.classList.remove('win-glow','lose-dim','choosing');
      wrapB.classList.remove('win-glow','lose-dim','choosing');
      // remove badge(s) if any
      document.querySelectorAll('.win-badge').forEach(el => el.remove());
      await loadPair();
      busy = false;
    }, 480);
  } catch (e) {
    document.getElementById('msg').textContent = `Submit failed: ${e}`;
    const wrapA = document.getElementById('wrapA');
    const wrapB = document.getElementById('wrapB');
    wrapA.classList.remove('choosing');
    wrapB.classList.remove('choosing');
    busy = false;
  }
}

window.addEventListener('DOMContentLoaded', () => {
  document.getElementById('btnNext').addEventListener('click', loadPair);
  document.getElementById('btnTie').addEventListener('click', () => curA && curB && vote(curA, curB, true));
  document.getElementById('btnA').addEventListener('click', () => curA && curB && vote(curA, curB));
  document.getElementById('btnB').addEventListener('click', () => curA && curB && vote(curB, curA));
  // no prompt input; only display caption/prompt from server
  // keyboard shortcuts
  window.addEventListener('keydown', (e) => {
    const ae = document.activeElement;
    if (ae && (ae.tagName === 'INPUT' || ae.tagName === 'TEXTAREA')) return;
    if (e.key === 'ArrowLeft' || e.key.toLowerCase() === 'a') { if (curA && curB) vote(curA, curB); }
    if (e.key === 'ArrowRight' || e.key.toLowerCase() === 'b') { if (curA && curB) vote(curB, curA); }
    if (e.key.toLowerCase() === 't') { if (curA && curB) vote(curA, curB, true); }
  });
  // ripple effect
  for (const btn of document.querySelectorAll('.btn')) {
    btn.addEventListener('click', (e) => {
      const r = document.createElement('span');
      r.className = 'ripple';
      const rect = btn.getBoundingClientRect();
      r.style.left = (e.clientX - rect.left) + 'px';
      r.style.top = (e.clientY - rect.top) + 'px';
      btn.appendChild(r);
      setTimeout(() => r.remove(), 700);
    });
  }
  loadPair();
});
