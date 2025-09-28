let pairs = [];
let curIdx = -1;
let curA = null;
let curB = null;
let busy = false;

async function runFromPromptFile() {
  const msg = document.getElementById('msg');
  msg.textContent = '';
  try {
    const btn = document.getElementById('btnNew');
    if (btn) { btn.disabled = true; btn.textContent = '生成中…'; }
    const promptEl = document.getElementById('promptInput');
    const negEl = document.getElementById('negInput');
    const prompt = (promptEl?.value || '').trim();
    const negative = (negEl?.value || '').trim();
    let res;
    const usedManual = !!(prompt || negative);
    if (usedManual) {
      res = await apiPost('/api/rt_generate', { prompt, negative });
    } else {
      res = await apiPost('/api/rt_next_from_prompts', {});
    }
    pairs = res.pairs || [];
    curIdx = -1;
    const pel = document.getElementById('currentPrompt');
    const ptxt = res.caption ? `Prompt: ${res.caption}` : '';
    pel.textContent = ptxt;
    pel.title = res.caption || '';
    nextPair();
    // 如果这次是手动 Prompt，生成完成后清空输入框，下一次将回到 prompt.json 随机模式
    if (usedManual) {
      if (promptEl) promptEl.value = '';
      if (negEl) negEl.value = '';
      const qi = document.getElementById('queueInfo');
      if (qi) qi.textContent = '';
    }
    if (btn) { btn.disabled = false; btn.textContent = '生成下一组'; }
  } catch (e) {
    msg.textContent = `生成失败: ${e}`;
    const btn = document.getElementById('btnNew');
    if (btn) { btn.disabled = false; btn.textContent = '生成下一组'; }
  }
}

function showPair(p) {
  const wrapA = document.getElementById('wrapA');
  const wrapB = document.getElementById('wrapB');
  const imgAEl = document.getElementById('imgA');
  const imgBEl = document.getElementById('imgB');
  wrapA.classList.add('loading');
  wrapB.classList.add('loading');
  imgAEl.classList.remove('appear'); imgAEl.classList.add('swap-out');
  imgBEl.classList.remove('appear'); imgBEl.classList.add('swap-out');
  curA = p.a; curB = p.b;
  imgAEl.onload = () => { wrapA.classList.remove('loading'); imgAEl.classList.remove('swap-out'); imgAEl.classList.add('appear'); };
  imgBEl.onload = () => { wrapB.classList.remove('loading'); imgBEl.classList.remove('swap-out'); imgBEl.classList.add('appear'); };
  imgAEl.src = p.a.relpath;
  imgBEl.src = p.b.relpath;
  // 不显示模型名称以避免偏见
  document.getElementById('labLeft').textContent = 'Left';
  document.getElementById('labRight').textContent = 'Right';
  document.getElementById('roundInfo').textContent = `第 1/1 对`;
}

function nextPair() {
  if (!pairs || pairs.length === 0) return;
  curIdx = Math.min(curIdx + 1, pairs.length - 1);
  showPair(pairs[curIdx]);
}

async function vote(winner, loser, tie=false) {
  try {
    if (busy) return; busy = true;
    const wrapA = document.getElementById('wrapA');
    const wrapB = document.getElementById('wrapB');
    const btnA = document.getElementById('btnA');
    const btnB = document.getElementById('btnB');
    const btnTie = document.getElementById('btnTie');
    wrapA.classList.add('choosing');
    wrapB.classList.add('choosing');
    if (!tie) {
      const w = (winner.id === curA.id) ? wrapA : wrapB;
      const l = (loser.id === curA.id) ? wrapA : wrapB;
      w.classList.add('win-glow');
      l.classList.add('lose-dim');
      // finish burst effect on winner
      const burst = document.createElement('div');
      burst.className = 'finish-burst';
      w.appendChild(burst);
      // confetti & sparkles
      launchConfetti(w, 26);
      launchSparkles(w, 8);
      // button celebrate
      const winBtn = (winner.id === curA.id) ? btnA : btnB;
      if (winBtn) winBtn.classList.add('celebrate');
    } else {
      wrapA.classList.add('win-glow');
      wrapB.classList.add('win-glow');
      const b1 = document.createElement('div'); b1.className = 'finish-burst neutral'; wrapA.appendChild(b1);
      const b2 = document.createElement('div'); b2.className = 'finish-burst neutral'; wrapB.appendChild(b2);
      launchConfetti(wrapA, 16);
      launchConfetti(wrapB, 16);
      launchSparkles(wrapA, 6);
      launchSparkles(wrapB, 6);
      if (btnTie) btnTie.classList.add('pulse');
    }
    const payload = (winner.model && loser.model) ? {
      winner_model: winner.model, loser_model: loser.model, tie
    } : { winner_id: winner.id, loser_id: loser.id, tie };
    await apiPost('/api/compare', payload);
    setTimeout(() => {
      wrapA.classList.remove('win-glow','lose-dim','choosing');
      wrapB.classList.remove('win-glow','lose-dim','choosing');
      // remove finish bursts
      document.querySelectorAll('.finish-burst').forEach(el => el.remove());
      // remove confetti/sparkles and button states
      document.querySelectorAll('.confetti-piece,.sparkle').forEach(el => el.remove());
      if (btnA) btnA.classList.remove('celebrate');
      if (btnB) btnB.classList.remove('celebrate');
      if (btnTie) btnTie.classList.remove('pulse');
      // 每组只测一次，自动进入下一组
      runFromPromptFile();
      busy = false;
    }, 360);
  } catch (e) {
    document.getElementById('msg').textContent = `提交失败: ${e}`;
    const wrapA = document.getElementById('wrapA');
    const wrapB = document.getElementById('wrapB');
    wrapA.classList.remove('choosing');
    wrapB.classList.remove('choosing');
    busy = false;
  }
}

window.addEventListener('DOMContentLoaded', () => {
  document.getElementById('btnNew').addEventListener('click', runFromPromptFile);
  document.getElementById('btnEnq').addEventListener('click', enqueueGenerate);
  document.getElementById('btnTie').addEventListener('click', () => curA && curB && vote(curA, curB, true));
  document.getElementById('btnA').addEventListener('click', () => curA && curB && vote(curA, curB));
  document.getElementById('btnB').addEventListener('click', () => curA && curB && vote(curB, curA));
  // keyboard
  window.addEventListener('keydown', (e) => {
    const ae = document.activeElement;
    if (ae && (ae.tagName === 'INPUT' || ae.tagName === 'TEXTAREA')) return;
    if (e.key === 'Enter') runFromPromptFile();
    if (e.key === 'ArrowLeft' || e.key.toLowerCase() === 'a') { if (curA && curB) vote(curA, curB); }
    if (e.key === 'ArrowRight' || e.key.toLowerCase() === 'b') { if (curA && curB) vote(curB, curA); }
    if (e.key.toLowerCase() === 't') { if (curA && curB) vote(curA, curB, true); }
  });
  // auto start one batch on load
  runFromPromptFile();
  // no SSE overlay for ratings
});

// Visual effects
function launchConfetti(container, count=24) {
  const colors = ['#f87171','#fbbf24','#34d399','#60a5fa','#a78bfa','#f472b6'];
  for (let i=0;i<count;i++) {
    const p = document.createElement('div');
    p.className = 'confetti-piece';
    const c = colors[i % colors.length];
    const left = (Math.random()*80 + 10).toFixed(2) + '%';
    const w = (Math.random()*4 + 4).toFixed(1) + 'px';
    const h = (Math.random()*8 + 8).toFixed(1) + 'px';
    const dur = (900 + Math.random()*900).toFixed(0) + 'ms';
    const delay = (Math.random()*120).toFixed(0) + 'ms';
    const dx = (Math.random()*120 - 60).toFixed(1) + 'px';
    p.style.left = left;
    p.style.setProperty('--w', w);
    p.style.setProperty('--h', h);
    p.style.setProperty('--color', c);
    p.style.setProperty('--dur', dur);
    p.style.setProperty('--delay', delay);
    p.style.setProperty('--dx', dx);
    container.appendChild(p);
    setTimeout(() => p.remove(), 2200);
  }
}

function launchSparkles(container, count=6) {
  for (let i=0;i<count;i++) {
    const s = document.createElement('div');
    s.className = 'sparkle';
    const size = (Math.random()*6 + 6).toFixed(1) + 'px';
    const l = (Math.random()*60 + 20).toFixed(2) + '%';
    const t = (Math.random()*60 + 20).toFixed(2) + '%';
    s.style.setProperty('--s', size);
    s.style.setProperty('--l', l);
    s.style.setProperty('--t', t);
    container.appendChild(s);
    setTimeout(() => s.remove(), 1200);
  }
}

// Queue-based generation
async function enqueueGenerate() {
  const msg = document.getElementById('msg');
  const qi = document.getElementById('queueInfo');
  try {
    const prompt = (document.getElementById('promptInput')?.value || '').trim();
    const negative = (document.getElementById('negInput')?.value || '').trim();
    const res = await apiPost('/api/rt_enqueue', { prompt, negative });
    const id = res.job_id;
    qi.textContent = `已入队 (#${id.slice(0,6)})，等待生成…`;
    // poll until done
    const poll = async () => {
      const jr = await apiGet(`/api/rt_job/${id}`);
      if (jr.status === 'done' && jr.result) {
        qi.textContent = `已完成 (#${id.slice(0,6)})`;
        pairs = jr.result.pairs || [];
        curIdx = -1;
        const pel = document.getElementById('currentPrompt');
        pel.textContent = jr.result.caption ? `Prompt: ${jr.result.caption}` : '';
        nextPair();
        // 清空输入框，使下一次回到 prompt.json 随机模式
        const promptEl = document.getElementById('promptInput');
        const negEl = document.getElementById('negInput');
        if (promptEl) promptEl.value = '';
        if (negEl) negEl.value = '';
        return;
      }
      if (jr.status === 'error') {
        msg.textContent = `生成失败：${jr.error || '未知错误'}`;
        qi.textContent = '';
        return;
      }
      setTimeout(poll, 900);
    };
    poll();
  } catch (e) {
    msg.textContent = `排队失败: ${e}`;
    qi.textContent = '';
  }
}
