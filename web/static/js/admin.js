function renderRatings(rows) {
  const tbody = document.querySelector('#ratingsTable tbody');
  tbody.innerHTML = '';
  for (const r of rows) {
    const tr = document.createElement('tr');
    tr.innerHTML = `
      <td>${r.rank}</td>
      <td>${r.relpath ? `<img src="${r.relpath}" alt="${r.model}">` : ''}</td>
      <td>${r.model}</td>
      <td>${r.rating.toFixed(1)}</td>
      <td>${r.appearances ?? 0}</td>
    `;
    tbody.appendChild(tr);
  }
}

async function refreshOnce() {
  const data = await apiGet('/api/ratings');
  const rows = data.ratings || [];
  renderRatings(rows);
  drawBarChart(rows);
  const hist = await apiGet('/api/history');
  drawLineChart(rows, hist.history || []);
}

window.addEventListener('DOMContentLoaded', () => {
  const status = document.getElementById('status');
  document.getElementById('btnRefresh').addEventListener('click', refreshOnce);
  // load config and populate inputs
  (async () => {
    try {
      const cfg = await apiGet('/api/config');
      const c = cfg.config || {};
      const ir = document.getElementById('initRating');
      const kf = document.getElementById('kFactor');
      if (ir) ir.value = c.initial_rating ?? 1200;
      if (kf) kf.value = c.k_factor ?? 64;
    } catch {}
  })();
  document.getElementById('btnApplyCfg').addEventListener('click', async () => {
    const cfgMsg = document.getElementById('cfgMsg');
    try {
      const ir = parseFloat(document.getElementById('initRating').value);
      const kf = parseInt(document.getElementById('kFactor').value, 10);
      await apiPost('/api/config', { initial_rating: ir, k_factor: kf });
      cfgMsg.textContent = 'Config saved';
    } catch (e) {
      cfgMsg.textContent = 'Failed to save config';
    }
  });
  document.getElementById('btnResetAll').addEventListener('click', async () => {
    const cfgMsg = document.getElementById('cfgMsg');
    try {
      const ir = parseFloat(document.getElementById('initRating').value);
      await apiPost('/api/reset', { scope: 'all', initial_rating: ir });
      cfgMsg.textContent = 'All ratings and history cleared';
      await refreshOnce();
    } catch (e) {
      cfgMsg.textContent = 'Reset failed';
    }
  });
  refreshOnce();
  sseConnect(async (evt) => {
    if (evt.type === 'snapshot' || evt.type === 'rating_update') {
      const rows = evt.payload.ratings || evt.payload || [];
      renderRatings(rows);
      drawBarChart(rows);
      try {
        const hist = await apiGet('/api/history');
        drawLineChart(rows, hist.history || []);
      } catch {}
    }
  }, () => {
    status.textContent = 'Connected';
  }, () => {
    status.textContent = 'Connection error';
  });
});

function getCanvas(id) { return document.getElementById(id); }

function drawBarChart(rows) {
  const c = getCanvas('barChart');
  if (!c) return;
  const ctx = c.getContext('2d');
  const dpr = window.devicePixelRatio || 1;
  const rect = c.getBoundingClientRect();
  const W = Math.max(1, Math.round(rect.width * dpr));
  // ensure a taller chart for readability
  const targetH = Math.max(rect.height, 220);
  const H = Math.max(1, Math.round(targetH * dpr));
  if (c.width !== W) c.width = W; if (c.height !== H) c.height = H;
  ctx.clearRect(0,0,W,H);
  const pad = 40 * dpr;
  const innerW = W - pad*2; const innerH = H - pad*2;
  // data
  const labels = rows.map(r => r.model);
  const values = rows.map(r => r.rating);
  let maxV = Math.max(1, Math.ceil(Math.max(...values, 0) / 50) * 50);
  // add headroom so tallest bar doesn't touch top
  maxV = Math.ceil((maxV * 1.08) / 50) * 50;
  const barW = innerW / Math.max(1, rows.length);
  // axes
  ctx.strokeStyle = '#274072'; ctx.lineWidth = 1 * dpr;
  ctx.beginPath();
  ctx.moveTo(pad, H - pad); ctx.lineTo(W - pad, H - pad); // X
  ctx.moveTo(pad, H - pad); ctx.lineTo(pad, pad); // Y
  ctx.stroke();
  // grid
  ctx.strokeStyle = '#22345c'; ctx.lineWidth = 1 * dpr;
  ctx.font = `${12*dpr}px ui-sans-serif`; ctx.fillStyle = '#9fb3d9';
  const ticks = 7;
  for (let i=0;i<=ticks;i++) {
    const v = (maxV / ticks) * i;
    const y = H - pad - innerH * (v / maxV);
    ctx.beginPath(); ctx.moveTo(pad, y); ctx.lineTo(W - pad, y); ctx.stroke();
    ctx.fillText(String(Math.round(v)), 6*dpr, y - 2*dpr);
  }
  // bars
  const palette = ['#60a5fa','#22d3ee','#a78bfa','#34d399','#f472b6', '#f59e0b', '#ef4444', '#10b981'];
  for (let i=0;i<rows.length;i++) {
    const v = values[i];
    const h = innerH * (v / maxV);
    const x = pad + i * barW + barW*0.15;
    const y = H - pad - h;
    const w = barW * 0.7;
    const col = palette[i % palette.length];
    const grad = ctx.createLinearGradient(0, y, 0, y+h);
    grad.addColorStop(0, col);
    grad.addColorStop(1, '#0f1b36');
    ctx.fillStyle = grad;
    ctx.fillRect(x, y, w, h);
    // label
    ctx.save();
    ctx.translate(x + w/2, H - pad + 14*dpr);
    ctx.rotate(-Math.PI/8);
    ctx.textAlign = 'center';
    ctx.fillStyle = '#b9c9ef';
    ctx.fillText(labels[i], 0, 0);
    ctx.restore();
  }
}

function drawLineChart(ratingsRows, history) {
  const c = getCanvas('lineChart');
  if (!c) return;
  const ctx = c.getContext('2d');
  const dpr = window.devicePixelRatio || 1;
  const rect = c.getBoundingClientRect();
  const W = Math.max(1, Math.round(rect.width * dpr));
  const targetH = Math.max(rect.height, 240);
  const H = Math.max(1, Math.round(targetH * dpr));
  if (c.width !== W) c.width = W; if (c.height !== H) c.height = H;
  ctx.clearRect(0,0,W,H);
  const pad = 44 * dpr;
  const innerW = W - pad*2; const innerH = H - pad*2;
  // Build per-model series (use event index as X)
  const topModels = ratingsRows.slice(0, 5).map(r => r.model);
  const series = {}; const last = {};
  for (const m of topModels) { series[m] = []; }
  // Iterate history in order
  for (let i=0;i<history.length;i++) {
    const h = history[i];
    const wi = h.winner_id || '';
    const li = h.loser_id || '';
    const wm = wi.startsWith('model:') ? wi.slice(6) : null;
    const lm = li.startsWith('model:') ? li.slice(6) : null;
    if (wm) last[wm] = h.winner_rating_after;
    if (lm) last[lm] = h.loser_rating_after;
    for (const m of topModels) {
      if (m in last) series[m].push({ x: i+1, y: last[m] });
    }
  }
  // Determine ranges (compress left empty by using minX of present points)
  let maxV = 0, minV = Infinity; let maxX = 1; let minX = Infinity; let any = false;
  for (const m of topModels) {
    for (const p of (series[m]||[])) {
      any = true;
      maxV = Math.max(maxV, p.y);
      minV = Math.min(minV, p.y);
      maxX = Math.max(maxX, p.x);
      minX = Math.min(minX, p.x);
    }
  }
  if (!any) return; // nothing to draw yet
  if (!isFinite(minV)) { minV = 0; maxV = 1; }
  if (maxV === minV) { maxV += 1; minV -= 1; }
  // add top/bottom margins (~5% range or at least 20)
  const r = maxV - minV;
  const margin = Math.max(20, r * 0.06);
  maxV += margin; minV -= margin;
  if (minV < 0) minV = 0;
  if (!isFinite(minX)) minX = 1;
  const xRange = Math.max(1, maxX - minX);
  // axes & grid
  ctx.strokeStyle = '#274072'; ctx.lineWidth = 1*dpr;
  ctx.beginPath(); ctx.moveTo(pad, H - pad); ctx.lineTo(W - pad, H - pad); ctx.moveTo(pad, H - pad); ctx.lineTo(pad, pad); ctx.stroke();
  ctx.strokeStyle = '#22345c'; ctx.font = `${12*dpr}px ui-sans-serif`; ctx.fillStyle = '#9fb3d9';
  const yticks = 8; const xticks = 8;
  for (let i=0;i<=yticks;i++) {
    const t = minV + (maxV - minV) * (i/yticks);
    const y = H - pad - innerH * ((t - minV)/(maxV - minV));
    ctx.beginPath(); ctx.moveTo(pad, y); ctx.lineTo(W - pad, y); ctx.stroke();
    ctx.fillText(String(Math.round(t)), 6*dpr, y - 2*dpr);
  }
  for (let i=0;i<=xticks;i++) {
    const x = pad + innerW * (i/xticks);
    ctx.beginPath(); ctx.moveTo(x, H - pad); ctx.lineTo(x, H - pad + 4*dpr); ctx.stroke();
    const label = Math.round(minX + xRange * (i/xticks));
    ctx.fillText(String(label), x - 4*dpr, H - pad + 16*dpr);
  }
  // colors
  const colors = ['#60a5fa','#22d3ee','#a78bfa','#34d399','#f472b6'];
  // draw lines
  let ci = 0;
  for (const m of topModels) {
    const pts = series[m] || [];
    if (!pts.length) continue;
    const col = colors[ci++ % colors.length];
    ctx.strokeStyle = col; ctx.lineWidth = 2*dpr;
    ctx.beginPath();
    for (let i=0;i<pts.length;i++) {
      const px = pad + innerW * ((pts[i].x - minX) / xRange);
      const py = H - pad - innerH * ((pts[i].y - minV)/(maxV - minV));
      if (i===0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
    }
    ctx.stroke();
    // legend
    ctx.fillStyle = col; ctx.fillRect(W - pad - 120*dpr, pad + (ci-1)*16*dpr, 10*dpr, 10*dpr);
    ctx.fillStyle = '#b9c9ef'; ctx.fillText(m, W - pad - 104*dpr, pad + (ci-1)*16*dpr + 10*dpr);
  }
}
