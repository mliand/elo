let curA = null;
let curB = null;

async function loadPair() {
  try {
    // show loading overlay
    document.getElementById('wrapA').classList.add('loading');
    document.getElementById('wrapB').classList.add('loading');
    const imgAEl = document.getElementById('imgA');
    const imgBEl = document.getElementById('imgB');
    const { a, b } = await apiGet('/api/next_pair');
    curA = a; curB = b;
    imgAEl.onload = () => document.getElementById('wrapA').classList.remove('loading');
    imgBEl.onload = () => document.getElementById('wrapB').classList.remove('loading');
    imgAEl.src = a.relpath;
    imgBEl.src = b.relpath;
    document.getElementById('metaA').textContent = `${a.filename} · ${a.rating.toFixed(1)}`;
    document.getElementById('metaB').textContent = `${b.filename} · ${b.rating.toFixed(1)}`;
    document.getElementById('msg').textContent = '';
  } catch (e) {
    document.getElementById('msg').textContent = `无法获取图片对：${e}`;
  }
}

async function vote(winner, loser, tie=false) {
  try {
    await apiPost('/api/compare', { winner_id: winner.id, loser_id: loser.id, tie });
    await loadPair();
  } catch (e) {
    document.getElementById('msg').textContent = `提交失败：${e}`;
  }
}

window.addEventListener('DOMContentLoaded', () => {
  document.getElementById('btnNext').addEventListener('click', loadPair);
  document.getElementById('btnTie').addEventListener('click', () => curA && curB && vote(curA, curB, true));
  document.getElementById('btnA').addEventListener('click', () => curA && curB && vote(curA, curB));
  document.getElementById('btnB').addEventListener('click', () => curA && curB && vote(curB, curA));
  loadPair();
});
