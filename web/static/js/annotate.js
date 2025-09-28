let images = [];
let current = null; // current image record
let boxes = []; // [{x,y,w,h,label}] in normalized coords
let drawing = false;
let startX = 0, startY = 0;

function $(id) { return document.getElementById(id); }

async function refreshList() {
  const res = await apiGet('/api/images');
  images = res.images || [];
  const list = $('imageList');
  list.innerHTML = '';
  for (const img of images) {
    const div = document.createElement('div');
    div.className = 'image-item';
    div.dataset.id = img.id;
    div.innerHTML = `<img src="${img.relpath}" alt="${img.filename}"><div class=\"name\" title=\"${img.filename}\">${img.filename}</div>`;
    div.addEventListener('click', () => selectImage(img));
    list.appendChild(div);
  }
}

async function selectImage(img) {
  current = img;
  $('annImg').src = img.relpath;
  $('annMsg').textContent = '';
  // highlight selection
  document.querySelectorAll('.image-item').forEach(el => {
    el.classList.toggle('active', el.dataset.id === img.id);
  });
  // fetch existing annotations
  const data = await apiGet(`/api/annotations/${img.id}`);
  boxes = data.annotations || [];
  renderBoxes();
}

function setupCanvas() {
  const img = $('annImg');
  const canvas = $('annCanvas');
  const wrap = canvas.parentElement;
  function syncSize() {
    // canvas overlays the image element
    const rect = img.getBoundingClientRect();
    const wrapRect = wrap.getBoundingClientRect();
    const width = wrap.clientWidth;
    const scale = rect.width / img.naturalWidth;
    canvas.width = rect.width;
    canvas.height = rect.height;
    canvas.style.width = rect.width + 'px';
    canvas.style.height = rect.height + 'px';
    canvas.style.left = rect.left - wrapRect.left + 'px';
    canvas.style.top = rect.top - wrapRect.top + 'px';
    renderBoxes();
  }
  img.addEventListener('load', syncSize);
  window.addEventListener('resize', syncSize);

  canvas.addEventListener('mousedown', (e) => {
    if (!$('btnAddBox').classList.contains('active')) return;
    const rect = canvas.getBoundingClientRect();
    drawing = true;
    startX = e.clientX - rect.left;
    startY = e.clientY - rect.top;
  });
  canvas.addEventListener('mousemove', (e) => {
    if (!drawing) return;
    renderBoxes();
    const ctx = canvas.getContext('2d');
    const rect = canvas.getBoundingClientRect();
    const curX = e.clientX - rect.left;
    const curY = e.clientY - rect.top;
    const x = Math.min(startX, curX);
    const y = Math.min(startY, curY);
    const w = Math.abs(curX - startX);
    const h = Math.abs(curY - startY);
    ctx.strokeStyle = '#4f7cff';
    ctx.lineWidth = 2;
    ctx.strokeRect(x, y, w, h);
  });
  canvas.addEventListener('mouseup', (e) => {
    if (!drawing) return;
    drawing = false;
    const rect = canvas.getBoundingClientRect();
    const curX = e.clientX - rect.left;
    const curY = e.clientY - rect.top;
    const x = Math.min(startX, curX);
    const y = Math.min(startY, curY);
    const w = Math.abs(curX - startX);
    const h = Math.abs(curY - startY);
    if (w < 4 || h < 4) return; // too small
    // normalize
    const nx = x / canvas.width;
    const ny = y / canvas.height;
    const nw = w / canvas.width;
    const nh = h / canvas.height;
    const label = $('annLabel').value.trim() || 'object';
    boxes.push({ x: nx, y: ny, w: nw, h: nh, label });
    renderBoxes();
    $('btnAddBox').classList.remove('active');
  });
}

function renderBoxes() {
  const canvas = $('annCanvas');
  const ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.lineWidth = 2;
  const list = $('boxList');
  list.innerHTML = '';
  boxes.forEach((b, i) => {
    const x = b.x * canvas.width;
    const y = b.y * canvas.height;
    const w = b.w * canvas.width;
    const h = b.h * canvas.height;
    ctx.strokeStyle = '#22c55e';
    ctx.strokeRect(x, y, w, h);
    ctx.fillStyle = 'rgba(34,197,94,0.15)';
    ctx.fillRect(x, y, w, h);
    ctx.fillStyle = '#e6e8ee';
    ctx.font = '12px sans-serif';
    ctx.fillText(b.label, x + 4, y + 14);

    const li = document.createElement('li');
    li.innerHTML = `<span>#${i+1} ${b.label} (${b.x.toFixed(2)}, ${b.y.toFixed(2)}, ${b.w.toFixed(2)}, ${b.h.toFixed(2)})</span>`;
    const del = document.createElement('button');
    del.textContent = '删除';
    del.className = 'btn';
    del.addEventListener('click', () => { boxes.splice(i,1); renderBoxes(); });
    li.appendChild(del);
    list.appendChild(li);
  });
}

async function saveAnnotations() {
  if (!current) { $('annMsg').textContent = '请先选择图片'; return; }
  try {
    await apiPost(`/api/annotations/${current.id}`, boxes);
    $('annMsg').textContent = '保存成功';
  } catch (e) {
    $('annMsg').textContent = '保存失败：' + e;
  }
}

// 上传和导入能力已移除（按需保留接口）。

window.addEventListener('DOMContentLoaded', () => {
  $('btnSaveAnn').addEventListener('click', saveAnnotations);
  $('btnClear').addEventListener('click', () => { boxes = []; renderBoxes(); });
  $('btnAddBox').addEventListener('click', (e) => e.target.classList.toggle('active'));
  setupCanvas();
  refreshList();
});
