function renderRatings(rows) {
  const tbody = document.querySelector('#ratingsTable tbody');
  tbody.innerHTML = '';
  for (const r of rows) {
    const tr = document.createElement('tr');
    tr.innerHTML = `
      <td>${r.rank}</td>
      <td><img src="${r.relpath}" alt="${r.filename}"></td>
      <td>${r.filename}</td>
      <td>${r.rating.toFixed(1)}</td>
    `;
    tbody.appendChild(tr);
  }
}

async function refreshOnce() {
  const data = await apiGet('/api/ratings');
  renderRatings(data.ratings || []);
}

window.addEventListener('DOMContentLoaded', () => {
  const status = document.getElementById('status');
  document.getElementById('btnRefresh').addEventListener('click', refreshOnce);
  refreshOnce();
  sseConnect((evt) => {
    if (evt.type === 'snapshot' || evt.type === 'rating_update') {
      renderRatings(evt.payload.ratings || evt.payload || []);
    }
  }, () => {
    status.textContent = '实时连接成功';
  }, () => {
    status.textContent = '实时连接异常';
  });
});

