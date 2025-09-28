const API_BASE = (typeof window !== 'undefined' && window.API_BASE) ? window.API_BASE : '';

async function apiGet(path) {
  const res = await fetch(API_BASE + path, { headers: { 'Accept': 'application/json' } });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

async function apiPost(path, body) {
  const res = await fetch(API_BASE + path, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body ?? {})
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

async function apiUploadFiles(files) {
  const fd = new FormData();
  for (const f of files) fd.append('files', f);
  const res = await fetch(API_BASE + '/api/upload', { method: 'POST', body: fd });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

function sseConnect(onEvent, onOpen, onError) {
  const es = new EventSource(API_BASE + '/api/events');
  es.onmessage = (ev) => {
    try { onEvent && onEvent(JSON.parse(ev.data)); } catch {}
  };
  es.onopen = () => onOpen && onOpen();
  es.onerror = (e) => onError && onError(e);
  return es;
}
