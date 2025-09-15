async function loadVideos() {
  const res = await fetch('/api/search?rated=all&offset=0&limit=50');
  const data = await res.json();
  const items = data.items || [];
  const box = document.getElementById('list');
  box.innerHTML = '';
  for (const v of items) {
    const el = document.createElement('div');
    el.className = 'card';
    const combo = v.params && v.params.sampler ? (v.params.sampler + '/' + (v.params.scheduler||'')) : (v.combo || 'N/A');
    const fps = v.params && v.params.fps || 'N/A';
    const w = v.params && v.params.width || 'N/A';
    const h = v.params && v.params.height || 'N/A';
    el.innerHTML = `
      <div class="row">
        <div>
          <div><strong>${v.name}</strong></div>
          <div class="small">${w}x${h}, fps=${fps}, combo=${combo}</div>
        </div>
        <div>
          <a href="/watch?name=${encodeURIComponent(v.name)}" style="margin-right:8px;color:#90caf9">Відкрити</a>
          <button onclick="banByVideo('${v.name}')">🚫 Ban combo</button>
        </div>
      </div>`;
    box.appendChild(el);
  }
}

async function banByVideo(videoName) {
  const res = await fetch('/api/ban_combo', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ video_name: videoName })
  });
  const data = await res.json();
  if (data.status === 'success') {
    alert('✅ Забанено: ' + (data.banned_combo || '') + '\nВідео позначено: ' + (data.video_marked ? 'так' : 'ні'));
    // Опційно перезавантажити список
    try { await loadVideos(); } catch(_) {}
  } else {
    alert('❌ Помилка: ' + (data.message || 'unknown'));
  }
}

window.addEventListener('load', loadVideos);


