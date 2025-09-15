async function banCurrent() {
  try {
    const iframe = document.getElementById('base');
    const doc = iframe.contentWindow.document;
    const hidden = doc.querySelector('input[name="video_name"]');
    let videoName = hidden ? hidden.value : null;
    // Fallback: parse from current video src
    if (!videoName) {
      const srcEl = doc.querySelector('video source');
      if (srcEl && srcEl.src) {
        try {
          const url = new URL(srcEl.src);
          const parts = url.pathname.split('/');
          videoName = decodeURIComponent(parts[parts.length - 1] || '');
        } catch (_) {}
      }
    }
    const payload = videoName ? { video_name: videoName } : {};
    const res = await fetch('/api/ban_combo', { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(payload) });
    const data = await res.json();
    if (iframe.contentWindow && typeof iframe.contentWindow.nextVideo === 'function') {
      iframe.contentWindow.nextVideo();
    }
    console.log('Ban result:', data);
  } catch(e) {
    alert('Ban failed: ' + e.message);
  }
}


