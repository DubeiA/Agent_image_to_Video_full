async function runSearch() {
  const params = new URLSearchParams();
  const get = id => document.getElementById(id).value.trim();
  const fields = ['q','rated','banned','reference','min_score','min_overall','fps','width','height','sampler','scheduler'];
  for (const f of fields) { const v = get(f); if (v) params.set(f, v); }
  const res = await fetch('/api/search?' + params.toString());
  const data = await res.json();
  const tb = document.querySelector('#res tbody');
  tb.innerHTML='';
  for (const it of data.items) {
    const tr = document.createElement('tr');
    const status = [];
    if (it.rated) status.push('rated');
    if (it.banned) status.push('banned');
    if (it.reference) status.push('ref');
    tr.innerHTML = `<td>${it.name}${it.exists?'':' <span class="pill">(no file)</span>'}</td>
                    <td>${it.score?.toFixed?.(3) ?? '-'}</td>
                    <td>${it.manual_overall ?? '-'}</td>
                    <td>${status.join(', ')||'-'}</td>
                    <td>${it.params.fps||'-'}fps, ${it.params.width||'-'}x${it.params.height||'-'}, ${it.params.sampler||'-'}/${it.params.scheduler||'-'}</td>
                    <td><a href="/watch?name=${encodeURIComponent(it.name)}">Відкрити</a></td>`;
    tb.appendChild(tr);
  }
}
runSearch();


