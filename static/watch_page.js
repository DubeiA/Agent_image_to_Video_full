const nameParam = new URLSearchParams(location.search).get('name');
async function loadDetails(){
  const title = document.getElementById('title');
  const vb = document.getElementById('video_box');
  const details = document.getElementById('details');
  const hiddenName = document.querySelector('input[name="video_name"]');
  title.textContent = 'Перегляд відео: ' + (nameParam || '(не обрано)');
  if (hiddenName) hiddenName.value = nameParam || '';
  if(!nameParam) { vb.textContent = 'Не обрано відео'; details.textContent=''; return; }
  const res = await fetch('/api/video_details?name=' + encodeURIComponent(nameParam));
  const data = await res.json();
  details.textContent = JSON.stringify(data, null, 2);
  if (data.exists) {
    vb.innerHTML = '<video controls muted><source src="/video/' + nameParam + '" type="video/mp4"></video>'
  } else {
    vb.textContent = 'Файл відео не знайдено';
  }
  if (data.manual_rating) {
    const f = document.getElementById('rate_form');
    for (const k in data.manual_rating) {
      if (f[k]) f[k].value = data.manual_rating[k];
      if (k === 'is_reference') f[k].checked = !!data.manual_rating[k];
    }
  }
}
document.getElementById('rate_form').addEventListener('submit', async (e)=>{
  e.preventDefault();
  const fd = new FormData(e.target);
  const rating = {};
  for (const [k,v] of fd.entries()){
    if (k==='video_name') continue;
    if (k==='is_reference') rating[k] = true; else rating[k] = isNaN(v)? v : parseFloat(v);
  }
  const res = await fetch('/api/rate', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({video_name: nameParam, rating})});
  const out = await res.json();
  document.getElementById('msg').textContent = JSON.stringify(out);
  if (out.status==='success') loadDetails();
});
loadDetails();


