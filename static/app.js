document.addEventListener('DOMContentLoaded', ()=>{
  const titleInput = document.getElementById('titleInput');
  const searchBtn = document.getElementById('searchBtn');
  const recList = document.getElementById('recList');
  const titlesList = document.getElementById('titlesList');

  // Load titles for autocomplete
  fetch('/titles')
    .then(r=>r.json())
    .then(data=>{
      // limit to first 2000 titles to avoid huge DOM
      data.slice(0,2000).forEach(t=>{
        const opt = document.createElement('option');
        opt.value = t;
        titlesList.appendChild(opt);
      });
    }).catch(()=>{});

  function showResults(items){
    recList.innerHTML = '';
    if(items.length === 0){
      recList.innerHTML = '<li>No recommendations found.</li>';
      return;
    }
    items.forEach(t=>{
      const li = document.createElement('li');
      li.textContent = t;
      recList.appendChild(li);
    });
  }

  searchBtn.addEventListener('click', ()=>{
    const title = titleInput.value.trim();
    if(!title) return;
    searchBtn.disabled = true;
    searchBtn.textContent = 'Searching...';
    fetch(`/recommend?title=${encodeURIComponent(title)}`)
      .then(r=>r.json())
      .then(data=>{
        if(data.recommendations) showResults(data.recommendations);
        else showResults([]);
      }).catch(()=> showResults([]))
      .finally(()=>{searchBtn.disabled=false;searchBtn.textContent='Recommend'});
  });

  titleInput.addEventListener('keydown', (e)=>{ if(e.key === 'Enter'){ searchBtn.click(); } });
});