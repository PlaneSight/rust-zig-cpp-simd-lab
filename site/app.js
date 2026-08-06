const state = { catalog: null, view: 'runtime' };
const $ = (id) => document.getElementById(id);
const filters = {
  family: $('family'), kernel: $('kernel'), language: $('language'), compiler: $('compiler'), isa: $('isa'), type: $('type'), search: $('search')
};

function esc(v) {
  return String(v ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
}
function optionList(select, values, label='All') {
  select.innerHTML = `<option value="">${label}</option>` + values.map(v => `<option>${esc(v)}</option>`).join('');
}
function fmtMetric(metric) {
  if (!metric) return '—';
  const value = Number(metric.value);
  const rendered = Number.isFinite(value) ? (Math.abs(value) >= 100 ? value.toFixed(1) : value.toFixed(4).replace(/0+$/,'').replace(/\.$/,'')) : String(metric.value);
  return `${rendered}${metric.unit ? ` ${esc(metric.unit)}` : ''}`;
}
function runtimeMetric(row, names) {
  for (const name of names) if (row.runtime?.[name]) return row.runtime[name];
  return null;
}
function matches(row) {
  const map = {family:'family',kernel:'kernel',language:'language',compiler:'compiler',isa:'isa',type:'data_type'};
  for (const [filter, key] of Object.entries(map)) if (filters[filter].value && String(row[key]) !== filters[filter].value) return false;
  const q = filters.search.value.trim().toLowerCase();
  if (!q) return true;
  const haystack = [row.family,row.kernel,row.language,row.compiler,row.compiler_version,row.implementation,row.variant,row.isa,row.data_type,row.target,row.dataset,row.semantics_contract].filter(Boolean).join(' ').toLowerCase();
  return haystack.includes(q);
}
function summarize(catalog) {
  const obs = catalog.observation_counts || {};
  $('summary').innerHTML = [
    ['Results', catalog.count], ['Runtime', obs.runtime || 0], ['Codegen', obs.codegen || 0], ['Semantics', obs.semantics || 0], ['Analyses', obs.analysis || 0]
  ].map(([label,value]) => `<div class="stat"><strong>${value}</strong><span>${label}</span></div>`).join('');
}
function baseCells(row) {
  return `<td class="primary">${esc(row.kernel)}</td><td>${esc(row.language || '—')}</td><td>${esc(row.implementation || '—')}</td><td><span class="badge">${esc(row.isa || row.target || 'portable')}</span></td><td>${esc(row.data_type || '—')}</td>`;
}
function runtimeTable(rows) {
  const shown = rows.filter(r => r.runtime && Object.keys(r.runtime).length);
  const body = shown.map(r => {
    const ns = runtimeMetric(r,['ns_per_element','ns/element','nanoseconds_per_element']);
    const bw = runtimeMetric(r,['gib_per_second','GiB/s','bandwidth_gib_s']);
    const cycles = runtimeMetric(r,['cycles_per_element']);
    return `<tr>${baseCells(r)}<td class="metric">${fmtMetric(ns)}</td><td class="metric">${fmtMetric(bw)}</td><td class="metric">${fmtMetric(cycles)}</td><td>${esc(r.compiler || '—')} ${esc(r.compiler_version || '')}</td></tr>`;
  }).join('');
  return {count: shown.length, html:`<thead><tr><th>Kernel</th><th>Lang</th><th>Implementation</th><th>ISA/target</th><th>Type</th><th>ns/element</th><th>GiB/s</th><th>cycles/element</th><th>Compiler</th></tr></thead><tbody>${body}</tbody>`};
}
function codegenTable(rows) {
  const shown = rows.filter(r => r.codegen);
  const body = shown.map(r => {
    const tracked = Object.entries(r.codegen.tracked_mnemonics || {}).sort((a,b)=>b[1]-a[1]).slice(0,6).map(([k,v])=>`${esc(k)}:${v}`).join(' · ');
    return `<tr>${baseCells(r)}<td class="metric">${r.codegen.instruction_count ?? '—'}</td><td class="metric">${r.codegen.vector_instruction_count ?? '—'}</td><td>${tracked || '—'}</td><td>${r.codegen.assembly_path ? `<code>${esc(r.codegen.assembly_path)}</code>` : '—'}</td></tr>`;
  }).join('');
  return {count: shown.length, html:`<thead><tr><th>Kernel</th><th>Lang</th><th>Implementation</th><th>ISA/target</th><th>Type</th><th>Instructions</th><th>Vector insns</th><th>Tracked mnemonics</th><th>Assembly</th></tr></thead><tbody>${body}</tbody>`};
}
function semanticsTable(rows) {
  const shown = rows.filter(r => r.semantics);
  const body = shown.map(r => `<tr>${baseCells(r)}<td><span class="badge ${esc(r.semantics.status)}">${esc(r.semantics.status)}</span></td><td>${esc(r.semantics.reference || '—')}</td><td class="metric">${r.semantics.mismatches ?? '—'} / ${r.semantics.cases ?? '—'}</td><td>${esc(r.semantics.summary || '—')}</td></tr>`).join('');
  return {count: shown.length, html:`<thead><tr><th>Kernel</th><th>Lang</th><th>Implementation</th><th>ISA/target</th><th>Type</th><th>Status</th><th>Reference</th><th>Mismatches</th><th>Summary</th></tr></thead><tbody>${body}</tbody>`};
}
function analysisTable(rows) {
  const shown = rows.filter(r => r.analysis?.length);
  const body = shown.map(r => `<tr>${baseCells(r)}<td><div class="analysis-list">${r.analysis.map(a => `<div class="analysis-item"><span class="badge ${esc(a.severity)}">${esc(a.severity || 'info')}</span> ${esc(a.summary)}</div>`).join('')}</div></td><td><code>${esc(r.commit?.slice(0,12) || '—')}</code></td></tr>`).join('');
  return {count: shown.length, html:`<thead><tr><th>Kernel</th><th>Lang</th><th>Implementation</th><th>ISA/target</th><th>Type</th><th>Analysis</th><th>Commit</th></tr></thead><tbody>${body}</tbody>`};
}
function render() {
  if (!state.catalog) return;
  const rows = state.catalog.results.filter(matches);
  const renderer = {runtime:runtimeTable,codegen:codegenTable,semantics:semanticsTable,analysis:analysisTable}[state.view];
  const result = renderer(rows);
  $('status').textContent = `${result.count} ${state.view} result${result.count === 1 ? '' : 's'} shown · ${rows.length} bundles match filters`;
  $('results').innerHTML = result.html;
  if (!state.catalog.count) {
    $('results').replaceChildren($('empty-template').content.cloneNode(true));
    $('status').textContent = 'Catalog is empty';
  }
}
async function boot() {
  try {
    const res = await fetch('./data/catalog.json', {cache:'no-store'});
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    state.catalog = await res.json();
    summarize(state.catalog);
    const f = state.catalog.facets || {};
    optionList(filters.family, f.family || []); optionList(filters.kernel, f.kernel || []); optionList(filters.language, f.language || []); optionList(filters.compiler, f.compiler || []); optionList(filters.isa, f.isa || []); optionList(filters.type, f.data_type || []);
    Object.values(filters).forEach(el => el.addEventListener(el.tagName === 'INPUT' ? 'input' : 'change', render));
    document.querySelectorAll('.tab').forEach(btn => btn.addEventListener('click', () => {
      state.view = btn.dataset.view;
      document.querySelectorAll('.tab').forEach(x => x.classList.toggle('active', x === btn));
      render();
    }));
    render();
  } catch (err) {
    $('status').textContent = `Could not load result catalog: ${err.message}`;
  }
}
boot();
