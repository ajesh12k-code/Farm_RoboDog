// WP10 orchard console (OS 5.5, MISSION-PRESETS 0.7): orchard map, tree card, preset bar, queue, progress chips,
// live band strips, crown gauge, recovery menu, assist dialog, job sheet download. Pure UI: every command goes out
// through send(msg); every input is a server message (map / preset_status / coverage / mast / recovery /
// tree_record / mark / jobsheet) or the 30 Hz state frame (robot pose). Hindi first.
import { getLang, reason, t } from './i18n.js';

const $ = (id) => document.getElementById(id);
const num = (v) => (typeof v === 'number' && Number.isFinite(v) ? v : null);
const CROP_COLOUR = { mango: '#f59e0b', guava: '#84cc16', papaya: '#fb923c', banana: '#facc15' };
const TREE_BUTTONS = {  // tree-card button -> preset id (harvest / prune depend on the crop)
  tree360: () => 'tree360', leaf_scan: () => 'leaf_scan', trunk_scan: () => 'trunk_scan',
  fruit_scout: () => 'fruit_scout', harvest: (crop) => `harvest_${crop}`,
  prune: (crop) => (crop === 'banana' ? 'prune_banana' : 'prune'),
};
const MARK_KINDS = ['fruit', 'prune', 'rootstock_sprout', 'skirt_lift', 'malformed_panicle', 'sanitation_deleaf',
  'sucker', 'bunch'];

function el(tag, attrs = {}, ...kids) {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === 'class') e.className = v;
    else if (k === 'text') e.textContent = v;
    else if (k.startsWith('on')) e.addEventListener(k.slice(2), v);
    else if (v !== undefined && v !== null && v !== false) e.setAttribute(k, v === true ? '' : v);
  }
  for (const c of kids) if (c !== null && c !== undefined) e.append(c);
  return e;
}

const fmtS = (s) => {
  if (num(s) === null) return t('na');
  const v = Math.max(0, Math.round(s));
  return v >= 60 ? `${Math.floor(v / 60)}:${String(v % 60).padStart(2, '0')}` : `${v} s`;
};
const pct = (v) => (num(v) === null ? t('na') : `${Math.round(v)}%`);
const label = (tile) => (getLang() === 'hi' ? (tile.hi || tile.tile) : (tile.en || tile.tile));

export class OrchardConsole {
  constructor(send, toast) {
    this.send = send;
    this.toast = toast;
    this.map = null;
    this.status = null;
    this.coverage = null;
    this.mast = null;
    this.recovery = null;
    this.records = new Map();
    this.marks = [];
    this.robot = null;
    this.selected = [];        // tree ids (tree card shows the first)
    this.lasso = false;
    this.drag = null;
    this.readOnly = false;
    this.view = { scale: 20, ox: 0, oy: 0 };
    this._bind();
    this._markDialogOptions();
  }

  // ---------------------------------------------------------------- inputs
  onMessage(msg) {
    switch (msg.type) {
      case 'map': this.map = msg; this._fitMap(); this._rowChips(); this._jobRows(); this.drawMap(); this.renderTree(); return true;
      case 'preset_status': this.status = msg; this.renderPresetBar(); this.renderQueue(); this.renderChips();
        this.renderAssist(); this.renderTree(); this.drawMap(); return true;
      case 'coverage': this.coverage = msg; this.renderProgress(); this.drawMap(); return true;
      case 'mast': this.mast = msg; this.renderCrown(); return true;
      case 'recovery': this.recovery = msg; this.renderRecovery(); return true;
      case 'tree_record': this.records.set(msg.tree_id, msg.record); this.renderTree(); this.drawMap(); return true;
      case 'mark': this.marks.push(msg.mark); if (this.marks.length > 500) this.marks.shift();
        this.renderMarks(); this.drawMap(); return true;
      case 'jobsheet': this.onJobSheet(msg); return true;
      default: return false;
    }
  }

  onState(s) {
    const tel = s.telemetry;
    if (tel && num(tel.x) !== null) {
      const moved = !this.robot || Math.hypot(tel.x - this.robot.x, tel.y - this.robot.y) > 0.02
        || Math.abs(tel.heading_deg - this.robot.h) > 2;
      this.robot = { x: tel.x, y: tel.y, h: tel.heading_deg };
      if (moved) this.drawMap();
    }
  }

  setReadOnly(ro) { this.readOnly = ro; this.renderPresetBar(); this.renderTree(); }

  relabel() {
    this.renderPresetBar(); this.renderTree(); this.renderQueue(); this.renderProgress(); this.renderCrown();
    this.renderRecovery(); this.renderChips(); this.renderMarks(); this._markDialogOptions(); this.drawMap();
  }

  // ---------------------------------------------------------------- top chips
  renderChips() {
    const box = $('chips');
    const chips = this.status?.chips ?? [];
    box.replaceChildren(...chips.map((c) => el('span', {
      class: `chip ${c.level}`, title: `${c.id}: ${c.value}`,
      text: c.id === 'blocked' ? `${t('blocked')}: ${reason(String(c.value).split(':')[0])}`
        : `${t(c.id)} ${c.id === 'soc' ? `${c.value}%` : c.id === 'wind' ? `${c.value} m/s`
          : c.id === 'slope' ? `${c.value}°` : c.value}`,
    })));
  }

  // ---------------------------------------------------------------- preset bar
  tileOf(id) { return (this.status?.tiles ?? []).find((x) => x.id === id) ?? null; }

  renderPresetBar() {
    const bar = $('presetbar');
    const tiles = this.status?.tiles ?? [];
    bar.replaceChildren(...tiles.filter((x) => x.kind !== 'recovery' || ['safe_park', 'return_home'].includes(x.id))
      .map((tile) => {
        const why = tile.enabled ? (tile.level === 'degraded' ? reason(tile.reason) : '') : reason(tile.reason);
        const b = el('button', {
          class: `tile ${tile.level ?? ''} ${tile.live ? '' : 'notlive'}`, 'data-op': '1',
          title: `${tile.en ?? tile.id} · ${tile.executor}${tile.live ? '' : ` (${t('not_live')})`}`
            + `${tile.season?.label ? `\n${tile.season.label}` : ''}${tile.chips?.length ? `\n${tile.chips.join('\n')}` : ''}`,
          onclick: () => this.startPreset(tile.id),
        }, el('span', { class: 't-hi', text: label(tile) }), el('span', { class: 't-en', text: getLang() === 'hi' ? (tile.tile ?? '') : (tile.hi ?? '') }),
        why ? el('span', { class: 't-why', text: why }) : null);
        b.disabled = this.readOnly || !tile.enabled;
        return b;
      }));
  }

  /** Start a preset on the current selection (tree presets need trees; recovery / dock / work run without). */
  startPreset(id, confirmed = false) {
    const tile = this.tileOf(id);
    if (!tile) { this.toast(`${id}: ${reason('preset_unknown')}`, false); return; }
    const needsTrees = !['recovery', 'work', 'dock'].includes(tile.kind);
    let targets = {};
    if (needsTrees) {
      if (!this.selected.length) { this.toast(t('select_tree'), false); this.setTab('map'); return; }
      targets = { tree_ids: this.selected.slice(0, 32) };
    }
    if (tile.level === 'degraded' && !confirmed) {
      if (!confirm(`${label(tile)}: ${t('confirm_degraded')}\n${(tile.reasons ?? []).map(reason).join(', ')}`)) return;
      confirmed = true;
    }
    this.send({ type: 'preset_start', preset: id, targets, params: {}, confirm: confirmed });
  }

  // ---------------------------------------------------------------- map
  _bind() {
    const c = $('map');
    c.addEventListener('pointerdown', (e) => this._down(e));
    c.addEventListener('pointermove', (e) => this._move(e));
    c.addEventListener('pointerup', (e) => this._up(e));
    window.addEventListener('resize', () => { this._fitMap(); this.drawMap(); });
    $('btn-lasso').addEventListener('click', () => {
      this.lasso = !this.lasso;
      $('btn-lasso').classList.toggle('on', this.lasso);
      $('map-note').textContent = this.lasso ? t('lasso_hint') : '';
    });
    $('btn-clear-sel').addEventListener('click', () => this.select([]));
    for (const b of document.querySelectorAll('[data-tc]')) {
      b.addEventListener('click', () => this._treeButton(b.dataset.tc));
    }
    $('btn-pause').addEventListener('click', () => this.send({ type: 'preset_pause' }));
    $('btn-resume').addEventListener('click', () => this.send({ type: 'preset_resume' }));
    $('btn-abort').addEventListener('click', () => this.send({ type: 'preset_abort' }));
    for (const b of document.querySelectorAll('[data-crown]')) {
      b.addEventListener('click', () => this.send({ type: 'crown', band: b.dataset.crown }));
    }
    for (const b of document.querySelectorAll('[data-rec]')) {
      b.addEventListener('click', () => this.send({ type: 'recovery', op: b.dataset.rec }));
    }
    for (const b of document.querySelectorAll('[data-assist]')) {
      b.addEventListener('click', () => { this.send({ type: 'assist_reply', reply: b.dataset.assist }); $('assist-dialog').close(); });
    }
    $('btn-drop-local').addEventListener('click', () => this.send({ type: 'link_test', link: 'local', dur_s: 6 }));
    $('btn-job-hi').addEventListener('click', () => this._jobRequest('hi'));
    $('btn-job-en').addEventListener('click', () => this._jobRequest('en'));
    $('mark-form').addEventListener('submit', (e) => {
      if (e.submitter?.value !== 'save' || !this.selected.length) return;
      const h = $('mark-h').value;
      this.send({ type: 'tree_mark', tree_id: this.selected[0], kind: $('mark-kind').value,
        clock: Number($('mark-clock').value), h_class: h });
    });
  }

  setTab(tab) { document.dispatchEvent(new CustomEvent('nandi-tab', { detail: tab })); }

  _fitMap() {
    const c = $('map');
    const r = c.getBoundingClientRect();
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    if (r.width > 0) { c.width = Math.round(r.width * dpr); c.height = Math.round(r.height * dpr); }
    const pts = [...(this.map?.geofence ?? []), ...(this.map?.trees ?? []).map((x) => x.xy)];
    if (!pts.length) return;
    const xs = pts.map((p) => p[0]); const ys = pts.map((p) => p[1]);
    const [x0, x1, y0, y1] = [Math.min(...xs), Math.max(...xs), Math.min(...ys), Math.max(...ys)];
    const pad = 12 * dpr;
    const s = Math.min((c.width - 2 * pad) / Math.max(1, x1 - x0), (c.height - 2 * pad) / Math.max(1, y1 - y0));
    this.view = { scale: s, ox: pad - x0 * s + ((c.width - 2 * pad) - (x1 - x0) * s) / 2,
      oy: c.height - pad + y0 * s - ((c.height - 2 * pad) - (y1 - y0) * s) / 2, dpr };
  }

  _px(p) { return [this.view.ox + p[0] * this.view.scale, this.view.oy - p[1] * this.view.scale]; }

  _world(e) {
    const c = $('map'); const r = c.getBoundingClientRect(); const d = this.view.dpr || 1;
    const x = (e.clientX - r.left) * d; const y = (e.clientY - r.top) * d;
    return [(x - this.view.ox) / this.view.scale, (this.view.oy - y) / this.view.scale];
  }

  _down(e) {
    if (!this.map) return;
    const w = this._world(e);
    if (this.lasso) { this.drag = { a: w, b: w }; $('map').setPointerCapture(e.pointerId); return; }
    let best = null; let bd = 1e9;
    for (const tr of this.map.trees) {
      const d = Math.hypot(tr.xy[0] - w[0], tr.xy[1] - w[1]);
      if (d < Math.max(0.8, tr.canopy_r_m ?? 0.6) && d < bd) { best = tr; bd = d; }
    }
    if (best) { this.select([best.tree_id]); this.setTab('tree'); }
  }

  _move(e) { if (this.drag) { this.drag.b = this._world(e); this.drawMap(); } }

  _up() {
    if (!this.drag) return;
    const { a, b } = this.drag; this.drag = null;
    const [x0, x1, y0, y1] = [Math.min(a[0], b[0]), Math.max(a[0], b[0]), Math.min(a[1], b[1]), Math.max(a[1], b[1])];
    this.select(this.map.trees.filter((tr) => tr.xy[0] >= x0 && tr.xy[0] <= x1 && tr.xy[1] >= y0 && tr.xy[1] <= y1)
      .map((tr) => tr.tree_id));
  }

  select(ids) {
    this.selected = [...new Set(ids)];
    const n = this.selected.length;
    $('sel-summary').textContent = n ? `${t('selected')}: ${this.selected.map((id) => this._tree(id)?.tag ?? id).join(', ')}` : '';
    this._rowChips();
    this.renderTree();
    this.drawMap();
  }

  _tree(id) { return this.map?.trees.find((x) => x.tree_id === id) ?? null; }

  _rowChips() {
    const box = $('row-chips');
    const rows = this.map?.rows ?? []; const blocks = this.map?.blocks ?? [];
    const sel = new Set(this.selected);
    box.replaceChildren(
      ...rows.map((r) => el('button', { class: `chip ${r.trees.length && r.trees.every((x) => sel.has(x)) ? 'sel' : ''}`,
        text: `${t('row')} ${r.row_id}`, onclick: () => this.select(r.trees) })),
      ...blocks.map((b) => el('button', { class: 'chip', text: `${t('block')} ${b.block_id}`, onclick: () => this.select(b.trees) })),
    );
    if (!this.map?.trees?.length) $('map-note').textContent = t('no_registry');
  }

  drawMap() {
    const c = $('map'); const g = c.getContext('2d');
    if (!c.width) return;
    g.clearRect(0, 0, c.width, c.height);
    const m = this.map;
    if (!m || !m.trees.length) {
      g.fillStyle = '#9fb0a0'; g.font = `${14 * (this.view.dpr || 1)}px sans-serif`;
      g.fillText(t('no_registry'), 12, 28); return;
    }
    const d = this.view.dpr || 1; const s = this.view.scale;
    const line = (pts, colour, w = 1, dash = []) => {
      if (!pts || pts.length < 2) return;
      g.beginPath(); g.setLineDash(dash); g.strokeStyle = colour; g.lineWidth = w * d;
      pts.forEach((p, i) => { const [x, y] = this._px(p); if (i) g.lineTo(x, y); else g.moveTo(x, y); });
      g.stroke(); g.setLineDash([]);
    };
    if (m.geofence.length) line([...m.geofence, m.geofence[0]], '#ef4444', 1.5, [6 * d, 4 * d]);
    for (const lat of m.laterals) line(lat.points, '#38bdf8', 1.2);
    for (const r of m.ropes) line(r.points, '#e5e7eb', 1, [2 * d, 2 * d]);
    const activeTrees = new Set(this.status?.active?.tree_ids ?? []);
    const sel = new Set(this.selected);
    const cov = this.coverage?.live ? this.coverage : null;
    for (const tr of m.trees) {
      const [x, y] = this._px(tr.xy);
      const r = Math.max(4 * d, (tr.canopy_r_m ?? 0.6) * s);
      if (sel.has(tr.tree_id) || activeTrees.has(tr.tree_id)) {  // TREE 360 orbit annulus r1
        const a = m.annuli?.[tr.tree_id];
        if (a) {
          g.beginPath(); g.setLineDash([4 * d, 4 * d]); g.strokeStyle = a.clear ? '#a3e635' : '#f97316'; g.lineWidth = 1.2 * d;
          g.arc(x, y, a.r1 * s, 0, Math.PI * 2); g.stroke(); g.setLineDash([]);
        }
      }
      g.beginPath(); g.fillStyle = CROP_COLOUR[tr.crop] ?? '#22c55e'; g.globalAlpha = 0.55;
      g.arc(x, y, r, 0, Math.PI * 2); g.fill(); g.globalAlpha = 1;
      if (sel.has(tr.tree_id)) { g.lineWidth = 3 * d; g.strokeStyle = '#ffffff'; g.stroke(); }
      if (activeTrees.has(tr.tree_id)) { g.lineWidth = 2 * d; g.strokeStyle = '#38bdf8'; g.stroke(); }
      if (cov && cov.tree_id === tr.tree_id) {  // live coverage donut (low band)
        const f = Math.min(1, (num(cov.coverage_pct?.low) ?? 0) / 100);
        g.beginPath(); g.strokeStyle = '#38bdf8'; g.lineWidth = 4 * d;
        g.arc(x, y, r + 5 * d, -Math.PI / 2, -Math.PI / 2 + f * Math.PI * 2); g.stroke();
      }
      g.fillStyle = '#fff'; g.font = `600 ${11 * d}px sans-serif`; g.textAlign = 'center';
      g.fillText(tr.tag, x, y + 4 * d);
      const nm = this.marks.filter((mk) => mk.tree_id === tr.tree_id).length;
      if (nm) { g.fillStyle = '#facc15'; g.fillText(`✕${nm}`, x + r, y - r); }
      if (this.records.has(tr.tree_id)) { g.fillStyle = '#a7f3d0'; g.fillText('•', x - r, y - r); }
    }
    if (m.home) { const [x, y] = this._px(m.home); g.fillStyle = '#e5e7eb'; g.fillRect(x - 5 * d, y - 5 * d, 10 * d, 10 * d); }
    if (this.robot) {
      const [x, y] = this._px([this.robot.x, this.robot.y]); const h = -(this.robot.h * Math.PI) / 180;
      g.save(); g.translate(x, y); g.rotate(h); g.beginPath(); g.fillStyle = '#ef4444';
      g.moveTo(10 * d, 0); g.lineTo(-7 * d, 6 * d); g.lineTo(-7 * d, -6 * d); g.closePath(); g.fill(); g.restore();
    }
    if (this.drag) {
      const [ax, ay] = this._px(this.drag.a); const [bx, by] = this._px(this.drag.b);
      g.strokeStyle = '#fff'; g.setLineDash([4 * d, 3 * d]); g.lineWidth = d; g.strokeRect(ax, ay, bx - ax, by - ay); g.setLineDash([]);
    }
  }

  // ---------------------------------------------------------------- tree card
  _treeButton(kind) {
    const tr = this._tree(this.selected[0]);
    if (!tr) { this.toast(t('select_tree'), false); return; }
    if (kind === 'mark') { $('mark-dialog').showModal(); return; }
    this.startPreset(TREE_BUTTONS[kind](tr.crop));
  }

  renderTree() {
    const tr = this._tree(this.selected[0]);
    $('tc-title').textContent = tr ? `${t('tree')} ${tr.tag} · ${tr.crop ?? ''} ${tr.cultivar ?? ''}` : t('select_tree');
    const facts = $('tc-facts');
    const strips = $('tc-strips');
    if (!tr) { facts.replaceChildren(); strips.replaceChildren(); $('tc-reason').textContent = ''; }
    else {
      const rows = [[t('row'), tr.row_id], [t('block'), tr.block_id], [t('height'), tr.height_m && `${tr.height_m} m`],
        [t('canopy'), tr.canopy_r_m && `r ${tr.canopy_r_m} m`], ['r1', this.map?.annuli?.[tr.tree_id]?.r1 && `${this.map.annuli[tr.tree_id].r1} m`],
        [t('marks'), String(this.marks.filter((mk) => mk.tree_id === tr.tree_id).length)]];
      facts.replaceChildren(...rows.flatMap(([k, v]) => [el('dt', { text: k }), el('dd', { text: v ?? t('na') })]));
    }
    const why = [];
    for (const b of document.querySelectorAll('[data-tc]')) {
      if (b.dataset.tc === 'mark') { b.disabled = this.readOnly || !tr; continue; }
      const id = tr ? TREE_BUTTONS[b.dataset.tc](tr.crop) : null;
      const tile = id ? this.tileOf(id) : null;
      const applies = tr && tile && (!tile.crops?.length || tile.crops.includes(tr.crop));
      b.disabled = this.readOnly || !applies || !tile.enabled;
      if (tr && tile && !tile.enabled) why.push(`${b.textContent}: ${reason(tile.reason)}`);
      if (tr && tile && tile.enabled && !tile.live) why.push(`${b.textContent}: ${t('not_live')}`);
      if (tr && !applies) why.push(`${b.textContent}: ${reason('not_in_mvp')}`);
    }
    $('tc-reason').textContent = why.join(' · ');
    const rec = tr ? this.records.get(tr.tree_id) : null;
    const box = $('tc-record');
    if (!rec) { box.textContent = t('no_record'); if (tr) strips.replaceChildren(); return; }
    const cov = rec.coverage?.az_deg_by_band ?? rec.coverage?.bands_by_side ?? null;
    const parts = [`${t('last_visit')}: ${rec.visit_t ?? t('na')} · ${(rec.presets ?? []).join(', ')}`];
    if (cov) parts.push(`${t('coverage')}: ${JSON.stringify(cov)}`);
    if (rec.coverage?.mode) parts.push(`mode ${rec.coverage.mode}, masked ${rec.coverage.masked_pct ?? t('na')}%`);
    parts.push(`${t('leads')}: ${rec.symptoms_n ?? 0}`);
    if (rec.fruit) parts.push(`${t('fruit')}: ${JSON.stringify(rec.fruit.visible_by_stage ?? rec.fruit).slice(0, 120)}`);
    if (rec.harvest) parts.push(`${t('btn_harvest')}: ${rec.harvest.picked_n ?? 0} / ${t('marks')} ${rec.harvest.marked_n ?? 0}`);
    parts.push(`${t('next_actions')}: ${(rec.next_actions ?? []).join(', ') || '-'}`);
    box.replaceChildren(...parts.map((p) => el('div', { text: p })));
    strips.replaceChildren(...(rec.media?.strips ?? []).map((s) => {
      const img = el('img', { src: s.url, alt: s.band, loading: 'lazy' });
      img.onerror = () => img.parentElement?.remove();
      return el('div', {}, el('span', { text: t(`band_${s.band}`) }), img);
    }));
  }

  _markDialogOptions() {
    const fill = (id, opts) => $(id).replaceChildren(...opts.map(([v, txt]) => el('option', { value: v, text: txt })));
    fill('mark-kind', MARK_KINDS.map((k) => [k, t(`kind_${k}`)]));
    fill('mark-clock', Array.from({ length: 12 }, (_, i) => [String(i + 1), String(i + 1)]));
    fill('mark-h', ['neeche', 'beech', 'upar'].map((h) => [h, t(`h_${h}`)]));
  }

  // ---------------------------------------------------------------- queue, chips, bands, crown
  renderQueue() {
    const st = this.status;
    const a = st?.active;
    const card = $('active-card');
    if (a) {
      card.replaceChildren(el('strong', { text: `${t('running')}: ${label(this.tileOf(a.preset) ?? { tile: a.preset })}` }),
        el('div', { class: 'small', text: `${(a.tree_ids ?? []).map((id) => this._tree(id)?.tag ?? id).join(', ')} · ${a.state}`
          + `${a.paused ? ` · ${t('paused_t')}` : ''} · ${t('elapsed')} ${fmtS(a.elapsed_s)} · ${a.executor}`
          + `${a.live ? '' : ` (${t('not_live')})`}` }));
    } else {
      card.replaceChildren(el('span', { class: 'muted', text: st?.paused ? `${t('queue_paused')}: ${reason(st.paused_reason)}` : t('queue_empty') }));
    }
    $('btn-pause').disabled = this.readOnly || !a || a.paused;
    $('btn-resume').disabled = this.readOnly || !((a && (a.paused || st?.relay_hold)) || (!a && (st?.paused || st?.relay_hold)));
    $('btn-abort').disabled = this.readOnly || !a;
    const q = st?.queue?.items ?? [];
    const order = q.map((it) => it.qid);
    $('queue-list').replaceChildren(...q.map((it, i) => {
      const move = (d) => {
        const o = order.slice(); const j = i + d; if (j < 0 || j >= o.length) return;
        [o[i], o[j]] = [o[j], o[i]]; this.send({ type: 'queue_reorder', order: o });
      };
      const tile = this.tileOf(it.preset) ?? { tile: it.preset };
      return el('li', {}, el('div', { text: `${label(tile)} · ${(it.tree_ids ?? []).map((id) => this._tree(id)?.tag ?? id).join(', ')}` }),
        el('div', { class: 'small muted', text: `${t('eta')} ${fmtS(it.eta_start_s)} → ${fmtS(it.eta_end_s)} · ${t('soc_end')} ${pct((it.soc_end ?? NaN) * 100)}`
          + ` · ${t('disk_end')} ${num(it.disk_free_end_mb) === null ? t('na') : `${Math.round(it.disk_free_end_mb)} MB`}`
          + `${it.blocked?.length ? ` · ${t('blocked')}: ${it.blocked.map(reason).join(', ')}` : ''}` }),
        el('div', { class: 'row' }, el('button', { 'data-op': '1', text: t('up'), onclick: () => move(-1) }),
          el('button', { 'data-op': '1', text: t('down'), onclick: () => move(1) }),
          el('button', { 'data-op': '1', class: 'warn', text: t('remove'), onclick: () => this.send({ type: 'preset_abort', qid: it.qid }) })));
    }));
    $('history-list').replaceChildren(...(st?.history ?? []).map((h) => el('li', {
      text: `${label(this.tileOf(h.preset) ?? { tile: h.preset })} · ${(h.tree_ids ?? []).map((id) => this._tree(id)?.tag ?? id).join(', ')}`
        + ` · ${reason(h.status)}${h.reason ? ` (${reason(h.reason)})` : ''} · ${fmtS(h.dur_s)}`,
    })));
    for (const b of document.querySelectorAll('#queue-list [data-op]')) b.disabled = this.readOnly;
  }

  renderProgress() {
    const c = this.coverage;
    const chips = [];
    if (c?.live) {
      chips.push(['ok', `${t('tree')} ${this._tree(c.tree_id)?.tag ?? c.tree_id}`]);
      chips.push([c.phase === 'aborted' ? 'fail' : 'ok', `${t('phase')}: ${c.phase}`]);
      if (c.stations) chips.push(['ok', `${t('station')} ${c.station ?? 0}/${c.stations}`]);
      if (num(c.r_err_m) !== null) chips.push([c.r_err_m > 0.15 ? 'warn' : 'ok', `${t('r_err')} ${c.r_err_m.toFixed(2)} m`]);
      if (num(c.reentries) !== null) chips.push([c.reentries > 1 ? 'fail' : c.reentries ? 'warn' : 'ok', `${t('reentries')} ${c.reentries}`]);
      if (c.mode) chips.push(['ok', c.mode]);
      if (c.paused) chips.push(['warn', t('paused_t')]);
      if (c.abort_reason) chips.push(['fail', reason(c.abort_reason)]);
    }
    $('progress-chips').replaceChildren(...chips.map(([lv, txt]) => el('span', { class: `chip ${lv}`, text: txt })));
    $('band-strips').replaceChildren(...['low', 'mid', 'high'].map((b) => {
      const v = num(c?.coverage_pct?.[b]);
      const fill = el('span', { class: v === null ? '' : v >= 90 ? 'good' : v >= 50 ? 'amber' : 'bad' });
      fill.style.width = `${Math.max(0, Math.min(100, v ?? 0))}%`;
      return el('div', { class: 'band' }, el('span', { text: t(`band_${b}`) }), el('span', { class: 'bar' }, fill),
        el('span', { text: pct(v) }));
    }));
  }

  renderCrown() {
    const m = this.mast;
    const fill = $('crown-fill');
    const ext = num(m?.ext_m);
    fill.style.height = `${Math.max(0, Math.min(1, (ext ?? 0) / 1.6)) * 100}%`;
    fill.classList.toggle('held', Boolean(m?.gated || m?.held_by || m?.braked));
    const rows = [[t('ext'), ext === null ? t('na') : `${ext.toFixed(2)} m (${t(m.band) ?? m.band})`],
      [t('gate'), m?.gated ? `${t('gate_held')}: ${reason(m.reason)} ${m.detail ?? ''}` : (m?.live ? t('gate_ok') : t('na'))],
      [t('arbiter'), m?.arbiter_source ?? '-'],
      [t('wind'), num(m?.wind?.speed_mps) === null ? t('na') : `${m.wind.speed_mps.toFixed(1)} m/s${m.wind.gust ? ' (gust)' : ''}`],
      ['margin', num(m?.gates?.margin_m) === null ? t('na') : `${m.gates.margin_m.toFixed(3)} m`]];
    $('crown-facts').replaceChildren(...rows.flatMap(([k, v]) => [el('dt', { text: k }), el('dd', { text: v })]));
  }

  // ---------------------------------------------------------------- recovery and assist
  renderRecovery() {
    const r = this.recovery;
    const links = r?.links ?? {};
    const lk = (k) => {
      const l = links[k] ?? {};
      return `${l.state ?? 'unknown'}${l.sim ? ` (${t('sim_handheld')})` : ''}${l.needs_confirm ? ' · Resume/Halt' : ''}`;
    };
    const rows = [[t('link_local'), lk('local')], [t('link_cloud'), lk('cloud')],
      [t('recovery'), r?.live ? (r.armed ? t('armed') : t('not_armed')) : t('na')],
      [t('active'), r?.active ? `${r.active} · ${r.step ?? ''}${r.outcome ? ` → ${r.outcome}` : ''}` : '-']];
    if (r?.confirms && Object.keys(r.confirms).length) rows.push(['confirms', JSON.stringify(r.confirms)]);
    $('rec-facts').replaceChildren(...rows.flatMap(([k, v]) => [el('dt', { text: k }), el('dd', { text: v })]));
    const p = $('rec-prompt');
    p.hidden = !r?.prompt;
    p.textContent = r?.prompt ?? '';
    const checks = r?.fallen_checks ?? [];
    $('fallen-checks').replaceChildren(...checks.map((c) => el('button', {
      'data-op': '1', class: r?.checklist?.[c] ? 'on' : '', text: t(`chk_${c}`),
      onclick: () => this.send({ type: 'recovery', op: 'check', item: c }),
    })));
    for (const b of document.querySelectorAll('#fallen-checks [data-op]')) b.disabled = this.readOnly || r?.active !== 'fallen';
  }

  renderAssist() {
    const a = this.status?.assist;
    const dlg = $('assist-dialog');
    if (a && !dlg.open && !this.readOnly) {
      $('assist-text').textContent = `${a.source}: ${reason(a.reason)}${a.tree_id ? ` · ${this._tree(a.tree_id)?.tag ?? a.tree_id}` : ''}`;
      dlg.showModal();
    } else if (!a && dlg.open) dlg.close();
  }

  // ---------------------------------------------------------------- jobs
  _jobRows() {
    const rows = this.map?.rows ?? [];
    $('job-row').replaceChildren(...rows.map((r) => el('option', { value: r.row_id, text: `${t('row')} ${r.row_id}` })));
  }

  _jobRequest(lang) {
    const row = $('job-row').value;
    if (!row) return;
    this._jobLang = lang;
    this.send({ type: 'jobsheet_request', row_id: row, lang });
  }

  onJobSheet(msg) {
    const s = msg.sheet;
    const hi = msg.lang !== 'en';
    const cols = hi ? s.columns_hi : s.columns_en;
    const rows = (s.lines ?? []).map((ln) => el('tr', {}, ...[ln.line_no, ln.physical_tag,
      `${hi ? ln.side_hi : ln.side} / ${ln.clock}`, hi ? ln.h_class_hi : ln.h_class_en, hi ? ln.job_hi : ln.job_en,
      hi ? ln.tool_hi : ln.tool_en, hi ? ln.notes_hi : ln.notes_en].map((v) => el('td', { text: String(v ?? '') }))));
    $('job-preview').replaceChildren(el('h3', { text: hi ? s.title_hi : s.title_en }),
      rows.length ? el('table', {}, el('tr', {}, ...cols.map((c) => el('th', { text: c }))), ...rows) : el('p', { text: t('no_jobs') }));
    // Printable HTML is always Hindi first, English second (WP9 job_sheet.render_html); the voice list is Hindi.
    const blob = new Blob([msg.html], { type: 'text/html;charset=utf-8' });
    const a = el('a', { href: URL.createObjectURL(blob), download: `jobsheet_${msg.row_id}${hi ? '.hi' : '.en'}.html` });
    document.body.append(a); a.click(); a.remove();
    setTimeout(() => URL.revokeObjectURL(a.href), 10000);
  }

  renderMarks() {
    $('mark-list').replaceChildren(...this.marks.slice(-60).reverse().map((m) => el('li', {
      text: `${m.physical_tag ?? m.tree_id} · ${t(`kind_${m.kind}`)} · ${m.clock} · ${t(`h_${m.h_class}`)}${m.reason ? ` · ${reason(m.reason)}` : ''}`,
    })));
  }
}
