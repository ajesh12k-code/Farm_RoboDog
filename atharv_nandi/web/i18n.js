// Console labels, Hindi first (OS 5.5), English second. Elements carry data-i18n="<key>"; t(key) for dynamic text.
// HINDI: machine-drafted for the pilot farm crew, needs native review (field pilot operators) before any release.
// tests/test_web_layout.py checks that every key used by the page and the scripts exists in both languages.
export const HI_REVIEW_NOTE = 'हिन्दी: मशीन से लिखा मसौदा, स्थानीय समीक्षा बाकी · Hindi: machine-drafted, needs native review';

export const STRINGS = {
  hi: {
    console: 'कंसोल',
    disconnected: 'सिम से कनेक्शन टूटा - दोबारा जोड़ रहे हैं...',
    fallen: 'रोबोट गिर गया है - वापसी मेनू देखें',
    estop: 'आपात रोक',
    estop_reset: 'आपात रोक हटाएँ',
    estop_latched: 'आपात रोक लगी है',
    estop_confirm: 'आपात रोक हटाएँ? रोबोट फिर से चल सकता है।',
    crops: 'फसल', falls: 'गिरना', time: 'समय', gait: 'चाल', arm: 'भुजा', pos: 'स्थिति', tilt: 'झुकाव', link: 'लिंक',
    battery: 'बैटरी',
    mission: 'मिशन', start_zone_a: 'ज़ोन A शुरू करें', abort: 'रद्द करें', state: 'अवस्था', progress: 'प्रगति',
    picked: 'तोड़ी गई', basket: 'टोकरी', detections: 'पहचान', run: 'रन', camera: 'कैमरा',
    hold_hint: 'चलने के लिए दबाए रखें, एक कदम के लिए टैप करें',
    controls: 'नियंत्रण', harvest: 'कटाई', switch_gait: 'चाल बदलें', release_arm: 'भुजा छोड़ें', reset: 'रीसेट',
    envelope: 'पहुँच क्षेत्र', follow: 'पीछा करें', arm_hint: '(स्लाइडर हिलाने से मैनुअल नियंत्रण)',
    keys_help: 'Space आपात रोक · X चाल रोकें · WASD चलें · Q/E मुड़ें · H कटाई · G चाल · R रीसेट · Esc भुजा छोड़ें · V क्षेत्र · F पीछा',
    auth_title: 'कंसोल टोकन', auth_prompt: 'इस कंसोल के लिए ऑपरेटर या दर्शक टोकन चाहिए।', sign_in: 'साइन इन',
    auth_bad: 'टोकन अस्वीकार - फिर कोशिश करें।',
    dev_badge: 'DEV: बिना प्रमाणीकरण', operator: 'ऑपरेटर', viewer: 'दर्शक (केवल देखें)',
    viewer_refused: 'दर्शक केवल देख सकते हैं (आपात रोक काम करता है)',
    na: 'उपलब्ध नहीं', not_connected: 'सिम से कनेक्ट नहीं',
    stop_t: 'रुकने का समय', stop_dist: 'दूरी',
    // views
    view_map: 'नक्शा', view_3d: '3D', view_tree: 'पेड़', view_queue: 'कतार', view_recovery: 'वापसी', view_jobs: 'काम',
    view_drive: 'चलाएँ',
    // preset bar / tree card
    presets: 'काम चुनें', select_tree: 'नक्शे पर पेड़ चुनें', no_registry: 'इस दुनिया में बाग़ का रजिस्टर नहीं (legacy)',
    tree: 'पेड़', crop: 'फसल', row: 'कतार', block: 'खंड', height: 'ऊँचाई', canopy: 'छतरी', last_visit: 'पिछली जाँच',
    never: 'कभी नहीं', coverage: 'कवरेज', leads: 'संकेत (पुष्टि बाकी)', fruit: 'फल', next_actions: 'अगले कदम',
    btn_360: '360', btn_leaf: 'पत्ती', btn_trunk: 'तना', btn_scout: 'फल गिनती', btn_harvest: 'तुड़ाई', btn_prune: 'छँटाई',
    btn_mark: 'निशान', start: 'शुरू करें', queued: 'कतार में', not_live: 'इस स्टैक में नहीं चलता',
    live: 'चालू', confirm_degraded: 'चेतावनी है। फिर भी शुरू करें?', strips: 'फ़िल्टर की हुई पट्टियाँ',
    no_record: 'अभी कोई रिकॉर्ड नहीं', lasso: 'घेरें', lasso_hint: 'नक्शे पर खींचकर पेड़ घेरें', clear: 'साफ़ करें',
    selected: 'चुने गए', rows: 'कतारें', marks: 'निशान', season: 'मौसम',
    // queue / progress
    running: 'चल रहा है', pause: 'रोकें', resume: 'फिर चलाएँ', queue_empty: 'कतार खाली है', eta: 'समय', soc_end: 'अंत में बैटरी',
    disk_end: 'अंत में जगह', blocked: 'रुका हुआ', remove: 'हटाएँ', up: 'ऊपर', down: 'नीचे', history: 'पिछले काम',
    queue_paused: 'कतार रुकी है', phase: 'चरण', station: 'स्टेशन', r_err: 'त्रिज्या त्रुटि', reentries: 'दोबारा प्रवेश',
    bands: 'पट्टियाँ', band_low: 'नीचे', band_mid: 'बीच', band_high: 'ऊपर', crown: 'मुकुट मस्तूल', ext: 'ऊँचाई',
    gate: 'गेट', wind: 'हवा', arbiter: 'आदेश स्रोत', stow: 'नीचे करें', mid: 'बीच', high: 'ऊपर',
    gate_ok: 'ठीक', gate_held: 'रोका गया', paused_t: 'रुका हुआ', elapsed: 'बीता समय',
    // recovery
    recovery: 'वापसी मेनू', safe_park: 'सुरक्षित रुको', return_home: 'घर लौटो', abort_recovery: 'वापसी रद्द',
    confirm_operator: 'ऑपरेटर पुष्टि (आपात रोक)', link_resume: 'फिर चलाएँ', link_halt: 'रुके रहो',
    link_local: 'स्थानीय सुरक्षा लिंक', link_cloud: 'कंसोल लिंक', sim_handheld: 'सिम हैंडहेल्ड', armed: 'सक्रिय',
    not_armed: 'निष्क्रिय', fallen_checklist: 'गिरने के बाद जाँच सूची', drop_local: 'सिम: लिंक 6 s काटें',
    local_confirm_note: 'स्थानीय पुष्टि केवल हैंडहेल्ड से (कंसोल से नहीं)', prompt: 'संदेश', active: 'सक्रिय काम',
    chk_crown_lock: 'मुकुट लॉक', chk_lift_points: 'उठाने के बिंदु', chk_two_people: 'दो लोग', chk_laterals_tools: 'पाइप और औज़ार',
    chk_mast_inspected: 'मस्तूल जाँचा', chk_person_clear: 'लोग दूर',
    // assist
    assist_title: 'सहायता चाहिए', take_over: 'नियंत्रण लें', skip: 'छोड़ें', halt: 'रोकें',
    // jobs
    jobsheet: 'काम की सूची', download_hi: 'सूची डाउनलोड (हिन्दी)', download_en: 'डाउनलोड (English)', no_jobs: 'कोई काम नहीं',
    mark_title: 'निशान लगाएँ', mark_kind: 'काम', clock: 'घड़ी', h_class: 'ऊँचाई', save: 'सहेजें', cancel: 'रद्द',
    h_neeche: 'नीचे', h_beech: 'बीच', h_upar: 'ऊपर', tool_hint: 'औज़ार',
    kind_fruit: 'फल', kind_prune: 'छँटाई', kind_rootstock_sprout: 'जड़ की कोंपल', kind_skirt_lift: 'नीचे की डालें',
    kind_malformed_panicle: 'विकृत बौर', kind_sanitation_deleaf: 'सूखी पत्ती', kind_sucker: 'पुत्ती', kind_bunch: 'गुच्छा',
    chips: 'स्थिति', soc: 'बैटरी', rtk: 'RTK', slope: 'ढलान',
  },
  en: {
    console: 'Console',
    disconnected: 'Disconnected from the sim - reconnecting...',
    fallen: 'Robot has fallen - see the Recovery menu',
    estop: 'E-STOP',
    estop_reset: 'Reset E-STOP',
    estop_latched: 'E-STOP latched',
    estop_confirm: 'Clear the E-STOP latch? The robot may move again.',
    crops: 'Crops', falls: 'Falls', time: 'Time', gait: 'Gait', arm: 'Arm', pos: 'Pos', tilt: 'Tilt', link: 'Link',
    battery: 'Battery',
    mission: 'Mission', start_zone_a: 'Start zone A', abort: 'Abort', state: 'State', progress: 'Progress',
    picked: 'Picked', basket: 'Basket', detections: 'Detections', run: 'Run', camera: 'Camera',
    hold_hint: 'Hold to walk, tap for one step',
    controls: 'Controls', harvest: 'Harvest', switch_gait: 'Switch Gait', release_arm: 'Release Arm', reset: 'Reset',
    envelope: 'Envelope', follow: 'Follow', arm_hint: '(moving a slider takes manual control)',
    keys_help: 'Space E-STOP · X stop gait · WASD move · Q/E turn · H harvest · G gait · R reset · Esc release arm · V envelope · F follow',
    auth_title: 'Console token', auth_prompt: 'This console requires an operator or viewer token.', sign_in: 'Sign in',
    auth_bad: 'Token rejected - try again.',
    dev_badge: 'DEV: no auth', operator: 'operator', viewer: 'viewer (read-only)',
    viewer_refused: 'Viewer role is read-only (E-STOP still works)',
    na: 'n/a', not_connected: 'Not connected to the sim',
    stop_t: 'stop', stop_dist: 'dist',
    view_map: 'Map', view_3d: '3D', view_tree: 'Tree', view_queue: 'Queue', view_recovery: 'Recovery', view_jobs: 'Jobs',
    view_drive: 'Drive',
    presets: 'Presets', select_tree: 'Pick a tree on the map', no_registry: 'No orchard registry in this world (legacy)',
    tree: 'Tree', crop: 'Crop', row: 'Row', block: 'Block', height: 'Height', canopy: 'Canopy', last_visit: 'Last visit',
    never: 'never', coverage: 'Coverage', leads: 'Leads (not confirmed)', fruit: 'Fruit', next_actions: 'Next actions',
    btn_360: '360', btn_leaf: 'Leaf', btn_trunk: 'Trunk', btn_scout: 'Scout', btn_harvest: 'Harvest', btn_prune: 'Prune',
    btn_mark: 'Mark', start: 'Start', queued: 'queued', not_live: 'not run by this stack',
    live: 'live', confirm_degraded: 'Warnings present. Start anyway?', strips: 'Filtered strips',
    no_record: 'No record yet', lasso: 'Lasso', lasso_hint: 'Drag on the map to select trees', clear: 'Clear',
    selected: 'Selected', rows: 'Rows', marks: 'Marks', season: 'Season',
    running: 'Running', pause: 'Pause', resume: 'Resume', queue_empty: 'Queue is empty', eta: 'ETA', soc_end: 'SoC at end',
    disk_end: 'Disk at end', blocked: 'Blocked', remove: 'Remove', up: 'Up', down: 'Down', history: 'History',
    queue_paused: 'Queue paused', phase: 'Phase', station: 'Station', r_err: 'Radius error', reentries: 'Re-entries',
    bands: 'Bands', band_low: 'low', band_mid: 'mid', band_high: 'high', crown: 'Crown mast', ext: 'Extension',
    gate: 'Gate', wind: 'Wind', arbiter: 'Arbiter source', stow: 'Stow', mid: 'Mid', high: 'High',
    gate_ok: 'ok', gate_held: 'held', paused_t: 'paused', elapsed: 'Elapsed',
    recovery: 'Recovery menu', safe_park: 'SAFE PARK', return_home: 'RETURN HOME', abort_recovery: 'Abort recovery',
    confirm_operator: 'Operator confirm (E-STOP)', link_resume: 'Resume', link_halt: 'Halt',
    link_local: 'Local safety link', link_cloud: 'Console link', sim_handheld: 'sim handheld', armed: 'armed',
    not_armed: 'not armed', fallen_checklist: 'Fallen checklist', drop_local: 'Sim: drop link 6 s',
    local_confirm_note: 'The local confirm comes only from the handheld (never the console)', prompt: 'Prompt',
    active: 'Active',
    chk_crown_lock: 'Crown lock', chk_lift_points: 'Lift points', chk_two_people: 'Two people',
    chk_laterals_tools: 'Laterals and tools', chk_mast_inspected: 'Mast inspected', chk_person_clear: 'People clear',
    assist_title: 'Assist needed', take_over: 'Take over', skip: 'Skip', halt: 'Halt',
    jobsheet: 'Job sheet', download_hi: 'Download (Hindi)', download_en: 'Download (English)', no_jobs: 'No jobs',
    mark_title: 'Add a MARK', mark_kind: 'Job', clock: 'Clock', h_class: 'Height', save: 'Save', cancel: 'Cancel',
    h_neeche: 'low (< 0.8 m)', h_beech: 'middle', h_upar: 'high (> 1.8 m)', tool_hint: 'Tool',
    kind_fruit: 'fruit', kind_prune: 'prune', kind_rootstock_sprout: 'rootstock sprout', kind_skirt_lift: 'skirt lift',
    kind_malformed_panicle: 'malformed panicle', kind_sanitation_deleaf: 'sanitation de-leaf', kind_sucker: 'sucker',
    kind_bunch: 'bunch',
    chips: 'Status', soc: 'SoC', rtk: 'RTK', slope: 'Slope',
  },
};

// MISSION-PRESETS 0.5 reason codes + the WP27 / WP34 extensions the Ready check emits: [hi, en].
export const REASONS = {
  out_of_reach_marked: ['पहुँच से बाहर, निशान लगाया', 'out of reach, marked'],
  orbit_blocked: ['चक्कर का रास्ता बंद', 'orbit blocked'], range_lost: ['दूरी नहीं मिली', 'range lost'],
  low_light: ['रोशनी कम', 'low light'], disk_low: ['जगह कम', 'disk low'], payload_full: ['टोकरी भरी', 'payload full'],
  tool_unavailable: ['औज़ार नहीं', 'tool unavailable'], rtk: ['RTK फ़िक्स नहीं', 'no RTK fix'],
  energy_reserve: ['बैटरी आरक्षित से कम', 'energy reserve'], stability: ['स्थिरता ठीक नहीं', 'stability'],
  crown_gate: ['मुकुट गेट बंद', 'crown gate'], crown_clearance: ['मुकुट के लिए जगह कम', 'crown clearance'],
  wind: ['हवा तेज़', 'wind'], slope: ['ढलान ज़्यादा', 'slope'], geofence: ['सीमा से बाहर', 'geofence'],
  person_hold: ['व्यक्ति पास है', 'person nearby'], animal_hold: ['जानवर पास है', 'animal nearby'],
  local_link_lost: ['स्थानीय लिंक नहीं', 'local link lost'], cloud_link_lost: ['कंसोल लिंक नहीं', 'console link lost'],
  stuck: ['फँसा', 'stuck'], sink: ['धँसा', 'sink'], over_capacity: ['डाल मोटी', 'over capacity'],
  paste_required: ['बोर्डो पेस्ट लगाएँ', 'paste required'], twist_failed: ['मरोड़ विफल', 'twist failed'],
  no_clearance: ['जगह नहीं', 'no clearance'], bagged: ['फल थैली में', 'bagged'], leaf_minimum: ['पत्ती कम', 'leaf minimum'],
  season: ['मौसम नहीं', 'season'], rain: ['बारिश', 'rain'], wet_foliage: ['पत्ते गीले', 'wet foliage'], heat: ['गर्मी', 'heat'],
  self_right_failed: ['खुद नहीं उठा', 'self-right failed'], registry_mismatch: ['रजिस्टर मेल नहीं', 'registry mismatch'],
  not_in_mvp: ['MVP में नहीं', 'not in MVP'], novnc_actors: ['noVNC चालू है', 'noVNC with actors'],
  param_invalid: ['गलत मान', 'invalid parameter'], preset_unknown: ['अनजान काम', 'unknown preset'],
  job_sheet_missing: ['काम की सूची नहीं', 'no job sheet'], tag_missing: ['टैग नहीं', 'no tag'],
  sensor_fault: ['सेंसर खराब', 'sensor fault'], confirm_required: ['पुष्टि चाहिए', 'confirm required'],
  estop_not_engaged: ['आपात रोक लगी नहीं', 'e-stop not engaged'], not_fallen: ['गिरा नहीं', 'not fallen'],
  queue_full: ['कतार भरी', 'queue full'], not_checked: ['जाँच नहीं हुई', 'not checked'], estop: ['आपात रोक', 'e-stop'],
  no_live_executor: ['इस स्टैक में चलाने वाला नहीं', 'no live executor in this stack'],
  aborted: ['रद्द', 'aborted'], done: ['पूरा', 'done'], refused: ['अस्वीकार', 'refused'], failed: ['विफल', 'failed'],
  operator: ['ऑपरेटर', 'operator'], assist_skip: ['छोड़ा', 'skipped'], assist_halt: ['रोका', 'halted'],
};

const STORE_KEY = 'nandi.lang';
let lang = 'hi';  // Hindi first (OS 5.5); English on request
try { lang = localStorage.getItem(STORE_KEY) === 'en' ? 'en' : 'hi'; } catch { /* private mode */ }

export const getLang = () => lang;

/** Label for key in the current language (falls back to Hindi, then English, then the key itself). */
export function t(key) {
  return STRINGS[lang][key] ?? STRINGS.hi[key] ?? STRINGS.en[key] ?? key;
}

/** A reason code in the current language ("code" stays visible for unknown codes). */
export function reason(code) {
  if (!code) return '';
  const base = String(code).split(':')[0];
  const r = REASONS[base];
  return r ? (lang === 'hi' ? r[0] : r[1]) : String(code);
}

export function applyI18n(root = document) {
  document.documentElement.lang = lang;
  for (const el of root.querySelectorAll('[data-i18n]')) {
    const text = t(el.dataset.i18n);
    if (el.textContent !== text) el.textContent = text;
  }
}

export function setLang(next) {
  lang = next === 'en' ? 'en' : 'hi';
  try { localStorage.setItem(STORE_KEY, lang); } catch { /* private mode */ }
  applyI18n();
}

export function toggleLang() {
  setLang(lang === 'hi' ? 'en' : 'hi');
  return lang;
}
