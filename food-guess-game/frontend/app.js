/* Chef Mystico — the front end.
 *
 * Three screens. The welcome page is pure black; the arena plays either game
 * mode over a looping video; the info screen holds About, Help and Future
 * scope. Every move between them is animated: an iris opens into the arena,
 * modes and pages cross-dissolve, questions glide out and in, and a lifting
 * cloche unveils every dish. All game logic lives on the server;
 * this file only asks it questions (through api.js) and stages the answers.
 */
(() => {
  'use strict';

  const { call, waitFor } = window.Mystico;
  // sound.js plays the clicks and hovers itself; these are the game's own moments
  const sound = window.MysticoSound || { press() {}, land() {}, boom() {}, wrong() {} };

  // ─────────────────────────────── helpers ───────────────────────────────
  const $ = (selector, root = document) => root.querySelector(selector);
  const $$ = (selector, root = document) => [...root.querySelectorAll(selector)];
  const EASE = 'cubic-bezier(0.16, 1, 0.3, 1)';
  const EASE_IN = 'cubic-bezier(0.55, 0, 1, 0.45)';
  const EASE_IN_OUT = 'cubic-bezier(0.65, 0, 0.35, 1)';
  const motion = matchMedia('(prefers-reduced-motion: reduce)');
  const calm = () => motion.matches;
  const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, calm() ? 0 : ms));
  const pad = (n) => String(n).padStart(2, '0');
  const esc = (value) => String(value).replace(/[&<>"']/g, (ch) => (
    { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[ch]
  ));

  // Run a Web Animation, then keep its end state as inline style and drop it,
  // so finished animations do not pile up on an element.
  function play(node, frames, options = {}) {
    if (!node || typeof node.animate !== 'function') return Promise.resolve();
    const settings = { duration: 450, easing: EASE, fill: 'forwards', ...options };
    if (calm()) Object.assign(settings, { duration: 1, delay: 0 });
    const animation = node.animate(frames, settings);
    return animation.finished.then(() => {
      try { animation.commitStyles(); } catch (err) { /* node left the page */ }
      animation.cancel();
    }, () => {});
  }

  function clearMotion(...nodes) {
    for (const node of nodes) {
      if (!node) continue;
      for (const prop of ['opacity', 'transform', 'translate', 'filter', 'clipPath']) node.style[prop] = '';
    }
  }

  // Staggered upward entrance for whatever in `root` is marked data-enter.
  // It moves `translate`, not `transform`, so buttons keep their CSS hover
  // and press transforms afterwards.
  const rise = (px) => [{ opacity: 0, translate: `0 ${px}px` }, { opacity: 1, translate: '0 0' }];
  function entrance(root, delay = 0) {
    $$('[data-enter]', root)
      .filter((node) => node.getClientRects().length > 0)
      .forEach((node, i) => play(node, rise(22), { duration: 760, delay: delay + i * 70, fill: 'both' }));
  }

  // Count a number up (or down) in place.
  function tween(node, to, width = 0) {
    if (!node) return;
    const target = String(to);
    const from = Number(node.dataset.value ?? node.textContent) || 0;
    node.dataset.value = target;
    const format = (v) => String(Math.round(v)).padStart(width, '0');
    if (calm() || !Number.isFinite(from) || from === to) { node.textContent = format(to); return; }
    const start = performance.now();
    const step = (now) => {
      if (node.dataset.value !== target) return;      // a newer count took over
      const t = Math.min(1, (now - start) / 700);
      node.textContent = format(from + (to - from) * (1 - Math.pow(1 - t, 4)));
      if (t < 1) requestAnimationFrame(step);
    };
    requestAnimationFrame(step);
  }

  function clientId() {
    try {
      let id = localStorage.getItem('mystico-client');
      if (!id) {
        id = Math.random().toString(36).slice(2) + Date.now().toString(36);
        localStorage.setItem('mystico-client', id);
      }
      return id;
    } catch (err) {
      return null;   // private windows may refuse storage; nothing depends on it
    }
  }

  function bump(node) {
    node.classList.remove('bump');
    void node.offsetWidth;
    node.classList.add('bump');
  }

  // ─────────────────────────────── elements ──────────────────────────────
  const W = {
    root: $('#welcome'),
    status: $('#welcome-status'),
    statusText: $('#welcome-status-text'),
    emblem: $('#emblem'),
  };
  const I = {
    root: $('#info'),
    tabs: $('#info-tabs'),
    pages: { about: $('#page-about'), help: $('#page-help'), future: $('#page-future') },
  };
  const A = {
    root: $('#arena'),
    video: $('#arena-video'),
    modeSwitch: $('#mode-switch'),
    scoreboard: $('#scoreboard'),
    panes: { forward: $('#pane-forward'), reverse: $('#pane-reverse') },
  };
  const telemetryFields = (root) => Object.fromEntries($$('[data-f]', root).map((n) => [n.dataset.f, n]));
  const F = {
    stage: $('#fw-stage'),
    bar: $('#fw-bar'),
    t: telemetryFields($('#fw-telemetry')),
    answers: $('#fw-answers'),
    form: $('#fw-form'),
    input: $('#fw-input'),
    after: $('#fw-after'),
    giveup: $('#fw-giveup'),
    ready: $('#fw-ready'),
    keys: $('#pane-forward .keys'),
  };
  const R = {
    stage: $('#rv-stage'),
    bar: $('#rv-bar'),
    t: telemetryFields($('#rv-telemetry')),
    form: $('#rv-form'),
    input: $('#rv-input'),
    feedback: $('#rv-feedback'),
    hint: $('#rv-hint'),
    giveup: $('#rv-giveup'),
    ready: $('#rv-ready'),
    foot: $('#rv-foot'),
    ledger: $('#rv-ledger'),
    ledgerEmpty: $('#rv-ledger-empty'),
  };
  const V = {
    root: $('#reveal'),
    eyebrow: $('#reveal-eyebrow'),
    glow: $('#reveal-glow'),
    card: $('#dish-card'),
    kicker: $('#dish-kicker'),
    name: $('#dish-name'),
    meta: $('#dish-meta'),
    ingredients: $('#dish-ingredients'),
    fact: $('#dish-fact'),
    badge: $('#dish-badge'),
    confidence: $('#dish-confidence'),
    lid: $('#cloche-lid'),
    plate: $('#cloche-plate'),
    note: $('#reveal-note'),
    review: $('#reveal-review'),
    actions: $('#reveal-actions'),
  };

  const score = { chef: 0, you: 0, points: 0 };

  // ───────────────────────────── the stage ───────────────────────────────
  // One item on stage at a time. The old one glides up and fades; the new one
  // slides up into its place. Both share one grid cell, so nothing jumps.
  const stageReadyAt = new WeakMap();   // when the next item may start to enter

  // Send whatever is on stage up and away. Called the moment the player
  // answers, so the question leaves at once rather than after the server replies.
  function dismissStage(stage) {
    const outgoing = $$(':scope > .stage-item:not(.leaving)', stage);
    for (const node of outgoing) {
      node.classList.add('leaving');
      node.setAttribute('aria-hidden', 'true');
      play(node,
        [{ opacity: 1, transform: 'translateY(0)', filter: 'blur(0px)' },
         { opacity: 0, transform: 'translateY(-44px)', filter: 'blur(8px)' }],
        { duration: 440, easing: EASE }).then(() => node.remove());
    }
    if (outgoing.length) stageReadyAt.set(stage, performance.now() + (calm() ? 0 : 210));
  }

  async function swapStage(stage, html) {
    dismissStage(stage);
    const incoming = document.createElement('div');
    incoming.className = 'stage-item';
    incoming.innerHTML = html;
    stage.appendChild(incoming);
    const delay = Math.max(0, (stageReadyAt.get(stage) || 0) - performance.now());
    await play(incoming,
      [{ opacity: 0, transform: 'translateY(34px)' }, { opacity: 1, transform: 'translateY(0)' }],
      { duration: 640, delay, fill: 'both' });
    return incoming;
  }

  const thinking = (words) =>
    `<p class="thinking">${esc(words)}<span class="dots"><i>.</i><i>.</i><i>.</i></span></p>`;

  // ─────────────────────── welcome: the emblem lives ─────────────────────
  const parallax = { x: 0, y: 0, tx: 0, ty: 0, running: false };
  const layers = $$('.layer', W.emblem);

  const dust = { t: 0, x: 0, y: 0 };
  window.addEventListener('pointermove', (event) => {
    parallax.tx = (event.clientX / innerWidth - 0.5) * 2;
    parallax.ty = (event.clientY / innerHeight - 0.5) * 2;

    // a trail of gold stardust behind the pointer, on the welcome page only
    if (view.screen !== 'welcome' || calm() || event.pointerType !== 'mouse') return;
    const now = performance.now();
    if (now - dust.t < 40 || Math.hypot(event.clientX - dust.x, event.clientY - dust.y) < 14) return;
    Object.assign(dust, { t: now, x: event.clientX, y: event.clientY });
    if (document.querySelectorAll('.stardust').length > 24) return;
    const speck = document.createElement('span');
    speck.className = 'stardust';
    speck.style.left = `${event.clientX}px`;
    speck.style.top = `${event.clientY}px`;
    document.body.appendChild(speck);
    const drift = (Math.random() - 0.5) * 30;
    const size = 0.5 + Math.random() * 0.8;
    vanish(speck.animate([
      { opacity: 0.95, transform: `translate(0, 0) scale(${size}) rotate(0deg)` },
      { opacity: 0, transform: `translate(${drift}px, ${22 + Math.random() * 26}px) scale(0.15) rotate(130deg)` },
    ], { duration: 900 + Math.random() * 500, easing: EASE }), speck);
  }, { passive: true });

  function startParallax() {
    if (parallax.running || calm()) return;
    parallax.running = true;
    const frame = () => {
      if (view.screen !== 'welcome') { parallax.running = false; return; }
      parallax.x += (parallax.tx - parallax.x) * 0.06;
      parallax.y += (parallax.ty - parallax.y) * 0.06;
      for (const layer of layers) {
        const depth = Number(layer.dataset.depth);
        layer.style.transform = `translate(${(parallax.x * depth).toFixed(2)}px, ${(parallax.y * depth * 0.7).toFixed(2)}px)`;
      }
      requestAnimationFrame(frame);
    };
    requestAnimationFrame(frame);
  }

  // The two welcome buttons lean toward the pointer.
  for (const button of $$('.magnetic')) {
    button.addEventListener('pointermove', (event) => {
      if (calm()) return;
      const box = button.getBoundingClientRect();
      const dx = event.clientX - (box.left + box.width / 2);
      const dy = event.clientY - (box.top + box.height / 2);
      button.style.translate = `${(dx * 0.12).toFixed(1)}px ${(dy * 0.22).toFixed(1)}px`;
    });
    button.addEventListener('pointerleave', () => { button.style.translate = ''; });
  }

  // ─────────────────────────────── the chef ──────────────────────────────
  // The emblem is a puppet and these are its moves. The welcome page has one
  // chef; the arena has a copy of him, the host, who talks to the player.
  const SVG_NS = 'http://www.w3.org/2000/svg';

  function vanish(animation, node) {
    animation.finished.then(() => node.remove(), () => node.remove());
  }

  // A four-pointed star centred on (cx, cy).
  function starPath(cx, cy, size) {
    const a = (0.28 * size).toFixed(2);
    const b = (0.72 * size).toFixed(2);
    return `M${cx} ${cy - size} l${a} ${b} l${b} ${a} l-${b} ${a} l-${a} ${b} l-${a} -${b} l-${b} -${a} l${b} -${a} z`;
  }

  function makeChef(svg) {
    const part = (name) => svg.querySelector(`.${name}`);
    const p = {
      magic: part('magic'), lift: part('cloche-lift'), arm: part('chef-arm'), halo: part('halo'),
      glint: part('glint'), eyesOpen: part('eyes-open'), eyesClosed: part('eyes-closed'),
      head: part('chef-head'), mustache: part('mustache'),
    };
    let lastSpell = 0;
    // Nothing to animate when motion is reduced or the chef is off screen.
    const still = () => calm() || document.hidden || svg.getClientRects().length === 0;

    function eyes(duration) {
      p.eyesOpen.animate([{ opacity: 0 }, { opacity: 1, offset: 0.1 }, { opacity: 1, offset: 0.86 }, { opacity: 0 }], { duration });
      p.eyesClosed.animate([{ opacity: 1 }, { opacity: 0, offset: 0.1 }, { opacity: 0, offset: 0.86 }, { opacity: 1 }], { duration });
    }

    // Gold dust rising off the cloche.
    function mote() {
      if (still()) return;
      const dot = document.createElementNS(SVG_NS, 'circle');
      dot.setAttribute('class', 'mote');
      dot.setAttribute('cx', (224 + Math.random() * 36).toFixed(1));
      dot.setAttribute('cy', (62 + Math.random() * 28).toFixed(1));
      dot.setAttribute('r', (0.5 + Math.random() * 1.1).toFixed(2));
      p.magic.appendChild(dot);
      const drift = ((Math.random() - 0.5) * 22).toFixed(1);
      const rise = (28 + Math.random() * 36).toFixed(1);
      vanish(dot.animate([
        { opacity: 0, transform: 'translate(0px, 0px)' },
        { opacity: 0.9, offset: 0.25 },
        { opacity: 0, transform: `translate(${drift}px, -${rise}px)` },
      ], { duration: 2200 + Math.random() * 1600, easing: 'cubic-bezier(0.2, 0.6, 0.4, 1)' }), dot);
    }

    // The spell: eyes open, the arm flourishes, the cloche lifts and turns,
    // a glint runs over the dome and stars burst from it.
    function spell() {
      if (still()) return;
      const now = performance.now();
      if (now - lastSpell < 1500) return;
      lastSpell = now;
      p.lift.animate([
        { transform: 'translateY(0) rotate(0deg)' },
        { transform: 'translateY(-18px) rotate(-7deg)', offset: 0.32 },
        { transform: 'translateY(-14px) rotate(5deg)', offset: 0.62 },
        { transform: 'translateY(0) rotate(0deg)' },
      ], { duration: 1800, easing: EASE_IN_OUT });
      p.arm.animate([
        { transform: 'rotate(0deg)' }, { transform: 'rotate(-5deg)', offset: 0.3 },
        { transform: 'rotate(2deg)', offset: 0.62 }, { transform: 'rotate(0deg)' },
      ], { duration: 1600, easing: 'ease-in-out' });
      p.halo.animate([{ transform: 'scale(1)' }, { transform: 'scale(1.8)', offset: 0.35 }, { transform: 'scale(1)' }],
        { duration: 1800, easing: EASE });
      p.glint.animate([
        { strokeDashoffset: 0.12, opacity: 0 }, { opacity: 1, offset: 0.2 }, { strokeDashoffset: -0.95, opacity: 0 },
      ], { duration: 1200, delay: 260, easing: 'ease-in-out' });
      eyes(1700);
      for (let i = 0; i < 10; i++) {
        const star = document.createElementNS(SVG_NS, 'path');
        star.setAttribute('class', 'magic-star');
        star.setAttribute('d', starPath(242, 72, 2 + Math.random() * 2.6));
        star.style.transformBox = 'fill-box';
        star.style.transformOrigin = 'center';
        p.magic.appendChild(star);
        const angle = (Math.PI * 2 * i) / 10 + Math.random() * 0.5;
        const reach = 26 + Math.random() * 32;
        vanish(star.animate([
          { opacity: 0, transform: 'translate(0px, 0px) scale(0.3) rotate(0deg)' },
          { opacity: 1, offset: 0.2 },
          { opacity: 0, transform: `translate(${(Math.cos(angle) * reach).toFixed(1)}px, ${(Math.sin(angle) * reach).toFixed(1)}px) scale(1.1) rotate(100deg)` },
        ], { duration: 1100 + Math.random() * 600, delay: 300, easing: EASE, fill: 'backwards' }), star);
      }
      for (let i = 0; i < 6; i++) setTimeout(mote, 250 + i * 70);
    }

    // Speaking: the mustache moves with the words and the eyes open.
    function talk(duration = 1200, gesture = true) {
      if (still()) return;
      const beats = Math.max(2, Math.round(duration / 230));
      const frames = [];
      for (let i = 0; i < beats; i++) {
        frames.push({ transform: 'translateY(0px) scaleY(1)' }, { transform: 'translateY(1.5px) scaleY(0.76)' });
      }
      frames.push({ transform: 'translateY(0px) scaleY(1)' });
      p.mustache.animate(frames, { duration, easing: 'ease-in-out' });
      eyes(duration + 400);
      if (gesture) {
        p.arm.animate([
          { transform: 'rotate(0deg)' }, { transform: 'rotate(-4deg)', offset: 0.35 },
          { transform: 'rotate(1.5deg)', offset: 0.7 }, { transform: 'rotate(0deg)' },
        ], { duration: Math.min(1500, duration + 300), easing: 'ease-in-out' });
      }
    }

    const head = (frames, duration) => { if (!still()) p.head.animate(frames, { duration, easing: 'ease-in-out' }); };
    return {
      mote,
      spell,
      talk,
      nod: () => head([
        { transform: 'translateY(0) rotate(0deg)' }, { transform: 'translateY(3px) rotate(2deg)', offset: 0.3 },
        { transform: 'translateY(0) rotate(0deg)', offset: 0.55 }, { transform: 'translateY(2px) rotate(1deg)', offset: 0.75 },
        { transform: 'translateY(0) rotate(0deg)' },
      ], 720),
      shake: () => head([0, -7, 6, -4, 2, 0].map((deg) => ({ transform: `rotate(${deg}deg)` })), 760),
      bow: () => {
        head([
          { transform: 'translateY(0) rotate(0deg)' }, { transform: 'translateY(4px) rotate(9deg)', offset: 0.35 },
          { transform: 'translateY(4px) rotate(9deg)', offset: 0.7 }, { transform: 'translateY(0) rotate(0deg)' },
        ], 1900);
        if (!still()) {
          p.arm.animate([
            { transform: 'rotate(0deg)' }, { transform: 'rotate(12deg)', offset: 0.35 },
            { transform: 'rotate(12deg)', offset: 0.7 }, { transform: 'rotate(0deg)' },
          ], { duration: 1900, easing: 'ease-in-out' });
        }
      },
      blink: () => { if (!still()) eyes(900); },
      think: (on) => svg.classList.toggle('thinking', on),
    };
  }

  // The welcome chef: dust all the time, a spell every few seconds, and one
  // whenever the pointer reaches him or he is clicked.
  const welcomeChef = makeChef(W.emblem);
  setInterval(() => welcomeChef.mote(), 380);
  setInterval(() => welcomeChef.spell(), 7000);
  setTimeout(() => welcomeChef.spell(), 3200);
  W.emblem.addEventListener('pointerenter', () => welcomeChef.spell());
  W.emblem.addEventListener('click', () => welcomeChef.spell());

  // The host: a copy of the welcome emblem, with its own ids so its masks and
  // gradients still resolve while the welcome page is hidden.
  function copyEmblem(mount) {
    mount.innerHTML = W.emblem.outerHTML
      .replace(/\sid="([^"]+)"/g, ' id="host-$1"')
      .replace(/url\(#([^)]+)\)/g, 'url(#host-$1)')
      .replace(/\saria-labelledby="[^"]*"/g, '')
      .replace(/\srole="img"/, '');
    const svg = mount.firstElementChild;
    svg.querySelector('title')?.remove();
    for (const layer of $$('.layer', svg)) layer.style.transform = '';
    return svg;
  }

  const H = {
    root: $('#host'),
    bubble: $('#host .host-bubble'),
    line: $('#host-line'),
    said: $('#host-said'),
    token: 0,
    last: 0,
    lastLine: '',
  };
  const hostChef = makeChef(copyEmblem($('#host-chef')));
  $('#host-chef').addEventListener('click', () => hostChef.spell());
  setInterval(() => hostChef.mote(), 650);
  setInterval(() => { if (performance.now() - H.last > 6500) hostChef.blink(); }, 4200);

  function pick(lines) {
    const fresh = lines.filter((line) => line !== H.lastLine);
    const line = fresh[Math.floor(Math.random() * fresh.length)] || lines[0];
    H.lastLine = line;
    return line;
  }

  // The host speaks: the bubble types the line while his mustache moves.
  // `move` is one of the chef's gestures: nod, shake, bow or spell.
  function say(text, move) {
    const token = ++H.token;
    H.last = performance.now();
    H.said.textContent = text;
    if (move) hostChef[move]();
    hostChef.talk(Math.min(2400, 300 + text.length * 22), move !== 'spell' && move !== 'bow');
    play(H.bubble, [{ opacity: 0.45, transform: 'scale(0.96)' }, { opacity: 1, transform: 'scale(1)' }], { duration: 420 });
    if (calm()) {
      H.line.textContent = text;
      return;
    }
    H.root.classList.add('typing');
    let shown = 0;
    const step = () => {
      if (token !== H.token) return;
      shown = Math.min(text.length, shown + 2);
      H.line.textContent = text.slice(0, shown);
      if (shown < text.length) setTimeout(step, 28);
      else H.root.classList.remove('typing');
    };
    step();
  }

  // Every game request goes through here, so the chef can look thoughtful
  // whenever the server takes a moment.
  async function ask(path, body) {
    const timer = setTimeout(() => hostChef.think(true), 260);
    try {
      return await call(path, body);
    } finally {
      clearTimeout(timer);
      hostChef.think(false);
    }
  }

  // A sleeping free server can take up to a minute to answer its first call.
  window.addEventListener('mystico:waking', () => {
    W.statusText.textContent = 'Waking the kitchen — the first visit can take up to a minute';
    W.status.hidden = false;
    if (ARENA.has(view.screen)) say('The kitchen is waking up… give me a moment.');
  });
  window.addEventListener('mystico:awake', () => { W.status.hidden = true; });

  let welcomeWaiting = false;
  let dishCount = 350;
  async function loadWelcome() {
    try {
      const health = await call('/api/health');
      dishCount = health.dishes;
      for (const node of $$('[data-live="dishes"]')) node.textContent = health.dishes;
      for (const node of $$('[data-live="questions"]')) node.textContent = health.questions;
      W.status.hidden = true;
      loadMenu();
    } catch (err) {
      W.statusText.textContent = 'Server offline — waiting for the kitchen';
      W.status.hidden = false;
      if (err.offline && !welcomeWaiting) {
        welcomeWaiting = true;
        waitFor(() => { welcomeWaiting = false; loadWelcome(); });
      }
    }
  }

  let menuLoaded = false;
  async function loadMenu() {
    if (menuLoaded) return;
    try {
      const menu = await call('/api/predict/dishes?limit=1000');
      const list = $('#dish-list');
      list.replaceChildren(...menu.dishes.map((name) => Object.assign(document.createElement('option'), { value: name })));
      menuLoaded = true;
    } catch (err) {
      // suggestions are a convenience; the game works without them
    }
  }

  // ───────────────────────────── navigation ──────────────────────────────
  // Screens: the welcome page, the arena in either mode, and the three pages.
  const view = { screen: 'welcome', busy: false };
  const ARENA = new Set(['forward', 'reverse']);
  const PAGES = new Set(['about', 'help', 'future']);
  const fromHash = () => {
    const name = location.hash.slice(1);
    return ARENA.has(name) || PAGES.has(name) ? name : 'welcome';
  };

  function remember(screen) {
    const hash = screen === 'welcome' ? '' : `#${screen}`;
    if (location.hash === hash) return;
    try {
      history.pushState(null, '', hash || location.pathname + location.search);
    } catch (err) {
      location.hash = hash || '#';   // file:// pages may refuse pushState
    }
  }

  async function go(target, origin) {
    if (view.busy || target === view.screen) return;
    view.busy = true;
    const from = view.screen;
    view.screen = target;
    remember(target);
    try {
      if (target === 'welcome') {
        if (ARENA.has(from)) await leaveArena();
        else await leaveInfo();
      } else if (ARENA.has(target)) {
        if (ARENA.has(from)) await switchMode(from, target);
        else await enterArena(target, origin, from === 'welcome' ? W.root : I.root);
      } else if (PAGES.has(from)) {
        await switchPage(from, target);
      } else {
        if (ARENA.has(from)) await leaveArena();
        await enterInfo(target);
      }
    } finally {
      view.busy = false;
    }
    if (fromHash() !== view.screen) go(fromHash());
  }
  window.addEventListener('popstate', () => go(fromHash()));
  window.addEventListener('hashchange', () => go(fromHash()));

  function centreOf(node) {
    if (!node) return { x: innerWidth / 2, y: innerHeight / 2 };
    const box = node.getBoundingClientRect();
    return { x: box.left + box.width / 2, y: box.top + box.height / 2 };
  }

  function startVideo() {
    const video = A.video;
    if (!video) return;
    const show = () => video.classList.add('ready');
    if (video.readyState >= 2) show();
    else video.addEventListener('loadeddata', show, { once: true });
    video.play().catch(() => {});
  }
  if (A.video) {
    const sources = $$('source', A.video);
    sources[sources.length - 1]?.addEventListener('error', () => A.video.remove());
  }

  function showMode(mode) {
    A.modeSwitch.dataset.mode = mode;
    $('.panes', A.root).dataset.mode = mode;
    for (const tab of $$('[role="tab"]', A.modeSwitch)) {
      tab.setAttribute('aria-selected', String(tab.dataset.mode === mode));
    }
    renderScoreboard();
  }

  // Welcome (or a page) → arena: an iris opens from the button that was pressed.
  async function enterArena(mode, origin, behind = W.root) {
    const { x, y } = centreOf(origin);
    const radius = Math.hypot(Math.max(x, innerWidth - x), Math.max(y, innerHeight - y));
    showMode(mode);
    A.panes[mode].hidden = false;
    A.panes[mode === 'forward' ? 'reverse' : 'forward'].hidden = true;
    A.root.hidden = false;
    startVideo();
    resume(mode);
    const iris = play(A.root,
      [{ clipPath: `circle(0px at ${x}px ${y}px)` }, { clipPath: `circle(${radius}px at ${x}px ${y}px)` }],
      { duration: 860, easing: EASE_IN_OUT });
    play(behind, [{ opacity: 1, transform: 'scale(1)' }, { opacity: 0.2, transform: 'scale(0.97)' }], { duration: 860 });
    entrance(A.root, 300);
    await iris;
    clearMotion(A.root, behind);
    behind.hidden = true;
  }

  // Arena → welcome: the arena dissolves away and the black page settles back.
  async function leaveArena() {
    if (reveal.open) closeReveal();
    W.root.hidden = false;
    startParallax();
    const out = play(A.root, [{ opacity: 1, transform: 'scale(1)' }, { opacity: 0, transform: 'scale(1.02)' }],
      { duration: 450, easing: EASE_IN });
    play(W.root, [{ opacity: 0, transform: 'scale(0.97)' }, { opacity: 1, transform: 'scale(1)' }],
      { duration: 650, delay: 150, fill: 'both' });
    entrance(W.root, 260);
    await out;
    A.root.hidden = true;
    clearMotion(A.root);
    A.video?.pause();
  }

  // Welcome ↔ pages: a cross-dissolve, the incoming screen rising from 0.97.
  function showPage(page) {
    I.tabs.dataset.tab = page;
    for (const tab of $$('[role="tab"]', I.tabs)) {
      tab.setAttribute('aria-selected', String(tab.dataset.tab === page));
    }
    for (const [name, node] of Object.entries(I.pages)) node.hidden = name !== page;
  }

  async function enterInfo(page) {
    showPage(page);
    I.root.hidden = false;
    I.root.scrollTop = 0;
    const out = play(W.root, [{ opacity: 1, transform: 'scale(1)' }, { opacity: 0, transform: 'scale(0.985)' }],
      { duration: 380, easing: EASE_IN });
    const into = play(I.root, [{ opacity: 0, transform: 'scale(0.97)' }, { opacity: 1, transform: 'scale(1)' }],
      { duration: 450, delay: 120, fill: 'both' });
    entrance(I.pages[page], 220);
    await out;
    W.root.hidden = true;
    clearMotion(W.root);
    await into;
    clearMotion(I.root);
  }

  async function leaveInfo() {
    W.root.hidden = false;
    startParallax();
    const out = play(I.root, [{ opacity: 1, transform: 'scale(1)' }, { opacity: 0, transform: 'scale(1.015)' }],
      { duration: 380, easing: EASE_IN });
    play(W.root, [{ opacity: 0, transform: 'scale(0.97)' }, { opacity: 1, transform: 'scale(1)' }],
      { duration: 600, delay: 140, fill: 'both' });
    entrance(W.root, 220);
    await out;
    I.root.hidden = true;
    clearMotion(I.root);
  }

  async function switchPage(from, to) {
    showPage(to);
    const outgoing = I.pages[from];
    const incoming = I.pages[to];
    outgoing.hidden = false;   // showPage hid it; keep it for the dissolve
    I.root.scrollTo({ top: 0, behavior: calm() ? 'auto' : 'smooth' });
    const out = play(outgoing,
      [{ opacity: 1, transform: 'translateY(0)', filter: 'blur(0px)' }, { opacity: 0, transform: 'translateY(-16px)', filter: 'blur(4px)' }],
      { duration: 300, easing: EASE_IN });
    const into = play(incoming, [{ opacity: 0, transform: 'scale(0.97)' }, { opacity: 1, transform: 'scale(1)' }],
      { duration: 450, delay: 150, fill: 'both' });
    entrance(incoming, 200);
    await out;
    outgoing.hidden = true;
    clearMotion(outgoing);
    await into;
    clearMotion(incoming);
  }

  // Mode ↔ mode: cross-dissolve with the incoming pane rising from 0.97.
  async function switchMode(from, to) {
    showMode(to);
    if (reveal.open) await closeReveal();
    const outgoing = A.panes[from];
    const incoming = A.panes[to];
    incoming.hidden = false;
    resume(to);
    const out = play(outgoing,
      [{ opacity: 1, transform: 'scale(1)', filter: 'blur(0px)' }, { opacity: 0, transform: 'scale(0.985)', filter: 'blur(4px)' }],
      { duration: 300, easing: EASE_IN });
    const into = play(incoming,
      [{ opacity: 0, transform: 'scale(0.97)' }, { opacity: 1, transform: 'scale(1)' }],
      { duration: 450, delay: 120, fill: 'both' });
    entrance(incoming, 180);
    await out;
    outgoing.hidden = true;
    clearMotion(outgoing);
    await into;
    clearMotion(incoming);
  }

  // Pick a mode back up where it was left. A fresh round always starts by
  // asking the player whether they are ready.
  function resume(mode) {
    if (mode === 'forward') {
      if (['idle', 'won', 'lost'].includes(fw.phase)) fwReady();
      else if (fw.phase === 'ready') say(pick(FW_SAYS.ready));
      else if (fw.phase === 'guess') fwPresentGuess(fw.game);
      else if (fw.phase === 'asking' || fw.phase === 'asking-dish') say('Where were we… ah, yes.');
    } else if (['idle', 'done'].includes(rv.phase)) {
      rvReady();
    } else if (rv.phase === 'ready') {
      say(pick(RV_SAYS.ready));
    } else if (rv.phase === 'playing') {
      say('Back to my mystery dish…');
    }
  }

  function renderScoreboard() {
    A.scoreboard.innerHTML = A.modeSwitch.dataset.mode === 'reverse'
      ? `<span>Score <b>${score.points}</b></span>`
      : `<span>Chef <b>${score.chef}</b></span><i>·</i><span>You <b>${score.you}</b></span>`;
  }

  // ───────────────────────────── trouble ─────────────────────────────────
  const waiting = { forward: false, reverse: false };

  function offlineHtml() {
    const eyebrow = '<p class="eyebrow" style="display:flex;align-items:center;gap:10px"><span class="pulse-dot"></span>Waiting for the kitchen</p>';
    if (!window.Mystico.local) {
      return `${eyebrow}<h2 class="question">The kitchen is closed for a moment.</h2>
        <p class="help">The game server isn’t answering right now. It may be restarting — the game picks up by
        itself as soon as it is back.</p>`;
    }
    return `${eyebrow}<h2 class="question">The kitchen is dark.</h2>
      <p class="help">No game server is answering at <code>${esc(window.Mystico.server)}</code>. Start one from the
      <code>food-guess-game</code> folder and the game picks up by itself:</p>
      <p class="cmd"><code>.venv/bin/python -m uvicorn app.main:app</code></p>`;
  }

  // `retry` runs once things work again; it is always a mode's ready step.
  function trouble(mode, err, retry) {
    const P = mode === 'forward' ? F : R;
    if (mode === 'forward') { fw.phase = 'offline'; fwControls('none'); }
    else { rv.phase = 'offline'; rvView('play'); rvEnabled(false); }
    if (reveal.open && reveal.owner === mode) closeReveal();
    if (view.screen === mode) {
      say(err.offline ? (window.Mystico.local ? 'My kitchen has gone dark… start the server and I’ll be right back.'
        : 'My kitchen is closed for a moment… I’ll be right back.')
        : 'Oh dear, something slipped in my kitchen.', 'bow');
    }

    if (!err.offline) {
      swapStage(P.stage,
        `<p class="eyebrow">Something went wrong</p><h2 class="question">The kitchen stumbled.</h2>
         <p class="help">${esc(err.message)}</p>
         <button class="pill pill--outline" type="button" data-retry>Start a new round</button>`)
        .then((item) => $('[data-retry]', item)?.addEventListener('click', retry, { once: true }));
      return;
    }
    swapStage(P.stage, offlineHtml());
    if (waiting[mode]) return;
    waiting[mode] = true;
    waitFor(() => {
      waiting[mode] = false;
      loadMenu();
      loadWelcome();
      retry();
    });
  }

  function riseIn(buttons) {
    buttons.forEach((button, i) => play(button,
      [{ opacity: 0, translate: '0 10px' }, { opacity: 1, translate: '0 0' }],
      { duration: 520, delay: 180 + i * 70, fill: 'both' }));
  }

  // ─────────────────────── forward: read the mind ────────────────────────
  // phases: idle → ready → loading → asking ⇄ guess → won | asking-dish → lost
  const fw = { game: null, phase: 'idle', busy: false };

  const FW_SAYS = {
    ready: ['Close your eyes and picture a dish. Any dish, from anywhere.', 'Think of a dish… don’t tell me. I’ll find it.'],
    notYet: ['No hurry. Think of its smell, its colour, that first bite…', 'Take your time. Picture the plate in front of you…'],
    first: ['Wonderful. Now answer me honestly…', 'Good. Let us begin — answer honestly.'],
    cold: ['Hmm…', 'Tell me more…', 'Interesting…', 'Let me look deeper…', 'The mist is still thick…'],
    warm: ['I see something forming…', 'The picture is getting clearer…', 'Ah, now we’re cooking…', 'Warmer, warmer…'],
    close: ['I can almost taste it…', 'So close now…', 'It’s on the tip of my tongue…'],
    unsure: ['No matter — I’ll ask another way.', 'That’s all right. Next question…'],
    misled: ['The spirits misled me. A few more questions…', 'Hmm, not that one. Let me look again…'],
  };

  function fwControls(which) {   // 'ready' | 'answers' | 'form' | 'after' | 'none'
    F.ready.hidden = which !== 'ready';
    F.answers.hidden = which === 'ready' || which === 'form' || which === 'after';
    F.form.hidden = which !== 'form';
    F.after.hidden = which !== 'after';
    const live = which === 'answers';
    for (const button of $$('button', F.answers)) button.disabled = !live;
    F.giveup.disabled = !live;
    F.giveup.hidden = which === 'ready';
    F.keys.textContent = which === 'ready' ? 'Keys · Y · N' : 'Keys · Y · N · U';
  }

  function fwTelemetry(state) {
    const asking = state.status === 'active' && !state.guess;
    const candidates = Math.max(1, Math.round(state.effective_candidates));
    tween(F.t.q, asking ? state.questions_asked + 1 : Math.max(1, state.questions_asked), 2);
    tween(F.t.c, candidates);
    F.t.cl.textContent = candidates === 1 ? 'candidate' : 'candidates';
    tween(F.t.p, Math.round(state.confidence * 100));
    F.bar.style.width = `${Math.round(state.progress * 100)}%`;
  }

  // Before every round: does the player have a dish in mind?
  function fwReady() {
    fw.phase = 'ready';
    fw.game = null;
    fwControls('ready');
    tween(F.t.q, 1, 2);
    tween(F.t.c, dishCount);
    F.t.cl.textContent = 'candidates';
    tween(F.t.p, 0);
    F.bar.style.width = '0%';
    swapStage(F.stage, '<p class="eyebrow">Before we begin</p><h2 class="question">Do you have a dish in your mind?</h2>' +
      '<p class="whisper ready-note">Picture it clearly, and don’t tell me what it is.</p>');
    riseIn($$('button', F.ready));
    if (view.screen === 'forward') say(pick(FW_SAYS.ready));
  }

  async function fwStart() {
    if (fw.phase !== 'ready') return;
    fw.phase = 'loading';
    fwControls('none');
    swapStage(F.stage, thinking('Reading your mind'));
    try {
      const state = await ask('/api/game/mind/new', { client_id: clientId() });
      await fwApply(state, 'first');
    } catch (err) {
      trouble('forward', err, fwReady);
    }
  }

  // `why` says what led here, so the chef can react to it.
  async function fwApply(state, why = '') {
    fw.game = state;
    fwTelemetry(state);
    if (state.status !== 'active') {
      return fwAskDish(state.message || 'You beat me.', 'You beat me! Tell me — what was it?');
    }
    if (state.guess) return fwPresentGuess(state);
    fw.phase = 'asking';
    if (why === 'first') say(pick(FW_SAYS.first), 'nod');
    else if (why === 'rejected') say(pick(FW_SAYS.misled), 'shake');
    else if (why === 'unsure') say(pick(FW_SAYS.unsure));
    else say(pick(state.confidence >= 0.6 ? FW_SAYS.close : state.confidence >= 0.2 ? FW_SAYS.warm : FW_SAYS.cold));
    await swapStage(F.stage, `<h2 class="question">${esc(state.question.text)}</h2>`);
    if (fw.phase !== 'asking') return;
    fwControls('answers');
    $$('button', F.answers).forEach((button, i) => play(button,
      [{ opacity: 0.35, translate: '0 8px' }, { opacity: 1, translate: '0 0' }],
      { duration: 480, delay: i * 50, fill: 'both' }));
  }

  async function fwAnswer(answer) {
    if (fw.phase !== 'asking' || fw.busy) return;
    const button = $(`[data-answer="${answer}"]`, F.answers);
    button.classList.remove('pressed');
    void button.offsetWidth;
    button.classList.add('pressed');
    hostChef.nod();
    fw.busy = true;
    fwControls('none');
    dismissStage(F.stage);
    try {
      const state = await ask('/api/game/mind/answer',
        { session_id: fw.game.session_id, key: fw.game.question.key, answer });
      await fwApply(state, answer === 'unsure' ? 'unsure' : '');
    } catch (err) {
      trouble('forward', err, fwReady);
    } finally {
      fw.busy = false;
    }
  }

  function fwPresentGuess(state) {
    fw.phase = 'guess';
    fwControls('none');
    swapStage(F.stage, '<p class="whisper">Hold that thought.</p><h2 class="question">I believe I have it.</h2>');
    if (view.screen !== 'forward') return;   // shown when the player comes back
    const sure = state.confidence >= 0.95;
    say(sure ? 'I sense it… is this your dish?' : 'My best guess… is it this?', 'spell');
    openReveal({
      owner: 'forward',
      eyebrow: sure ? 'I sense it' : 'My best guess',
      kicker: `After ${state.questions_asked} question${state.questions_asked === 1 ? '' : 's'}`,
      dish: state.guess,
      confidence: state.confidence,
      note: 'Is this your dish?',
      actions: [
        { label: 'Yes, that’s it', style: 'white', key: 'y', run: () => fwVerdict(true) },
        { label: 'No', style: 'outline', key: 'n', run: () => fwVerdict(false) },
      ],
    });
  }

  async function fwVerdict(correct) {
    if (fw.busy || fw.phase !== 'guess') return;
    fw.busy = true;
    revealActionsEnabled(false);
    try {
      const state = await ask('/api/game/mind/verdict', { session_id: fw.game.session_id, correct });
      fw.game = state;
      if (correct) {
        fw.phase = 'won';
        score.chef += 1;
        renderScoreboard();
        bump(A.scoreboard);
        say(`Ha! ${state.message}`, 'spell');
        await updateReveal({
          eyebrow: 'Mind read',
          note: state.message,
          actions: [
            { label: 'Play again', style: 'white', run: () => closeReveal().then(fwReady) },
            { label: 'Try reverse mode', style: 'outline', run: () => closeReveal().then(() => go('reverse')) },
          ],
        });
        sparkle(V.badge.hidden ? V.name : V.badge);
      } else {
        await closeReveal({ recover: true });
        await fwApply(state, 'rejected');
      }
    } catch (err) {
      trouble('forward', err, fwReady);
    } finally {
      fw.busy = false;
    }
  }

  async function fwAskDish(message, line) {
    fw.phase = 'asking-dish';
    fwControls('none');
    say(line, 'bow');
    await swapStage(F.stage,
      `<p class="whisper">${esc(message)}</p><h2 class="question">What were you thinking of?</h2>`);
    fwControls('form');
    F.input.value = '';
    F.input.focus({ preventScroll: true });
  }

  F.form.addEventListener('submit', async (event) => {
    event.preventDefault();
    const text = F.input.value.trim();
    if (!text || fw.busy || fw.phase !== 'asking-dish') return;
    fw.busy = true;
    F.input.disabled = true;
    try {
      const state = await ask('/api/game/mind/reveal', { session_id: fw.game.session_id, actual: text });
      fw.game = state;
      fw.phase = 'lost';
      score.you += 1;
      renderScoreboard();
      bump(A.scoreboard);
      fwControls('after');
      if (state.revealed) {
        say(`Ah, ${state.revealed.name}! I’ll remember that one.`, 'nod');
        swapStage(F.stage, `<p class="whisper">${esc(state.revealed.name)}.</p><h2 class="question">You beat me.</h2>`);
        openReveal({
          owner: 'forward',
          eyebrow: 'You win — it was',
          kicker: 'Noted for next time',
          dish: state.revealed,
          note: state.message,
          review: state.review,
          actions: [
            { label: 'Play again', style: 'white', run: () => closeReveal().then(fwReady) },
            { label: 'Try reverse mode', style: 'outline', run: () => closeReveal().then(() => go('reverse')) },
          ],
        });
      } else {
        say('That one isn’t in my book yet… you win this round.', 'bow');
        await swapStage(F.stage, `<p class="whisper">${esc(state.message)}</p><h2 class="question">You beat me.</h2>`);
      }
    } catch (err) {
      trouble('forward', err, fwReady);
    } finally {
      fw.busy = false;
      F.input.disabled = false;
    }
  });

  for (const button of $$('[data-answer]', F.answers)) {
    button.addEventListener('click', () => fwAnswer(button.dataset.answer));
  }
  $('#fw-ready-yes').addEventListener('click', fwStart);
  $('#fw-ready-wait').addEventListener('click', () => {
    if (fw.phase === 'ready') say(pick(FW_SAYS.notYet));
  });
  F.giveup.addEventListener('click', () => {
    if (fw.phase === 'asking' && !fw.busy) fwAskDish('So be it.', 'So be it. What were you thinking of?');
  });
  $('[data-run="again"]', F.after).addEventListener('click', fwReady);
  $('[data-run="other"]', F.after).addEventListener('click', () => go('reverse'));

  // ──────────────────── reverse: name the chef's dish ────────────────────
  // phases: idle → ready → loading → playing → done
  const rv = { round: null, phase: 'idle', busy: false, recent: [], hero: 0 };

  const RV_SAYS = {
    ready: ['I’ve hidden a dish under my cloche…', 'I’m thinking of a dish. Can you read MY mind?'],
    first: ['Here’s your first clue. Listen carefully…', 'Your first clue. Take your time…'],
    trait: ['Here’s another hint…', 'Does this help?', 'Getting warmer?', 'Think carefully…', 'One more piece of the puzzle…'],
    clue: ['A secret from my kitchen…', 'Here’s a proper clue…'],
    broke: ['Not quite! Remember what I told you…', 'Ah-ah. Look at my hints again.'],
    correct: ['Magnifique! You read MY mind.', 'Bravo! That’s the one.'],
  };

  function rvView(which) {   // 'ready' | 'play'
    R.ready.hidden = which !== 'ready';
    R.form.hidden = which === 'ready';
    R.feedback.hidden = which === 'ready';
    R.foot.hidden = which === 'ready';
  }

  function rvEnabled(on) {
    const playing = on && rv.phase === 'playing';
    R.input.disabled = !playing;
    $('button[type="submit"]', R.form).disabled = !playing;
    R.giveup.disabled = !playing;
    R.hint.disabled = !playing || !rv.round || rv.round.hints_revealed >= rv.round.hints_total;
  }

  function rvTelemetry(state) {
    tween(R.t.h, state.hints_revealed, 2);
    tween(R.t.t, state.hints_total, 2);
    tween(R.t.c, state.candidates_left);
    R.t.cl.textContent = state.candidates_left === 1 ? 'dish fits' : 'dishes fit';
    tween(R.t.p, state.potential_score);
    R.bar.style.width = `${Math.round((100 * state.hints_revealed) / Math.max(1, state.hints_total))}%`;
  }

  function hintHtml(number, hint) {
    const label = hint.kind === 'clue' ? 'A clue from the chef' : 'A trait';
    const match = /^Unlike (.+?): (.*)$/.exec(hint.text);
    const text = match
      ? `<span class="contrast">Unlike ${esc(match[1])}:</span> ${esc(match[2])}`
      : esc(hint.text);
    return `<p class="eyebrow">${label} · ${pad(number)}</p><h2 class="hint">${text}</h2>`;
  }

  function addToLedger(number, hint) {
    R.ledgerEmpty.hidden = true;
    const item = document.createElement('li');
    item.innerHTML = `<span class="n">${pad(number)}</span>` +
      `<span class="tag ${hint.kind === 'clue' ? 'clue' : ''}">${hint.kind === 'clue' ? 'Clue' : 'Trait'}</span>` +
      `<p>${esc(hint.text)}</p>`;
    R.ledger.appendChild(item);
    play(item, [{ opacity: 0, transform: 'translateY(-14px)' }, { opacity: 1, transform: 'translateY(0)' }],
      { duration: 620, delay: 120, fill: 'both' });
    item.closest('.ledger')?.scrollTo({ top: 1e6, behavior: calm() ? 'auto' : 'smooth' });
  }

  // The newest hint takes the stage; the one it replaces moves to the ledger.
  async function rvShow(state) {
    rvTelemetry(state);
    const history = state.history || [];
    if (history.length <= rv.hero) return;
    for (let i = Math.max(0, rv.hero - 1); i < history.length - 1; i++) addToLedger(i + 1, history[i]);
    rv.hero = history.length;
    const latest = history[history.length - 1];
    const contrast = /^Unlike (.+?): /.exec(latest.text);
    if (history.length === 1) say(pick(RV_SAYS.first));
    else if (contrast) say(`That’s why it can’t be ${contrast[1]}.`, 'shake');
    else say(pick(latest.kind === 'clue' ? RV_SAYS.clue : RV_SAYS.trait), 'nod');
    await swapStage(R.stage, hintHtml(history.length, latest));
  }

  function feedback(html) {
    R.feedback.innerHTML = html;
    if (html) play(R.feedback, [{ opacity: 0, transform: 'translateY(4px)' }, { opacity: 1, transform: 'translateY(0)' }], { duration: 380 });
  }

  // Before every round: the chef has chosen; is the player ready?
  function rvReady() {
    rv.phase = 'ready';
    rv.round = null;
    rv.hero = 0;
    rvView('ready');
    rvEnabled(false);
    feedback('');
    R.input.value = '';
    R.ledger.replaceChildren();
    R.ledgerEmpty.hidden = false;
    for (const [field, text] of [['h', '00'], ['t', '—'], ['c', '—'], ['p', '—']]) {
      R.t[field].textContent = text;
      delete R.t[field].dataset.value;
    }
    R.t.cl.textContent = 'dishes fit';
    R.bar.style.width = '0%';
    swapStage(R.stage, '<p class="eyebrow">The chef has chosen</p>' +
      '<h2 class="question">I have a dish in my mind. <em>Are you ready to guess?</em></h2>');
    riseIn($$('button', R.ready));
    if (view.screen === 'reverse') say(pick(RV_SAYS.ready));
  }

  async function rvStart() {
    if (rv.phase !== 'ready') return;
    rv.phase = 'loading';
    rvView('play');
    rvEnabled(false);
    swapStage(R.stage, thinking('Lifting a corner of the cloche'));
    try {
      const state = await ask('/api/game/clue/new', { client_id: clientId(), exclude: rv.recent });
      rv.round = state;
      rv.phase = 'playing';
      await rvShow(state);
      rvEnabled(true);
      if (view.screen === 'reverse' && matchMedia('(hover: hover)').matches) R.input.focus({ preventScroll: true });
    } catch (err) {
      trouble('reverse', err, rvReady);
    }
  }

  async function rvNextHint() {
    if (rv.busy || rv.phase !== 'playing') return;
    rv.busy = true;
    rvEnabled(false);
    const lastHint = rv.round.hints_revealed >= rv.round.hints_total;
    if (!lastHint) dismissStage(R.stage);
    try {
      const state = await ask('/api/game/clue/next', { session_id: rv.round.session_id });
      rv.round = state;
      if (state.clue) {
        feedback('');
        await rvShow(state);
      } else {
        rvTelemetry(state);
        say('That’s everything I have. Your best guess?');
        feedback('That is every hint I have. Make your best guess.');
      }
      rv.busy = false;
      rvEnabled(true);
    } catch (err) {
      rv.busy = false;
      trouble('reverse', err, rvReady);
    }
  }

  function rememberDish(name) {
    rv.recent.push(name);
    if (rv.recent.length > 25) rv.recent.shift();
  }

  function shake(node) {
    node.classList.remove('shake');
    void node.offsetWidth;
    node.classList.add('shake');
  }

  async function rvGuess(text) {
    if (!text || rv.busy || rv.phase !== 'playing') return;
    rv.busy = true;
    rvEnabled(false);
    try {
      const result = await ask('/api/game/clue/guess', { session_id: rv.round.session_id, guess: text });
      rv.round = result.state;
      rvTelemetry(result.state);
      if (result.correct) {
        rv.phase = 'done';
        rv.busy = false;
        score.points += result.score_awarded;
        renderScoreboard();
        bump(A.scoreboard);
        rememberDish(result.reveal.name);
        R.input.value = '';
        feedback('');
        say(pick(RV_SAYS.correct), 'spell');
        await openReveal({
          owner: 'reverse',
          eyebrow: `Correct · +${result.score_awarded} points`,
          kicker: `Named after ${result.state.hints_revealed} hint${result.state.hints_revealed === 1 ? '' : 's'}`,
          dish: result.reveal,
          actions: [
            { label: 'Next dish', style: 'white', run: () => closeReveal().then(rvReady) },
            { label: 'Try predict mode', style: 'outline', run: () => closeReveal().then(() => go('forward')) },
          ],
        });
        sparkle(V.name);
        return;
      }
      shake(R.input);
      sound.wrong();
      if (result.feedback) {
        say(pick(RV_SAYS.broke), 'shake');
        feedback(`<span class="gold">${esc(result.feedback)}</span>`);
      } else if (result.matched) {
        say(`Not ${result.matched}. Want another hint?`, 'shake');
        feedback(`Not ${esc(result.matched)}. Ask for another hint and I’ll show you why.`);
      } else if (result.near_miss) {
        say('So close… check the spelling.', 'nod');
        feedback(`Did you mean <button type="button" data-fill="${esc(result.near_miss)}">${esc(result.near_miss)}</button>?`);
      } else {
        say('Hmm, I don’t know that dish…', 'shake');
        feedback('That isn’t a dish I know. Try another name.');
      }
      rv.busy = false;
      rvEnabled(true);
      R.input.select();
    } catch (err) {
      rv.busy = false;
      trouble('reverse', err, rvReady);
    }
  }

  async function rvGiveUp() {
    if (rv.busy || rv.phase !== 'playing') return;
    rv.busy = true;
    rvEnabled(false);
    try {
      const outcome = await ask('/api/game/clue/giveup', { session_id: rv.round.session_id });
      rv.phase = 'done';
      rv.busy = false;
      rememberDish(outcome.dish.name);
      feedback('');
      say(`It was ${outcome.dish.name}! Shall we play another?`, 'nod');
      await openReveal({
        owner: 'reverse',
        eyebrow: 'The dish was',
        kicker: 'Under the cloche',
        dish: outcome.dish,
        note: 'Another one?',
        actions: [
          { label: 'Next dish', style: 'white', run: () => closeReveal().then(rvReady) },
          { label: 'Try predict mode', style: 'outline', run: () => closeReveal().then(() => go('forward')) },
        ],
      });
    } catch (err) {
      rv.busy = false;
      trouble('reverse', err, rvReady);
    }
  }

  R.form.addEventListener('submit', (event) => {
    event.preventDefault();
    rvGuess(R.input.value.trim());
  });
  R.feedback.addEventListener('click', (event) => {
    const button = event.target.closest('[data-fill]');
    if (!button) return;
    R.input.value = button.dataset.fill;
    R.input.focus();
  });
  $('#rv-ready-go').addEventListener('click', rvStart);
  R.hint.addEventListener('click', rvNextHint);
  R.giveup.addEventListener('click', rvGiveUp);

  // ─────────────────── the reveal: lift the cloche ───────────────────────
  const reveal = { open: false, owner: null, token: 0, actions: [] };

  function fillCard({ kicker, dish, confidence }) {
    V.kicker.textContent = kicker || 'The chef presents';
    V.name.textContent = dish.name;
    V.meta.textContent = [dish.cuisine, dish.country, dish.course].filter(Boolean).join(' · ');
    V.ingredients.textContent = (dish.key_ingredients || []).join(' · ');
    V.fact.textContent = dish.fact || '';
    V.badge.hidden = confidence == null;
    V.confidence.dataset.value = '0';
    V.confidence.textContent = '0';
  }

  function setActions(actions) {
    reveal.actions = actions.map((action) => {
      const button = document.createElement('button');
      button.type = 'button';
      button.className = `pill pill--${action.style || 'outline'}`;
      button.textContent = action.label;
      button.addEventListener('click', () => action.run());
      return { ...action, button };
    });
    V.actions.replaceChildren(...reveal.actions.map((a) => a.button));
  }

  function revealActionsEnabled(on) {
    for (const action of reveal.actions) action.button.disabled = !on;
  }

  function setReview(review) {
    V.review.replaceChildren();
    V.review.hidden = !review || review.length === 0;
    if (V.review.hidden) return;
    const shown = review.slice(0, 4);
    for (const row of shown) {
      const item = document.createElement('li');
      item.innerHTML = `<span class="said">You said ${esc(row.answered)}</span>` +
        `<q>${esc(row.question)}</q> My book says: ${esc(row.catalogue)}`;
      V.review.appendChild(item);
    }
    if (review.length > shown.length) {
      const more = document.createElement('li');
      more.textContent = `…and ${review.length - shown.length} more.`;
      V.review.appendChild(more);
    }
  }

  async function openReveal(options) {
    const token = ++reveal.token;
    reveal.open = true;
    reveal.owner = options.owner;
    fillCard(options);
    V.eyebrow.textContent = options.eyebrow || '';
    V.note.textContent = '';
    setReview(null);
    setActions([]);
    clearMotion(V.card, V.lid, V.plate, V.glow, V.eyebrow);
    V.root.hidden = false;
    V.root.scrollTop = 0;

    // 1. the room dims and the covered dish arrives
    play(V.root, [{ opacity: 0 }, { opacity: 1 }], { duration: 380 });
    play(V.eyebrow, [{ opacity: 0, transform: 'translateY(8px)', letterSpacing: '0.5em' }, { opacity: 1, transform: 'translateY(0)', letterSpacing: '0.3em' }], { duration: 900, fill: 'both' });
    play(V.plate, [{ opacity: 0, transform: 'scaleX(0.6)' }, { opacity: 1, transform: 'scaleX(1)' }], { duration: 700, fill: 'both' });
    await play(V.lid, [{ opacity: 0, transform: 'translateY(-48px)' }, { opacity: 1, transform: 'translateY(0)' }], { duration: 640, fill: 'both' });
    if (token !== reveal.token) return;
    sound.land();

    // 2. a moment of anticipation
    await play(V.lid, [
      { transform: 'rotate(0deg)' }, { transform: 'rotate(-2.4deg)' }, { transform: 'rotate(1.8deg)' },
      { transform: 'rotate(-0.8deg)' }, { transform: 'rotate(0deg)' },
    ], { duration: 620, easing: 'ease-in-out' });
    if (token !== reveal.token) return;

    // 3. the cloche lifts away and the dish rises into the light
    sparkle(V.lid);
    sound.boom();
    play(V.glow, [{ opacity: 0, transform: 'scale(0.55)' }, { opacity: 1, transform: 'scale(1)' }], { duration: 1300 });
    play(V.lid, [{ opacity: 1, transform: 'translateY(0) rotate(0deg)' }, { opacity: 0, transform: 'translateY(-190px) rotate(-8deg)' }], { duration: 1050 });
    play(V.plate, [{ opacity: 1 }, { opacity: 0 }], { duration: 600, delay: 220 });
    if (options.confidence != null) setTimeout(() => tween(V.confidence, Math.round(options.confidence * 100)), calm() ? 0 : 420);
    await play(V.card, [{ opacity: 0, transform: 'translateY(38px) scale(0.94)' }, { opacity: 1, transform: 'translateY(0) scale(1)' }], { duration: 950, delay: 160 });
    if (token !== reveal.token) return;

    // 4. the question, and what can be done about it
    await updateReveal(options, token);
  }

  async function updateReveal({ eyebrow, note, actions = [], review = null }, token = reveal.token) {
    if (eyebrow && eyebrow !== V.eyebrow.textContent) {
      await play(V.eyebrow, [{ opacity: 1 }, { opacity: 0 }], { duration: 180 });
      V.eyebrow.textContent = eyebrow;
      play(V.eyebrow, [{ opacity: 0 }, { opacity: 1 }], { duration: 400 });
    }
    if (token !== reveal.token) return;
    V.note.textContent = note || '';
    setReview(review);
    setActions(actions);
    [V.note, ...V.review.children, ...reveal.actions.map((a) => a.button)]
      .forEach((node, i) => play(node, rise(14), { duration: 620, delay: i * 70, fill: 'both' }));
    reveal.actions[0]?.button.focus({ preventScroll: true });
  }

  async function closeReveal({ recover = false } = {}) {
    if (!reveal.open) return;
    const token = ++reveal.token;
    reveal.open = false;
    revealActionsEnabled(false);
    if (recover && !calm()) {
      // the cloche comes back down over a dish that was not the one
      play(V.card, [{ opacity: 1, transform: 'scale(1)' }, { opacity: 0, transform: 'scale(0.95)' }], { duration: 420, easing: EASE_IN });
      play(V.plate, [{ opacity: 0 }, { opacity: 1 }], { duration: 320 });
      play(V.glow, [{ opacity: 1 }, { opacity: 0 }], { duration: 500 });
      await play(V.lid, [{ opacity: 0, transform: 'translateY(-190px) rotate(-8deg)' }, { opacity: 1, transform: 'translateY(0) rotate(0deg)' }], { duration: 680 });
      sound.land();
      await sleep(140);
    }
    await play(V.root, [{ opacity: 1 }, { opacity: 0 }], { duration: 340, easing: EASE_IN });
    if (token !== reveal.token) return;
    V.root.hidden = true;
    setActions([]);
    clearMotion(V.root, V.card, V.lid, V.plate, V.glow);
  }

  // A small burst of gold stars from the middle of a node.
  function sparkle(node) {
    if (calm() || !node) return;
    const { x, y } = centreOf(node);
    for (let i = 0; i < 12; i++) {
      const star = document.createElement('span');
      star.className = 'spark-burst';
      star.style.left = `${x}px`;
      star.style.top = `${y}px`;
      document.body.appendChild(star);
      const angle = (Math.PI * 2 * i) / 12 + Math.random() * 0.4;
      const distance = 70 + Math.random() * 90;
      const size = 0.5 + Math.random() * 0.9;
      play(star, [
        { opacity: 1, transform: `translate(0, 0) scale(${size}) rotate(0deg)` },
        { opacity: 0, transform: `translate(${Math.cos(angle) * distance}px, ${Math.sin(angle) * distance}px) scale(0.2) rotate(140deg)` },
      ], { duration: 900 + Math.random() * 500, easing: EASE }).then(() => star.remove());
    }
  }

  // ───────────────────────────── keyboard ────────────────────────────────
  document.addEventListener('keydown', (event) => {
    if (event.metaKey || event.ctrlKey || event.altKey || event.repeat) return;
    if (event.target.closest('input, textarea, select')) return;
    const key = event.key.toLowerCase();
    // a focused button already answers Enter and Space by itself
    if ((key === 'enter' || key === ' ') && event.target.closest('button, a')) return;
    if (PAGES.has(view.screen) && key === 'escape') { go('welcome'); return; }
    if (reveal.open) {
      const action = reveal.actions.find((a) => a.key === key && !a.button.disabled);
      if (action) { event.preventDefault(); sound.press(action.style === 'white'); action.button.click(); }
      return;
    }
    if (view.screen === 'forward' && fw.phase === 'ready') {
      if (key === 'y' || key === 'enter') { event.preventDefault(); sound.press(true); fwStart(); }
      if (key === 'n') { event.preventDefault(); sound.press(); $('#fw-ready-wait').click(); }
    } else if (view.screen === 'reverse' && rv.phase === 'ready') {
      if (key === 'enter' || key === 'y') { event.preventDefault(); sound.press(true); rvStart(); }
    } else if (view.screen === 'forward' && fw.phase === 'asking') {
      const answer = { y: 'yes', n: 'no', u: 'unsure' }[key];
      if (answer) { event.preventDefault(); sound.press(answer === 'yes'); fwAnswer(answer); }
    } else if (view.screen === 'reverse' && rv.phase === 'playing') {
      if (key === 'h' && !R.hint.disabled) { event.preventDefault(); sound.press(); rvNextHint(); }
      if (key === '/') { event.preventDefault(); R.input.focus(); }
    } else if (view.screen === 'welcome' && key === 'enter' && event.target === document.body) {
      go('forward', $('[data-go="forward"]'));
    }
  });

  // ───────────────────────────── wiring ──────────────────────────────────
  for (const button of $$('[data-go]')) {
    button.addEventListener('click', () => go(button.dataset.go, button));
  }
  for (const tab of $$('[role="tab"]', A.modeSwitch)) {
    tab.addEventListener('click', () => go(tab.dataset.mode));
  }
  $('#to-menu').addEventListener('click', () => go('welcome'));

  // ───────────────────────────── start ───────────────────────────────────
  renderScoreboard();
  loadWelcome();
  const initial = fromHash();
  if (initial === 'welcome') {
    A.video?.pause();
    entrance(W.root, 650);
    startParallax();
  } else {
    go(initial);
  }
})();
