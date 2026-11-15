/* Chef Mystico — sound.
 *
 * Every sound is synthesised with the Web Audio API, so there are no files to
 * download:
 *
 *   press   a heavy, muffled thud when anything pressable is clicked or tapped
 *   land    the cloche settling onto its plate
 *   boom    a deep thud and a shimmer as the cloche lifts
 *   magic   a soft thud and a shimmer when the chef casts a spell
 *   wrong   a dull double thud for a wrong guess
 *   spark   a tiny glassy ting for each speck of stardust behind the cursor
 *
 * Browsers only allow sound after the visitor's first click or key press, so
 * nothing plays before that. The button in the corner turns sound off, and the
 * choice is remembered on this device.
 */
(() => {
  'use strict';

  const STORE = 'mystico-sound';
  const PRESSABLE = 'button, a[href], [role="tab"], summary, .emblem, .host-chef';
  const MAGICAL = '.emblem, .host-chef';

  let on = true;
  try { on = localStorage.getItem(STORE) !== 'off'; } catch (err) { /* storage may be blocked */ }

  let unlocked = false;
  let ctx = null;
  let out = null;

  // ───────────────────────────── voices ──────────────────────────────────
  // Each voice draws into any audio context, at time t, through `dest`.

  const noiseFor = new WeakMap();
  function noise(c) {
    if (!noiseFor.has(c)) {
      const length = Math.floor(c.sampleRate * 0.25);
      const buffer = c.createBuffer(1, length, c.sampleRate);
      const data = buffer.getChannelData(0);
      for (let i = 0; i < length; i++) data[i] = Math.random() * 2 - 1;
      noiseFor.set(c, buffer);
    }
    return noiseFor.get(c);
  }

  // Soft clipping. A pure 45 Hz tone is felt rather than heard, and laptop and
  // phone speakers cannot play it at all; the harmonics this adds can.
  let curve = null;
  function warmth(c) {
    if (!curve) {
      curve = new Float32Array(1024);
      for (let i = 0; i < curve.length; i++) {
        const x = (i / (curve.length - 1)) * 2 - 1;
        curve[i] = Math.tanh(3 * x) / Math.tanh(3);
      }
    }
    const shaper = c.createWaveShaper();
    shaper.curve = curve;
    shaper.oversample = '2x';
    return shaper;
  }

  // A thud: a low tone falling fast in pitch, plus a muffled knock of noise at
  // the moment of impact. `weight` is loudness, `pitch` scales the frequencies.
  function thud(c, dest, t, { weight = 1, pitch = 1, length = 0.3 } = {}) {
    const tone = c.createOscillator();
    tone.type = 'sine';
    tone.frequency.setValueAtTime(160 * pitch, t);
    tone.frequency.exponentialRampToValueAtTime(45 * pitch, t + 0.14);
    const drive = c.createGain();
    drive.gain.value = 1.8;
    const body = c.createGain();
    body.gain.setValueAtTime(0.0001, t);
    body.gain.exponentialRampToValueAtTime(0.5 * weight, t + 0.004);
    body.gain.exponentialRampToValueAtTime(0.0001, t + length);
    tone.connect(drive).connect(warmth(c)).connect(body).connect(dest);
    tone.start(t);
    tone.stop(t + length + 0.05);

    const hit = c.createBufferSource();
    hit.buffer = noise(c);
    const muffle = c.createBiquadFilter();
    muffle.type = 'lowpass';
    muffle.frequency.setValueAtTime(1100 * pitch, t);
    muffle.frequency.exponentialRampToValueAtTime(180, t + 0.05);
    const knock = c.createGain();
    knock.gain.setValueAtTime(0.0001, t);
    knock.gain.exponentialRampToValueAtTime(0.35 * weight, t + 0.002);
    knock.gain.exponentialRampToValueAtTime(0.0001, t + 0.06);
    hit.connect(muffle).connect(knock).connect(dest);
    hit.start(t);
    hit.stop(t + 0.08);
  }

  // A shimmer: four high partials of an open fifth, entering one after another.
  function shimmer(c, dest, t, level = 0.045) {
    [1318.5, 1975.5, 2637, 3951].forEach((frequency, i) => {
      const start = t + i * 0.045;
      const tone = c.createOscillator();
      tone.type = 'sine';
      tone.frequency.value = frequency;
      const gain = c.createGain();
      gain.gain.setValueAtTime(0.0001, start);
      gain.gain.exponentialRampToValueAtTime(level, start + 0.012);
      gain.gain.exponentialRampToValueAtTime(0.0001, start + 0.7 + i * 0.1);
      tone.connect(gain).connect(dest);
      tone.start(start);
      tone.stop(start + 1);
    });
  }

  // A spark: one short, high, glassy note with a breath of crackle. The notes
  // come from one pentatonic scale, so a sweep of the cursor sounds like a wind
  // chime, and each spark is panned to where the cursor is.
  const SPARK_NOTES = [1760, 1975.5, 2217.5, 2637, 2960, 3520, 3951];
  function spark(c, dest, t, { pan = 0, level = 0.03 } = {}) {
    let target = dest;
    if (c.createStereoPanner) {
      const panner = c.createStereoPanner();
      panner.pan.value = Math.max(-1, Math.min(1, pan));
      panner.connect(dest);
      target = panner;
    }
    const note = SPARK_NOTES[Math.floor(Math.random() * SPARK_NOTES.length)];
    const ring = 0.08 + Math.random() * 0.06;
    [[1, 1], [2.01, 0.35]].forEach(([ratio, share]) => {
      const tone = c.createOscillator();
      tone.type = 'sine';
      tone.frequency.setValueAtTime(note * ratio * 0.985, t);
      tone.frequency.exponentialRampToValueAtTime(note * ratio, t + 0.015);
      const gain = c.createGain();
      gain.gain.setValueAtTime(0.0001, t);
      gain.gain.exponentialRampToValueAtTime(level * share, t + 0.003);
      gain.gain.exponentialRampToValueAtTime(0.0001, t + ring / ratio);
      tone.connect(gain).connect(target);
      tone.start(t);
      tone.stop(t + ring + 0.02);
    });
    const crackle = c.createBufferSource();
    crackle.buffer = noise(c);
    const thin = c.createBiquadFilter();
    thin.type = 'highpass';
    thin.frequency.value = 6000;
    const gain = c.createGain();
    gain.gain.setValueAtTime(0.0001, t);
    gain.gain.exponentialRampToValueAtTime(level * 0.6, t + 0.001);
    gain.gain.exponentialRampToValueAtTime(0.0001, t + 0.012);
    crackle.connect(thin).connect(gain).connect(target);
    crackle.start(t);
    crackle.stop(t + 0.02);
  }

  const VOICES = {
    spark,
    press: (c, d, t) => thud(c, d, t, { weight: 0.9, pitch: 1, length: 0.28 }),
    strong: (c, d, t) => thud(c, d, t, { weight: 1.1, pitch: 0.85, length: 0.34 }),
    land: (c, d, t) => thud(c, d, t, { weight: 0.75, pitch: 0.8, length: 0.3 }),
    boom: (c, d, t) => { thud(c, d, t, { weight: 1.4, pitch: 0.6, length: 0.75 }); shimmer(c, d, t + 0.12); },
    magic: (c, d, t) => { thud(c, d, t, { weight: 0.45, pitch: 1.25, length: 0.2 }); shimmer(c, d, t); },
    wrong: (c, d, t) => {
      thud(c, d, t, { weight: 0.8, pitch: 0.65, length: 0.2 });
      thud(c, d, t + 0.1, { weight: 0.6, pitch: 0.55, length: 0.25 });
    },
  };

  // ───────────────────────────── playing ─────────────────────────────────
  function context() {
    if (!unlocked) return null;
    if (!ctx) {
      const Context = window.AudioContext || window.webkitAudioContext;
      if (!Context) return null;
      ctx = new Context();
      const glue = ctx.createDynamicsCompressor();   // keeps overlapping thuds from clipping
      glue.threshold.value = -16;
      glue.knee.value = 10;
      glue.ratio.value = 5;
      glue.attack.value = 0.002;
      glue.release.value = 0.2;
      out = ctx.createGain();
      out.gain.value = 0.75;
      out.connect(glue).connect(ctx.destination);
    }
    if (ctx.state === 'suspended') ctx.resume();
    return ctx;
  }

  function play(name, options) {
    if (!on) return;
    const c = context();
    if (c) VOICES[name](c, out, c.currentTime + 0.005, options);
  }

  // At most one spark every 85 ms, however fast the cursor moves.
  let sparkedAt = 0;
  function sparkle(pan = 0, energy = 0.5) {
    const now = performance.now();
    if (now - sparkedAt < 85) return;
    sparkedAt = now;
    play('spark', { pan: pan * 0.7, level: 0.016 + 0.024 * Math.max(0, Math.min(1, energy)) });
  }

  // Render a voice offline, without speakers — for checking how it sounds.
  async function render(name, seconds = 1.2) {
    const Offline = window.OfflineAudioContext || window.webkitOfflineAudioContext;
    const c = new Offline(1, Math.ceil(44100 * seconds), 44100);
    const gain = c.createGain();
    gain.gain.value = 0.75;
    gain.connect(c.destination);
    VOICES[name](c, gain, 0.005);
    return (await c.startRendering()).getChannelData(0);
  }

  // ─────────────────────────── the pointer ───────────────────────────────
  const unlock = () => { unlocked = true; };
  for (const type of ['pointerdown', 'keydown', 'touchend']) {
    window.addEventListener(type, unlock, { capture: true, passive: true });
  }

  document.addEventListener('pointerdown', (event) => {
    if (event.button > 0) return;
    unlock();
    const target = event.target.closest && event.target.closest(PRESSABLE);
    if (!target || target.disabled || target.closest('.sound-toggle')) return;
    if (target.matches(MAGICAL)) play('magic');
    else play(target.matches('.pill--white') ? 'strong' : 'press');
  }, { capture: true, passive: true });

  // Enter or Space on a focused button presses it too.
  document.addEventListener('keydown', (event) => {
    if (event.repeat || (event.key !== 'Enter' && event.key !== ' ')) return;
    const target = event.target.closest && event.target.closest(PRESSABLE);
    if (target && !target.disabled && !target.closest('.sound-toggle')) {
      play(target.matches('.pill--white') ? 'strong' : 'press');
    }
  }, { capture: true });

  // ─────────────────────────── the switch ────────────────────────────────
  const style = document.createElement('style');
  style.textContent = `
    .sound-toggle {
      position: fixed; left: 16px; bottom: 16px; z-index: 9;
      display: grid; place-items: center; width: 40px; height: 40px; padding: 0;
      border: 1px solid rgba(250, 250, 250, 0.18); border-radius: 50%;
      background: rgba(9, 9, 11, 0.6); -webkit-backdrop-filter: blur(16px); backdrop-filter: blur(16px);
      color: rgba(250, 250, 250, 0.72); cursor: pointer;
      transition: color 240ms, border-color 240ms, transform 300ms cubic-bezier(0.16, 1, 0.3, 1);
    }
    .sound-toggle:hover { color: #FAFAFA; border-color: rgba(250, 250, 250, 0.46); }
    .sound-toggle:active { transform: scale(0.94); }
    .sound-toggle:focus-visible { outline: 1px solid #C8A96E; outline-offset: 3px; }
    .sound-toggle svg { width: 18px; height: 18px; fill: none; stroke: currentColor; stroke-width: 1.6;
      stroke-linecap: round; stroke-linejoin: round; }
    .sound-toggle .wave { stroke: #C8A96E; }
    .sound-toggle .mute { display: none; }
    .sound-toggle[aria-pressed="false"] .wave { display: none; }
    .sound-toggle[aria-pressed="false"] .mute { display: inline; }
    @media (max-width: 560px) { .sound-toggle { left: 12px; bottom: 12px; width: 36px; height: 36px; } }
  `;
  document.head.appendChild(style);

  const toggle = document.createElement('button');
  toggle.type = 'button';
  toggle.className = 'sound-toggle';
  toggle.innerHTML = '<svg viewBox="0 0 24 24" aria-hidden="true">' +
    '<path d="M4 9.5v5h3.5L12 18.5v-13L7.5 9.5H4z"/>' +
    '<path class="wave" d="M15.5 9a4 4 0 0 1 0 6"/><path class="wave" d="M18 6.5a7.5 7.5 0 0 1 0 11"/>' +
    '<path class="mute" d="M16 9.5l5 5M21 9.5l-5 5"/></svg>';
  const show = () => {
    toggle.setAttribute('aria-pressed', String(on));
    toggle.setAttribute('aria-label', on ? 'Sound on — turn off' : 'Sound off — turn on');
    toggle.title = on ? 'Sound on' : 'Sound off';
  };
  toggle.addEventListener('click', () => {
    on = !on;
    try { localStorage.setItem(STORE, on ? 'on' : 'off'); } catch (err) { /* not remembered, still works */ }
    show();
    if (on) { unlock(); play('press'); }
  });
  show();
  document.body.appendChild(toggle);

  window.MysticoSound = {
    press: (strong = false) => play(strong ? 'strong' : 'press'),
    land: () => play('land'),
    boom: () => play('boom'),
    magic: () => play('magic'),
    wrong: () => play('wrong'),
    spark: sparkle,
    render,
    get on() { return on; },
  };
})();
