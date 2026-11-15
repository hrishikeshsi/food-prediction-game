/* Chef Mystico — finding and talking to the game server.
 *
 * Where the server is depends on where the page is:
 *
 *   served by the game server itself      → the same address the page came from
 *   on this machine (file://, Live Server) → the local server, LOCAL_API
 *   anywhere else (GitHub Pages)           → the deployed server, LIVE_API
 *
 * Add ?api=https://host to a page's address to name a server explicitly.
 */
(() => {
  'use strict';

  // The deployed game server (see render.yaml). If Render gives the service a
  // different address, change it here.
  const LIVE_API = 'https://craving-jadugar.onrender.com';
  const LOCAL_API = 'http://127.0.0.1:8000';

  const OVERRIDE = new URLSearchParams(location.search).get('api');
  const ON_THIS_MACHINE = location.protocol === 'file:'
    || ['localhost', '127.0.0.1', '[::1]', ''].includes(location.hostname);
  const FALLBACK = ON_THIS_MACHINE ? LOCAL_API : LIVE_API;
  // GitHub Pages serves the page but has no game server to ask.
  const STATIC_HOST = /\.github\.io$/.test(location.hostname);
  const WAKING_AFTER_MS = 3000;   // a free server asleep takes up to a minute

  let base = null;
  let probing = null;

  function offline(message) {
    const error = new Error(message);
    error.offline = true;
    return error;
  }

  // The first address that answers /api/health the way the game server does.
  async function probe() {
    const candidates = OVERRIDE
      ? [OVERRIDE.replace(/\/+$/, '')]
      : [...(location.protocol.startsWith('http') && !STATIC_HOST ? [''] : []), FALLBACK];
    const slow = setTimeout(() => window.dispatchEvent(new CustomEvent('mystico:waking')), WAKING_AFTER_MS);
    try {
      for (const candidate of candidates) {
        try {
          const response = await fetch(candidate + '/api/health', { cache: 'no-store' });
          const health = response.ok ? await response.json() : null;
          if (health && health.status === 'ok' && 'dishes' in health) {
            base = candidate;
            window.dispatchEvent(new CustomEvent('mystico:awake'));
            return base;
          }
        } catch (err) {
          // nothing there, or something that is not the game server
        }
      }
    } finally {
      clearTimeout(slow);
    }
    throw offline(`no game server is answering at ${(OVERRIDE || FALLBACK).replace(/^https?:\/\//, '')}`);
  }

  function find() {
    if (base !== null) return Promise.resolve(base);
    if (!probing) probing = probe().finally(() => { probing = null; });
    return probing;
  }

  // GET when there is no body, POST with a JSON body otherwise.
  async function call(path, body) {
    const root = await find();
    let response;
    try {
      response = await fetch(root + path, body === undefined
        ? { cache: 'no-store' }
        : { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
    } catch (err) {
      base = null;   // it went away; look for it again next time
      throw offline('the game server stopped answering');
    }
    if (!response.ok) {
      const detail = await response.json().catch(() => ({}));
      throw new Error(typeof detail.detail === 'string' ? detail.detail : `the server said ${response.status}`);
    }
    return response.json();
  }

  // Check every few seconds until a server answers, then run onBack once.
  function waitFor(onBack, every = 3000) {
    const timer = setInterval(async () => {
      try {
        await find();
      } catch (stillDown) {
        return;
      }
      clearInterval(timer);
      onBack();
    }, every);
    return () => clearInterval(timer);
  }

  window.Mystico = {
    find,
    call,
    waitFor,
    local: ON_THIS_MACHINE && !OVERRIDE,
    server: (OVERRIDE || FALLBACK).replace(/^https?:\/\//, ''),
  };
})();
