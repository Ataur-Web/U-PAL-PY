import Head from 'next/head';
import { useState, useEffect, useCallback } from 'react';

const PAGE_SIZE = 20;

const ACCENTS = [
  { name: 'Rust',   hue: 28  },
  { name: 'Olive',  hue: 120 },
  { name: 'Teal',   hue: 190 },
  { name: 'Indigo', hue: 260 },
  { name: 'Plum',   hue: 330 },
];

function computeStats(data) {
  const total = data.length;
  if (total === 0) return { total: 0, avg: null, helpfulPct: null, langPct: null, withComments: 0 };
  const avg          = (data.reduce((s, d) => s + d.satisfaction, 0) / total).toFixed(1);
  const helpful      = data.filter(d => d.helpfulAnswer === true).length;
  const correct      = data.filter(d => d.correctLanguage === true).length;
  const withComments = data.filter(d => d.comments && d.comments.trim()).length;
  return { total, avg, helpfulPct: Math.round((helpful / total) * 100), langPct: Math.round((correct / total) * 100), withComments };
}

function exportCSV(data) {
  if (!data.length) return;
  const headers = ['#', 'Timestamp', 'Rating', 'Helpful', 'Correct Language', 'Comments'];
  const rows = data.map((d, i) => [
    i + 1, d.timestamp, d.satisfaction,
    d.helpfulAnswer === true ? 'Yes' : d.helpfulAnswer === false ? 'No' : '',
    d.correctLanguage === true ? 'Yes' : d.correctLanguage === false ? 'No' : '',
    '"' + (d.comments || '').replace(/"/g, '""') + '"',
  ]);
  const csv  = [headers, ...rows].map(r => r.join(',')).join('\n');
  const blob = new Blob([csv], { type: 'text/csv' });
  const url  = URL.createObjectURL(blob);
  const a    = Object.assign(document.createElement('a'), { href: url, download: `upal-feedback-${new Date().toISOString().slice(0, 10)}.csv` });
  a.click();
  URL.revokeObjectURL(url);
}

function Stars({ n }) {
  return (
    <span className="admin-stars">
      <span className="admin-stars-filled">{'★'.repeat(n)}</span>
      <span className="admin-stars-empty">{'★'.repeat(5 - n)}</span>
    </span>
  );
}

function Badge({ val }) {
  if (val === true)  return <span className="admin-badge admin-badge-yes">Yes</span>;
  if (val === false) return <span className="admin-badge admin-badge-no">No</span>;
  return <span className="admin-badge admin-badge-null">N/A</span>;
}

function StatCard({ value, label, sub }) {
  return (
    <div className="admin-stat">
      <div className="admin-stat-val">{value}</div>
      <div className="admin-stat-lbl">{label}</div>
      {sub && <div className="admin-stat-sub">{sub}</div>}
    </div>
  );
}

function SlidersIcon() {
  return (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" width="18" height="18">
      <line x1="4" y1="6" x2="20" y2="6"/><circle cx="8" cy="6" r="2" fill="currentColor" stroke="none"/>
      <line x1="4" y1="12" x2="20" y2="12"/><circle cx="16" cy="12" r="2" fill="currentColor" stroke="none"/>
      <line x1="4" y1="18" x2="20" y2="18"/><circle cx="10" cy="18" r="2" fill="currentColor" stroke="none"/>
    </svg>
  );
}

// localStorage keys for the theme controls. kept in one place so the
// admin page and the main page can share the same saved preferences if
// we ever want them to.
// ref: https://developer.mozilla.org/en-US/docs/Web/API/Window/localStorage
const LS_THEME  = 'upal.theme';
const LS_FONT   = 'upal.font';
const LS_ACCENT = 'upal.accent';

export default function Admin() {
  // useState with initial 'light' / 'editorial' / 28 so the server-rendered
  // HTML matches the first client render, we pull the saved values out of
  // localStorage in the effect below (localStorage is a browser API and
  // cannot run during SSR).
  const [theme,      setTheme]      = useState('light');
  const [font,       setFont]       = useState('editorial');
  const [accent,     setAccent]     = useState(28);
  const [tweaksOpen, setTweaksOpen] = useState(false);

  // one-shot restore from localStorage on mount. wrapped in try/catch
  // because Safari in private mode throws on any localStorage access.
  useEffect(() => {
    if (typeof window === 'undefined') return;
    try {
      const savedTheme  = localStorage.getItem(LS_THEME);
      const savedFont   = localStorage.getItem(LS_FONT);
      const savedAccent = localStorage.getItem(LS_ACCENT);
      if (savedTheme === 'light' || savedTheme === 'dark') setTheme(savedTheme);
      if (savedFont)  setFont(savedFont);
      if (savedAccent && !Number.isNaN(Number(savedAccent))) setAccent(Number(savedAccent));
    } catch (_) { /* private mode or disabled storage, fall through to defaults */ }
  }, []);

  // apply the theme to the document element whenever it changes, and
  // persist the new values so the next visit restores them.
  useEffect(() => {
    if (typeof window === 'undefined') return;
    document.documentElement.dataset.theme = theme;
    document.documentElement.dataset.font  = font;
    document.documentElement.style.setProperty('--accent-h', String(accent));
    try {
      localStorage.setItem(LS_THEME,  theme);
      localStorage.setItem(LS_FONT,   font);
      localStorage.setItem(LS_ACCENT, String(accent));
    } catch (_) { /* ignore, theme still applies for this session */ }
  }, [theme, font, accent]);

  const [password,  setPassword]  = useState('');
  const [authed,    setAuthed]    = useState(false);
  const [authError, setAuthError] = useState('');
  const [logging,   setLogging]   = useState(false);
  const [creds,     setCreds]     = useState('');

  const [data,     setData]     = useState([]);
  const [stats,    setStats]    = useState(null);
  const [page,     setPage]     = useState(1);
  const [loading,  setLoading]  = useState(false);
  const [fetchErr, setFetchErr] = useState('');

  const fetchData = useCallback(async (encodedCreds) => {
    setLoading(true); setFetchErr('');
    try {
      const res  = await fetch('/api/feedback', { headers: { Authorization: `Basic ${encodedCreds}` } });
      if (res.status === 401) { setAuthed(false); setAuthError('Session expired. Please sign in again.'); return; }
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const json = await res.json();
      setData(json); setStats(computeStats(json)); setPage(1);
    } catch (e) { setFetchErr(`Failed to load: ${e.message}`); }
    setLoading(false);
  }, []);

  async function login(e) {
    e.preventDefault();
    setLogging(true); setAuthError('');
    const encoded = btoa(`admin:${password}`);
    try {
      const res  = await fetch('/api/feedback', { headers: { Authorization: `Basic ${encoded}` } });
      if (res.status === 401) { setAuthError('Incorrect password. Try again.'); setLogging(false); return; }
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const json = await res.json();
      setData(json); setStats(computeStats(json)); setCreds(encoded); setAuthed(true);
    } catch (e) { setAuthError(`Error: ${e.message}`); }
    setLogging(false);
  }

  const totalPages = Math.ceil(data.length / PAGE_SIZE);
  const slice      = data.slice((page - 1) * PAGE_SIZE, page * PAGE_SIZE);

  return (
    <>
      <Head>
        <title>U-Pal Admin Feedback Dashboard</title>
        <meta name="viewport" content="width=device-width, initial-scale=1" />
      </Head>

      <style>{`
        .admin-shell {
          min-height: 100vh;
          background: var(--bg);
          display: flex;
          flex-direction: column;
        }

        /* ---- Login screen ---- */
        .admin-login {
          flex: 1;
          display: flex;
          align-items: center;
          justify-content: center;
          padding: var(--s-6);
          background-image:
            linear-gradient(to right,  var(--grid) 1px, transparent 1px),
            linear-gradient(to bottom, var(--grid) 1px, transparent 1px);
          background-size: 48px 48px;
        }
        .admin-login-card {
          background: var(--bg);
          border: 1px solid var(--line-2);
          border-radius: 16px;
          padding: var(--s-8) var(--s-7);
          width: 100%;
          max-width: 400px;
          display: flex;
          flex-direction: column;
          gap: var(--s-5);
          box-shadow: 0 4px 32px color-mix(in oklab, var(--ink) 6%, transparent);
        }
        .admin-login-brand {
          display: flex;
          align-items: center;
          gap: var(--s-3);
        }
        .admin-login-mark {
          width: 44px; height: 44px;
          border-radius: 12px;
          background: var(--accent);
          color: #fff;
          font-size: 18px; font-weight: 700;
          display: flex; align-items: center; justify-content: center;
          flex-shrink: 0;
        }
        .admin-login-title {
          font-family: var(--h-font);
          font-size: 22px;
          font-weight: var(--h-weight);
          color: var(--ink);
          line-height: 1.1;
        }
        .admin-login-title em { font-style: italic; color: var(--accent-ink); }
        .admin-login-sub {
          font-family: var(--mono);
          font-size: 11px;
          color: var(--ink-3);
          letter-spacing: 0.04em;
          text-transform: uppercase;
          margin-top: 3px;
        }
        .admin-login-divider { height: 1px; background: var(--line); }
        .admin-field-lbl {
          display: block;
          font-family: var(--mono);
          font-size: 10.5px;
          letter-spacing: 0.07em;
          text-transform: uppercase;
          color: var(--ink-3);
          margin-bottom: var(--s-2);
        }
        .admin-field-input {
          width: 100%;
          border: 1px solid var(--line-2);
          border-radius: var(--r-2);
          padding: 10px var(--s-3);
          background: var(--bg-2);
          font-family: var(--sans);
          font-size: 14px;
          color: var(--ink);
          outline: none;
          transition: border-color .15s, background .15s;
        }
        .admin-field-input:focus { border-color: var(--accent); background: var(--bg); }
        .admin-login-btn {
          width: 100%;
          padding: 12px;
          background: var(--accent);
          color: #fff;
          border: none;
          border-radius: var(--r-2);
          font-family: var(--sans);
          font-size: 14px;
          font-weight: 600;
          cursor: pointer;
          transition: opacity .15s, transform .1s;
        }
        .admin-login-btn:hover:not(:disabled) { opacity: 0.88; transform: translateY(-1px); }
        .admin-login-btn:disabled { opacity: .5; cursor: not-allowed; }
        .admin-login-err {
          font-size: 13px;
          color: oklch(0.45 0.16 25);
          background: oklch(0.97 0.03 25);
          border: 1px solid oklch(0.88 0.07 25);
          border-radius: var(--r-2);
          padding: var(--s-2) var(--s-3);
        }
        .admin-login-footer {
          text-align: center;
          font-family: var(--mono);
          font-size: 10.5px;
          color: var(--ink-4);
          letter-spacing: 0.04em;
        }

        /* ---- Topbar ---- */
        .admin-topbar {
          height: 56px;
          border-bottom: 1px solid var(--line);
          background: color-mix(in oklab, var(--bg) 88%, transparent);
          backdrop-filter: saturate(180%) blur(14px);
          -webkit-backdrop-filter: saturate(180%) blur(14px);
          display: flex;
          align-items: center;
          padding: 0 var(--s-6);
          gap: var(--s-4);
          position: sticky;
          top: 0;
          z-index: 40;
        }
        .admin-brand {
          display: flex;
          align-items: center;
          gap: var(--s-3);
          text-decoration: none;
        }
        .admin-mark {
          width: 32px; height: 32px;
          border-radius: 8px;
          background: var(--accent);
          color: #fff;
          font-size: 13px; font-weight: 700;
          display: flex; align-items: center; justify-content: center;
          flex-shrink: 0;
        }
        .admin-wordmark {
          font-size: 14px;
          font-weight: 600;
          color: var(--ink);
          letter-spacing: -0.01em;
        }
        .admin-wordmark em { font-style: italic; color: var(--accent-ink); }
        .admin-topbar-sep { width: 1px; height: 20px; background: var(--line-2); flex-shrink: 0; }
        .admin-topbar-pill {
          font-family: var(--mono);
          font-size: 10px;
          letter-spacing: 0.08em;
          text-transform: uppercase;
          color: var(--ink-3);
          border: 1px solid var(--line-2);
          border-radius: 20px;
          padding: 3px 10px;
        }
        .admin-topbar-spacer { flex: 1; }
        .admin-topbar-actions { display: flex; align-items: center; gap: var(--s-2); }
        .admin-topbar-btn {
          font-family: var(--sans);
          font-size: 12.5px;
          font-weight: 600;
          color: var(--ink-2);
          background: var(--bg-2);
          border: 1px solid var(--line-2);
          border-radius: 20px;
          padding: 5px 14px;
          cursor: pointer;
          transition: border-color .15s, color .15s, background .15s;
        }
        .admin-topbar-btn:hover { border-color: var(--ink); color: var(--ink); }
        .admin-topbar-btn:disabled { opacity: .5; cursor: not-allowed; }

        /* ---- Body layout ---- */
        .admin-body {
          flex: 1;
          padding: var(--s-8) var(--s-6) var(--s-10);
        }
        .admin-wrap {
          max-width: 1160px;
          margin: 0 auto;
          display: flex;
          flex-direction: column;
          gap: var(--s-7);
        }

        /* ---- Page header ---- */
        .admin-page-header {
          display: flex;
          align-items: flex-end;
          justify-content: space-between;
          gap: var(--s-4);
          flex-wrap: wrap;
          padding-bottom: var(--s-5);
          border-bottom: 1px solid var(--line);
        }
        .admin-page-eyebrow {
          font-family: var(--mono);
          font-size: 10.5px;
          letter-spacing: 0.08em;
          text-transform: uppercase;
          color: var(--ink-3);
          margin-bottom: var(--s-2);
          display: flex;
          align-items: center;
          gap: var(--s-2);
        }
        .admin-page-eyebrow::before {
          content: '';
          width: 16px; height: 1px;
          background: var(--accent);
        }
        .admin-page-title {
          font-family: var(--h-font);
          font-size: clamp(26px, 3vw, 36px);
          font-weight: var(--h-weight);
          letter-spacing: var(--h-tracking);
          color: var(--ink);
          line-height: 1.1;
        }
        .admin-page-title em { font-style: italic; color: var(--accent-ink); }
        .admin-page-actions { display: flex; gap: var(--s-2); align-items: center; }
        .admin-action-btn {
          font-family: var(--sans);
          font-size: 13px;
          font-weight: 600;
          padding: var(--s-2) var(--s-4);
          border-radius: var(--r-2);
          border: 1px solid var(--line-2);
          background: var(--bg);
          color: var(--ink-2);
          cursor: pointer;
          transition: border-color .15s, color .15s, background .15s;
        }
        .admin-action-btn:hover:not(:disabled) {
          border-color: var(--accent);
          color: var(--accent-ink);
          background: var(--accent-tint);
        }
        .admin-action-btn:disabled { opacity: .4; cursor: not-allowed; }
        .admin-action-btn.primary {
          background: var(--accent);
          color: #fff;
          border-color: var(--accent);
        }
        .admin-action-btn.primary:hover:not(:disabled) { opacity: 0.88; }

        /* ---- Error bar ---- */
        .admin-err {
          padding: var(--s-3) var(--s-4);
          background: oklch(0.97 0.03 25);
          border: 1px solid oklch(0.88 0.07 25);
          border-radius: var(--r-3);
          font-size: 13.5px;
          color: oklch(0.45 0.16 25);
        }

        /* ---- Stats grid ---- */
        .admin-stats {
          display: grid;
          grid-template-columns: repeat(5, 1fr);
          gap: 1px;
          border: 1px solid var(--line-2);
          border-radius: var(--r-4);
          overflow: hidden;
          background: var(--line-2);
        }
        .admin-stat {
          background: var(--bg);
          padding: var(--s-5) var(--s-5);
          display: flex;
          flex-direction: column;
          gap: var(--s-1);
        }
        .admin-stat:first-child { border-radius: var(--r-4) 0 0 var(--r-4); }
        .admin-stat:last-child  { border-radius: 0 var(--r-4) var(--r-4) 0; }
        .admin-stat-val {
          font-family: var(--h-font);
          font-size: clamp(28px, 3vw, 40px);
          font-weight: var(--h-weight);
          letter-spacing: var(--h-tracking);
          color: var(--accent-ink);
          line-height: 1;
        }
        .admin-stat-lbl {
          font-family: var(--mono);
          font-size: 10px;
          letter-spacing: 0.08em;
          text-transform: uppercase;
          color: var(--ink-3);
        }
        .admin-stat-sub { font-size: 11px; color: var(--ink-4); font-family: var(--mono); }

        /* ---- Feedback table card ---- */
        .admin-card {
          border: 1px solid var(--line-2);
          border-radius: var(--r-4);
          overflow: hidden;
          background: var(--bg);
        }
        .admin-card-head {
          display: flex;
          align-items: center;
          justify-content: space-between;
          padding: var(--s-4) var(--s-5);
          border-bottom: 1px solid var(--line);
          background: var(--bg-2);
          flex-wrap: wrap;
          gap: var(--s-3);
        }
        .admin-card-title {
          font-size: 14px;
          font-weight: 600;
          color: var(--ink);
          letter-spacing: -0.01em;
        }
        .admin-card-meta {
          font-family: var(--mono);
          font-size: 10.5px;
          color: var(--ink-3);
          margin-top: 3px;
        }
        .admin-table-wrap { overflow-x: auto; }
        .admin-table {
          width: 100%;
          border-collapse: collapse;
          font-size: 13px;
          color: var(--ink);
        }
        .admin-table thead th {
          background: var(--bg-2);
          padding: var(--s-2) var(--s-4);
          text-align: left;
          font-family: var(--mono);
          font-size: 10px;
          font-weight: 500;
          text-transform: uppercase;
          letter-spacing: 0.08em;
          color: var(--ink-3);
          white-space: nowrap;
          border-bottom: 1px solid var(--line-2);
        }
        .admin-table tbody tr { border-top: 1px solid var(--line); transition: background .1s; }
        .admin-table tbody tr:hover { background: var(--bg-2); }
        .admin-table td { padding: var(--s-3) var(--s-4); vertical-align: middle; }
        .admin-num { font-family: var(--mono); font-size: 11px; color: var(--ink-4); }
        .admin-date { font-size: 13px; font-weight: 500; color: var(--ink); }
        .admin-time { font-family: var(--mono); font-size: 10.5px; color: var(--ink-3); margin-top: 2px; }
        .admin-stars { letter-spacing: 1px; }
        .admin-stars-filled { color: var(--accent); }
        .admin-stars-empty  { color: var(--line-2); }
        .admin-badge {
          display: inline-block;
          padding: 2px 10px;
          border-radius: 20px;
          font-family: var(--mono);
          font-size: 10px;
          font-weight: 500;
          letter-spacing: 0.04em;
        }
        .admin-badge-yes  { background: var(--accent-tint); color: var(--accent-ink); }
        .admin-badge-no   { background: oklch(0.97 0.03 25); color: oklch(0.45 0.16 25); }
        .admin-badge-null { background: var(--bg-3); color: var(--ink-4); }
        .admin-comment {
          max-width: 260px;
          font-size: 12.5px;
          color: var(--ink-2);
          font-style: italic;
          white-space: nowrap;
          overflow: hidden;
          text-overflow: ellipsis;
        }
        .admin-empty td {
          text-align: center;
          padding: var(--s-10) var(--s-4);
          color: var(--ink-3);
          font-size: 13.5px;
        }
        .admin-spinner {
          width: 22px; height: 22px;
          border: 2px solid var(--line-2);
          border-top-color: var(--accent);
          border-radius: 50%;
          animation: adminSpin .7s linear infinite;
          margin: 0 auto var(--s-3);
        }
        @keyframes adminSpin { to { transform: rotate(360deg); } }

        /* ---- Pagination ---- */
        .admin-pag {
          display: flex;
          align-items: center;
          justify-content: space-between;
          padding: var(--s-3) var(--s-5);
          border-top: 1px solid var(--line);
          flex-wrap: wrap;
          gap: var(--s-3);
        }
        .admin-pag-info { font-family: var(--mono); font-size: 10.5px; color: var(--ink-3); }
        .admin-pag-btns { display: flex; gap: var(--s-2); }
        .admin-pag-btn {
          font-family: var(--sans);
          font-size: 12.5px;
          font-weight: 600;
          padding: var(--s-1) var(--s-4);
          border: 1px solid var(--line-2);
          background: var(--bg);
          color: var(--ink-2);
          border-radius: var(--r-2);
          cursor: pointer;
          transition: border-color .15s, background .15s, color .15s;
        }
        .admin-pag-btn:disabled { opacity: .3; cursor: not-allowed; }
        .admin-pag-btn:not(:disabled):hover {
          border-color: var(--accent);
          background: var(--accent-tint);
          color: var(--accent-ink);
        }

        /* ---- Inline theme controls (always visible in topbar) ---- */
        .admin-theme-controls {
          display: flex;
          align-items: center;
          gap: var(--s-3);
          padding: 0 var(--s-3);
          border-left: 1px solid var(--line-2);
          border-right: 1px solid var(--line-2);
        }
        .admin-theme-group {
          display: flex;
          align-items: center;
          gap: var(--s-1);
        }
        .admin-theme-swatch {
          width: 16px; height: 16px;
          border-radius: 50%;
          border: 2px solid transparent;
          cursor: pointer;
          transition: transform .15s, border-color .15s;
          padding: 0;
          flex-shrink: 0;
        }
        .admin-theme-swatch:hover { transform: scale(1.2); }
        .admin-theme-swatch.on { border-color: var(--ink); transform: scale(1.15); }
        .admin-theme-seg {
          display: flex;
          border: 1px solid var(--line-2);
          border-radius: var(--r-2);
          overflow: hidden;
        }
        .admin-theme-seg-btn {
          font-family: var(--mono);
          font-size: 10px;
          letter-spacing: 0.04em;
          text-transform: uppercase;
          padding: 3px 9px;
          border: none;
          background: transparent;
          color: var(--ink-3);
          cursor: pointer;
          transition: background .12s, color .12s;
        }
        .admin-theme-seg-btn + .admin-theme-seg-btn { border-left: 1px solid var(--line-2); }
        .admin-theme-seg-btn.on { background: var(--accent); color: #fff; }
        .admin-theme-seg-btn:not(.on):hover { background: var(--bg-2); color: var(--ink); }

        /* ---- Mobile themes toggle (hidden on desktop where controls are always visible) ---- */
        .admin-topbar-themes-mob {
          display: none;
          padding: 5px 8px;
        }

        /* ---- Responsive ---- */
        @media (max-width: 900px) {
          .admin-theme-controls { display: none; }
          .admin-topbar-themes-mob { display: flex; align-items: center; justify-content: center; }
        }
        @media (max-width: 760px) {
          .admin-stats { grid-template-columns: repeat(2, 1fr); }
          .admin-stat:first-child { border-radius: var(--r-4) 0 0 0; }
          .admin-stat:last-child  { border-radius: 0 0 var(--r-4) 0; }
          .admin-body { padding: var(--s-5) var(--s-4) var(--s-9); }
          .admin-topbar { padding: 0 var(--s-4); }
          .admin-topbar-pill { display: none; }
          .tweaks-fab         { display: none; }
          .tweaks-panel { right: 10px; left: auto; width: 260px; bottom: auto; top: 56px; }
        }
        @media (max-width: 480px) {
          .admin-stats { grid-template-columns: 1fr 1fr; }
          .admin-login-card { padding: var(--s-6) var(--s-5); }
          .tweaks-panel { right: 8px; width: calc(100vw - 16px); top: 56px; }
        }
      `}</style>

      <div className="admin-shell">

        {/* Login screen */}
        {!authed && (
          <div className="admin-login">
            <div className="admin-login-card">
              <div className="admin-login-brand">
                <div className="admin-login-mark">U</div>
                <div>
                  <div className="admin-login-title">U-Pal <em>Admin</em></div>
                  <div className="admin-login-sub">UWTSD · Feedback dashboard</div>
                </div>
              </div>

              <div className="admin-login-divider" />

              <form onSubmit={login} style={{ display: 'flex', flexDirection: 'column', gap: 'var(--s-4)' }}>
                <div>
                  <label className="admin-field-lbl" htmlFor="pw">Admin password</label>
                  <input
                    id="pw"
                    className="admin-field-input"
                    type="password"
                    value={password}
                    onChange={e => setPassword(e.target.value)}
                    placeholder="Enter password…"
                    autoFocus
                  />
                </div>
                <button className="admin-login-btn" type="submit" disabled={!password || logging}>
                  {logging ? 'Signing in…' : 'Sign in'}
                </button>
                {authError && <div className="admin-login-err">{authError}</div>}
              </form>

              <div className="admin-login-footer">U-Pal · BSc Dissertation · UWTSD 2026</div>
            </div>
          </div>
        )}

        {/* Dashboard */}
        {authed && (
          <>
            <header className="admin-topbar">
              <a href="/" className="admin-brand">
                <div className="admin-mark">U</div>
                <div className="admin-wordmark">U-Pal <em>admin</em></div>
              </a>
              <div className="admin-topbar-sep" />
              <span className="admin-topbar-pill">Feedback dashboard</span>
              <div className="admin-topbar-spacer" />

              {/* Always-on theme controls, accent, font, light/dark */}
              <div className="admin-theme-controls">
                <div className="admin-theme-group">
                  {ACCENTS.map(a => (
                    <button
                      key={a.hue}
                      className={`admin-theme-swatch${accent === a.hue ? ' on' : ''}`}
                      title={a.name}
                      style={{ background: `oklch(0.62 0.15 ${a.hue})` }}
                      onClick={() => setAccent(a.hue)}
                    />
                  ))}
                </div>
                <div className="admin-theme-seg">
                  {['editorial', 'grotesk', 'plex'].map(f => (
                    <button
                      key={f}
                      className={`admin-theme-seg-btn${font === f ? ' on' : ''}`}
                      onClick={() => setFont(f)}
                    >
                      {f === 'editorial' ? 'Serif' : f === 'grotesk' ? 'Sans' : 'Mono'}
                    </button>
                  ))}
                </div>
                <div className="admin-theme-seg">
                  {['light', 'dark'].map(t => (
                    <button
                      key={t}
                      className={`admin-theme-seg-btn${theme === t ? ' on' : ''}`}
                      onClick={() => setTheme(t)}
                    >
                      {t === 'light' ? 'Light' : 'Dark'}
                    </button>
                  ))}
                </div>
              </div>

              <div className="admin-topbar-actions">
                {/* Themes button, only shown below 900px where inline controls are hidden */}
                <button
                  className="admin-topbar-btn admin-topbar-themes-mob"
                  onClick={() => setTweaksOpen(v => !v)}
                  title="Themes"
                >
                  <SlidersIcon />
                </button>
                <button className="admin-topbar-btn" onClick={() => fetchData(creds)} disabled={loading}>
                  {loading ? 'Refreshing…' : 'Refresh'}
                </button>
                <button
                  className="admin-topbar-btn"
                  onClick={async () => {
                    // wipe local auth state
                    setAuthed(false);
                    setPassword('');
                    setCreds('');
                    setData([]);
                    setStats(null);
                    // hit /api/logout so the browser drops its cached Basic
                    // Auth creds, otherwise Chrome / Edge replay them on the
                    // next /admin visit and the user is silently signed in
                    // again.
                    // ref: https://developer.mozilla.org/en-US/docs/Web/HTTP/Authentication
                    try { await fetch('/api/logout'); } catch (_) { /* non-fatal */ }
                    // send the user back to the main site
                    window.location.href = '/';
                  }}
                >
                  Sign out
                </button>
              </div>
            </header>

            <div className="admin-body">
              <div className="admin-wrap">

                <div className="admin-page-header">
                  <div>
                    <div className="admin-page-eyebrow">Research data · UWTSD 2026</div>
                    <h1 className="admin-page-title">Feedback <em>results</em></h1>
                  </div>
                  <div className="admin-page-actions">
                    <button className="admin-action-btn primary" onClick={() => exportCSV(data)} disabled={!data.length}>
                      Export CSV
                    </button>
                  </div>
                </div>

                {fetchErr && <div className="admin-err">{fetchErr}</div>}

                {stats && (
                  <div className="admin-stats">
                    <StatCard value={stats.total} label="Total responses" />
                    <StatCard value={stats.avg ? `${stats.avg} ★` : 'N/A'} label="Avg rating" />
                    <StatCard value={stats.helpfulPct != null ? `${stats.helpfulPct}%` : 'N/A'} label="Found helpful" />
                    <StatCard value={stats.langPct != null ? `${stats.langPct}%` : 'N/A'} label="Correct language" />
                    <StatCard value={stats.withComments} label="Left comments" />
                  </div>
                )}

                <div className="admin-card">
                  <div className="admin-card-head">
                    <div>
                      <div className="admin-card-title">All feedback entries</div>
                      <div className="admin-card-meta">
                        {data.length} {data.length === 1 ? 'entry' : 'entries'} · sorted newest first
                      </div>
                    </div>
                    <button className="admin-action-btn" onClick={() => fetchData(creds)} disabled={loading}>
                      {loading ? 'Loading…' : 'Reload'}
                    </button>
                  </div>

                  <div className="admin-table-wrap">
                    <table className="admin-table">
                      <thead>
                        <tr>
                          <th style={{ width: 40 }}>#</th>
                          <th>Date &amp; time</th>
                          <th>Rating</th>
                          <th>Helpful?</th>
                          <th>Correct language?</th>
                          <th>Comments</th>
                        </tr>
                      </thead>
                      <tbody>
                        {loading && (
                          <tr className="admin-empty">
                            <td colSpan={6}>
                              <div className="admin-spinner" />
                              Loading feedback…
                            </td>
                          </tr>
                        )}
                        {!loading && data.length === 0 && (
                          <tr className="admin-empty">
                            <td colSpan={6}>No feedback submitted yet.</td>
                          </tr>
                        )}
                        {!loading && slice.map((entry, i) => {
                          const num  = (page - 1) * PAGE_SIZE + i + 1;
                          const date = new Date(entry.timestamp);
                          return (
                            <tr key={entry._id || i}>
                              <td><span className="admin-num">{num}</span></td>
                              <td>
                                <div className="admin-date">
                                  {date.toLocaleDateString('en-GB', { day: '2-digit', month: 'short', year: 'numeric' })}
                                </div>
                                <div className="admin-time">
                                  {date.toLocaleTimeString('en-GB', { hour: '2-digit', minute: '2-digit' })}
                                </div>
                              </td>
                              <td><Stars n={entry.satisfaction} /></td>
                              <td><Badge val={entry.helpfulAnswer} /></td>
                              <td><Badge val={entry.correctLanguage} /></td>
                              <td>
                                <div className="admin-comment" title={entry.comments || ''}>
                                  {entry.comments || <span style={{ color: 'var(--ink-4)' }}>N/A</span>}
                                </div>
                              </td>
                            </tr>
                          );
                        })}
                      </tbody>
                    </table>
                  </div>

                  <div className="admin-pag">
                    <span className="admin-pag-info">
                      {data.length > 0
                        ? `Showing ${(page - 1) * PAGE_SIZE + 1}–${Math.min(page * PAGE_SIZE, data.length)} of ${data.length}`
                        : '0 entries'}
                    </span>
                    <div className="admin-pag-btns">
                      <button className="admin-pag-btn" onClick={() => setPage(p => p - 1)} disabled={page <= 1}>Prev</button>
                      <button className="admin-pag-btn" onClick={() => setPage(p => p + 1)} disabled={page >= totalPages}>Next</button>
                    </div>
                  </div>
                </div>

              </div>
            </div>

            {/* Mobile themes panel (visible below 900px where topbar controls are hidden) */}
            {tweaksOpen && (
              <div className="tweaks-panel">
                <div className="tweaks-head">
                  <span className="dot" />
                  Themes
                  <button className="close-btn" onClick={() => setTweaksOpen(false)} aria-label="Close themes">×</button>
                </div>
                <div className="tweaks-body">
                  <div className="tweak-row">
                    <div className="tweak-label">Accent colour</div>
                    <div className="tweak-swatches">
                      {ACCENTS.map(a => (
                        <button
                          key={a.hue}
                          className={`tweak-sw${accent === a.hue ? ' on' : ''}`}
                          title={a.name}
                          style={{ background: `oklch(0.62 0.15 ${a.hue})` }}
                          onClick={() => setAccent(a.hue)}
                        />
                      ))}
                    </div>
                  </div>
                  <div className="tweak-row">
                    <div className="tweak-label">Font pairing</div>
                    <div className="tweak-fonts">
                      {['editorial', 'grotesk', 'plex'].map(f => (
                        <button key={f} className={`tweak-btn${font === f ? ' on' : ''}`} onClick={() => setFont(f)}>
                          {f === 'editorial' ? 'Editorial' : f === 'grotesk' ? 'Grotesk' : 'Plex'}
                        </button>
                      ))}
                    </div>
                  </div>
                  <div className="tweak-row">
                    <div className="tweak-label">Theme</div>
                    <div className="tweak-toggle">
                      {['light', 'dark'].map(t => (
                        <button key={t} className={`tweak-btn${theme === t ? ' on' : ''}`} onClick={() => setTheme(t)}>
                          {t === 'light' ? 'Light' : 'Dark'}
                        </button>
                      ))}
                    </div>
                  </div>
                </div>
              </div>
            )}
          </>
        )}

      </div>
    </>
  );
}
