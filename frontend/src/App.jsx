import { useState, useRef, useEffect } from 'react'
import {
  AreaChart, Area, LineChart, Line, BarChart, Bar,
  XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ReferenceLine,
} from 'recharts'

const C = {
  bg:      '#07090f',
  surface: '#0c0f1a',
  card:    '#111827',
  border:  '#1e2a3a',
  accent:  '#06b6d4',
  purple:  '#818cf8',
  success: '#10b981',
  danger:  '#ef4444',
  warning: '#f59e0b',
  text:    '#f1f5f9',
  muted:   '#64748b',
  gold:    '#fbbf24',
}

const pct    = v => v != null ? `${(+v).toFixed(2)}%`   : 'N/A'
const dollar = v => v != null ? `$${(+v).toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}` : 'N/A'
const sign   = v => (+v) > 0 ? C.success : C.danger

// ── Universe presets (kept in sync with scanner.py) ───────────────────────────

const NASDAQ100_TICKERS = ['AAPL','MSFT','GOOG','AMZN','META','NVDA','TSLA','AVGO','COST','NFLX','ADBE','AMD','INTC','CSCO','QCOM','INTU','PYPL','AMGN','SBUX','MDLZ']
const SP500_SAMPLE      = ['AAPL','MSFT','AMZN','GOOG','META','BRK-B','JNJ','V','PG','JPM','UNH','HD','MA','DIS','PYPL','ADBE','NFLX','CMCSA','XOM','T']
const FAANG_TICKERS     = ['META','AAPL','AMZN','NFLX','GOOG','MSFT']
const MEGA8_TICKERS     = ['AAPL','MSFT','GOOG','AMZN','META','NVDA','TSLA','AVGO']

const UNIVERSE_PRESETS = {
  nasdaq100: { label: 'NASDAQ 100', tickers: NASDAQ100_TICKERS },
  sp500:     { label: 'S&P 500',    tickers: SP500_SAMPLE },
  faang:     { label: 'FAANG+',     tickers: FAANG_TICKERS },
  mega8:     { label: 'Mega 8',     tickers: MEGA8_TICKERS },
  custom:    { label: 'Custom',     tickers: [] },
}

// Full-Kaggle dynamic universe options (Kaggle data source only)
const FULL_KAGGLE_OPTIONS = {
  full_kaggle_stocks: { label: 'All Stocks',      desc: '7,195 US stocks' },
  full_kaggle_all:    { label: 'Stocks + ETFs',   desc: '8,500+ instruments' },
}

const SCAN_RULES = [
  { key: 'top_volume',   label: 'Highest Volume'        },
  { key: 'top_gainers',  label: 'Biggest Daily Gainers' },
  { key: 'top_momentum', label: 'Strongest Momentum'    },
  { key: 'lowest_rsi',   label: 'Most Oversold (RSI)'   },
]

const RISK_DEFAULTS = {
  conservative: { stopLoss: 2,  takeProfit: 6,  positionSize: 30, maxDD: 10 },
  moderate:     { stopLoss: 5,  takeProfit: 10, positionSize: 50, maxDD: 20 },
  aggressive:   { stopLoss: 8,  takeProfit: 20, positionSize: 95, maxDD: 35 },
}

const TEMPLATES = [
  { short: 'EMA Cross',   full: 'Trade using a 15/50 EMA crossover' },
  { short: 'RSI Bounce',  full: 'Buy when RSI(14) drops below 30, sell when it recovers above 50' },
  { short: 'Bollinger',   full: 'Buy on lower Bollinger Band touch, sell at the upper band' },
  { short: 'Top Mover',   full: 'Each week trade the stock with the strongest 20-day momentum' },
]

// ── Shared UI components ──────────────────────────────────────────────────────

function Label({ children }) {
  return (
    <p style={{ fontSize: 11, fontWeight: 600, color: C.muted, textTransform: 'uppercase', letterSpacing: '.06em', marginBottom: 10 }}>
      {children}
    </p>
  )
}

function Card({ children, style = {} }) {
  return (
    <div style={{ background: C.card, border: `1px solid ${C.border}`, borderRadius: 12, ...style }}>
      {children}
    </div>
  )
}

function Tab({ label, active, onClick }) {
  return (
    <button onClick={onClick} style={{
      padding: '7px 16px', border: 'none', borderRadius: 8, fontFamily: 'inherit',
      fontSize: 13, fontWeight: 500, cursor: 'pointer', transition: 'all .15s',
      background: active ? C.accent + '22' : 'transparent',
      color: active ? C.accent : C.muted,
      borderBottom: active ? `2px solid ${C.accent}` : '2px solid transparent',
    }}>{label}</button>
  )
}

function MetricCard({ title, display, color }) {
  return (
    <div style={{ background: C.card, border: `1px solid ${C.border}`, borderRadius: 12, padding: '14px 18px' }}>
      <p style={{ color: C.muted, fontSize: 11, textTransform: 'uppercase', letterSpacing: '.06em', marginBottom: 6 }}>{title}</p>
      <p style={{ color: color ?? C.text, fontSize: 21, fontWeight: 700, fontFamily: 'monospace', letterSpacing: '-.02em' }}>{display ?? 'N/A'}</p>
    </div>
  )
}

function ChartTip({ active, payload, label, unit = '%' }) {
  if (!active || !payload?.length) return null
  return (
    <div style={{ background: C.surface, border: `1px solid ${C.border}`, borderRadius: 8, padding: '9px 13px', fontSize: 12 }}>
      <p style={{ color: C.muted, marginBottom: 4 }}>Iter {label}</p>
      {payload.map((p, i) => (
        <p key={i} style={{ color: p.color, fontFamily: 'monospace' }}>
          {p.name}: {(+p.value).toFixed(2)}{unit}
        </p>
      ))}
    </div>
  )
}

function ChartBox({ label, children }) {
  return (
    <div style={{ background: C.card, border: `1px solid ${C.border}`, borderRadius: 12, padding: '14px 16px' }}>
      <p style={{ fontSize: 12, color: C.muted, marginBottom: 10 }}>{label}</p>
      {children}
    </div>
  )
}

function Stat({ label, value, color }) {
  return (
    <div>
      <p style={{ fontSize: 11, color: C.muted, marginBottom: 2 }}>{label}</p>
      <p style={{ fontSize: 14, fontWeight: 600, fontFamily: 'monospace', color }}>{value}</p>
    </div>
  )
}

function Badge({ label, color }) {
  return (
    <span style={{
      background: color + '22', border: `1px solid ${color}55`,
      borderRadius: 6, padding: '2px 8px', fontSize: 11, color, fontWeight: 600,
    }}>{label}</span>
  )
}

function downloadJSON(data, filename) {
  const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' })
  const url  = URL.createObjectURL(blob)
  const a    = document.createElement('a')
  a.href = url; a.download = filename; a.click()
  URL.revokeObjectURL(url)
}

// ── Setup-panel sub-components ────────────────────────────────────────────────

function FormSection({ title, children }) {
  return (
    <div style={{ marginBottom: 18 }}>
      <p style={{ fontSize: 10, fontWeight: 700, color: C.muted, textTransform: 'uppercase', letterSpacing: '.08em', marginBottom: 8 }}>{title}</p>
      {children}
    </div>
  )
}

function Chip({ label, active, onClick, color, fullWidth = false }) {
  const c = color || C.accent
  return (
    <button onClick={onClick} style={{
      flex: fullWidth ? 'none' : 1,
      width: fullWidth ? '100%' : undefined,
      padding: '7px 6px',
      border: `1px solid ${active ? c : C.border}`,
      borderRadius: 8,
      background: active ? c + '22' : C.card,
      color: active ? c : C.muted,
      fontSize: 12,
      fontWeight: active ? 600 : 400,
      cursor: 'pointer',
      transition: 'all .12s',
      textAlign: 'center',
    }}>{label}</button>
  )
}

function NumField({ label, value, onChange, suffix }) {
  return (
    <div>
      <p style={{ fontSize: 10, color: C.muted, marginBottom: 4 }}>{label}</p>
      <div style={{ display: 'flex', alignItems: 'center', background: C.surface, border: `1px solid ${C.border}`, borderRadius: 8, overflow: 'hidden' }}>
        <input
          type="number" value={value}
          onChange={e => onChange(+e.target.value)}
          min={0} max={100}
          style={{ flex: 1, padding: '6px 8px', background: 'transparent', border: 'none', color: C.text, fontSize: 13, fontFamily: 'monospace', fontWeight: 600, width: 0 }}
        />
        <span style={{ padding: '0 8px', color: C.muted, fontSize: 12, flexShrink: 0 }}>{suffix}</span>
      </div>
    </div>
  )
}

// ── Main App ──────────────────────────────────────────────────────────────────

export default function App() {

  // ── Setup form state ────────────────────────────────────────────────────────
  const [form, setForm] = useState({
    strategy:       '',
    dataSource:     'yfinance',
    mode:           'single',
    ticker:         'AAPL',
    universePreset: 'nasdaq100',  // used when universeType === 'preset'
    universeType:   'preset',     // 'preset' | 'full_kaggle_stocks' | 'full_kaggle_all'
    customTickers:  '',
    scanRule:       'top_momentum',
    scanTopN:       1,
    riskLevel:      'moderate',
    stopLoss:       5,
    takeProfit:     10,
    positionSize:   50,
    maxDD:          20,
    horizon:        'weeks',
    optScope:       'all',
  })

  const setField = (k, v) => setForm(f => ({ ...f, [k]: v }))
  const applyRisk = level => {
    const d = RISK_DEFAULTS[level]
    setForm(f => ({ ...f, riskLevel: level, stopLoss: d.stopLoss, takeProfit: d.takeProfit, positionSize: d.positionSize, maxDD: d.maxDD }))
  }

  // ── AI assistant chat ───────────────────────────────────────────────────────
  const [aiMessages, setAiMessages] = useState([
    { type: 'ai', text: 'Describe your strategy above — I can help clarify it, or just fill the settings and hit Run.' },
  ])
  const [aiInput,   setAiInput]   = useState('')
  const [analyzing, setAnalyzing] = useState(false)
  const chatEndRef = useRef(null)
  useEffect(() => { chatEndRef.current?.scrollIntoView({ behavior: 'smooth' }) }, [aiMessages])

  const sendAiMessage = async (msg) => {
    if (!msg.trim() || analyzing) return
    setAiInput('')
    setAnalyzing(true)
    setAiMessages(prev => [...prev,
      { type: 'user', text: msg },
      { type: 'ai',   text: '…',  loading: true },
    ])
    try {
      const res  = await fetch('/api/analyze', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ strategy: form.strategy || msg, question_answer: msg }),
      })
      const data = await res.json()
      const reply = data.message + (data.question ? '\n\n' + data.question : '')
      setAiMessages(prev => [...prev.slice(0, -1), { type: 'ai', text: reply }])

      // Auto-fill form from AI inference
      if (data.mode) setField('mode', data.mode)
      if (data.ticker) { setField('ticker', data.ticker); if (!data.mode) setField('mode', 'single') }
      if (data.universe_preset && UNIVERSE_PRESETS[data.universe_preset]) {
        setField('universePreset', data.universe_preset)
        setField('mode', 'multi')
      }
      if (data.scan_rule)  setField('scanRule', data.scan_rule)
      if (data.scan_top_n) setField('scanTopN', data.scan_top_n)
    } catch (err) {
      setAiMessages(prev => [...prev.slice(0, -1), { type: 'ai', text: `Error: ${err.message}` }])
    } finally {
      setAnalyzing(false)
    }
  }

  // ── Results state ───────────────────────────────────────────────────────────
  const [results,         setResults]         = useState(null)
  const [loading,         setLoading]         = useState(false)
  const [statusMsg,       setStatusMsg]       = useState(null)   // {text, type}
  const [activeTab,       setActiveTab]       = useState('dashboard')
  const [startAmount,     setStartAmount]     = useState(100000)
  const [history,         setHistory]         = useState(null)
  const [benchmarkTicker, setBenchmarkTicker] = useState('SPY')
  const [benchmarkInput,  setBenchmarkInput]  = useState('SPY')

  useEffect(() => { if (activeTab === 'history' && history === null) fetchHistory() }, [activeTab])

  const fetchHistory = async () => {
    try { setHistory(await (await fetch('/api/history')).json()) }
    catch { setHistory([]) }
  }

  // ── Run backtest ────────────────────────────────────────────────────────────
  const runBacktest = async () => {
    if (!form.strategy.trim()) {
      setStatusMsg({ text: 'Please describe a trading strategy first.', type: 'error' })
      return
    }

    const isFullKaggle = form.mode === 'multi' && form.universeType !== 'preset'

    let tickers
    if (form.mode === 'single') {
      tickers = [(form.ticker.trim().toUpperCase() || 'AAPL')]
    } else if (isFullKaggle) {
      tickers = []  // backend will discover tickers dynamically
    } else {
      tickers = form.universePreset === 'custom'
        ? form.customTickers.split(/[,\s]+/).map(t => t.trim().toUpperCase()).filter(t => t.length >= 1 && t.length <= 6)
        : (UNIVERSE_PRESETS[form.universePreset]?.tickers ?? NASDAQ100_TICKERS)
      if (!tickers.length) {
        setStatusMsg({ text: 'Enter at least one ticker in the Custom universe.', type: 'error' })
        return
      }
    }

    const params = {
      prompt:           form.strategy + ` [Risk: ${form.riskLevel}, SL=${form.stopLoss}%, TP=${form.takeProfit}%, size=${form.positionSize}%]`,
      benchmark_ticker: benchmarkTicker,
      data_source:      form.dataSource,
      tickers,
      is_multi_stock:   form.mode === 'multi',
      scan_rule:        form.scanRule,
      scan_top_n:       form.scanTopN,
      universe_type:    isFullKaggle ? form.universeType : 'preset',
      risk_profile: {
        level:               form.riskLevel,
        stop_loss_pct:       form.stopLoss    / 100,
        take_profit_pct:     form.takeProfit  / 100,
        position_size_pct:   form.positionSize / 100,
        max_drawdown_limit:  form.maxDD        / 100,
        investment_horizon:  form.horizon,
      },
      optimization_scope: [form.optScope],
    }

    setLoading(true)
    setStatusMsg({ text: 'Running backtest & optimizing across 3 iterations…', type: 'info' })

    try {
      const res  = await fetch('/api/backtest', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(params),
      })
      const data = await res.json()
      if (!res.ok || data.error) throw new Error(data.error || `HTTP ${res.status}`)

      setResults(data)
      setActiveTab('dashboard')
      setHistory(null)

      const m = data.best_configuration?.metrics ?? {}
      const n = data.all_iterations?.length ?? 0
      setStatusMsg({
        text: `Done! ${n} iterations — CAGR: ${pct(m.cagr)} | Max DD: ${pct(m.max_drawdown)}`,
        type: 'success',
      })
    } catch (err) {
      setStatusMsg({ text: `Error: ${err.message}`, type: 'error' })
    } finally {
      setLoading(false)
    }
  }

  // ── Derived chart data ──────────────────────────────────────────────────────
  const bm       = results?.best_configuration?.metrics ?? {}
  const bestConf = results?.best_configuration?.config  ?? {}
  const iters    = results?.all_iterations  ?? []
  const genCode  = results?.generated_code  ?? ''
  const explain  = results?.explanation     ?? ''
  const riskProf = results?.risk_profile    ?? {}
  const optExpls = results?.optimization_explanations ?? []

  const chartData = iters.map((it, i) => ({
    n:        it.iteration ?? i + 1,
    cagr:     +(it.metrics?.cagr        ?? 0),
    drawdown: +(it.metrics?.max_drawdown ?? 0),
    winRate:  +(it.metrics?.win_rate     ?? 0),
    expect:   +(it.metrics?.expectancy   ?? 0),
  }))

  const barData = results ? [
    { name: 'CAGR',     value: +(bm.cagr        ?? 0) },
    { name: 'Win Rate', value: +(bm.win_rate     ?? 0) },
    { name: 'Drawdown', value: Math.abs(+(bm.max_drawdown ?? 0)) },
  ] : []

  const totalReturnPct = +(bm.total_return_pct ?? 0)
  const scaledFinal    = startAmount * (1 + totalReturnPct / 100)
  const profit         = scaledFinal - startAmount
  const isProfit       = profit >= 0
  const scaleFactor    = startAmount / 100_000

  const equityCurve  = (bm.portfolio_values ?? []).map(d => ({ date: d.date.slice(0, 7), value: Math.round(d.value * scaleFactor) }))
  const bmMap        = new Map((results?.benchmark_values ?? []).map(d => [d.date.slice(0, 7), Math.round(d.value * scaleFactor)]))
  const mergedCurve  = equityCurve.map(p => ({ date: p.date, portfolio: p.value, benchmark: bmMap.get(p.date) ?? null }))
  const activeBmLabel = results?.benchmark_ticker ?? benchmarkTicker
  const startLabel    = mergedCurve[0]?.date   ?? 'Start'
  const endLabel      = mergedCurve.at(-1)?.date ?? 'End'

  // Universe info line
  const universePreview = form.mode === 'multi'
    ? (form.universePreset === 'custom'
        ? `${form.customTickers.split(/[,\s]+/).filter(t => t.trim()).length} custom tickers`
        : (() => { const p = UNIVERSE_PRESETS[form.universePreset]; return `${p.tickers.length} tickers · ${p.tickers.slice(0, 4).join(', ')}…` })())
    : null

  // ── Render ──────────────────────────────────────────────────────────────────
  return (
    <div style={{ height: '100vh', background: C.bg, color: C.text, overflow: 'hidden', display: 'flex', flexDirection: 'column', fontFamily: "'Inter', -apple-system, BlinkMacSystemFont, sans-serif" }}>
      <style>{`
        @keyframes dot  { from { opacity:.3;transform:scale(.7) } to { opacity:1;transform:scale(1) } }
        @keyframes spin { to   { transform:rotate(360deg) } }
        *, *::before, *::after { box-sizing:border-box; margin:0; padding:0 }
        ::-webkit-scrollbar       { width:4px; height:4px }
        ::-webkit-scrollbar-thumb { background:${C.border}; border-radius:4px }
        ::-webkit-scrollbar-track { background:transparent }
        textarea, input, button { font-family:inherit; outline:none }
        strong { font-weight:700 }
        input[type=number]::-webkit-inner-spin-button { opacity:.4 }
      `}</style>

      {/* Header */}
      <header style={{ display:'flex', alignItems:'center', justifyContent:'space-between', padding:'11px 22px', background:C.surface, borderBottom:`1px solid ${C.border}`, flexShrink:0 }}>
        <div style={{ display:'flex', alignItems:'center', gap:11 }}>
          <div style={{ width:34, height:34, borderRadius:9, background:`linear-gradient(135deg,${C.accent},${C.purple})`, display:'flex', alignItems:'center', justifyContent:'center', fontSize:18 }}>📈</div>
          <div>
            <h1 style={{ fontSize:15, fontWeight:700, letterSpacing:'-.02em' }}>AlgoTrader AI</h1>
            <p style={{ fontSize:11, color:C.muted }}>AI-Powered Strategy Backtesting & Optimization</p>
          </div>
        </div>
        <div style={{ display:'flex', alignItems:'center', gap:8 }}>
          {results?.data_source && <Badge label={results.data_source === 'kaggle' ? 'Kaggle' : 'Yahoo Finance'} color={results.data_source === 'kaggle' ? C.purple : C.accent}/>}
          {results?.risk_profile?.level && <Badge label={results.risk_profile.level} color={results.risk_profile.level==='aggressive'?C.danger:results.risk_profile.level==='conservative'?C.success:C.warning}/>}
          <div style={{ width:8, height:8, borderRadius:'50%', background: results?C.success:loading?C.warning:C.muted, boxShadow:results?`0 0 8px ${C.success}88`:loading?`0 0 8px ${C.warning}88`:'none' }}/>
          <span style={{ fontSize:12, color:C.muted }}>{loading?'Running…':results?'Results ready':'Idle'}</span>
        </div>
      </header>

      <div style={{ display:'flex', flex:1, overflow:'hidden' }}>

        {/* ─── LEFT: Setup Panel ─────────────────────────────────────────────── */}
        <div style={{ width:340, display:'flex', flexDirection:'column', borderRight:`1px solid ${C.border}`, background:C.surface, flexShrink:0 }}>

          {/* Scrollable form body */}
          <div style={{ flex:1, overflowY:'auto', padding:'14px 12px 8px' }}>

            {/* Strategy */}
            <FormSection title="Strategy Description">
              <textarea
                value={form.strategy}
                onChange={e => setField('strategy', e.target.value)}
                placeholder={"Describe what you want to trade\ne.g. 'Trade the highest-volume NASDAQ stock each week using momentum'"}
                rows={4}
                style={{ width:'100%', background:C.card, border:`1px solid ${C.border}`, borderRadius:10, padding:'10px 12px', color:C.text, fontSize:13, lineHeight:1.6, resize:'vertical' }}
              />
              <div style={{ display:'flex', flexWrap:'wrap', gap:5, marginTop:7 }}>
                {TEMPLATES.map(t => (
                  <button key={t.short} onClick={() => setField('strategy', t.full)}
                    style={{ background:C.card, border:`1px solid ${C.border}`, borderRadius:6, padding:'4px 9px', color:C.muted, fontSize:11, cursor:'pointer' }}
                    onMouseEnter={e => e.currentTarget.style.color = C.text}
                    onMouseLeave={e => e.currentTarget.style.color = C.muted}
                  >{t.short}</button>
                ))}
              </div>
            </FormSection>

            {/* AI Assistant (mini chat) */}
            <div style={{ marginBottom:18, background:C.card, border:`1px solid ${C.border}`, borderRadius:10, overflow:'hidden' }}>
              <p style={{ fontSize:10, fontWeight:700, color:C.muted, textTransform:'uppercase', letterSpacing:'.08em', padding:'7px 10px 0' }}>AI Assistant</p>
              <div style={{ padding:'6px 10px 6px', maxHeight:120, overflowY:'auto', display:'flex', flexDirection:'column', gap:6 }}>
                {aiMessages.map((m, i) => (
                  <div key={i} style={{ display:'flex', gap:6, justifyContent: m.type==='user' ? 'flex-end' : 'flex-start' }}>
                    {m.type === 'ai' && (
                      <div style={{ width:17, height:17, borderRadius:'50%', background:`linear-gradient(135deg,${C.accent},${C.purple})`, display:'flex', alignItems:'center', justifyContent:'center', fontSize:8, fontWeight:700, color:'#fff', flexShrink:0, marginTop:2 }}>AI</div>
                    )}
                    <p style={{ background: m.type==='user' ? '#1e3a8a' : C.surface, borderRadius:8, padding:'5px 9px', fontSize:11.5, color: m.loading ? C.muted : C.text, maxWidth:'85%', whiteSpace:'pre-wrap', lineHeight:1.5 }}>
                      {m.loading
                        ? <span>…<span style={{ display:'inline-flex', gap:3, marginLeft:6, verticalAlign:'middle' }}>{[0,1,2].map(j=><span key={j} style={{ width:4, height:4, borderRadius:'50%', background:C.accent, display:'inline-block', animation:`dot .9s ${j*.22}s ease-in-out infinite alternate`}}/>)}</span></span>
                        : m.text}
                    </p>
                  </div>
                ))}
                <div ref={chatEndRef}/>
              </div>
              <div style={{ display:'flex', borderTop:`1px solid ${C.border}` }}>
                <input
                  value={aiInput}
                  onChange={e => setAiInput(e.target.value)}
                  onKeyDown={e => { if (e.key === 'Enter') sendAiMessage(aiInput) }}
                  placeholder="Ask AI or answer its question…"
                  disabled={analyzing || loading}
                  style={{ flex:1, padding:'7px 10px', background:'transparent', border:'none', color:C.text, fontSize:12 }}
                />
                <button
                  onClick={() => sendAiMessage(aiInput)}
                  disabled={analyzing || loading || !aiInput.trim()}
                  style={{ padding:'7px 12px', background:'transparent', border:'none', color: (analyzing || !aiInput.trim()) ? C.muted : C.accent, fontSize:12, cursor: (analyzing || !aiInput.trim()) ? 'not-allowed' : 'pointer', fontWeight:600 }}
                >{analyzing ? '…' : 'Ask'}</button>
              </div>
            </div>

            {/* Data Source */}
            <FormSection title="Data Source">
              <div style={{ display:'flex', gap:6 }}>
                <Chip label="Yahoo Finance" active={form.dataSource==='yfinance'}
                  onClick={() => setForm(f => ({ ...f, dataSource:'yfinance', universeType:'preset' }))} />
                <Chip label="Kaggle Local"  active={form.dataSource==='kaggle'}
                  onClick={() => setField('dataSource','kaggle')} color={C.purple} />
              </div>
            </FormSection>

            {/* Universe */}
            <FormSection title="Universe">
              <div style={{ display:'flex', gap:6, marginBottom:10 }}>
                <Chip label="Single Stock" active={form.mode==='single'} onClick={()=>setField('mode','single')} />
                <Chip label="Multi-Stock"  active={form.mode==='multi'}  onClick={()=>setField('mode','multi')}  color={C.purple} />
              </div>

              {form.mode === 'single' ? (
                <div>
                  <p style={{ fontSize:10, color:C.muted, marginBottom:5 }}>Ticker symbol</p>
                  <input
                    value={form.ticker}
                    onChange={e => setField('ticker', e.target.value.toUpperCase().replace(/[^A-Z.-]/g, ''))}
                    placeholder="AAPL"
                    maxLength={8}
                    style={{ width:'100%', background:C.card, border:`1px solid ${C.border}`, borderRadius:8, padding:'8px 12px', color:C.text, fontSize:16, fontFamily:'monospace', fontWeight:700 }}
                  />
                </div>
              ) : (
                <>
                  {/* ── Full-Kaggle dynamic scan (only when Kaggle data source) ── */}
                  {form.dataSource === 'kaggle' && (
                    <div style={{ marginBottom:10 }}>
                      <p style={{ fontSize:10, color:C.gold, fontWeight:700, marginBottom:5, textTransform:'uppercase', letterSpacing:'.06em' }}>
                        Dynamic — full dataset
                      </p>
                      <div style={{ display:'flex', gap:5, marginBottom:4 }}>
                        {Object.entries(FULL_KAGGLE_OPTIONS).map(([key, opt]) => (
                          <button key={key}
                            onClick={() => setField('universeType', form.universeType === key ? 'preset' : key)}
                            style={{
                              flex:1, padding:'7px 6px',
                              border:`1px solid ${form.universeType===key ? C.gold : C.border}`,
                              borderRadius:8, background: form.universeType===key ? C.gold+'22' : C.card,
                              color: form.universeType===key ? C.gold : C.muted,
                              fontSize:11, fontWeight: form.universeType===key ? 700 : 400,
                              cursor:'pointer', display:'flex', flexDirection:'column', alignItems:'center', gap:2,
                            }}>
                            <span>{opt.label}</span>
                            <span style={{ fontSize:10, color:C.muted }}>{opt.desc}</span>
                          </button>
                        ))}
                      </div>
                      {form.universeType !== 'preset' && (
                        <p style={{ fontSize:10, color:C.gold+'bb', lineHeight:1.5 }}>
                          First scan takes ~30s and is cached for future runs.
                        </p>
                      )}
                      <div style={{ borderBottom:`1px solid ${C.border}`, margin:'8px 0 10px' }}/>
                    </div>
                  )}

                  {/* ── Preset groups (always visible when universeType === 'preset') ── */}
                  {form.universeType === 'preset' && (
                    <>
                      <p style={{ fontSize:10, color:C.muted, marginBottom:5, textTransform:'uppercase', letterSpacing:'.06em', fontWeight:700 }}>Preset groups</p>
                      <div style={{ display:'grid', gridTemplateColumns:'1fr 1fr', gap:5, marginBottom:7 }}>
                        {Object.entries(UNIVERSE_PRESETS).map(([key, p]) => (
                          <button key={key} onClick={() => setField('universePreset', key)} style={{
                            padding:'7px 8px', border:`1px solid ${form.universePreset===key ? C.purple : C.border}`,
                            borderRadius:8, background: form.universePreset===key ? C.purple+'22' : C.card,
                            color: form.universePreset===key ? C.purple : C.muted,
                            fontSize:12, fontWeight: form.universePreset===key ? 600 : 400,
                            cursor:'pointer', display:'flex', flexDirection:'column', alignItems:'center', gap:2,
                          }}>
                            <span>{p.label}</span>
                            {key !== 'custom' && <span style={{ fontSize:10, color:C.muted }}>{p.tickers.length} tickers</span>}
                          </button>
                        ))}
                      </div>

                      {form.universePreset === 'custom' && (
                        <textarea
                          value={form.customTickers}
                          onChange={e => setField('customTickers', e.target.value.toUpperCase())}
                          placeholder="AAPL, MSFT, GOOG, TSLA, AMZN…"
                          rows={2}
                          style={{ width:'100%', background:C.card, border:`1px solid ${C.border}`, borderRadius:8, padding:'8px 10px', color:C.text, fontSize:12, resize:'vertical', marginBottom:6 }}
                        />
                      )}

                      {universePreview && (
                        <p style={{ fontSize:10, color:C.muted, marginBottom:10 }}>{universePreview}</p>
                      )}
                    </>
                  )}

                  {/* ── Scan rule + top N (always shown in multi mode) ── */}
                  <p style={{ fontSize:10, color:C.muted, marginBottom:6 }}>
                    {form.universeType !== 'preset' ? 'Monthly selection rule' : 'Daily selection rule'}
                  </p>
                  <div style={{ display:'flex', flexDirection:'column', gap:4 }}>
                    {SCAN_RULES.map(r => (
                      <button key={r.key} onClick={() => setField('scanRule', r.key)} style={{
                        padding:'6px 10px', border:`1px solid ${form.scanRule===r.key ? C.accent : C.border}`,
                        borderRadius:7, background: form.scanRule===r.key ? C.accent+'18' : C.card,
                        color: form.scanRule===r.key ? C.accent : C.muted, fontSize:12, textAlign:'left', cursor:'pointer',
                      }}>{r.label}</button>
                    ))}
                  </div>

                  <div style={{ display:'flex', alignItems:'center', gap:8, marginTop:10 }}>
                    <span style={{ fontSize:11, color:C.muted }}>Select top</span>
                    <input
                      type="number" value={form.scanTopN}
                      onChange={e => setField('scanTopN', Math.max(1, Math.min(10, +e.target.value)))}
                      min={1} max={10}
                      style={{ width:50, background:C.card, border:`1px solid ${C.border}`, borderRadius:7, padding:'5px 8px', color:C.text, fontSize:13, fontFamily:'monospace', fontWeight:700 }}
                    />
                    <span style={{ fontSize:11, color:C.muted }}>stock(s) per month</span>
                  </div>
                </>
              )}
            </FormSection>

            {/* Risk Profile */}
            <FormSection title="Risk Profile">
              <div style={{ display:'flex', gap:5, marginBottom:10 }}>
                <Chip label="Conservative" active={form.riskLevel==='conservative'} onClick={()=>applyRisk('conservative')} color={C.success} />
                <Chip label="Moderate"     active={form.riskLevel==='moderate'}     onClick={()=>applyRisk('moderate')}     color={C.warning} />
                <Chip label="Aggressive"   active={form.riskLevel==='aggressive'}   onClick={()=>applyRisk('aggressive')}   color={C.danger}  />
              </div>
              <div style={{ display:'grid', gridTemplateColumns:'1fr 1fr', gap:7 }}>
                <NumField label="Stop Loss"     value={form.stopLoss}     onChange={v=>setField('stopLoss',v)}     suffix="%" />
                <NumField label="Take Profit"   value={form.takeProfit}   onChange={v=>setField('takeProfit',v)}   suffix="%" />
                <NumField label="Position Size" value={form.positionSize} onChange={v=>setField('positionSize',v)} suffix="%" />
                <NumField label="Max Drawdown"  value={form.maxDD}        onChange={v=>setField('maxDD',v)}        suffix="%" />
              </div>
            </FormSection>

            {/* Holding Period */}
            <FormSection title="Holding Period">
              <div style={{ display:'flex', gap:5 }}>
                {[{v:'days',l:'Days'},{v:'weeks',l:'Weeks'},{v:'months',l:'Months'},{v:'years',l:'Years'}].map(h => (
                  <Chip key={h.v} label={h.l} active={form.horizon===h.v} onClick={()=>setField('horizon',h.v)} />
                ))}
              </div>
            </FormSection>

            {/* Optimization Scope */}
            <FormSection title="Optimize">
              <div style={{ display:'flex', gap:5 }}>
                <Chip label="All Params"    active={form.optScope==='all'}      onClick={()=>setField('optScope','all')}      />
                <Chip label="Risk Only"     active={form.optScope==='risk'}     onClick={()=>setField('optScope','risk')}     />
                <Chip label="Strategy Only" active={form.optScope==='strategy'} onClick={()=>setField('optScope','strategy')} />
              </div>
            </FormSection>

          </div>{/* end scrollable form */}

          {/* Footer: status + run */}
          <div style={{ padding:'10px 12px', borderTop:`1px solid ${C.border}`, background:C.bg, flexShrink:0 }}>
            {statusMsg && (
              <p style={{
                fontSize:11.5, marginBottom:8, lineHeight:1.5,
                color: statusMsg.type==='error' ? C.danger : statusMsg.type==='success' ? C.success : C.muted,
              }}>{statusMsg.text}</p>
            )}
            <button
              onClick={runBacktest}
              disabled={loading}
              style={{
                width:'100%', padding:'11px 0',
                background: loading ? C.border : `linear-gradient(135deg,${C.success},#059669)`,
                border:'none', borderRadius:10,
                color: loading ? C.muted : '#fff',
                fontWeight:700, fontSize:14, cursor: loading ? 'not-allowed' : 'pointer',
                display:'flex', alignItems:'center', justifyContent:'center', gap:8, transition:'all .15s',
              }}
            >
              {loading
                ? <><span style={{ width:14, height:14, border:`2px solid ${C.muted}`, borderTopColor:C.accent, borderRadius:'50%', display:'inline-block', animation:'spin .8s linear infinite' }}/> Running…</>
                : '▶  Run Backtest'}
            </button>
          </div>

        </div>{/* end left panel */}

        {/* ─── RIGHT: Results Panel ──────────────────────────────────────────── */}
        <div style={{ flex:1, display:'flex', flexDirection:'column', overflow:'hidden' }}>

          {/* Tab bar */}
          <div style={{ display:'flex', gap:4, padding:'10px 20px', borderBottom:`1px solid ${C.border}`, background:C.surface, flexShrink:0 }}>
            {[
              { id:'dashboard', label:'📊 Dashboard' },
              { id:'portfolio', label:'💼 Portfolio' },
              { id:'code',      label:'🧑‍💻 Code' },
              { id:'history',   label:'🕑 History' },
              { id:'settings',  label:'⚙️ Settings' },
            ].map(t => <Tab key={t.id} label={t.label} active={activeTab===t.id} onClick={() => setActiveTab(t.id)}/>)}
          </div>

          <div style={{ flex:1, overflowY:'auto', padding:22 }}>

            {/* ── DASHBOARD ───────────────────────────────────────── */}
            {activeTab === 'dashboard' && (
              !results
                ? <EmptyState onSelect={s => setField('strategy', s)} />
                : (
                  <div style={{ display:'flex', flexDirection:'column', gap:22 }}>

                    {explain && (
                      <section>
                        <Label>AI Analysis</Label>
                        <Card style={{ padding:'16px 20px' }}>
                          <div style={{ display:'flex', gap:14, alignItems:'flex-start' }}>
                            <div style={{ width:36, height:36, borderRadius:10, flexShrink:0, background:`linear-gradient(135deg,${C.accent},${C.purple})`, display:'flex', alignItems:'center', justifyContent:'center', fontSize:18 }}>🤖</div>
                            <p style={{ color:C.text, fontSize:14, lineHeight:1.7 }}>{explain}</p>
                          </div>
                        </Card>
                      </section>
                    )}

                    {optExpls.length > 0 && (
                      <section>
                        <Label>Optimization Journey</Label>
                        <Card style={{ padding:'14px 18px' }}>
                          {optExpls.map((e, i) => (
                            <div key={i} style={{ display:'flex', gap:12, alignItems:'flex-start', marginBottom: i < optExpls.length-1 ? 10 : 0 }}>
                              <div style={{ width:22, height:22, borderRadius:'50%', background:C.purple+'33', border:`1px solid ${C.purple}55`, display:'flex', alignItems:'center', justifyContent:'center', fontSize:11, color:C.purple, fontWeight:700, flexShrink:0 }}>{i+2}</div>
                              <p style={{ fontSize:13, color:C.muted, lineHeight:1.6 }}>{e}</p>
                            </div>
                          ))}
                        </Card>
                      </section>
                    )}

                    {riskProf?.level && (
                      <section>
                        <Label>Risk Profile Applied</Label>
                        <Card style={{ padding:'14px 18px' }}>
                          <div style={{ display:'flex', gap:12, flexWrap:'wrap' }}>
                            <Badge label={riskProf.level?.toUpperCase()} color={riskProf.level==='aggressive'?C.danger:riskProf.level==='conservative'?C.success:C.warning}/>
                            <Stat label="Stop Loss"     value={pct(riskProf.stop_loss_pct*100)}     color={C.danger}/>
                            <Stat label="Take Profit"   value={pct(riskProf.take_profit_pct*100)}   color={C.success}/>
                            <Stat label="Position Size" value={pct(riskProf.position_size_pct*100)} color={C.accent}/>
                          </div>
                        </Card>
                      </section>
                    )}

                    <section>
                      <div style={{ display:'flex', alignItems:'center', justifyContent:'space-between', marginBottom:10 }}>
                        <Label>Best Configuration</Label>
                        <button onClick={() => downloadJSON({ config:bestConf, metrics:bm }, `best_config_${results.strategy_id??'run'}.json`)}
                          style={{ background:C.success+'22', border:`1px solid ${C.success}55`, borderRadius:8, padding:'5px 12px', color:C.success, fontSize:12, fontWeight:600, cursor:'pointer' }}>
                          ⬇ Download JSON
                        </button>
                      </div>
                      <Card style={{ padding:'14px 18px', fontFamily:'monospace', fontSize:13, color:C.success, whiteSpace:'pre-wrap', wordBreak:'break-all' }}>
                        {typeof bestConf==='string' ? bestConf : JSON.stringify(bestConf, null, 2)}
                      </Card>
                    </section>

                    <section>
                      <Label>Key Metrics — Best Config</Label>
                      <div style={{ display:'grid', gridTemplateColumns:'repeat(3,1fr)', gap:12 }}>
                        <MetricCard title="CAGR"         display={pct(bm.cagr)}             color={sign(bm.cagr)}/>
                        <MetricCard title="Total Return" display={pct(bm.total_return_pct)}  color={sign(bm.total_return_pct)}/>
                        <MetricCard title="Max Drawdown" display={pct(bm.max_drawdown)}      color={C.danger}/>
                        <MetricCard title="Win Rate"     display={pct(bm.win_rate)}           color={(+bm.win_rate)>50?C.success:C.warning}/>
                        <MetricCard title="Expectancy"   display={dollar(bm.expectancy)}      color={sign(bm.expectancy)}/>
                        <MetricCard title="Total Trades" display={bm.total_trades??'N/A'}     color={C.accent}/>
                        <MetricCard title="Avg Win"      display={dollar(bm.avg_win)}          color={C.success}/>
                        <MetricCard title="Avg Loss"     display={dollar(bm.avg_loss)}          color={C.danger}/>
                        <MetricCard title="Final Value"  display={dollar(bm.final_portfolio_value)} color={C.gold}/>
                      </div>
                    </section>

                    {chartData.length > 1 && (
                      <section>
                        <Label>Optimization Progress</Label>
                        <div style={{ display:'grid', gridTemplateColumns:'1fr 1fr', gap:14 }}>
                          <ChartBox label="CAGR by Iteration">
                            <ResponsiveContainer width="100%" height={150}>
                              <AreaChart data={chartData} margin={{ top:4, right:4, bottom:0, left:-10 }}>
                                <defs><linearGradient id="g1" x1="0" y1="0" x2="0" y2="1"><stop offset="5%" stopColor={C.success} stopOpacity={.35}/><stop offset="95%" stopColor={C.success} stopOpacity={0}/></linearGradient></defs>
                                <CartesianGrid strokeDasharray="3 3" stroke={C.border}/>
                                <XAxis dataKey="n" stroke={C.muted} fontSize={11}/>
                                <YAxis stroke={C.muted} fontSize={11} tickFormatter={v=>`${v.toFixed(1)}%`}/>
                                <Tooltip content={<ChartTip/>}/>
                                <Area type="monotone" dataKey="cagr" name="CAGR" stroke={C.success} fill="url(#g1)" strokeWidth={2} dot={false}/>
                              </AreaChart>
                            </ResponsiveContainer>
                          </ChartBox>
                          <ChartBox label="Drawdown by Iteration">
                            <ResponsiveContainer width="100%" height={150}>
                              <AreaChart data={chartData} margin={{ top:4, right:4, bottom:0, left:-10 }}>
                                <defs><linearGradient id="g2" x1="0" y1="0" x2="0" y2="1"><stop offset="5%" stopColor={C.danger} stopOpacity={.35}/><stop offset="95%" stopColor={C.danger} stopOpacity={0}/></linearGradient></defs>
                                <CartesianGrid strokeDasharray="3 3" stroke={C.border}/>
                                <XAxis dataKey="n" stroke={C.muted} fontSize={11}/>
                                <YAxis stroke={C.muted} fontSize={11} tickFormatter={v=>`${v.toFixed(1)}%`}/>
                                <Tooltip content={<ChartTip/>}/>
                                <Area type="monotone" dataKey="drawdown" name="Drawdown" stroke={C.danger} fill="url(#g2)" strokeWidth={2} dot={false}/>
                              </AreaChart>
                            </ResponsiveContainer>
                          </ChartBox>
                          <ChartBox label="Win Rate by Iteration">
                            <ResponsiveContainer width="100%" height={150}>
                              <LineChart data={chartData} margin={{ top:4, right:4, bottom:0, left:-10 }}>
                                <CartesianGrid strokeDasharray="3 3" stroke={C.border}/>
                                <XAxis dataKey="n" stroke={C.muted} fontSize={11}/>
                                <YAxis stroke={C.muted} fontSize={11} tickFormatter={v=>`${v.toFixed(1)}%`}/>
                                <Tooltip content={<ChartTip/>}/>
                                <Line type="monotone" dataKey="winRate" name="Win Rate" stroke={C.accent} strokeWidth={2} dot={false}/>
                              </LineChart>
                            </ResponsiveContainer>
                          </ChartBox>
                          <ChartBox label="Best Config Metrics">
                            <ResponsiveContainer width="100%" height={150}>
                              <BarChart data={barData} margin={{ top:4, right:4, bottom:0, left:-10 }}>
                                <CartesianGrid strokeDasharray="3 3" stroke={C.border}/>
                                <XAxis dataKey="name" stroke={C.muted} fontSize={10}/>
                                <YAxis stroke={C.muted} fontSize={11} tickFormatter={v=>`${v.toFixed(0)}%`}/>
                                <Tooltip contentStyle={{ background:C.surface, border:`1px solid ${C.border}`, borderRadius:8, fontSize:12 }} formatter={v=>`${(+v).toFixed(2)}%`}/>
                                <Bar dataKey="value" fill={C.accent} radius={[4,4,0,0]}/>
                              </BarChart>
                            </ResponsiveContainer>
                          </ChartBox>
                        </div>
                      </section>
                    )}

                    {iters.length > 0 && (
                      <section>
                        <Label>All Iterations ({iters.length})</Label>
                        <Card style={{ overflow:'hidden' }}>
                          <div style={{ overflowX:'auto' }}>
                            <table style={{ width:'100%', borderCollapse:'collapse', fontSize:13 }}>
                              <thead>
                                <tr style={{ background:C.surface }}>
                                  {['#','Config','CAGR','Drawdown','Win Rate','Expectancy','Change'].map(h => (
                                    <th key={h} style={{ padding:'10px 16px', textAlign:'left', color:C.muted, fontWeight:600, fontSize:11, textTransform:'uppercase', letterSpacing:'.05em', borderBottom:`1px solid ${C.border}`, whiteSpace:'nowrap' }}>{h}</th>
                                  ))}
                                </tr>
                              </thead>
                              <tbody>
                                {iters.map((it, i) => {
                                  const m   = it.metrics ?? {}
                                  const exp = optExpls[i-1] ?? ''
                                  return (
                                    <tr key={i} style={{ borderBottom:`1px solid ${C.border}` }}
                                      onMouseEnter={e => e.currentTarget.style.background=C.surface}
                                      onMouseLeave={e => e.currentTarget.style.background='transparent'}>
                                      <td style={{ padding:'10px 16px', color:C.muted, fontFamily:'monospace' }}>{it.iteration??i+1}</td>
                                      <td style={{ padding:'10px 16px', fontFamily:'monospace', fontSize:12, color:C.accent, maxWidth:180, overflow:'hidden', textOverflow:'ellipsis', whiteSpace:'nowrap' }}>{typeof it.config==='string'?it.config:JSON.stringify(it.config)}</td>
                                      <td style={{ padding:'10px 16px', fontFamily:'monospace', color:m.cagr>0?C.success:C.danger }}>{pct(m.cagr)}</td>
                                      <td style={{ padding:'10px 16px', fontFamily:'monospace', color:C.danger }}>{pct(m.max_drawdown)}</td>
                                      <td style={{ padding:'10px 16px', fontFamily:'monospace' }}>{pct(m.win_rate)}</td>
                                      <td style={{ padding:'10px 16px', fontFamily:'monospace', color:m.expectancy>0?C.success:C.danger }}>{dollar(m.expectancy)}</td>
                                      <td style={{ padding:'10px 16px', fontSize:11, color:C.muted, maxWidth:200, overflow:'hidden', textOverflow:'ellipsis', whiteSpace:'nowrap' }}>{exp || (i===0?'Baseline':'—')}</td>
                                    </tr>
                                  )
                                })}
                              </tbody>
                            </table>
                          </div>
                        </Card>
                      </section>
                    )}

                  </div>
                )
            )}

            {/* ── PORTFOLIO ───────────────────────────────────────── */}
            {activeTab === 'portfolio' && (
              <div style={{ display:'flex', flexDirection:'column', gap:22, maxWidth:760 }}>
                <section>
                  <Label>Starting Capital</Label>
                  <Card style={{ padding:'18px 20px' }}>
                    <div style={{ display:'flex', alignItems:'center', gap:14, flexWrap:'wrap' }}>
                      <span style={{ color:C.muted, fontSize:14 }}>Starting amount ($)</span>
                      <input type="number" value={startAmount} onChange={e => setStartAmount(Math.max(1, +e.target.value))} min={1}
                        style={{ background:C.surface, border:`1px solid ${C.border}`, borderRadius:8, padding:'8px 14px', color:C.text, fontSize:16, fontFamily:'monospace', fontWeight:700, width:180 }}/>
                      <div style={{ display:'flex', gap:6 }}>
                        {[10000,50000,100000,500000].map(v => (
                          <button key={v} onClick={() => setStartAmount(v)} style={{ background:startAmount===v?C.accent+'33':C.card, border:`1px solid ${startAmount===v?C.accent:C.border}`, borderRadius:6, padding:'5px 10px', color:startAmount===v?C.accent:C.muted, fontSize:12, cursor:'pointer' }}>
                            {v>=1000?`$${v/1000}k`:`$${v}`}
                          </button>
                        ))}
                      </div>
                    </div>
                  </Card>
                </section>

                {!results ? (
                  <div style={{ textAlign:'center', padding:'48px 0', color:C.muted }}>
                    <div style={{ fontSize:48, marginBottom:12 }}>💼</div>
                    <p>Run a backtest first to see portfolio results.</p>
                  </div>
                ) : (
                  <>
                    <section>
                      <Label>Portfolio Performance</Label>
                      <div style={{ display:'grid', gridTemplateColumns:'repeat(3,1fr)', gap:12 }}>
                        <MetricCard title="Starting Capital" display={dollar(startAmount)}   color={C.text}/>
                        <MetricCard title="Final Value"      display={dollar(scaledFinal)}   color={isProfit?C.success:C.danger}/>
                        <MetricCard title="Total P&L"        display={`${isProfit?'+':''}${dollar(profit)}`} color={isProfit?C.success:C.danger}/>
                        <MetricCard title="Total Return"     display={pct(totalReturnPct)}   color={sign(totalReturnPct)}/>
                        <MetricCard title="CAGR"             display={pct(bm.cagr)}          color={sign(bm.cagr)}/>
                        <MetricCard title="Max Drawdown"     display={pct(bm.max_drawdown)}  color={C.danger}/>
                      </div>
                    </section>

                    <section>
                      <Label>Portfolio vs {activeBmLabel} ({startLabel} → {endLabel})</Label>
                      <Card style={{ padding:'18px 20px' }}>
                        {mergedCurve.length > 1 ? (
                          <ResponsiveContainer width="100%" height={260}>
                            <AreaChart data={mergedCurve} margin={{ top:10, right:10, bottom:5, left:20 }}>
                              <defs>
                                <linearGradient id="eq" x1="0" y1="0" x2="0" y2="1"><stop offset="5%" stopColor={isProfit?C.success:C.danger} stopOpacity={.25}/><stop offset="95%" stopColor={isProfit?C.success:C.danger} stopOpacity={0}/></linearGradient>
                                <linearGradient id="bmg" x1="0" y1="0" x2="0" y2="1"><stop offset="5%" stopColor={C.gold} stopOpacity={.15}/><stop offset="95%" stopColor={C.gold} stopOpacity={0}/></linearGradient>
                              </defs>
                              <CartesianGrid strokeDasharray="3 3" stroke={C.border}/>
                              <XAxis dataKey="date" stroke={C.muted} fontSize={11} tickFormatter={d=>d.slice(0,4)} interval={Math.floor(mergedCurve.length/8)}/>
                              <YAxis stroke={C.muted} fontSize={11} tickFormatter={v=>`$${(v/1000).toFixed(0)}k`}/>
                              <Tooltip contentStyle={{ background:C.surface, border:`1px solid ${C.border}`, borderRadius:8, fontSize:12 }} formatter={(v,name)=>[dollar(v), name==='portfolio'?'Strategy':activeBmLabel]} labelStyle={{ color:C.muted }}/>
                              <ReferenceLine y={startAmount} stroke={C.muted} strokeDasharray="4 4" label={{ value:'Start', fill:C.muted, fontSize:11 }}/>
                              <Area type="monotone" dataKey="benchmark" name="benchmark" stroke={C.gold}    fill="url(#bmg)" strokeWidth={1.5} dot={false} strokeDasharray="5 3" connectNulls/>
                              <Area type="monotone" dataKey="portfolio" name="portfolio" stroke={isProfit?C.success:C.danger} fill="url(#eq)" strokeWidth={2} dot={false}/>
                            </AreaChart>
                          </ResponsiveContainer>
                        ) : (
                          <p style={{ color:C.muted, fontSize:13, textAlign:'center', padding:'32px 0' }}>No time-series data for this run.</p>
                        )}
                        <div style={{ display:'flex', gap:20, marginTop:12 }}>
                          <div style={{ display:'flex', alignItems:'center', gap:6 }}><div style={{ width:20, height:2, background:isProfit?C.success:C.danger, borderRadius:1 }}/><span style={{ fontSize:12, color:C.muted }}>Strategy</span></div>
                          <div style={{ display:'flex', alignItems:'center', gap:6 }}><div style={{ width:20, height:2, background:C.gold, borderRadius:1, opacity:.7 }}/><span style={{ fontSize:12, color:C.muted }}>{activeBmLabel}</span></div>
                        </div>
                        <div style={{ marginTop:16, padding:'14px 20px', borderRadius:10, background:isProfit?C.success+'15':C.danger+'15', border:`1px solid ${isProfit?C.success+'44':C.danger+'44'}`, display:'flex', alignItems:'center', justifyContent:'space-between' }}>
                          <span style={{ color:C.muted, fontSize:14 }}>{isProfit?'✅ Profitable strategy':'❌ Strategy lost money'}</span>
                          <span style={{ color:isProfit?C.success:C.danger, fontSize:18, fontWeight:700, fontFamily:'monospace' }}>{isProfit?'+':''}{dollar(profit)} ({pct(totalReturnPct)})</span>
                        </div>
                      </Card>
                    </section>

                    <section>
                      <Label>Risk Context</Label>
                      <Card style={{ padding:'16px 20px' }}>
                        <div style={{ display:'grid', gridTemplateColumns:'1fr 1fr', gap:16 }}>
                          <div><p style={{ fontSize:12, color:C.muted, marginBottom:4 }}>Worst-case drawdown on your capital</p><p style={{ fontSize:18, fontWeight:700, fontFamily:'monospace', color:C.danger }}>-{dollar(startAmount*Math.abs(+(bm.max_drawdown??0))/100)}</p></div>
                          <div><p style={{ fontSize:12, color:C.muted, marginBottom:4 }}>Expected gain per trade</p><p style={{ fontSize:18, fontWeight:700, fontFamily:'monospace', color:sign(bm.expectancy) }}>{dollar(bm.expectancy)}</p></div>
                          <div><p style={{ fontSize:12, color:C.muted, marginBottom:4 }}>Avg win per trade</p><p style={{ fontSize:18, fontWeight:700, fontFamily:'monospace', color:C.success }}>{dollar(bm.avg_win)}</p></div>
                          <div><p style={{ fontSize:12, color:C.muted, marginBottom:4 }}>Avg loss per trade</p><p style={{ fontSize:18, fontWeight:700, fontFamily:'monospace', color:C.danger }}>-{dollar(bm.avg_loss)}</p></div>
                        </div>
                      </Card>
                    </section>
                  </>
                )}
              </div>
            )}

            {/* ── CODE ─────────────────────────────────────────────── */}
            {activeTab === 'code' && (
              <div style={{ display:'flex', flexDirection:'column', gap:16 }}>
                <div style={{ display:'flex', alignItems:'center', justifyContent:'space-between' }}>
                  <Label>Generated Backtrader Strategy</Label>
                  {genCode && (
                    <button onClick={() => { const b=new Blob([genCode],{type:'text/x-python'}); const u=URL.createObjectURL(b); const a=document.createElement('a'); a.href=u; a.download=`strategy_${results?.strategy_id??'run'}.py`; a.click(); URL.revokeObjectURL(u) }}
                      style={{ background:C.purple+'22', border:`1px solid ${C.purple}55`, borderRadius:8, padding:'5px 12px', color:C.purple, fontSize:12, fontWeight:600, cursor:'pointer' }}>
                      ⬇ Download .py
                    </button>
                  )}
                </div>
                {!genCode ? (
                  <div style={{ textAlign:'center', padding:'48px 0', color:C.muted }}><div style={{ fontSize:48, marginBottom:12 }}>🧑‍💻</div><p>Run a backtest to see the generated code.</p></div>
                ) : (
                  <Card style={{ overflow:'hidden' }}>
                    <div style={{ display:'flex', alignItems:'center', justifyContent:'space-between', padding:'10px 16px', borderBottom:`1px solid ${C.border}`, background:C.surface }}>
                      <span style={{ fontSize:12, color:C.muted, fontFamily:'monospace' }}>strategy.py</span>
                      <div style={{ display:'flex', gap:5 }}>{['#ef4444','#f59e0b','#10b981'].map(c=><div key={c} style={{ width:10, height:10, borderRadius:'50%', background:c }}/>)}</div>
                    </div>
                    <pre style={{ padding:'16px 18px', margin:0, fontFamily:"'JetBrains Mono','Fira Code','Courier New',monospace", fontSize:13, lineHeight:1.65, color:'#a5f3fc', background:'transparent', overflowX:'auto', whiteSpace:'pre' }}>{genCode}</pre>
                  </Card>
                )}
              </div>
            )}

            {/* ── HISTORY ───────────────────────────────────────────── */}
            {activeTab === 'history' && (
              <div style={{ display:'flex', flexDirection:'column', gap:16 }}>
                <div style={{ display:'flex', alignItems:'center', justifyContent:'space-between', marginBottom:2 }}>
                  <Label>Past Runs</Label>
                  <button onClick={fetchHistory} style={{ background:C.card, border:`1px solid ${C.border}`, borderRadius:8, padding:'5px 12px', color:C.muted, fontSize:12, cursor:'pointer' }}>↻ Refresh</button>
                </div>
                {history===null ? <div style={{ textAlign:'center', padding:'32px 0', color:C.muted }}>Loading…</div>
                 : history.length===0 ? <div style={{ textAlign:'center', padding:'48px 0', color:C.muted }}><div style={{ fontSize:48, marginBottom:12 }}>🕑</div><p>No past runs yet.</p></div>
                 : history.map(h => (
                    <Card key={h.id} style={{ padding:'16px 18px' }}>
                      <div style={{ display:'flex', alignItems:'flex-start', justifyContent:'space-between', gap:12 }}>
                        <div style={{ flex:1 }}>
                          <p style={{ fontSize:13.5, color:C.text, marginBottom:5, fontWeight:500 }}>{h.prompt}</p>
                          <div style={{ display:'flex', gap:8, flexWrap:'wrap', marginBottom:4 }}>
                            <Badge label={h.data_source==='kaggle'?'Kaggle':'Yahoo Finance'} color={h.data_source==='kaggle'?C.purple:C.accent}/>
                            {h.is_multi_stock && <Badge label="Multi-stock" color={C.warning}/>}
                          </div>
                          <p style={{ fontSize:11, color:C.muted }}>{h.timestamp}</p>
                        </div>
                        <span style={{ background:C.surface, border:`1px solid ${C.border}`, borderRadius:6, padding:'2px 8px', color:C.muted, fontSize:11, flexShrink:0 }}>#{h.id}</span>
                      </div>
                      {h.best_cagr!=null && (
                        <div style={{ display:'flex', gap:20, marginTop:12, flexWrap:'wrap' }}>
                          <Stat label="CAGR"       value={pct(h.best_cagr)}          color={sign(h.best_cagr)}/>
                          <Stat label="Drawdown"   value={pct(h.best_drawdown)}      color={C.danger}/>
                          <Stat label="Win Rate"   value={pct(h.best_win_rate)}      color={C.accent}/>
                          <Stat label="Expectancy" value={dollar(h.best_expectancy)} color={sign(h.best_expectancy)}/>
                        </div>
                      )}
                    </Card>
                  ))
                }
              </div>
            )}

            {/* ── SETTINGS ─────────────────────────────────────────── */}
            {activeTab === 'settings' && (
              <div style={{ display:'flex', flexDirection:'column', gap:22, maxWidth:600 }}>

                <section>
                  <Label>Benchmark Comparison</Label>
                  <Card style={{ padding:'20px 22px' }}>
                    <p style={{ fontSize:13.5, color:C.muted, marginBottom:16, lineHeight:1.6 }}>
                      Choose the ticker to compare against on the Portfolio equity curve.
                    </p>
                    <div style={{ display:'flex', gap:10, alignItems:'center', flexWrap:'wrap' }}>
                      <input value={benchmarkInput} onChange={e=>setBenchmarkInput(e.target.value.toUpperCase())} onKeyDown={e=>{ if(e.key==='Enter') setBenchmarkTicker(benchmarkInput) }} placeholder="e.g. SPY" maxLength={8}
                        style={{ background:C.surface, border:`1px solid ${C.border}`, borderRadius:8, padding:'9px 14px', color:C.text, fontSize:15, fontFamily:'monospace', fontWeight:700, width:120 }}/>
                      <button onClick={() => setBenchmarkTicker(benchmarkInput)} style={{ background:`linear-gradient(135deg,${C.accent},${C.purple})`, border:'none', borderRadius:8, padding:'9px 18px', color:'#fff', fontWeight:600, fontSize:13, cursor:'pointer' }}>Set</button>
                      <span style={{ fontSize:12, color:C.muted }}>Current: <span style={{ color:C.gold, fontFamily:'monospace', fontWeight:600 }}>{benchmarkTicker}</span></span>
                    </div>
                    <div style={{ marginTop:16, display:'flex', gap:8, flexWrap:'wrap' }}>
                      {[{ticker:'SPY',label:'S&P 500'},{ticker:'QQQ',label:'Nasdaq'},{ticker:'DIA',label:'Dow'},{ticker:'IWM',label:'Russell'},{ticker:'GLD',label:'Gold'},{ticker:'BTC-USD',label:'Bitcoin'}].map(({ticker,label})=>(
                        <button key={ticker} onClick={()=>{setBenchmarkInput(ticker);setBenchmarkTicker(ticker)}} style={{ background:benchmarkTicker===ticker?C.gold+'22':C.card, border:`1px solid ${benchmarkTicker===ticker?C.gold:C.border}`, borderRadius:8, padding:'6px 14px', cursor:'pointer', display:'flex', flexDirection:'column', alignItems:'center', gap:2 }}>
                          <span style={{ fontFamily:'monospace', fontSize:13, fontWeight:700, color:benchmarkTicker===ticker?C.gold:C.text }}>{ticker}</span>
                          <span style={{ fontSize:10, color:C.muted }}>{label}</span>
                        </button>
                      ))}
                    </div>
                    <p style={{ fontSize:12, color:C.muted, marginTop:16, fontStyle:'italic' }}>Changes take effect on the next backtest run.</p>
                  </Card>
                </section>

                <section>
                  <Label>Data Source Guide</Label>
                  <Card style={{ padding:'18px 20px' }}>
                    <div style={{ display:'grid', gridTemplateColumns:'1fr 1fr', gap:14 }}>
                      <div style={{ padding:'14px 16px', background:C.surface, borderRadius:10, border:`1px solid ${C.border}` }}>
                        <p style={{ fontWeight:700, color:C.accent, marginBottom:6 }}>Yahoo Finance</p>
                        <p style={{ fontSize:12, color:C.muted, lineHeight:1.6 }}>Live data fetched on demand. Covers all current stocks and ETFs. Best for recent tickers.</p>
                      </div>
                      <div style={{ padding:'14px 16px', background:C.surface, borderRadius:10, border:`1px solid ${C.border}` }}>
                        <p style={{ fontWeight:700, color:C.purple, marginBottom:6 }}>Kaggle Local</p>
                        <p style={{ fontSize:12, color:C.muted, lineHeight:1.6 }}>Pre-downloaded CSV files. Fast loading. ~7,500 US stocks & ETFs. Data up to ~2017.</p>
                      </div>
                    </div>
                  </Card>
                </section>

                <section>
                  <Label>Risk Profiles</Label>
                  <Card style={{ padding:'18px 20px' }}>
                    <div style={{ display:'flex', flexDirection:'column', gap:10 }}>
                      {[
                        { level:'Conservative', color:C.success, sl:'2%', tp:'6%',  size:'30%', dd:'10%' },
                        { level:'Moderate',     color:C.warning, sl:'5%', tp:'10%', size:'50%', dd:'20%' },
                        { level:'Aggressive',   color:C.danger,  sl:'8%', tp:'20%', size:'95%', dd:'35%' },
                      ].map(p => (
                        <div key={p.level} style={{ padding:'12px 16px', background:C.surface, borderRadius:10, border:`1px solid ${p.color}33` }}>
                          <div style={{ display:'flex', justifyContent:'space-between', alignItems:'center', marginBottom:8 }}>
                            <span style={{ fontWeight:700, color:p.color }}>{p.level}</span>
                          </div>
                          <div style={{ display:'flex', gap:16, flexWrap:'wrap' }}>
                            <Stat label="Stop Loss"    value={p.sl}   color={C.danger}/>
                            <Stat label="Take Profit"  value={p.tp}   color={C.success}/>
                            <Stat label="Position Size" value={p.size} color={C.accent}/>
                            <Stat label="Max Drawdown" value={p.dd}   color={C.warning}/>
                          </div>
                        </div>
                      ))}
                    </div>
                  </Card>
                </section>

              </div>
            )}

          </div>{/* end tab content */}
        </div>{/* end right panel */}

      </div>{/* end body */}
    </div>
  )
}

// ── Empty state ───────────────────────────────────────────────────────────────

function EmptyState({ onSelect }) {
  const examples = [
    'Trade AAPL using a 15/50 EMA crossover',
    'SPY RSI(14) oversold below 30 strategy',
    'Each week trade the highest-volume NASDAQ stock',
  ]
  return (
    <div style={{ display:'flex', flexDirection:'column', alignItems:'center', justifyContent:'center', minHeight:'70vh', gap:14, color:C.muted, textAlign:'center' }}>
      <div style={{ fontSize:56 }}>📊</div>
      <h2 style={{ fontSize:19, fontWeight:600, color:C.text }}>Dashboard</h2>
      <p style={{ fontSize:13.5, maxWidth:380, lineHeight:1.65 }}>
        Fill in the setup panel on the left, or pick a quick-start below.
      </p>
      <div style={{ display:'flex', flexDirection:'column', gap:6, marginTop:6, width:'100%', maxWidth:340 }}>
        {examples.map(s => (
          <button key={s} onClick={() => onSelect(s)}
            style={{ background:C.card, border:`1px solid ${C.border}`, borderRadius:8, padding:'8px 14px', color:C.muted, fontSize:13, cursor:'pointer', textAlign:'left' }}
            onMouseEnter={e => e.currentTarget.style.color = C.text}
            onMouseLeave={e => e.currentTarget.style.color = C.muted}
          >{s}</button>
        ))}
      </div>
    </div>
  )
}