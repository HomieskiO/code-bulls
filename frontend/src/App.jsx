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

// Values from backend are already in % form (cagr=15 means 15%) or dollars
const pct    = v => v != null ? `${(+v).toFixed(2)}%`   : 'N/A'
const dollar = v => v != null ? `$${(+v).toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}` : 'N/A'
const num    = v => v != null ? `${(+v).toFixed(2)}`    : 'N/A'
const sign   = v => (+v) > 0 ? C.success : C.danger

// ─── Small reusables ─────────────────────────────────────────────────────────

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

// ─── Metric Card ─────────────────────────────────────────────────────────────

function MetricCard({ title, value, display, color }) {
  const shown = display ?? value ?? 'N/A'
  const col   = color ?? C.text
  return (
    <div style={{ background: C.card, border: `1px solid ${C.border}`, borderRadius: 12, padding: '14px 18px' }}>
      <p style={{ color: C.muted, fontSize: 11, textTransform: 'uppercase', letterSpacing: '.06em', marginBottom: 6 }}>
        {title}
      </p>
      <p style={{ color: col, fontSize: 21, fontWeight: 700, fontFamily: 'monospace', letterSpacing: '-.02em' }}>
        {shown}
      </p>
    </div>
  )
}

// ─── Chart Tooltip ────────────────────────────────────────────────────────────

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

// ─── Chat Message ─────────────────────────────────────────────────────────────

function Msg({ m }) {
  const isUser = m.type === 'user'
  return (
    <div style={{ display: 'flex', justifyContent: isUser ? 'flex-end' : 'flex-start', marginBottom: 10 }}>
      {!isUser && (
        <div style={{
          width: 26, height: 26, borderRadius: '50%', flexShrink: 0,
          background: `linear-gradient(135deg,${C.accent},${C.purple})`,
          display: 'flex', alignItems: 'center', justifyContent: 'center',
          fontSize: 11, fontWeight: 700, color: '#fff', marginRight: 8, marginTop: 2,
        }}>AI</div>
      )}
      <div style={{
        maxWidth: '82%',
        background: isUser ? `linear-gradient(135deg,#1e3a8a,#2563eb)` : C.card,
        border: `1px solid ${isUser ? 'transparent' : C.border}`,
        borderRadius: isUser ? '14px 14px 4px 14px' : '14px 14px 14px 4px',
        padding: '9px 13px', color: C.text, fontSize: 13.5, lineHeight: 1.55,
        whiteSpace: 'pre-wrap',
      }}>
        {m.content}
        {m.thinking && (
          <span style={{ display: 'inline-flex', gap: 4, marginLeft: 8, verticalAlign: 'middle' }}>
            {[0, 1, 2].map(i => (
              <span key={i} style={{
                width: 5, height: 5, borderRadius: '50%', background: C.accent,
                display: 'inline-block',
                animation: `dot .9s ${i * 0.22}s ease-in-out infinite alternate`,
              }} />
            ))}
          </span>
        )}
      </div>
      {isUser && (
        <div style={{
          width: 26, height: 26, borderRadius: '50%', flexShrink: 0,
          background: '#1e3a8a', display: 'flex', alignItems: 'center', justifyContent: 'center',
          fontSize: 11, fontWeight: 700, color: '#fff', marginLeft: 8, marginTop: 2,
        }}>U</div>
      )}
    </div>
  )
}

// ─── Download helper ──────────────────────────────────────────────────────────

function downloadJSON(data, filename) {
  const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' })
  const url  = URL.createObjectURL(blob)
  const a    = document.createElement('a')
  a.href = url; a.download = filename; a.click()
  URL.revokeObjectURL(url)
}

// ─── Main App ─────────────────────────────────────────────────────────────────

// Dataset (Kaggle Stock Market) ends on 2017-11-10 — hard max for all backtests
const DATASET_MAX_END = '2017-11-10'
const DEFAULT_START   = '2010-01-01'
const DEFAULT_END     = DATASET_MAX_END

export default function App() {
  const [mode,             setMode]             = useState('single')   // 'single' | 'screened'
  const [messages,         setMessages]         = useState([{
    type: 'ai',
    content: "Welcome to AlgoTrader AI.\n\nDescribe your trading strategy, pick a start & end date, and I'll backtest and optimize it for you.",
  }])
  const [input,            setInput]            = useState("Trade AAPL. Buy when the 15-day EMA crosses above the 50-day EMA. Sell when it crosses below.")
  const [screeningInput,   setScreeningInput]   = useState("Top 1% of stocks with the biggest price move over the past 1 month")
  const [strategyInput,    setStrategyInput]    = useState("Buy when the stock reaches a new all-time high (ATH). Sell when it closes below the 50-day SMA.")
  const [startDate,        setStartDate]        = useState(DEFAULT_START)
  const [endDate,          setEndDate]          = useState(DEFAULT_END)
  // Position sizing: % of available cash per trade (defaults differ by mode)
  const [positionSizeSingle, setPositionSizeSingle] = useState(100)
  const [positionSizeMulti,  setPositionSizeMulti]  = useState(10)
  const [loading,          setLoading]          = useState(false)
  const [results,          setResults]          = useState(null)
  const [activeTab,        setActiveTab]        = useState('dashboard')
  const [startAmount,      setStartAmount]      = useState(100000)
  const [history,          setHistory]          = useState(null)
  const [benchmarkTicker,  setBenchmarkTicker]  = useState('SPY')
  const [benchmarkInput,   setBenchmarkInput]   = useState('SPY')
  const [debugData,        setDebugData]        = useState(null)   // partial data on failed runs
  const chatEnd = useRef(null)

  const clampEnd = (d) => (d && d > DATASET_MAX_END ? DATASET_MAX_END : d)
  const datesValid = Boolean(
    startDate && endDate && startDate < endDate && endDate <= DATASET_MAX_END
  )
  const positionSizePct = mode === 'screened' ? positionSizeMulti : positionSizeSingle
  const setPositionSizePct = mode === 'screened' ? setPositionSizeMulti : setPositionSizeSingle
  const positionSizeValid = positionSizePct >= 1 && positionSizePct <= 100

  useEffect(() => { chatEnd.current?.scrollIntoView({ behavior: 'smooth' }) }, [messages])

  // Load history when tab selected
  useEffect(() => {
    if (activeTab === 'history' && history === null) fetchHistory()
  }, [activeTab])

  const fetchHistory = async () => {
    try {
      const res  = await fetch('/api/history')
      const data = await res.json()
      setHistory(data)
    } catch {
      setHistory([])
    }
  }

  const run = async () => {
    const isScreened = mode === 'screened'
    const canRun = isScreened
      ? (screeningInput.trim() && strategyInput.trim())
      : input.trim()
    if (!canRun || loading) return

    if (!startDate || !endDate) {
      setMessages(prev => [...prev, { type: 'ai', content: 'Please set both a start date and an end date before running.' }])
      return
    }
    if (startDate >= endDate) {
      setMessages(prev => [...prev, { type: 'ai', content: 'Start date must be before end date.' }])
      return
    }
    if (endDate > DATASET_MAX_END) {
      setMessages(prev => [...prev, {
        type: 'ai',
        content: `End date cannot be after ${DATASET_MAX_END} (last available date in the market dataset).`,
      }])
      return
    }
    if (!positionSizeValid) {
      setMessages(prev => [...prev, {
        type: 'ai',
        content: 'Position size must be between 1% and 100% of available cash.',
      }])
      return
    }

    const periodStart = startDate
    const periodEnd   = clampEnd(endDate)
    const stakePct    = Math.min(100, Math.max(1, Number(positionSizePct) || (isScreened ? 10 : 100)))

    setLoading(true)

    if (isScreened) {
      const userMsg =
        `[Screening] ${screeningInput.trim()}\n[Strategy] ${strategyInput.trim()}\n` +
        `[Period] ${periodStart} → ${periodEnd}\n` +
        `[Position size] ${stakePct}% of available cash per trade`
      setMessages(prev => [
        ...prev,
        { type: 'user', content: userMsg },
        { type: 'ai',   content: 'Generating screener code, running screening, then backtesting…', thinking: true },
      ])
      try {
        const res  = await fetch('/api/screen-backtest', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            strategy_prompt:    strategyInput.trim(),
            screening_prompt:   screeningInput.trim(),
            start_date:         periodStart,
            end_date:           periodEnd,
            benchmark_ticker:   benchmarkTicker,
            position_size_pct:  stakePct,
          }),
        })
        const data = await res.json()
        if (!res.ok || data.error) {
          if (data.generated_code || data.screening_code) {
            setDebugData({ error: data.error || `HTTP ${res.status}`, generated_code: data.generated_code || '', screening_code: data.screening_code || '' })
            setActiveTab('code')
          }
          throw new Error(data.error || `HTTP ${res.status}`)
        }

        setDebugData(null)
        setResults({ ...data, _mode: 'screened' })
        setActiveTab('dashboard')
        setHistory(null)

        const m = data.best_configuration?.metrics ?? {}
        const n = data.all_iterations?.length ?? 0
        const s = data.screening_summary ?? {}
        const period = data.period ?? { start: periodStart, end: periodEnd }
        setMessages(prev => [
          ...prev.slice(0, -1),
          {
            type: 'ai',
            content:
              `Screened backtest complete after ${n} iteration${n !== 1 ? 's' : ''}.\n\n` +
              `Period: ${period.start} → ${period.end}\n` +
              `Tickers traded: ${s.unique_tickers?.length ?? 0} unique symbols\n` +
              `CAGR: ${pct(m.cagr)} | Drawdown: ${pct(m.max_drawdown)} | Win Rate: ${pct(m.win_rate)}\n\n` +
              `Full results, charts & explanation are on the right →`,
          },
        ])
      } catch (err) {
        setMessages(prev => [
          ...prev.slice(0, -1),
          { type: 'ai', content: `Error: ${err.message}` },
        ])
      } finally {
        setLoading(false)
      }
    } else {
      const prompt = input.trim()
      setInput('')
      setMessages(prev => [
        ...prev,
        {
          type: 'user',
          content:
            `${prompt}\n[Period] ${periodStart} → ${periodEnd}\n` +
            `[Position size] ${stakePct}% of available cash per trade`,
        },
        { type: 'ai',   content: 'Running backtest & optimizing…', thinking: true },
      ])
      try {
        const res  = await fetch('/api/backtest', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            prompt,
            start_date: periodStart,
            end_date: periodEnd,
            benchmark_ticker: benchmarkTicker,
            position_size_pct: stakePct,
          }),
        })
        const data = await res.json()
        if (!res.ok || data.error) {
          if (data.generated_code) {
            setDebugData({ error: data.error || `HTTP ${res.status}`, generated_code: data.generated_code, screening_code: '' })
            setActiveTab('code')
          }
          throw new Error(data.error || `HTTP ${res.status}`)
        }

        setDebugData(null)
        setResults({ ...data, _mode: 'single' })
        setActiveTab('dashboard')
        setHistory(null)

        const m = data.best_configuration?.metrics ?? {}
        const n = data.all_iterations?.length ?? 0
        const period = data.period ?? { start: periodStart, end: periodEnd }
        setMessages(prev => [
          ...prev.slice(0, -1),
          {
            type: 'ai',
            content: `Optimization complete after ${n} iteration${n !== 1 ? 's' : ''}.\n\n` +
              `Period: ${period.start} → ${period.end}\n` +
              `CAGR: ${pct(m.cagr)} | Drawdown: ${pct(m.max_drawdown)} | Win Rate: ${pct(m.win_rate)}\n\n` +
              `Full results, charts & explanation are on the right →`,
          },
        ])
      } catch (err) {
        setMessages(prev => [
          ...prev.slice(0, -1),
          { type: 'ai', content: `Error: ${err.message}` },
        ])
      } finally {
        setLoading(false)
      }
    }
  }

  // Derived data
  const bm        = results?.best_configuration?.metrics ?? {}
  const bestConf  = results?.best_configuration?.config  ?? {}
  const iters     = results?.all_iterations ?? []
  const genCode   = results?.generated_code ?? ''
  const explain   = results?.explanation    ?? ''

  const chartData = iters.map((it, i) => ({
    n:        it.iteration ?? i + 1,
    cagr:     +(it.metrics?.cagr          ?? 0),
    drawdown: +(it.metrics?.max_drawdown   ?? 0),
    winRate:  +(it.metrics?.win_rate       ?? 0),
    expect:   +(it.metrics?.expectancy     ?? 0),
  }))

  const barData = results ? [
    { name: 'CAGR',     value: +(bm.cagr        ?? 0) },
    { name: 'Win Rate', value: +(bm.win_rate      ?? 0) },
    { name: 'Drawdown', value: Math.abs(+(bm.max_drawdown ?? 0)) },
  ] : []

  // Portfolio tab calculations
  const totalReturnPct = +(bm.total_return_pct ?? 0)
  const scaledFinal    = startAmount * (1 + totalReturnPct / 100)
  const profit         = scaledFinal - startAmount
  const isProfit       = profit >= 0

  // Equity curve — scale backend's $100k values to user's startAmount
  const scaleFactor = startAmount / 100_000
  const equityCurve = (bm.portfolio_values ?? []).map(d => ({
    date:  d.date.slice(0, 7),
    value: Math.round(d.value * scaleFactor),
  }))

  // Benchmark curve — same scaling
  const bmMap = new Map(
    (results?.benchmark_values ?? []).map(d => [d.date.slice(0, 7), Math.round(d.value * scaleFactor)])
  )
  const mergedCurve = equityCurve.map(p => ({
    date:      p.date,
    portfolio: p.value,
    benchmark: bmMap.get(p.date) ?? null,
  }))

  const activeBenchmarkLabel = results?.benchmark_ticker ?? benchmarkTicker
  const startLabel = mergedCurve[0]?.date    ?? 'Start'
  const endLabel   = mergedCurve.at(-1)?.date ?? 'End'

  return (
    <div style={{
      height: '100vh', background: C.bg, color: C.text, overflow: 'hidden',
      display: 'flex', flexDirection: 'column',
      fontFamily: "'Inter', -apple-system, BlinkMacSystemFont, sans-serif",
    }}>
      <style>{`
        @keyframes dot  { from { opacity:.3;transform:scale(.7) } to { opacity:1;transform:scale(1) } }
        @keyframes spin { to   { transform:rotate(360deg) } }
        *, *::before, *::after { box-sizing:border-box; margin:0; padding:0 }
        ::-webkit-scrollbar       { width:4px; height:4px }
        ::-webkit-scrollbar-thumb { background:${C.border}; border-radius:4px }
        ::-webkit-scrollbar-track { background:transparent }
        textarea, input, button { font-family:inherit; outline:none }
      `}</style>

      {/* ── Header ────────────────────────────────────────────────────── */}
      <header style={{
        display: 'flex', alignItems: 'center', justifyContent: 'space-between',
        padding: '11px 22px', background: C.surface, borderBottom: `1px solid ${C.border}`, flexShrink: 0,
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 11 }}>
          <div style={{
            width: 34, height: 34, borderRadius: 9,
            background: `linear-gradient(135deg,${C.accent},${C.purple})`,
            display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: 18,
          }}>📈</div>
          <div>
            <h1 style={{ fontSize: 15, fontWeight: 700, letterSpacing: '-.02em' }}>AlgoTrader AI</h1>
            <p style={{ fontSize: 11, color: C.muted }}>AI-Powered Strategy Backtesting & Optimization</p>
          </div>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          <div style={{
            width: 8, height: 8, borderRadius: '50%',
            background: results ? C.success : loading ? C.warning : C.muted,
            boxShadow: results ? `0 0 8px ${C.success}88` : loading ? `0 0 8px ${C.warning}88` : 'none',
          }} />
          <span style={{ fontSize: 12, color: C.muted }}>
            {loading ? 'Running…' : results ? 'Results ready' : 'Idle'}
          </span>
        </div>
      </header>

      {/* ── Body ──────────────────────────────────────────────────────── */}
      <div style={{ display: 'flex', flex: 1, overflow: 'hidden' }}>

        {/* ── Chat Panel ────────────────────────────────────────────── */}
        <div style={{
          width: 340, display: 'flex', flexDirection: 'column',
          borderRight: `1px solid ${C.border}`, background: C.surface, flexShrink: 0,
        }}>
          <div style={{ flex: 1, overflowY: 'auto', padding: '14px 10px' }}>
            {messages.map((m, i) => <Msg key={i} m={m} />)}
            <div ref={chatEnd} />
          </div>

          <div style={{ padding: 10, borderTop: `1px solid ${C.border}`, background: C.bg }}>

            {/* Mode switcher */}
            <div style={{ display: 'flex', gap: 4, marginBottom: 8 }}>
              {[
                { id: 'single',   label: 'Single Stock' },
                { id: 'screened', label: 'Screened Multi-Stock' },
              ].map(m => (
                <button
                  key={m.id}
                  onClick={() => setMode(m.id)}
                  style={{
                    flex: 1, padding: '6px 0', border: 'none', borderRadius: 8,
                    fontFamily: 'inherit', fontSize: 12, fontWeight: 600, cursor: 'pointer',
                    background: mode === m.id ? C.accent + '33' : C.card,
                    color: mode === m.id ? C.accent : C.muted,
                    borderBottom: mode === m.id ? `2px solid ${C.accent}` : `2px solid ${C.border}`,
                    transition: 'all .15s',
                  }}
                >{m.label}</button>
              ))}
            </div>

            {/* Backtest period — required for every strategy */}
            <DateRangePicker
              startDate={startDate}
              endDate={endDate}
              onStart={d => setStartDate(d)}
              onEnd={d => setEndDate(clampEnd(d))}
              disabled={loading}
              maxEnd={DATASET_MAX_END}
            />

            {/* Position sizing — % of available cash per trade */}
            <PositionSizeInput
              value={positionSizePct}
              onChange={setPositionSizePct}
              disabled={loading}
              mode={mode}
            />

            {mode === 'single' ? (
              /* ── Single-stock input ── */
              <>
                <div style={{ background: C.card, border: `1px solid ${C.border}`, borderRadius: 12, overflow: 'hidden' }}>
                  <textarea
                    value={input}
                    onChange={e => setInput(e.target.value)}
                    onKeyDown={e => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); run() } }}
                    placeholder="Describe your trading strategy…"
                    disabled={loading}
                    rows={3}
                    style={{
                      width: '100%', padding: '11px 13px',
                      background: 'transparent', border: 'none',
                      color: C.text, fontSize: 13.5, resize: 'none', lineHeight: 1.5,
                    }}
                  />
                  <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', padding: '7px 11px', borderTop: `1px solid ${C.border}` }}>
                    <span style={{ fontSize: 11, color: C.muted }}>↵ send · ⇧↵ newline</span>
                    <RunButton loading={loading} disabled={!input.trim() || !datesValid || !positionSizeValid} onClick={run} />
                  </div>
                </div>
                <div style={{ marginTop: 8, display: 'flex', flexDirection: 'column', gap: 4 }}>
                  {[
                    'AAPL 15/50 EMA crossover',
                    'SPY RSI(14) oversold <30 strategy',
                    'TSLA Bollinger Bands breakout',
                  ].map(s => (
                    <button key={s} onClick={() => setInput(s)} style={{
                      background: C.card, border: `1px solid ${C.border}`, borderRadius: 8,
                      padding: '6px 12px', color: C.muted, fontSize: 12, textAlign: 'left', cursor: 'pointer',
                    }}
                      onMouseEnter={e => e.currentTarget.style.color = C.text}
                      onMouseLeave={e => e.currentTarget.style.color = C.muted}
                    >{s}</button>
                  ))}
                </div>
              </>
            ) : (
              /* ── Screened multi-stock input ── */
              <>
                <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
                  <div style={{ background: C.card, border: `1px solid ${C.border}`, borderRadius: 12, overflow: 'hidden' }}>
                    <p style={{ padding: '7px 13px 0', fontSize: 10, fontWeight: 700, color: C.accent, textTransform: 'uppercase', letterSpacing: '.06em' }}>
                      Stock Screening Logic
                    </p>
                    <textarea
                      value={screeningInput}
                      onChange={e => setScreeningInput(e.target.value)}
                      placeholder="e.g. top 1% of stocks by 1-month return"
                      disabled={loading}
                      rows={2}
                      style={{
                        width: '100%', padding: '6px 13px 10px',
                        background: 'transparent', border: 'none',
                        color: C.text, fontSize: 13, resize: 'none', lineHeight: 1.5,
                      }}
                    />
                  </div>
                  <div style={{ background: C.card, border: `1px solid ${C.border}`, borderRadius: 12, overflow: 'hidden' }}>
                    <p style={{ padding: '7px 13px 0', fontSize: 10, fontWeight: 700, color: C.purple, textTransform: 'uppercase', letterSpacing: '.06em' }}>
                      Trading Strategy
                    </p>
                    <textarea
                      value={strategyInput}
                      onChange={e => setStrategyInput(e.target.value)}
                      onKeyDown={e => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); run() } }}
                      placeholder="e.g. buy on new ATH, sell when close < 50 SMA"
                      disabled={loading}
                      rows={2}
                      style={{
                        width: '100%', padding: '6px 13px 10px',
                        background: 'transparent', border: 'none',
                        color: C.text, fontSize: 13, resize: 'none', lineHeight: 1.5,
                      }}
                    />
                  </div>
                  <div style={{ display: 'flex', justifyContent: 'flex-end' }}>
                    <RunButton loading={loading} disabled={!screeningInput.trim() || !strategyInput.trim() || !datesValid || !positionSizeValid} onClick={run} />
                  </div>
                </div>
                <div style={{ marginTop: 8, display: 'flex', flexDirection: 'column', gap: 4 }}>
                  {[
                    { screen: 'Top 1% stocks by 1-month return', strategy: 'Buy when the stock reaches a new ATH. Sell when it closes below the 50-day SMA.' },
                    { screen: 'Stocks with 3x average volume spike', strategy: 'Buy when screened. Sell when close drops below the 20-day EMA.' },
                    { screen: 'Stocks breaking above their 52-week high', strategy: 'Buy breakout when screened. Sell when price drops below EMA 20.' },
                  ].map(({ screen, strategy }) => (
                    <button key={screen} onClick={() => { setScreeningInput(screen); setStrategyInput(strategy) }} style={{
                      background: C.card, border: `1px solid ${C.border}`, borderRadius: 8,
                      padding: '6px 12px', color: C.muted, fontSize: 11, textAlign: 'left', cursor: 'pointer',
                    }}
                      onMouseEnter={e => e.currentTarget.style.color = C.text}
                      onMouseLeave={e => e.currentTarget.style.color = C.muted}
                    >
                      <span style={{ color: C.accent }}>Screen:</span> {screen}<br />
                      <span style={{ color: C.purple }}>Strategy:</span> {strategy}
                    </button>
                  ))}
                </div>
              </>
            )}
          </div>
        </div>

        {/* ── Right Panel ───────────────────────────────────────────── */}
        <div style={{ flex: 1, display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>

          {/* Tab bar */}
          <div style={{
            display: 'flex', gap: 4, padding: '10px 20px',
            borderBottom: `1px solid ${C.border}`, background: C.surface, flexShrink: 0,
          }}>
            {[
              { id: 'dashboard', label: '📊 Dashboard' },
              { id: 'portfolio', label: '💼 Portfolio' },
              { id: 'trades',    label: '📋 Trades' },
              { id: 'code',      label: '🧑‍💻 Strategy Code' },
              { id: 'history',   label: '🕑 History' },
              { id: 'settings',  label: '⚙️ Settings' },
            ].map(t => (
              <Tab key={t.id} label={t.label} active={activeTab === t.id} onClick={() => setActiveTab(t.id)} />
            ))}
          </div>

          {/* Tab content */}
          <div style={{ flex: 1, overflowY: 'auto', padding: 22 }}>

            {/* ── DASHBOARD TAB ─────────────────────────────────────── */}
            {activeTab === 'dashboard' && (
              !results ? (
                <EmptyState onSelect={setInput} />
              ) : (
                <div style={{ display: 'flex', flexDirection: 'column', gap: 22 }}>

                  {/* Backtest period + position sizing */}
                  {(results.period || startDate) && (
                    <section>
                      <Label>Backtest Period & Position Size</Label>
                      <Card style={{ padding: '14px 18px' }}>
                        <div style={{ display: 'flex', alignItems: 'center', gap: 16, flexWrap: 'wrap' }}>
                          <div>
                            <p style={{ fontSize: 11, color: C.muted, marginBottom: 4 }}>Start</p>
                            <p style={{ fontFamily: 'monospace', fontWeight: 700, color: C.accent, fontSize: 15 }}>
                              {results.period?.start ?? startDate}
                            </p>
                          </div>
                          <span style={{ color: C.muted, fontSize: 18 }}>→</span>
                          <div>
                            <p style={{ fontSize: 11, color: C.muted, marginBottom: 4 }}>End</p>
                            <p style={{ fontFamily: 'monospace', fontWeight: 700, color: C.purple, fontSize: 15 }}>
                              {results.period?.end ?? endDate}
                            </p>
                          </div>
                          <div>
                            <p style={{ fontSize: 11, color: C.muted, marginBottom: 4 }}>Position size</p>
                            <p style={{ fontFamily: 'monospace', fontWeight: 700, color: C.gold, fontSize: 15 }}>
                              {results.position_size_pct ?? positionSizePct}% cash / trade
                            </p>
                          </div>
                          <p style={{ fontSize: 12, color: C.muted, marginLeft: 'auto', maxWidth: 260, lineHeight: 1.5 }}>
                            Metrics are over this window. Each entry uses the stake % of available cash.
                          </p>
                        </div>
                      </Card>
                    </section>
                  )}

                  {/* Screening Summary (screened mode only) */}
                  {results._mode === 'screened' && results.screening_summary && (
                    <section>
                      <Label>Screening Summary</Label>
                      <Card style={{ padding: '14px 18px' }}>
                        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3,1fr)', gap: 12 }}>
                          <MetricCard title="Unique Tickers" display={results.screening_summary.unique_tickers?.length ?? 0} color={C.accent} />
                          <MetricCard title="Ticker-Day Pairs" display={(results.screening_summary.total_ticker_days ?? 0).toLocaleString()} color={C.purple} />
                          <MetricCard title="Date Range"
                            display={`${results.screening_summary.date_range?.start?.slice(0,7) ?? '?'} → ${results.screening_summary.date_range?.end?.slice(0,7) ?? '?'}`}
                            color={C.muted}
                          />
                        </div>
                        {results.screening_summary.unique_tickers?.length > 0 && (
                          <div style={{ marginTop: 12 }}>
                            <p style={{ fontSize: 11, color: C.muted, marginBottom: 6 }}>Tickers in screening universe</p>
                            <div style={{ display: 'flex', flexWrap: 'wrap', gap: 4, maxHeight: 72, overflowY: 'auto' }}>
                              {results.screening_summary.unique_tickers.map(t => (
                                <span key={t} style={{
                                  background: C.surface, border: `1px solid ${C.border}`,
                                  borderRadius: 5, padding: '2px 7px',
                                  fontSize: 11, fontFamily: 'monospace', color: C.accent,
                                }}>{t}</span>
                              ))}
                            </div>
                          </div>
                        )}
                      </Card>
                    </section>
                  )}

                  {/* AI Explanation */}
                  {explain && (
                    <section>
                      <Label>AI Analysis</Label>
                      <Card style={{ padding: '16px 20px' }}>
                        <div style={{ display: 'flex', gap: 14, alignItems: 'flex-start' }}>
                          <div style={{
                            width: 36, height: 36, borderRadius: 10, flexShrink: 0,
                            background: `linear-gradient(135deg,${C.accent},${C.purple})`,
                            display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: 18,
                          }}>🤖</div>
                          <p style={{ color: C.text, fontSize: 14, lineHeight: 1.7, margin: 0 }}>{explain}</p>
                        </div>
                      </Card>
                    </section>
                  )}

                  {/* Best Config + Download */}
                  <section>
                    <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 10 }}>
                      <Label>Best Configuration</Label>
                      <button
                        onClick={() => downloadJSON(
                          { config: bestConf, metrics: bm },
                          `best_config_${results.strategy_id ?? 'run'}.json`
                        )}
                        style={{
                          background: C.success + '22', border: `1px solid ${C.success}55`,
                          borderRadius: 8, padding: '5px 12px', color: C.success,
                          fontSize: 12, fontWeight: 600, cursor: 'pointer', display: 'flex', alignItems: 'center', gap: 5,
                        }}
                      >⬇ Download JSON</button>
                    </div>
                    <Card style={{ padding: '14px 18px', fontFamily: 'monospace', fontSize: 13, color: C.success, whiteSpace: 'pre-wrap', wordBreak: 'break-all' }}>
                      {typeof bestConf === 'string' ? bestConf : JSON.stringify(bestConf, null, 2)}
                    </Card>
                  </section>

                  {/* Key Metrics */}
                  <section>
                    <Label>Key Metrics — Best Config</Label>
                    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3,1fr)', gap: 12 }}>
                      <MetricCard title="CAGR"           display={pct(bm.cagr)}              color={sign(bm.cagr)} />
                      <MetricCard title="Total Return"   display={pct(bm.total_return_pct)}   color={sign(bm.total_return_pct)} />
                      <MetricCard title="Max Drawdown"   display={pct(bm.max_drawdown)}       color={C.danger} />
                      <MetricCard title="Win Rate"       display={pct(bm.win_rate)}            color={(+bm.win_rate) > 50 ? C.success : C.warning} />
                      <MetricCard title="Expectancy"     display={dollar(bm.expectancy)}       color={sign(bm.expectancy)} />
                      <MetricCard title="Total Trades"   display={bm.total_trades ?? 'N/A'}    color={C.accent} />
                      <MetricCard title="Avg Win"        display={dollar(bm.avg_win)}           color={C.success} />
                      <MetricCard title="Avg Loss"       display={dollar(bm.avg_loss)}          color={C.danger} />
                      <MetricCard title="Final Value"    display={dollar(bm.final_portfolio_value)} color={C.gold} />
                    </div>
                  </section>

                  {/* Charts */}
                  {chartData.length > 1 && (
                    <section>
                      <Label>Optimization Progress</Label>
                      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 14 }}>
                        <ChartBox label="CAGR by Iteration">
                          <ResponsiveContainer width="100%" height={150}>
                            <AreaChart data={chartData} margin={{ top:4, right:4, bottom:0, left:-10 }}>
                              <defs>
                                <linearGradient id="g1" x1="0" y1="0" x2="0" y2="1">
                                  <stop offset="5%"  stopColor={C.success} stopOpacity={.35}/>
                                  <stop offset="95%" stopColor={C.success} stopOpacity={0}/>
                                </linearGradient>
                              </defs>
                              <CartesianGrid strokeDasharray="3 3" stroke={C.border}/>
                              <XAxis dataKey="n"    stroke={C.muted} fontSize={11}/>
                              <YAxis               stroke={C.muted} fontSize={11} tickFormatter={v => `${v.toFixed(1)}%`}/>
                              <Tooltip content={<ChartTip/>}/>
                              <Area type="monotone" dataKey="cagr" name="CAGR" stroke={C.success} fill="url(#g1)" strokeWidth={2} dot={false}/>
                            </AreaChart>
                          </ResponsiveContainer>
                        </ChartBox>

                        <ChartBox label="Drawdown by Iteration">
                          <ResponsiveContainer width="100%" height={150}>
                            <AreaChart data={chartData} margin={{ top:4, right:4, bottom:0, left:-10 }}>
                              <defs>
                                <linearGradient id="g2" x1="0" y1="0" x2="0" y2="1">
                                  <stop offset="5%"  stopColor={C.danger} stopOpacity={.35}/>
                                  <stop offset="95%" stopColor={C.danger} stopOpacity={0}/>
                                </linearGradient>
                              </defs>
                              <CartesianGrid strokeDasharray="3 3" stroke={C.border}/>
                              <XAxis dataKey="n"    stroke={C.muted} fontSize={11}/>
                              <YAxis               stroke={C.muted} fontSize={11} tickFormatter={v => `${v.toFixed(1)}%`}/>
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
                              <YAxis             stroke={C.muted} fontSize={11} tickFormatter={v => `${v.toFixed(1)}%`}/>
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
                              <YAxis               stroke={C.muted} fontSize={11} tickFormatter={v => `${v.toFixed(0)}%`}/>
                              <Tooltip contentStyle={{ background: C.surface, border: `1px solid ${C.border}`, borderRadius: 8, fontSize: 12 }} formatter={v => `${(+v).toFixed(2)}%`}/>
                              <Bar dataKey="value" fill={C.accent} radius={[4,4,0,0]}/>
                            </BarChart>
                          </ResponsiveContainer>
                        </ChartBox>
                      </div>
                    </section>
                  )}

                  {/* Iterations Table */}
                  {iters.length > 0 && (
                    <section>
                      <Label>All Iterations ({iters.length})</Label>
                      <Card style={{ overflow: 'hidden' }}>
                        <div style={{ overflowX: 'auto' }}>
                          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
                            <thead>
                              <tr style={{ background: C.surface }}>
                                {['#', 'Config', 'CAGR', 'Drawdown', 'Win Rate', 'Expectancy'].map(h => (
                                  <th key={h} style={{ padding: '10px 16px', textAlign: 'left', color: C.muted, fontWeight: 600, fontSize: 11, textTransform: 'uppercase', letterSpacing: '.05em', borderBottom: `1px solid ${C.border}`, whiteSpace: 'nowrap' }}>{h}</th>
                                ))}
                              </tr>
                            </thead>
                            <tbody>
                              {iters.map((it, i) => {
                                const m = it.metrics ?? {}
                                return (
                                  <tr key={i} style={{ borderBottom: `1px solid ${C.border}` }}
                                    onMouseEnter={e => e.currentTarget.style.background = C.surface}
                                    onMouseLeave={e => e.currentTarget.style.background = 'transparent'}
                                  >
                                    <td style={{ padding: '10px 16px', color: C.muted, fontFamily: 'monospace' }}>{it.iteration ?? i + 1}</td>
                                    <td style={{ padding: '10px 16px', fontFamily: 'monospace', fontSize: 12, color: C.accent, maxWidth: 220, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{typeof it.config === 'string' ? it.config : JSON.stringify(it.config)}</td>
                                    <td style={{ padding: '10px 16px', fontFamily: 'monospace', color: m.cagr > 0 ? C.success : C.danger }}>{pct(m.cagr)}</td>
                                    <td style={{ padding: '10px 16px', fontFamily: 'monospace', color: C.danger }}>{pct(m.max_drawdown)}</td>
                                    <td style={{ padding: '10px 16px', fontFamily: 'monospace', color: C.text }}>{pct(m.win_rate)}</td>
                                    <td style={{ padding: '10px 16px', fontFamily: 'monospace', color: m.expectancy > 0 ? C.success : C.danger }}>{dollar(m.expectancy)}</td>
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

            {/* ── PORTFOLIO TAB ─────────────────────────────────────── */}
            {activeTab === 'portfolio' && (
              <div style={{ display: 'flex', flexDirection: 'column', gap: 22, maxWidth: 760 }}>
                <section>
                  <Label>Starting Capital</Label>
                  <Card style={{ padding: '18px 20px' }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: 14 }}>
                      <span style={{ color: C.muted, fontSize: 14 }}>Starting amount ($)</span>
                      <input
                        type="number"
                        value={startAmount}
                        onChange={e => setStartAmount(Math.max(1, +e.target.value))}
                        min={1}
                        style={{
                          background: C.surface, border: `1px solid ${C.border}`,
                          borderRadius: 8, padding: '8px 14px', color: C.text,
                          fontSize: 16, fontFamily: 'monospace', fontWeight: 700,
                          width: 180,
                        }}
                      />
                      <div style={{ display: 'flex', gap: 6 }}>
                        {[10000, 50000, 100000, 500000].map(v => (
                          <button key={v} onClick={() => setStartAmount(v)} style={{
                            background: startAmount === v ? C.accent + '33' : C.card,
                            border: `1px solid ${startAmount === v ? C.accent : C.border}`,
                            borderRadius: 6, padding: '5px 10px', color: startAmount === v ? C.accent : C.muted,
                            fontSize: 12, cursor: 'pointer',
                          }}>{v >= 1000 ? `$${v/1000}k` : `$${v}`}</button>
                        ))}
                      </div>
                    </div>
                  </Card>
                </section>

                {!results ? (
                  <div style={{ textAlign: 'center', padding: '48px 0', color: C.muted }}>
                    <div style={{ fontSize: 48, marginBottom: 12 }}>💼</div>
                    <p>Run a backtest first to see portfolio results.</p>
                  </div>
                ) : (
                  <>
                    {/* P&L Summary Cards */}
                    <section>
                      <Label>Portfolio Performance</Label>
                      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3,1fr)', gap: 12 }}>
                        <MetricCard title="Starting Capital" display={dollar(startAmount)}    color={C.text} />
                        <MetricCard title="Final Value"      display={dollar(scaledFinal)}    color={isProfit ? C.success : C.danger} />
                        <MetricCard title="Total P&L"        display={`${isProfit ? '+' : ''}${dollar(profit)}`} color={isProfit ? C.success : C.danger} />
                        <MetricCard title="Total Return"     display={pct(totalReturnPct)}    color={sign(totalReturnPct)} />
                        <MetricCard title="CAGR"             display={pct(bm.cagr)}           color={sign(bm.cagr)} />
                        <MetricCard title="Max Drawdown"     display={pct(bm.max_drawdown)}   color={C.danger} />
                      </div>
                    </section>

                    {/* Equity curve */}
                    <section>
                      <Label>Portfolio vs {activeBenchmarkLabel} ({startLabel} → {endLabel})</Label>
                      <Card style={{ padding: '18px 20px' }}>
                        {mergedCurve.length > 1 ? (
                          <ResponsiveContainer width="100%" height={260}>
                            <AreaChart data={mergedCurve} margin={{ top:10, right:10, bottom:5, left:20 }}>
                              <defs>
                                <linearGradient id="eq" x1="0" y1="0" x2="0" y2="1">
                                  <stop offset="5%"  stopColor={isProfit ? C.success : C.danger} stopOpacity={.25}/>
                                  <stop offset="95%" stopColor={isProfit ? C.success : C.danger} stopOpacity={0}/>
                                </linearGradient>
                                <linearGradient id="bm" x1="0" y1="0" x2="0" y2="1">
                                  <stop offset="5%"  stopColor={C.gold} stopOpacity={.15}/>
                                  <stop offset="95%" stopColor={C.gold} stopOpacity={0}/>
                                </linearGradient>
                              </defs>
                              <CartesianGrid strokeDasharray="3 3" stroke={C.border}/>
                              <XAxis
                                dataKey="date" stroke={C.muted} fontSize={11}
                                tickFormatter={d => d.slice(0, 4)}
                                interval={Math.floor(mergedCurve.length / 8)}
                              />
                              <YAxis stroke={C.muted} fontSize={11} tickFormatter={v => `$${(v/1000).toFixed(0)}k`}/>
                              <Tooltip
                                contentStyle={{ background: C.surface, border: `1px solid ${C.border}`, borderRadius: 8, fontSize: 12 }}
                                formatter={(v, name) => [dollar(v), name === 'portfolio' ? 'Strategy' : activeBenchmarkLabel]}
                                labelStyle={{ color: C.muted }}
                              />
                              <ReferenceLine y={startAmount} stroke={C.muted} strokeDasharray="4 4" label={{ value: 'Start', fill: C.muted, fontSize: 11 }}/>
                              <Area type="monotone" dataKey="benchmark" name="benchmark" stroke={C.gold}    fill="url(#bm)" strokeWidth={1.5} dot={false} strokeDasharray="5 3" connectNulls/>
                              <Area type="monotone" dataKey="portfolio" name="portfolio" stroke={isProfit ? C.success : C.danger} fill="url(#eq)" strokeWidth={2}   dot={false}/>
                            </AreaChart>
                          </ResponsiveContainer>
                        ) : (
                          <p style={{ color: C.muted, fontSize: 13, textAlign: 'center', padding: '32px 0' }}>
                            No time-series data available for this run.
                          </p>
                        )}

                        {/* Legend */}
                        <div style={{ display: 'flex', gap: 20, marginTop: 12, marginBottom: 4 }}>
                          <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                            <div style={{ width: 20, height: 2, background: isProfit ? C.success : C.danger, borderRadius: 1 }}/>
                            <span style={{ fontSize: 12, color: C.muted }}>Strategy</span>
                          </div>
                          <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                            <div style={{ width: 20, height: 2, background: C.gold, borderRadius: 1, opacity: .7 }}/>
                            <span style={{ fontSize: 12, color: C.muted }}>{activeBenchmarkLabel}</span>
                          </div>
                        </div>

                        {/* P&L highlight */}
                        <div style={{
                          marginTop: 16, padding: '14px 20px', borderRadius: 10,
                          background: isProfit ? C.success + '15' : C.danger + '15',
                          border: `1px solid ${isProfit ? C.success + '44' : C.danger + '44'}`,
                          display: 'flex', alignItems: 'center', justifyContent: 'space-between',
                        }}>
                          <span style={{ color: C.muted, fontSize: 14 }}>
                            {isProfit ? '✅ Profitable strategy' : '❌ Strategy lost money'}
                          </span>
                          <span style={{ color: isProfit ? C.success : C.danger, fontSize: 18, fontWeight: 700, fontFamily: 'monospace' }}>
                            {isProfit ? '+' : ''}{dollar(profit)} ({pct(totalReturnPct)})
                          </span>
                        </div>
                      </Card>
                    </section>

                    {/* Risk context */}
                    <section>
                      <Label>Risk Context</Label>
                      <Card style={{ padding: '16px 20px' }}>
                        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
                          <div>
                            <p style={{ fontSize: 12, color: C.muted, marginBottom: 4 }}>Worst-case drawdown on your capital</p>
                            <p style={{ fontSize: 18, fontWeight: 700, fontFamily: 'monospace', color: C.danger }}>
                              -{dollar(startAmount * Math.abs(+(bm.max_drawdown ?? 0)) / 100)}
                            </p>
                          </div>
                          <div>
                            <p style={{ fontSize: 12, color: C.muted, marginBottom: 4 }}>Expected gain per trade</p>
                            <p style={{ fontSize: 18, fontWeight: 700, fontFamily: 'monospace', color: sign(bm.expectancy) }}>
                              {dollar(bm.expectancy)}
                            </p>
                          </div>
                          <div>
                            <p style={{ fontSize: 12, color: C.muted, marginBottom: 4 }}>Avg win per trade</p>
                            <p style={{ fontSize: 18, fontWeight: 700, fontFamily: 'monospace', color: C.success }}>
                              {dollar(bm.avg_win)}
                            </p>
                          </div>
                          <div>
                            <p style={{ fontSize: 12, color: C.muted, marginBottom: 4 }}>Avg loss per trade</p>
                            <p style={{ fontSize: 18, fontWeight: 700, fontFamily: 'monospace', color: C.danger }}>
                              -{dollar(bm.avg_loss)}
                            </p>
                          </div>
                        </div>
                      </Card>
                    </section>
                  </>
                )}
              </div>
            )}

            {/* ── TRADES TAB ────────────────────────────────────────── */}
            {activeTab === 'trades' && (() => {
              const trades = bm.trades ?? results?.best_configuration?.metrics?.trades ?? []
              const wins   = trades.filter(t => (+t.pnl) > 0)
              const losses = trades.filter(t => (+t.pnl) <= 0)
              const sumPnl = trades.reduce((s, t) => s + (+t.pnl || 0), 0)
              const scale  = startAmount / 100_000

              const downloadCsv = () => {
                const headers = ['id','ticker','side','size','entry_date','entry_price','exit_date','exit_price','pnl','pnl_pct','commission']
                const rows = trades.map(t => headers.map(h => {
                  const v = t[h]
                  return v == null ? '' : String(v).includes(',') ? `"${v}"` : v
                }).join(','))
                const csv  = [headers.join(','), ...rows].join('\n')
                const blob = new Blob([csv], { type: 'text/csv' })
                const url  = URL.createObjectURL(blob)
                const a    = document.createElement('a')
                a.href = url
                a.download = `trades_${results?.strategy_id ?? 'run'}.csv`
                a.click()
                URL.revokeObjectURL(url)
              }

              return (
                <div style={{ display: 'flex', flexDirection: 'column', gap: 22 }}>
                  {!results ? (
                    <div style={{ textAlign: 'center', padding: '48px 0', color: C.muted }}>
                      <div style={{ fontSize: 48, marginBottom: 12 }}>📋</div>
                      <p>Run a backtest to see the trade list for the best configuration.</p>
                    </div>
                  ) : (
                    <>
                      <section>
                        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 10 }}>
                          <Label>Trade Summary — Best Config</Label>
                          {trades.length > 0 && (
                            <button
                              onClick={downloadCsv}
                              style={{
                                background: C.success + '22', border: `1px solid ${C.success}55`,
                                borderRadius: 8, padding: '5px 12px', color: C.success,
                                fontSize: 12, fontWeight: 600, cursor: 'pointer',
                              }}
                            >⬇ Download CSV</button>
                          )}
                        </div>
                        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4,1fr)', gap: 12 }}>
                          <MetricCard title="Closed Trades" display={trades.length} color={C.accent} />
                          <MetricCard title="Winners" display={wins.length} color={C.success} />
                          <MetricCard title="Losers" display={losses.length} color={C.danger} />
                          <MetricCard
                            title="Sum P&L (sim)"
                            display={`${sumPnl >= 0 ? '+' : ''}${dollar(sumPnl * scale)}`}
                            color={sign(sumPnl)}
                          />
                        </div>
                        <p style={{ fontSize: 11, color: C.muted, marginTop: 10 }}>
                          PnL is from the $100k simulation; summary scales to your portfolio starting capital when set on Portfolio.
                          Only closed trades from the best configuration are listed.
                        </p>
                      </section>

                      <section>
                        <Label>All Closed Trades ({trades.length})</Label>
                        {trades.length === 0 ? (
                          <Card style={{ padding: '32px 18px', textAlign: 'center', color: C.muted }}>
                            No closed trades were recorded for this run.
                          </Card>
                        ) : (
                          <Card style={{ overflow: 'hidden' }}>
                            <div style={{ overflowX: 'auto', maxHeight: '60vh' }}>
                              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12.5 }}>
                                <thead>
                                  <tr style={{ background: C.surface, position: 'sticky', top: 0 }}>
                                    {['#', 'Ticker', 'Side', 'Size', 'Entry', 'Entry $', 'Exit', 'Exit $', 'P&L', 'P&L %', 'Comm'].map(h => (
                                      <th key={h} style={{
                                        padding: '10px 12px', textAlign: 'left', color: C.muted,
                                        fontWeight: 600, fontSize: 11, textTransform: 'uppercase',
                                        letterSpacing: '.05em', borderBottom: `1px solid ${C.border}`,
                                        whiteSpace: 'nowrap',
                                      }}>{h}</th>
                                    ))}
                                  </tr>
                                </thead>
                                <tbody>
                                  {trades.map((t, i) => {
                                    const pnl = +t.pnl || 0
                                    return (
                                      <tr
                                        key={t.id ?? i}
                                        style={{ borderBottom: `1px solid ${C.border}` }}
                                        onMouseEnter={e => e.currentTarget.style.background = C.surface}
                                        onMouseLeave={e => e.currentTarget.style.background = 'transparent'}
                                      >
                                        <td style={{ padding: '9px 12px', color: C.muted, fontFamily: 'monospace' }}>{t.id ?? i + 1}</td>
                                        <td style={{ padding: '9px 12px', fontFamily: 'monospace', fontWeight: 700, color: C.accent }}>{t.ticker ?? '—'}</td>
                                        <td style={{
                                          padding: '9px 12px', fontFamily: 'monospace',
                                          color: t.side === 'short' ? C.purple : C.text,
                                          textTransform: 'uppercase', fontSize: 11,
                                        }}>{t.side ?? 'long'}</td>
                                        <td style={{ padding: '9px 12px', fontFamily: 'monospace', color: C.text }}>{num(t.size)}</td>
                                        <td style={{ padding: '9px 12px', fontFamily: 'monospace', color: C.muted, whiteSpace: 'nowrap' }}>{t.entry_date ?? '—'}</td>
                                        <td style={{ padding: '9px 12px', fontFamily: 'monospace', color: C.text }}>{t.entry_price != null ? dollar(t.entry_price) : '—'}</td>
                                        <td style={{ padding: '9px 12px', fontFamily: 'monospace', color: C.muted, whiteSpace: 'nowrap' }}>{t.exit_date ?? '—'}</td>
                                        <td style={{ padding: '9px 12px', fontFamily: 'monospace', color: C.text }}>{t.exit_price != null ? dollar(t.exit_price) : '—'}</td>
                                        <td style={{ padding: '9px 12px', fontFamily: 'monospace', fontWeight: 600, color: sign(pnl) }}>
                                          {pnl >= 0 ? '+' : ''}{dollar(pnl)}
                                        </td>
                                        <td style={{ padding: '9px 12px', fontFamily: 'monospace', color: sign(t.pnl_pct) }}>
                                          {t.pnl_pct != null ? pct(t.pnl_pct) : '—'}
                                        </td>
                                        <td style={{ padding: '9px 12px', fontFamily: 'monospace', color: C.muted }}>
                                          {t.commission != null ? dollar(t.commission) : '—'}
                                        </td>
                                      </tr>
                                    )
                                  })}
                                </tbody>
                              </table>
                            </div>
                          </Card>
                        )}
                      </section>
                    </>
                  )}
                </div>
              )
            })()}

            {/* ── CODE TAB ──────────────────────────────────────────── */}
            {activeTab === 'code' && (() => {
              const codeSource  = results ?? debugData
              const dispCode    = codeSource?.generated_code ?? genCode
              const screenCode  = codeSource?.screening_code ?? results?.screening_code ?? ''
              const errorMsg    = debugData?.error ?? null
              const runId       = results?.strategy_id ?? 'run'

              const downloadPy = (code, name) => {
                const blob = new Blob([code], { type: 'text/x-python' })
                const url  = URL.createObjectURL(blob)
                const a    = document.createElement('a'); a.href = url; a.download = name; a.click()
                URL.revokeObjectURL(url)
              }

              return (
                <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>

                  {/* Error banner (only when the run failed) */}
                  {errorMsg && (
                    <div style={{
                      background: C.danger + '15', border: `1px solid ${C.danger}55`,
                      borderRadius: 10, padding: '14px 18px',
                      display: 'flex', gap: 12, alignItems: 'flex-start',
                    }}>
                      <span style={{ fontSize: 18, flexShrink: 0 }}>❌</span>
                      <div>
                        <p style={{ fontWeight: 700, color: C.danger, fontSize: 13, marginBottom: 4 }}>Run failed — showing generated code for debugging</p>
                        <pre style={{ color: C.text, fontSize: 12, lineHeight: 1.6, whiteSpace: 'pre-wrap', wordBreak: 'break-word', margin: 0 }}>{errorMsg}</pre>
                      </div>
                    </div>
                  )}

                  {/* Screening code block */}
                  {screenCode ? (
                    <div>
                      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 8 }}>
                        <Label>Generated Screening Code</Label>
                        <button
                          onClick={() => downloadPy(screenCode, `screening_${runId}.py`)}
                          style={{ background: C.warning + '22', border: `1px solid ${C.warning}55`, borderRadius: 8, padding: '5px 12px', color: C.warning, fontSize: 12, fontWeight: 600, cursor: 'pointer' }}
                        >⬇ Download .py</button>
                      </div>
                      <Card style={{ overflow: 'hidden' }}>
                        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', padding: '10px 16px', borderBottom: `1px solid ${C.border}`, background: C.surface }}>
                          <span style={{ fontSize: 12, color: C.warning, fontFamily: 'monospace' }}>screening.py</span>
                          <div style={{ display: 'flex', gap: 5 }}>{['#ef4444','#f59e0b','#10b981'].map(c => <div key={c} style={{ width: 10, height: 10, borderRadius: '50%', background: c }}/>)}</div>
                        </div>
                        <pre style={{ padding: '16px 18px', margin: 0, fontFamily: "'JetBrains Mono','Fira Code','Courier New',monospace", fontSize: 13, lineHeight: 1.65, color: '#fde68a', background: 'transparent', overflowX: 'auto', whiteSpace: 'pre' }}>{screenCode}</pre>
                      </Card>
                    </div>
                  ) : null}

                  {/* Strategy code block */}
                  <div>
                    <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 8 }}>
                      <Label>Generated Backtrader Strategy</Label>
                      {dispCode && (
                        <button
                          onClick={() => downloadPy(dispCode, `strategy_${runId}.py`)}
                          style={{ background: C.purple + '22', border: `1px solid ${C.purple}55`, borderRadius: 8, padding: '5px 12px', color: C.purple, fontSize: 12, fontWeight: 600, cursor: 'pointer' }}
                        >⬇ Download .py</button>
                      )}
                    </div>
                    {!dispCode ? (
                      <div style={{ textAlign: 'center', padding: '48px 0', color: C.muted }}>
                        <div style={{ fontSize: 48, marginBottom: 12 }}>🧑‍💻</div>
                        <p>Run a backtest to see the generated strategy code.</p>
                      </div>
                    ) : (
                      <Card style={{ overflow: 'hidden' }}>
                        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', padding: '10px 16px', borderBottom: `1px solid ${C.border}`, background: C.surface }}>
                          <span style={{ fontSize: 12, color: C.muted, fontFamily: 'monospace' }}>strategy.py</span>
                          <div style={{ display: 'flex', gap: 5 }}>{['#ef4444','#f59e0b','#10b981'].map(c => <div key={c} style={{ width: 10, height: 10, borderRadius: '50%', background: c }}/>)}</div>
                        </div>
                        <pre style={{ padding: '16px 18px', margin: 0, fontFamily: "'JetBrains Mono','Fira Code','Courier New',monospace", fontSize: 13, lineHeight: 1.65, color: '#a5f3fc', background: 'transparent', overflowX: 'auto', whiteSpace: 'pre' }}>{dispCode}</pre>
                      </Card>
                    )}
                  </div>

                </div>
              )
            })()}

            {/* ── HISTORY TAB ───────────────────────────────────────── */}
            {activeTab === 'history' && (
              <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
                <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 2 }}>
                  <Label>Past Runs</Label>
                  <button onClick={fetchHistory} style={{
                    background: C.card, border: `1px solid ${C.border}`,
                    borderRadius: 8, padding: '5px 12px', color: C.muted, fontSize: 12, cursor: 'pointer',
                  }}>↻ Refresh</button>
                </div>

                {history === null ? (
                  <div style={{ textAlign: 'center', padding: '32px 0', color: C.muted }}>Loading…</div>
                ) : history.length === 0 ? (
                  <div style={{ textAlign: 'center', padding: '48px 0', color: C.muted }}>
                    <div style={{ fontSize: 48, marginBottom: 12 }}>🕑</div>
                    <p>No past runs yet. Run a backtest to see history here.</p>
                  </div>
                ) : (
                  history.map(h => (
                    <Card key={h.id} style={{ padding: '16px 18px' }}>
                      <div style={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between', gap: 12 }}>
                        <div style={{ flex: 1 }}>
                          <p style={{ fontSize: 13.5, color: C.text, marginBottom: 5, fontWeight: 500 }}>
                            {h.prompt}
                          </p>
                          <p style={{ fontSize: 11, color: C.muted }}>{h.timestamp}</p>
                        </div>
                        <span style={{
                          background: C.surface, border: `1px solid ${C.border}`,
                          borderRadius: 6, padding: '2px 8px', color: C.muted, fontSize: 11, flexShrink: 0,
                        }}>#{h.id}</span>
                      </div>
                      {h.best_cagr != null && (
                        <div style={{ display: 'flex', gap: 20, marginTop: 12, flexWrap: 'wrap' }}>
                          <Stat label="CAGR"      value={pct(h.best_cagr)}       color={sign(h.best_cagr)} />
                          <Stat label="Drawdown"  value={pct(h.best_drawdown)}   color={C.danger} />
                          <Stat label="Win Rate"  value={pct(h.best_win_rate)}   color={C.accent} />
                          <Stat label="Expectancy" value={dollar(h.best_expectancy)} color={sign(h.best_expectancy)} />
                        </div>
                      )}
                    </Card>
                  ))
                )}
              </div>
            )}

            {/* ── SETTINGS TAB ──────────────────────────────────────── */}
            {activeTab === 'settings' && (
              <div style={{ display: 'flex', flexDirection: 'column', gap: 22, maxWidth: 600 }}>

                <section>
                  <Label>Benchmark Comparison</Label>
                  <Card style={{ padding: '20px 22px' }}>
                    <p style={{ fontSize: 13.5, color: C.muted, marginBottom: 16, lineHeight: 1.6 }}>
                      Choose the ticker to compare against on the Portfolio equity curve. The benchmark will be normalised to the same starting capital so you can directly compare performance.
                    </p>
                    <div style={{ display: 'flex', gap: 10, alignItems: 'center', flexWrap: 'wrap' }}>
                      <input
                        value={benchmarkInput}
                        onChange={e => setBenchmarkInput(e.target.value.toUpperCase())}
                        onKeyDown={e => { if (e.key === 'Enter') { setBenchmarkTicker(benchmarkInput); } }}
                        placeholder="e.g. SPY"
                        maxLength={8}
                        style={{
                          background: C.surface, border: `1px solid ${C.border}`,
                          borderRadius: 8, padding: '9px 14px',
                          color: C.text, fontSize: 15, fontFamily: 'monospace', fontWeight: 700,
                          width: 120, letterSpacing: '.04em',
                        }}
                      />
                      <button
                        onClick={() => setBenchmarkTicker(benchmarkInput)}
                        style={{
                          background: `linear-gradient(135deg,${C.accent},${C.purple})`,
                          border: 'none', borderRadius: 8, padding: '9px 18px',
                          color: '#fff', fontWeight: 600, fontSize: 13, cursor: 'pointer',
                        }}
                      >Set Benchmark</button>
                      <span style={{ fontSize: 12, color: C.muted }}>
                        Current: <span style={{ color: C.gold, fontFamily: 'monospace', fontWeight: 600 }}>{benchmarkTicker}</span>
                      </span>
                    </div>

                    {/* Quick picks */}
                    <div style={{ marginTop: 16 }}>
                      <p style={{ fontSize: 12, color: C.muted, marginBottom: 8 }}>Quick picks</p>
                      <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
                        {[
                          { ticker: 'SPY',  label: 'S&P 500'     },
                          { ticker: 'QQQ',  label: 'Nasdaq 100'  },
                          { ticker: 'DIA',  label: 'Dow Jones'   },
                          { ticker: 'IWM',  label: 'Russell 2000'},
                          { ticker: 'GLD',  label: 'Gold'        },
                          { ticker: 'BTC-USD', label: 'Bitcoin'  },
                        ].map(({ ticker, label }) => (
                          <button key={ticker}
                            onClick={() => { setBenchmarkInput(ticker); setBenchmarkTicker(ticker); }}
                            style={{
                              background: benchmarkTicker === ticker ? C.gold + '22' : C.card,
                              border: `1px solid ${benchmarkTicker === ticker ? C.gold : C.border}`,
                              borderRadius: 8, padding: '6px 14px', cursor: 'pointer',
                              display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 2,
                            }}
                          >
                            <span style={{ fontFamily: 'monospace', fontSize: 13, fontWeight: 700, color: benchmarkTicker === ticker ? C.gold : C.text }}>{ticker}</span>
                            <span style={{ fontSize: 10, color: C.muted }}>{label}</span>
                          </button>
                        ))}
                      </div>
                    </div>

                    <p style={{ fontSize: 12, color: C.muted, marginTop: 16, fontStyle: 'italic' }}>
                      Changes take effect on the next backtest run.
                    </p>
                  </Card>
                </section>

              </div>
            )}

          </div>
        </div>
      </div>
    </div>
  )
}

// ─── Tiny helpers ─────────────────────────────────────────────────────────────

function EmptyState({ onSelect }) {
  return (
    <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', minHeight: '70vh', gap: 14, color: C.muted, textAlign: 'center' }}>
      <div style={{ fontSize: 56 }}>📊</div>
      <h2 style={{ fontSize: 19, fontWeight: 600, color: C.text }}>Dashboard</h2>
      <p style={{ fontSize: 13.5, maxWidth: 360, lineHeight: 1.65 }}>
        Enter a trading strategy in the chat panel and hit Run. Results, charts, explanation & iteration stats will appear here.
      </p>
      <div style={{ display: 'flex', flexDirection: 'column', gap: 6, marginTop: 6, width: '100%', maxWidth: 320 }}>
        {[
          'AAPL 15/50 EMA crossover, 2020–2024',
          'SPY RSI(14) oversold <30 strategy',
          'TSLA Bollinger Bands breakout',
        ].map(s => (
          <button key={s} onClick={() => onSelect(s)} style={{
            background: C.card, border: `1px solid ${C.border}`, borderRadius: 8,
            padding: '8px 14px', color: C.muted, fontSize: 13, cursor: 'pointer', textAlign: 'left',
          }}
            onMouseEnter={e => e.currentTarget.style.color = C.text}
            onMouseLeave={e => e.currentTarget.style.color = C.muted}
          >{s}</button>
        ))}
      </div>
    </div>
  )
}

function RunButton({ loading, disabled, onClick }) {
  const off = loading || disabled
  return (
    <button
      onClick={onClick}
      disabled={off}
      style={{
        background: off ? C.border : `linear-gradient(135deg,${C.accent},${C.purple})`,
        border: 'none', borderRadius: 8, padding: '6px 14px',
        color: off ? C.muted : '#fff',
        fontWeight: 600, fontSize: 13,
        display: 'flex', alignItems: 'center', gap: 6, transition: 'all .2s',
        cursor: off ? 'not-allowed' : 'pointer',
      }}
    >
      {loading
        ? <span style={{ width: 13, height: 13, border: `2px solid ${C.muted}`, borderTopColor: C.accent, borderRadius: '50%', display: 'inline-block', animation: 'spin .8s linear infinite' }} />
        : '▶'}
      {loading ? 'Running' : 'Run'}
    </button>
  )
}

function DateRangePicker({ startDate, endDate, onStart, onEnd, disabled, maxEnd }) {
  const invalid = startDate && endDate && startDate >= endDate
  const inputStyle = {
    width: '100%',
    background: C.surface,
    border: `1px solid ${invalid ? C.danger : C.border}`,
    borderRadius: 8,
    padding: '7px 10px',
    color: C.text,
    fontSize: 12,
    fontFamily: 'monospace',
    colorScheme: 'dark',
  }
  return (
    <div style={{
      background: C.card, border: `1px solid ${C.border}`, borderRadius: 12,
      padding: '10px 12px', marginBottom: 8,
    }}>
      <p style={{
        fontSize: 10, fontWeight: 700, color: C.gold,
        textTransform: 'uppercase', letterSpacing: '.06em', marginBottom: 8,
      }}>
        Backtest Period
      </p>
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8 }}>
        <div>
          <p style={{ fontSize: 10, color: C.muted, marginBottom: 4 }}>Start</p>
          <input
            type="date"
            value={startDate}
            min="1970-01-01"
            max={endDate || maxEnd}
            onChange={e => onStart(e.target.value)}
            disabled={disabled}
            style={inputStyle}
          />
        </div>
        <div>
          <p style={{ fontSize: 10, color: C.muted, marginBottom: 4 }}>End</p>
          <input
            type="date"
            value={endDate}
            min={startDate || '1970-01-01'}
            max={maxEnd}
            onChange={e => onEnd(e.target.value)}
            disabled={disabled}
            style={inputStyle}
          />
        </div>
      </div>
      {invalid ? (
        <p style={{ fontSize: 11, color: C.danger, marginTop: 6 }}>Start must be before end.</p>
      ) : (
        <p style={{ fontSize: 10, color: C.muted, marginTop: 6 }}>
          Data available through {maxEnd}. Defaults: 2010-01-01 → {maxEnd}.
        </p>
      )}
    </div>
  )
}

function PositionSizeInput({ value, onChange, disabled, mode }) {
  const invalid = !(value >= 1 && value <= 100)
  const isMulti = mode === 'screened'
  const presets = isMulti ? [5, 10, 20, 25] : [25, 50, 75, 100]
  const inputStyle = {
    width: 88,
    background: C.surface,
    border: `1px solid ${invalid ? C.danger : C.border}`,
    borderRadius: 8,
    padding: '7px 10px',
    color: C.text,
    fontSize: 13,
    fontFamily: 'monospace',
    fontWeight: 700,
  }
  return (
    <div style={{
      background: C.card, border: `1px solid ${C.border}`, borderRadius: 12,
      padding: '10px 12px', marginBottom: 8,
    }}>
      <p style={{
        fontSize: 10, fontWeight: 700, color: C.success,
        textTransform: 'uppercase', letterSpacing: '.06em', marginBottom: 8,
      }}>
        Position Size / Trade
      </p>
      <div style={{ display: 'flex', alignItems: 'center', gap: 8, flexWrap: 'wrap' }}>
        <input
          type="number"
          min={1}
          max={100}
          step={1}
          value={value}
          disabled={disabled}
          onChange={e => {
            const n = Number(e.target.value)
            onChange(Number.isFinite(n) ? n : (isMulti ? 10 : 100))
          }}
          style={inputStyle}
        />
        <span style={{ fontSize: 13, color: C.text, fontWeight: 600 }}>%</span>
        <span style={{ fontSize: 11, color: C.muted }}>of available cash</span>
        <div style={{ display: 'flex', gap: 4, marginLeft: 'auto', flexWrap: 'wrap' }}>
          {presets.map(p => (
            <button
              key={p}
              type="button"
              disabled={disabled}
              onClick={() => onChange(p)}
              style={{
                background: value === p ? C.success + '33' : C.surface,
                border: `1px solid ${value === p ? C.success : C.border}`,
                borderRadius: 6, padding: '4px 8px',
                color: value === p ? C.success : C.muted,
                fontSize: 11, fontFamily: 'monospace', cursor: disabled ? 'not-allowed' : 'pointer',
              }}
            >{p}%</button>
          ))}
        </div>
      </div>
      {invalid ? (
        <p style={{ fontSize: 11, color: C.danger, marginTop: 6 }}>Enter a value from 1 to 100.</p>
      ) : (
        <p style={{ fontSize: 10, color: C.muted, marginTop: 6 }}>
          Default for {isMulti ? 'multi-stock' : 'single-stock'}: {isMulti ? '10%' : '100%'}.
          Strategy code uses stake_pct and the backtest sizer.
        </p>
      )}
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