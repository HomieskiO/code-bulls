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

// ─── Reusable components ──────────────────────────────────────────────────────

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

function MetricCard({ title, value, display, color }) {
  const shown = display ?? value ?? 'N/A'
  return (
    <div style={{ background: C.card, border: `1px solid ${C.border}`, borderRadius: 12, padding: '14px 18px' }}>
      <p style={{ color: C.muted, fontSize: 11, textTransform: 'uppercase', letterSpacing: '.06em', marginBottom: 6 }}>{title}</p>
      <p style={{ color: color ?? C.text, fontSize: 21, fontWeight: 700, fontFamily: 'monospace', letterSpacing: '-.02em' }}>{shown}</p>
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

// ─── Chat Message ─────────────────────────────────────────────────────────────

function Msg({ m }) {
  const isUser = m.type === 'user'
  // Render **bold** and `code` in AI messages
  const renderContent = (text) => {
    if (isUser) return text
    const parts = text.split(/(\*\*[^*]+\*\*|`[^`]+`)/g)
    return parts.map((p, i) => {
      if (p.startsWith('**') && p.endsWith('**'))
        return <strong key={i} style={{ color: C.accent }}>{p.slice(2, -2)}</strong>
      if (p.startsWith('`') && p.endsWith('`'))
        return <code key={i} style={{ background: C.surface, borderRadius: 4, padding: '1px 5px', fontSize: 12, color: C.gold }}>{p.slice(1, -1)}</code>
      return p
    })
  }
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
        padding: '9px 13px', color: C.text, fontSize: 13.5, lineHeight: 1.65,
        whiteSpace: 'pre-wrap',
      }}>
        {m.thinking
          ? <span style={{ color: C.muted }}>{m.content}
              <span style={{ display: 'inline-flex', gap: 4, marginLeft: 8, verticalAlign: 'middle' }}>
                {[0,1,2].map(i => <span key={i} style={{ width: 5, height: 5, borderRadius: '50%', background: C.accent, display: 'inline-block', animation: `dot .9s ${i*.22}s ease-in-out infinite alternate` }}/>)}
              </span>
            </span>
          : renderContent(m.content)
        }
      </div>
      {isUser && (
        <div style={{ width: 26, height: 26, borderRadius: '50%', flexShrink: 0, background: '#1e3a8a', display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: 11, fontWeight: 700, color: '#fff', marginLeft: 8, marginTop: 2 }}>U</div>
      )}
    </div>
  )
}

// ─── Run Backtest Confirmation Banner ─────────────────────────────────────────

function ReadyBanner({ params, riskSummary, onRun, onReset, loading }) {
  return (
    <div style={{ margin: '10px 0', padding: '14px 16px', background: C.success + '15', border: `1px solid ${C.success}44`, borderRadius: 12 }}>
      <p style={{ fontSize: 13, fontWeight: 600, color: C.success, marginBottom: 8 }}>Setup complete — ready to run!</p>
      {riskSummary && <p style={{ fontSize: 12, color: C.muted, marginBottom: 10 }}>{riskSummary}</p>}
      <div style={{ display: 'flex', gap: 8 }}>
        <button onClick={onRun} disabled={loading} style={{
          background: loading ? C.border : `linear-gradient(135deg,${C.success},#059669)`,
          border: 'none', borderRadius: 8, padding: '8px 18px',
          color: loading ? C.muted : '#fff', fontWeight: 600, fontSize: 13, cursor: loading ? 'not-allowed' : 'pointer',
          display: 'flex', alignItems: 'center', gap: 6,
        }}>
          {loading
            ? <><span style={{ width: 12, height: 12, border: `2px solid ${C.muted}`, borderTopColor: C.accent, borderRadius: '50%', display: 'inline-block', animation: 'spin .8s linear infinite' }}/> Running</>
            : '▶ Run Backtest'}
        </button>
        <button onClick={onReset} disabled={loading} style={{
          background: 'transparent', border: `1px solid ${C.border}`, borderRadius: 8,
          padding: '8px 14px', color: C.muted, fontSize: 13, cursor: 'pointer',
        }}>Start over</button>
      </div>
    </div>
  )
}

// ─── Main App ─────────────────────────────────────────────────────────────────

const WELCOME_MESSAGE = `Welcome to AlgoTrader AI.

I'll guide you through setting up your backtest step by step.

To start: describe the trading strategy you'd like to test.
Example: "Trade AAPL using a 15/50 EMA crossover"`

export default function App() {
  const [messages,        setMessages]        = useState([{ type: 'ai', content: WELCOME_MESSAGE }])
  const [input,           setInput]           = useState('')
  const [loading,         setLoading]         = useState(false)
  const [results,         setResults]         = useState(null)
  const [activeTab,       setActiveTab]       = useState('dashboard')
  const [startAmount,     setStartAmount]     = useState(100000)
  const [history,         setHistory]         = useState(null)
  const [benchmarkTicker, setBenchmarkTicker] = useState('SPY')
  const [benchmarkInput,  setBenchmarkInput]  = useState('SPY')

  // Conversation state
  const [sessionId,       setSessionId]       = useState(null)
  const [convStage,       setConvStage]       = useState('initial')
  const [readyToRun,      setReadyToRun]      = useState(false)
  const [backtestParams,  setBacktestParams]  = useState(null)
  const [riskSummary,     setRiskSummary]     = useState(null)

  const chatEnd = useRef(null)

  useEffect(() => { chatEnd.current?.scrollIntoView({ behavior: 'smooth' }) }, [messages])
  useEffect(() => { if (activeTab === 'history' && history === null) fetchHistory() }, [activeTab])

  const fetchHistory = async () => {
    try { setHistory(await (await fetch('/api/history')).json()) }
    catch { setHistory([]) }
  }

  // Send a message to the conversation agent
  const sendMessage = async () => {
    if (!input.trim() || loading) return
    const msg = input.trim()
    setInput('')
    setLoading(true)
    setMessages(prev => [...prev, { type: 'user', content: msg }, { type: 'ai', content: 'Thinking…', thinking: true }])

    try {
      const res  = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ session_id: sessionId, message: msg }),
      })
      const data = await res.json()

      setSessionId(data.session_id)
      setConvStage(data.stage)
      setMessages(prev => [...prev.slice(0, -1), { type: 'ai', content: data.response }])

      if (data.ready_to_run) {
        setReadyToRun(true)
        setBacktestParams(data.backtest_params)
        setRiskSummary(data.risk_summary)
      }
    } catch (err) {
      setMessages(prev => [...prev.slice(0, -1), { type: 'ai', content: `Error: ${err.message}` }])
    } finally {
      setLoading(false)
    }
  }

  // Run the actual backtest once params are collected
  const runBacktest = async () => {
    if (!backtestParams || loading) return
    setReadyToRun(false)
    setLoading(true)

    const params = { ...backtestParams, benchmark_ticker: benchmarkTicker }
    setMessages(prev => [...prev, { type: 'ai', content: 'Running backtest & optimizing across 3 iterations…', thinking: true }])

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
      const optLines = (data.optimization_explanations ?? [])
        .map((e, i) => `  Iter ${i+2}: ${e}`)
        .join('\n')

      setMessages(prev => [
        ...prev.slice(0, -1),
        {
          type: 'ai',
          content:
            `Optimization complete after ${n} iteration${n!==1?'s':''}.\n\n` +
            `CAGR: ${pct(m.cagr)} | Drawdown: ${pct(m.max_drawdown)} | Win Rate: ${pct(m.win_rate)}\n` +
            (optLines ? `\nParameter changes:\n${optLines}\n` : '') +
            `\nFull results on the right →`,
        },
      ])
    } catch (err) {
      setMessages(prev => [...prev.slice(0, -1), { type: 'ai', content: `Error: ${err.message}` }])
    } finally {
      setLoading(false)
    }
  }

  const resetConversation = () => {
    setSessionId(null)
    setConvStage('initial')
    setReadyToRun(false)
    setBacktestParams(null)
    setRiskSummary(null)
    setMessages([{ type: 'ai', content: WELCOME_MESSAGE }])
  }

  // Derived data
  const bm       = results?.best_configuration?.metrics ?? {}
  const bestConf = results?.best_configuration?.config  ?? {}
  const iters    = results?.all_iterations  ?? []
  const genCode  = results?.generated_code  ?? ''
  const explain  = results?.explanation     ?? ''
  const riskProf = results?.risk_profile    ?? {}
  const optExpls = results?.optimization_explanations ?? []

  const chartData = iters.map((it, i) => ({
    n:        it.iteration ?? i+1,
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

  const equityCurve = (bm.portfolio_values ?? []).map(d => ({
    date: d.date.slice(0, 7),
    value: Math.round(d.value * scaleFactor),
  }))
  const bmMap = new Map((results?.benchmark_values ?? []).map(d => [d.date.slice(0,7), Math.round(d.value*scaleFactor)]))
  const mergedCurve = equityCurve.map(p => ({ date: p.date, portfolio: p.value, benchmark: bmMap.get(p.date) ?? null }))
  const activeBmLabel = results?.benchmark_ticker ?? benchmarkTicker
  const startLabel = mergedCurve[0]?.date ?? 'Start'
  const endLabel   = mergedCurve.at(-1)?.date ?? 'End'

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

      {/* Body */}
      <div style={{ display:'flex', flex:1, overflow:'hidden' }}>

        {/* Chat Panel */}
        <div style={{ width:340, display:'flex', flexDirection:'column', borderRight:`1px solid ${C.border}`, background:C.surface, flexShrink:0 }}>
          <div style={{ flex:1, overflowY:'auto', padding:'14px 10px' }}>
            {messages.map((m, i) => <Msg key={i} m={m}/>)}

            {readyToRun && (
              <ReadyBanner
                params={backtestParams}
                riskSummary={riskSummary}
                onRun={runBacktest}
                onReset={resetConversation}
                loading={loading}
              />
            )}
            <div ref={chatEnd}/>
          </div>

          <div style={{ padding:10, borderTop:`1px solid ${C.border}`, background:C.bg }}>
            <div style={{ background:C.card, border:`1px solid ${C.border}`, borderRadius:12, overflow:'hidden' }}>
              <textarea
                value={input}
                onChange={e => setInput(e.target.value)}
                onKeyDown={e => { if (e.key==='Enter' && !e.shiftKey) { e.preventDefault(); sendMessage() } }}
                placeholder={readyToRun ? "Type 'yes' to confirm or ask to change something…" : "Reply to the AI…"}
                disabled={loading}
                rows={3}
                style={{ width:'100%', padding:'11px 13px', background:'transparent', border:'none', color:C.text, fontSize:13.5, resize:'none', lineHeight:1.5 }}
              />
              <div style={{ display:'flex', alignItems:'center', justifyContent:'space-between', padding:'7px 11px', borderTop:`1px solid ${C.border}` }}>
                <span style={{ fontSize:11, color:C.muted }}>↵ send · ⇧↵ newline</span>
                <button
                  onClick={sendMessage}
                  disabled={loading || !input.trim()}
                  style={{
                    background: loading||!input.trim() ? C.border : `linear-gradient(135deg,${C.accent},${C.purple})`,
                    border:'none', borderRadius:8, padding:'6px 14px',
                    color: loading||!input.trim() ? C.muted : '#fff',
                    fontWeight:600, fontSize:13, display:'flex', alignItems:'center', gap:6, cursor:loading||!input.trim()?'not-allowed':'pointer',
                  }}
                >
                  {loading
                    ? <span style={{ width:13, height:13, border:`2px solid ${C.muted}`, borderTopColor:C.accent, borderRadius:'50%', display:'inline-block', animation:'spin .8s linear infinite' }}/>
                    : '▶'}
                  {loading ? 'Working' : 'Send'}
                </button>
              </div>
            </div>

            {/* Quick-start templates */}
            {convStage === 'initial' && (
              <div style={{ marginTop:8, display:'flex', flexDirection:'column', gap:4 }}>
                <p style={{ fontSize:11, color:C.muted, padding:'0 2px' }}>Quick start</p>
                {[
                  'Trade AAPL using a 15/50 EMA crossover',
                  'SPY RSI(14) oversold below 30 strategy',
                  'TSLA Bollinger Bands breakout strategy',
                ].map(s => (
                  <button key={s} onClick={() => setInput(s)} style={{ background:C.card, border:`1px solid ${C.border}`, borderRadius:8, padding:'6px 12px', color:C.muted, fontSize:12, textAlign:'left', cursor:'pointer' }}
                    onMouseEnter={e => e.currentTarget.style.color=C.text}
                    onMouseLeave={e => e.currentTarget.style.color=C.muted}
                  >{s}</button>
                ))}
              </div>
            )}

            {/* New session button once a conversation is in progress */}
            {convStage !== 'initial' && !readyToRun && (
              <button onClick={resetConversation} style={{ marginTop:8, width:'100%', background:'transparent', border:`1px solid ${C.border}`, borderRadius:8, padding:'6px 12px', color:C.muted, fontSize:12, cursor:'pointer' }}>
                ↺ Start a new conversation
              </button>
            )}
          </div>
        </div>

        {/* Right Panel */}
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

            {/* ── DASHBOARD TAB ─────────────────────────────────────── */}
            {activeTab === 'dashboard' && (
              !results ? <EmptyState onSelect={setInput}/> : (
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

                  {/* Optimization journey */}
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

                  {/* Risk profile summary */}
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

            {/* ── PORTFOLIO TAB ─────────────────────────────────────── */}
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

            {/* ── CODE TAB ──────────────────────────────────────────── */}
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

            {/* ── HISTORY TAB ───────────────────────────────────────── */}
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
                          <Stat label="CAGR"       value={pct(h.best_cagr)}        color={sign(h.best_cagr)}/>
                          <Stat label="Drawdown"   value={pct(h.best_drawdown)}    color={C.danger}/>
                          <Stat label="Win Rate"   value={pct(h.best_win_rate)}    color={C.accent}/>
                          <Stat label="Expectancy" value={dollar(h.best_expectancy)} color={sign(h.best_expectancy)}/>
                        </div>
                      )}
                    </Card>
                  ))
                }
              </div>
            )}

            {/* ── SETTINGS TAB ──────────────────────────────────────── */}
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
                        <p style={{ fontSize:12, color:C.muted, lineHeight:1.6 }}>Live data fetched on demand. Covers all current stocks and ETFs. Slower to load. Best for recent tickers.</p>
                      </div>
                      <div style={{ padding:'14px 16px', background:C.surface, borderRadius:10, border:`1px solid ${C.border}` }}>
                        <p style={{ fontWeight:700, color:C.purple, marginBottom:6 }}>Kaggle Local</p>
                        <p style={{ fontSize:12, color:C.muted, lineHeight:1.6 }}>Pre-downloaded CSV files. Fast loading. ~7,500 US stocks & ETFs. Data up to ~2017. Best for historical analysis.</p>
                      </div>
                    </div>
                    <p style={{ fontSize:12, color:C.muted, marginTop:14, fontStyle:'italic' }}>
                      The conversation agent will ask which source to use when setting up a new backtest.
                    </p>
                  </Card>
                </section>

                <section>
                  <Label>Risk Profiles</Label>
                  <Card style={{ padding:'18px 20px' }}>
                    <div style={{ display:'flex', flexDirection:'column', gap:10 }}>
                      {[
                        { level:'Conservative', color:C.success, sl:'2%', tp:'6%', size:'30%', dd:'10%' },
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
                    <p style={{ fontSize:12, color:C.muted, marginTop:14, fontStyle:'italic' }}>
                      These defaults are applied when you select a risk level during the conversation. You can always customize individual values.
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

// ─── Empty state ─────────────────────────────────────────────────────────────

function EmptyState({ onSelect }) {
  return (
    <div style={{ display:'flex', flexDirection:'column', alignItems:'center', justifyContent:'center', minHeight:'70vh', gap:14, color:C.muted, textAlign:'center' }}>
      <div style={{ fontSize:56 }}>📊</div>
      <h2 style={{ fontSize:19, fontWeight:600, color:C.text }}>Dashboard</h2>
      <p style={{ fontSize:13.5, maxWidth:380, lineHeight:1.65 }}>
        Use the chat on the left to set up your backtest. The AI will guide you through the parameters step by step.
      </p>
      <div style={{ display:'flex', flexDirection:'column', gap:6, marginTop:6, width:'100%', maxWidth:340 }}>
        {[
          'Trade AAPL using a 15/50 EMA crossover',
          'SPY RSI(14) oversold below 30 strategy',
          'TSLA Bollinger Bands breakout strategy',
        ].map(s => (
          <button key={s} onClick={() => onSelect(s)} style={{ background:C.card, border:`1px solid ${C.border}`, borderRadius:8, padding:'8px 14px', color:C.muted, fontSize:13, cursor:'pointer', textAlign:'left' }}
            onMouseEnter={e => e.currentTarget.style.color=C.text}
            onMouseLeave={e => e.currentTarget.style.color=C.muted}
          >{s}</button>
        ))}
      </div>
    </div>
  )
}