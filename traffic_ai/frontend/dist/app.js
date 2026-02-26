const { useState, useEffect, useRef, useCallback } = React;
const {
  LineChart, Line, AreaChart, Area, BarChart, Bar,
  XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid, Legend
} = Recharts;

// ── API helpers ──────────────────────────────────────────────────────────────
const api = {
  get:  (path) => fetch(path).then(r => r.json()),
  post: (path, body) => fetch(path, { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(body||{}) }).then(r=>r.json()),
};

// ── Custom Tooltip ────────────────────────────────────────────────────────────
function CTooltip({ active, payload, label }) {
  if (!active || !payload?.length) return null;
  return React.createElement('div', { className: 'custom-tooltip' },
    React.createElement('div', { style: { color:'#4a6278', marginBottom:4 } }, `Step ${label}`),
    ...payload.map(p =>
      React.createElement('div', { key: p.dataKey, style: { color: p.color } },
        `${p.name}: ${typeof p.value === 'number' ? p.value.toFixed(2) : p.value}`
      )
    )
  );
}

// ── Direction names ───────────────────────────────────────────────────────────
const DIR_NAMES = ['North', 'South', 'East', 'West'];
const DIR_SHORT = ['N', 'S', 'E', 'W'];

// ── Intersection Card ─────────────────────────────────────────────────────────
function IntersectionCard({ data }) {
  if (!data) return null;
  const { id, phase, phase_locked, queues, waits, throughput } = data;
  const phaseLabel = phase === 0 ? 'N-S GREEN' : 'E-W GREEN';
  const phaseClass = phase_locked ? 'phase-lock' : (phase === 0 ? 'phase-ns' : 'phase-ew');
  const phaseText  = phase_locked ? 'LOCKED' : phaseLabel;

  const maxQ = 20;
  const gridLabels = [
    ['Intersection', String(id + 1).padStart(2, '0')],
    ['Throughput', throughput],
  ];

  return React.createElement('div', { className: 'intersection-card fade-in' },
    React.createElement('div', { className: 'int-header' },
      React.createElement('div', { style: { display:'flex', alignItems:'center', gap:10 } },
        React.createElement('div', { className: 'signal-light' },
          [0,1,2,3].map(d => React.createElement('div', {
            key: d,
            className: `light-dot ${
              phase === 0 && (d===0||d===1) ? 'on-ns' :
              phase === 1 && (d===2||d===3) ? 'on-ew' : 'off'
            }`
          }))
        ),
        React.createElement('div', { className: 'int-title' }, `INT ${id + 1}`)
      ),
      React.createElement('div', { className: `phase-badge ${phaseClass}` }, phaseText)
    ),
    React.createElement('div', { className: 'directions' },
      DIR_NAMES.map((name, d) => {
        const q   = queues[d] || 0;
        const pct = Math.min((q / maxQ) * 100, 100);
        const isNS = d < 2;
        const isHigh = q >= 12;
        const isCrit = q >= 18;
        return React.createElement('div', { key: d, className: 'dir-row' },
          React.createElement('div', { className: 'dir-label' },
            React.createElement('span', null, name),
            React.createElement('span', { className: 'dir-count' }, q + ' cars')
          ),
          React.createElement('div', { className: 'bar-track' },
            React.createElement('div', {
              className: `bar-fill ${isCrit ? 'critical' : isHigh ? (isNS ? 'bar-ns high' : 'bar-ew high') : (isNS ? 'bar-ns' : 'bar-ew')}`,
              style: { width: pct + '%' }
            })
          )
        );
      })
    ),
    React.createElement('div', { style:{ marginTop:'0.8rem', display:'flex', justifyContent:'space-between' } },
      React.createElement('div', { style:{fontFamily:'var(--mono)', fontSize:'0.6rem', color:'var(--text-dim)'} },
        'THROUGHPUT'
      ),
      React.createElement('div', { style:{fontFamily:'var(--mono)', fontSize:'0.7rem', color:'var(--accent3)'} },
        throughput + ' vehicles'
      )
    )
  );
}

// ── Stat Card ─────────────────────────────────────────────────────────────────
function StatCard({ label, value, unit, color, delta, deltaLabel }) {
  return React.createElement('div', { className: `stat-card ${color}` },
    React.createElement('div', { className: 'stat-label' }, label),
    React.createElement('div', { className: 'stat-value' }, value),
    unit && React.createElement('div', { className: 'stat-unit' }, unit),
    delta !== undefined && React.createElement('div', { className: 'stat-delta' },
      React.createElement('span', { className: delta >= 0 ? 'delta-pos' : 'delta-neg' },
        (delta >= 0 ? '▲' : '▼') + ' ' + Math.abs(delta).toFixed(1) + '% ' + (deltaLabel || '')
      )
    )
  );
}

// ── Training Panel ────────────────────────────────────────────────────────────
function TrainingPanel({ onTrainComplete }) {
  const [status,   setStatus]   = useState('idle'); // idle | running | done
  const [progress, setProgress] = useState(0);
  const [log,      setLog]      = useState([]);
  const [lastEp,   setLastEp]   = useState(null);
  const logRef = useRef(null);

  useEffect(() => {
    if (status !== 'running') return;
    const iv = setInterval(async () => {
      const data = await api.get('/api/training_log');
      setLog(data.log);
      setProgress((data.log.length / 300) * 100);
      if (data.log.length > 0) setLastEp(data.log[data.log.length - 1]);
      if (data.done) {
        setStatus('done');
        clearInterval(iv);
        onTrainComplete();
      }
    }, 800);
    return () => clearInterval(iv);
  }, [status]);

  useEffect(() => {
    if (logRef.current) logRef.current.scrollTop = logRef.current.scrollHeight;
  }, [log]);

  const startTraining = async () => {
    setStatus('running');
    setLog([]);
    setProgress(0);
    await api.post('/api/train');
  };

  // Chart data — last 50 episodes
  const chartData = log.slice(-60).map(l => ({ ep: l.ep, reward: -l.reward, wait: l.wait, queue: l.queue }));

  return React.createElement('div', { className: 'training-card' },
    React.createElement('div', { className: 'training-header' },
      React.createElement('div', { className: 'training-title' }, '⚡ Agent Training'),
      status === 'idle' && React.createElement('button', { className: 'btn btn-primary', style:{width:'auto',padding:'8px 20px'}, onClick: startTraining }, 'START TRAINING'),
      status === 'done' && React.createElement('span', { style:{ fontFamily:'var(--mono)', fontSize:'0.65rem', color:'var(--accent3)' } }, '✓ TRAINING COMPLETE'),
    ),

    status === 'idle' && React.createElement('div', { className: 'idle-state' },
      React.createElement('div', { className: 'idle-icon' }, '🧠'),
      React.createElement('div', { className: 'idle-text' }, 'AGENT NOT TRAINED — CLICK START TO BEGIN'),
    ),

    status !== 'idle' && React.createElement('div', null,
      React.createElement('div', { className: 'progress-wrap' },
        React.createElement('div', { style:{ display:'flex', justifyContent:'space-between', fontFamily:'var(--mono)', fontSize:'0.6rem', color:'var(--text-dim)' } },
          React.createElement('span', null, 'TRAINING PROGRESS'),
          React.createElement('span', { style:{color:'var(--accent)'} }, Math.round(progress) + '%'),
        ),
        React.createElement('div', { className: 'progress-bar-track' },
          React.createElement('div', { className: 'progress-bar-fill', style:{ width: progress + '%' } })
        ),
      ),

      lastEp && React.createElement('div', { style:{ display:'grid', gridTemplateColumns:'repeat(4,1fr)', gap:'0.5rem', marginBottom:'1rem' } },
        [
          { k:'Episode', v: lastEp.ep + '/300' },
          { k:'Reward',  v: lastEp.reward.toFixed(2) },
          { k:'Epsilon', v: lastEp.eps.toFixed(3) },
          { k:'Queue',   v: lastEp.queue.toFixed(2) },
        ].map(({ k, v }) =>
          React.createElement('div', { key: k, style:{ background:'var(--bg)', border:'1px solid var(--border)', padding:'8px', textAlign:'center' } },
            React.createElement('div', { style:{ fontFamily:'var(--mono)', fontSize:'0.55rem', color:'var(--text-dim)', marginBottom:4 } }, k),
            React.createElement('div', { style:{ fontFamily:'var(--mono)', fontSize:'0.8rem', color:'var(--accent)' } }, v)
          )
        )
      ),

      chartData.length > 2 && React.createElement('div', { style:{ height: 140 } },
        React.createElement(ResponsiveContainer, { width:'100%', height:'100%' },
          React.createElement(AreaChart, { data: chartData, margin:{ top:4, right:4, left:-20, bottom:0 } },
            React.createElement(CartesianGrid, { strokeDasharray:'3 3', stroke:'var(--border)' }),
            React.createElement(XAxis, { dataKey:'ep', tick:{ fill:'var(--text-dim)', fontSize:10 } }),
            React.createElement(YAxis, { tick:{ fill:'var(--text-dim)', fontSize:10 } }),
            React.createElement(Tooltip, { content: React.createElement(CTooltip) }),
            React.createElement(Area, { type:'monotone', dataKey:'reward', name:'Loss', stroke:'var(--accent2)', fill:'rgba(255,107,53,0.1)', strokeWidth:1.5, dot:false }),
            React.createElement(Area, { type:'monotone', dataKey:'queue',  name:'Queue', stroke:'var(--accent)', fill:'rgba(0,212,255,0.07)', strokeWidth:1.5, dot:false }),
          )
        )
      ),
    )
  );
}

// ── Live Simulation Panel ──────────────────────────────────────────────────────
function SimPanel({ agentReady }) {
  const [running,     setRunning]     = useState(false);
  const [mode,        setMode]        = useState('rl');
  const [rate,        setRate]        = useState(0.8);
  const [snapshot,    setSnapshot]    = useState(null);
  const [history,     setHistory]     = useState([]);

  useEffect(() => {
    if (!running) return;
    const iv = setInterval(async () => {
      const data = await api.get('/api/sim/snapshot');
      if (data.ready) {
        setSnapshot(data.snapshot);
        setHistory(data.history || []);
      }
    }, 300);
    return () => clearInterval(iv);
  }, [running]);

  const start = async () => {
    await api.post('/api/sim/start', { mode, arrival_rate: rate });
    setRunning(true);
  };
  const stop = async () => {
    await api.post('/api/sim/stop');
    setRunning(false);
  };
  const switchMode = async (m) => {
    setMode(m);
    if (running) await api.post('/api/sim/config', { mode: m });
  };
  const switchRate = async (r) => {
    setRate(r);
    if (running) await api.post('/api/sim/config', { arrival_rate: r });
  };

  const totalQ    = snapshot?.total_queue || 0;
  const totalThru = snapshot?.total_throughput || 0;
  const totalWait = snapshot?.total_wait || 0;
  const reward    = snapshot?.reward || 0;

  // Chart data
  const chartQ = history.map(h => ({ step: h.step, queue: h.total_queue }));
  const chartW = history.map(h => ({ step: h.step, wait: Math.min(h.total_wait / 100, 50) }));

  return React.createElement('div', { style:{ display:'flex', flexDirection:'column', gap:'1rem' } },

    // Controls row
    React.createElement('div', { style:{ display:'flex', gap:'1rem', alignItems:'center', flexWrap:'wrap' } },
      React.createElement('div', { style:{flex:'0 0 auto'} },
        React.createElement('div', { style:{ fontFamily:'var(--mono)', fontSize:'0.55rem', color:'var(--text-dim)', marginBottom:6, letterSpacing:2 } }, 'CONTROLLER MODE'),
        React.createElement('div', { className: 'mode-toggle', style:{width:200} },
          React.createElement('button', {
            className: `mode-btn ${mode==='rl' ? 'active-rl' : ''}`,
            onClick: () => switchMode('rl'),
            disabled: !agentReady,
          }, '🧠 RL AGENT'),
          React.createElement('button', {
            className: `mode-btn ${mode==='fixed' ? 'active-fx' : ''}`,
            onClick: () => switchMode('fixed'),
          }, '⏱ FIXED')
        )
      ),
      React.createElement('div', { className:'slider-wrap', style:{flex:'1', minWidth:180} },
        React.createElement('div', { className:'slider-label' },
          React.createElement('span', null, 'ARRIVAL RATE'),
          React.createElement('span', { className:'slider-val' }, rate.toFixed(1) + ' cars/step')
        ),
        React.createElement('input', { type:'range', min:0.3, max:1.2, step:0.1, value:rate,
          onChange: e => switchRate(parseFloat(e.target.value)) })
      ),
      running
        ? React.createElement('button', { className:'btn btn-danger', style:{width:140}, onClick:stop }, '⏹ STOP SIM')
        : React.createElement('button', { className:'btn btn-success', style:{width:140}, onClick:start,
            disabled: mode==='rl' && !agentReady }, '▶ START SIM'),
    ),

    // Stats
    snapshot && React.createElement('div', { className:'stats-row' },
      React.createElement(StatCard, { label:'TOTAL QUEUE', value: totalQ, unit:'vehicles waiting', color:'orange' }),
      React.createElement(StatCard, { label:'THROUGHPUT', value: totalThru, unit:'vehicles cleared', color:'green' }),
      React.createElement(StatCard, { label:'TOTAL WAIT', value: totalWait, unit:'vehicle-steps', color:'blue' }),
      React.createElement(StatCard, { label:'STEP REWARD', value: reward.toFixed(3), unit:'normalized [-1,0]', color:'amber' }),
    ),

    // Intersections
    snapshot && React.createElement('div', null,
      React.createElement('div', { className:'section-header', style:{marginBottom:'0.75rem'} },
        React.createElement('div', { className:'section-title' }, '4 Intersections — Live State'),
        React.createElement('div', { className:'section-line' })
      ),
      React.createElement('div', { className:'intersections-grid' },
        (snapshot.intersections || []).map(inter =>
          React.createElement(IntersectionCard, { key: inter.id, data: inter })
        )
      )
    ),

    // Charts
    history.length > 2 && React.createElement('div', { className:'charts-row' },
      React.createElement('div', { className:'chart-card' },
        React.createElement('div', { className:'chart-title' }, '📊 Queue Length Over Time'),
        React.createElement('div', { style:{ height:180 } },
          React.createElement(ResponsiveContainer, { width:'100%', height:'100%' },
            React.createElement(AreaChart, { data: chartQ.slice(-80), margin:{top:4,right:4,left:-20,bottom:0} },
              React.createElement(CartesianGrid, { strokeDasharray:'3 3', stroke:'var(--border)' }),
              React.createElement(XAxis, { dataKey:'step', tick:{fill:'var(--text-dim)', fontSize:10} }),
              React.createElement(YAxis, { tick:{fill:'var(--text-dim)', fontSize:10} }),
              React.createElement(Tooltip, { content: React.createElement(CTooltip) }),
              React.createElement(Area, { type:'monotone', dataKey:'queue', name:'Queue', stroke:'var(--accent2)', fill:'rgba(255,107,53,0.12)', strokeWidth:2, dot:false }),
            )
          )
        )
      ),
      React.createElement('div', { className:'chart-card' },
        React.createElement('div', { className:'chart-title' }, '⏱ Wait Time Over Time'),
        React.createElement('div', { style:{ height:180 } },
          React.createElement(ResponsiveContainer, { width:'100%', height:'100%' },
            React.createElement(AreaChart, { data: chartW.slice(-80), margin:{top:4,right:4,left:-20,bottom:0} },
              React.createElement(CartesianGrid, { strokeDasharray:'3 3', stroke:'var(--border)' }),
              React.createElement(XAxis, { dataKey:'step', tick:{fill:'var(--text-dim)', fontSize:10} }),
              React.createElement(YAxis, { tick:{fill:'var(--text-dim)', fontSize:10} }),
              React.createElement(Tooltip, { content: React.createElement(CTooltip) }),
              React.createElement(Area, { type:'monotone', dataKey:'wait', name:'Wait', stroke:'var(--accent)', fill:'rgba(0,212,255,0.1)', strokeWidth:2, dot:false }),
            )
          )
        )
      )
    ),

    !snapshot && React.createElement('div', { className:'idle-state' },
      React.createElement('div', { className:'idle-icon' }, '🚦'),
      React.createElement('div', { className:'idle-text' }, running ? 'WAITING FOR FIRST STEP...' : 'PRESS ▶ START SIM TO BEGIN')
    )
  );
}

// ── Results Panel ─────────────────────────────────────────────────────────────
function ResultsPanel() {
  // Static results from pre-run evaluation
  const results = [
    { metric: 'Wait Time', rl: 1164, fixed: 1787, unit: 'vehicle-steps', lower: true },
    { metric: 'Queue Length', rl: 0.95, fixed: 4.0, unit: 'avg per intersection', lower: true },
    { metric: 'Throughput', rl: 617, fixed: 605, unit: 'vehicles cleared', lower: false },
  ];

  const improvements = [
    { label: 'WAIT REDUCTION', value: 34.9, color: 'var(--accent3)' },
    { label: 'QUEUE REDUCTION', value: 76.2, color: 'var(--accent)' },
    { label: 'THROUGHPUT GAIN', value: 2.0, color: 'var(--amber)' },
  ];

  const barData = results.map(r => ({
    metric: r.metric,
    'RL Agent': r.rl,
    'Fixed-Time': r.fixed,
  }));

  return React.createElement('div', { style:{ display:'flex', flexDirection:'column', gap:'1.2rem' } },
    React.createElement('div', { style:{ display:'grid', gridTemplateColumns:'repeat(3,1fr)', gap:'1rem' } },
      improvements.map(imp =>
        React.createElement('div', { key: imp.label,
          style:{ background:'var(--panel)', border:'1px solid var(--border)', padding:'1.2rem', textAlign:'center' } },
          React.createElement('div', { style:{ fontFamily:'var(--mono)', fontSize:'0.55rem', letterSpacing:2, color:'var(--text-dim)', marginBottom:8 } }, imp.label),
          React.createElement('div', { style:{ fontFamily:'var(--cond)', fontSize:'2.8rem', fontWeight:900, color: imp.color } },
            '+' + imp.value + '%'
          ),
          React.createElement('div', { style:{ fontFamily:'var(--mono)', fontSize:'0.6rem', color:'var(--text-dim)', marginTop:4 } }, 'VS FIXED-TIME')
        )
      )
    ),

    React.createElement('div', { className:'chart-card' },
      React.createElement('div', { className:'chart-title' }, '📊 RL Agent vs Fixed-Time Baseline'),
      React.createElement('div', { style:{ height:220 } },
        React.createElement(ResponsiveContainer, { width:'100%', height:'100%' },
          React.createElement(BarChart, { data: barData, margin:{top:4,right:20,left:0,bottom:0} },
            React.createElement(CartesianGrid, { strokeDasharray:'3 3', stroke:'var(--border)' }),
            React.createElement(XAxis, { dataKey:'metric', tick:{fill:'var(--text-dim)', fontSize:11} }),
            React.createElement(YAxis, { tick:{fill:'var(--text-dim)', fontSize:10} }),
            React.createElement(Tooltip, { contentStyle:{ background:'var(--panel)', border:'1px solid var(--border)', fontFamily:'var(--mono)', fontSize:'0.65rem' } }),
            React.createElement(Legend, { wrapperStyle:{ fontFamily:'var(--mono)', fontSize:'0.65rem' } }),
            React.createElement(Bar, { dataKey:'RL Agent',    fill:'var(--accent)',  radius:[2,2,0,0] }),
            React.createElement(Bar, { dataKey:'Fixed-Time',  fill:'var(--accent2)', radius:[2,2,0,0] }),
          )
        )
      )
    )
  );
}

// ── Main App ──────────────────────────────────────────────────────────────────
function App() {
  const [tab,        setTab]        = useState('sim');   // sim | train | results
  const [agentReady, setAgentReady] = useState(false);
  const [simRunning, setSimRunning] = useState(false);

  // Poll status
  useEffect(() => {
    const iv = setInterval(async () => {
      const s = await api.get('/api/status');
      setAgentReady(s.agent_ready);
      setSimRunning(s.sim_running);
    }, 2000);
    return () => clearInterval(iv);
  }, []);

  const tabs = [
    { id:'sim',     label:'🚦 Live Simulation' },
    { id:'train',   label:'🧠 Train Agent' },
    { id:'results', label:'📊 Results' },
  ];

  return React.createElement('div', null,

    // Header
    React.createElement('header', { className:'header' },
      React.createElement('div', { className:'header-logo' },
        React.createElement('span', { className:'logo-icon' }, '🚦'),
        React.createElement('div', null,
          React.createElement('div', { className:'logo-text' }, 'Traffic AI'),
          React.createElement('div', { className:'logo-sub' }, 'Multi-Intersection Control Room'),
        )
      ),
      React.createElement('div', { className:'header-status' },
        React.createElement('div', { className:'status-pill', style:{ color: agentReady ? 'var(--accent3)' : 'var(--text-dim)', borderColor: agentReady ? 'var(--accent3)' : 'var(--border)' } },
          React.createElement('div', { className:'status-dot', style:{ background: agentReady ? 'var(--accent3)' : 'var(--text-dim)' } }),
          agentReady ? 'AGENT READY' : 'AGENT UNTRAINED'
        ),
        React.createElement('div', { className:'status-pill', style:{ color: simRunning ? 'var(--accent)' : 'var(--text-dim)', borderColor: simRunning ? 'var(--accent)' : 'var(--border)' } },
          React.createElement('div', { className:'status-dot', style:{ background: simRunning ? 'var(--accent)' : 'var(--text-dim)' } }),
          simRunning ? 'SIM LIVE' : 'SIM IDLE'
        ),
      )
    ),

    // Body
    React.createElement('div', { className:'app-body' },

      // Sidebar
      React.createElement('aside', { className:'sidebar' },
        React.createElement('div', { className:'sidebar-section' },
          React.createElement('div', { className:'sidebar-label' }, 'Navigation'),
          ...tabs.map(t =>
            React.createElement('button', {
              key: t.id,
              className: `btn ${tab===t.id ? 'btn-primary' : ''}`,
              style: tab !== t.id ? { color:'var(--text-dim)', borderColor:'var(--border)' } : {},
              onClick: () => setTab(t.id),
            }, t.label)
          )
        ),

        React.createElement('div', { className:'sidebar-section' },
          React.createElement('div', { className:'sidebar-label' }, 'System Info'),
          ...[
            ['Intersections', '4 (2×2 grid)'],
            ['Directions', '4 per intersection'],
            ['Algorithm', 'Tabular Q-Learning'],
            ['Agents', 'Independent × 4'],
            ['Min Green Time', '5 steps'],
            ['Max Queue', '20 vehicles'],
          ].map(([k, v]) =>
            React.createElement('div', { key:k, style:{ display:'flex', justifyContent:'space-between', fontFamily:'var(--mono)', fontSize:'0.6rem', padding:'4px 0', borderBottom:'1px solid var(--border)' } },
              React.createElement('span', { style:{color:'var(--text-dim)'} }, k),
              React.createElement('span', { style:{color:'var(--accent)'} }, v),
            )
          )
        ),

        React.createElement('div', { className:'sidebar-section' },
          React.createElement('div', { className:'sidebar-label' }, 'Legend'),
          ...[
            ['var(--accent3)', 'N-S GREEN phase'],
            ['var(--accent)',  'E-W GREEN phase'],
            ['var(--amber)',   'Phase locked (min-green)'],
            ['var(--red)',     'Critical queue (18+ cars)'],
          ].map(([color, label]) =>
            React.createElement('div', { key:label, style:{ display:'flex', alignItems:'center', gap:8, fontFamily:'var(--mono)', fontSize:'0.6rem', color:'var(--text-dim)' } },
              React.createElement('div', { style:{ width:10, height:10, borderRadius:'50%', background:color, flexShrink:0 } }),
              label
            )
          )
        )
      ),

      // Main content
      React.createElement('main', { className:'main' },
        tab === 'train'   && React.createElement(TrainingPanel, { onTrainComplete: () => setAgentReady(true) }),
        tab === 'sim'     && React.createElement(SimPanel, { agentReady }),
        tab === 'results' && React.createElement(ResultsPanel),
      )
    )
  );
}

ReactDOM.render(React.createElement(App), document.getElementById('root'));
