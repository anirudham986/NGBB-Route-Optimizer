import React, { useState, useEffect, useCallback } from 'react';
import HeaderBar from './components/HeaderBar';
import ControlToolbar from './components/ControlToolbar';
import SolverCanvas from './components/SolverCanvas';
import StatsDashboard from './components/StatsDashboard';
import VsBadge from './components/VsBadge';
import { generateInstance } from './utils/instanceGenerator';
import { runTraditionalSolver } from './utils/traditionalSolver';
import { runNGBBSolver } from './utils/ngbbSolver';
import { runStudentSolver } from './utils/studentSolver';

export default function App() {
  const [config, setConfig] = useState({
    nodeCount: 20, seed: 42, problemType: 'cvrp', vehicleCapacity: 50,
  });
  const [instance, setInstance] = useState(null);
  const [tradResult, setTradResult] = useState(null);
  const [ngbbResult, setNgbbResult] = useState(null);
  const [studentResult, setStudentResult] = useState(null);
  const [frame, setFrame] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [speed, setSpeed] = useState(1.0);

  const generate = useCallback(() => {
    const inst = generateInstance(config.nodeCount, config.seed, config.problemType, config.vehicleCapacity);
    setInstance(inst);
    setTradResult(runTraditionalSolver(inst, config.seed));
    setNgbbResult(runNGBBSolver(inst, config.seed));
    setStudentResult(runStudentSolver(inst, config.seed));
    setFrame(0);
    setIsPlaying(false);
  }, [config]);

  useEffect(() => { generate(); }, [generate]);

  // Animation loop
  useEffect(() => {
    if (!isPlaying || !tradResult || !ngbbResult || !studentResult) return;
    let raf;
    let lastTime = 0;
    const tick = (now) => {
      if (now - lastTime > 1000 / (24 * speed)) {
        setFrame(f => {
          const maxF = Math.max(
            tradResult.events.length,
            ngbbResult.events.length,
            studentResult.events.length
          ) - 1;
          if (f >= maxF) { setIsPlaying(false); return f; }
          return f + 1;
        });
        lastTime = now;
      }
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [isPlaying, speed, tradResult, ngbbResult, studentResult]);

  const maxFrame = tradResult && ngbbResult && studentResult
    ? Math.max(tradResult.events.length, ngbbResult.events.length, studentResult.events.length) - 1 : 0;

  const handleConfigChange = (key, value) => {
    setConfig(prev => ({ ...prev, [key]: value }));
  };

  return (
    <div className="app-container animated-gradient font-display">
      {/* Floating particles */}
      <div className="particle w-1 h-1 bg-teal/30 top-[20%] left-[10%]" />
      <div className="particle w-1.5 h-1.5 bg-purple-light/20 top-[60%] right-[15%]" style={{animationDelay: '2s'}} />
      <div className="particle w-1 h-1 bg-amber/20 top-[40%] left-[80%]" style={{animationDelay: '4s'}} />

      <HeaderBar
        config={config}
        onConfigChange={handleConfigChange}
        onGenerate={() => setConfig(prev => ({ ...prev, seed: Math.floor(Math.random() * 9999) }))}
      />

      <ControlToolbar
        isPlaying={isPlaying}
        speed={speed}
        frame={frame}
        maxFrame={maxFrame}
        onPlay={() => setIsPlaying(true)}
        onPause={() => setIsPlaying(false)}
        onStep={() => setFrame(f => Math.min(f + 1, maxFrame))}
        onReset={() => { setFrame(0); setIsPlaying(false); }}
        onRunToEnd={() => { setFrame(maxFrame); setIsPlaying(false); }}
        onSpeedChange={setSpeed}
      />

      {/* 3-solver canvas row */}
      <div className="canvas-row py-3">
        <SolverCanvas label="Traditional B&B" subtitle="Random Branching"
          result={tradResult} currentFrame={frame}
          colorScheme="navy" glowClass="canvas-glow-navy" />

        <VsBadge tradResult={tradResult} otherResult={ngbbResult} frame={frame}
          label="vs" />

        <SolverCanvas label="Teacher NGBB" subtitle="GNN-Guided (Full Model)"
          result={ngbbResult} currentFrame={frame}
          colorScheme="teal" glowClass="canvas-glow-teal" />

        <VsBadge tradResult={ngbbResult} otherResult={studentResult} frame={frame}
          label="KD" />

        <SolverCanvas label="Student KD" subtitle="Distilled Agent (Deployed)"
          result={studentResult} currentFrame={frame}
          colorScheme="purple" glowClass="canvas-glow-purple" />
      </div>

      <StatsDashboard
        tradResult={tradResult}
        ngbbResult={ngbbResult}
        studentResult={studentResult}
        frame={frame}
      />
    </div>
  );
}
