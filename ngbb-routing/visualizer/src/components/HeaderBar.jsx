import React from 'react';
import { Shuffle, Truck, Zap } from 'lucide-react';

export default function HeaderBar({ config, onConfigChange, onGenerate }) {
  return (
    <header className="glass-surface px-6 py-3 flex flex-wrap items-center gap-4 relative z-10">
      {/* Brand */}
      <div className="flex items-center gap-3 mr-4">
        <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-teal to-purple-500 flex items-center justify-center shadow-lg shadow-teal/20">
          <Truck size={20} className="text-white" />
        </div>
        <div>
          <h1 className="text-lg font-extrabold tracking-tight leading-tight">
            <span className="gradient-text">NGBB</span>
            <span className="text-gray-400 font-medium text-sm ml-1.5">Route Optimizer</span>
          </h1>
          <p className="text-[10px] text-gray-500 font-medium tracking-wider uppercase">
            Neural-Guided Branch & Bound + Knowledge Distillation
          </p>
        </div>
      </div>

      {/* Controls */}
      <div className="flex items-center gap-3 flex-wrap">
        {/* Node count */}
        <label className="flex items-center gap-2 text-xs font-medium">
          <span className="text-gray-500">Locations</span>
          <input type="range" min="10" max="60" step="1" value={config.nodeCount}
            onChange={e => onConfigChange('nodeCount', parseInt(e.target.value))}
            className="w-20 accent-teal-light h-1" />
          <span className="font-mono text-sm font-bold text-teal-light min-w-[1.5rem]">{config.nodeCount}</span>
        </label>

        {/* Seed */}
        <label className="flex items-center gap-2 text-xs font-medium">
          <span className="text-gray-500">Seed</span>
          <input type="number" min="0" max="9999" value={config.seed}
            onChange={e => onConfigChange('seed', parseInt(e.target.value) || 0)}
            className="w-16 px-2 py-1 rounded-lg bg-gray-800/80 border border-gray-700/50 text-white text-sm font-mono focus:border-teal/50 focus:outline-none transition" />
        </label>

        {/* Problem type */}
        <label className="flex items-center gap-2 text-xs font-medium">
          <span className="text-gray-500">Mode</span>
          <select value={config.problemType} onChange={e => onConfigChange('problemType', e.target.value)}
            className="px-2 py-1 rounded-lg bg-gray-800/80 border border-gray-700/50 text-white text-sm focus:border-teal/50 focus:outline-none transition">
            <option value="cvrp">CVRP</option>
            <option value="tsp">TSP</option>
          </select>
        </label>

        {config.problemType === 'cvrp' && (
          <label className="flex items-center gap-2 text-xs font-medium">
            <span className="text-gray-500">Cap</span>
            <input type="number" min="20" max="100" value={config.vehicleCapacity}
              onChange={e => onConfigChange('vehicleCapacity', parseInt(e.target.value) || 50)}
              className="w-14 px-2 py-1 rounded-lg bg-gray-800/80 border border-gray-700/50 text-white text-sm font-mono focus:border-teal/50 focus:outline-none transition" />
          </label>
        )}
      </div>

      <div className="flex-1" />

      {/* Generate button */}
      <button onClick={onGenerate}
        className="flex items-center gap-2 px-4 py-2 rounded-xl bg-gradient-to-r from-teal to-teal-dark text-white text-sm font-bold hover:brightness-110 transition-all shadow-lg shadow-teal/25 active:scale-95">
        <Zap size={14} /> New Instance
      </button>
    </header>
  );
}
