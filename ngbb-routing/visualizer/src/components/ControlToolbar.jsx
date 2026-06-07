import React from 'react';
import { Play, Pause, SkipForward, RotateCcw, FastForward } from 'lucide-react';

export default function ControlToolbar({ isPlaying, speed, frame, maxFrame, onPlay, onPause, onStep, onReset, onRunToEnd, onSpeedChange }) {
  const progress = maxFrame > 0 ? (frame / maxFrame * 100) : 0;

  return (
    <div className="glass-surface px-6 py-2.5 flex flex-wrap items-center gap-3 relative z-10">
      {/* Transport controls */}
      <div className="flex items-center gap-1.5">
        <button onClick={isPlaying ? onPause : onPlay}
          className="flex items-center gap-1.5 px-3.5 py-1.5 rounded-lg bg-gradient-to-r from-teal/20 to-teal/10 text-teal-light hover:from-teal/30 hover:to-teal/20 text-sm font-semibold transition-all active:scale-95 border border-teal/20">
          {isPlaying ? <Pause size={13} /> : <Play size={13} />}
          {isPlaying ? 'Pause' : 'Play'}
        </button>

        <button onClick={onStep} disabled={isPlaying}
          className={`p-2 rounded-lg text-sm transition-all ${isPlaying ? 'opacity-30 cursor-not-allowed' : 'bg-gray-800/60 text-gray-300 hover:bg-gray-700/60 active:scale-95 border border-gray-700/30'}`}>
          <SkipForward size={14} />
        </button>

        <button onClick={onReset}
          className="p-2 rounded-lg bg-gray-800/60 text-gray-300 hover:bg-gray-700/60 transition-all active:scale-95 border border-gray-700/30">
          <RotateCcw size={14} />
        </button>

        <button onClick={onRunToEnd}
          className="p-2 rounded-lg bg-gray-800/60 text-gray-300 hover:bg-gray-700/60 transition-all active:scale-95 border border-gray-700/30">
          <FastForward size={14} />
        </button>
      </div>

      <div className="w-px h-5 bg-gray-700/50" />

      {/* Speed */}
      <label className="flex items-center gap-2 text-xs font-medium">
        <span className="text-gray-500">Speed</span>
        <input type="range" min="0.25" max="8" step="0.25" value={speed}
          onChange={e => onSpeedChange(parseFloat(e.target.value))}
          className="w-16 accent-teal-light h-1" />
        <span className="font-mono text-sm font-bold text-teal-light min-w-[2.5rem]">{speed}×</span>
      </label>

      <div className="w-px h-5 bg-gray-700/50" />

      {/* Progress bar */}
      <div className="flex items-center gap-2 flex-1 min-w-[120px]">
        <span className="text-xs text-gray-500 font-mono">{frame}</span>
        <div className="flex-1 h-1.5 bg-gray-800 rounded-full overflow-hidden">
          <div className="h-full bg-gradient-to-r from-teal to-purple-500 rounded-full transition-all duration-150"
            style={{ width: `${progress}%` }} />
        </div>
        <span className="text-xs text-gray-500 font-mono">{maxFrame}</span>
      </div>
    </div>
  );
}
