import React from 'react';

export default function VsBadge({ tradResult, otherResult, frame, label }) {
  const bothRunning = tradResult && otherResult &&
    frame < Math.max(tradResult.events.length, otherResult.events.length) - 1;

  return (
    <div className="flex flex-col items-center justify-center px-2">
      <div className={`w-8 h-8 rounded-full flex items-center justify-center text-[10px] font-black uppercase tracking-widest border ${
        bothRunning
          ? 'bg-amber/10 border-amber/30 text-amber pulse-glow'
          : 'bg-gray-800/50 border-gray-700/30 text-gray-500'
      }`}>
        {label}
      </div>
    </div>
  );
}
