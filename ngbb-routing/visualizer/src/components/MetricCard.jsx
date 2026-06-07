import React from 'react';

export default function MetricCard({ title, values, best, icon }) {
  // values = [{label, value, color}]
  return (
    <div className="glass-card px-3 py-2.5 rounded-xl">
      <div className="flex items-center gap-1.5 mb-2">
        {icon && <span className="text-gray-500">{icon}</span>}
        <span className="text-[10px] font-semibold text-gray-500 uppercase tracking-wider">{title}</span>
      </div>
      <div className="flex items-end gap-3">
        {values.map((v, i) => (
          <div key={i} className="flex-1 min-w-0">
            <div className="text-[9px] text-gray-600 mb-0.5 truncate">{v.label}</div>
            <div className={`text-sm font-bold font-mono metric-value ${v.color || 'text-gray-300'} ${best === i ? 'underline decoration-2 underline-offset-2' : ''}`}>
              {v.value}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
