import React from 'react';
import { Zap, TrendingDown } from 'lucide-react';

export default function NodeReductionBadge({ tradNodes, ngbbNodes, studentNodes, visible }) {
  if (!visible || !tradNodes) return null;

  const ngbbReduction = Math.round((tradNodes - ngbbNodes) / tradNodes * 100);
  const studentReduction = Math.round((tradNodes - studentNodes) / tradNodes * 100);
  const kdVsTeacher = Math.round((ngbbNodes - studentNodes) / Math.max(ngbbNodes, 1) * 100);

  return (
    <div className="badge-enter flex flex-wrap items-center justify-center gap-3 mt-2">
      {/* Teacher reduction */}
      <div className="glass-card glow-teal px-4 py-2 rounded-xl flex items-center gap-2">
        <TrendingDown size={16} className="text-teal-light" />
        <div>
          <div className="text-lg font-black text-teal-light metric-value">{ngbbReduction}%</div>
          <div className="text-[9px] text-gray-500 font-medium">Teacher vs Traditional</div>
        </div>
      </div>

      {/* Student reduction */}
      <div className="glass-card glow-purple px-4 py-2 rounded-xl flex items-center gap-2">
        <Zap size={16} className="text-purple-400" />
        <div>
          <div className="text-lg font-black text-purple-300 metric-value">{studentReduction}%</div>
          <div className="text-[9px] text-gray-500 font-medium">Student vs Traditional</div>
        </div>
      </div>

      {/* KD efficiency */}
      <div className="glass-card px-4 py-2 rounded-xl flex items-center gap-2 border border-amber/20">
        <Zap size={16} className="text-amber" />
        <div>
          <div className="text-lg font-black gradient-text-warm metric-value">
            {kdVsTeacher > 0 ? `+${kdVsTeacher}%` : `${kdVsTeacher}%`}
          </div>
          <div className="text-[9px] text-gray-500 font-medium">Student KD vs Teacher</div>
        </div>
      </div>
    </div>
  );
}
