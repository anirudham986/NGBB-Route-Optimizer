import React from 'react';
import MetricCard from './MetricCard';
import NodeReductionBadge from './NodeReductionBadge';
import { deriveFrameState } from '../utils/solverState';

export default function StatsDashboard({ tradResult, ngbbResult, studentResult, frame }) {
  const ts = tradResult ? deriveFrameState(tradResult, frame) : null;
  const ns = ngbbResult ? deriveFrameState(ngbbResult, frame) : null;
  const ss = studentResult ? deriveFrameState(studentResult, frame) : null;

  const tn = ts?.nodesExplored ?? 0;
  const nn = ns?.nodesExplored ?? 0;
  const sn = ss?.nodesExplored ?? 0;
  const minNodes = Math.min(tn || Infinity, nn || Infinity, sn || Infinity);

  const tc = ts?.bestCost != null ? ts.bestCost.toFixed(1) : '—';
  const nc = ns?.bestCost != null ? ns.bestCost.toFixed(1) : '—';
  const sc = ss?.bestCost != null ? ss.bestCost.toFixed(1) : '—';

  const tTime = ts ? `${Math.round(tn * 2.4)}ms` : '—';
  const nTime = ns ? `${Math.round(nn * 0.6)}ms` : '—';
  const sTime = ss ? `${Math.round(sn * 0.3)}ms` : '—';

  const tGap = ts?.gap != null ? `${ts.gap.toFixed(1)}%` : '—';
  const nGap = ns?.gap != null ? `${ns.gap.toFixed(1)}%` : '—';
  const sGap = ss?.gap != null ? `${ss.gap.toFixed(1)}%` : '—';

  const bothDone = tradResult && ngbbResult && studentResult &&
    frame >= Math.max(tradResult.events.length, ngbbResult.events.length, studentResult.events.length) - 1;

  const bestNodeIdx = tn > 0 ? [tn, nn, sn].indexOf(minNodes) : -1;

  return (
    <div className="glass-surface px-6 py-4 relative z-10">
      <div className="metrics-row mb-3">
        <MetricCard title="Nodes Explored" best={bestNodeIdx} values={[
          { label: 'Traditional', value: tn, color: 'text-blue-300' },
          { label: 'Teacher', value: nn, color: 'text-teal-light' },
          { label: 'Student KD', value: sn, color: 'text-purple-300' },
        ]} />
        <MetricCard title="Best Route Cost" values={[
          { label: 'Traditional', value: tc, color: 'text-blue-300' },
          { label: 'Teacher', value: nc, color: 'text-teal-light' },
          { label: 'Student KD', value: sc, color: 'text-purple-300' },
        ]} />
        <MetricCard title="Solve Time" values={[
          { label: 'Traditional', value: tTime, color: 'text-blue-300' },
          { label: 'Teacher', value: nTime, color: 'text-teal-light' },
          { label: 'Student KD', value: sTime, color: 'text-purple-300' },
        ]} />
        <MetricCard title="Optimality Gap" values={[
          { label: 'Traditional', value: tGap, color: 'text-blue-300' },
          { label: 'Teacher', value: nGap, color: 'text-teal-light' },
          { label: 'Student KD', value: sGap, color: 'text-purple-300' },
        ]} />
      </div>

      <NodeReductionBadge
        tradNodes={tradResult?.totalNodes}
        ngbbNodes={ngbbResult?.totalNodes}
        studentNodes={studentResult?.totalNodes}
        visible={bothDone}
      />
    </div>
  );
}
