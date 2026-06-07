import React, { useRef, useEffect } from 'react';
import { drawFrame } from '../utils/canvasRenderer';
import { deriveFrameState } from '../utils/solverState';
import { CheckCircle } from 'lucide-react';

export default function SolverCanvas({ label, subtitle, result, currentFrame, colorScheme, glowClass }) {
  const canvasRef = useRef(null);
  const rafRef = useRef(null);

  useEffect(() => {
    if (!result || !canvasRef.current) return;
    const ctx = canvasRef.current.getContext('2d');
    const render = (timestamp) => {
      const frameData = deriveFrameState(result, currentFrame);
      drawFrame(ctx, result.instance, frameData, colorScheme, 420, 320, timestamp);
      rafRef.current = requestAnimationFrame(render);
    };
    rafRef.current = requestAnimationFrame(render);
    return () => { if (rafRef.current) cancelAnimationFrame(rafRef.current); };
  }, [result, currentFrame, colorScheme]);

  const isSolved = result && currentFrame >= result.events.length - 1;

  const schemeColors = {
    navy: { bg: 'from-blue-900/20 to-blue-800/10', text: 'text-blue-300', badge: 'bg-navy', border: 'border-blue-800/30' },
    teal: { bg: 'from-teal/20 to-teal-dark/10', text: 'text-teal-light', badge: 'bg-teal', border: 'border-teal/30' },
    purple: { bg: 'from-purple-900/20 to-purple-800/10', text: 'text-purple-300', badge: 'bg-purple-600', border: 'border-purple-700/30' },
  };
  const cs = schemeColors[colorScheme] || schemeColors.teal;

  return (
    <div className="flex flex-col items-center gap-2">
      {/* Label */}
      <div className="text-center">
        <div className={`text-xs font-bold tracking-wider uppercase ${cs.text}`}>{label}</div>
        {subtitle && <div className="text-[10px] text-gray-500 mt-0.5">{subtitle}</div>}
      </div>

      {/* Canvas */}
      <div className={`relative rounded-2xl overflow-hidden ${glowClass} border ${cs.border}`}>
        <canvas ref={canvasRef} width={420} height={320} style={{ display: 'block' }} />
        {isSolved && (
          <div className={`absolute bottom-2 left-1/2 -translate-x-1/2 flex items-center gap-1.5 px-3 py-1 rounded-full text-[11px] font-bold text-white shadow-lg badge-enter ${cs.badge}`}>
            <CheckCircle size={12} />
            {result.totalNodes} nodes · {result.solvedAtFrame} steps
          </div>
        )}
      </div>
    </div>
  );
}
