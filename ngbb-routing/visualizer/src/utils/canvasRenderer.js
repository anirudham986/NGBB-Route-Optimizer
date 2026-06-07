import * as d3 from 'd3';

const SCHEMES = {
  navy:   { primary: '#3B82F6', accent: '#1E3A5F', glow: 'rgba(59,130,246,0.3)' },
  teal:   { primary: '#0D7377', accent: '#10908F', glow: 'rgba(13,115,119,0.3)' },
  purple: { primary: '#8B5CF6', accent: '#7C3AED', glow: 'rgba(139,92,246,0.3)' },
};

export function drawFrame(ctx, instance, frameState, colorScheme, width, height, timestamp = 0) {
  if (!instance || !frameState) return;
  const scaleX = d3.scaleLinear().domain([0, 100]).range([35, width - 35]);
  const scaleY = d3.scaleLinear().domain([0, 100]).range([35, height - 35]);
  const nodes = instance.nodes;
  const scheme = SCHEMES[colorScheme] || SCHEMES.teal;

  // 1. Clear + dark background
  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = '#0A0F1A';
  ctx.fillRect(0, 0, width, height);

  // 2. Subtle grid
  ctx.strokeStyle = 'rgba(255,255,255,0.025)';
  ctx.lineWidth = 0.5;
  for (let g = 0; g <= 100; g += 10) {
    ctx.beginPath(); ctx.moveTo(scaleX(g), scaleY(0)); ctx.lineTo(scaleX(g), scaleY(100)); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(scaleX(0), scaleY(g)); ctx.lineTo(scaleX(100), scaleY(g)); ctx.stroke();
  }

  // 3. Faint potential edges (only short ones)
  ctx.strokeStyle = 'rgba(255,255,255,0.04)';
  ctx.lineWidth = 0.3;
  ctx.setLineDash([]);
  for (let i = 0; i < nodes.length; i++) {
    for (let j = i + 1; j < nodes.length; j++) {
      if (instance.dist[i][j] < instance.maxDist * 0.35) {
        ctx.beginPath();
        ctx.moveTo(scaleX(nodes[i].x), scaleY(nodes[i].y));
        ctx.lineTo(scaleX(nodes[j].x), scaleY(nodes[j].y));
        ctx.stroke();
      }
    }
  }

  // 4. Active edges (partial solution)
  if (frameState.activeEdges) {
    ctx.strokeStyle = scheme.primary;
    ctx.lineWidth = 2;
    ctx.globalAlpha = 0.7;
    ctx.setLineDash([]);
    ctx.shadowColor = scheme.glow;
    ctx.shadowBlur = 4;
    frameState.activeEdges.forEach(([from, to]) => {
      if (from < nodes.length && to < nodes.length) {
        ctx.beginPath();
        ctx.moveTo(scaleX(nodes[from].x), scaleY(nodes[from].y));
        ctx.lineTo(scaleX(nodes[to].x), scaleY(nodes[to].y));
        ctx.stroke();
      }
    });
    ctx.shadowBlur = 0;
    ctx.globalAlpha = 1;
  }

  // 5. Candidate edge (marching ants)
  if (frameState.candidateEdge) {
    const [from, to] = frameState.candidateEdge;
    if (from < nodes.length && to < nodes.length) {
      ctx.strokeStyle = '#F59E0B';
      ctx.lineWidth = 2;
      ctx.setLineDash([6, 4]);
      ctx.lineDashOffset = -(frameState.frame || 0) * 0.8;
      ctx.beginPath();
      ctx.moveTo(scaleX(nodes[from].x), scaleY(nodes[from].y));
      ctx.lineTo(scaleX(nodes[to].x), scaleY(nodes[to].y));
      ctx.stroke();
      ctx.setLineDash([]);
    }
  }

  // 6. Optimal route (thick glow)
  if (frameState.isSolved && frameState.optimalRoute) {
    ctx.strokeStyle = scheme.primary;
    ctx.lineWidth = 3;
    ctx.setLineDash([]);
    ctx.shadowColor = scheme.glow;
    ctx.shadowBlur = 10;
    for (let i = 0; i < frameState.optimalRoute.length - 1; i++) {
      const from = frameState.optimalRoute[i];
      const to = frameState.optimalRoute[i + 1];
      if (from < nodes.length && to < nodes.length) {
        ctx.beginPath();
        ctx.moveTo(scaleX(nodes[from].x), scaleY(nodes[from].y));
        ctx.lineTo(scaleX(nodes[to].x), scaleY(nodes[to].y));
        ctx.stroke();
      }
    }
    ctx.shadowBlur = 0;
  }

  // 7. Nodes
  nodes.forEach((node) => {
    const px = scaleX(node.x), py = scaleY(node.y);
    if (node.isDepot) {
      // Depot: gradient circle
      const grad = ctx.createRadialGradient(px, py, 0, px, py, 13);
      grad.addColorStop(0, '#1E3A5F');
      grad.addColorStop(1, '#0F1D32');
      ctx.beginPath(); ctx.arc(px, py, 13, 0, Math.PI * 2);
      ctx.fillStyle = grad; ctx.fill();
      ctx.strokeStyle = 'rgba(255,255,255,0.15)'; ctx.lineWidth = 1; ctx.stroke();
      ctx.fillStyle = '#FFF'; ctx.font = 'bold 10px Inter';
      ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
      ctx.fillText('D', px, py);
    } else {
      const isActive = frameState.candidateEdge &&
        (frameState.candidateEdge[0] === node.id || frameState.candidateEdge[1] === node.id);
      const isOnOptimal = frameState.isSolved && frameState.optimalRoute?.includes(node.id);

      if (isActive) {
        // Pulse ring
        const pulse = 0.3 + 0.7 * (Math.sin((timestamp || 0) / 400 * Math.PI) * 0.5 + 0.5);
        ctx.globalAlpha = pulse;
        ctx.beginPath(); ctx.arc(px, py, 14, 0, Math.PI * 2);
        ctx.strokeStyle = scheme.primary; ctx.lineWidth = 1.5; ctx.stroke();
        ctx.globalAlpha = 1;
        ctx.beginPath(); ctx.arc(px, py, 8, 0, Math.PI * 2);
        ctx.fillStyle = scheme.primary; ctx.fill();
      } else if (isOnOptimal) {
        ctx.beginPath(); ctx.arc(px, py, 7, 0, Math.PI * 2);
        ctx.fillStyle = scheme.accent; ctx.fill();
        ctx.strokeStyle = 'rgba(255,255,255,0.2)'; ctx.lineWidth = 1; ctx.stroke();
      } else {
        ctx.beginPath(); ctx.arc(px, py, 6, 0, Math.PI * 2);
        ctx.fillStyle = '#1A2332';
        ctx.fill();
        ctx.strokeStyle = 'rgba(255,255,255,0.1)'; ctx.lineWidth = 1; ctx.stroke();
      }

      // Demand label
      ctx.fillStyle = (isActive || isOnOptimal) ? '#FFF' : 'rgba(255,255,255,0.4)';
      ctx.font = '8px Inter'; ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
      ctx.fillText(node.demand, px, py);
    }
  });
}
