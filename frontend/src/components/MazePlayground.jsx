import { useState, useRef, useEffect } from 'react';
import './MazePlayground.css';

const INITIAL_MAZE = [
  [2, 1, 1, 2, 1, 2, 2, 2, 2, 2, 2, 1, 2, 2, 1, 2, 2, 2, 1, 1, 1, 2, 1, 2, 2, 2, 2, 2, 2, 2],
  [1, 2, 2, 1, 2, 1, 1, 2, 2, 2, 2, 2, 1, 2, 2, 1, 2, 1, 2, 1, 2, 1, 2, 2, 1, 2, 2, 2, 2, 2],
  [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 2, 1, 1, 1, 2, 1, 2, 2, 1, 1, 2, 2, 2, 2, 1, 1, 2],
  [2, 2, 2, 2, 1, 2, 1, 2, 2, 2, 2, 2, 1, 2, 2, 1, 2, 2, 2, 2, 1, 1, 2, 2, 2, 2, 1, 2, 1, 2],
  [1, 2, 1, 2, 2, 2, 1, 2, 1, 2, 2, 1, 1, 2, 2, 2, 2, 2, 1, 2, 2, 2, 1, 2, 2, 2, 2, 2, 1, 2],
  [2, 2, 2, 2, 2, 1, 1, 2, 1, 1, 2, 2, 2, 1, 1, 2, 2, 1, 2, 1, 1, 2, 1, 2, 2, 2, 1, 1, 2, 2],
  [2, 2, 2, 1, 2, 1, 1, 2, 1, 2, 1, 2, 2, 2, 1, 2, 2, 1, 2, 2, 2, 1, 1, 2, 2, 1, 1, 2, 2, 2],
  [2, 1, 2, 2, 1, 1, 1, 1, 1, 2, 1, 2, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 1, 2, 1, 2, 2, 2, 2],
  [2, 1, 1, 2, 2, 2, 1, 2, 2, 2, 2, 1, 2, 2, 1, 2, 1, 1, 2, 2, 2, 2, 2, 2, 1, 2, 1, 2, 2, 2],
  [1, 1, 2, 2, 2, 2, 2, 1, 2, 1, 1, 2, 1, 1, 2, 2, 2, 2, 2, 1, 1, 1, 2, 1, 2, 2, 2, 2, 2, 2],
  [2, 2, 2, 1, 1, 2, 2, 2, 2, 2, 2, 1, 1, 2, 2, 1, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2],
  [2, 2, 1, 2, 2, 2, 1, 2, 2, 2, 2, 2, 1, 1, 2, 2, 1, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1],
  [2, 1, 2, 1, 2, 1, 2, 1, 1, 1, 2, 1, 2, 2, 1, 1, 2, 2, 2, 2, 2, 2, 1, 1, 2, 2, 2, 1, 2, 2],
  [2, 2, 2, 1, 1, 2, 1, 2, 1, 1, 2, 1, 1, 1, 2, 1, 1, 2, 2, 2, 2, 1, 2, 2, 1, 2, 2, 2, 1, 2],
  [2, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 1, 2, 2, 1, 1, 2, 1, 2, 2, 2, 2, 1, 2, 2, 2, 2, 1, 2, 2],
  [2, 2, 1, 2, 4, 1, 2, 1, 1, 1, 2, 2, 2, 1, 1, 1, 1, 1, 2, 2, 2, 1, 1, 2, 2, 1, 2, 1, 1, 2],
  [2, 2, 1, 1, 2, 2, 1, 1, 2, 2, 2, 2, 2, 1, 1, 2, 1, 1, 1, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2],
  [1, 1, 2, 1, 1, 2, 2, 2, 2, 1, 1, 1, 2, 1, 2, 2, 2, 2, 2, 2, 2, 1, 1, 2, 2, 2, 2, 2, 2, 1],
  [1, 2, 2, 2, 1, 1, 1, 1, 1, 2, 2, 1, 2, 1, 2, 1, 2, 2, 2, 1, 2, 1, 1, 2, 2, 2, 2, 2, 2, 2],
  [2, 1, 2, 2, 1, 2, 1, 2, 2, 2, 1, 1, 1, 1, 2, 1, 2, 2, 1, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2],
  [2, 2, 1, 2, 2, 2, 2, 1, 2, 2, 2, 2, 1, 1, 2, 1, 2, 2, 2, 2, 2, 1, 1, 1, 2, 1, 2, 1, 1, 2],
  [2, 2, 2, 2, 1, 2, 2, 2, 2, 1, 2, 1, 2, 1, 1, 1, 1, 2, 2, 1, 1, 2, 2, 2, 2, 1, 1, 1, 1, 1],
  [2, 2, 1, 2, 2, 2, 1, 2, 2, 2, 2, 1, 2, 1, 2, 1, 1, 1, 1, 1, 2, 2, 2, 1, 2, 2, 1, 2, 2, 2],
  [2, 2, 1, 2, 2, 2, 1, 1, 2, 2, 2, 1, 1, 1, 2, 2, 2, 1, 2, 1, 2, 1, 1, 1, 2, 2, 2, 2, 2, 2],
  [2, 2, 3, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2, 1, 2, 2, 1, 1, 2, 2, 1, 2, 2, 1, 2, 2, 2, 2, 1, 2],
  [1, 2, 2, 2, 2, 2, 2, 2, 2, 1, 2, 1, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 2, 1, 2],
  [1, 1, 2, 2, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 1, 2, 1, 1, 2, 2, 2, 2, 2, 1, 1, 2, 2, 1, 2],
  [2, 2, 2, 1, 2, 2, 2, 2, 1, 2, 2, 1, 2, 2, 1, 2, 2, 2, 1, 2, 1, 1, 2, 1, 1, 1, 2, 1, 1, 2],
  [1, 2, 1, 2, 2, 1, 2, 2, 2, 2, 2, 2, 1, 2, 1, 2, 1, 2, 2, 1, 1, 2, 2, 2, 2, 2, 2, 2, 1, 2],
  [1, 2, 2, 1, 2, 2, 2, 2, 2, 2, 1, 1, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 1, 2, 1, 2, 2, 1, 2, 1],
];

function MazeCanvas({ grid, scale = 12, onClick, theme }) {
  const canvasRef = useRef(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const rows = grid.length;
    const cols = grid[0]?.length || 0;

    // Check if dark mode is enabled
    const isDark = theme === 'dark' || document.documentElement.getAttribute('data-theme') === 'dark';

    const valueToColor = (v) => {
      // Requested mapping:
      // 1 = wall → black (#000000)
      // 2 = open → white/light gray (#FFFFFF or #EEEEEE)
      // 3 = start → red (#FF0000)
      // 4 = goal → green (#00FF00)
      // 5 = predicted path → blue (#1E90FF)
      // 0 = pad → white (#FFFFFF)

      const lightPalette = {
        0: '#FFFFFF',    // pad
        1: '#000000',    // wall
        2: '#EEEEEE',    // open
        3: '#FF0000',    // start (red)
        4: '#00FF00',    // goal (green)
        5: '#1E90FF',    // predicted path (blue)
      };

      const darkPalette = {
        0: '#FFFFFF',    // pad - keep white for clarity
        1: '#000000',    // wall
        2: '#FFFFFF',    // open - use white in dark mode too so it contrasts with walls
        3: '#FF0000',    // start (red)
        4: '#00FF00',    // goal (green)
        5: '#1E90FF',    // predicted path (blue)
      };

      if (isDark) {
        return darkPalette[v] || '#FFFFFF';
      } else {
        return lightPalette[v] || '#FFFFFF';
      }
    };

    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < cols; c++) {
        const v = grid[r]?.[c];
        if (v !== undefined) {
          ctx.fillStyle = valueToColor(v);
          ctx.fillRect(c * scale, r * scale, scale, scale);
        }
      }
    }

    ctx.strokeStyle = isDark ? 'rgba(255, 255, 255, 0.1)' : 'rgba(0, 0, 0, 0.1)';
    ctx.lineWidth = 0.5;

    for (let r = 0; r <= rows; r++) {
      ctx.beginPath();
      ctx.moveTo(0, r * scale);
      ctx.lineTo(cols * scale, r * scale);
      ctx.stroke();
    }

    for (let c = 0; c <= cols; c++) {
      ctx.beginPath();
      ctx.moveTo(c * scale, 0);
      ctx.lineTo(c * scale, rows * scale);
      ctx.stroke();
    }
  }, [grid, scale, theme]);

  const handleClick = (e) => {
    if (!onClick) return;
    const canvas = canvasRef.current;
    if (!canvas) return;

    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    const col = Math.floor(x / scale);
    const row = Math.floor(y / scale);

    if (row >= 0 && row < grid.length && col >= 0 && col < (grid[0]?.length || 0)) {
      onClick(row, col);
    }
  };

  const rows = grid.length;
  const cols = grid[0]?.length || 0;

  return (
    <canvas
      ref={canvasRef}
      width={cols * scale}
      height={rows * scale}
      onClick={handleClick}
      className={onClick ? 'maze-canvas clickable' : 'maze-canvas'}
    />
  );
}

function MazePlayground() {
  const [inputMaze, setInputMaze] = useState(INITIAL_MAZE.map(row => [...row]));
  const [solvedMaze, setSolvedMaze] = useState(INITIAL_MAZE.map(row => [...row]));
  const [isGenerating, setIsGenerating] = useState(false);
  const [theme, setTheme] = useState('light');

  useEffect(() => {
    // Detect initial theme
    const currentTheme = document.documentElement.getAttribute('data-theme') || 'light';
    setTheme(currentTheme);

    // Watch for theme changes
    const observer = new MutationObserver((mutations) => {
      mutations.forEach((mutation) => {
        if (mutation.attributeName === 'data-theme') {
          const newTheme = document.documentElement.getAttribute('data-theme') || 'light';
          setTheme(newTheme);
        }
      });
    });

    observer.observe(document.documentElement, {
      attributes: true,
      attributeFilter: ['data-theme']
    });

    return () => observer.disconnect();
  }, []);

  // Mode for moving start/goal: null | 'start' | 'goal'
  const [moveMode, setMoveMode] = useState(null);

  const handleCellClick = (row, col) => {
    const newMaze = inputMaze.map(r => [...r]);
    if (newMaze[row]?.[col] === undefined) return;

    // If user is in move mode, place start or goal at clicked cell and
    // clear the previous one (set to open=2). Then exit move mode.
    if (moveMode === 'start') {
      // Clear any existing start (3) to open (2)
      for (let r = 0; r < newMaze.length; r++) {
        for (let c = 0; c < newMaze[0].length; c++) {
          if (newMaze[r][c] === 3) newMaze[r][c] = 2;
        }
      }
      // Place start unless it's a wall (1) - convert it to start regardless
      newMaze[row][col] = 3;
      setInputMaze(newMaze);
      setMoveMode(null);
      return;
    }

    if (moveMode === 'goal') {
      // Clear any existing goal (4) to open (2)
      for (let r = 0; r < newMaze.length; r++) {
        for (let c = 0; c < newMaze[0].length; c++) {
          if (newMaze[r][c] === 4) newMaze[r][c] = 2;
        }
      }
      newMaze[row][col] = 4;
      setInputMaze(newMaze);
      setMoveMode(null);
      return;
    }

    // Normal click behavior: toggle walls (1) and open (2).
    // Do NOT allow clicks to change start(3) or goal(4) into walls.
    const current = newMaze[row][col];
    if (current === 3 || current === 4) {
      // no-op: prevent turning start/goal into walls
      return;
    }

    newMaze[row][col] = current === 1 ? 2 : 1;
    setInputMaze(newMaze);
  };

  const handleGenerate = async () => {
    setIsGenerating(true);

    try {
      // Simple behavior: snapshot the current input maze and send it directly
      // to the realtime predict endpoint with cache-busting. No extra
      // normalization or complex checks — what you see is what we send.
      const payloadGrid = inputMaze.map(row => [...row]);

      const requestId = `${Date.now()}-${Math.random().toString(36).slice(2,8)}`;
      const url = `https://alphaxiv--tinyrecursive-eval-predict-realtime.modal.run?_r=${requestId}`;

      console.info('Sending payload to realtime predict:', { requestId, grid: payloadGrid });

      const response = await fetch(url, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Cache-Control': 'no-cache, no-store',
          'Pragma': 'no-cache',
          'X-Request-Id': requestId,
        },
        body: JSON.stringify({ task: 'maze', grid: payloadGrid, request_id: requestId }),
      });

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }

      const data = await response.json();

      console.info("data", data);
      
      if (data.solved_maze) {
        let grid = data.solved_maze;
        
        if (!Array.isArray(grid[0])) {
          const flat = grid;
          const L = flat.length;
          const side = Math.sqrt(L);
          if (Number.isInteger(side)) {
            const newGrid = [];
            for (let r = 0; r < side; r++) {
              newGrid.push(flat.slice(r * side, (r + 1) * side));
            }
            grid = newGrid;
          }
        }
        
        setSolvedMaze(grid);
      } else {
        throw new Error('Response missing solved_maze field');
      }
    } catch (err) {
      console.error('Error generating maze:', err);
    } finally {
      setIsGenerating(false);
    }
  };

  return (
    <div className="maze-playground">
      <div className="description">
        <p>
          Click on the maze cells to toggle between walls (black) and open spaces (light gray). Then
          click Generate to see the model solve the maze with a path (blue).
        </p>
      </div>

      <div className="generate-button-container">
        <button
          onClick={handleGenerate}
          disabled={isGenerating}
          className="generate-button"
        >
          <svg 
            className="play-icon" 
            viewBox="0 0 24 24" 
            fill="currentColor"
          >
            <path d="M8 5v14l11-7z" />
          </svg>
          {isGenerating ? 'Generating...' : 'Generate Solution'}
        </button>
  <div style={{display: 'flex', alignItems: 'center', gap: '0.5rem', marginLeft: '1rem'}}>
          <button
            className="generate-button"
            onClick={() => setMoveMode(moveMode === 'start' ? null : 'start')}
            type="button"
          >
            {moveMode === 'start' ? 'Cancel Move Start' : 'Move Start'}
          </button>

          <button
            className="generate-button"
            onClick={() => setMoveMode(moveMode === 'goal' ? null : 'goal')}
            type="button"
          >
            {moveMode === 'goal' ? 'Cancel Move Goal' : 'Move Goal'}
          </button>
        </div>
      </div>

      <div className="mazes-wrapper">
        <div className="mazes-grid">
          <div className="maze-section">
            <h3 className="maze-title">Input Maze</h3>
            <MazeCanvas 
              grid={inputMaze} 
              scale={8}
              onClick={handleCellClick}
              theme={theme}
            />
          </div>

          <div className="maze-section">
            <h3 className="maze-title">Solved Maze</h3>
            <MazeCanvas 
              grid={solvedMaze}
              scale={8}
              theme={theme}
            />
          </div>
        </div>
      </div>

      <div className="legend">
        <h4 className="legend-title">Legend</h4>
          <div className="legend-items">
            <div className="legend-item">
              <div className="legend-color" style={{ backgroundColor: '#000000' }}></div>
              <span>1 = Wall</span>
            </div>
            <div className="legend-item">
              <div className="legend-color" style={{ backgroundColor: theme === 'dark' ? '#FFFFFF' : '#EEEEEE' }}></div>
              <span>2 = Open</span>
            </div>
            <div className="legend-item">
              <div className="legend-color" style={{ backgroundColor: '#FF0000' }}></div>
              <span>3 = Start</span>
            </div>
            <div className="legend-item">
              <div className="legend-color" style={{ backgroundColor: '#00FF00' }}></div>
              <span>4 = Goal</span>
            </div>
            <div className="legend-item">
              <div className="legend-color" style={{ backgroundColor: '#1E90FF' }}></div>
              <span>5 = Predicted Path</span>
            </div>
            <div className="legend-item">
              <div className="legend-color" style={{ backgroundColor: '#FFFFFF' }}></div>
              <span>0 = Pad</span>
            </div>
          </div>
      </div>
    </div>
  );
}

export default MazePlayground;

