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
      if (isDark) {
        // Dark mode palette
        const darkPalette = {
          0: '#FFFFFF',      // White (empty/unused)
          1: '#000000',      // Black (walls)
          2: '#64748b',      // Slate gray (open spaces) - more distinct from black
          3: '#00AA00',      // Green (goal)
          4: '#FFFF00',      // Yellow (start)
          5: '#1E90FF',      // Blue (solution path)
          6: '#FF00FF',      // Magenta
        };
        return darkPalette[v] || '#64748b';
      } else {
        // Light mode palette
        const lightPalette = {
          0: '#FFFFFF',      // White
          1: '#000000',      // Black (walls)
          2: '#EEEEEE',      // Light gray (open spaces)
          3: '#00AA00',      // Green (goal)
          4: '#FFFF00',      // Yellow (start)
          5: '#1E90FF',      // Blue (solution path)
          6: '#FF00FF',      // Magenta
        };
        return lightPalette[v] || '#CCCCCC';
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

  const handleCellClick = (row, col) => {
    const newMaze = inputMaze.map(r => [...r]);
    if (newMaze[row]?.[col] !== undefined) {
      newMaze[row][col] = newMaze[row][col] === 1 ? 2 : 1;
      setInputMaze(newMaze);
    }
  };

  const handleGenerate = async () => {
    setIsGenerating(true);
    
    try {
      const response = await fetch(
        'https://alphaxiv--tinyrecursive-eval-predict-realtime.modal.run',
        {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            task: 'maze',
            grid: inputMaze,
          }),
        }
      );

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
            <span>Wall</span>
          </div>
          <div className="legend-item">
            <div className="legend-color" style={{ backgroundColor: theme === 'dark' ? '#64748b' : '#EEEEEE' }}></div>
            <span>Open</span>
          </div>
          <div className="legend-item">
            <div className="legend-color" style={{ backgroundColor: '#FFFF00' }}></div>
            <span>Start</span>
          </div>
          <div className="legend-item">
            <div className="legend-color" style={{ backgroundColor: '#00AA00' }}></div>
            <span>Goal</span>
          </div>
          <div className="legend-item">
            <div className="legend-color" style={{ backgroundColor: '#1E90FF' }}></div>
            <span>Solution Path</span>
          </div>
        </div>
      </div>
    </div>
  );
}

export default MazePlayground;

