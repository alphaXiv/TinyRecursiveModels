import { useEffect } from 'react'
import MazePlayground from './components/MazePlayground'
import './App.css'

function App() {
  useEffect(() => {
    // Check URL parameter for theme - handle both search and hash
    const searchParams = new URLSearchParams(window.location.search)
    const hashParams = new URLSearchParams(window.location.hash.split('?')[1])
    const themeParam = searchParams.get('theme') || hashParams.get('theme')
    
    if (themeParam === 'dark') {
      document.documentElement.setAttribute('data-theme', 'dark')
    } else {
      document.documentElement.setAttribute('data-theme', 'light')
    }
  }, [])

  return (
    <div className="app">
      <MazePlayground />
    </div>
  )
}

export default App
