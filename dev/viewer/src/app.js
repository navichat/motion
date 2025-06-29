/**
 * Main Application Entry Point
 */

import React from 'react';
import ReactDOM from 'react-dom/client';
import MotionViewer from './components/MotionViewer.jsx';
import './styles/viewer.scss';

// Application state
let app = null;

// Initialize the motion viewer application
async function initializeApp() {
    console.log('Initializing Motion Viewer Application...');
    
    try {
        // Get the container element
        const container = document.getElementById('motion-viewer-container');
        if (!container) {
            throw new Error('Motion viewer container not found. Please add <div id="motion-viewer-container"></div> to your HTML.');
        }
        
        // Create React root
        const root = ReactDOM.createRoot(container);
        
        // Render the main component
        root.render(
            <React.StrictMode>
                <MotionViewerApp />
            </React.StrictMode>
        );
        
        app = { root, container };
        
        console.log('Motion Viewer Application initialized successfully');
        
    } catch (error) {
        console.error('Failed to initialize Motion Viewer:', error);
        
        // Show error message to user
        const container = document.getElementById('motion-viewer-container');
        if (container) {
            container.innerHTML = `
                <div style="
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    height: 400px;
                    background: #f8f9fa;
                    border: 2px dashed #dee2e6;
                    border-radius: 8px;
                    color: #6c757d;
                    text-align: center;
                    font-family: Arial, sans-serif;
                ">
                    <div>
                        <h3>Failed to Initialize Motion Viewer</h3>
                        <p>${error.message}</p>
                        <button onclick="location.reload()" style="
                            background: #007bff;
                            color: white;
                            border: none;
                            padding: 8px 16px;
                            border-radius: 4px;
                            cursor: pointer;
                            margin-top: 12px;
                        ">Retry</button>
                    </div>
                </div>
            `;
        }
    }
}

// Main application component
const MotionViewerApp = () => {
    const [config, setConfig] = React.useState({
        width: 800,
        height: 600,
        environment: 'classroom',
        showControls: true
    });
    
    const [logs, setLogs] = React.useState([]);
    const [showLogs, setShowLogs] = React.useState(false);
    
    // Add log entry
    const addLog = (message, type = 'info') => {
        const timestamp = new Date().toLocaleTimeString();
        setLogs(prev => [...prev.slice(-49), { timestamp, message, type }]);
    };
    
    // Event handlers
    const handleAvatarLoaded = (avatarInstance) => {
        addLog(`Avatar loaded: ${avatarInstance.name || 'Unknown'}`, 'success');
    };
    
    const handleAnimationStarted = (data) => {
        addLog(`Animation started: ${data.clip.name}`, 'info');
    };
    
    const handleError = (error) => {
        addLog(`Error: ${error.message}`, 'error');
    };
    
    // Responsive sizing
    React.useEffect(() => {
        const updateSize = () => {
            const container = document.getElementById('motion-viewer-container');
            if (container) {
                const rect = container.getBoundingClientRect();
                setConfig(prev => ({
                    ...prev,
                    width: Math.max(400, rect.width - 40),
                    height: Math.max(300, Math.min(600, rect.height - 80))
                }));
            }
        };
        
        updateSize();
        window.addEventListener('resize', updateSize);
        
        return () => window.removeEventListener('resize', updateSize);
    }, []);
    
    return (
        <div className="motion-viewer-app">
            <header className="app-header">
                <h1>Motion Viewer - 3D Avatar Showcase</h1>
                <div className="header-controls">
                    <button 
                        className={`log-toggle ${showLogs ? 'active' : ''}`}
                        onClick={() => setShowLogs(!showLogs)}
                    >
                        {showLogs ? 'Hide Logs' : 'Show Logs'}
                    </button>
                    <div className="environment-quick-switch">
                        <label>Environment:</label>
                        <select 
                            value={config.environment}
                            onChange={(e) => setConfig(prev => ({ ...prev, environment: e.target.value }))}
                        >
                            <option value="classroom">Classroom</option>
                            <option value="stage">Stage</option>
                            <option value="studio">Studio</option>
                            <option value="outdoor">Outdoor</option>
                        </select>
                    </div>
                </div>
            </header>
            
            <main className="app-main">
                <div className="viewer-container">
                    <MotionViewer
                        width={config.width}
                        height={config.height}
                        environment={config.environment}
                        showControls={config.showControls}
                        onAvatarLoaded={handleAvatarLoaded}
                        onAnimationStarted={handleAnimationStarted}
                        onError={handleError}
                    />
                </div>
                
                {showLogs && (
                    <div className="logs-panel">
                        <div className="logs-header">
                            <h3>Application Logs</h3>
                            <button onClick={() => setLogs([])}>Clear</button>
                        </div>
                        <div className="logs-content">
                            {logs.length === 0 ? (
                                <div className="no-logs">No logs yet...</div>
                            ) : (
                                logs.map((log, index) => (
                                    <div key={index} className={`log-entry ${log.type}`}>
                                        <span className="log-time">{log.timestamp}</span>
                                        <span className="log-message">{log.message}</span>
                                    </div>
                                ))
                            )}
                        </div>
                    </div>
                )}
            </main>
            
            <footer className="app-footer">
                <div className="footer-info">
                    <p>Motion Viewer v1.0 - Integrated 3D Avatar System</p>
                    <p>Phase 1: Chat Interface Integration | Phase 2: RSMT Neural Networks</p>
                </div>
            </footer>
        </div>
    );
};

// Expose initialization function globally
window.initializeMotionViewer = initializeApp;

// Auto-initialize if container exists on page load
document.addEventListener('DOMContentLoaded', () => {
    if (document.getElementById('motion-viewer-container')) {
        initializeApp();
    }
});

// Export for module usage
export { initializeApp as initializeMotionViewer, MotionViewerApp };
export default MotionViewer;
