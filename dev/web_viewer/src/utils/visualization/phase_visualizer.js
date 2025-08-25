/**
 * RSMT Phase Manifold Visualizer
 * 
 * Real-time visualization of DeepPhase 2D manifold coordinates
 * Shows how motion cycles through the phase space over time
 */

class PhaseManifoldVisualizer {
    constructor(canvasId, width = 300, height = 300) {
        this.canvas = document.getElementById(canvasId);
        this.ctx = this.canvas.getContext('2d');
        this.width = width;
        this.height = height;
        
        this.canvas.width = width;
        this.canvas.height = height;
        
        // Phase data storage
        this.phaseHistory = [];
        this.maxHistoryLength = 200;
        this.currentPhase = [0, 0];
        
        // Visualization settings
        this.scale = Math.min(width, height) * 0.4;
        this.centerX = width / 2;
        this.centerY = height / 2;
        
        // Colors
        this.colors = {
            background: '#1a1a2e',
            grid: '#333',
            trajectory: '#00d4ff',
            current: '#ff4444',
            fade: '#00d4ff40'
        };
        
        this.setupCanvas();
        this.startAnimation();
    }
    
    setupCanvas() {
        // Set canvas style
        this.canvas.style.border = '1px solid #444';
        this.canvas.style.borderRadius = '8px';
        this.canvas.style.background = this.colors.background;
    }
    
    updatePhase(sx, sy) {
        this.currentPhase = [sx, sy];
        
        // Add to history with timestamp
        this.phaseHistory.push({
            x: sx,
            y: sy,
            timestamp: Date.now()
        });
        
        // Limit history length
        if (this.phaseHistory.length > this.maxHistoryLength) {
            this.phaseHistory.shift();
        }
    }
    
    clearHistory() {
        this.phaseHistory = [];
    }
    
    worldToCanvas(x, y) {
        return {
            x: this.centerX + x * this.scale,
            y: this.centerY - y * this.scale  // Flip Y axis
        };
    }
    
    drawGrid() {
        this.ctx.strokeStyle = this.colors.grid;
        this.ctx.lineWidth = 1;
        this.ctx.setLineDash([2, 2]);
        
        // Draw unit circle
        this.ctx.beginPath();
        this.ctx.arc(this.centerX, this.centerY, this.scale, 0, Math.PI * 2);
        this.ctx.stroke();
        
        // Draw axes
        this.ctx.beginPath();
        this.ctx.moveTo(this.centerX - this.scale, this.centerY);
        this.ctx.lineTo(this.centerX + this.scale, this.centerY);
        this.ctx.moveTo(this.centerX, this.centerY - this.scale);
        this.ctx.lineTo(this.centerX, this.centerY + this.scale);
        this.ctx.stroke();
        
        this.ctx.setLineDash([]);
    }
    
    drawTrajectory() {
        if (this.phaseHistory.length < 2) return;
        
        const now = Date.now();
        const fadeTime = 3000; // 3 seconds fade
        
        this.ctx.lineWidth = 2;
        
        for (let i = 1; i < this.phaseHistory.length; i++) {
            const prev = this.phaseHistory[i - 1];
            const curr = this.phaseHistory[i];
            
            // Calculate fade based on age
            const age = now - curr.timestamp;
            const alpha = Math.max(0, 1 - age / fadeTime);
            
            this.ctx.strokeStyle = `rgba(0, 212, 255, ${alpha * 0.7})`;
            
            const prevCanvas = this.worldToCanvas(prev.x, prev.y);
            const currCanvas = this.worldToCanvas(curr.x, curr.y);
            
            this.ctx.beginPath();
            this.ctx.moveTo(prevCanvas.x, prevCanvas.y);
            this.ctx.lineTo(currCanvas.x, currCanvas.y);
            this.ctx.stroke();
        }
    }
    
    drawCurrentPoint() {
        const canvasPos = this.worldToCanvas(this.currentPhase[0], this.currentPhase[1]);
        
        // Draw glow effect
        const gradient = this.ctx.createRadialGradient(
            canvasPos.x, canvasPos.y, 0,
            canvasPos.x, canvasPos.y, 15
        );
        gradient.addColorStop(0, 'rgba(255, 68, 68, 0.8)');
        gradient.addColorStop(1, 'rgba(255, 68, 68, 0)');
        
        this.ctx.fillStyle = gradient;
        this.ctx.beginPath();
        this.ctx.arc(canvasPos.x, canvasPos.y, 15, 0, Math.PI * 2);
        this.ctx.fill();
        
        // Draw center point
        this.ctx.fillStyle = this.colors.current;
        this.ctx.beginPath();
        this.ctx.arc(canvasPos.x, canvasPos.y, 4, 0, Math.PI * 2);
        this.ctx.fill();
    }
    
    drawLabels() {
        this.ctx.fillStyle = '#888';
        this.ctx.font = '12px Arial';
        this.ctx.textAlign = 'center';
        
        // Title
        this.ctx.fillText('Phase Manifold (sx, sy)', this.width / 2, 20);
        
        // Axis labels
        this.ctx.textAlign = 'center';
        this.ctx.fillText('1', this.centerX + this.scale, this.centerY - 5);
        this.ctx.fillText('-1', this.centerX - this.scale, this.centerY - 5);
        this.ctx.fillText('1', this.centerX + 5, this.centerY - this.scale);
        this.ctx.fillText('-1', this.centerX + 5, this.centerY + this.scale);
        
        // Current coordinates
        this.ctx.textAlign = 'left';
        this.ctx.fillStyle = '#00d4ff';
        this.ctx.fillText(
            `(${this.currentPhase[0].toFixed(3)}, ${this.currentPhase[1].toFixed(3)})`,
            10, this.height - 10
        );
    }
    
    render() {
        // Clear canvas
        this.ctx.clearRect(0, 0, this.width, this.height);
        
        // Draw components
        this.drawGrid();
        this.drawTrajectory();
        this.drawCurrentPoint();
        this.drawLabels();
    }
    
    startAnimation() {
        const animate = () => {
            this.render();
            requestAnimationFrame(animate);
        };
        animate();
    }
    
    // Mock phase generation for demo
    generateMockPhase(time) {
        const frequency = 0.002; // Slow cycling
        const sx = Math.cos(time * frequency);
        const sy = Math.sin(time * frequency * 1.3); // Slightly different frequency for lissajous
        this.updatePhase(sx, sy);
    }
}

/**
 * Quality Metrics Dashboard
 * Shows real-time transition quality metrics
 */
class QualityMetricsDashboard {
    constructor(containerId) {
        this.container = document.getElementById(containerId);
        this.metrics = {
            smoothness: 0,
            style_preservation: 0,
            foot_skating: 0,
            overall_quality: 0
        };
        
        this.setupDashboard();
    }
    
    setupDashboard() {
        this.container.innerHTML = `
            <div class="metrics-dashboard">
                <h3>Transition Quality Metrics</h3>
                <div class="metric-item">
                    <label>Smoothness:</label>
                    <div class="metric-bar">
                        <div class="metric-fill" id="smoothness-fill"></div>
                    </div>
                    <span class="metric-value" id="smoothness-value">0%</span>
                </div>
                <div class="metric-item">
                    <label>Style Preservation:</label>
                    <div class="metric-bar">
                        <div class="metric-fill" id="style-fill"></div>
                    </div>
                    <span class="metric-value" id="style-value">0%</span>
                </div>
                <div class="metric-item">
                    <label>Foot Contact:</label>
                    <div class="metric-bar">
                        <div class="metric-fill" id="foot-fill"></div>
                    </div>
                    <span class="metric-value" id="foot-value">0%</span>
                </div>
                <div class="metric-item overall">
                    <label>Overall Quality:</label>
                    <div class="metric-bar">
                        <div class="metric-fill" id="overall-fill"></div>
                    </div>
                    <span class="metric-value" id="overall-value">0%</span>
                </div>
            </div>
        `;
        
        // Add styles
        const style = document.createElement('style');
        style.textContent = `
            .metrics-dashboard {
                background: rgba(0, 0, 0, 0.8);
                padding: 15px;
                border-radius: 8px;
                border: 1px solid #333;
                min-width: 250px;
            }
            
            .metrics-dashboard h3 {
                color: #00d4ff;
                margin: 0 0 15px 0;
                font-size: 14px;
                text-align: center;
            }
            
            .metric-item {
                display: flex;
                align-items: center;
                margin-bottom: 10px;
                font-size: 12px;
            }
            
            .metric-item.overall {
                border-top: 1px solid #333;
                padding-top: 10px;
                margin-top: 10px;
            }
            
            .metric-item label {
                width: 100px;
                color: #ccc;
                font-size: 11px;
            }
            
            .metric-bar {
                flex: 1;
                height: 12px;
                background: #333;
                border-radius: 6px;
                overflow: hidden;
                margin: 0 8px;
                position: relative;
            }
            
            .metric-fill {
                height: 100%;
                background: linear-gradient(90deg, #ff4444, #ffaa00, #00ff88);
                width: 0%;
                transition: width 0.5s ease;
                border-radius: 6px;
            }
            
            .overall .metric-fill {
                background: linear-gradient(90deg, #00d4ff, #00ff88);
            }
            
            .metric-value {
                width: 35px;
                text-align: right;
                font-size: 11px;
                color: #00d4ff;
                font-weight: bold;
            }
        `;
        document.head.appendChild(style);
    }
    
    updateMetrics(metrics) {
        this.metrics = { ...this.metrics, ...metrics };
        
        // Update smoothness
        const smoothness = (this.metrics.smoothness || 0) * 100;
        document.getElementById('smoothness-fill').style.width = `${smoothness}%`;
        document.getElementById('smoothness-value').textContent = `${smoothness.toFixed(0)}%`;
        
        // Update style preservation
        const style = (this.metrics.style_preservation || 0) * 100;
        document.getElementById('style-fill').style.width = `${style}%`;
        document.getElementById('style-value').textContent = `${style.toFixed(0)}%`;
        
        // Update foot contact (inverted - lower is better)
        const foot = Math.max(0, (1 - (this.metrics.foot_skating || 0)) * 100);
        document.getElementById('foot-fill').style.width = `${foot}%`;
        document.getElementById('foot-value').textContent = `${foot.toFixed(0)}%`;
        
        // Update overall quality
        const overall = (this.metrics.overall_quality || 0) * 100;
        document.getElementById('overall-fill').style.width = `${overall}%`;
        document.getElementById('overall-value').textContent = `${overall.toFixed(0)}%`;
    }
}

// Export for use in other scripts
if (typeof window !== 'undefined') {
    window.PhaseManifoldVisualizer = PhaseManifoldVisualizer;
    window.QualityMetricsDashboard = QualityMetricsDashboard;
}
