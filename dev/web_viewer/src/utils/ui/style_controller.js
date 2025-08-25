/**
 * Real-time Style Interpolation Controls
 * 
 * Interactive interface for live style mixing and transition control
 */

class StyleInterpolationController {
    constructor(containerId) {
        this.container = document.getElementById(containerId);
        this.currentStyles = {};
        this.targetStyles = {};
        this.interpolationWeights = {};
        this.isLiveMode = false;
        
        this.setupInterface();
        this.bindEvents();
    }
    
    setupInterface() {
        this.container.innerHTML = `
            <div class="style-controller">
                <div class="controller-header">
                    <h3>🎨 Real-time Style Mixing</h3>
                    <div class="header-controls">
                        <button class="live-mode-btn ${this.isLiveMode ? 'active' : ''}" id="live-mode-btn">
                            ${this.isLiveMode ? 'Live: ON' : 'Live Mode'}
                        </button>
                        <button class="reset-btn" onclick="styleController.resetStyle()">Reset</button>
                    </div>
                </div>
                
                <div class="style-grid">
                    <div class="style-category">
                        <h4>Emotional Styles</h4>
                        <div class="style-slider-group">
                            <div class="style-slider">
                                <label>Aggressive ↔ Calm</label>
                                <input type="range" id="emotion-slider" min="-100" max="100" value="0" class="slider">
                                <span class="slider-value">0</span>
                            </div>
                            <div class="style-slider">
                                <label>Excited ↔ Depressed</label>
                                <input type="range" id="energy-emotion-slider" min="-100" max="100" value="0" class="slider">
                                <span class="slider-value">0</span>
                            </div>
                        </div>
                    </div>
                    
                    <div class="style-category">
                        <h4>Physical Characteristics</h4>
                        <div class="style-slider-group">
                            <div class="style-slider">
                                <label>Young ↔ Elderly</label>
                                <input type="range" id="age-slider" min="0" max="100" value="30" class="slider">
                                <span class="slider-value">30</span>
                            </div>
                            <div class="style-slider">
                                <label>Human ↔ Robot</label>
                                <input type="range" id="mechanical-slider" min="0" max="100" value="0" class="slider">
                                <span class="slider-value">0</span>
                            </div>
                        </div>
                    </div>
                    
                    <div class="style-category">
                        <h4>Movement Energy</h4>
                        <div class="style-slider-group">
                            <div class="style-slider">
                                <label>Speed</label>
                                <input type="range" id="speed-slider" min="0.5" max="2.0" step="0.1" value="1.0" class="slider">
                                <span class="slider-value">1.0x</span>
                            </div>
                            <div class="style-slider">
                                <label>Stride Length</label>
                                <input type="range" id="stride-slider" min="0.5" max="1.5" step="0.1" value="1.0" class="slider">
                                <span class="slider-value">1.0x</span>
                            </div>
                        </div>
                    </div>
                </div>
                
                <div class="style-presets">
                    <h4>Quick Presets</h4>
                    <div class="preset-buttons">
                        <button class="preset-btn" data-preset="neutral">Neutral</button>
                        <button class="preset-btn" data-preset="aggressive">Aggressive</button>
                        <button class="preset-btn" data-preset="elderly">Elderly</button>
                        <button class="preset-btn" data-preset="robotic">Robotic</button>
                        <button class="preset-btn" data-preset="excited">Excited</button>
                        <button class="preset-btn" data-preset="confident">Confident</button>
                    </div>
                </div>
                
                <div class="blend-controls">
                    <h4>Transition Controls</h4>
                    <div class="blend-settings">
                        <div class="setting-row">
                            <label>Transition Duration:</label>
                            <input type="range" id="duration-slider" min="0.5" max="5.0" step="0.1" value="1.0" class="slider">
                            <span class="slider-value">1.0s</span>
                        </div>
                        <div class="setting-row">
                            <label>Blend Curve:</label>
                            <select id="curve-select">
                                <option value="linear">Linear</option>
                                <option value="smooth" selected>Smooth</option>
                                <option value="ease_in">Ease In</option>
                                <option value="ease_out">Ease Out</option>
                                <option value="bounce">Bounce</option>
                            </select>
                        </div>
                    </div>
                </div>
                
                <div class="style-feedback">
                    <div class="current-style">
                        <strong>Current Style:</strong> <span id="style-description">Neutral Walking</span>
                    </div>
                    <div class="blend-status">
                        <strong>Status:</strong> <span id="blend-status">Ready</span>
                    </div>
                </div>
            </div>
        `;
        
        this.addStyles();
    }
    
    addStyles() {
        const style = document.createElement('style');
        style.textContent = `
            .style-controller {
                background: rgba(0, 0, 0, 0.9);
                border: 1px solid #333;
                border-radius: 10px;
                padding: 20px;
                color: white;
                font-family: Arial, sans-serif;
                max-width: 400px;
                max-height: 600px;
                overflow-y: auto;
            }
            
            .controller-header {
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin-bottom: 20px;
                border-bottom: 1px solid #333;
                padding-bottom: 10px;
            }
            
            .controller-header h3 {
                margin: 0;
                color: #00d4ff;
                font-size: 16px;
            }
            
            .header-controls {
                display: flex;
                gap: 8px;
            }
            
            .live-mode-btn, .reset-btn {
                background: rgba(0, 212, 255, 0.2);
                border: 1px solid #00d4ff;
                color: #00d4ff;
                padding: 6px 12px;
                border-radius: 4px;
                cursor: pointer;
                font-size: 12px;
                transition: all 0.3s ease;
            }
            
            .live-mode-btn:hover, .reset-btn:hover {
                background: rgba(0, 212, 255, 0.4);
            }
            
            .live-mode-btn.active {
                background: #00d4ff;
                color: #1a1a2e;
            }
            
            .reset-btn {
                border-color: #ffa500;
                color: #ffa500;
                background: rgba(255, 165, 0, 0.2);
            }
            
            .reset-btn:hover {
                background: rgba(255, 165, 0, 0.4);
            }
            
            .style-category {
                margin-bottom: 20px;
            }
            
            .style-category h4 {
                color: #ffa500;
                margin: 0 0 10px 0;
                font-size: 14px;
            }
            
            .style-slider {
                margin-bottom: 12px;
            }
            
            .style-slider label {
                display: block;
                font-size: 12px;
                margin-bottom: 4px;
                color: #ccc;
            }
            
            .slider {
                width: 100%;
                height: 6px;
                border-radius: 3px;
                background: #333;
                outline: none;
                margin-right: 10px;
            }
            
            .slider::-webkit-slider-thumb {
                appearance: none;
                width: 16px;
                height: 16px;
                border-radius: 50%;
                background: #00d4ff;
                cursor: pointer;
            }
            
            .slider::-moz-range-thumb {
                width: 16px;
                height: 16px;
                border-radius: 50%;
                background: #00d4ff;
                cursor: pointer;
                border: none;
            }
            
            .slider-value {
                display: inline-block;
                width: 40px;
                text-align: right;
                font-size: 11px;
                color: #00d4ff;
                font-weight: bold;
            }
            
            .preset-buttons {
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 8px;
                margin: 10px 0;
            }
            
            .preset-btn {
                background: rgba(255, 165, 0, 0.2);
                border: 1px solid #ffa500;
                color: #ffa500;
                padding: 8px 12px;
                border-radius: 4px;
                cursor: pointer;
                font-size: 11px;
                transition: all 0.3s ease;
            }
            
            .preset-btn:hover {
                background: rgba(255, 165, 0, 0.4);
            }
            
            .setting-row {
                display: flex;
                align-items: center;
                margin-bottom: 10px;
                font-size: 12px;
            }
            
            .setting-row label {
                width: 120px;
                color: #ccc;
            }
            
            .setting-row select {
                background: #333;
                border: 1px solid #555;
                color: white;
                padding: 4px 8px;
                border-radius: 4px;
                font-size: 11px;
            }
            
            .style-feedback {
                margin-top: 15px;
                padding-top: 15px;
                border-top: 1px solid #333;
                font-size: 12px;
            }
            
            .style-feedback div {
                margin-bottom: 8px;
            }
            
            #style-description {
                color: #00ff88;
            }
            
            #blend-status {
                color: #00d4ff;
            }
        `;
        document.head.appendChild(style);
    }
    
    bindEvents() {
        // Live mode toggle
        const liveModeBtn = document.getElementById('live-mode-btn');
        liveModeBtn.addEventListener('click', () => this.toggleLiveMode());
        
        // Slider events
        const sliders = this.container.querySelectorAll('.slider');
        sliders.forEach(slider => {
            slider.addEventListener('input', (e) => this.handleSliderChange(e));
            slider.addEventListener('change', (e) => this.handleSliderChange(e));
        });
        
        // Preset buttons
        const presetBtns = this.container.querySelectorAll('.preset-btn');
        presetBtns.forEach(btn => {
            btn.addEventListener('click', (e) => this.applyPreset(e.target.dataset.preset));
        });
        
        // Blend curve selection
        const curveSelect = document.getElementById('curve-select');
        curveSelect.addEventListener('change', (e) => this.updateBlendCurve(e.target.value));
    }
    
    toggleLiveMode() {
        this.isLiveMode = !this.isLiveMode;
        const btn = document.getElementById('live-mode-btn');
        
        if (this.isLiveMode) {
            btn.textContent = 'Disable Live Mode';
            btn.classList.add('active');
            this.updateBlendStatus('Live mode active - real-time style mixing');
            this.startLiveUpdates();
        } else {
            btn.textContent = 'Enable Live Mode';
            btn.classList.remove('active');
            this.updateBlendStatus('Live mode disabled');
            this.stopLiveUpdates();
        }
    }
    
    handleSliderChange(event) {
        const slider = event.target;
        const value = parseFloat(slider.value);
        const valueSpan = slider.parentElement.querySelector('.slider-value');
        
        // Update display value
        switch (slider.id) {
            case 'speed-slider':
            case 'stride-slider':
                valueSpan.textContent = value.toFixed(1) + 'x';
                break;
            case 'duration-slider':
                valueSpan.textContent = value.toFixed(1) + 's';
                break;
            default:
                valueSpan.textContent = Math.round(value);
        }
        
        // Update style description
        this.updateStyleDescription();
        
        // Apply changes if in live mode
        if (this.isLiveMode) {
            this.applyCurrentStyle();
        }
    }
    
    updateStyleDescription() {
        const emotion = document.getElementById('emotion-slider').value;
        const energy = document.getElementById('energy-emotion-slider').value;
        const age = document.getElementById('age-slider').value;
        const mechanical = document.getElementById('mechanical-slider').value;
        const speed = document.getElementById('speed-slider').value;
        
        let description = '';
        
        // Age component
        if (age > 70) description += 'Elderly ';
        else if (age < 30) description += 'Youthful ';
        
        // Mechanical component
        if (mechanical > 50) description += 'Robotic ';
        
        // Emotional component
        if (emotion > 30) description += 'Aggressive ';
        else if (emotion < -30) description += 'Calm ';
        
        if (energy > 30) description += 'Excited ';
        else if (energy < -30) description += 'Depressed ';
        
        // Speed component
        if (speed > 1.3) description += 'Fast ';
        else if (speed < 0.8) description += 'Slow ';
        
        description += 'Walking';
        
        document.getElementById('style-description').textContent = description;
    }
    
    applyPreset(presetName) {
        const presets = {
            neutral: { emotion: 0, energy: 0, age: 30, mechanical: 0, speed: 1.0, stride: 1.0 },
            aggressive: { emotion: 70, energy: 40, age: 25, mechanical: 0, speed: 1.3, stride: 1.2 },
            elderly: { emotion: -20, energy: -40, age: 80, mechanical: 0, speed: 0.7, stride: 0.8 },
            robotic: { emotion: 0, energy: 0, age: 30, mechanical: 80, speed: 1.0, stride: 1.0 },
            excited: { emotion: 20, energy: 80, age: 20, mechanical: 0, speed: 1.4, stride: 1.3 },
            confident: { emotion: 30, energy: 20, age: 35, mechanical: 0, speed: 1.1, stride: 1.1 }
        };
        
        const preset = presets[presetName];
        if (!preset) return;
        
        // Apply preset values to sliders
        document.getElementById('emotion-slider').value = preset.emotion;
        document.getElementById('energy-emotion-slider').value = preset.energy;
        document.getElementById('age-slider').value = preset.age;
        document.getElementById('mechanical-slider').value = preset.mechanical;
        document.getElementById('speed-slider').value = preset.speed;
        document.getElementById('stride-slider').value = preset.stride;
        
        // Update all slider displays
        this.container.querySelectorAll('.slider').forEach(slider => {
            const event = new Event('input');
            slider.dispatchEvent(event);
        });
        
        // Apply if in live mode
        if (this.isLiveMode) {
            this.applyCurrentStyle();
        }
        
        this.updateBlendStatus(`Applied ${presetName} preset`);
    }
    
    updateBlendCurve(curve) {
        this.updateBlendStatus(`Blend curve set to ${curve}`);
    }
    
    updateBlendStatus(message) {
        document.getElementById('blend-status').textContent = message;
    }
    
    getCurrentStyleParameters() {
        return {
            emotion: parseFloat(document.getElementById('emotion-slider').value),
            energyEmotion: parseFloat(document.getElementById('energy-emotion-slider').value),
            age: parseFloat(document.getElementById('age-slider').value),
            mechanical: parseFloat(document.getElementById('mechanical-slider').value),
            speed: parseFloat(document.getElementById('speed-slider').value),
            stride: parseFloat(document.getElementById('stride-slider').value),
            duration: parseFloat(document.getElementById('duration-slider').value),
            curve: document.getElementById('curve-select').value
        };
    }
    
    applyCurrentStyle() {
        const params = this.getCurrentStyleParameters();
        
        // Trigger style application (would integrate with main system)
        if (window.applyStyleParameters) {
            window.applyStyleParameters(params);
        }
        
        this.updateBlendStatus('Applying style parameters...');
    }
    
    startLiveUpdates() {
        this.liveUpdateInterval = setInterval(() => {
            if (this.isLiveMode) {
                this.applyCurrentStyle();
            }
        }, 100); // 10 FPS updates
    }
    
    stopLiveUpdates() {
        if (this.liveUpdateInterval) {
            clearInterval(this.liveUpdateInterval);
            this.liveUpdateInterval = null;
        }
    }
    
    resetStyle() {
        // Reset all sliders to default values
        const sliders = this.container.querySelectorAll('.slider');
        sliders.forEach(slider => {
            const defaultValue = slider.dataset.default || slider.min || '0';
            slider.value = defaultValue;
            
            // Update display
            const valueSpan = slider.parentElement.querySelector('.slider-value');
            if (valueSpan) {
                valueSpan.textContent = this.formatSliderValue(slider);
            }
        });
        
        // Reset current styles
        this.currentStyles = {};
        
        // Update status
        this.updateBlendStatus('Styles reset to defaults');
        
        // Apply if live mode is active
        if (this.isLiveMode) {
            this.applyCurrentStyle();
        }
    }
    
    formatSliderValue(slider) {
        let value = parseFloat(slider.value);
        
        // Format based on slider type
        if (slider.id.includes('speed') || slider.id.includes('stride')) {
            return value.toFixed(1) + 'x';
        } else if (slider.id.includes('duration')) {
            return value.toFixed(1) + 's';
        } else {
            return Math.round(value).toString();
        }
    }
    
    getCurrentStyle() {
        // Return current style parameters for motion application
        return {
            emotional: {
                intensity: this.getSliderValue('energy-slider') / 100,
                valence: (this.getSliderValue('happiness-slider') - 50) / 50 // -1 to 1
            },
            physical: {
                age: this.getSliderValue('age-slider') / 100,
                build: this.getSliderValue('weight-slider') / 100
            },
            energy: {
                level: this.getSliderValue('energy-slider') / 100,
                tempo: this.getSliderValue('speed-slider')
            },
            mechanical: this.getSliderValue('mechanical-slider') / 100
        };
    }
    
    getSliderValue(sliderId) {
        const slider = document.getElementById(sliderId);
        return slider ? parseFloat(slider.value) : 0;
    }
}

// Export for use in other scripts
if (typeof window !== 'undefined') {
    window.StyleInterpolationController = StyleInterpolationController;
}
