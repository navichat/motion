/**
 * Motion Capture Integration
 * 
 * Real-time motion capture from webcam using MediaPipe or similar
 * For demonstration purposes, includes mouse-based pose control
 */

class MotionCaptureInterface {
    constructor(canvasId) {
        this.canvas = document.getElementById(canvasId);
        this.ctx = this.canvas.getContext('2d');
        this.width = 640;
        this.height = 480;
        
        this.canvas.width = this.width;
        this.canvas.height = this.height;
        
        // Motion capture state
        this.isCapturing = false;
        this.capturedPoses = [];
        this.currentPose = null;
        this.maxCaptureFrames = 300; // 5 seconds at 60fps
        
        // Mouse control for demo
        this.mouseControl = {
            isActive: false,
            lastX: 0,
            lastY: 0,
            joints: this.initializeJoints()
        };
        
        this.setupInterface();
        this.bindEvents();
        this.startRender();
    }
    
    setupInterface() {
        // Set canvas style
        this.canvas.style.border = '2px solid #00d4ff';
        this.canvas.style.borderRadius = '8px';
        this.canvas.style.background = '#1a1a2e';
        this.canvas.style.cursor = 'crosshair';
        
        // Create control panel
        const controlPanel = document.createElement('div');
        controlPanel.className = 'mocap-controls';
        controlPanel.innerHTML = `
            <div class="mocap-header">
                <h3>🎥 Motion Capture Interface</h3>
                <div class="mocap-status" id="mocap-status">Ready</div>
            </div>
            <div class="mocap-buttons">
                <button id="start-capture-btn" class="mocap-btn primary">Start Capture</button>
                <button id="stop-capture-btn" class="mocap-btn" disabled>Stop Capture</button>
                <button id="clear-capture-btn" class="mocap-btn">Clear</button>
                <button id="mouse-mode-btn" class="mocap-btn">Mouse Mode</button>
            </div>
            <div class="mocap-info">
                <div class="info-row">
                    <span>Frames Captured:</span>
                    <span id="frame-count">0</span>
                </div>
                <div class="info-row">
                    <span>Duration:</span>
                    <span id="capture-duration">0.0s</span>
                </div>
                <div class="info-row">
                    <span>Mode:</span>
                    <span id="capture-mode">Demo</span>
                </div>
            </div>
            <div class="mocap-actions">
                <button id="apply-capture-btn" class="mocap-btn success" disabled>Apply to Animation</button>
                <button id="export-capture-btn" class="mocap-btn" disabled>Export BVH</button>
            </div>
        `;
        
        this.canvas.parentNode.appendChild(controlPanel);
        this.addStyles();
    }
    
    addStyles() {
        const style = document.createElement('style');
        style.textContent = `
            .mocap-controls {
                background: rgba(0, 0, 0, 0.9);
                border: 1px solid #333;
                border-radius: 8px;
                padding: 15px;
                margin-top: 10px;
                color: white;
                font-family: Arial, sans-serif;
                max-width: 640px;
            }
            
            .mocap-header {
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin-bottom: 15px;
                border-bottom: 1px solid #333;
                padding-bottom: 10px;
            }
            
            .mocap-header h3 {
                margin: 0;
                color: #00d4ff;
                font-size: 16px;
            }
            
            .mocap-status {
                padding: 4px 12px;
                border-radius: 4px;
                font-size: 12px;
                font-weight: bold;
                background: rgba(0, 255, 136, 0.2);
                color: #00ff88;
                border: 1px solid #00ff88;
            }
            
            .mocap-status.capturing {
                background: rgba(255, 68, 68, 0.2);
                color: #ff4444;
                border-color: #ff4444;
            }
            
            .mocap-buttons {
                display: grid;
                grid-template-columns: 1fr 1fr 1fr 1fr;
                gap: 10px;
                margin-bottom: 15px;
            }
            
            .mocap-btn {
                background: rgba(0, 212, 255, 0.2);
                border: 1px solid #00d4ff;
                color: #00d4ff;
                padding: 8px 12px;
                border-radius: 4px;
                cursor: pointer;
                font-size: 12px;
                transition: all 0.3s ease;
            }
            
            .mocap-btn:hover:not(:disabled) {
                background: rgba(0, 212, 255, 0.4);
            }
            
            .mocap-btn:disabled {
                opacity: 0.5;
                cursor: not-allowed;
            }
            
            .mocap-btn.primary {
                background: rgba(0, 255, 136, 0.2);
                border-color: #00ff88;
                color: #00ff88;
            }
            
            .mocap-btn.primary:hover:not(:disabled) {
                background: rgba(0, 255, 136, 0.4);
            }
            
            .mocap-btn.success {
                background: rgba(255, 165, 0, 0.2);
                border-color: #ffa500;
                color: #ffa500;
            }
            
            .mocap-btn.success:hover:not(:disabled) {
                background: rgba(255, 165, 0, 0.4);
            }
            
            .mocap-info {
                display: grid;
                grid-template-columns: 1fr 1fr 1fr;
                gap: 15px;
                margin-bottom: 15px;
                font-size: 12px;
            }
            
            .info-row {
                display: flex;
                flex-direction: column;
                align-items: center;
            }
            
            .info-row span:first-child {
                color: #ccc;
                margin-bottom: 4px;
            }
            
            .info-row span:last-child {
                color: #00d4ff;
                font-weight: bold;
            }
            
            .mocap-actions {
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 10px;
            }
        `;
        document.head.appendChild(style);
    }
    
    initializeJoints() {
        // Simplified skeleton for demo
        return {
            head: { x: this.width / 2, y: 100 },
            neck: { x: this.width / 2, y: 130 },
            leftShoulder: { x: this.width / 2 - 40, y: 140 },
            rightShoulder: { x: this.width / 2 + 40, y: 140 },
            leftElbow: { x: this.width / 2 - 60, y: 180 },
            rightElbow: { x: this.width / 2 + 60, y: 180 },
            leftHand: { x: this.width / 2 - 80, y: 220 },
            rightHand: { x: this.width / 2 + 80, y: 220 },
            spine: { x: this.width / 2, y: 200 },
            hips: { x: this.width / 2, y: 260 },
            leftHip: { x: this.width / 2 - 20, y: 260 },
            rightHip: { x: this.width / 2 + 20, y: 260 },
            leftKnee: { x: this.width / 2 - 25, y: 320 },
            rightKnee: { x: this.width / 2 + 25, y: 320 },
            leftFoot: { x: this.width / 2 - 30, y: 380 },
            rightFoot: { x: this.width / 2 + 30, y: 380 }
        };
    }
    
    bindEvents() {
        // Control buttons
        document.getElementById('start-capture-btn').addEventListener('click', () => this.startCapture());
        document.getElementById('stop-capture-btn').addEventListener('click', () => this.stopCapture());
        document.getElementById('clear-capture-btn').addEventListener('click', () => this.clearCapture());
        document.getElementById('mouse-mode-btn').addEventListener('click', () => this.toggleMouseMode());
        document.getElementById('apply-capture-btn').addEventListener('click', () => this.applyCapture());
        document.getElementById('export-capture-btn').addEventListener('click', () => this.exportCapture());
        
        // Mouse controls for demo
        this.canvas.addEventListener('mousedown', (e) => this.handleMouseDown(e));
        this.canvas.addEventListener('mousemove', (e) => this.handleMouseMove(e));
        this.canvas.addEventListener('mouseup', () => this.handleMouseUp());
        this.canvas.addEventListener('mouseleave', () => this.handleMouseUp());
    }
    
    startCapture() {
        this.isCapturing = true;
        this.capturedPoses = [];
        this.updateStatus('Capturing...', 'capturing');
        this.updateButtons();
        
        // Start capture loop
        this.captureStartTime = Date.now();
        this.captureLoop();
    }
    
    stopCapture() {
        this.isCapturing = false;
        this.updateStatus('Capture Complete', 'complete');
        this.updateButtons();
    }
    
    clearCapture() {
        this.capturedPoses = [];
        this.updateStatus('Ready', 'ready');
        this.updateButtons();
        this.updateCaptureInfo();
    }
    
    toggleMouseMode() {
        this.mouseControl.isActive = !this.mouseControl.isActive;
        const btn = document.getElementById('mouse-mode-btn');
        
        if (this.mouseControl.isActive) {
            btn.textContent = 'Exit Mouse Mode';
            btn.style.background = 'rgba(255, 68, 68, 0.2)';
            btn.style.borderColor = '#ff4444';
            btn.style.color = '#ff4444';
            this.updateStatus('Mouse Control Active', 'mouse');
        } else {
            btn.textContent = 'Mouse Mode';
            btn.style.background = 'rgba(0, 212, 255, 0.2)';
            btn.style.borderColor = '#00d4ff';
            btn.style.color = '#00d4ff';
            this.updateStatus('Ready', 'ready');
        }
    }
    
    captureLoop() {
        if (!this.isCapturing) return;
        
        // Capture current pose (in demo mode, use mouse-controlled pose)
        const currentPose = this.getCurrentPose();
        this.capturedPoses.push({
            timestamp: Date.now() - this.captureStartTime,
            pose: currentPose
        });
        
        // Update info
        this.updateCaptureInfo();
        
        // Continue if under max frames
        if (this.capturedPoses.length < this.maxCaptureFrames) {
            setTimeout(() => this.captureLoop(), 16.67); // ~60 FPS
        } else {
            this.stopCapture();
        }
    }
    
    getCurrentPose() {
        // Return current joint positions (simplified)
        return JSON.parse(JSON.stringify(this.mouseControl.joints));
    }
    
    handleMouseDown(e) {
        if (!this.mouseControl.isActive) return;
        
        const rect = this.canvas.getBoundingClientRect();
        this.mouseControl.lastX = e.clientX - rect.left;
        this.mouseControl.lastY = e.clientY - rect.top;
        this.mouseControl.isDragging = true;
        
        // Find closest joint
        this.mouseControl.selectedJoint = this.findClosestJoint(
            this.mouseControl.lastX, 
            this.mouseControl.lastY
        );
    }
    
    handleMouseMove(e) {
        if (!this.mouseControl.isActive || !this.mouseControl.isDragging) return;
        
        const rect = this.canvas.getBoundingClientRect();
        const mouseX = e.clientX - rect.left;
        const mouseY = e.clientY - rect.top;
        
        // Update selected joint position
        if (this.mouseControl.selectedJoint) {
            this.mouseControl.joints[this.mouseControl.selectedJoint].x = mouseX;
            this.mouseControl.joints[this.mouseControl.selectedJoint].y = mouseY;
            
            // Apply basic inverse kinematics constraints
            this.applyIKConstraints(this.mouseControl.selectedJoint);
            
            // Apply real-time motion capture update
            this.applyToCurrentMotion();
        }
    }
    
    handleMouseUp() {
        this.mouseControl.isDragging = false;
        this.mouseControl.selectedJoint = null;
    }
    
    findClosestJoint(x, y) {
        let closestJoint = null;
        let minDistance = Infinity;
        
        for (const [jointName, joint] of Object.entries(this.mouseControl.joints)) {
            const distance = Math.sqrt(Math.pow(joint.x - x, 2) + Math.pow(joint.y - y, 2));
            if (distance < minDistance && distance < 30) { // 30px selection radius
                minDistance = distance;
                closestJoint = jointName;
            }
        }
        
        return closestJoint;
    }
    
    applyIKConstraints(movedJoint) {
        // Simple IK constraints for demo
        const joints = this.mouseControl.joints;
        
        switch (movedJoint) {
            case 'leftHand':
                // Update elbow to maintain arm length
                const leftArmLength = 60;
                const leftShoulderToHand = this.getDistance(joints.leftShoulder, joints.leftHand);
                if (leftShoulderToHand > leftArmLength * 2) {
                    const angle = Math.atan2(
                        joints.leftHand.y - joints.leftShoulder.y,
                        joints.leftHand.x - joints.leftShoulder.x
                    );
                    joints.leftElbow.x = joints.leftShoulder.x + Math.cos(angle) * leftArmLength;
                    joints.leftElbow.y = joints.leftShoulder.y + Math.sin(angle) * leftArmLength;
                }
                break;
                
            case 'rightHand':
                // Similar for right arm
                const rightArmLength = 60;
                const rightShoulderToHand = this.getDistance(joints.rightShoulder, joints.rightHand);
                if (rightShoulderToHand > rightArmLength * 2) {
                    const angle = Math.atan2(
                        joints.rightHand.y - joints.rightShoulder.y,
                        joints.rightHand.x - joints.rightShoulder.x
                    );
                    joints.rightElbow.x = joints.rightShoulder.x + Math.cos(angle) * rightArmLength;
                    joints.rightElbow.y = joints.rightShoulder.y + Math.sin(angle) * rightArmLength;
                }
                break;
        }
    }
    
    getDistance(joint1, joint2) {
        return Math.sqrt(Math.pow(joint1.x - joint2.x, 2) + Math.pow(joint1.y - joint2.y, 2));
    }
    
    updateStatus(message, type) {
        const statusEl = document.getElementById('mocap-status');
        statusEl.textContent = message;
        statusEl.className = `mocap-status ${type}`;
    }
    
    updateButtons() {
        const startBtn = document.getElementById('start-capture-btn');
        const stopBtn = document.getElementById('stop-capture-btn');
        const applyBtn = document.getElementById('apply-capture-btn');
        const exportBtn = document.getElementById('export-capture-btn');
        
        startBtn.disabled = this.isCapturing;
        stopBtn.disabled = !this.isCapturing;
        applyBtn.disabled = this.capturedPoses.length === 0;
        exportBtn.disabled = this.capturedPoses.length === 0;
    }
    
    updateCaptureInfo() {
        document.getElementById('frame-count').textContent = this.capturedPoses.length;
        document.getElementById('capture-duration').textContent = 
            (this.capturedPoses.length / 60).toFixed(1) + 's';
        document.getElementById('capture-mode').textContent = 
            this.mouseControl.isActive ? 'Mouse Control' : 'Demo';
    }
    
    applyCapture() {
        if (this.capturedPoses.length === 0) return;
        
        // Convert captured poses to motion data format
        const motionData = this.convertPosesToMotionData();
        
        // Apply to main animation system
        if (window.applyMotionCapture) {
            window.applyMotionCapture(motionData);
        }
        
        this.updateStatus('Applied to Animation', 'applied');
    }
    
    exportCapture() {
        if (this.capturedPoses.length === 0) return;
        
        // Convert to BVH format (simplified)
        const bvhData = this.convertToBVH();
        
        // Download as file
        const blob = new Blob([bvhData], { type: 'text/plain' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `mocap_${Date.now()}.bvh`;
        a.click();
        
        this.updateStatus('Exported BVH', 'exported');
    }
    
    convertPosesToMotionData() {
        // Convert captured poses to motion frames format compatible with BVH
        return this.capturedPoses.map(frame => {
            const pose = frame.pose;
            return this.convertSinglePoseToBVH(pose);
        });
    }
    
    convertSinglePoseToBVH(pose) {
        // Create a BVH frame array (69 channels for 100STYLE format)
        const frame = new Array(69).fill(0);
        
        // Map pose joints to BVH channels
        if (pose.hips) {
            frame[0] = pose.hips.x || 0; // Hip position X
            frame[1] = pose.hips.y || 100; // Hip position Y (offset from ground)
            frame[2] = pose.hips.z || 0; // Hip position Z
            frame[3] = 0; // Hip rotation Y
            frame[4] = 0; // Hip rotation X
            frame[5] = 0; // Hip rotation Z
        }
        
        // Calculate rotations from joint positions
        if (pose.neck && pose.head) {
            const headAngle = Math.atan2(pose.head.y - pose.neck.y, pose.head.x - pose.neck.x);
            frame[21] = headAngle * 180 / Math.PI; // Head rotation Y
        }
        
        // Right arm
        if (pose.rightShoulder && pose.rightElbow) {
            const shoulderAngle = Math.atan2(
                pose.rightElbow.y - pose.rightShoulder.y,
                pose.rightElbow.x - pose.rightShoulder.x
            );
            frame[27] = shoulderAngle * 180 / Math.PI; // Right shoulder rotation Y
        }
        
        if (pose.rightElbow && pose.rightHand) {
            const elbowAngle = Math.atan2(
                pose.rightHand.y - pose.rightElbow.y,
                pose.rightHand.x - pose.rightElbow.x
            );
            frame[30] = elbowAngle * 180 / Math.PI; // Right elbow rotation Y
        }
        
        // Left arm
        if (pose.leftShoulder && pose.leftElbow) {
            const shoulderAngle = Math.atan2(
                pose.leftElbow.y - pose.leftShoulder.y,
                pose.leftElbow.x - pose.leftShoulder.x
            );
            frame[39] = shoulderAngle * 180 / Math.PI; // Left shoulder rotation Y
        }
        
        if (pose.leftElbow && pose.leftHand) {
            const elbowAngle = Math.atan2(
                pose.leftHand.y - pose.leftElbow.y,
                pose.leftHand.x - pose.leftElbow.x
            );
            frame[42] = elbowAngle * 180 / Math.PI; // Left elbow rotation Y
        }
        
        // Legs (simplified)
        if (pose.hips && pose.leftKnee) {
            const hipAngle = Math.atan2(
                pose.leftKnee.y - pose.hips.y,
                pose.leftKnee.x - pose.hips.x
            );
            frame[60] = hipAngle * 180 / Math.PI; // Left hip rotation Y
        }
        
        if (pose.hips && pose.rightKnee) {
            const hipAngle = Math.atan2(
                pose.rightKnee.y - pose.hips.y,
                pose.rightKnee.x - pose.hips.x
            );
            frame[48] = hipAngle * 180 / Math.PI; // Right hip rotation Y
        }
        
        return frame;
    }
    
    applyToCurrentMotion() {
        if (!this.mouseControl.isActive || !this.currentPose) return null;
        
        // Convert current pose to BVH frame
        const bvhFrame = this.convertSinglePoseToBVH(this.mouseControl.joints);
        
        // Apply to main viewer
        if (window.applyMotionCaptureFrame) {
            window.applyMotionCaptureFrame(bvhFrame);
        }
        
        return bvhFrame;
    }
    
    convertToBVH() {
        // Simplified BVH export
        let bvh = `HIERARCHY
ROOT Hips
{
    OFFSET 0.0 0.0 0.0
    CHANNELS 6 Xposition Yposition Zposition Zrotation Xrotation Yrotation
}
MOTION
Frames: ${this.capturedPoses.length}
Frame Time: 0.016667
`;
        
        this.capturedPoses.forEach(frame => {
            const pose = frame.pose;
            bvh += `${pose.hips.x} ${pose.hips.y} 0 0 0 0\n`;
        });
        
        return bvh;
    }
    
    render() {
        // Clear canvas
        this.ctx.clearRect(0, 0, this.width, this.height);
        
        // Draw background grid
        this.drawGrid();
        
        // Draw skeleton
        this.drawSkeleton();
        
        // Draw UI elements
        this.drawUI();
    }
    
    drawGrid() {
        this.ctx.strokeStyle = '#333';
        this.ctx.lineWidth = 1;
        this.ctx.setLineDash([2, 2]);
        
        // Vertical lines
        for (let x = 0; x < this.width; x += 40) {
            this.ctx.beginPath();
            this.ctx.moveTo(x, 0);
            this.ctx.lineTo(x, this.height);
            this.ctx.stroke();
        }
        
        // Horizontal lines
        for (let y = 0; y < this.height; y += 40) {
            this.ctx.beginPath();
            this.ctx.moveTo(0, y);
            this.ctx.lineTo(this.width, y);
            this.ctx.stroke();
        }
        
        this.ctx.setLineDash([]);
    }
    
    drawSkeleton() {
        const joints = this.mouseControl.joints;
        
        // Draw bones
        this.ctx.strokeStyle = '#00d4ff';
        this.ctx.lineWidth = 3;
        
        const bones = [
            ['head', 'neck'],
            ['neck', 'leftShoulder'],
            ['neck', 'rightShoulder'],
            ['leftShoulder', 'leftElbow'],
            ['leftElbow', 'leftHand'],
            ['rightShoulder', 'rightElbow'],
            ['rightElbow', 'rightHand'],
            ['neck', 'spine'],
            ['spine', 'hips'],
            ['hips', 'leftHip'],
            ['hips', 'rightHip'],
            ['leftHip', 'leftKnee'],
            ['leftKnee', 'leftFoot'],
            ['rightHip', 'rightKnee'],
            ['rightKnee', 'rightFoot']
        ];
        
        bones.forEach(([joint1, joint2]) => {
            if (joints[joint1] && joints[joint2]) {
                this.ctx.beginPath();
                this.ctx.moveTo(joints[joint1].x, joints[joint1].y);
                this.ctx.lineTo(joints[joint2].x, joints[joint2].y);
                this.ctx.stroke();
            }
        });
        
        // Draw joints
        Object.entries(joints).forEach(([name, joint]) => {
            this.ctx.fillStyle = name === this.mouseControl.selectedJoint ? '#ff4444' : '#00ff88';
            this.ctx.beginPath();
            this.ctx.arc(joint.x, joint.y, 6, 0, Math.PI * 2);
            this.ctx.fill();
        });
    }
    
    drawUI() {
        // Draw instructions
        if (this.mouseControl.isActive) {
            this.ctx.fillStyle = '#ffa500';
            this.ctx.font = '14px Arial';
            this.ctx.fillText('Mouse Mode: Click and drag joints to pose', 10, 25);
        }
        
        // Draw capture indicator
        if (this.isCapturing) {
            this.ctx.fillStyle = '#ff4444';
            this.ctx.font = 'bold 16px Arial';
            this.ctx.fillText('● RECORDING', this.width - 120, 25);
        }
    }
    
    startRender() {
        const renderLoop = () => {
            this.render();
            requestAnimationFrame(renderLoop);
        };
        renderLoop();
    }
}

// Export for use in other scripts
if (typeof window !== 'undefined') {
    window.MotionCaptureInterface = MotionCaptureInterface;
}
