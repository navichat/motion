/**
 * Motion Viewer - Main React component for the 3D avatar viewer
 */

import React, { useEffect, useRef, useState, useCallback } from 'react';
import { Player } from '../src/Player.js';
import { AvatarLoader } from '../src/AvatarLoader.js';
import { AnimationController } from '../src/AnimationController.js';
import { SceneManager } from '../src/SceneManager.js';

const MotionViewer = ({
    width = 800,
    height = 600,
    environment = 'classroom',
    showControls = true,
    onAvatarLoaded,
    onAnimationStarted,
    onError
}) => {
    const canvasRef = useRef(null);
    const playerRef = useRef(null);
    const sceneManagerRef = useRef(null);
    const avatarLoaderRef = useRef(null);
    const animationControllerRef = useRef(null);
    
    const [isLoading, setIsLoading] = useState(true);
    const [currentAvatar, setCurrentAvatar] = useState(null);
    const [availableAvatars, setAvailableAvatars] = useState([]);
    const [availableAnimations, setAvailableAnimations] = useState([]);
    const [playbackState, setPlaybackState] = useState({
        isPlaying: false,
        currentTime: 0,
        duration: 0,
        speed: 1.0
    });
    const [error, setError] = useState(null);
    
    // Initialize the 3D viewer
    useEffect(() => {
        if (!canvasRef.current) return;
        
        const initViewer = async () => {
            try {
                setIsLoading(true);
                
                // Initialize core components
                const player = new Player(canvasRef.current);
                const sceneManager = new SceneManager(player.scene);
                const avatarLoader = new AvatarLoader();
                
                playerRef.current = player;
                sceneManagerRef.current = sceneManager;
                avatarLoaderRef.current = avatarLoader;
                
                // Load environment
                await sceneManager.loadEnvironment(environment);
                
                // Load available avatars
                const avatars = await avatarLoader.getAvailableAvatars();
                setAvailableAvatars(avatars);
                
                // Load available animations
                const animations = await loadAvailableAnimations();
                setAvailableAnimations(animations);
                
                // Handle canvas resize
                const handleResize = () => {
                    player.handleCanvasResize();
                };
                
                window.addEventListener('resize', handleResize);
                handleResize();
                
                setIsLoading(false);
                
                // Auto-load first avatar if available
                if (avatars.length > 0) {
                    await loadAvatar(avatars[0]);
                }
                
                return () => {
                    window.removeEventListener('resize', handleResize);
                };
                
            } catch (err) {
                console.error('Failed to initialize viewer:', err);
                setError(err.message);
                setIsLoading(false);
                onError?.(err);
            }
        };
        
        initViewer();
        
        // Cleanup on unmount
        return () => {
            if (animationControllerRef.current) {
                animationControllerRef.current.dispose();
            }
            if (sceneManagerRef.current) {
                sceneManagerRef.current.dispose();
            }
            if (avatarLoaderRef.current) {
                avatarLoaderRef.current.dispose();
            }
            if (playerRef.current) {
                playerRef.current.dispose();
            }
        };
    }, [environment]);
    
    // Load avatar
    const loadAvatar = useCallback(async (avatarConfig) => {
        if (!avatarLoaderRef.current || !sceneManagerRef.current) return;
        
        try {
            setIsLoading(true);
            
            // Remove current avatar if exists
            if (currentAvatar) {
                sceneManagerRef.current.removeAvatar('main');
            }
            
            // Load new avatar
            const avatarInstance = await avatarLoaderRef.current.loadAvatar(avatarConfig);
            
            // Add to scene
            sceneManagerRef.current.addAvatar('main', avatarInstance);
            
            // Create animation controller
            const animController = new AnimationController(avatarInstance);
            animationControllerRef.current = animController;
            
            // Set up animation event listeners
            animController.on('animationStarted', (data) => {
                onAnimationStarted?.(data);
            });
            
            animController.on('update', (state) => {
                setPlaybackState(state);
            });
            
            // Update player tick to include animation controller
            playerRef.current.on('tick', (delta) => {
                animController.update(delta);
            });
            
            setCurrentAvatar(avatarInstance);
            setIsLoading(false);
            
            onAvatarLoaded?.(avatarInstance);
            
        } catch (err) {
            console.error('Failed to load avatar:', err);
            setError(`Failed to load avatar: ${err.message}`);
            setIsLoading(false);
            onError?.(err);
        }
    }, [currentAvatar, onAvatarLoaded, onAnimationStarted, onError]);
    
    // Load available animations from assets
    const loadAvailableAnimations = async () => {
        try {
            // Load chat animations (JSON format)
            const chatAnimsResponse = await fetch('/assets/animations/index.json');
            const chatAnimations = await chatAnimsResponse.json();
            
            // Phase 2: Load RSMT animations (BVH format)
            // const rsmtAnimsResponse = await fetch('/assets/animations/100style/index.json');
            // const rsmtAnimations = await rsmtAnimsResponse.json();
            
            return [
                ...chatAnimations.map(anim => ({ ...anim, format: 'json', source: 'chat' }))
                // Phase 2: ...rsmtAnimations.map(anim => ({ ...anim, format: 'bvh', source: 'rsmt' }))
            ];
            
        } catch (error) {
            console.warn('Could not load animation index, using defaults');
            return [];
        }
    };
    
    // Animation controls
    const playAnimation = useCallback((animationData, options = {}) => {
        if (!animationControllerRef.current) return;
        
        animationControllerRef.current.playAnimation(animationData, options);
    }, []);
    
    const pauseAnimation = useCallback(() => {
        if (!animationControllerRef.current) return;
        animationControllerRef.current.pause();
    }, []);
    
    const resumeAnimation = useCallback(() => {
        if (!animationControllerRef.current) return;
        animationControllerRef.current.resume();
    }, []);
    
    const stopAnimation = useCallback(() => {
        if (!animationControllerRef.current) return;
        animationControllerRef.current.stop();
    }, []);
    
    const setAnimationSpeed = useCallback((speed) => {
        if (!animationControllerRef.current) return;
        animationControllerRef.current.setSpeed(speed);
    }, []);
    
    const setAnimationTime = useCallback((time) => {
        if (!animationControllerRef.current) return;
        animationControllerRef.current.setCurrentTime(time);
    }, []);
    
    // Environment controls
    const changeEnvironment = useCallback(async (environmentType) => {
        if (!sceneManagerRef.current) return;
        
        try {
            await sceneManagerRef.current.loadEnvironment(environmentType);
        } catch (err) {
            console.error('Failed to change environment:', err);
            setError(`Failed to change environment: ${err.message}`);
        }
    }, []);
    
    return (
        <div className="motion-viewer" style={{ width, height, position: 'relative' }}>
            <canvas
                ref={canvasRef}
                style={{
                    width: '100%',
                    height: '100%',
                    display: 'block',
                    background: 'linear-gradient(to bottom, #87CEEB, #E0F6FF)'
                }}
            />
            
            {isLoading && (
                <div className="loading-overlay">
                    <div className="loading-spinner"></div>
                    <div className="loading-text">Loading 3D Viewer...</div>
                </div>
            )}
            
            {error && (
                <div className="error-overlay">
                    <div className="error-message">
                        <h3>Error</h3>
                        <p>{error}</p>
                        <button onClick={() => setError(null)}>Dismiss</button>
                    </div>
                </div>
            )}
            
            {showControls && !isLoading && !error && (
                <ViewerControls
                    availableAvatars={availableAvatars}
                    availableAnimations={availableAnimations}
                    currentAvatar={currentAvatar}
                    playbackState={playbackState}
                    onLoadAvatar={loadAvatar}
                    onPlayAnimation={playAnimation}
                    onPauseAnimation={pauseAnimation}
                    onResumeAnimation={resumeAnimation}
                    onStopAnimation={stopAnimation}
                    onSetSpeed={setAnimationSpeed}
                    onSetTime={setAnimationTime}
                    onChangeEnvironment={changeEnvironment}
                />
            )}
        </div>
    );
};

// Control panel component
const ViewerControls = ({
    availableAvatars,
    availableAnimations,
    currentAvatar,
    playbackState,
    onLoadAvatar,
    onPlayAnimation,
    onPauseAnimation,
    onResumeAnimation,
    onStopAnimation,
    onSetSpeed,
    onSetTime,
    onChangeEnvironment
}) => {
    const [selectedAnimation, setSelectedAnimation] = useState(null);
    const [showAvatarSelector, setShowAvatarSelector] = useState(false);
    const [showAnimationSelector, setShowAnimationSelector] = useState(false);
    
    return (
        <div className="viewer-controls">
            <div className="control-panel">
                {/* Avatar Selection */}
                <div className="control-group">
                    <label>Avatar:</label>
                    <button 
                        className="selector-button"
                        onClick={() => setShowAvatarSelector(!showAvatarSelector)}
                    >
                        {currentAvatar ? currentAvatar.name || 'Current Avatar' : 'Select Avatar'}
                    </button>
                    {showAvatarSelector && (
                        <div className="dropdown-menu">
                            {availableAvatars.map((avatar, index) => (
                                <button
                                    key={index}
                                    className="dropdown-item"
                                    onClick={() => {
                                        onLoadAvatar(avatar);
                                        setShowAvatarSelector(false);
                                    }}
                                >
                                    {avatar.name}
                                </button>
                            ))}
                        </div>
                    )}
                </div>
                
                {/* Animation Selection */}
                <div className="control-group">
                    <label>Animation:</label>
                    <button 
                        className="selector-button"
                        onClick={() => setShowAnimationSelector(!showAnimationSelector)}
                    >
                        {selectedAnimation ? selectedAnimation.name : 'Select Animation'}
                    </button>
                    {showAnimationSelector && (
                        <div className="dropdown-menu">
                            {availableAnimations.map((animation, index) => (
                                <button
                                    key={index}
                                    className="dropdown-item"
                                    onClick={() => {
                                        setSelectedAnimation(animation);
                                        setShowAnimationSelector(false);
                                    }}
                                >
                                    {animation.name} ({animation.source})
                                </button>
                            ))}
                        </div>
                    )}
                </div>
                
                {/* Playback Controls */}
                <div className="control-group">
                    <div className="playback-controls">
                        <button
                            onClick={() => selectedAnimation && onPlayAnimation(selectedAnimation)}
                            disabled={!selectedAnimation || !currentAvatar}
                            className="play-button"
                        >
                            Play
                        </button>
                        <button
                            onClick={playbackState.isPlaying ? onPauseAnimation : onResumeAnimation}
                            disabled={!currentAvatar}
                            className="pause-button"
                        >
                            {playbackState.isPlaying ? 'Pause' : 'Resume'}
                        </button>
                        <button
                            onClick={onStopAnimation}
                            disabled={!currentAvatar}
                            className="stop-button"
                        >
                            Stop
                        </button>
                    </div>
                </div>
                
                {/* Speed Control */}
                <div className="control-group">
                    <label>Speed: {playbackState.speed?.toFixed(1)}x</label>
                    <input
                        type="range"
                        min="0.1"
                        max="3.0"
                        step="0.1"
                        value={playbackState.speed || 1.0}
                        onChange={(e) => onSetSpeed(parseFloat(e.target.value))}
                        className="speed-slider"
                    />
                </div>
                
                {/* Timeline */}
                <div className="control-group">
                    <label>
                        Time: {playbackState.currentTime?.toFixed(1)}s / {playbackState.duration?.toFixed(1)}s
                    </label>
                    <input
                        type="range"
                        min="0"
                        max={playbackState.duration || 1}
                        step="0.1"
                        value={playbackState.currentTime || 0}
                        onChange={(e) => onSetTime(parseFloat(e.target.value))}
                        className="timeline-slider"
                        disabled={!playbackState.duration}
                    />
                </div>
                
                {/* Environment Selection */}
                <div className="control-group">
                    <label>Environment:</label>
                    <select 
                        onChange={(e) => onChangeEnvironment(e.target.value)}
                        className="environment-selector"
                    >
                        <option value="classroom">Classroom</option>
                        <option value="stage">Stage</option>
                        <option value="studio">Studio</option>
                        <option value="outdoor">Outdoor</option>
                    </select>
                </div>
            </div>
        </div>
    );
};

export default MotionViewer;
