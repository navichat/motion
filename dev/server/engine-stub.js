/**
 * Engine stub for loading resources and animations
 * Provides compatibility with the original @navi/engine while using local assets
 */

import fs from 'fs/promises';
import path from 'path';

export async function loadResources({ ctx }) {
    console.log('Loading resources from', ctx.paths.resources);
    
    try {
        // Load avatars
        const avatarsPath = path.join(ctx.paths.resources, 'avatars');
        const avatarFiles = await fs.readdir(avatarsPath).catch(() => []);
        
        ctx.resources = ctx.resources || {};
        ctx.resources.avatars = {};
        
        for (const file of avatarFiles) {
            if (file.endsWith('.json')) {
                const avatarData = JSON.parse(
                    await fs.readFile(path.join(avatarsPath, file), 'utf-8')
                );
                const avatarId = path.basename(file, '.json');
                ctx.resources.avatars[avatarId] = avatarData;
            }
        }
        
        // Load scenes
        const scenesPath = path.join(ctx.paths.resources, 'scenes');
        const sceneFiles = await fs.readdir(scenesPath).catch(() => []);
        
        ctx.resources.scenes = {};
        
        for (const file of sceneFiles) {
            if (file.endsWith('.json')) {
                const sceneData = JSON.parse(
                    await fs.readFile(path.join(scenesPath, file), 'utf-8')
                );
                const sceneId = path.basename(file, '.json');
                ctx.resources.scenes[sceneId] = sceneData;
            }
        }
        
        console.log(`Loaded ${Object.keys(ctx.resources.avatars).length} avatars and ${Object.keys(ctx.resources.scenes).length} scenes`);
        
    } catch (error) {
        console.error('Error loading resources:', error);
        ctx.resources = { avatars: {}, scenes: {} };
    }
}

export async function loadAnimations({ ctx }) {
    console.log('Loading animations from', ctx.paths.resources);
    
    try {
        const animationsPath = path.join(ctx.paths.resources, 'animations');
        const animationFiles = await fs.readdir(animationsPath).catch(() => []);
        
        ctx.animations = {};
        
        for (const file of animationFiles) {
            if (file.endsWith('.json') && file !== 'manifest.json') {
                const animationData = JSON.parse(
                    await fs.readFile(path.join(animationsPath, file), 'utf-8')
                );
                const animationId = path.basename(file, '.json');
                ctx.animations[animationId] = animationData;
            }
        }
        
        // Load manifest if it exists
        try {
            const manifestPath = path.join(animationsPath, 'manifest.json');
            const manifest = JSON.parse(await fs.readFile(manifestPath, 'utf-8'));
            ctx.animationManifest = manifest;
        } catch (error) {
            console.log('No animation manifest found, continuing without it');
        }
        
        console.log(`Loaded ${Object.keys(ctx.animations).length} animations`);
        
    } catch (error) {
        console.error('Error loading animations:', error);
        ctx.animations = {};
    }
}

export async function loadPlayerClasses() {
    // Return the player classes from the viewer module
    const { Player, AvatarInstance, FreeCameraDriver } = await import('../viewer/src/Player.js');
    
    return {
        Player,
        AvatarInstance, 
        FreeCameraDriver
    };
}
