/**
 * Asset verification and setup script
 * Ensures all components can access the shared assets properly
 */

import fs from 'fs/promises';
import path from 'path';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));

async function verifyAssets() {
    console.log('🔍 Verifying asset structure...');
    
    const assetPaths = [
        'assets/animations',
        'assets/avatars', 
        'assets/scenes'
    ];
    
    for (const assetPath of assetPaths) {
        const fullPath = path.join(__dirname, assetPath);
        try {
            const stats = await fs.stat(fullPath);
            if (stats.isDirectory()) {
                const files = await fs.readdir(fullPath);
                console.log(`✅ ${assetPath}: ${files.length} files`);
            }
        } catch (error) {
            console.log(`❌ ${assetPath}: Not found or not accessible`);
        }
    }
}

async function createSymlinks() {
    console.log('🔗 Creating asset symlinks...');
    
    const symlinks = [
        { source: '../assets', target: 'webapp/assets' },
        { source: '../assets', target: 'viewer/assets' }
    ];
    
    for (const { source, target } of symlinks) {
        const targetPath = path.join(__dirname, target);
        const sourcePath = path.join(__dirname, source);
        
        try {
            // Remove existing symlink or directory
            try {
                await fs.unlink(targetPath);
            } catch (error) {
                // File doesn't exist, that's fine
            }
            
            // Create new symlink
            await fs.symlink(sourcePath, targetPath);
            console.log(`✅ Created symlink: ${target} -> ${source}`);
        } catch (error) {
            console.log(`❌ Failed to create symlink ${target}: ${error.message}`);
        }
    }
}

async function verifyDependencies() {
    console.log('📦 Verifying package dependencies...');
    
    const packages = ['psyche', 'webapp', 'viewer'];
    
    for (const pkg of packages) {
        const packageJsonPath = path.join(__dirname, pkg, 'package.json');
        const nodeModulesPath = path.join(__dirname, pkg, 'node_modules');
        
        try {
            await fs.access(packageJsonPath);
            await fs.access(nodeModulesPath);
            console.log(`✅ ${pkg}: Dependencies installed`);
        } catch (error) {
            console.log(`❌ ${pkg}: Missing dependencies`);
        }
    }
}

async function generateAssetManifest() {
    console.log('📋 Generating asset manifest...');
    
    const manifest = {
        generated: new Date().toISOString(),
        animations: {},
        avatars: {},
        scenes: {}
    };
    
    // Scan animations
    const animationsPath = path.join(__dirname, 'assets/animations');
    try {
        const animationFiles = await fs.readdir(animationsPath);
        for (const file of animationFiles) {
            if (file.endsWith('.json') && file !== 'manifest.json') {
                try {
                    const data = JSON.parse(
                        await fs.readFile(path.join(animationsPath, file), 'utf-8')
                    );
                    const id = path.basename(file, '.json');
                    manifest.animations[id] = {
                        file: file,
                        name: data.name || id,
                        duration: data.duration || null
                    };
                } catch (error) {
                    console.log(`⚠️  Failed to parse animation ${file}`);
                }
            }
        }
    } catch (error) {
        console.log('❌ Failed to scan animations directory');
    }
    
    // Scan avatars
    const avatarsPath = path.join(__dirname, 'assets/avatars');
    try {
        const avatarFiles = await fs.readdir(avatarsPath);
        for (const file of avatarFiles) {
            if (file.endsWith('.json')) {
                try {
                    const data = JSON.parse(
                        await fs.readFile(path.join(avatarsPath, file), 'utf-8')
                    );
                    const id = path.basename(file, '.json');
                    manifest.avatars[id] = {
                        file: file,
                        name: data.name || id,
                        model: data.model || null
                    };
                } catch (error) {
                    console.log(`⚠️  Failed to parse avatar ${file}`);
                }
            }
        }
    } catch (error) {
        console.log('❌ Failed to scan avatars directory');
    }
    
    // Scan scenes
    const scenesPath = path.join(__dirname, 'assets/scenes');
    try {
        const sceneFiles = await fs.readdir(scenesPath);
        for (const file of sceneFiles) {
            if (file.endsWith('.json')) {
                try {
                    const data = JSON.parse(
                        await fs.readFile(path.join(scenesPath, file), 'utf-8')
                    );
                    const id = path.basename(file, '.json');
                    manifest.scenes[id] = {
                        file: file,
                        name: data.name || id,
                        environment: data.environment || null
                    };
                } catch (error) {
                    console.log(`⚠️  Failed to parse scene ${file}`);
                }
            }
        }
    } catch (error) {
        console.log('❌ Failed to scan scenes directory');
    }
    
    // Write manifest
    const manifestPath = path.join(__dirname, 'assets/manifest.json');
    await fs.writeFile(manifestPath, JSON.stringify(manifest, null, 2));
    
    console.log(`✅ Generated manifest with ${Object.keys(manifest.animations).length} animations, ${Object.keys(manifest.avatars).length} avatars, ${Object.keys(manifest.scenes).length} scenes`);
}

async function main() {
    console.log('🚀 Motion Platform Asset Setup\n');
    
    await verifyAssets();
    console.log('');
    
    await createSymlinks();
    console.log('');
    
    await verifyDependencies();
    console.log('');
    
    await generateAssetManifest();
    console.log('');
    
    console.log('✅ Asset setup complete!');
}

main().catch(console.error);
