const fs = require('fs');
const path = require('path');

/**
 * Split a large ONNX file into smaller chunks for Git storage
 * @param {string} modelPath - Path to the original ONNX model
 * @param {number} chunkSizeMB - Size of each chunk in MB (default: 90MB)
 */
function splitOnnxModel(modelPath, chunkSizeMB = 90) {
    const chunkSize = chunkSizeMB * 1024 * 1024; // Convert to bytes
    const modelData = fs.readFileSync(modelPath);
    const modelSizeMB = (modelData.length / 1024 / 1024).toFixed(2);
    
    console.log(`📦 Splitting ${path.basename(modelPath)} (${modelSizeMB} MB) into ${chunkSizeMB}MB chunks...`);
    
    const baseName = path.basename(modelPath, '.onnx');
    const dirName = path.dirname(modelPath);
    
    let chunkIndex = 0;
    let offset = 0;
    const chunks = [];
    
    while (offset < modelData.length) {
        const remainingBytes = modelData.length - offset;
        const currentChunkSize = Math.min(chunkSize, remainingBytes);
        
        const chunkFileName = `${baseName}.chunk.${chunkIndex.toString().padStart(3, '0')}`;
        const chunkPath = path.join(dirName, chunkFileName);
        
        const chunkData = modelData.slice(offset, offset + currentChunkSize);
        fs.writeFileSync(chunkPath, chunkData);
        
        chunks.push({
            index: chunkIndex,
            filename: chunkFileName,
            size: currentChunkSize,
            sizeMB: (currentChunkSize / 1024 / 1024).toFixed(2)
        });
        
        console.log(`  ✅ Created ${chunkFileName} (${(currentChunkSize / 1024 / 1024).toFixed(2)} MB)`);
        
        offset += currentChunkSize;
        chunkIndex++;
    }
    
    // Create metadata file
    const metadata = {
        originalFile: path.basename(modelPath),
        originalSize: modelData.length,
        originalSizeMB: modelSizeMB,
        chunks: chunks,
        totalChunks: chunks.length,
        chunkSize: chunkSize,
        createdAt: new Date().toISOString()
    };
    
    const metadataPath = path.join(dirName, `${baseName}.chunks.json`);
    fs.writeFileSync(metadataPath, JSON.stringify(metadata, null, 2));
    
    // Create rebuild script
    const rebuildScript = `
const fs = require('fs');
const path = require('path');

/**
 * Rebuild ONNX model from chunks
 */
function rebuildModel() {
    const metadata = JSON.parse(fs.readFileSync('${baseName}.chunks.json', 'utf8'));
    console.log(\`🔧 Rebuilding \${metadata.originalFile} from \${metadata.totalChunks} chunks...\`);
    
    const chunks = [];
    
    for (const chunk of metadata.chunks) {
        const chunkData = fs.readFileSync(chunk.filename);
        chunks.push(chunkData);
        console.log(\`  ✅ Loaded \${chunk.filename} (\${chunk.sizeMB} MB)\`);
    }
    
    const rebuiltModel = Buffer.concat(chunks);
    fs.writeFileSync(metadata.originalFile, rebuiltModel);
    
    console.log(\`🎉 Successfully rebuilt \${metadata.originalFile} (\${metadata.originalSizeMB} MB)\`);
    console.log(\`📊 Verification: Expected \${metadata.originalSize} bytes, got \${rebuiltModel.length} bytes\`);
    
    if (rebuiltModel.length === metadata.originalSize) {
        console.log('✅ File integrity verified!');
        return true;
    } else {
        console.error('❌ File integrity check failed!');
        return false;
    }
}

if (require.main === module) {
    rebuildModel();
}

module.exports = { rebuildModel };
`;
    
    const rebuildScriptPath = path.join(dirName, `rebuild_${baseName}.js`);
    fs.writeFileSync(rebuildScriptPath, rebuildScript);
    
    console.log(`\n📋 Split Summary:`);
    console.log(`  Original: ${path.basename(modelPath)} (${modelSizeMB} MB)`);
    console.log(`  Chunks: ${chunks.length} files`);
    console.log(`  Metadata: ${baseName}.chunks.json`);
    console.log(`  Rebuild Script: rebuild_${baseName}.js`);
    console.log(`\n🗑️  You can now delete the original ${path.basename(modelPath)} file`);
    console.log(`📤 Commit the chunks, metadata, and rebuild script to Git`);
    console.log(`🔧 To rebuild: node rebuild_${baseName}.js`);
    
    return {
        chunks,
        metadata: metadataPath,
        rebuildScript: rebuildScriptPath
    };
}

if (require.main === module) {
    const modelPath = process.argv[2];
    const chunkSize = parseInt(process.argv[3]) || 90;
    
    if (!modelPath) {
        console.error('Usage: node split_onnx_model.js <model_path> [chunk_size_mb]');
        process.exit(1);
    }
    
    if (!fs.existsSync(modelPath)) {
        console.error(`Error: Model file ${modelPath} not found`);
        process.exit(1);
    }
    
    splitOnnxModel(modelPath, chunkSize);
}

module.exports = { splitOnnxModel };
