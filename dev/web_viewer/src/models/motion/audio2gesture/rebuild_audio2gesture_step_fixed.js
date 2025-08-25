
const fs = require('fs');
const path = require('path');

/**
 * Rebuild ONNX model from chunks
 */
function rebuildModel() {
    const metadata = JSON.parse(fs.readFileSync('audio2gesture_step_fixed.chunks.json', 'utf8'));
    console.log(`🔧 Rebuilding ${metadata.originalFile} from ${metadata.totalChunks} chunks...`);
    
    const chunks = [];
    
    for (const chunk of metadata.chunks) {
        const chunkData = fs.readFileSync(chunk.filename);
        chunks.push(chunkData);
        console.log(`  ✅ Loaded ${chunk.filename} (${chunk.sizeMB} MB)`);
    }
    
    const rebuiltModel = Buffer.concat(chunks);
    fs.writeFileSync(metadata.originalFile, rebuiltModel);
    
    console.log(`🎉 Successfully rebuilt ${metadata.originalFile} (${metadata.originalSizeMB} MB)`);
    console.log(`📊 Verification: Expected ${metadata.originalSize} bytes, got ${rebuiltModel.length} bytes`);
    
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
