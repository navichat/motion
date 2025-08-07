#!/usr/bin/env node

/**
 * ONNX Model Validation and Debugging Tool
 * Comprehensive analysis of ONNX files for the avatar AI inference system
 */

const fs = require('fs');
const path = require('path');

class ONNXValidator {
    constructor() {
        this.modelPaths = {
            'RSMT': '../../../../RSMT-Realtime-Stylized-Motion-Transition/output/web_viewer/deepphase.onnx',
            'DeepMimic': '../../../deepmimic/data/policies_onnx/compatible_humanoid3d_humanoid3d_walk.onnx',
            'Audio2Gesture': '../../../audio2gesture_step_fixed.onnx',
            'FaceFormer': '../../../engine/web_porting_poc/faceformer/faceformer_core_step.onnx'
        };
    }

    validateONNXHeader(filePath) {
        try {
            const buffer = fs.readFileSync(filePath);
            
            // ONNX files should start with protobuf magic bytes
            const magicBytes = buffer.slice(0, 4);
            const header = buffer.slice(0, 20);
            
            console.log(`📄 File: ${path.basename(filePath)}`);
            console.log(`📏 Size: ${buffer.length} bytes (${(buffer.length / 1024 / 1024).toFixed(2)} MB)`);
            console.log(`🔢 Magic bytes: ${Array.from(magicBytes).map(b => '0x' + b.toString(16).padStart(2, '0')).join(' ')}`);
            console.log(`📋 Header hex: ${Array.from(header).map(b => b.toString(16).padStart(2, '0')).join(' ')}`);
            console.log(`📋 Header ASCII: ${Array.from(header).map(b => b >= 32 && b <= 126 ? String.fromCharCode(b) : '.').join('')}`);
            
            // Check for common ONNX patterns
            const headerStr = header.toString('ascii', 0, 20);
            let validity = 'UNKNOWN';
            let issues = [];
            
            // ONNX files typically start with version info
            if (headerStr.includes('pytorch') || headerStr.includes('ONNX')) {
                validity = 'VALID_ONNX';
            } else if (buffer[0] === 0x08) {
                // Protobuf files often start with 0x08 (field 1, varint)
                validity = 'LIKELY_PROTOBUF';
            } else if (buffer[0] === 0x00 && buffer[1] === 0x00) {
                validity = 'POSSIBLE_CORRUPTION';
                issues.push('File starts with null bytes');
            }
            
            // Check for wire type 4 issue (the specific error we're seeing)
            for (let i = 0; i < Math.min(100, buffer.length - 1); i++) {
                const byte = buffer[i];
                const wireType = byte & 0x07;
                if (wireType === 4) {
                    issues.push(`Wire type 4 found at offset ${i} (0x${i.toString(16)}) - this causes parsing errors`);
                }
            }
            
            return {
                filePath,
                size: buffer.length,
                validity,
                issues,
                isAccessible: true
            };
        } catch (error) {
            return {
                filePath,
                size: 0,
                validity: 'ERROR',
                issues: [error.message],
                isAccessible: false
            };
        }
    }

    async validateAllModels() {
        console.log('🔍 ONNX Model Validation Report');
        console.log('================================\n');
        
        const results = [];
        const workingDir = '/home/barberb/motion/dev/web_viewer/js/workers';
        
        for (const [modelName, relativePath] of Object.entries(this.modelPaths)) {
            console.log(`🤖 Validating ${modelName}:`);
            
            const fullPath = path.resolve(workingDir, relativePath);
            const result = this.validateONNXHeader(fullPath);
            result.modelName = modelName;
            result.relativePath = relativePath;
            result.fullPath = fullPath;
            
            results.push(result);
            
            // Color coding for terminal output
            const statusColor = result.validity === 'VALID_ONNX' ? '✅' : 
                               result.validity === 'LIKELY_PROTOBUF' ? '⚠️' : '❌';
            
            console.log(`   ${statusColor} Status: ${result.validity}`);
            console.log(`   📍 Path: ${result.fullPath}`);
            console.log(`   🔗 Relative: ${result.relativePath}`);
            
            if (!result.isAccessible) {
                console.log(`   ❌ File not accessible`);
            }
            
            if (result.issues.length > 0) {
                console.log(`   ⚠️  Issues found:`);
                result.issues.forEach(issue => {
                    console.log(`      • ${issue}`);
                });
            }
            console.log('');
        }
        
        // Summary
        console.log('📊 SUMMARY:');
        const validFiles = results.filter(r => r.validity === 'VALID_ONNX').length;
        const accessibleFiles = results.filter(r => r.isAccessible).length;
        const filesWithIssues = results.filter(r => r.issues.length > 0).length;
        
        console.log(`   • Total models checked: ${results.length}`);
        console.log(`   • Accessible files: ${accessibleFiles}/${results.length}`);
        console.log(`   • Valid ONNX files: ${validFiles}/${results.length}`);
        console.log(`   • Files with issues: ${filesWithIssues}/${results.length}`);
        
        // Recommendations
        console.log('\n💡 RECOMMENDATIONS:');
        
        const corruptedFiles = results.filter(r => r.validity === 'POSSIBLE_CORRUPTION');
        if (corruptedFiles.length > 0) {
            console.log(`   🔧 Re-export these models: ${corruptedFiles.map(f => f.modelName).join(', ')}`);
        }
        
        const inaccessibleFiles = results.filter(r => !r.isAccessible);
        if (inaccessibleFiles.length > 0) {
            console.log(`   📁 Fix file paths for: ${inaccessibleFiles.map(f => f.modelName).join(', ')}`);
        }
        
        const wireType4Issues = results.filter(r => r.issues.some(i => i.includes('Wire type 4')));
        if (wireType4Issues.length > 0) {
            console.log(`   🔧 Wire type 4 issues in: ${wireType4Issues.map(f => f.modelName).join(', ')}`);
            console.log(`      These files may need to be re-exported with newer ONNX tools`);
        }
        
        return results;
    }

    generateFixScript(results) {
        const script = `#!/bin/bash
# Auto-generated ONNX model fix script

echo "🔧 ONNX Model Fix Script"
echo "======================="

`;
        
        const inaccessibleFiles = results.filter(r => !r.isAccessible);
        const wireType4Files = results.filter(r => r.issues.some(i => i.includes('Wire type 4')));
        
        if (inaccessibleFiles.length > 0) {
            script += `
# Fix missing model files
echo "📁 Checking for alternative model locations..."
`;
            inaccessibleFiles.forEach(file => {
                script += `
find /home/barberb/motion -name "${path.basename(file.fullPath)}" -type f 2>/dev/null | head -1 | while read found_file; do
    if [ -n "$found_file" ]; then
        echo "Found ${file.modelName} at: $found_file"
        mkdir -p "$(dirname "${file.fullPath}")"
        ln -sf "$found_file" "${file.fullPath}"
        echo "✅ Created symlink for ${file.modelName}"
    fi
done
`;
            });
        }
        
        if (wireType4Files.length > 0) {
            script += `
# Check ONNX file integrity
echo "🔍 Checking ONNX file integrity..."
`;
            wireType4Files.forEach(file => {
                script += `
if command -v python3 &> /dev/null; then
    python3 -c "
import onnx
try:
    model = onnx.load('${file.fullPath}')
    print('✅ ${file.modelName}: ONNX file is valid')
except Exception as e:
    print('❌ ${file.modelName}: ONNX validation failed - ' + str(e))
    print('   Consider re-exporting this model')
"
fi
`;
            });
        }
        
        return script + `
echo "🏁 Fix script completed"
`;
    }
}

// Run the validator
async function main() {
    const validator = new ONNXValidator();
    const results = await validator.validateAllModels();
    
    // Generate fix script
    const fixScript = validator.generateFixScript(results);
    fs.writeFileSync('/home/barberb/motion/dev/web_viewer/fix-onnx-models.sh', fixScript);
    console.log('\n📜 Fix script generated at: fix-onnx-models.sh');
    console.log('   Run: chmod +x fix-onnx-models.sh && ./fix-onnx-models.sh');
}

if (require.main === module) {
    main().catch(console.error);
}

module.exports = ONNXValidator;
