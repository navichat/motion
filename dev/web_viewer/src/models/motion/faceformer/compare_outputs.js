// JavaScript Model Output Comparison Script
// Runs the JS FaceFormer model and compares with Python outputs

class FaceFormerComparison {
    constructor() {
        this.pythonResults = null;
        this.jsResults = null;
    }

    async loadPythonResults() {
        console.log('📥 Loading Python model results...');
        try {
            const response = await fetch('./python_model_outputs.json');
            this.pythonResults = await response.json();
            console.log('✅ Python results loaded successfully');
            return true;
        } catch (error) {
            console.error('❌ Failed to load Python results:', error);
            return false;
        }
    }

    async runJavaScriptModel() {
        console.log('🟨 Running JavaScript FaceFormer model...');
        
        try {
            // Initialize the JS model
            const generator = new FaceFormerWebGeneratorFixed();
            await generator.initialize();
            
            // Load test data (same as Python)
            const testDataResponse = await fetch('./faceformer_minimal_test_data.json');
            const testData = await testDataResponse.json();
            
            const inputs = testData.inputs;
            
            // Extract inputs
            const audioFeatures = inputs.audio_features[0][0];  // Flatten to 1D array
            const template = inputs.template[0][0];             // Flatten to 1D array  
            const oneHot = inputs.one_hot[0];                   // Already 1D
            
            console.log('📊 JS Input data:');
            console.log(`  Audio features length: ${audioFeatures.length}`);
            console.log(`  Template length: ${template.length}`);
            console.log(`  One-hot length: ${oneHot.length}`);
            
            // Run the model for a single step
            const results = await this.runSingleInferenceStep(generator, audioFeatures, template, oneHot);
            
            this.jsResults = {
                outputs: results,
                shapes: {
                    new_vertice_out: [1, 1, results.new_vertice_out.length],
                    updated_vertice_emb: [1, 1, results.updated_vertice_emb.length]
                },
                model_type: "JavaScript FaceFormer Minimal"
            };
            
            console.log('✅ JavaScript model completed');
            console.log('📐 JS Output shapes:');
            Object.entries(this.jsResults.shapes).forEach(([key, shape]) => {
                console.log(`  ${key}: [${shape.join(', ')}]`);
            });
            
            return true;
            
        } catch (error) {
            console.error('❌ JavaScript model failed:', error);
            return false;
        }
    }

    async runSingleInferenceStep(generator, audioFeatures, template, oneHot) {
        // Run a single inference step using the minimal model
        console.log('🔄 Running single inference step...');
        
        try {
            // Create tensors for ONNX model
            const batchSize = 1;
            
            // Prepare inputs in correct shapes
            const audioTensor = new ort.Tensor('float32', new Float32Array(audioFeatures), [batchSize, 1, 768]);
            const templateTensor = new ort.Tensor('float32', new Float32Array(template), [batchSize, 1, 15069]);
            const oneHotTensor = new ort.Tensor('float32', new Float32Array(oneHot), [batchSize, 3]);
            
            // Initialize vertex embedding with small values
            const initialVerticeEmb = new Array(64).fill(0.01);
            const verticeEmbTensor = new ort.Tensor('float32', new Float32Array(initialVerticeEmb), [batchSize, 1, 64]);
            
            // Prepare model inputs
            const feeds = {
                audio_features: audioTensor,
                vertice_emb: verticeEmbTensor,
                one_hot: oneHotTensor,
                template: templateTensor
            };
            
            console.log('📋 Running ONNX inference...');
            const modelResults = await generator.session.run(feeds);
            
            // Extract results
            const newVerticeOut = Array.from(modelResults.new_vertice_out.data);
            const updatedVerticeEmb = Array.from(modelResults.updated_vertice_emb.data);
            
            console.log(`✅ Single step completed - Generated ${newVerticeOut.length} vertices`);
            
            return {
                new_vertice_out: newVerticeOut,
                updated_vertice_emb: updatedVerticeEmb,
                raw_outputs: {
                    new_vertice_out_tensor: modelResults.new_vertice_out,
                    updated_vertice_emb_tensor: modelResults.updated_vertice_emb
                }
            };
            
        } catch (error) {
            console.error('❌ Single inference step failed:', error);
            throw error;
        }
    }

    compareOutputs() {
        if (!this.pythonResults || !this.jsResults) {
            console.error('❌ Missing results for comparison');
            return null;
        }

        console.log('🔍 Comparing Python vs JavaScript outputs...');
        
        // Get outputs to compare
        const pythonOutputs = this.pythonResults.simplified_results?.simplified_python_outputs;
        const jsOutputs = this.jsResults.outputs;
        
        if (!pythonOutputs || !jsOutputs) {
            console.error('❌ Missing output data for comparison');
            return null;
        }

        // Compare shapes
        const shapeComparison = this.compareShapes();
        
        // Compare values
        const valueComparison = this.compareValues(pythonOutputs, jsOutputs);
        
        // Statistical analysis
        const stats = this.computeStatistics(pythonOutputs, jsOutputs);
        
        const comparison = {
            timestamp: new Date().toISOString(),
            shapes: shapeComparison,
            values: valueComparison,
            statistics: stats,
            summary: this.generateSummary(shapeComparison, valueComparison, stats)
        };
        
        this.displayComparison(comparison);
        return comparison;
    }

    compareShapes() {
        const pythonShapes = this.pythonResults.simplified_results?.shapes;
        const jsShapes = this.jsResults.shapes;
        
        const comparison = {};
        
        // Compare new_vertice_out shapes
        if (pythonShapes?.new_vertices && jsShapes?.new_vertice_out) {
            comparison.new_vertice_out = {
                python: pythonShapes.new_vertices,
                js: jsShapes.new_vertice_out,
                match: JSON.stringify(pythonShapes.new_vertices) === JSON.stringify(jsShapes.new_vertice_out)
            };
        }
        
        // Compare updated_vertice_emb shapes  
        if (pythonShapes?.updated_emb && jsShapes?.updated_vertice_emb) {
            comparison.updated_vertice_emb = {
                python: pythonShapes.updated_emb,
                js: jsShapes.updated_vertice_emb,
                match: JSON.stringify(pythonShapes.updated_emb) === JSON.stringify(jsShapes.updated_vertice_emb)
            };
        }
        
        return comparison;
    }

    compareValues(pythonOutputs, jsOutputs) {
        const comparison = {};
        
        // Compare new_vertice_out values
        if (pythonOutputs.new_vertice_out && jsOutputs.new_vertice_out) {
            comparison.new_vertice_out = this.compareArrays(
                pythonOutputs.new_vertice_out, 
                jsOutputs.new_vertice_out,
                'New Vertices'
            );
        }
        
        // Compare updated_vertice_emb values
        if (pythonOutputs.updated_vertice_emb && jsOutputs.updated_vertice_emb) {
            comparison.updated_vertice_emb = this.compareArrays(
                pythonOutputs.updated_vertice_emb,
                jsOutputs.updated_vertice_emb, 
                'Updated Embedding'
            );
        }
        
        return comparison;
    }

    compareArrays(array1, array2, name) {
        if (array1.length !== array2.length) {
            return {
                error: `Length mismatch: ${array1.length} vs ${array2.length}`,
                name: name
            };
        }
        
        const diffs = [];
        let sumSquaredDiff = 0;
        let maxDiff = 0;
        let totalValues = array1.length;
        
        for (let i = 0; i < array1.length; i++) {
            const diff = Math.abs(array1[i] - array2[i]);
            diffs.push(diff);
            sumSquaredDiff += diff * diff;
            maxDiff = Math.max(maxDiff, diff);
        }
        
        const meanSquaredError = sumSquaredDiff / totalValues;
        const rootMeanSquaredError = Math.sqrt(meanSquaredError);
        const meanAbsoluteError = diffs.reduce((a, b) => a + b, 0) / totalValues;
        
        return {
            name: name,
            length: totalValues,
            max_difference: maxDiff,
            mean_absolute_error: meanAbsoluteError,
            root_mean_squared_error: rootMeanSquaredError,
            first_10_diffs: diffs.slice(0, 10),
            is_close: maxDiff < 1e-3 && meanAbsoluteError < 1e-4
        };
    }

    computeStatistics(pythonOutputs, jsOutputs) {
        const stats = {};
        
        // Compute statistics for each output
        ['new_vertice_out', 'updated_vertice_emb'].forEach(key => {
            if (pythonOutputs[key] && jsOutputs[key]) {
                const pythonArray = pythonOutputs[key];
                const jsArray = jsOutputs[key];
                
                stats[key] = {
                    python: this.arrayStats(pythonArray),
                    js: this.arrayStats(jsArray)
                };
            }
        });
        
        return stats;
    }

    arrayStats(array) {
        const sum = array.reduce((a, b) => a + b, 0);
        const mean = sum / array.length;
        const variance = array.reduce((a, b) => a + (b - mean) ** 2, 0) / array.length;
        const std = Math.sqrt(variance);
        const min = Math.min(...array);
        const max = Math.max(...array);
        
        return {
            mean: mean,
            std: std,
            min: min,
            max: max,
            sum: sum
        };
    }

    generateSummary(shapeComparison, valueComparison, stats) {
        const summary = {
            shapes_match: true,
            values_close: true,
            issues: []
        };
        
        // Check shape matches
        Object.entries(shapeComparison).forEach(([key, comp]) => {
            if (!comp.match) {
                summary.shapes_match = false;
                summary.issues.push(`Shape mismatch in ${key}: Python ${JSON.stringify(comp.python)} vs JS ${JSON.stringify(comp.js)}`);
            }
        });
        
        // Check value closeness
        Object.entries(valueComparison).forEach(([key, comp]) => {
            if (comp.error) {
                summary.values_close = false;
                summary.issues.push(`${key}: ${comp.error}`);
            } else if (!comp.is_close) {
                summary.values_close = false;
                summary.issues.push(`${key}: Values not close enough (max_diff=${comp.max_difference.toFixed(6)}, mae=${comp.mean_absolute_error.toFixed(6)})`);
            }
        });
        
        // Overall assessment
        if (summary.shapes_match && summary.values_close) {
            summary.overall = "✅ MODELS MATCH - Outputs are consistent!";
        } else if (summary.shapes_match) {
            summary.overall = "⚠️ PARTIAL MATCH - Shapes match but values differ";
        } else {
            summary.overall = "❌ MISMATCH - Significant differences detected";
        }
        
        return summary;
    }

    displayComparison(comparison) {
        console.log('\n' + '='.repeat(60));
        console.log('🔍 FACEFORMER MODEL COMPARISON RESULTS');
        console.log('='.repeat(60));
        
        console.log('\n📐 Shape Comparison:');
        Object.entries(comparison.shapes).forEach(([key, comp]) => {
            const status = comp.match ? '✅' : '❌';
            console.log(`  ${status} ${key}:`);
            console.log(`    Python: [${comp.python.join(', ')}]`);
            console.log(`    JS:     [${comp.js.join(', ')}]`);
        });
        
        console.log('\n📊 Value Comparison:');
        Object.entries(comparison.values).forEach(([key, comp]) => {
            if (comp.error) {
                console.log(`  ❌ ${key}: ${comp.error}`);
            } else {
                const status = comp.is_close ? '✅' : '⚠️';
                console.log(`  ${status} ${comp.name}:`);
                console.log(`    Max Difference: ${comp.max_difference.toFixed(6)}`);
                console.log(`    Mean Absolute Error: ${comp.mean_absolute_error.toFixed(6)}`);
                console.log(`    Root Mean Squared Error: ${comp.root_mean_squared_error.toFixed(6)}`);
            }
        });
        
        console.log('\n📈 Statistics Summary:');
        Object.entries(comparison.statistics).forEach(([key, stats]) => {
            console.log(`  ${key}:`);
            console.log(`    Python - Mean: ${stats.python.mean.toFixed(4)}, Std: ${stats.python.std.toFixed(4)}, Range: [${stats.python.min.toFixed(4)}, ${stats.python.max.toFixed(4)}]`);
            console.log(`    JS     - Mean: ${stats.js.mean.toFixed(4)}, Std: ${stats.js.std.toFixed(4)}, Range: [${stats.js.min.toFixed(4)}, ${stats.js.max.toFixed(4)}]`);
        });
        
        console.log('\n🎯 Summary:');
        console.log(`  ${comparison.summary.overall}`);
        if (comparison.summary.issues.length > 0) {
            console.log('\n⚠️ Issues Found:');
            comparison.summary.issues.forEach((issue, i) => {
                console.log(`  ${i + 1}. ${issue}`);
            });
        }
        
        console.log('\n' + '='.repeat(60));
    }

    async saveComparisonResults(comparison) {
        const data = {
            comparison: comparison,
            python_results: this.pythonResults,
            js_results: this.jsResults,
            timestamp: new Date().toISOString()
        };
        
        // For browser environment, we'll log the data
        // In a real environment, you'd save this to a file
        console.log('💾 Full comparison data:', data);
        
        // Also create a downloadable JSON
        const blob = new Blob([JSON.stringify(data, null, 2)], {type: 'application/json'});
        const url = URL.createObjectURL(blob);
        
        const a = document.createElement('a');
        a.href = url;
        a.download = 'faceformer_comparison_results.json';
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
        
        console.log('💾 Comparison results downloaded as JSON file');
    }

    async runFullComparison() {
        console.log('🚀 Starting Full FaceFormer Model Comparison...');
        
        // Step 1: Load Python results
        const pythonLoaded = await this.loadPythonResults();
        if (!pythonLoaded) {
            console.error('❌ Cannot proceed without Python results');
            return false;
        }
        
        // Step 2: Run JavaScript model
        const jsRan = await this.runJavaScriptModel();
        if (!jsRan) {
            console.error('❌ Cannot proceed without JavaScript results');
            return false;
        }
        
        // Step 3: Compare outputs
        const comparison = this.compareOutputs();
        if (!comparison) {
            console.error('❌ Comparison failed');
            return false;
        }
        
        // Step 4: Save results
        await this.saveComparisonResults(comparison);
        
        console.log('✅ Full comparison completed successfully!');
        return true;
    }
}

// Export for use
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { FaceFormerComparison };
}

// Make available globally for web use
if (typeof window !== 'undefined') {
    window.FaceFormerComparison = FaceFormerComparison;
}
