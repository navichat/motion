/**
 * Real Job Factory - Creates diverse computational jobs for realistic queue testing
 */

class RealJobFactory {
    constructor() {
        this.jobTypes = [
            // WASM CPU Jobs
            'WASMMatrix', 'WASMPrime', 'WASMFractal',
            // WebGPU Jobs  
            'WebGPUMatrix', 'WebGPUImage', 'WebGPUParticle',
            // WebNN Jobs
            'WebNNImageClassification', 'WebNNTextProcessing', 'WebNNAudioProcessing'
        ];
        
        this.jobCounter = 0;
    }

    createRealisticWorkload(jobCount = 50) {
        const jobs = [];
        
        for (let i = 0; i < jobCount; i++) {
            const job = this.createRandomJob();
            jobs.push({
                job: job,
                priority: this.generateRealisticPriority(job.type),
                scheduledTime: this.generateScheduledTime(),
                options: {
                    maxRetries: 2,
                    timeout: job.duration * 2
                }
            });
        }
        
        return jobs;
    }

    createRandomJob() {
        const jobType = this.jobTypes[Math.floor(Math.random() * this.jobTypes.length)];
        const complexity = Math.floor(Math.random() * 3) + 1; // 1-3
        const id = `job_${Date.now()}_${this.jobCounter++}`;
        
        switch (jobType) {
            case 'WASMMatrix':
                return new WASMMatrixJob(id, 
                    128 + Math.random() * 256, // Size 128-384
                    complexity);
                    
            case 'WASMPrime':
                return new WASMPrimeJob(id,
                    50000 + Math.random() * 100000, // Limit 50k-150k
                    complexity);
                    
            case 'WASMFractal':
                return new WASMFractalJob(id,
                    256 + Math.random() * 256, // Size 256-512
                    50 + Math.random() * 100, // Iterations 50-150
                    complexity);
                    
            case 'WebGPUMatrix':
                return new WebGPUMatrixJob(id,
                    256 + Math.random() * 512, // Size 256-768
                    complexity);
                    
            case 'WebGPUImage':
                return new WebGPUImageJob(id,
                    512 + Math.random() * 512, // Width 512-1024
                    512 + Math.random() * 512, // Height 512-1024
                    complexity);
                    
            case 'WebGPUParticle':
                return new WebGPUParticleJob(id,
                    10000 + Math.random() * 40000, // Particles 10k-50k
                    50 + Math.random() * 100, // Steps 50-150
                    complexity);
                    
            case 'WebNNImageClassification':
                return new WebNNImageClassificationJob(id,
                    8 + Math.random() * 24, // Batch size 8-32
                    224, // Standard ImageNet size
                    complexity);
                    
            case 'WebNNTextProcessing':
                return new WebNNTextProcessingJob(id,
                    256 + Math.random() * 256, // Sequence length 256-512
                    4 + Math.random() * 12, // Batch size 4-16
                    complexity);
                    
            case 'WebNNAudioProcessing':
                return new WebNNAudioProcessingJob(id,
                    8000 + Math.random() * 16000, // Audio length 0.5-1.5s
                    4 + Math.random() * 8, // Batch size 4-12
                    complexity);
                    
            default:
                return new WASMMatrixJob(id, 256, 1);
        }
    }

    generateRealisticPriority(jobType) {
        // Assign realistic priorities based on job types
        const priorityMaps = {
            // Real-time jobs (higher priority = lower number)
            'WebNNAudioProcessing': () => Math.floor(Math.random() * 3), // 0-2 (highest)
            'WebGPUParticle': () => Math.floor(Math.random() * 3), // 0-2 (real-time sim)
            
            // Interactive jobs
            'WebNNImageClassification': () => 2 + Math.floor(Math.random() * 3), // 2-4
            'WebGPUImage': () => 2 + Math.floor(Math.random() * 3), // 2-4
            
            // Batch processing jobs
            'WebNNTextProcessing': () => 4 + Math.floor(Math.random() * 3), // 4-6
            'WebGPUMatrix': () => 4 + Math.floor(Math.random() * 3), // 4-6
            
            // Background computation jobs
            'WASMMatrix': () => 6 + Math.floor(Math.random() * 3), // 6-8
            'WASMPrime': () => 7 + Math.floor(Math.random() * 3), // 7-9
            'WASMFractal': () => 7 + Math.floor(Math.random() * 3), // 7-9
        };
        
        const priorityFn = priorityMaps[jobType] || (() => Math.floor(Math.random() * 10));
        return priorityFn();
    }

    generateScheduledTime() {
        // Some jobs are immediate, others are scheduled for future
        const now = Date.now();
        const delay = Math.random();
        
        if (delay < 0.7) {
            return null; // Immediate execution (70%)
        } else if (delay < 0.9) {
            return now + Math.random() * 5000; // 0-5 seconds delay (20%)
        } else {
            return now + 5000 + Math.random() * 10000; // 5-15 seconds delay (10%)
        }
    }

    createStressTestWorkload(intensity = 'medium') {
        const intensitySettings = {
            light: { jobCount: 20, maxComplexity: 1 },
            medium: { jobCount: 50, maxComplexity: 2 },
            heavy: { jobCount: 100, maxComplexity: 3 },
            extreme: { jobCount: 200, maxComplexity: 3 }
        };
        
        const settings = intensitySettings[intensity] || intensitySettings.medium;
        const jobs = [];
        
        for (let i = 0; i < settings.jobCount; i++) {
            const job = this.createRandomJob();
            // Override complexity for stress test
            if (job.complexity !== undefined) {
                job.complexity = Math.min(job.complexity, settings.maxComplexity);
            }
            
            jobs.push({
                job: job,
                priority: this.generateRealisticPriority(job.type),
                scheduledTime: this.generateScheduledTime(),
                options: {
                    maxRetries: 1, // Fewer retries for stress test
                    timeout: job.duration * 1.5
                }
            });
        }
        
        return jobs;
    }

    createMLPipelineWorkload() {
        // Create a realistic ML pipeline workload
        const jobs = [];
        
        // Data preprocessing jobs (high priority)
        for (let i = 0; i < 5; i++) {
            jobs.push({
                job: new WebNNImageClassificationJob(`preprocess_${i}`, 16, 224, 1),
                priority: 1,
                scheduledTime: null
            });
        }
        
        // Feature extraction (medium priority)
        for (let i = 0; i < 8; i++) {
            jobs.push({
                job: new WebNNTextProcessingJob(`feature_${i}`, 512, 8, 2),
                priority: 3,
                scheduledTime: null
            });
        }
        
        // Model training/inference (mixed priority)
        for (let i = 0; i < 10; i++) {
            const isTraining = i < 3;
            jobs.push({
                job: new WebGPUMatrixJob(`model_${i}`, 512, isTraining ? 3 : 1),
                priority: isTraining ? 2 : 5,
                scheduledTime: null
            });
        }
        
        // Audio processing (real-time, highest priority)
        for (let i = 0; i < 6; i++) {
            jobs.push({
                job: new WebNNAudioProcessingJob(`audio_${i}`, 16000, 4, 1),
                priority: 0,
                scheduledTime: i * 2000 // Staggered every 2 seconds
            });
        }
        
        // Background computation (lowest priority)
        for (let i = 0; i < 15; i++) {
            const jobTypes = [WASMPrimeJob, WASMFractalJob, WASMMatrixJob];
            const JobClass = jobTypes[i % jobTypes.length];
            jobs.push({
                job: new JobClass(`background_${i}`, 100000, 2),
                priority: 8,
                scheduledTime: null
            });
        }
        
        return jobs;
    }
}

// Export factory
window.RealJobFactory = RealJobFactory;

// Convenience function for testing
window.createRealisticWorkload = function(intensity = 'medium') {
    const factory = new RealJobFactory();
    return factory.createStressTestWorkload(intensity);
};

window.createMLPipelineWorkload = function() {
    const factory = new RealJobFactory();
    return factory.createMLPipelineWorkload();
};
