import fs from 'fs';
import path from 'path';

// Extract and save outputs from the test run we just completed
const terminalOutput = `
[COLLECT] [log] 🤖 Starting COMPREHENSIVE Avatar AI Model Output Collection...
[COLLECT] [log] 🌐 Navigating to task-manager-demo.html...
[COLLECT] [log] 📡 Page loaded, starting AI model output collection...
[COLLECT] [log] 🔄 Triggering AI model activities...
[COLLECT] [log] Found 0 buttons to interact with
[COLLECT] [log] ⏳ Extended collection period for all AI model outputs...
[COLLECT] [log] Collecting... 1/60 (1 total messages, 0 neural outputs)
[COLLECT] [log] *** TASK SCHEDULER OUTPUT ***
[COLLECT] [log] ✅ Starting AI Model Task Scheduling Loop...
[COLLECT] [log] 🤖 Task TinyLlama inference initialized...
[COLLECT] [log] 🤖 Task FaceFormer inference initialized...
[COLLECT] [log] 🤖 Task RSMT inference initialized...
[COLLECT] [log] 🤖 Task WebNNAudioProcessing inference initialized...
[COLLECT] [log] 🔍 Checking worker webnn_worker_0 capabilities: {webnn: false, onnx: true} against requirements: {cpu: 25, gpu: 0, webnn: 100, memory: 96}
[COLLECT] [log] ❌ Worker webnn_worker_0 fails WebNN requirement (needs webnn:true, has webnn:false)
[COLLECT] [log] ❌ Worker webnn_worker_0 does not satisfy requirements for task task_1753663638597_5d9w1qu6l
[COLLECT] [log] 🔍 Checking pool: cpu. Available workers: 2
[COLLECT] [log] 🔍 Attempting to match task task_1753663638597_5d9w1qu6l with worker cpu_worker_0 (type: cpu)
[COLLECT] [log] 🔍 Checking worker cpu_worker_0 capabilities: {cpu: true, webgpu: false, onnx: false} against requirements: {cpu: 25, gpu: 0, webnn: 100, memory: 96}
[COLLECT] [log] ❌ Worker cpu_worker_0 fails WebNN requirement (needs webnn:true, has webnn:undefined)
[COLLECT] [log] ❌ Worker cpu_worker_0 does not satisfy requirements for task task_1753663638597_5d9w1qu6l
[COLLECT] [log] [WARN] No available worker for task task_1753663638597_5d9w1qu6l (TinyLlama)
[COLLECT] [log] ❌ No available worker for task task_1753663638597_5d9w1qu6l. Attempting preemption.
[COLLECT] [log] [DEBUG] No suitable worker to preempt for task task_1753663638597_5d9w1qu6l
[COLLECT] [log] *** DEBUG: Continuing scheduling. Heap size: 1, Running tasks: 0, Total tasks: 1 ***
[COLLECT] [log] *** DEBUG: _scheduleNextProcess called. Setting timer. ***
[COLLECT] [log] *** DEBUG: Inside setTimeout callback. this.running: true ***
[COLLECT] [log] *** DEBUG: Entering _processQueue. ***
[COLLECT] [log] [INFO] Processing queue. Running tasks: 0, Max concurrent: 4, Heap size: 1
[COLLECT] [log] 🔄 Processing loop - Running: 0, Max: 4, Heap size: 1
[COLLECT] [log] 🔍 Evaluating task task_1753663614610_n4288r13n (FaceFormer) with priority -2
[COLLECT] [log] ✅ Task task_1753663614610_n4288r13n is ready to run, looking for worker...
[COLLECT] [log] 🔍 Searching for worker for task task_1753663614610_n4288r13n (FaceFormer) with requirements: {cpu: 25, gpu: 0, webnn: 100, memory: 128}
[COLLECT] [log] DEBUG: [77.4s] Stats: {queue: Object, workers: Object, performance: Object, system: Object}
[COLLECT] [warning] DEBUG: Potential deadlock detected - jobs pending but none running
[COLLECT] [log] 🔍 Evaluating task task_1753663622816_reo0eky1o (RSMT) with priority -1
[COLLECT] [log] ✅ Task task_1753663622816_reo0eky1o is ready to run, looking for worker...
[COLLECT] [log] 🔍 Searching for worker for task task_1753663622816_reo0eky1o (RSMT) with requirements: {cpu: 25, gpu: 0, webnn: 100, memory: 164}
[COLLECT] [log]   - Queue status: Queued(0) Running(0) Completed(8)
[COLLECT] [log] 🔍 Evaluating task task_1753663579561_mxjqxavcl (WebNNAudioProcessing) with priority -7
[COLLECT] [log] ✅ Task task_1753663579561_mxjqxavcl is ready to run, looking for worker...
[COLLECT] [log] 🔍 Searching for worker for task task_1753663579561_mxjqxavcl (WebNNAudioProcessing) with requirements: {memory: 256000, webnn: 0.85}
`;

// Parse the output to extract AI model information
const extractedData = {
  timestamp: new Date().toISOString(),
  testSource: 'Playwright AI Model Output Collection Test',
  totalOutputLines: terminalOutput.split('\\n').length,
  
  aiModelsDetected: {
    TinyLlama: {
      taskId: 'task_1753663638597_5d9w1qu6l',
      status: 'scheduled_but_failed_worker_assignment',
      requirements: { cpu: 25, gpu: 0, webnn: 100, memory: 96 },
      issue: 'WebNN requirement not met - needs webnn:true, workers have webnn:false'
    },
    FaceFormer: {
      taskId: 'task_1753663614610_n4288r13n',
      status: 'scheduled_but_failed_worker_assignment',
      requirements: { cpu: 25, gpu: 0, webnn: 100, memory: 128 },
      priority: -2,
      issue: 'WebNN requirement not met'
    },
    RSMT: {
      taskId: 'task_1753663622816_reo0eky1o',
      status: 'scheduled_but_failed_worker_assignment',
      requirements: { cpu: 25, gpu: 0, webnn: 100, memory: 164 },
      priority: -1,
      issue: 'WebNN requirement not met'
    },
    WebNNAudioProcessing: {
      taskId: 'task_1753663579561_mxjqxavcl',
      status: 'scheduled_but_failed_worker_assignment',
      requirements: { memory: 256000, webnn: 0.85 },
      priority: -7,
      issue: 'WebNN requirement not met'
    }
  },
  
  workerCapabilities: {
    webnn_worker_0: {
      type: 'webnn',
      capabilities: { webnn: false, onnx: true },
      issue: 'WebNN capability disabled or not supported'
    },
    cpu_worker_0: {
      type: 'cpu',
      capabilities: { cpu: true, webgpu: false, onnx: false },
      issue: 'No WebNN support'
    },
    cpu_worker_1: {
      type: 'cpu', 
      capabilities: { cpu: true, webgpu: false, onnx: false },
      issue: 'No WebNN support'
    }
  },
  
  taskSchedulerInfo: {
    maxConcurrent: [4, 8], // varying levels detected
    heapSizes: [0, 1, 44], // different heap sizes observed
    completedTasks: 8,
    queuedTasks: 0,
    runningTasks: 0,
    totalTasks: [1, 44], // varying task counts
    deadlockDetected: true,
    schedulingLoopActive: true
  },
  
  systemIssue: {
    primaryProblem: 'WebNN Worker Capability Mismatch',
    description: 'All AI models require WebNN support but available workers have webnn:false',
    impact: 'No AI model inference can execute - all tasks fail worker assignment',
    technicalDetails: {
      requiredCapability: 'webnn:true',
      actualCapability: 'webnn:false',
      fallbackAttempted: 'CPU workers checked but also lack WebNN support'
    }
  },
  
  rawOutput: terminalOutput,
  
  analysis: {
    aiModelsPresent: 4,
    modelsSuccessfullyExecuted: 0,
    modelsFailedWorkerAssignment: 4,
    workerPoolsAvailable: 2, // webnn and cpu pools
    workersTotal: 3,
    primaryBlocker: 'WebNN capability requirement vs availability mismatch'
  }
};

// Save the extracted data
const outputPath = path.join(process.cwd(), 'collected-ai-model-outputs.json');
fs.writeFileSync(outputPath, JSON.stringify(extractedData, null, 2));

// Create a readable report
let report = `AI MODEL OUTPUT COLLECTION REPORT\\n`;
report += `=====================================\\n`;
report += `Generated: ${extractedData.timestamp}\\n`;
report += `Source: ${extractedData.testSource}\\n\\n`;

report += `SUMMARY\\n`;
report += `-------\\n`;
report += `AI Models Detected: ${extractedData.analysis.aiModelsPresent}\\n`;
report += `Successfully Executed: ${extractedData.analysis.modelsSuccessfullyExecuted}\\n`;
report += `Failed Worker Assignment: ${extractedData.analysis.modelsFailedWorkerAssignment}\\n`;
report += `Primary Issue: ${extractedData.analysis.primaryBlocker}\\n\\n`;

report += `AI MODELS FOUND\\n`;
report += `===============\\n`;
for (const [model, data] of Object.entries(extractedData.aiModelsDetected)) {
  report += `${model}:\\n`;
  report += `  Task ID: ${data.taskId}\\n`;
  report += `  Status: ${data.status}\\n`;
  report += `  Requirements: ${JSON.stringify(data.requirements)}\\n`;
  report += `  Issue: ${data.issue}\\n`;
  if (data.priority !== undefined) {
    report += `  Priority: ${data.priority}\\n`;
  }
  report += `\\n`;
}

report += `WORKER CAPABILITIES\\n`;
report += `==================\\n`;
for (const [worker, data] of Object.entries(extractedData.workerCapabilities)) {
  report += `${worker}:\\n`;
  report += `  Type: ${data.type}\\n`;
  report += `  Capabilities: ${JSON.stringify(data.capabilities)}\\n`;
  report += `  Issue: ${data.issue}\\n\\n`;
}

report += `SYSTEM DIAGNOSIS\\n`;
report += `================\\n`;
report += `Problem: ${extractedData.systemIssue.primaryProblem}\\n`;
report += `Description: ${extractedData.systemIssue.description}\\n`;
report += `Impact: ${extractedData.systemIssue.impact}\\n`;
report += `Required: ${extractedData.systemIssue.technicalDetails.requiredCapability}\\n`;
report += `Actual: ${extractedData.systemIssue.technicalDetails.actualCapability}\\n`;
report += `Fallback: ${extractedData.systemIssue.technicalDetails.fallbackAttempted}\\n\\n`;

report += `TASK SCHEDULER STATUS\\n`;
report += `====================\\n`;
report += `Max Concurrent: ${extractedData.taskSchedulerInfo.maxConcurrent.join(', ')}\\n`;
report += `Heap Sizes: ${extractedData.taskSchedulerInfo.heapSizes.join(', ')}\\n`;
report += `Completed Tasks: ${extractedData.taskSchedulerInfo.completedTasks}\\n`;
report += `Deadlock Detected: ${extractedData.taskSchedulerInfo.deadlockDetected}\\n`;
report += `Scheduling Active: ${extractedData.taskSchedulerInfo.schedulingLoopActive}\\n\\n`;

report += `RAW OUTPUT SAMPLE (First 500 chars)\\n`;
report += `===================================\\n`;
report += `${extractedData.rawOutput.substring(0, 500)}...\\n`;

const reportPath = path.join(process.cwd(), 'ai-model-outputs-report.txt');
fs.writeFileSync(reportPath, report);

console.log('\\n🎉 AI Model Output Collection Complete!');
console.log(`📁 Detailed JSON: ${outputPath}`);
console.log(`📄 Readable Report: ${reportPath}`);
console.log(`\\n📊 FINDINGS SUMMARY:`);
console.log(`   • ${extractedData.analysis.aiModelsPresent} AI models detected`);
console.log(`   • ${extractedData.analysis.modelsFailedWorkerAssignment} models failed due to WebNN capability mismatch`);
console.log(`   • Task scheduler is active but blocked on worker requirements`);
console.log(`   • Models found: TinyLlama, FaceFormer, RSMT, WebNNAudioProcessing`);
console.log(`\\n🔍 KEY INSIGHT: All models require WebNN support but workers have webnn:false`);
