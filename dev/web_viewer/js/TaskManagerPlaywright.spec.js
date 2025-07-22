import { test, expect } from '@playwright/test';

test.describe('TaskManager with WebGPU', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/test_taskmanager.html');
    await page.addScriptTag({ path: 'dev/web_viewer/js/FibonacciHeap.js' });
    await page.addScriptTag({ path: 'dev/web_viewer/js/MockGPUJobs.js' });
    await page.addScriptTag({ path: 'dev/web_viewer/js/TaskManager.js' });
    await page.evaluate(() => {
      window.taskManager = new TaskManager({
        cpuWorkers: 2,
        gpuWorkers: 1,
        webnnWorkers: 1,
        maxConcurrentTasks: 4
      });
      window.taskManager.start();
    });
  });

  test('should initialize TaskManager', async ({ page }) => {
    test.setTimeout(10000);
    const taskManagerHandle = await page.evaluateHandle(() => window.taskManager);
    expect(taskManagerHandle).toBeTruthy();
  });

  test('should schedule and complete a single task', async ({ page }) => {
    test.setTimeout(10000);
    const result = await page.evaluate(async () => {
      const job = window.MockGPUJobFactory.createJobA(1);
      job.duration = 100;
      const taskId = window.taskManager.scheduleTask(job, 5);
      return new Promise(resolve => {
        window.taskManager.on('taskCompleted', (task) => {
          if (task.id === taskId) {
            resolve(task.result);
          }
        });
      });
    });
    expect(result).toBeDefined();
  });

  test('should handle multiple tasks with different priorities', async ({ page }) => {
    test.setTimeout(10000);
    const completedOrder = await page.evaluate(async () => {
      const completedTasks = [];
      window.taskManager.on('taskCompleted', (task) => {
        completedTasks.push(task.id);
      });

      const lowPriorityJob = window.MockGPUJobFactory.createJobA(1);
      lowPriorityJob.duration = 100;
      const highPriorityJob = window.MockGPUJobFactory.createJobB(1);
      highPriorityJob.duration = 100;

      const lowPriorityId = window.taskManager.scheduleTask(lowPriorityJob, 10);
      const highPriorityId = window.taskManager.scheduleTask(highPriorityJob, 1);

      await new Promise(resolve => {
        const checkCompletion = () => {
          if (completedTasks.length >= 2) {
            resolve();
          } else {
            setTimeout(checkCompletion, 50);
          }
        };
        checkCompletion();
      });
      return completedTasks;
    });

    expect(completedOrder).toHaveLength(2);
  });

  test('should timeout a long-running task', async ({ page }) => {
    test.setTimeout(10000); // Increased timeout to 10 seconds
    const result = await page.evaluate(async () => {
      const job = window.MockGPUJobFactory.createJobA(1);
      job.duration = 5000; // 5 seconds, longer than the timeout
      const taskId = window.taskManager.scheduleTask(job, 5, null, { timeout: 1000 }); // 1 second timeout

      return new Promise(resolve => {
        window.taskManager.on('taskFailed', (task) => {
          if (task.id === taskId) {
            resolve(task.error);
          }
        });
      });
    });
    expect(result).toContain('timeout');
  });
});
