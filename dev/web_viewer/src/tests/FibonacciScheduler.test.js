
/**
 * FibonacciScheduler.test.js
 *
 * Basic unit tests for the FibonacciScheduler.
 * These tests are conceptual and would require a testing framework like Jest or Playwright's test runner.
 */

import FibonacciScheduler from '../animation/core/FibonacciScheduler.js';
import AnimationTaskFactory from '../animation/core/AnimationTaskFactory.js';

describe('FibonacciScheduler', () => {
    let scheduler;
    let mockWorker;
    let taskFactory;

    beforeEach(() => {
        // Mock a Web Worker
        mockWorker = {
            postMessage: jest.fn(),
            onmessage: null,
            onerror: null,
        };
        scheduler = new FibonacciScheduler();
        scheduler.init(mockWorker);
        taskFactory = new AnimationTaskFactory();
    });

    test('should add a task and dispatch it to the worker', () => {
        const mockTask = taskFactory.createTask('faceformer', {}, 10, () => {}, () => {});
        scheduler.addTask(mockTask);

        expect(mockWorker.postMessage).toHaveBeenCalledWith({
            type: 'startTask',
            task: expect.objectContaining({ id: mockTask.id, sourceType: 'faceformer' })
        });
        expect(scheduler.runningTask.id).toBe(mockTask.id);
    });

    test('should prioritize higher priority tasks', () => {
        const lowPriorityTask = taskFactory.createTask('audio2gesture', {}, 5, () => {}, () => {});
        const highPriorityTask = taskFactory.createTask('faceformer', {}, 10, () => {}, () => {});

        scheduler.addTask(lowPriorityTask);
        // Simulate completion of the first task to allow the next to be scheduled
        scheduler._handleWorkerMessage({ data: { type: 'taskComplete', taskId: lowPriorityTask.id, result: [] } });

        scheduler.addTask(highPriorityTask);

        expect(mockWorker.postMessage).toHaveBeenCalledWith({
            type: 'startTask',
            task: expect.objectContaining({ id: highPriorityTask.id, sourceType: 'faceformer' })
        });
        expect(scheduler.runningTask.id).toBe(highPriorityTask.id);
    });

    test('should preempt a lower priority running task with a higher priority new task', () => {
        const lowPriorityTask = taskFactory.createTask('audio2gesture', {}, 5, () => {}, jest.fn());
        const highPriorityTask = taskFactory.createTask('faceformer', {}, 10, () => {}, () => {});

        scheduler.addTask(lowPriorityTask);
        expect(scheduler.runningTask.id).toBe(lowPriorityTask.id);

        // Add high priority task while low priority is running
        scheduler.addTask(highPriorityTask);

        // Expect the low priority task's onError to be called due to preemption
        expect(lowPriorityTask.onError).toHaveBeenCalledWith(expect.any(Error));
        expect(lowPriorityTask.onError.mock.calls[0][0].message).toBe("Task preempted");

        // Expect the high priority task to be dispatched
        expect(mockWorker.postMessage).toHaveBeenCalledWith({
            type: 'startTask',
            task: expect.objectContaining({ id: highPriorityTask.id, sourceType: 'faceformer' })
        });
        expect(scheduler.runningTask.id).toBe(highPriorityTask.id);
    });

    test('should handle task completion', () => {
        const mockTask = taskFactory.createTask('faceformer', {}, 10, jest.fn(), () => {});
        scheduler.addTask(mockTask);

        // Simulate worker completing the task
        scheduler._handleWorkerMessage({ data: { type: 'taskComplete', taskId: mockTask.id, result: ["bvh_data"] } });

        expect(mockTask.onComplete).toHaveBeenCalledWith(["bvh_data"]);
        expect(scheduler.runningTask).toBeNull();
    });

    test('should handle task error', () => {
        const mockTask = taskFactory.createTask('faceformer', {}, 10, () => {}, jest.fn());
        scheduler.addTask(mockTask);

        // Simulate worker reporting an error
        scheduler._handleWorkerMessage({ data: { type: 'taskError', taskId: mockTask.id, error: "Failed to generate" } });

        expect(mockTask.onError).toHaveBeenCalledWith("Failed to generate");
        expect(scheduler.runningTask).toBeNull();
    });

    test('should change task priority and re-sort', () => {
        const taskA = taskFactory.createTask('faceformer', {}, 5, () => {}, () => {});
        const taskB = taskFactory.createTask('audio2gesture', {}, 10, () => {}, () => {});

        scheduler.addTask(taskA);
        scheduler.addTask(taskB);

        // Simulate taskA completing so taskB becomes running
        scheduler._handleWorkerMessage({ data: { type: 'taskComplete', taskId: taskA.id, result: [] } });

        // Change priority of taskA to be higher than taskB (which is now running)
        scheduler.changeTaskPriority(taskA.id, 15);

        // Expect taskB to be preempted and taskA to be dispatched
        expect(mockWorker.postMessage).toHaveBeenCalledWith({
            type: 'startTask',
            task: expect.objectContaining({ id: taskA.id, priority: 15 })
        });
        expect(scheduler.runningTask.id).toBe(taskA.id);
    });
});
