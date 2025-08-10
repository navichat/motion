# Comprehensive Implementation Plan: Advanced BVH Animation System

**I. Core Architecture & Principles**

*   **Modular Design:** Each component (task factory, scheduler, animation source, timeline, avatar puppeteer) will be designed as a loosely coupled module to facilitate independent development, testing, and future expansion.
*   **Asynchronous Operations:** Leverage Web Workers and asynchronous programming (Promises, async/await) to prevent UI blocking during computationally intensive animation generation and processing.
*   **Real-time Performance:** Prioritize low-latency processing for animation preemption and smooth avatar puppeting.
*   **Extensibility:** Design interfaces that allow easy integration of new animation sources or avatar types.

**II. Components**

**A. Fibonacci Heap Task Scheduler (`FibonacciScheduler.js`)**

*   **Purpose:** Manage and prioritize animation generation tasks, enabling efficient preemption and dynamic task ordering.
*   **Implementation Details:**
    *   Implement a Fibonacci Heap data structure to store tasks.
    *   Each task will have a priority (e.g., based on urgency, user interaction, or animation type).
    *   Support `insert`, `extractMin`, `decreaseKey` (for priority changes), and `delete` operations.
    *   Integrate with Web Workers to distribute task execution.
*   **Key Features:**
    *   **Preemption:** Allow higher-priority tasks (e.g., speech-driven animation) to interrupt and take precedence over lower-priority tasks (e.g., idle animation).
    *   **Dynamic Prioritization:** Ability to change a task's priority mid-execution or while queued.

**B. Task Factory (`AnimationTaskFactory.js`)**

*   **Purpose:** Centralized creation and configuration of animation generation tasks.
*   **Implementation Details:**
    *   Provide methods for creating tasks for each animation source (FaceFormer, Audio2Gesture, RSMT, DeepMimic, raw BVH files).
    *   Each task object will encapsulate:
        *   `id`: Unique task identifier.
        *   `sourceType`: e.g., 'faceformer', 'audio2gesture', 'deepmimic', 'bvh_file'.
        *   `inputData`: Specific data required by the source (e.g., audio file path, text, DeepMimic parameters, BVH file content).
        *   `priority`: Initial priority for the scheduler.
        *   `outputFormat`: Ensure output is standardized BVH frame data.
        *   `status`: (e.g., 'pending', 'running', 'completed', 'interrupted').
        *   `onComplete`, `onError`, `onProgress` callbacks.
*   **Integration:** The Task Factory will submit created tasks to the Fibonacci Heap Scheduler.

**C. Animation Source Adapters (`FaceFormerAdapter.js`, `Audio2GestureAdapter.js`, `RSMTAdapter.js`, `DeepMimicAdapter.js`, `BVHFileLoader.js`)**

*   **Purpose:** Abstract the specifics of each animation generation method and provide a unified interface for the Task Factory.
*   **Implementation Details:**
    *   Each adapter will be responsible for:
        *   Loading/initializing its respective model/library (e.g., FaceFormer, Audio2Gesture, DeepMimic runtime).
        *   Taking input data and generating BVH animation frames.
        *   Handling any specific dependencies or environment setups (e.g., WebAssembly, WebGPU for neural networks).
        *   Normalizing output to a common BVH frame data structure.
*   **Existing Work Integration:** Leverage existing `engine/` and `dev/web_viewer/` components for model loading and inference where applicable (e.g., `engine/web_porting_poc`, `dev/web_viewer/models`).

**D. BVH Timeline Management (`BVHTimeline.js`, `TimelineComposer.js`)**

*   **Purpose:** Store, manage, and composite BVH animation frames from various sources into a single, coherent timeline for avatar puppeting.
*   **`BVHTimeline.js`:**
    *   Represents an independent sequence of BVH frames.
    *   Methods for adding, removing, and querying frames.
    *   Support for time-based indexing.
*   **`TimelineComposer.js`:**
    *   Takes multiple `BVHTimeline` instances as input.
    *   Provides methods to blend, layer, and transition between different timelines.
    *   Handles interpolation and smoothing during transitions to avoid jerky movements.
    *   Crucial for seamless preemption: when a new animation starts, the composer will smoothly transition from the old timeline to the new one.

**E. VRM Avatar Puppeteer (`VRMAvatarPuppeteer.js`)**

*   **Purpose:** Apply the composited BVH animation data to the Ichika VRM avatar.
*   **Implementation Details:**
    *   Utilize a VRM loading and rendering library (e.g., `three-vrm` if using Three.js, which seems likely given `three_test_simple.html`).
    *   Map BVH joint rotations and positions to the VRM avatar's bone structure.
    *   Handle avatar specific features (e.g., blend shapes for facial animation, spring bones).
    *   Receive updates from the `TimelineComposer` and render the avatar accordingly.

**III. Workflow & Interaction**

1.  **User Input/Event:** An event triggers the need for new animation (e.g., speech detected, user interaction).
2.  **Task Creation:** The `AnimationTaskFactory` creates a new animation task with a specific `sourceType`, `inputData`, and `priority`.
3.  **Task Submission:** The new task is submitted to the `FibonacciScheduler`.
4.  **Scheduler Action:**
    *   If the new task has higher priority than currently running tasks, the scheduler signals for preemption.
    *   The scheduler dispatches tasks to available Web Workers based on priority.
5.  **Animation Generation (in Web Worker):**
    *   The assigned Web Worker uses the appropriate `Animation Source Adapter` to generate BVH frames.
    *   Progress updates are sent back to the main thread.
6.  **BVH Timeline Population:** As frames are generated, they are added to a new, independent `BVHTimeline` instance.
7.  **Timeline Composition & Preemption:**
    *   Once the new `BVHTimeline` is ready (or partially ready for streaming), the `TimelineComposer` is instructed to transition to this new timeline.
    *   The `TimelineComposer` handles the smooth blending from the currently playing animation to the new one.
8.  **Avatar Puppeting:** The `VRMAvatarPuppeteer` continuously receives the current pose from the `TimelineComposer` and updates the Ichika VRM avatar's pose in the rendering engine.

**IV. Scaffolding & Testing Strategy**

**A. Documentation (`docs/BVH_ANIMATION_SYSTEM_PLAN.md`)**

*   This entire plan will be written into this Markdown file.

**B. Directory Structure (Proposed additions to `dev/web_viewer/src/`)**

```
dev/web_viewer/src/
├── animation/
│   ├── adapters/
│   │   ├── Audio2GestureAdapter.js
│   │   ├── BVHFileLoader.js
│   │   ├── DeepMimicAdapter.js
│   │   ├── FaceFormerAdapter.js
│   │   └── RSMTAdapter.js
│   ├── core/
│   │   ├── AnimationTaskFactory.js
│   │   ├── BVHTimeline.js
│   │   ├── FibonacciScheduler.js
│   │   └── TimelineComposer.js
│   └── avatar/
│       └── VRMAvatarPuppeteer.js
├── workers/
│   ├── animation-worker.js  (Web Worker for offloading animation generation)
│   └── scheduler-worker.js  (Optional: if scheduler logic itself needs offloading)
└── main.js (Or existing entry point, to integrate new system)
```

**C. Scaffolding (Placeholder Files)**

*   Create empty `.js` files for each component listed above in the proposed directory structure.
*   Add basic class/function definitions with comments outlining their intended purpose and methods.

**D. Testing Strategy**

*   **Unit Tests:**
    *   For `FibonacciScheduler`: Test `insert`, `extractMin`, `decreaseKey`, `delete` operations, and priority handling.
    *   For `AnimationTaskFactory`: Test task creation with various parameters.
    *   For `BVHTimeline`: Test frame addition, retrieval, and time-based queries.
    *   For `TimelineComposer`: Test blending, layering, and transition logic with mock BVH data.
    *   For `VRMAvatarPuppeteer`: Test mapping BVH data to a simplified avatar model (mock VRM).
*   **Integration Tests:**
    *   Test the flow from Task Factory -> Scheduler -> Adapter -> Timeline -> Composer -> Puppeteer with simplified mock data for animation sources.
    *   Test preemption scenarios: start a low-priority task, then introduce a high-priority task and verify the transition.
*   **End-to-End Tests (using Playwright, given existing `e2e-*.spec.js` files):**
    *   Simulate user interaction (e.g., speech input) triggering animation.
    *   Verify visual output of the Ichika avatar.
    *   Test preemption by triggering a new animation mid-way through an existing one and observing smooth transitions.
    *   Leverage existing Playwright setup in `dev/` and `dev/web_viewer/`.

**V. Next Steps**

1.  Create the documentation file `docs/BVH_ANIMATION_SYSTEM_PLAN.md` with the above plan.
2.  Create the proposed directory structure and placeholder files.
3.  Set up basic test files for the core components.