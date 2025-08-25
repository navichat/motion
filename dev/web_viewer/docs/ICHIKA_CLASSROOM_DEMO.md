# Ichika Classroom Demo (BVH + Speech)

This demo shows an Ichika VRM avatar orchestrated by a BVH timeline with speech-driven face/gestures.

What’s included:
- BVHTimeline: bone-mask-aware blending and metadata propagation (faceViseme, gestureEnergy).
- VRM integration: maps visemes to blendshapes via AvatarBinder.
- ClipRegistry + BVHClipLibrary: load static BVH clips and schedule them through the orchestrator.
- Demo (`demos/ichika_classroom_demo.html`): Start Idle, Point, Wave, Speak.

Testing posture:
- HTTP-only smokes (fast):
  - `dev/web_viewer/e2e-smoke.spec.js` (index DOM)
  - `dev/web_viewer/e2e-smoke-http.spec.js` (VRM + classroom demo fetches)
- Unit-web tests (require dev server, per-file timeouts, CI-skipped by `NO_WEBSERVER`):
  - Speech preemption, Wave and Point button logs.
- Serverless tests (no server): bone-mask blending, viseme→blendshape mapping, prebuilt clip passthrough.

Notes:
- The demo attempts to load a small BVH idle clip; if blocked (CORS/offline), it falls back to a manifest idle.
- Real inference paths can be added behind an env gate (e.g., `RUN_REAL_INFERENCE=1`) to keep CI green.

Try it quickly:
```bash
cd dev/web_viewer
python3 serve_with_headers.py
xdg-open http://127.0.0.1:8080/demos/ichika_classroom_demo.html || true

# Smokes
npm run test:smoke

# Unit-web Ichika subset
npx playwright test --project=unit-web -g "Ichika demo:" --reporter=line
```

More details:
- See `docs/ICHIKA_CLASSROOM_VRM_IMPLEMENTATION_PLAN.md` for architecture and test strategy.
