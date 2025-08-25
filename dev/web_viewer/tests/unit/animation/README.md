# Animation Unit-Web Tests

This folder contains deterministic unit-web tests for the BVH animation pipeline:

- Composition and ordering across tracks (base vs audio)
- Fade-in envelope semantics and composedFrom weights
- Metadata propagation (face viseme, gesture energy)
- Buffering behaviors (prebuffering, invalidation on clip removal)
- VRM integration smoke via a mock adapter
- TimelineMixer smokes (standalone + with real BVHTimelines)

Most tests require the dev web server (skip when `NO_WEBSERVER=1`). The server is started automatically by Playwright via `playwright.config.js` with enforced shell timeouts.

## Quick runs

- All animation unit-web tests:
  - npm run test:unit:animation
- VRM integration smoke only:
  - npm run test:unit:animation:vrm-smoke
- Mixer smokes:
  - npm run test:unit:animation:mixer

## Running just VRM-related tests

Use a title grep to target VRM tests:

```sh
# With the unit-web project (requires dev server)
PW_GREP="VRM|viseme" npm run test:unit:web

# Or run a single spec
npx playwright test --project=unit-web dev/web_viewer/tests/unit/animation/vrm-viseme-driver-web.spec.js --reporter=line
```

## Notes

- Tests load modules in the page context with a CommonJS shim to avoid `window` shape issues.
- Some tests set a bone mapping so motionData indices map to names used by track influence sets: `timeline.setBoneMapping(new Map([[5, 'head'], [7, 'leftArm']]))`.
- ComposedFrom weight reflects the clip’s configured weight; fade-in envelope is tracked on frames (e.g., `weightEnvelope`) but isn’t multiplied into composedFrom.
