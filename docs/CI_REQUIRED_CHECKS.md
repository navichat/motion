# Required Status Checks (Recommended)

To prevent merges that break the deterministic Ichika conversation E2E, enable required checks for the two jobs in the workflow `e2e-ultimate-conversation.yml`.

Recommended required checks (exact GitHub contexts):

- `e2e-ultimate-conversation / test`
- `e2e-ultimate-conversation / vite`

These correspond to the jobs:

- `test` → Ultimate conversation (deterministic; Python web server)
- `vite` → Ultimate conversation (Vite self-serve)

Both jobs already include shell timeouts and use IPv4 to avoid ::1 issues. Vite is invoked via `npx -y vite@^6` to avoid conflicts with system binaries.

## How to enable (GitHub UI)

1. Repository → Settings → Branches → Branch protection rules
2. Edit (or add) a rule for your default branch (e.g., `main`)
3. Check “Require status checks to pass before merging”
4. Add the two checks by name:
   - `e2e-ultimate-conversation / test`
   - `e2e-ultimate-conversation / vite`
5. Save changes

Notes:
- You need admin permissions on the repository to change branch protection.
- If you rename jobs or the workflow file, update the required check names accordingly.
