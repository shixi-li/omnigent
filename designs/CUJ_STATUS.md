# Intelligent Routing CUJ status

Living checklist, maintained by the lead session. Legend:
- ✅ **user** — Bryan confirmed it live
- ✅ **evidence** — verified from process-level ground truth (logs / DB / harness-written files), not just UI
- 🟡 **ui-only** — the UI claims it, but process reality is unverified or known to diverge
- ❌ — confirmed broken (fix status noted)
- ⬜ — not yet tested live

"UI" = what chips/dropdowns/panels display. "Process" = what the harness
process actually runs (rollout files, panes, config, spawned models).

Last updated: 2026-07-29 night — all engineering rows closed; remaining items are user decisions/confirms.

## 1. Claude Code CUJ (IR main session)

| Layer | Status | Evidence / notes |
|---|---|---|
| IR selectable in Configure Claude Code, Effort greys | ✅ user | 2026-07-29 |
| Sticky IR default next session, same harness | ✅ user | 2026-07-29 |
| Session created with routing flag, no pin | ✅ evidence | DB session_overrides |
| Router decision + chip (below message) | ✅ user | rationale correct (task_v1 cc) |
| Gateway env prepared at launch (ucode) | ✅ evidence | `configured=True`, ANTHROPIC_BASE_URL, apiKeyHelper |
| **Process runs the routed model** | ✅ evidence | live pane proof post-fix (82cac6fa): `/model sonnet` typed under the inject lock, banner Opus 5 → Sonnet 5, second turn idempotent; root cause was model_override dropped in _run_turn_bg + alias vocabulary |

## 2. Codex CUJ (IR main session)

| Layer | Status | Evidence / notes |
|---|---|---|
| IR selectable in Configure Codex | ✅ user | |
| Router decision + chip | ✅ evidence | headless battery: "hi"→luna, complex refactor→sol, both exact servable matches |
| **Process runs the routed model** | ✅ evidence | live re-run post push+mirror+hardening (51801530): model_override holds databricks-gpt-5-6-luna across the turn and the session config.toml reads model = databricks-gpt-5-6-luna |
| Codex TUI `/model` reflects the live model | ❌ | shows thread default (`databricks-gpt-5-5`), not the routed per-turn model. Design in flight (`live-model-state`) |
| Post-launch model push (re-route / lost launch race) | ❌ by design gap | no omnigent→codex push exists; claude-style injection equivalent needed. Design in flight (`live-model-state`) |

## 3. Auto harness CUJ

| Layer | Status | Evidence / notes |
|---|---|---|
| "Auto" chip + dropdown item + description | ✅ user | naming iterations settled |
| Configure Auto = Permissions only, locked Default | ✅ user | payload carries no permission override (test-pinned) |
| Harness+model decision at session start | ✅ evidence | headless auto CUJ: session-scope decision picked codex/luna from the five-arm menu, model persisted, session running |
| Cross-harness subagents allowed ONLY here | ⬜ | constraint shipped `0fb7ea95` w/ tests; not confirmed live |

## 4. Subagent routing

| Layer | Status | Evidence / notes |
|---|---|---|
| Claude subagent decisions (chips per spawn) | ✅ user | decisions fire |
| **Claude subagent spawns get the routed model** | ✅ evidence | live re-test post-fix: Explore spawn ran to completion with routed sonnet-5 decision (was 7ms schema failure) |
| Same-harness constraint (no codex suggestions in CC) | 🟡 | user: "seems like it's being followed" — not deliberately exercised yet |
| **Codex subagent hooks execute at all** | ✅ evidence | live E2E post -I fix (518376ba): canary fired, PreToolUse gate ran, SubagentStart audit recorded the spawn on the routed model (luna). Root causes: app-server ignores the bypass flag (persisted trust handshake added) + cwd shadowing killed hook imports (python -I) |
| Canary → `subagent_routing_unenforced` warning banner | ✅ evidence | watcher posted every 30s and the warning surfaced on the session snapshot during the shadowing incident — the watcher is what caught the bug |
| In-session Subagent routing row (IR/Default, inherit) | ✅ user | toggle enables/disables routing as expected |
| Mid-session toggle affects next spawn (process level) | ✅ evidence | live round-trip: off → gate declined per call ("subagent routing disabled" logged, no decision persisted, spawn proceeded); on → next spawn routed (decision count 1→2) |
| Fork spawns exempt (v1 policy) | ⬜ | test-pinned only |

## 5. Visibility & telemetry

| Layer | Status | Evidence / notes |
|---|---|---|
| Decision chips show raw→applied divergence | ✅ user | this is how two real bugs were caught — keep it |
| Chip below the user message | 🟡 | render rule shipped (8fa280ea, streaming-tested both arrival orders) — awaiting user visual confirm |
| Per-subagent routed model in sub-agents panel | 🟡 ui-only | displays child model_override, which for claude ≠ actual spawned model until apply fix lands |
| Session warning banner renders when server publishes | ⬜ | component shipped; never seen live (see canary ❌) |
| `omnigent.routing.*` OTel events | ⬜ | needs OTEL endpoint configured to observe |
| Switch-off / fork telemetry triggers | ⬜ | |

## 6. Meta

| Item | Status |
|---|---|
| Router contract (task_v1 scenarios, live probe 6/6) | ✅ evidence (`scripts/probe_routing_api.sh`) |
| Fail-open on router outage w/ reason | ✅ evidence (task_v1 rollback incident, 400s → unrouted + logged) |
| Gate INFO logs name every no-route reason | ✅ evidence |
| SAFE flag (universe), L6 live E2E suite, PR demo shots | ⬜ outstanding |

## The one-line summary

Decision layer solid everywhere; apply layer verified real only on codex
main sessions; claude apply (main + subagents) and codex subagent
enforcement are the open holes, all with fixes in flight. UI currently
reports intent, not reality — the `live-model-state` workstream closes that.
