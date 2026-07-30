# Intelligent Routing CUJ status

Living checklist, maintained by the lead session. Legend:
- ✅ **user** — Bryan confirmed it live
- ✅ **evidence** — verified from process-level ground truth (logs / DB / harness-written files), not just UI
- 🟡 **ui-only** — the UI claims it, but process reality is unverified or known to diverge
- ❌ — confirmed broken (fix status noted)
- ⬜ — not yet tested live

"UI" = what chips/dropdowns/panels display. "Process" = what the harness
process actually runs (rollout files, panes, config, spawned models).

Last updated: 2026-07-29 late night — child constraint, chip pairing, rename, telemetry rework landed; top-level Smart Routing harness in flight.

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
| Codex TUI reflects the live model | ✅ evidence | thread-level push (thread/settings/update) live-updates the TUI status bar (probed); /model picker highlight is upstream codex behavior — noted in designs/LIVE_MODEL_STATE.md |
| Post-launch model push (re-route / lost launch race) | ✅ evidence | first-turn push + config mirror + forwarder hardening (51801530); live re-run held luna in model_override and config.toml |

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
| Same-harness constraint (native spawns + omnigent children) | ✅ evidence | hook path shipped earlier; child-session gap found live (codex parent → 9 forced-auto children, some claude-opus) and fixed (5a397d6f): children stay in the parent's family unless the parent is genuinely Smart Routing |
| **Codex subagent hooks execute at all** | ✅ evidence | live E2E post -I fix (518376ba): canary fired, PreToolUse gate ran, SubagentStart audit recorded the spawn on the routed model (luna). Root causes: app-server ignores the bypass flag (persisted trust handshake added) + cwd shadowing killed hook imports (python -I) |
| Canary → `subagent_routing_unenforced` warning banner | ✅ evidence | watcher posted every 30s and the warning surfaced on the session snapshot during the shadowing incident — the watcher is what caught the bug |
| In-session Subagent routing row (IR/Default, inherit) | ✅ user | toggle enables/disables routing as expected |
| Mid-session toggle affects next spawn (process level) | ✅ evidence | live round-trip: off → gate declined per call ("subagent routing disabled" logged, no decision persisted, spawn proceeded); on → next spawn routed (decision count 1→2) |
| Fork spawns exempt (v1 policy) | ⬜ | test-pinned only |

## 5. Visibility & telemetry

| Layer | Status | Evidence / notes |
|---|---|---|
| Decision chips show raw→applied divergence | ✅ user | this is how two real bugs were caught — keep it |
| Chip below the user message | 🟡 | render rule (8fa280ea) + claude fix: the injected /model echo broke pairing on claude only, now skipped (25b75c62) — awaiting user visual confirm on a fresh claude session |
| Per-subagent routed model in sub-agents panel | 🟡 | apply fixes landed on both harnesses, so the displayed override now matches reality on fresh sessions; not re-eyeballed since |
| Session warning banner renders when server publishes | ✅ user | Bryan screenshotted the live banner during the shadowing incident; over-warning on routing-off sessions fixed same day |
| Routing analytics (OSS telemetry pipeline) | 🟡 | reworked per PR review (c7f78f26): RoutingDecisionEvent/RoutingSettingChangedEvent with family/tier-only model labels; OTel helper deleted; not yet observed against a live ingestion endpoint |
| Switch-off / fork telemetry triggers | ⬜ | browser-side spans only (routingTelemetry.ts); server-side toggle event ships in RoutingSettingChangedEvent |

### Renames & new asks (2026-07-29 late)

| Item | Status |
|---|---|
| All UI labels renamed to "Smart Routing" | ✅ user-directed, shipped e5c8a160 |
| Top-level Smart Routing harness in the landing dropdown (agentless auto over native claude/codex) | 🔨 in flight (`smart-routing-harness` agent) |
| GLM absent from codex model list (eng-ml-agent-platform) | ❌ external: codex's client-side model registry/ucode has no glm-5-2; gateway serves no /models on the codex path — isaac/ucode distribution question, not omnigent |
| task_v1 escalates clear+contained prompts to opus (well-written spawn prompts always pay opus) | 📝 recipe feedback for Ivan — frozen router, needs task_v2 |

## 6. Meta

| Item | Status |
|---|---|
| Router contract (task_v1 scenarios, live probe 6/6) | ✅ evidence (`scripts/probe_routing_api.sh`) |
| Fail-open on router outage w/ reason | ✅ evidence (task_v1 rollback incident, 400s → unrouted + logged) |
| Gate INFO logs name every no-route reason | ✅ evidence |
| SAFE flag (universe), L6 live E2E suite, PR demo shots | ⬜ outstanding |

## The one-line summary

Decision and apply layers evidence-verified on both harnesses, main and
subagents, with mid-session toggles live. Open: user visual confirms
(chip placement, fresh-session panel), the top-level Smart Routing
harness (in flight), fork routing policy, and the external GLM/codex
model-list gap.
