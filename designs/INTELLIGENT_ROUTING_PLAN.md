# Intelligent Routing × AIGW: MVP Engineering Plan (Bryan's workstreams)

Owner: bryan.qiu@databricks.com
Status: ACTIVE — updated 2026-07-28 against main @ `c62bfc2f`; router API behavior
confirmed by live probes (see §1). MVP target: **Jul 31, 2026**.
Branch strategy: **all MVP work lands on the single branch `routing-mvp`**
(worktree `~/omnigent-routing-mvp`, cut from latest main). No per-task branches —
tasks are partitioned by file ownership (§4) so parallel agents don't collide.
Split into reviewable PRs only after end-to-end testing on this branch.

MVP requirements (from the Jul 28 brainstorm/meeting):

1. When using AIGW for inference, intelligent routing goes to the AIGW routing API.
2. Subagent routing is deterministic — including subagents spawned *natively* by
   Claude Code (Task tool) and Codex (`spawn_agent`), not just Omnigent-spawned
   child sessions.
3. All routing decisions are visible in the UI.
4. Telemetry: capture when intelligent routing is enabled, when users switch OFF
   of it mid-flight, and forks off an intelligent-routing session (OTel).
5. SAFE flag with isaac to default-on for specific users (lives in universe, not
   this repo — tracked in §7, not a task packet here).

---

## 1. Confirmed router API behavior (live probes, 2026-07-28)

Probed `POST {workspace}/ai-gateway/routing/v1/routes:select` with
`router_name=task_v0` on **eng-ml-inference** and **eng-ml-agent-platform**
staging (bearer via `databricks auth token`). Identical behavior on both:

- **Live and working.** Response shape:
  `{"route_selection": [{"route_option": {"model", "harness"}, "params": {}}], "rationale": "..."}`.
- **Static required model set.** The recipe requires *exactly*
  `[claude-opus-4-8, glm-5-2, gpt-5-4-mini, gpt-5-5, gpt-5-6-luna]` in
  `route_options`. Any subset → `400 BAD_REQUEST` naming the missing ids.
  This is per Ivan/Mason: the model set is static for now; a caller-supplied
  model+harness set comes later.
- **NOT harness-constrained server-side.** The `harness` field on a route
  option is echoed back on the pick but does not constrain it: offering all
  five models tagged `harness=codex` (no claude-sdk option at all) still
  returned `claude-opus-4-8` (with `harness: "codex"` echoed verbatim).
  **Harness feasibility is therefore the client's job today.** Per the meeting,
  the router will constrain the *returned* model by the harnesses passed in
  eventually — design for that without depending on it (§2).
- **Required ids need not exist as serving endpoints.** Both staging workspaces
  lack `databricks-gpt-5-4-mini` and `databricks-gpt-5-6-luna` endpoints, yet
  the router demands those ids in `route_options`. Consequence: a strictly
  catalog-derived option list 400s. The client must send the router's
  vocabulary, then map the pick back to something actually servable.
- Trivial-task prompts route to `claude-opus-4-8` ("bug fix rewards deeper
  reasoning"); mid/complex prompts routed to `gpt-5-5`. Recipe quality is
  Ivan's problem, not ours — but transcript rationale display (C2) matters
  precisely because of picks like this.

Dev-loop config (worktree `.omnigent-local/config.yaml`, isolated via
`OMNIGENT_CONFIG_HOME`/`OMNIGENT_DATA_DIR`): routing `provider: external`,
`base_url: https://eng-ml-inference.staging.cloud.databricks.com/ai-gateway/routing/v1`,
`router_name: task_v0`, profile `eng-ml-inference`. Note
`OMNIGENT_SMART_ROUTING=1` from the Jul 24 runbook is obsolete — the client is
built from the `routing:` config block alone (`cli.py::_build_external_routing_client`).

## 2. Design principle: one swappable route-options boundary

Today's contract (static 5-model vocabulary, client-side harness filtering,
picks that may not be servable) is temporary. Mason will later accept dynamic
model+harness sets and enforce harness constraints server-side. To make that a
config/adapter change rather than a refactor, **all knowledge of the router's
contract lives in exactly one place**: a `RouteOptionSource` seam inside
`omnigent/server/smart_routing.py`:

- `build_route_options(harnesses, catalog) -> list[RouteOption]` — v0
  implementation returns the static required vocabulary annotated with the
  requesting harness(es), ignoring the catalog except for prefix mapping.
  A future `CatalogRouteOptionSource` returns the caller's real model set.
- `resolve_selection(pick, harnesses, catalog) -> ResolvedRoute` — v0 applies
  the harness-compatibility correction client-side (the existing
  `_HARNESS_EXCLUDED_MODELS` / `_redirect_incompatible_pick` logic moves here
  instead of being deleted) and maps router vocabulary → servable catalog id
  (prefix restore + nearest-available fallback when the picked id has no
  endpoint). When the server-side constraint ships, this collapses to prefix
  restore only.

Callers (`route_session_harness`, `route_turn`, the B0 subagent endpoint) never
see router vocabulary or harness-correction rules. Router recipe name stays a
config key (`routing.router_name`), never hardcoded in logic.

## 3. State of the world on main (do not rebuild)

| Capability | Where |
|---|---|
| AIGW routing client (`routes:select`, proto, Databricks OAuth per-call refresh, model-prefix mapping) | `omnigent/server/smart_routing.py::ExternalRoutingClient`, `omnigent/api/routing/v1/routing.proto` |
| Local LLM-judge fallback router | `smart_routing.py::LLMRoutingClient` (built from server `llm:` block) |
| Pluggable router | `RuntimeCaps.routing_client` (`omnigent/runtime/caps.py`) |
| Auto-harness session routing (SDK harnesses `claude-sdk`/`codex`/`pi`) | `smart_routing.py::route_session_harness` ← `server/routes/_sessions/orchestration.py` (~3614) |
| Per-turn routing (`cost_control_mode_override == "on"`, incl. `/model` injection into live Claude-native sessions) | `smart_routing.py::route_turn` → `orchestration.py` (~3734, ~3995) |
| Omnigent-spawned child routing via parent catalog | `orchestration.py` (~3697) |
| Deterministic model-override plumbing (native `--model` argv, SDK `HARNESS_<H>_MODEL` env, family-mismatch guard) | `omnigent/model_override.py` |
| Routing-decision transcript item (applied vs would-have-picked, rationale, child→parent mirroring) | `omnigent/entities/conversation.py::RoutingDecisionData` |
| UI Smart Routing sentinel (hard-disabled: `smartRoutingEligible = false`) | `web/src/components/HarnessConfigControls.tsx`, `web/src/shell/NewChatDialog.tsx` (~2165) |
| `sys_advise_models` tool (present only when routing enabled) | `omnigent/runner/tool_dispatch.py` (~284) |

**Gaps this plan closes:** (G1) native in-harness subagents are unrouted —
Claude-native hooks are observe-only (`claude_native_forwarder.py`), Codex's
private `CODEX_HOME` symlinks the *user's* `hooks.json` (`inner/codex_executor.py`);
(G2) router-contract knowledge is smeared across the client (static
`MODEL_LISTS`, `_HARNESS_EXCLUDED_MODELS` post-correction) instead of the §2
seam; (G3) no routing telemetry; (G4) decisions not visible per-subagent.

Enforcement primitives verified externally (2026-07-28 research reports):

- **Claude Code**: `PreToolUse` hook on the Agent/Task tool can deny **and**
  rewrite `tool_input` (incl. `model`) via `hookSpecificOutput.updatedInput` +
  `permissionDecision: "allow"`. Settings-level hooks recurse to nested
  subagents. Works in CLI and Agent SDK.
- **Codex** (live-verified on codex-cli 0.145.0): Claude-compatible hooks;
  `PreToolUse` on `spawn_agent` can deny and rewrite args **including
  injecting `model`** (not LLM-visible in the schema; harness accepts it).
  Caveats: flattened tool name is `collaborationspawn_agent` — match
  `.*spawn_agent` by regex; the spawn `message` field is **encrypted** in hook
  payloads (route on `task_name` + metadata, never prompt text); unmanaged
  hooks are **silently skipped** unless trusted; `SubagentStart` payload
  carries actual `agent_id` + `model` for audit.

---

## 4. Task packets (optimized for parallel Opus 5 subagents)

Rules of engagement for subagents:

- One packet = one agent = an **exclusive set of owned files**. Never edit
  another packet's files; integration points go through the frozen contracts
  in §5. Shared-file exceptions are called out explicitly per packet.
- Every packet lands its own unit tests alongside the code and must pass
  `pre-commit run --files <owned files>` + its own `uv run pytest` targets
  before committing to `routing-mvp`.
- Commit per packet (small, revertable), all on `routing-mvp`.

### Wave 1 — six packets, fully parallel

**P1 — Route-options seam + default-on AIGW router** *(req 1, G2)*
Owns: `omnigent/server/smart_routing.py`, `omnigent/cli.py`
(`_build_external_routing_client` region only), `tests/server/test_smart_routing*.py`.
- Introduce `RouteOptionSource` per §2; move `_HARNESS_EXCLUDED_MODELS` /
  `_redirect_incompatible_pick` / `MODEL_LISTS` behind it as the v0 static
  implementation (required vocabulary = the task_v0 five, config-overridable
  via `routing.required_models`).
- Map unservable picks to the nearest servable catalog id; emit the raw pick
  in the decision payload either way (UI shows what the router said).
- Default-on: when the server's provider is Databricks (`kind: databricks`)
  and no `routing:` block exists, synthesize the external client against that
  workspace's `/ai-gateway/routing/v1` with
  `model_prefixes=["databricks-", "system.ai."]` and profile auth. Explicit
  `routing:` config always wins; `routing.provider: none` opts out.
- Fallback ordering unchanged: external → LLM judge → disabled.
- Tests: static-vocabulary request shape (all five present regardless of
  catalog), harness correction on a codex-tagged opus pick, unservable-pick
  mapping, default-on synthesis with/without Databricks creds, 400-missing-
  models surfaced as a warning + fallback (not a crash).

**P2 — Runner `route-subagent` endpoint** *(req 2 backbone)*
Owns: new `omnigent/runner/subagent_routing.py`, the runner HTTP surface file
that exposes it, server relay route (mirror the existing
`/v1/sessions/{id}/models` proxy pattern used by `smart_routing.fetch_runner_models`),
`tests/server/test_subagent_routing.py`.
- Implements the §5.1 contract. Internally calls `RuntimeCaps.routing_client`
  through the server relay; applies policy server-side so hook scripts stay
  dumb: `fork=true` → `allow` unchanged (v1: don't route forks);
  router unreachable → configurable `routing.subagent_fail_mode`
  (`open`=allow unchanged [default], `closed`=deny) — pilot runs `closed`.
- Caches per (session, task-hash) to keep the blocking spawn path fast.
- Persists every decision as a transcript item via the §5.2 shape.
- Tests with a fake routing client: rewrite/redirect/deny/fork/outage paths.

**P3 — Claude PreToolUse router hook** *(req 2, CC native + SDK)*
Owns: `omnigent/claude_native_bridge.py` (hook-provisioning region), new hook
script under `omnigent/inner/hook_scripts/`, `omnigent/inner/claude_sdk_executor.py`
(hook registration only), matching tests.
- Generated settings gain a `PreToolUse` entry on the Agent/Task tool calling
  the packaged script (stdlib-only Python; must be fast — it blocks spawns).
  SDK path registers the same decision logic as an in-process hook callback.
- Script maps §5.1 responses: `rewrite` → allow + `updatedInput` with routed
  `model`; `redirect` → deny with reason
  `"Router selected <harness>/<model>. Use sys_session_send with args.harness=…, args.model=… instead."`;
  `deny` → deny with router reason. Fork-typed spawns pass `fork=true`.
- Until P2 merges, develop against a stub of the §5.1 contract (frozen).
- Tests: fixture hook payloads → exact hook JSON out; settings-generation
  snapshot; recursion (nested Task) uses the same settings file.

**P4 — Codex hooks.json generation + trust + canary** *(req 2, Codex)*
Owns: `omnigent/inner/codex_executor.py`, new codex hook script under
`omnigent/inner/hook_scripts/`, matching tests.
- Stop symlinking the user's `hooks.json` when routing is on; generate a
  merged one: user hooks + Omnigent `PreToolUse` matcher `.*spawn_agent`
  (regex — flattened name is `collaborationspawn_agent` on 0.145.x; re-check
  the tool name inside the script), generous timeout, + `SessionStart` canary
  (touch file in bridge dir) + `SubagentStart` audit writer (`agent_id`,
  `model` → bridge dir for reconciliation against `decision_id`).
- Add `--dangerously-bypass-hook-trust` to Omnigent-launched codex argv (we
  own the generated file in a private home); document the managed
  `requirements.toml` path for fleet later. Canary absent after launch →
  emit the §5.3 warning event instead of failing open silently.
- Codex constraints: spawn `message` is encrypted — pass through verbatim on
  rewrite, route on `task_name`/parent-model/metadata only; injected `model`
  must come from the harness's spawn-eligible set or the handler errors.
- Tests: hooks.json merge snapshot, canary detection, audit-file parsing,
  argv assembly.

**P5 — Decision data model + telemetry** *(reqs 3+4 backbone)*
Owns: `omnigent/entities/conversation.py` (`RoutingDecisionData` only),
OTel emission in `omnigent/runtime/telemetry.py` call sites it adds, matching tests.
- Extend `RoutingDecisionData` per §5.2 (additive, defaulted fields — old
  rows must deserialize).
- OTel events (namespaced `omnigent.routing.*`): `decision` (every §5.2 item,
  incl. scope + applied), `disabled_mid_session` (user switches off IR),
  `enabled` (session starts with IR on), `fork_from_routed_session`.
  Emit from the entity-adjacent helper so call sites stay one-liners.
- Tests: serialization round-trip incl. legacy rows; event payload shapes.

**P6 — UI: per-subagent visibility + toggle telemetry wiring** *(reqs 3+4)*
Owns: `web/src/` only — transcript routing card, `web/src/shell/Sidebar.tsx`,
`subagentStatus.ts`, `NewChatDialog.tsx`, jest tests.
- Transcript card: add harness + scope badge ("subagent: <name>"), raw pick
  vs applied model when they differ, rationale as today.
- Sidebar child/subagent rows: show routed model per subagent (ankit req #1).
- Surface the P4 canary warning on the session header
  ("subagent routing not enforced").
- Fire the switch-off/fork telemetry triggers (client → existing event
  plumbing) when the user leaves IR mid-session or forks a routed session.
- NewChatDialog: keep Auto harness as the entry point; leave the dead
  `smartRoutingEligible` sentinel untouched this wave (design decision with
  Ajay/Tomu pending — §7).
- Develops against §5.2 as fixture data; jest only (`npx jest web/src/...`).

### Wave 2 — integration (start once the relevant Wave-1 packets merge)

**P7 — Hook↔endpoint integration + override precedence** *(needs P2+P3+P4)*
- Wire the real endpoint URL/token into the generated Claude settings and
  Codex hooks.json (bridge-dir env file), replacing the P3/P4 stubs.
- B3 determinism check: when routing is on, an LLM-supplied `args.model` on
  `sys_session_send` must NOT override the router (`tools/builtins/spawn.py`);
  record the attempted override in the decision item. Add the test.
- SubagentStart-vs-decision reconciliation in the codex forwarder; mismatch →
  warning event.
- Integration tests with a fake router: spawn rewritten in-harness, blocked
  cross-harness with redirect text, transcript item present (extend the
  `tests/server/integration/test_sessions_child_sessions.py` pattern).

**P8 — Live E2E + codex probe** *(needs P7)*
- E2E (conventions from `tests/e2e/test_polly_subagent_model_e2e.py`):
  Claude-native Task spawn model rewritten; Codex `spawn_agent` rewritten and
  `SubagentStart` model matches the decision; cross-harness redirect followed
  by the model at least once.
- Standalone codex hook probe (deny + rewrite smoke) runnable on every codex
  version bump; pin the installed codex version (`harness_install_spec.py`).
- Manual pass against staging AIGW via the worktree dev stack
  (`run-server.sh` / `run-host.sh` / `run-frontend.sh`, ports 6868/5273).

### Parallelism summary

```
Wave 1 (parallel):  P1   P2   P3   P4   P5   P6
                      \   |  /  \  |   /   |
Wave 2:                P7 (P2+P3+P4[+P1])  P6 finishes against P5 shapes
Wave 3:                P8 (all)
```

P3/P4/P6 code against frozen contracts (§5), not against P2/P5's merged code —
that's what makes Wave 1 six-wide. P7 is the only packet allowed to touch
multiple packets' files (it's the integrator; schedule it after Wave 1 merges).

---

## 5. Frozen interface contracts (agents code against these, not each other)

### 5.1 `route-subagent` endpoint (P2 serves; P3/P4 consume)

`POST {runner_local}/v1/sessions/{session_id}/route-subagent`

```json
// request
{
  "harness": "claude-sdk" | "codex" | "claude-native" | "codex-native",
  "task_name": "string",            // subagent_type / task_name
  "prompt": "string | null",        // null on codex (encrypted upstream)
  "fork": false,
  "parent_model": "string | null"
}
// response
{
  "action": "allow" | "rewrite" | "redirect" | "deny",
  "model": "string | null",         // servable id, set for rewrite/redirect
  "harness": "string | null",       // set for redirect
  "raw_model": "string | null",     // router-vocabulary pick, pre-resolution
  "rationale": "string",
  "decision_id": "uuid"
}
```

### 5.2 `RoutingDecisionData` additions (P5 defines; P2/P6/P7 consume)

Additive fields: `harness: str | None`,
`scope: Literal["session","turn","child_session","native_subagent"]`,
`decision_id: str | None`, `raw_model: str | None`,
`attempted_override: str | None`. All defaulted for legacy rows.

### 5.3 Canary warning event (P4 emits; P6 renders)

Session-scoped warning `subagent_routing_unenforced` with `{harness, reason}`,
delivered on the existing session-status channel.

---

## 6. MVP definition of done (Jul 31)

- AIGW `routes:select` drives session/turn/child routing by default on
  Databricks-backed deployments (P1), surviving the confirmed contract quirks:
  static vocabulary, no server-side harness constraint, unservable picks.
- A Claude Task spawn and a Codex `spawn_agent` cannot proceed on a
  non-router-approved model; failure mode is "didn't spawn", never "wrong
  model" (P2–P4, P7).
- Every decision renders in the transcript; per-subagent routed model in the
  sidebar; canary warning when enforcement is off (P5, P6).
- `omnigent.routing.*` OTel events for enabled / switched-off / fork (P5, P6).
- E2E green on the live dev stack against staging AIGW (P8).

## 7. Outside this branch / open items

- **SAFE flag** for isaac default-on cohort — universe repo, after this branch
  is testable.
- **NewChatDialog reconciliation** (Auto harness vs per-harness IR toggle,
  naming "Intelligent Routing") — blocked on Ajay/Tomu design call.
- **v3 gateway model-listing API** — explicitly not-MVP per the checklist.
- **Feed to Mason**: (a) harness-constrained returns (client correction is a
  stopgap — §1); (b) spawn-eligible model subsets per harness (Codex spawns
  support fewer models than sessions); (c) required-vocabulary ids should not
  400 when absent from the caller's workspace.
- **Feed to Ivan**: encrypted Codex spawn prompts cap task-aware quality for
  in-harness codex subagent routing (signal lives in `task_name` only).

## 8. Risks

1. **Codex hook fragility** (tool-name flattening, trust gate, v1/v2
   selection): regex matcher + canary + pinned version + standalone probe;
   budget breakage on bumps.
2. **Router latency blocks spawns**: p99 of `routes:select` sits on the
   PreToolUse path; mitigated by P2 caching + fail-mode knob.
3. **Redirect compliance is soft**: deny+redirect relies on the model calling
   `sys_session_send`; worst case non-spawn, never wrong-model. Track
   redirect-follow rate via decision items during the pilot.
4. **Static vocabulary drift**: if Ivan's recipe changes its required set, the
   client 400s until `routing.required_models` is updated; P1 must degrade to
   fallback with a loud warning, and the config knob is the escape hatch.
5. **Fork/cache-miss economics**: v1 doesn't route forks; a real cost model
   needs inherited-context size that only Omnigent can supply out-of-band.
