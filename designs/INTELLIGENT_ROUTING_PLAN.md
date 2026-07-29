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

## 1. Confirmed router API behavior (live probes, 2026-07-28/29)

Probed `POST {workspace}/ai-gateway/routing/v1/routes:select` on
**eng-ml-inference** and **eng-ml-agent-platform** staging (bearer via
`databricks auth token`). Two routers are registered: `task_v0` and `task_v1`
(the error for an unknown name enumerates them). **`task_v1` is the
harness-aware recipe — use it.** Response shape for both:
`{"route_selection": [{"route_option": {"model", "harness"}, "params": {}}], "rationale": "..."}`.

**`task_v1` (target):**

- **Scenario inference from model arms, not harness tags.** The router infers
  a scenario from *which model families appear* in `route_options`:
  Claude arms `{claude-opus-4-8, claude-sonnet-5}` → scenario `cc`;
  Codex arms `{glm-5-2, gpt-5-6-sol, gpt-5-6-luna}` → scenario `codex`;
  both families → scenario `both`. Offering no recognized arm (e.g. only
  `gpt-5-5`) → 400 "could not infer a scenario".
- **Each scenario requires its full fixed menu** (`cc` = both Claude arms,
  `codex` = all three Codex arms, `both` = all five). A partial menu → 400
  naming the missing arms. **Extra non-arm models are tolerated and ignored**
  (verified: menu + `gpt-5-5`/`claude-haiku-4-5`/`kimi-k2` routes fine) — so
  omnigent can keep sending a catalog superset as long as the full menu for
  the intended scenario is present.
- **This arm-menu selection IS the harness constraint**: send only Codex arms
  for P0 (within-codex), only Claude arms for P1 (within-CC), all five for
  the auto/cross-harness CUJ. The pick always comes from the offered menu.
- **The `harness` field itself is still passthrough**, on both routers:
  swapping tags (Claude models tagged `codex` and vice versa) neither changes
  the pick nor errors — the tag is echoed back verbatim on the selection,
  even when nonsensical. Harness intent is expressed via which arms you
  offer, not via the tag; treat the echoed harness as untrusted.
- Recipe is a rule tree (rationales expose predicates: `prompt<300`,
  `not_crosscutting`, `low_ambiguity`, "cheapest arm … never escalate";
  defaults `claude-sonnet-5` / `gpt-5-6-sol`).

**`task_v0` (previous placeholder, still registered):** static required set
`[claude-opus-4-8, glm-5-2, gpt-5-4-mini, gpt-5-5, gpt-5-6-luna]`, no
tolerance for missing ids, no harness behavior at all. Ignore except for
backward-compat testing.

**Menu ids need not exist as serving endpoints.** eng-ml-inference has
endpoints for `claude-opus-4-8`, `claude-sonnet-5`, `glm-5-2`, but **not**
`gpt-5-6-sol` / `gpt-5-6-luna` — yet task_v1 requires those ids in
`route_options` and happily *selects* them. Consequences: (a) a strictly
catalog-derived option list 400s (the two missing arms never enter it), so the
client must inject the router vocabulary; (b) a pick may be unservable in the
workspace and must be mapped to a servable id (or the decision degraded to
fallback) — see §2.

### 1.1 Backend implementation notes (universe `ai-gateway/src/routing/`)

Read the server source; it confirms the probes and adds contract facts the
client design should exploit:

- **Versioned routers are frozen.** A `router_name` pin means "same decision
  forever"; behavior changes ship as `task_v2`, never edits to v1
  (`router/CLAUDE.md`, the FROZEN banner). So scenario menus can only drift
  when *we* change `routing.router_name` — pinning the name pins the menu,
  which de-fangs risk #4 (drift is opt-in, not ambient).
- **Every task_v1 call makes an LLM extraction self-call under the caller's
  identity** (three axes: `expected_change_scope`, `prompt_ambiguity`,
  `difficulty`) — even Rule-0 needs `llm difficulty == easy`, so there is no
  LLM-free fast path. Extraction model = `route_selector.config.model` if
  set, else the frozen default `gpt-5-4-mini`, resolved as
  `system.ai.<model>` in the caller's workspace. Two implications:
  (a) routing latency ≈ one small-model call — real, budget it on the spawn
  path (risk #2); (b) **the caller needs query access to the extraction
  model** — P1 should pass `routing.selection_model` through as
  `route_selector.config.model` so deployments can pin one they have.
- **`task.prompt` is the entire routing signal.** Deterministic features are
  regexes/lengths over the prompt (stack traces, file paths, `` `symbols` ``,
  code fences; buckets at 400/1200/3000 chars; trivial cutoff 300). Send the
  user's raw task text, not a wrapper/summary — wrapping changes routing.
  Corollary for B0/P2: Codex's encrypted spawn prompts mean hook-path codex
  routing degenerates to short-prompt defaults; the redirect path
  (plaintext) is where task-aware quality lives.
- **`harness` is never read by any router.** `RouteOption.harness` is
  documented "optional for a native harness, required for a metaharness";
  selection matches on model only ("first harness wins on a duplicate"), and
  post-validation just checks the picked option was offered verbatim.
  Confirms: derive harness client-side from the picked arm's family.
- **Malformed model ids are silently dropped** from `route_options`
  (normalizer: lowercase, `.`→`-`), not 400ed — catalog junk is safe to send.
- **`session_history` exists in the request schema but no shipped router
  reads it** — the designed hook for per-turn / "sidekick" routing (open
  questions 6/10). The client should be shaped to populate it later (P2's
  decision cache already retains per-session picks).
- **`routes:select` does no access or existence checks** on offered options —
  explains unservable arms being required *and* selectable; resolution is
  entirely ours (§2).
- **No server-side decision logging yet** (TODO in `RoutingHandler`) — until
  AIGW's background-activity log lands, omnigent's decision items + OTel (P5)
  are the *only* record of routing decisions. Raises the stakes on P5.
- The wire types in universe are temporary "until Omnigent's routing protos
  sync" — `omnigent/api/routing/v1/routing.proto` is the source of truth, so
  contract extensions (codebase metadata, fork context, spawn-eligible sets)
  start as PRs on *our* proto. Endpoint is SAFE-gated server-side
  (`routeSelectionEnabled`).

Dev-loop config (worktree `.omnigent-local/config.yaml`, isolated via
`OMNIGENT_CONFIG_HOME`/`OMNIGENT_DATA_DIR`): routing `provider: external`,
`base_url: https://eng-ml-inference.staging.cloud.databricks.com/ai-gateway/routing/v1`,
`router_name: task_v1`, profile `eng-ml-inference`. Note
`OMNIGENT_SMART_ROUTING=1` from the Jul 24 runbook is obsolete — the client is
built from the `routing:` config block alone (`cli.py::_build_external_routing_client`).

## 2. Design principle: one swappable route-options boundary

Today's contract (fixed per-scenario arm menus, harness expressed by arm
choice rather than a first-class field, picks that may not be servable) is
temporary. Mason will later accept dynamic model+harness sets and enforce
harness constraints server-side. To make that a config/adapter change rather
than a refactor, **all knowledge of the router's contract lives in exactly one
place**: a `RouteOptionSource` seam inside `omnigent/server/smart_routing.py`:

- `build_route_options(harnesses, catalog) -> list[RouteOption]` — v0
  implementation selects the task_v1 scenario menu from the requesting
  harness set (codex-only → Codex arms, claude-only → Claude arms, mixed →
  all five; menus config-overridable via `routing.scenario_menus`), injecting
  menu ids even when absent from the catalog. A future
  `CatalogRouteOptionSource` returns the caller's real model set.
- `resolve_selection(pick, harnesses, catalog) -> ResolvedRoute` — v0 ignores
  the router's echoed `harness` (passthrough, untrusted — §1), derives the
  harness from the picked arm's family (keeping
  `_HARNESS_EXCLUDED_MODELS`-style compatibility data here rather than
  deleting it), and maps router vocabulary → servable catalog id (prefix
  restore + nearest-available fallback when the picked id has no endpoint,
  e.g. `gpt-5-6-sol`/`gpt-5-6-luna` on eng-ml-inference today). When
  server-side constraints ship, this collapses to prefix restore only.

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
  implementation (task_v1 scenario menus per §1, config-overridable via
  `routing.scenario_menus`; default `routing.router_name` = `task_v1` — menus
  are keyed by router version since versions are frozen, §1.1).
- Map unservable picks to the nearest servable catalog id; emit the raw pick
  in the decision payload either way (UI shows what the router said).
- Pass `routing.selection_model` config through as
  `route_selector.config.model` (extraction self-call runs under caller
  identity — §1.1); send the raw task text as `task.prompt`, never a
  wrapped/summarized prompt.
- Default-on: when the server's provider is Databricks (`kind: databricks`)
  and no `routing:` block exists, synthesize the external client against that
  workspace's `/ai-gateway/routing/v1` with
  `model_prefixes=["databricks-", "system.ai."]` and profile auth. Explicit
  `routing:` config always wins; `routing.provider: none` opts out.
- Fallback ordering unchanged: external → LLM judge → disabled.
- Tests: scenario-menu request shape (correct arm menu per harness set,
  injected even when absent from catalog; catalog extras preserved), harness
  derived from picked arm's family (echoed harness ignored), unservable-pick
  mapping (`gpt-5-6-sol`/`luna`), default-on synthesis with/without
  Databricks creds, 400-missing-arms surfaced as a warning + fallback (not a
  crash).

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
  Databricks-backed deployments (P1) using `task_v1`, surviving the confirmed
  contract quirks: fixed scenario menus, passthrough harness tags, unservable
  picks.
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
4. **Scenario-menu drift**: routers are frozen per version (§1.1), so menus
   only change when we bump `routing.router_name` (e.g. to a future task_v2)
   — bumping the name and the menus must happen together (`scenario_menus`
   keyed by router version enforces this). P1 still degrades to fallback with
   a loud warning if they're ever mismatched.
5. **Fork/cache-miss economics**: v1 doesn't route forks; a real cost model
   needs inherited-context size that only Omnigent can supply out-of-band.
