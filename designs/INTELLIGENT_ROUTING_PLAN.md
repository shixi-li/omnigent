# Intelligent Routing × AIGW: MVP Engineering Plan (Bryan's workstreams)

Owner: bryan.qiu@databricks.com
Status: ACTIVE — updated 2026-07-29 against main @ `c62bfc2f`. Router API behavior
confirmed by live probes (§1); backend read from universe `ai-gateway/src/routing`
(§1.1); **every omnigent-side claim below verified against code** (file:line
refs are exact as of `c62bfc2f`). MVP target: **Jul 31, 2026**.
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
   this repo — tracked in §8, not a task packet here).

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
no-routing) — see §2.

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
  (Client already truncates to 4000 chars — `smart_routing.py:505` — which
  preserves the "long" bucket boundary at 3000; keep the truncation.)
  Corollary for P2: Codex's encrypted spawn prompts mean hook-path codex
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

Callers (`route_session_harness`, `route_turn`, the P2 subagent endpoint)
never see router vocabulary or harness-correction rules. Router recipe name
stays a config key (`routing.router_name`), never hardcoded in logic.

**Failure semantics (corrected):** the "external vs LLM judge" choice is
**build-time** — `cli.py:3420` constructs exactly one client into
`RuntimeCaps.routing_client`; there is no runtime chain. At runtime a router
failure returns `None` with `last_error` set (`smart_routing.py:518-582`),
and callers proceed **unrouted** with the reason surfaced
(`route_session_harness` returns the error string for the UI). MVP keeps
this fail-open posture for sessions/turns; the P2 subagent path adds the
configurable strict mode (§5.1) because "unrouted spawn" there means the
determinism guarantee silently lapses.

## 3. State of the world on main (verified 2026-07-29, `c62bfc2f`)

| Capability | Where (exact) |
|---|---|
| AIGW routing client: proto-typed request (`routing_pb2` + `json_format`, snake_case), per-call Databricks OAuth refresh off-thread, 4000-char prompt cap, `last_error` surfacing | `omnigent/server/smart_routing.py:378-587` |
| Routing proto — **already has** `RouteSelector.config` (Struct), `SessionHistory`, `RouteSelection.params`; client just doesn't populate them | `omnigent/api/routing/v1/routing.proto` |
| Local LLM-judge router (built INSTEAD of external when `provider != external`) | `smart_routing.py:250`; `cli.py:107,166,3420` |
| Pluggable router | `omnigent/runtime/caps.py:80` (`routing_client`) |
| Auto-harness session routing | `smart_routing.py:678` ← `orchestration.py:3625` |
| Omnigent-spawned child routing via parent catalog | `orchestration.py:3717` (`catalog_session_id=parent`) |
| Per-turn routing (`cost_control_mode_override == "on"`; native `/model` injection) | `smart_routing.py:823` ← `orchestration.py:3748, 3995-4012` |
| Post-verdict harness correction | `smart_routing.py:636-675` (`_HARNESS_EXCLUDED_MODELS`, `_redirect_incompatible_pick`) |
| Live catalog fetch (server→runner `/v1/sessions/{id}/models`) | `smart_routing.py:122` (`fetch_runner_models`) |
| Deterministic model-override plumbing (native `--model`, SDK `HARNESS_<H>_MODEL`, `model_family_mismatch` guard) | `omnigent/model_override.py:30,112,254` |
| `sys_session_send` child model: **create-time-only** `args.model`, family-guarded, → `model_override` | `omnigent/tools/builtins/spawn.py:1562-1662,1778-1781` |
| Routing-decision transcript item — fields today: `model`, `applied`, `rationale`, `agent` | `omnigent/entities/conversation.py:512-560` |
| UI decision rendering: `RoutingDecisionChip` (turn) + `RoutingDecisionCard` (session) | `web/src/components/blocks/StatusBlocks.tsx:130,172` |
| UI Smart Routing sentinel, hard-disabled for Auto harness | `HarnessConfigControls.tsx:17`, `NewChatDialog.tsx:2165` |
| Claude-native hook provisioning — settings dict built in code, passed via `--settings`; **a deny-capable PreToolUse policy hook already exists** (AskUserQuestion matcher + policy eval when `ap_server_url` set) | `claude_native_bridge.py:1118-1364` (`build_hook_settings`), `:1322-1325` |
| Runner-local HTTP endpoint pattern for hooks: tool relay on `127.0.0.1:0`, bearer token advertised via `tool_relay.json` in the bridge dir, discovery env vars | `claude_native_bridge.py:3291,3317,3326`; env `HARNESS_CLAUDE_NATIVE_BRIDGE_DIR` / `..._REQUEST_SESSION_ID` (`:804-805`) |
| Codex per-session private home: `auth.json` symlinked, `hooks.json` **symlinked from the user's** `~/.codex`, `config.toml` copied | `omnigent/inner/codex_executor.py:112-120,656,760,1380` |
| Codex hook-trust groundwork **already present**: `_CODEX_BYPASS_HOOK_TRUST_FLAG` + min-version gates (policy hooks ≥0.129.0, bypass-trust ≥0.131.0) — defined, not yet applied to omnigent-launched argv | `omnigent/inner/codex_native_app_server.py:89-94,1877` |
| Fork detection (session-level: `/fork`, `/branch`, `forkedFrom` markers) | `claude_native_forwarder.py:233,2311,2457`; `claude_native_bridge.py:1716` |
| Existing hook-script precedents (no shared dir yet) | `omnigent/inner/cursor_policy_hook.py`, `omnigent/inner/hermes_policy_hook.py` |
| Telemetry: `omnigent` tracer, `span()` ctx manager, `record_llm_usage`/`record_error`; **no event-emission helper yet** | `omnigent/runtime/telemetry.py:598,632,789,816` |

**Gaps this plan closes:** (G1) native in-harness subagents are unrouted —
Claude SDK executor registers **no** hooks today, Claude-native's PreToolUse
hook doesn't cover Task spawns, Codex symlinks the user's `hooks.json`
untouched; (G2) router-contract knowledge is smeared across the client
(static `MODEL_LISTS`, post-correction) instead of the §2 seam, and
`route_selector.config` is never sent; (G3) no routing telemetry, and
`RoutingDecisionData` lacks harness/scope/decision identity; (G4) decisions
are not visible per-subagent — `ChildSessionInfo`
(`web/src/hooks/useChildSessions.ts:25-54`, fed by
`GET /v1/sessions/{id}/child_sessions`) carries **no model field at all**, and
there is no generic session-header warning banner to surface enforcement
state.

Enforcement primitives verified externally (2026-07-28 research reports):

- **Claude Code**: `PreToolUse` hook on the Agent/Task tool can deny **and**
  rewrite `tool_input` (incl. `model`) via `hookSpecificOutput.updatedInput` +
  `permissionDecision: "allow"`. Settings-level hooks recurse to nested
  subagents. Works in CLI and Agent SDK (SDK: in-process hook callbacks).
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
  `uvx pre-commit run --files <owned files>` + its own test targets
  (`uv run pytest <paths>`; `cd web && npx vitest run <paths>` — the web
  suite is **vitest**, not jest) before committing to `routing-mvp`.
- Commit per packet (small, revertable), all on `routing-mvp`.

### Wave 1 — six packets, fully parallel

**P1 — Route-options seam + config + default-on AIGW router** *(req 1, G2)*
Owns: `omnigent/server/smart_routing.py`, `omnigent/cli.py` (routing-client
build region ~107-166 only), `tests/server/test_smart_routing.py`.
- Introduce `RouteOptionSource` per §2; move `_HARNESS_EXCLUDED_MODELS` /
  `_redirect_incompatible_pick` / `MODEL_LISTS` behind it (task_v1 scenario
  menus per §1, config-overridable via `routing.scenario_menus`, keyed by
  router version; default `routing.router_name` = `task_v1`).
- Map unservable picks to the nearest servable catalog id; keep the raw pick
  for the decision payload (UI shows what the router said).
- Populate `route_selector.config.model` from new `routing.selection_model`
  (proto field already exists — zero proto work); keep raw prompt + existing
  4000-char cap.
- **Owns ALL new `routing.*` config parsing** (including P2's
  `subagent_fail_mode` and cache TTL) into the frozen `RoutingSettings`
  dataclass (§5.4) hung on `RuntimeCaps` — other packets read the dataclass,
  never cli.py. This is what keeps cli.py single-owner.
- Default-on: when the server's provider is Databricks (`kind: databricks`)
  and no `routing:` block exists, synthesize the external client against that
  workspace's `/ai-gateway/routing/v1` with
  `model_prefixes=["databricks-", "system.ai."]` and profile auth. Explicit
  `routing:` config always wins; `routing.provider: none` opts out.
- Failure semantics unchanged for sessions/turns: `None` + `last_error`,
  caller proceeds unrouted with the reason (§2) — do NOT invent a runtime
  LLM-judge chain in this packet.

**P2 — Runner `route-subagent` endpoint** *(req 2 backbone)*
Owns: new `omnigent/runner/subagent_routing.py`, its wiring into the runner's
existing local HTTP surface, new `tests/server/test_subagent_routing.py`.
- **Follow the existing tool-relay pattern** (`claude_native_bridge.py:3291`):
  loopback HTTP on `127.0.0.1:0`, bearer token + URL advertised via a JSON
  file in the session bridge dir — do not invent a new auth scheme. Serves
  the §5.1 contract.
- Reaches the server's `RuntimeCaps.routing_client` the same way live
  catalogs flow today (the runner↔server hop behind
  `fetch_runner_models`, `smart_routing.py:122`) — P2 adds the inverse
  relay route next to the existing `/v1/sessions/{id}/models` handler.
- Policy lives server-side so hook scripts stay dumb: `fork=true` → `allow`
  unchanged (v1: don't route forks); router unreachable →
  `RoutingSettings.subagent_fail_mode` (`open`=allow unchanged [default],
  `closed`=deny) — pilot runs `closed`.
- Caches per (session, task-hash) with TTL to keep the blocking spawn path
  fast; cache retains per-session picks shaped for future `session_history`.
- Persists every decision as a transcript item via the §5.2 shape.
- Tests with `_FakeRoutingClient` + the `_caps` patch pattern
  (`tests/server/test_smart_routing.py:62,305-317`): rewrite / redirect /
  deny / fork-exempt / outage-open / outage-closed paths.

**P3 — Claude PreToolUse router hook (native + SDK)** *(req 2)*
Owns: `omnigent/claude_native_bridge.py` (the `build_hook_settings` region
only), new `omnigent/inner/hook_scripts/` dir (create it) + claude router
hook script, `omnigent/inner/claude_sdk_executor.py` (hook registration
only), matching tests.
- Native: extend `build_hook_settings` (`:1118-1364`) with a `PreToolUse`
  entry matching the Agent/Task tool — **model on the existing deny-capable
  PreToolUse policy-hook entry at `:1322-1325`** and the
  `cursor_policy_hook.py` / `hermes_policy_hook.py` script precedents.
  Script discovers the P2 endpoint via the bridge-dir advertisement file
  (same discovery as `tool_relay.json`; bridge dir comes in on
  `HARNESS_CLAUDE_NATIVE_BRIDGE_DIR` / `--bridge-dir` argv). Stdlib-only,
  fast — it blocks spawns.
- SDK: `claude_sdk_executor.py` registers **no hooks today** (verified) —
  add the same decision logic as an in-process `claude-agent-sdk`
  `PreToolUse` callback; no subprocess.
- Decision mapping per §5.1: `rewrite` → allow + `updatedInput` with routed
  `model`; `redirect` → deny with reason
  `"Router selected <harness>/<model>. Use sys_session_send with args.harness=…, args.model=… instead."`;
  `deny` → deny with router reason. Fork-typed spawns send `fork=true`.
- Until P7 wires the live endpoint, develop against a stub honoring §5.1.
- Tests: fixture hook payloads → exact hook JSON out; settings-generation
  snapshot (extend existing `build_hook_settings` tests); recursion note
  (settings hooks apply to nested Task spawns).

**P4 — Codex hooks.json generation + trust + canary** *(req 2)*
Owns: `omnigent/inner/codex_executor.py`, `omnigent/inner/codex_native_app_server.py`
(argv/trust region only), codex router hook script in
`omnigent/inner/hook_scripts/`, matching tests.
- Stop symlinking the user's `hooks.json` when routing is on
  (`codex_executor.py:113,760`); generate a merged file: user hooks +
  Omnigent `PreToolUse` matcher `.*spawn_agent` (regex — flattened name is
  `collaborationspawn_agent` on 0.145.x; re-check inside the script),
  generous timeout, + `SessionStart` canary (touch file in bridge dir) +
  `SubagentStart` audit writer (`agent_id`, `model` → bridge dir).
- Trust: **the flag and version gates already exist** —
  `_CODEX_BYPASS_HOOK_TRUST_FLAG` and the ≥0.129/≥0.131 constants
  (`codex_native_app_server.py:89-94,1877`); apply the flag to
  omnigent-launched codex argv when the generated hooks file is in play,
  behind the existing version check. Document the managed
  `requirements.toml` path for fleet later. Canary absent after launch →
  emit the §5.3 warning event instead of failing open silently.
- Codex constraints: spawn `message` is encrypted — pass through verbatim on
  rewrite, route on `task_name`/parent-model/metadata only; injected `model`
  must come from the harness's spawn-eligible set or the handler errors.
- Tests: hooks.json merge snapshot (user hooks preserved), canary detection,
  audit-file parsing, argv assembly incl. version-gated flag.

**P5 — Decision data model + child-sessions API + telemetry** *(reqs 3+4 backbone)*
Owns: `omnigent/entities/conversation.py` (`RoutingDecisionData` only), the
`GET /v1/sessions/{id}/child_sessions` handler (routed-model field addition),
a new `emit_routing_event` helper in `omnigent/runtime/telemetry.py`, matching
tests.
- Extend `RoutingDecisionData` per §5.2 — additive, defaulted (current fields
  are exactly `model/applied/rationale/agent`; legacy rows must deserialize).
- Add `routed_model` (+ `routing_decision_id`) to the child-sessions API
  payload so the sidebar can render per-subagent models — **this is the
  server half of G4; P6 must not need server edits.**
- Telemetry: `telemetry.py` has span helpers but **no event API** (verified)
  — add one `emit_routing_event(name, attrs)` helper (span-event or log-record
  based, matching `tests/runtime/test_telemetry_logs.py` conventions) and
  emit `omnigent.routing.decision` / `.enabled` / `.disabled_mid_session` /
  `.fork_from_routed_session`.
- Tests: serialization round-trip incl. legacy rows; child-sessions payload;
  event emission asserted via `InMemorySpanExporter`
  (pattern: `tests/inner/test_tracing_genai_semconv.py`).

**P6 — Web UI: per-subagent visibility + warning banner + toggle telemetry** *(reqs 3+4)*
Owns: `web/src/` only — `StatusBlocks.tsx`, `Sidebar.tsx`,
`subagentStatus.ts`, `useChildSessions.ts`, `NewChatDialog.tsx`, vitest tests.
- Extend `RoutingDecisionChip`/`RoutingDecisionCard`
  (`StatusBlocks.tsx:130,172`) with harness + scope badge and raw-pick vs
  applied model.
- `useChildSessions.ts`: add the §5.2-mirrored `routed_model` field to
  `ChildSessionInfo`; render it on sidebar child rows (ankit req #1).
  Develops against fixture payloads matching P5's API addition.
- **Build the session-header warning banner** (none exists — verified; the
  closest precedents are `ReconnectSessionDialog` states and the sidebar
  `AlertTriangleIcon` usage) and render §5.3
  `subagent_routing_unenforced` on it.
- Fire switch-off/fork telemetry triggers through the existing event
  plumbing when the user leaves IR mid-session or forks a routed session.
- NewChatDialog: leave the dead `smartRoutingEligible` sentinel untouched
  this wave (design decision with Ajay/Tomu pending — §8).
- Tests: **vitest** (`cd web && npx vitest run src/...`), not jest.

### Wave 2 — integration (start once the relevant Wave-1 packets merge)

**P7 — Hook↔endpoint integration + override precedence** *(needs P2+P3+P4)*
The only packet allowed to touch multiple packets' files.
- Wire the real P2 endpoint advertisement into the generated Claude settings
  and Codex hooks.json (bridge-dir file), replacing the P3/P4 stubs.
- Override precedence: when routing is on, an LLM-supplied `args.model` on
  `sys_session_send` must NOT override the router. Note `args.model` is
  **create-time-only** (`spawn.py:1778`) — the precedence gate goes where
  `model_override` enters `create_body`, and the attempted override lands in
  the decision item (`attempted_override`, §5.2). Add the test.
- `SubagentStart`-vs-decision reconciliation in the codex forwarder;
  mismatch → §5.3 warning event.
- Integration tests: fake router + `ControllableMockClient`
  (`tests/server/conftest.py:177`), extending the
  `tests/server/integration/test_sessions_child_sessions.py` pattern —
  spawn rewritten in-harness, blocked cross-harness with redirect text,
  transcript item present, child-sessions API carries the routed model.

**P8 — Live E2E + probes + version pin** *(needs P7)* — see §6 for the full
test matrix this packet executes.
- Add a codex version pin: `harness_install_spec.py` has `min_version` /
  `max_version_exclusive` fields but **no codex pin today** (verified) — set
  one covering the hook-verified range (≥0.145.0, < next-untested-major).

### Parallelism summary

```
Wave 1 (parallel):  P1   P2   P3   P4   P5   P6
                      \   |  /  \  |   /   |
Wave 2:                P7 (P2+P3+P4[+P1])  P6 finishes against P5 fixtures
Wave 3:                P8 (all)
```

File-ownership conflicts eliminated by construction: cli.py + smart_routing.py
are P1-only (P2 reads `RoutingSettings`, not config); entities + server API
additions are P5-only (P6 consumes fixtures); the two executors split cleanly
(P3 = claude files, P4 = codex files); `omnigent/inner/hook_scripts/` is new
but P3/P4 add disjoint files inside it. P7 is the sole integrator.

---

## 5. Frozen interface contracts (agents code against these, not each other)

### 5.1 `route-subagent` endpoint (P2 serves; P3/P4 consume)

Advertised to hook scripts via a bridge-dir JSON file
(`subagent_router.json`: `{url, token}`), same pattern as `tool_relay.json`.

`POST {url}/v1/sessions/{session_id}/route-subagent` (Bearer token)

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

Today: `model: str`, `applied: bool`, `rationale: str`, `agent: str | None`.
Additive fields: `harness: str | None`,
`scope: Literal["session","turn","child_session","native_subagent"]`,
`decision_id: str | None`, `raw_model: str | None`,
`attempted_override: str | None`. All defaulted for legacy rows.
Child-sessions API mirrors `routed_model: str | null` +
`routing_decision_id: str | null` per child row.

### 5.3 Canary warning event (P4 emits; P6 renders)

Session-scoped warning `subagent_routing_unenforced` with `{harness, reason}`,
delivered on the existing session-status channel.

### 5.4 `RoutingSettings` (P1 defines & parses; P2 reads)

Frozen dataclass on `RuntimeCaps`:

```python
@dataclass(frozen=True)
class RoutingSettings:
    router_name: str = "task_v1"
    selection_model: str | None = None          # -> route_selector.config.model
    scenario_menus: Mapping[str, Mapping[str, tuple[str, ...]]] = TASK_V1_MENUS
    subagent_fail_mode: Literal["open", "closed"] = "open"
    subagent_cache_ttl_s: float = 300.0
```

---

## 6. Testing plan

Layered; each layer names its runner and when it gates.

**L1 — per-packet unit (gates every commit).**
`uv run pytest tests/server/test_smart_routing.py tests/server/test_subagent_routing.py <packet tests>`
and `cd web && npx vitest run src/...`. Baseline to protect: the existing 47
tests in `test_smart_routing.py` (they lock prefix round-tripping, worker-name
mapping, redirect correction, last_error surfacing) must stay green through
the P1 refactor — they are the regression net for the seam extraction.

**L2 — router contract fixtures (gates P1).** Freeze §1 as recorded
request/response fixtures and assert the client against them: (a) codex-only
harness set → request contains exactly the Codex arm menu (+catalog extras);
(b) claude-only → Claude arms; (c) mixed → all five; (d) 400 "requires its
full menu" → `None` + `last_error`, session proceeds unrouted; (e) pick of an
endpoint-less arm (`gpt-5-6-sol`) → resolved to a servable id with `raw_model`
preserved; (f) echoed nonsense harness ignored (harness derived from arm
family); (g) `route_selector.config.model` present iff `selection_model` set.

**L3 — live contract probe (manual/CI-cron, not commit-gating).**
`scripts/probe_routing_api.sh` — the recorded curl battery from 2026-07-28/29
(scenario inference, full-menu 400s, extras tolerated, tag passthrough)
against eng-ml-inference staging via `databricks auth token`. Run before
demos and whenever AIGW deploys; alerts us if a task_v2 lands or menus move.

**L4 — hook-layer unit (gates P3/P4).** Hook scripts are pure functions
around the §5.1 call: fixture stdin payloads → exact hook JSON out for
allow/rewrite/redirect/deny × fork × endpoint-down (×fail_mode). Codex
additionally: merged-hooks.json snapshot preserving user hooks; canary
present/absent; audit-record parsing; version-gated argv flag.

**L5 — server integration with fake router (gates P7).** Boot the test
server (`ControllableMockClient` + `_caps` patch), drive a session:
1. Omnigent child spawn (`sys_session_send`) → router decision wins over
   `args.model`, attempted override recorded.
2. Native-subagent decision via the P2 endpoint → transcript
   `RoutingDecisionData(scope="native_subagent")`, child-sessions API row
   carries `routed_model`.
3. `subagent_fail_mode=closed` + dead router → deny; `=open` → allow
   unchanged; both leave a decision item.
4. Decision cache: two identical spawns, one router call.

**L6 — live-harness E2E (gates P8; needs real claude/codex CLIs).**
Conventions from `tests/e2e/test_polly_subagent_model_e2e.py`:
- Claude native: session with router hook → Task spawn's model rewritten
  (assert via hooks.jsonl SubagentStart mirror).
- Claude SDK: same via in-process callback.
- Codex: `spawn_agent` rewritten; `SubagentStart` payload model ==
  decision_id's model (audit reconciliation); canary fires; with hooks
  untrusted and bypass flag stripped → canary absent → warning event.
- Cross-harness: deny+redirect reason emitted; model follows with
  `sys_session_send` at least once (track follow rate, don't hard-assert).
- Standalone `scripts/probe_codex_hooks.py` (deny + rewrite smoke) —
  rerun on every codex version bump; paired with the P8 version pin.

**L7 — manual CUJ pass on the dev stack (release gate, ~30 min).**
Stack: `./run-server.sh` / `./run-host.sh` / `./run-frontend.sh` in
`~/omnigent-routing-mvp` (ports 6868/5273, isolated config, staging AIGW,
`task_v1`). Checklist mirrors the brainstorm CUJs:
1. **Codex CUJ**: codex harness + IR on → server log shows Codex-arm-menu
   request; picked model applied; subagent spawn shows decision in
   transcript + sidebar.
2. **Claude Code CUJ**: same with claude; verify `/model` injection on a
   routed turn.
3. **Auto CUJ**: Polly + AUTO gear → harness+model pick lands; Omnigent
   child sessions routed via parent catalog; cross-harness redirect visible.
4. **Visibility**: every decision has a chip/card with rationale; per-subagent
   model in sidebar; kill the router mid-session → fail-mode behavior +
   warning banner.
5. **Telemetry**: `omnigent.routing.*` events visible in OTel export
   (`OTEL_EXPORTER_OTLP_ENDPOINT` set); switch IR off mid-session and fork a
   routed session → both events present.
6. **Isolation regression**: user-level `~/.omnigent` untouched (config home
   + data dir remain worktree-local).

**L8 — full-suite regression (before PR split).**
`uv run pytest tests/server tests/runtime tests/inner` +
`cd web && npx vitest run` + `uvx pre-commit run --all-files`.

---

## 7. MVP definition of done (Jul 31)

- AIGW `routes:select` drives session/turn/child routing by default on
  Databricks-backed deployments (P1) using `task_v1`, surviving the confirmed
  contract quirks: fixed scenario menus, passthrough harness tags, unservable
  picks. L2 fixtures green; L3 probe clean against staging.
- A Claude Task spawn and a Codex `spawn_agent` cannot proceed on a
  non-router-approved model; failure mode is "didn't spawn", never "wrong
  model" (P2–P4, P7). L5/L6 green.
- Every decision renders in the transcript; per-subagent routed model in the
  sidebar; warning banner when enforcement is off (P5, P6).
- `omnigent.routing.*` OTel events for decision / enabled / switched-off /
  fork (P5, P6).
- L7 manual CUJ pass recorded (notes or screen capture) on the live dev stack.

## 8. Outside this branch / open items

- **SAFE flag** for isaac default-on cohort — universe repo, after this branch
  is testable.
- **NewChatDialog reconciliation** (Auto harness vs per-harness IR toggle,
  naming "Intelligent Routing") — blocked on Ajay/Tomu design call.
- **v3 gateway model-listing API** — explicitly not-MVP per the checklist.
- **Feed to Mason**: (a) harness-constrained returns (client correction is a
  stopgap — §1); (b) spawn-eligible model subsets per harness (Codex spawns
  support fewer models than sessions); (c) menu ids should not be required
  when absent from the caller's workspace.
- **Feed to Ivan**: encrypted Codex spawn prompts cap task-aware quality for
  in-harness codex subagent routing (signal lives in `task_name` only).

## 9. Risks

1. **Codex hook fragility** (tool-name flattening, trust gate, v1/v2
   selection): regex matcher + canary + new version pin + L6 standalone
   probe; budget breakage on bumps. Groundwork (bypass flag + version gates)
   already exists in `codex_native_app_server.py`.
2. **Router latency blocks spawns**: task_v1 always makes an LLM extraction
   self-call (§1.1), so p99 sits on the PreToolUse path; mitigated by P2
   caching + fail-mode knob; measure in L6.
3. **Redirect compliance is soft**: deny+redirect relies on the model calling
   `sys_session_send`; worst case non-spawn, never wrong-model. Track
   redirect-follow rate via decision items during the pilot.
4. **Scenario-menu drift**: routers are frozen per version (§1.1), so menus
   only change when we bump `routing.router_name` — `scenario_menus` keyed by
   router version keeps them moving together; L3 probe catches server-side
   surprises; P1 degrades to unrouted-with-warning on mismatch.
5. **Fork/cache-miss economics**: v1 doesn't route forks; a real cost model
   needs inherited-context size that only Omnigent can supply out-of-band.
6. **Extraction-model access**: task_v1's self-call needs the caller to have
   `system.ai.gpt-5-4-mini` (or `routing.selection_model` pinned to one they
   do have); a workspace without it breaks routing invisibly — L3 probe +
   `last_error` surfacing cover it.
