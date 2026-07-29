import { describe, expect, it } from "vitest";
import { routingExtras, routingExtrasFromWire, subagentScopeLabel } from "./routingDecision";

describe("routingExtrasFromWire", () => {
  it("maps the snake_case wire fields to camelCase", () => {
    expect(
      routingExtrasFromWire({
        model: "databricks-claude-sonnet-5",
        harness: "codex-native",
        scope: "native_subagent",
        decision_id: "dec_1",
        raw_model: "gpt-5-6-sol",
        attempted_override: "databricks-claude-opus-4-8",
      }),
    ).toEqual({
      harness: "codex-native",
      scope: "native_subagent",
      decisionId: "dec_1",
      rawModel: "gpt-5-6-sol",
      attemptedOverride: "databricks-claude-opus-4-8",
    });
  });

  it("returns nothing for a legacy payload", () => {
    // Rows written before routing grew these fields must not gain keys —
    // the UI branches on presence to stay backward compatible.
    expect(routingExtrasFromWire({ model: "m", applied: true, rationale: "r" })).toEqual({});
  });

  it("drops blank values and unknown scopes", () => {
    expect(
      routingExtrasFromWire({ harness: "", scope: "galaxy", decision_id: null, raw_model: 7 }),
    ).toEqual({});
  });
});

describe("routingExtras", () => {
  it("copies only the set fields", () => {
    expect(routingExtras({ harness: "codex", scope: null, rawModel: "gpt-5-6-sol" })).toEqual({
      harness: "codex",
      rawModel: "gpt-5-6-sol",
    });
  });
});

describe("subagentScopeLabel", () => {
  it("labels sub-agent scopes with the agent name", () => {
    expect(subagentScopeLabel("native_subagent", "researcher")).toBe("subagent: researcher");
    expect(subagentScopeLabel("child_session", "coder")).toBe("subagent: coder");
  });

  it("falls back to a bare label when the agent name is missing", () => {
    expect(subagentScopeLabel("native_subagent", null)).toBe("subagent");
    expect(subagentScopeLabel("child_session", "  ")).toBe("subagent");
  });

  it("returns null for session/turn scopes and for an absent scope", () => {
    expect(subagentScopeLabel("session", "x")).toBeNull();
    expect(subagentScopeLabel("turn", "x")).toBeNull();
    expect(subagentScopeLabel(undefined, "x")).toBeNull();
  });
});
