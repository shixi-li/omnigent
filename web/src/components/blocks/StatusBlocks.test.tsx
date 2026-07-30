import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it } from "vitest";
import { RoutingDecisionCard, RoutingDecisionChip } from "./StatusBlocks";

afterEach(cleanup);

describe("RoutingDecisionChip — smart routing", () => {
  it("applied verdict: names the active model with its tier, plus the rationale line", () => {
    render(
      <RoutingDecisionChip
        model="databricks-claude-opus-4-8"
        applied
        rationale="multi-file refactor needs deep reasoning"
      />,
    );
    const chip = screen.getByTestId("routing-decision-chip");
    // Visible without hovering: the short model name + tier render in the
    // chip text. A missing model/tier would mean the verdict didn't thread
    // through the block pipeline.
    expect(chip).toHaveTextContent("Smart routing");
    expect(chip).toHaveTextContent("opus");
    expect(chip.textContent).not.toContain("(expensive)");
    // The rationale shows as a muted second line (not hover-only).
    expect(chip).toHaveTextContent("multi-file refactor needs deep reasoning");
    expect(chip.getAttribute("data-applied")).toBe("true");
    // No hover required: the rationale is in the rendered DOM, not just title.
    expect(chip.querySelector("[data-testid]")).toBeNull();
  });

  it("shadow verdict: reads 'would have picked' instead of naming the active model", () => {
    render(
      <RoutingDecisionChip
        model="databricks-claude-haiku-4-5"
        applied={false}
        rationale="trivial question"
      />,
    );
    const chip = screen.getByTestId("routing-decision-chip");
    // applied=false → "would have picked" framing; a flip to the applied
    // copy would falsely claim the brain ran on the router's pick.
    expect(chip).toHaveTextContent("would have picked");
    expect(chip).toHaveTextContent("haiku");
    expect(chip.getAttribute("data-applied")).toBe("false");
  });

  it("renders nothing for the rationale line when rationale is empty", () => {
    render(<RoutingDecisionChip model="databricks-claude-sonnet-4-6" applied rationale="" />);
    const chip = screen.getByTestId("routing-decision-chip");
    // Empty rationale still renders the primary line, just no second line —
    // a stray empty <span> would add visual noise to the transcript.
    expect(chip).toHaveTextContent("sonnet");
    expect(chip.textContent).not.toContain("(medium)");
  });

  it("never uses the old 'model control' vocabulary (rename sweep)", () => {
    render(<RoutingDecisionChip model="databricks-claude-opus-4-8" applied rationale="x" />);
    const chip = screen.getByTestId("routing-decision-chip");
    // The feature was renamed from "Intelligent model control"; the chip
    // must carry the new name and never the retired one.
    expect(chip.textContent).toContain("Smart routing");
    expect(chip.textContent).not.toContain("model control");
    expect(chip.textContent).not.toContain("Model Control");
  });
});

describe("RoutingDecisionCard — session-level auto-routing", () => {
  it("applied verdict: shows model pill with tier and rationale", () => {
    render(
      <RoutingDecisionCard
        model="databricks-claude-opus-4-8"
        applied
        rationale="Multi-file refactor needs deep reasoning."
      />,
    );
    const card = screen.getByTestId("routing-decision-card");
    expect(card).toHaveTextContent("Smart routing");
    expect(card).toHaveTextContent("· applied");
    expect(card).toHaveTextContent("Session");
    expect(card).toHaveTextContent("opus");
    expect(card).toHaveTextContent("Multi-file refactor needs deep reasoning.");
    expect(card.getAttribute("data-applied")).toBe("true");
  });

  it("advisory verdict: shows '· advisory' and the model that would have been picked", () => {
    render(
      <RoutingDecisionCard
        model="databricks-claude-haiku-4-5"
        applied={false}
        rationale="Trivial question."
      />,
    );
    const card = screen.getByTestId("routing-decision-card");
    expect(card).toHaveTextContent("· advisory");
    expect(card).toHaveTextContent("haiku");
    expect(card.getAttribute("data-applied")).toBe("false");
  });

  it("shows agent name as row label when mirrored into parent session", () => {
    render(
      <RoutingDecisionCard
        model="databricks-claude-haiku-4-5"
        applied
        rationale="Simple task."
        agent="claude_code"
      />,
    );
    const card = screen.getByTestId("routing-decision-card");
    // The agent name replaces the generic "Session" label so the
    // orchestrator's transcript identifies which sub-agent was routed.
    expect(card).toHaveTextContent("claude_code");
    expect(card.textContent).not.toContain("Session");
  });

  it("expands raw verdict JSON behind the chevron", () => {
    render(
      <RoutingDecisionCard
        model="databricks-claude-opus-4-8"
        applied
        rationale="Deep reasoning required."
      />,
    );
    // Collapsed by default — raw JSON not visible.
    expect(screen.queryByText(/"rationale"/)).toBeNull();
    fireEvent.click(screen.getByTestId("routing-decision-raw-toggle"));
    expect(screen.getByText(/"rationale"/)).toBeInTheDocument();
  });
});

describe("routing decision — harness / scope / raw pick", () => {
  it("chip: renders harness and the sub-agent scope badge", () => {
    render(
      <RoutingDecisionChip
        model="databricks-claude-sonnet-5"
        applied
        rationale="short task"
        agent="researcher"
        routing={{ harness: "claude-native", scope: "native_subagent" }}
      />,
    );
    const chip = screen.getByTestId("routing-decision-chip");
    // Harness + which sub-agent the decision covers: without them a
    // native-subagent decision is indistinguishable from a session one.
    expect(chip).toHaveTextContent("claude-native");
    expect(screen.getByTestId("routing-decision-scope")).toHaveTextContent("subagent: researcher");
  });

  it("chip: session/turn scopes get no sub-agent badge", () => {
    render(
      <RoutingDecisionChip
        model="databricks-claude-sonnet-5"
        applied
        rationale="x"
        routing={{ harness: "codex", scope: "turn" }}
      />,
    );
    expect(screen.queryByTestId("routing-decision-scope")).toBeNull();
  });

  it("chip: shows the raw router pick when it differs from the applied model", () => {
    render(
      <RoutingDecisionChip
        model="databricks-claude-sonnet-5"
        applied
        rationale="x"
        routing={{ rawModel: "gpt-5-6-sol" }}
      />,
    );
    // The router's vocabulary pick had no endpoint and was mapped to a
    // servable id — both must be visible or the mapping is invisible.
    expect(screen.getByTestId("routing-decision-raw-model")).toHaveTextContent("gpt-5-6-sol");
    expect(screen.getByTestId("routing-decision-chip")).toHaveTextContent("sonnet");
  });

  it("chip: hides the raw pick when it resolves to the same short name", () => {
    render(
      <RoutingDecisionChip
        model="databricks-claude-sonnet-5"
        applied
        rationale="x"
        routing={{ rawModel: "claude-sonnet-5" }}
      />,
    );
    expect(screen.queryByTestId("routing-decision-raw-model")).toBeNull();
  });

  it("chip: renders exactly as before when no new field is set", () => {
    render(<RoutingDecisionChip model="databricks-claude-opus-4-8" applied rationale="deep" />);
    const chip = screen.getByTestId("routing-decision-chip");
    expect(chip).toHaveTextContent("Smart routing");
    expect(chip).toHaveTextContent("opus");
    expect(screen.queryByTestId("routing-decision-harness")).toBeNull();
    expect(screen.queryByTestId("routing-decision-scope")).toBeNull();
    expect(screen.queryByTestId("routing-decision-raw-model")).toBeNull();
  });

  it("card: renders harness, scope badge, raw pick, and the extras in the raw JSON", () => {
    render(
      <RoutingDecisionCard
        model="databricks-claude-sonnet-5"
        applied
        rationale="short task"
        agent="researcher"
        routing={{
          harness: "codex-native",
          scope: "child_session",
          decisionId: "dec_123",
          rawModel: "gpt-5-6-sol",
          attemptedOverride: "databricks-claude-opus-4-8",
        }}
      />,
    );
    expect(screen.getByTestId("routing-decision-harness")).toHaveTextContent("codex-native");
    expect(screen.getByTestId("routing-decision-scope")).toHaveTextContent("subagent: researcher");
    expect(screen.getByTestId("routing-decision-raw-model")).toHaveTextContent("gpt-5-6-sol");
    // Identity + attempted override are audit data — they belong in the
    // expandable verdict, not the glance row.
    fireEvent.click(screen.getByTestId("routing-decision-raw-toggle"));
    expect(screen.getByText(/"decision_id"/)).toBeInTheDocument();
    expect(screen.getByText(/"attempted_override"/)).toBeInTheDocument();
  });

  it("card: omits the new rows and JSON keys when the fields are absent", () => {
    render(
      <RoutingDecisionCard model="databricks-claude-opus-4-8" applied rationale="deep reasoning" />,
    );
    expect(screen.queryByTestId("routing-decision-harness")).toBeNull();
    expect(screen.queryByTestId("routing-decision-scope")).toBeNull();
    expect(screen.queryByTestId("routing-decision-raw-model")).toBeNull();
    fireEvent.click(screen.getByTestId("routing-decision-raw-toggle"));
    expect(screen.queryByText(/"harness"/)).toBeNull();
    expect(screen.queryByText(/"raw_model"/)).toBeNull();
  });
});
