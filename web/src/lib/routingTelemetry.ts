// Client-side intelligent-routing telemetry.
//
// The web app has no bespoke event endpoint — its only telemetry channel is
// the browser OpenTelemetry provider set up in `telemetry.ts`. So the two
// user-initiated routing events (leaving routing mid-session, forking a
// routed session) are emitted as zero-duration spans named to match the
// server's `omnigent.routing.*` events, which lands them in the same OTel
// export the server writes to.
//
// No-op unless `VITE_OTEL_EXPORTER_OTLP_ENDPOINT` is configured: without a
// provider `trace.getTracer` returns the API's no-op tracer.

import { trace } from "@opentelemetry/api";

const TRACER_NAME = "omnigent.web.routing";

/** Event name for "the user switched intelligent routing off mid-session". */
export const ROUTING_DISABLED_MID_SESSION = "omnigent.routing.disabled_mid_session";

/** Event name for "the user forked a session that had routing enabled". */
export const ROUTING_FORK_FROM_ROUTED_SESSION = "omnigent.routing.fork_from_routed_session";

function emit(name: string, attributes: Record<string, string>): void {
  const span = trace.getTracer(TRACER_NAME).startSpan(name);
  span.setAttributes(attributes);
  span.end();
}

/**
 * Record that the user turned intelligent routing off (or back to the spec
 * default) on a session that had it on.
 *
 * @param sessionId - Session being changed, e.g. `"conv_abc123"`.
 * @param mode - The new switch value: `"off"`, or `"default"` for `null`.
 */
export function recordRoutingDisabledMidSession(sessionId: string, mode: "off" | "default"): void {
  emit(ROUTING_DISABLED_MID_SESSION, { "session.id": sessionId, "routing.mode": mode });
}

/**
 * Record that a session with intelligent routing enabled was forked.
 *
 * @param sourceSessionId - Session that was forked, e.g. `"conv_abc123"`.
 * @param forkSessionId - The created clone, e.g. `"conv_def456"`.
 */
export function recordForkFromRoutedSession(sourceSessionId: string, forkSessionId: string): void {
  emit(ROUTING_FORK_FROM_ROUTED_SESSION, {
    "session.id": sourceSessionId,
    "routing.fork_session_id": forkSessionId,
  });
}
