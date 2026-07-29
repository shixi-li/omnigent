// Session-scoped warning strip under the chat header.
//
// Surfaces degraded-but-running conditions the user must know about while
// the session keeps working — today only `subagent_routing_unenforced`,
// published when a harness's router hook never fired, so sub-agent model
// routing is advisory rather than enforced for that harness.

import { AlertTriangleIcon } from "lucide-react";
import { useMemo } from "react";
import { useSession } from "@/hooks/useSession";
import type { SessionWarning } from "@/lib/types";

/** Warning code for "a harness ran without the router hook enforcing picks". */
export const SUBAGENT_ROUTING_UNENFORCED = "subagent_routing_unenforced";

// Codes this banner renders. Unknown codes are ignored rather than shown
// raw, so a future server-side warning can't leak a machine string into
// the header before the UI has copy for it.
const RENDERED_CODES = new Set<string>([SUBAGENT_ROUTING_UNENFORCED]);

/**
 * Session warnings this banner knows how to render, in wire order.
 *
 * @param warnings - The snapshot's warnings, e.g. from `Session.warnings`.
 * @returns Only the renderable entries; empty when there is nothing to show.
 */
export function renderableWarnings(
  warnings: SessionWarning[] | null | undefined,
): SessionWarning[] {
  return (warnings ?? []).filter((warning) => RENDERED_CODES.has(warning.code));
}

/**
 * Live session warnings for a conversation, read off the session snapshot
 * (`GET /v1/sessions/{id}`, shared cache with `chatStore.bindStream`).
 *
 * The server does not publish `warnings` yet — until P7 wires the canary
 * event into the session-status channel this returns an empty list and the
 * banner stays hidden.
 *
 * @param conversationId - Session to read, or `null` to disable.
 * @returns Renderable warnings for that session.
 */
export function useSessionWarnings(conversationId: string | null | undefined): SessionWarning[] {
  const { session } = useSession(conversationId ?? null);
  const warnings = session?.warnings;
  return useMemo(() => renderableWarnings(warnings), [warnings]);
}

/** One line of copy per warning code: headline + optional detail. */
function warningTitle(warning: SessionWarning): string {
  const harness = warning.harness?.trim();
  return harness
    ? `Sub-agent routing isn't enforced on ${harness}`
    : "Sub-agent routing isn't enforced";
}

/**
 * Warning strip for the active session. Renders nothing when the session
 * has no warning the UI knows about — the common case — so the header keeps
 * its current layout.
 *
 * @param warnings - The session snapshot's warnings.
 */
export function SessionWarningBanner({ warnings }: { warnings?: SessionWarning[] | null }) {
  const visible = renderableWarnings(warnings);
  if (visible.length === 0) return null;
  return (
    <div data-testid="session-warning-banner" className="flex flex-col">
      {visible.map((warning) => (
        <div
          key={`${warning.code}:${warning.harness ?? ""}`}
          data-testid={`session-warning-${warning.code}`}
          className="flex items-start gap-2 border-b border-border bg-warning/10 px-3 py-1.5 text-xs text-foreground"
        >
          <AlertTriangleIcon aria-hidden="true" className="mt-0.5 size-3.5 shrink-0 text-warning" />
          <span className="min-w-0">
            <span className="font-medium">{warningTitle(warning)}</span>
            {warning.reason ? (
              <span className="text-muted-foreground"> · {warning.reason}</span>
            ) : null}
          </span>
        </div>
      ))}
    </div>
  );
}
