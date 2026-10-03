// Pi extension for the dsv41-sparks provider (DeepSeek-V4.1-Flash on two DGX Sparks).
// Pi always sends an output-token limit (model maxTokens, clamped to its own context estimate). This server has no
// generation cap: requests end at EOS or when the client cancels. Remove the limit fields for this provider only,
// so a long answer is never cut off and a low context estimate can never push prompt + limit past 262,144.
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";

export default function (pi: ExtensionAPI) {
  pi.on("before_provider_request", (event, ctx) => {
    if (ctx.model?.provider !== "dsv41-sparks") return;
    const body = { ...(event.payload as Record<string, unknown>) };
    delete body.max_tokens;
    delete body.max_completion_tokens;
    return body;
  });
}
