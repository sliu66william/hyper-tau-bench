
---

## Runtime facts

These hold in local tests and in evaluation.

### Model call arguments

`model_gateway.generate` accepts these arguments besides the model's own
constraints: `frequency_penalty`, `max_completion_tokens`, `max_tokens`,
`modalities`, `presence_penalty`, `reasoning_effort`, `response_format`,
`seed`, `stop`, `temperature`, `thinking` and `top_p`. Any other argument,
`parallel_tool_calls` included, is rejected and the call fails. They are not
passed through to the provider.

### Tool results

After a turn with one tool call, `generate_next_message` receives a plain
`ToolMessage`. After a turn with several, it receives a `MultiToolMessage`
holding one `ToolMessage` per call. Handle both.

### Conversations that end early

A conversation ends with a score of 0 when its tools have returned five errors,
or when it reaches 200 steps (messages from the agent, the customer and the
tools together). Local tests default to 100 steps (configurable up to 200).

### Sealed runtime

The agent runs without network access and with a read-only workspace. Each
request to the agent may take at most 300 seconds, 32 model calls and 256
Client API calls; a request that exceeds a limit fails.

### Conversation transfers

The conversation transfer endpoint accepts only a `summary` string. When the
domain's operating materials define structured transfer reason codes, include
the selected reason code verbatim in `summary`.

