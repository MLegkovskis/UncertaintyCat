import { createServer } from "node:http";

const host = "127.0.0.1";
const port = 8790;
const localKey = "e2e-local-only";

const responseObject = {
  equations: [
    {
      latex: String.raw`\mathbf{y}=f(\mathbf{x})`,
      limitation:
        "This formal mapping is used because the fixture does not reinterpret private model source.",
    },
  ],
  modelOverview:
    "The authenticated model maps the supplied uncertain inputs to one validated output through the retained OpenTURNS function.",
  inputUncertainty: [
    "Each input uses the distribution and parameters retained by isolated validation.",
  ],
  dependenceAndPropagation:
    "Dependence follows the validated copula metadata supplied with the model.",
  validatedPilotBehaviour:
    "The bounded pilot executed successfully; its retained facts remain the numerical authority.",
  questionsToConfirm: ["Which physical units apply to the model variables?"],
};

function json(response, status, body) {
  response.writeHead(status, { "content-type": "application/json" });
  response.end(JSON.stringify(body));
}

const server = createServer((request, response) => {
  if (request.method === "GET" && request.url === "/health") {
    json(response, 200, { ok: true });
    return;
  }
  if (
    request.method !== "POST" ||
    request.url !== "/openai/v1/chat/completions"
  ) {
    json(response, 404, { error: { message: "Not found" } });
    return;
  }
  if (request.headers.authorization !== `Bearer ${localKey}`) {
    json(response, 401, { error: { message: "Unauthorized" } });
    return;
  }

  let raw = "";
  request.setEncoding("utf8");
  request.on("data", (chunk) => {
    raw += chunk;
  });
  request.on("end", () => {
    try {
      const body = JSON.parse(raw);
      const jsonSchema = body.response_format?.json_schema;
      if (
        body.response_format?.type !== "json_schema" ||
        jsonSchema?.strict !== true ||
        jsonSchema?.schema?.additionalProperties !== false
      ) {
        json(response, 422, {
          error: { message: "Expected the strict Model Understanding schema" },
        });
        return;
      }
      json(response, 200, {
        id: "chatcmpl-uncertaintycat-e2e",
        object: "chat.completion",
        created: 0,
        model: body.model,
        choices: [
          {
            index: 0,
            finish_reason: "stop",
            message: {
              role: "assistant",
              content: JSON.stringify(responseObject),
            },
          },
        ],
        usage: {
          prompt_tokens: 1,
          completion_tokens: 1,
          total_tokens: 2,
        },
      });
    } catch {
      json(response, 400, { error: { message: "Invalid JSON" } });
    }
  });
});

server.listen(port, host);

for (const signal of ["SIGINT", "SIGTERM"]) {
  process.on(signal, () => server.close(() => process.exit(0)));
}
