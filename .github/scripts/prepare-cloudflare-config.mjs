import { readFile, writeFile } from "node:fs/promises";
import { resolve } from "node:path";

const databaseId = process.env.CLOUDFLARE_D1_DATABASE_ID?.trim();
if (!databaseId || !/^[0-9a-f-]{36}$/i.test(databaseId)) {
  throw new Error(
    "CLOUDFLARE_D1_DATABASE_ID must be a valid D1 database UUID.",
  );
}

const source = resolve("apps/api/wrangler.production.jsonc");
const target = resolve("apps/api/wrangler.generated.jsonc");
const sourceText = await readFile(source, "utf8");
const config = JSON.parse(sourceText.replace(/,\s*([}\]])/g, "$1"));
config.d1_databases[0].database_id = databaseId;
await writeFile(target, `${JSON.stringify(config, null, 2)}\n`, {
  mode: 0o600,
});
console.log(
  "Prepared production Wrangler configuration with the configured D1 binding.",
);
