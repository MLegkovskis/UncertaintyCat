import { createHash } from "node:crypto";
import { readdirSync, readFileSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const repositoryRoot = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const examplesDirectory = join(repositoryRoot, "examples");
const assetsDirectory = join(repositoryRoot, "apps", "web", "dist", "assets");

const publicAssets = readdirSync(assetsDirectory)
  .filter((name) => name.endsWith(".js") || name.endsWith(".js.map"))
  .map((name) => ({
    name,
    content: readFileSync(join(assetsDirectory, name), "utf8"),
  }));

const leakedExamples = [];
for (const filename of readdirSync(examplesDirectory).filter((name) => name.endsWith(".py"))) {
  const source = readFileSync(join(examplesDirectory, filename), "utf8");
  const hash = createHash("sha256").update(source).digest("hex");
  const sourceMarkers = source
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter((line) => line.length >= 48)
    .sort((left, right) => right.length - left.length)
    .slice(0, 5);
  const leakedBy = publicAssets.find(
    ({ content }) => content.includes(hash) || sourceMarkers.some((marker) => content.includes(marker)),
  );
  if (leakedBy) leakedExamples.push(`${filename} in ${leakedBy.name}`);
}

if (leakedExamples.length) {
  throw new Error(
    `Authenticated example source leaked into public web assets:\n${leakedExamples.join("\n")}`,
  );
}

console.log("Public web bundle contains no canonical authenticated example source.");
