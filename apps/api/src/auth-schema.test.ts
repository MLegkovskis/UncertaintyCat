import { describe, expect, it } from "vitest";
import { getTableConfig } from "drizzle-orm/sqlite-core";

import { account } from "./auth-schema";

describe("Better Auth account identity schema", () => {
  it("stores a required issuer and uniquely scopes each provider subject", () => {
    const config = getTableConfig(account);
    const issuer = config.columns.find((column) => column.name === "issuer");
    const identityIndex = config.indexes.find(
      (index) => index.config.name === "account_issuer_accountId_uidx",
    );

    expect(issuer?.notNull).toBe(true);
    expect(identityIndex?.config.unique).toBe(true);
    expect(
      identityIndex?.config.columns.map((column) =>
        "name" in column ? column.name : undefined,
      ),
    ).toEqual(["issuer", "accountId"]);
  });
});
