import { readFileSync } from "node:fs";
import { DatabaseSync } from "node:sqlite";

import { describe, expect, it } from "vitest";
import { getTableConfig } from "drizzle-orm/sqlite-core";

import { account } from "./auth-schema";

describe("Better Auth account identity schema", () => {
  it("retains legacy issuers without requiring new inserts and scopes provider subjects", () => {
    const config = getTableConfig(account);
    const issuer = config.columns.find((column) => column.name === "issuer");
    const identityIndex = config.indexes.find(
      (index) => index.config.name === "account_provider_account_uidx",
    );

    expect(issuer?.notNull).toBe(false);
    expect(identityIndex?.config.unique).toBe(true);
    expect(
      identityIndex?.config.columns.map((column) =>
        "name" in column ? column.name : undefined,
      ),
    ).toEqual(["providerId", "accountId"]);
    expect(
      config.indexes.some(
        (index) => index.config.name === "account_issuer_accountId_uidx",
      ),
    ).toBe(false);
  });

  it("migrates populated accounts without losing their identity or requiring new issuers", () => {
    const db = new DatabaseSync(":memory:");
    const migration = (name: string) =>
      readFileSync(new URL(`../migrations/${name}`, import.meta.url), "utf8");
    try {
      db.exec(migration("0001_initial.sql"));
      db.exec(migration("0002_auth_account_issuer.sql"));
      db.exec(
        "INSERT INTO user (id, name, email, createdAt, updatedAt) VALUES ('user-1', 'Test', 'test@example.com', 1, 1)",
      );
      db.exec(
        "INSERT INTO account (id, issuer, accountId, providerId, userId, createdAt, updatedAt) VALUES ('account-1', 'https://example.com/issuer', 'subject-1', 'cloudflare', 'user-1', 1, 1)",
      );

      db.exec(migration("0008_auth_account_issuer_compat.sql"));

      expect(
        db.prepare("SELECT issuer, providerId, accountId, userId FROM account WHERE id = 'account-1'").get(),
      ).toMatchObject({
        issuer: "https://example.com/issuer",
        providerId: "cloudflare",
        accountId: "subject-1",
        userId: "user-1",
      });
      expect(
        db.prepare("PRAGMA table_info(account)").all().find((column) => column.name === "issuer"),
      ).toMatchObject({ notnull: 0 });
      expect(
        db.prepare("PRAGMA index_list(account)").all().map((index) => index.name),
      ).toContain("account_provider_account_uidx");
      expect(
        db.prepare("PRAGMA index_list(account)").all().map((index) => index.name),
      ).not.toContain("account_issuer_accountId_uidx");
      db.exec(
        "INSERT INTO account (id, accountId, providerId, userId, createdAt, updatedAt) VALUES ('account-2', 'subject-2', 'cloudflare', 'user-1', 2, 2)",
      );
      expect(
        db.prepare("SELECT issuer FROM account WHERE id = 'account-2'").get(),
      ).toMatchObject({ issuer: null });
      expect(() =>
        db.exec(
          "INSERT INTO account (id, accountId, providerId, userId, createdAt, updatedAt) VALUES ('account-3', 'subject-1', 'cloudflare', 'user-1', 3, 3)",
        ),
      ).toThrow();
      expect(db.prepare("PRAGMA foreign_key_check").all()).toEqual([]);
    } finally {
      db.close();
    }
  });
});
