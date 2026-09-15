-- Better Auth 1.7.3+ identifies accounts by (providerId, accountId) and no
-- longer writes issuer. Preserve the issuer values of existing accounts, but
-- relax the 1.7.0-1.7.2 NOT NULL constraint and remove its obsolete index.
-- SQLite/D1 cannot DROP NOT NULL in place, so rebuild only the account table.
DROP INDEX account_issuer_accountId_uidx;

CREATE TABLE account_nullable_issuer (
  id TEXT PRIMARY KEY,
  issuer TEXT,
  accountId TEXT NOT NULL,
  providerId TEXT NOT NULL,
  userId TEXT NOT NULL REFERENCES user(id) ON DELETE CASCADE,
  accessToken TEXT,
  refreshToken TEXT,
  idToken TEXT,
  accessTokenExpiresAt INTEGER,
  refreshTokenExpiresAt INTEGER,
  scope TEXT,
  password TEXT,
  createdAt INTEGER NOT NULL,
  updatedAt INTEGER NOT NULL
);

INSERT INTO account_nullable_issuer (
  id, issuer, accountId, providerId, userId, accessToken, refreshToken,
  idToken, accessTokenExpiresAt, refreshTokenExpiresAt, scope, password,
  createdAt, updatedAt
)
SELECT
  id, issuer, accountId, providerId, userId, accessToken, refreshToken,
  idToken, accessTokenExpiresAt, refreshTokenExpiresAt, scope, password,
  createdAt, updatedAt
FROM account;

DROP TABLE account;
ALTER TABLE account_nullable_issuer RENAME TO account;

CREATE INDEX account_user_idx ON account(userId);
CREATE UNIQUE INDEX account_provider_account_uidx ON account(providerId, accountId);
