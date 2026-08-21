-- Better Auth 1.7 scopes an external account by its trusted issuer and subject.
-- Rebuild the SQLite table so issuer is genuinely NOT NULL (SQLite cannot add
-- that constraint to an existing column). The only configured OAuth provider
-- before this migration is Cloudflare Access; the fallback preserves the
-- documented Better Auth namespace for any plain OAuth rows created locally.
CREATE TABLE account_with_issuer (
  id TEXT PRIMARY KEY,
  issuer TEXT NOT NULL,
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

INSERT INTO account_with_issuer (
  id,
  issuer,
  accountId,
  providerId,
  userId,
  accessToken,
  refreshToken,
  idToken,
  accessTokenExpiresAt,
  refreshTokenExpiresAt,
  scope,
  password,
  createdAt,
  updatedAt
)
SELECT
  id,
  CASE providerId
    WHEN 'cloudflare' THEN 'https://uncertaintycat.cloudflareaccess.com/cdn-cgi/access/sso/oidc/77b32fa216cbcfad9abe98328816b6a759aa15950a43f4b071e186ffdd1e595d'
    WHEN 'credential' THEN 'local:credential'
    ELSE 'local:oauth:' || replace(replace(replace(providerId, '%', '%25'), '/', '%2F'), ' ', '%20')
  END,
  accountId,
  providerId,
  userId,
  accessToken,
  refreshToken,
  idToken,
  accessTokenExpiresAt,
  refreshTokenExpiresAt,
  scope,
  password,
  createdAt,
  updatedAt
FROM account;

DROP TABLE account;
ALTER TABLE account_with_issuer RENAME TO account;

CREATE INDEX account_user_idx ON account(userId);
CREATE UNIQUE INDEX account_issuer_accountId_uidx ON account(issuer, accountId);
