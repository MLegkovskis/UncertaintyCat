import { describe, expect, it } from "vitest";

import { formatIdentity } from "./Shell";

describe("formatIdentity", () => {
  it("uses first and last initials for a non-empty name", () => {
    expect(
      formatIdentity({ authenticated: true, name: " Ada Lovelace ", email: "ada@example.com" }),
    ).toMatchObject({ initials: "AL", label: "Ada Lovelace", fallbackIcon: false });
  });

  it("treats blank OIDC names as missing and uses email-local initials", () => {
    expect(
      formatIdentity({ authenticated: true, name: "  ", email: "mark.legkovskis@example.com" }),
    ).toMatchObject({ initials: "MA", label: "mark.legkovskis@example.com" });
  });

  it("uses a user icon only when no usable authenticated claim exists", () => {
    expect(formatIdentity({ authenticated: true, name: "", email: "@example.com" })).toEqual({
      initials: "",
      label: "Account",
      fallbackIcon: true,
    });
  });

  it("keeps the cat identity for a guest", () => {
    expect(formatIdentity({ authenticated: false, name: "", email: "" })).toEqual({
      initials: "UC",
      label: "Guest workspace",
      fallbackIcon: false,
    });
  });
});
