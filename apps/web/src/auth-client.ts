import { createAuthClient } from "better-auth/react";
import { useState } from "react";

export const authClient = createAuthClient();

const signInUnavailable =
  "Cloudflare sign-in could not start. Please try again shortly.";

export function useCloudflareSignIn() {
  const [signInError, setSignInError] = useState<string | null>(null);
  const [isSigningIn, setIsSigningIn] = useState(false);

  const beginSignIn = async (callbackURL: string) => {
    setSignInError(null);
    setIsSigningIn(true);
    try {
      const result = await authClient.signIn.social({
        provider: "cloudflare",
        callbackURL,
      });
      if (result.error) setSignInError(signInUnavailable);
    } catch {
      setSignInError(signInUnavailable);
    } finally {
      setIsSigningIn(false);
    }
  };

  return { beginSignIn, isSigningIn, signInError };
}
