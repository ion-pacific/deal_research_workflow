/**
 * Sign-in + request credentials.
 *
 * Entra mode (VITE_AZURE_CLIENT_ID set): Microsoft sign-in via MSAL
 * redirect flow; every API call carries `Authorization: Bearer <access
 * token>` for the API scope. The backend validates it (app/auth.py) and
 * derives the user's email from the token -- the client can't choose it.
 *
 * Stub mode (no client id, local dev): the old typed-email
 * X-User-Email header.
 */
import {
  InteractionRequiredAuthError,
  PublicClientApplication,
  type AccountInfo,
} from "@azure/msal-browser";

import { useUI } from "../stores/ui";

const CLIENT_ID = (import.meta.env.VITE_AZURE_CLIENT_ID as string | undefined) ?? "";
const TENANT_ID = (import.meta.env.VITE_AZURE_TENANT_ID as string | undefined) ?? "";
const API_SCOPE =
  (import.meta.env.VITE_API_SCOPE as string | undefined) ??
  (CLIENT_ID ? `api://${CLIENT_ID}/access_as_user` : "");

export const ENTRA_ENABLED = Boolean(CLIENT_ID && TENANT_ID);

const msal = ENTRA_ENABLED
  ? new PublicClientApplication({
      auth: {
        clientId: CLIENT_ID,
        authority: `https://login.microsoftonline.com/${TENANT_ID}`,
        redirectUri: window.location.origin,
        postLogoutRedirectUri: window.location.origin,
      },
      // sessionStorage: tokens don't outlive the tab; MSAL's own SSO
      // cookie on login.microsoftonline.com keeps re-login silent.
      cache: { cacheLocation: "sessionStorage" },
    })
  : null;

let account: AccountInfo | null = null;

/**
 * Call once before rendering. Returns true when the app may render;
 * false when a redirect to Microsoft sign-in is under way.
 */
export async function initAuth(): Promise<boolean> {
  if (!msal) return true;
  await msal.initialize();
  const result = await msal.handleRedirectPromise();
  account = result?.account ?? msal.getAllAccounts()[0] ?? null;
  if (!account) {
    await msal.loginRedirect({ scopes: [API_SCOPE] });
    return false;
  }
  msal.setActiveAccount(account);
  // Display only -- the backend never trusts this value in Entra mode.
  useUI.getState().setUserEmail(account.username.toLowerCase());
  return true;
}

/** Headers that identify the user on an API request. */
export async function authHeaders(): Promise<Record<string, string>> {
  if (!msal) {
    const email = useUI.getState().userEmail;
    return email ? { "X-User-Email": email } : {};
  }
  if (!account) throw new Error("Not signed in");
  try {
    const r = await msal.acquireTokenSilent({ scopes: [API_SCOPE], account });
    return { Authorization: `Bearer ${r.accessToken}` };
  } catch (e) {
    if (e instanceof InteractionRequiredAuthError) {
      // Session expired / consent needed: bounce through sign-in.
      await msal.acquireTokenRedirect({ scopes: [API_SCOPE], account });
    }
    throw e;
  }
}

export async function signOut(): Promise<void> {
  useUI.getState().setUserEmail("");
  if (msal) await msal.logoutRedirect({ account: account ?? undefined });
}
