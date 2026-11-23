import React from "react";
import ReactDOM from "react-dom/client";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { BrowserRouter } from "react-router-dom";

import App from "./App";
import { initAuth } from "./lib/auth";
import "./index.css";

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      // Sessions/versions are server-of-record; keep cache fresh on focus,
      // but don't hammer on every render.
      staleTime: 30_000,
      refetchOnWindowFocus: true,
    },
  },
});

const root = ReactDOM.createRoot(document.getElementById("root")!);

// Sign in (Entra mode) before the first render, so no component ever
// fires an unauthenticated request. initAuth() resolves false while the
// browser is being redirected to Microsoft sign-in.
initAuth()
  .then((ready) => {
    if (!ready) return;
    root.render(
      <React.StrictMode>
        <QueryClientProvider client={queryClient}>
          <BrowserRouter>
            <App />
          </BrowserRouter>
        </QueryClientProvider>
      </React.StrictMode>
    );
  })
  .catch((e) => {
    console.error("Sign-in failed", e);
    root.render(
      <div style={{ padding: 24, fontFamily: "sans-serif" }}>
        Sign-in failed: {String(e?.message ?? e)}. Reload to try again.
      </div>
    );
  });
