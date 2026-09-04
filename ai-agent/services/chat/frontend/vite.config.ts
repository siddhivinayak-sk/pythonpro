import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// Dev server proxies API calls to the chat backend so the SPA and API share an origin.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 3000,
    proxy: {
      "/v1": "http://localhost:8080",
      "/healthz": "http://localhost:8080",
    },
  },
});
