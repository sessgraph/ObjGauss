import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

export default defineConfig({
  plugins: [react()],
  server: {
    watch: {
      ignored: [
        "**/.venv/**",
        "**/.pytest_cache/**",
        "**/objgauss.egg-info/**",
        "**/dist/**",
        "**/outputs/**",
      ],
    },
  },
});
