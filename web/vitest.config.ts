// Vitest config — picks up __tests__/**/*.test.ts.
import { defineConfig } from "vitest/config";

export default defineConfig({
  test: {
    include: ["__tests__/**/*.test.ts"],
    environment: "node",
    globals: false,
    coverage: {
      reporter: ["text", "html"],
      include: ["src/**/*.ts"],
    },
  },
});
