// Library-mode build config for @rollingcat/spoof-detector.
// Externalizes onnxruntime-web and @mediapipe/tasks-vision so consumers
// can supply their own version (and we don't ship a 30 MB bundle).

import { defineConfig } from "vite";
import { resolve } from "node:path";

export default defineConfig({
  build: {
    lib: {
      entry: resolve(__dirname, "src/index.ts"),
      name: "SpoofDetector",
      formats: ["es", "umd"],
      fileName: (format) =>
        format === "es" ? "spoof-detector.js" : "spoof-detector.umd.cjs",
    },
    sourcemap: true,
    rollupOptions: {
      external: ["onnxruntime-web", "@mediapipe/tasks-vision"],
      output: {
        globals: {
          "onnxruntime-web": "ort",
          "@mediapipe/tasks-vision": "MediaPipeTasksVision",
        },
      },
    },
    target: "es2020",
    emptyOutDir: true,
  },
  server: {
    port: 5180,
    open: "/examples/demo.html",
  },
});
