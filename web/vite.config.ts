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
      // formats + fileName intentionally omitted — driven by the
      // `rollupOptions.output` array below so we can give each format
      // its own chunk-splitting policy (ES code-splits; UMD doesn't).
    },
    sourcemap: true,
    rollupOptions: {
      external: ["onnxruntime-web", "@mediapipe/tasks-vision"],
      // Phase 5E-1: split the 3 heavy analyzers into their own chunks so
      // the main `spoof-detector.js` bundle doesn't carry them on first
      // paint. Only applied to the ES output — UMD library builds
      // require a single inline chunk, so we leave UMD untouched.
      output: [
        {
          format: "es",
          entryFileNames: "spoof-detector.js",
          chunkFileNames: "spoof-detector-[name]-[hash].js",
          globals: {
            "onnxruntime-web": "ort",
            "@mediapipe/tasks-vision": "MediaPipeTasksVision",
          },
          manualChunks(id: string): string | undefined {
            if (id.includes("/analyzers/MoireAnalyzer")) return "MoireAnalyzer";
            if (id.includes("/analyzers/TextureAnalyzer")) {
              return "TextureAnalyzer";
            }
            if (id.includes("/analyzers/ScreenReplayAnalyzer")) {
              return "ScreenReplayAnalyzer";
            }
            return undefined;
          },
        },
        {
          format: "umd",
          name: "SpoofDetector",
          entryFileNames: "spoof-detector.umd.cjs",
          inlineDynamicImports: true,
          globals: {
            "onnxruntime-web": "ort",
            "@mediapipe/tasks-vision": "MediaPipeTasksVision",
          },
        },
      ],
    },
    target: "es2020",
    emptyOutDir: true,
  },
  server: {
    port: 5180,
    open: "/examples/demo.html",
  },
});
