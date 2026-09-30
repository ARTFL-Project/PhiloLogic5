import vue from "@vitejs/plugin-vue";
import { defineConfig } from "vite";
import compression from "vite-plugin-compression2";

// Relative paths: the UI can be served at any URL prefix (behind a reverse proxy), its routes are in the URL hash
export default defineConfig({
    base: "./",
    plugins: [
        vue(),
        compression({ algorithms: ["brotliCompress"], filter: /\.(js|css|html|svg|json)$/i, deleteOriginalAssets: false }),
    ],
    css: {
        // Bootstrap's Sass still uses @import and old color functions: its deprecation warnings are only noise here
        preprocessorOptions: { scss: { quietDeps: true, silenceDeprecations: ["import", "global-builtin", "color-functions"] } },
    },
    server: {
        // npm run dev: the API of a UI started with philologic5-webui-loader --port 8765
        proxy: { "/api": "http://localhost:8765" },
    },
    test: {
        environment: "jsdom",
        globals: true,
    },
});
