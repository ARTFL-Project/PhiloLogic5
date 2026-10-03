import VueI18nPlugin from "@intlify/unplugin-vue-i18n/vite";
import vue from "@vitejs/plugin-vue";
import { dirname, resolve } from "node:path";
import { fileURLToPath, URL } from "node:url";
import { defineConfig } from "vite";
import compression from "vite-plugin-compression2";

export default defineConfig({
    test: {
        environment: "jsdom",
        globals: true,
    },
    plugins: [
        vue(),
        VueI18nPlugin({
            include: resolve(
                dirname(fileURLToPath(import.meta.url)),
                "./src/locales/**"
            ),
        }),
        compression({
            algorithm: "brotliCompress", // Use Brotli for compression
            ext: ".br", // File extension for Brotli compressed files
            threshold: 0, // Compress all assets (even small ones)
            deleteOriginFile: false, // Keep the original files for fallback
            compressionOptions: { level: 11 }, // Maximize compression level for Brotli
            filter: /\.(js|css|html|svg|json)$/i, // Only compress specific file types
        }),
    ],
    // Paths relative to the <base href> the server gives each page, the database's own path: the client is built for
    // no host or prefix in particular, so a database can be served anywhere, behind a proxy (EZproxy) as under its
    // own name, or copied to another machine, without rebuilding it
    base: process.env.NODE_ENV === "production" ? "./" : "/",
    resolve: {
        alias: {
            "@": fileURLToPath(new URL("./src", import.meta.url)),
        },
        // TODO: Remove by explicitely adding extension in imports
        extensions: [".js", ".json", ".vue"],
    },
    server: {
        hmr: {
            overlay: false,
        },
    },
});
