import VueI18nPlugin from "@intlify/unplugin-vue-i18n/vite";
import vue from "@vitejs/plugin-vue";
import fs from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath, URL } from "node:url";
import { defineConfig } from "vite";
import compression from "vite-plugin-compression2";

export default defineConfig({
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
    base: process.env.NODE_ENV === "production" ? getBaseUrl() : "/",
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

function getBaseUrl() {
    let appConfig = fs.readFileSync("appConfig.json");
    let dbUrl = JSON.parse(appConfig).dbUrl;
    if (dbUrl == "") {
        let dbPath = __dirname.replace(/app$/, "");
        let dbname = dbPath.split("/").reverse()[1];
        let config = fs.readFileSync("/etc/philologic/philologic5.cfg", "utf8");
        let re = /url_root = ["']([^"]+)["']/gm;
        let match = re.exec(config);
        let rootPath = match[1];
        if (rootPath.endsWith("/")) {
            rootPath = rootPath.slice(0, -1);
        }
        dbUrl = rootPath + "/" + dbname + "/";
        let jsonString = JSON.stringify({ dbUrl: dbUrl });
        fs.writeFileSync("./appConfig.json", jsonString);
        return dbUrl;
    }
    return dbUrl;
}
