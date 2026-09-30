import { createI18n } from "vue-i18n";
import en from "./locales/en.json";
import fr from "./locales/fr.json";

const saved = localStorage.getItem("philologic5-webui-loader-lang");
const browser = (navigator.language || "en").slice(0, 2);

export default createI18n({
    legacy: false,
    locale: saved || (["en", "fr"].includes(browser) ? browser : "en"),
    fallbackLocale: "en",
    messages: { en, fr },
});
