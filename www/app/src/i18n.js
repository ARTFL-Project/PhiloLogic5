import messages from "@intlify/unplugin-vue-i18n/messages";
import { createI18n } from "vue-i18n";

export default createI18n({
    legacy: false,
    locale: "en",
    fallbackLocale: "en",
    availableLocales: ["en", "fr"],
    messages: messages,
});
