import messages from "@intlify/unplugin-vue-i18n/messages";
import { createI18n } from "vue-i18n";

const availableLocales = ["en", "fr", "es"];

// Messages in two forms, "singular | plural", chosen as each language counts: French says "0 occurrence"
function pluralRule(locale) {
    const rules = new Intl.PluralRules(locale);
    return (count) => (rules.select(count) === "one" ? 0 : 1);
}

export default createI18n({
    legacy: false,
    locale: "en",
    fallbackLocale: "en",
    availableLocales,
    messages: messages,
    pluralRules: Object.fromEntries(availableLocales.map((locale) => [locale, pluralRule(locale)])),
});
