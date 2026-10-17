import js from "@eslint/js";
import pluginVue from "eslint-plugin-vue";
import globals from "globals";

export default [
    { ignores: ["dist/"] },
    js.configs.recommended,
    ...pluginVue.configs["flat/essential"],
    {
        languageOptions: {
            ecmaVersion: "latest",
            sourceType: "module",
            globals: { ...globals.browser, ...globals.node },
        },
        rules: {
            "no-console": "off",
            "vue/multi-word-component-names": "off",
            "vue/no-multiple-template-root": "off",
            "vue/no-use-v-if-with-v-for": "off",
            // the metadata forms fill in the objects their parent gives them
            "vue/no-mutating-props": ["error", { shallowOnly: true }],
        },
    },
];
