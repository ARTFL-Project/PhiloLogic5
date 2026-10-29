import { describe, it, expect } from "vitest";
import { mount } from "@vue/test-utils";
import GroupRanking from "../src/components/GroupRanking.vue";

describe("GroupRanking", () => {
    it("names each group's button by what it shows, after what choosing it does", () => {
        const wrapper = mount(GroupRanking, {
            props: {
                title: "Distinctive Author results",
                items: [["Voltaire", 12.3, ["raison", "esprit"]]],
                explainerLabel: "Top collocates",
                selectAriaLabelPrefix: "Comparative Collocation",
            },
        });
        const button = wrapper.find(".list-group-item");
        expect(button.attributes("aria-label")).toBeUndefined();
        const shown = [...button.element.childNodes].filter(
            (n) => n.nodeType !== Node.COMMENT_NODE && !n.classList?.contains("visually-hidden")
        );
        const visible = shown.map((n) => n.textContent).join(" ").replace(/\s+/g, " ").trim();
        expect(visible).toBe("Voltaire 12.3 Top collocates: raison, esprit");
        expect(button.text().replace(/\s+/g, " ")).toBe(`Comparative Collocation ${visible}`);
        wrapper.unmount();
    });

    it("says what the score is to screen readers, before the number", () => {
        const wrapper = mount(GroupRanking, {
            props: {
                items: [["Voltaire", 12.3, ["raison", "esprit"]]],
                explainerLabel: "Top collocates",
                scoreLabel: "distinctiveness score",
            },
        });
        const button = wrapper.find(".list-group-item");
        expect(button.find(".badge").text()).toBe("12.3");
        expect(button.text().replace(/\s+/g, " ")).toBe("Voltaire distinctiveness score 12.3 Top collocates: raison, esprit");
        wrapper.unmount();
    });
});
