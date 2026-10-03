import { describe, it, expect, vi, afterEach } from "vitest";
import { Modal } from "bootstrap";
import router from "../src/router/index.js";

function dialog(className) {
    const element = document.createElement("div");
    element.className = className;
    element.innerHTML = '<div class="modal-dialog"></div>';
    document.body.appendChild(element);
    const modal = new Modal(element);
    return vi.spyOn(modal, "hide");
}

afterEach(() => {
    document.body.innerHTML = "";
});

describe("router", () => {
    it("closes the open dialog a navigation starts from", async () => {
        // The results' titles dialog links to texts: the results page went with the dialog in it, and its backdrop
        // stayed over the text
        const open = dialog("modal fade show");
        const closed = dialog("modal fade");
        await router.push("/no/such/page");
        expect(open).toHaveBeenCalledOnce();
        expect(closed).not.toHaveBeenCalled();
    });
});
