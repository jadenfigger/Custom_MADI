/* Browser-local history and shortcuts. Numeric data stays on the server. */
(function () {
    "use strict";
    const api = {bindings: [], current: null, states: [], cursor: -1, timer: null, restoring: false};
    const clone = value => JSON.parse(JSON.stringify(value));
    const same = (left, right) => JSON.stringify(left) === JSON.stringify(right);
    const set = (id, props) => window.dash_clientside.set_props(id, props);
    const noUpdate = () => window.dash_clientside.no_update;

    function status() {
        set("undo-state", {disabled: api.cursor <= 0});
        set("redo-state", {disabled: api.cursor >= api.states.length - 1});
        set("history-note", {children: `${Math.max(0, api.cursor)} undo steps · history holds up to 50 states in this tab`});
    }

    function remember(value) {
        if (api.cursor >= 0 && same(api.states[api.cursor], value)) return;
        api.states = api.states.slice(0, api.cursor + 1);
        api.states.push(clone(value));
        if (api.states.length > 50) api.states.shift();
        api.cursor = api.states.length - 1;
    }

    function flush() {
        clearTimeout(api.timer);
        if (!api.current) return;
        if (api.restoring) {
            api.states[api.cursor] = clone(api.current);
            api.restoring = false;
        } else {
            remember(api.current);
        }
        set("session-current", {data: clone(api.current)});
        status();
    }

    function restore(value) {
        clearTimeout(api.timer);
        api.restoring = true;
        api.current = clone(value);
        for (const binding of api.bindings) {
            set(binding.id, {[binding.property]: clone(value[binding.key])});
        }
        api.timer = setTimeout(flush, 350);
        status();
    }

    api.record = function (bindings, values) {
        api.bindings = bindings;
        api.current = Object.fromEntries(bindings.map((binding, i) =>
            [binding.key, values[i] === undefined ? null : values[i]]));
        clearTimeout(api.timer);
        api.timer = setTimeout(flush, 350);
        return noUpdate();
    };

    api.history = function () {
        flush();
        const triggered = window.dash_clientside.callback_context.triggered_id;
        const next = api.cursor + (triggered === "undo-state" ? -1 : 1);
        if (next >= 0 && next < api.states.length) {
            api.cursor = next;
            restore(api.states[next]);
        }
        return Date.now();
    };

    api.load = function (value) {
        if (!value || !value.controls) return noUpdate();
        flush();
        remember(value.controls);
        restore(value.controls);
        return value.nonce;
    };

    document.addEventListener("keydown", event => {
        const target = event.target;
        const input = target.closest("input");
        const editingInput = input && !["checkbox", "radio", "range", "button", "submit", "reset"].includes(input.type);
        if (event.repeat || editingInput || target.closest("textarea,select,[contenteditable=true],[role=textbox]")) return;
        let button = null;
        if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === "s") button = "save-session";
        if (event.altKey && event.key.toLowerCase() === "z") button = event.shiftKey ? "redo-state" : "undo-state";
        if (event.altKey && event.key.toLowerCase() === "r") button = "snap-reference";
        if (event.altKey && ["1", "2", "3", "4"].includes(event.key)) {
            event.preventDefault();
            set("view", {value: ["slice", "fisher", "widths", "inspect"][Number(event.key) - 1]});
        }
        if (button) {
            const element = document.getElementById(button);
            if (element && !element.disabled) {
                event.preventDefault();
                flush();
                element.click();
            }
        }
    });
    window.manifoldWorkspace = api;
}());
