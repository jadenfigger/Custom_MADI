"use strict";
const assert = require("node:assert/strict");
const {readFileSync} = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");
const {test} = require("node:test");

function browser() {
    const updates = {}, pending = new Map(), listeners = {}, clicked = [];
    let serial = 0;
    const context = {
        window: {dash_clientside: {
            no_update: null, callback_context: {},
            set_props: (id, props) => { updates[typeof id === "string" ? id : JSON.stringify(id)] = props; },
        }},
        document: {
            addEventListener: (name, listener) => { listeners[name] = listener; },
            getElementById: id => ({disabled: false, click: () => clicked.push(id)}),
        },
        setTimeout: fn => { pending.set(++serial, fn); return serial; },
        clearTimeout: id => pending.delete(id),
    };
    vm.runInNewContext(readFileSync(path.join(__dirname,
        "../../tools/manifold_explorer/assets/workspace.js"), "utf8"), context);
    const api = context.window.manifoldWorkspace;
    const bindings = [{id: "ref", property: "data", key: "ref"},
                      {id: "view", property: "value", key: "view"}];
    const drain = () => { for (const [id, fn] of [...pending]) { pending.delete(id); fn(); } };
    const record = (row, view) => { api.record(bindings, [row, view]); drain(); };
    const history = action => {
        context.window.dash_clientside.callback_context.triggered_id = action;
        api.history(); drain();
    };
    return {api, bindings, updates, drain, record, history, listeners, clicked};
}

test("missing optional values survive JSON serialization as null", () => {
    const b = browser();
    b.record(undefined, "slice");
    assert.equal(JSON.stringify(b.updates["session-current"].data), '{"ref":null,"view":"slice"}');
});

test("undo/redo restore controls, imports are undoable, edits discard the redo branch", () => {
    const b = browser();
    b.record(1, "slice"); b.record(2, "fisher");
    b.history("undo-state");
    assert.equal(b.updates.ref.data, 1);
    assert.equal(b.updates.view.value, "slice");
    b.history("redo-state");
    assert.equal(b.updates.ref.data, 2);
    b.api.load({controls: {ref: 8, view: "inspect"}, nonce: "import"}); b.drain();
    assert.equal(b.updates.ref.data, 8);
    b.history("undo-state");
    assert.equal(b.updates.ref.data, 2);
    b.record(3, "widths");
    b.history("redo-state");
    assert.equal(b.updates["session-current"].data.ref, 3);
    assert.equal(b.updates["redo-state"].disabled, true);
});

test("rapid edits coalesce and history has a fixed bound", () => {
    const b = browser();
    b.record(0, "slice");
    for (let row = 1; row < 10; row++) b.api.record(b.bindings, [row, "slice"]);
    b.drain();
    assert.equal(b.api.states.length, 2);
    for (let row = 10; row < 100; row++) b.record(row, "slice");
    assert.equal(b.api.states.length, 50);
    assert.equal(b.api.cursor, 49);
});

test("keyboard shortcuts do not intercept text editing", () => {
    const b = browser();
    b.record(1, "slice");
    const event = {key: "s", ctrlKey: true, preventDefault() {}, target: {closest: () => true}};
    b.listeners.keydown(event);
    assert.equal(b.clicked.length, 0);
    event.target.closest = () => false;
    b.listeners.keydown(event);
    assert.deepEqual(b.clicked, ["save-session"]);
    b.listeners.keydown({...event, ctrlKey: false, altKey: true, key: "4"});
    assert.equal(b.updates.view.value, "inspect");
    event.target.closest = selector => selector === "input" ? {type: "radio"} : null;
    b.listeners.keydown({...event, ctrlKey: false, altKey: true, key: "2"});
    assert.equal(b.updates.view.value, "fisher");
});
