import * as assert from "assert";
import * as vscode from "vscode";

suite("extension activation", () => {
  test("activates and registers trelix.search / trelix.ask commands", async () => {
    const extension = vscode.extensions.getExtension("trelix.trelix-vscode");
    assert.ok(extension, "extension not found — check publisher.name in package.json");
    await extension!.activate();

    const commands = await vscode.commands.getCommands(true);
    assert.ok(commands.includes("trelix.search"), "trelix.search not registered");
    assert.ok(commands.includes("trelix.ask"), "trelix.ask not registered");
  });
});
