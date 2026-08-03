import * as assert from "assert";
import * as vscode from "vscode";
import { TrelixHoverProvider } from "../../hover-provider";
import { TrelixMcpClient, Symbol } from "../../mcp-client";

function fakeSymbol(overrides: Partial<Symbol> = {}): Symbol {
  return {
    name: "validate_token",
    qualifiedName: "AuthService.validate_token",
    kind: "method",
    file: "src/auth.py",
    lineStart: 10,
    lineEnd: 25,
    signature: "def validate_token(self, token: str) -> bool",
    docstring: "Validate a JWT token.",
    body: "def validate_token(self, token): ...",
    language: "python",
    ...overrides,
  };
}

async function openDoc(content: string): Promise<vscode.TextDocument> {
  return vscode.workspace.openTextDocument({ content, language: "python" });
}

const NO_CANCEL: vscode.CancellationToken = {
  isCancellationRequested: false,
  onCancellationRequested: () => ({ dispose: () => undefined }),
};

suite("TrelixHoverProvider.provideHover", () => {
  test("returns a Hover with signature and docstring for a resolvable word", async () => {
    const doc = await openDoc("validate_token(x)\n");
    const position = new vscode.Position(0, 2);

    let calls = 0;
    const fakeClient = {
      getSymbol: async () => {
        calls++;
        return fakeSymbol();
      },
    } as unknown as TrelixMcpClient;

    const provider = new TrelixHoverProvider(
      async () => fakeClient,
      () => "/repo"
    );

    const hover = await provider.provideHover(doc, position, NO_CANCEL);

    assert.ok(hover, "expected a Hover to be returned");
    assert.strictEqual(calls, 1);
    const md = hover!.contents[0] as vscode.MarkdownString;
    const text = md.value;
    assert.ok(text.includes("validate_token"));
    assert.ok(text.includes("Validate a JWT token."));
    assert.ok(text.includes("src/auth.py:10-25"));
  });

  test("returns undefined when get_symbol resolves to null (not found)", async () => {
    const doc = await openDoc("some_unknown_identifier\n");
    const position = new vscode.Position(0, 2);

    const fakeClient = {
      getSymbol: async () => null,
    } as unknown as TrelixMcpClient;

    const provider = new TrelixHoverProvider(
      async () => fakeClient,
      () => "/repo"
    );

    const hover = await provider.provideHover(doc, position, NO_CANCEL);

    assert.strictEqual(hover, undefined);
  });

  test("returns undefined (silent no-op) when getClient throws", async () => {
    const doc = await openDoc("validate_token(x)\n");
    const position = new vscode.Position(0, 2);

    const provider = new TrelixHoverProvider(
      async () => {
        throw new Error("not connected");
      },
      () => "/repo"
    );

    const hover = await provider.provideHover(doc, position, NO_CANCEL);

    assert.strictEqual(hover, undefined);
  });

  test("caches a resolved symbol and does not call getSymbol twice for the same word+repo", async () => {
    const doc = await openDoc("validate_token(x)\nvalidate_token(y)\n");

    let calls = 0;
    const fakeClient = {
      getSymbol: async () => {
        calls++;
        return fakeSymbol();
      },
    } as unknown as TrelixMcpClient;

    const provider = new TrelixHoverProvider(
      async () => fakeClient,
      () => "/repo"
    );

    await provider.provideHover(doc, new vscode.Position(0, 2), NO_CANCEL);
    await provider.provideHover(doc, new vscode.Position(1, 2), NO_CANCEL);

    assert.strictEqual(calls, 1, "expected the second lookup to hit the cache");
  });
});
