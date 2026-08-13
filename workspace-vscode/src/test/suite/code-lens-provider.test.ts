import * as assert from "assert";
import * as vscode from "vscode";
import {
    TrelixCodeLensProvider,
    TrelixCodeLens,
} from "../../code-lens-provider";
import { TrelixMcpClient, BlastRadiusEntry } from "../../mcp-client";

const NO_CANCEL: vscode.CancellationToken = {
    isCancellationRequested: false,
    onCancellationRequested: () => ({ dispose: () => undefined }),
};

const CANCELLED: vscode.CancellationToken = {
    isCancellationRequested: true,
    onCancellationRequested: () => ({ dispose: () => undefined }),
};

function fakeEntry(
    overrides: Partial<BlastRadiusEntry> = {},
): BlastRadiusEntry {
    return {
        file: "src/caller.py",
        symbol: "caller",
        kind: "function",
        lineStart: 42,
        language: "python",
        ...overrides,
    };
}

/**
 * A fake client whose every method bumps a shared counter. The CRITICAL perf
 * assertion is that provideCodeLenses leaves this counter at 0. `blastRadius`
 * can be overridden per-test to feed resolveCodeLens.
 */
function countingClient(
    blastRadius: (
        name: string,
        repo: string,
    ) => Promise<BlastRadiusEntry[]> = async () => [],
): { client: TrelixMcpClient; calls: () => number; blastCalls: () => number } {
    let calls = 0;
    let blastCalls = 0;
    const client = {
        getSymbol: async () => {
            calls++;
            return null;
        },
        search: async () => {
            calls++;
            return { results: [], nextCursor: null, totalAvailable: 0 };
        },
        ask: async () => {
            calls++;
            return { answer: "", sessionId: "", turnCount: 0 };
        },
        blastRadius: async (name: string, repo: string) => {
            calls++;
            blastCalls++;
            return blastRadius(name, repo);
        },
    } as unknown as TrelixMcpClient;
    return { client, calls: () => calls, blastCalls: () => blastCalls };
}

function docSymbol(
    name: string,
    line: number,
    children: vscode.DocumentSymbol[] = [],
): vscode.DocumentSymbol {
    const range = new vscode.Range(line, 0, line, 10);
    const ds = new vscode.DocumentSymbol(
        name,
        "",
        vscode.SymbolKind.Function,
        range,
        range,
    );
    ds.children = children;
    return ds;
}

async function openDoc(
    content: string,
    language = "plaintext",
): Promise<vscode.TextDocument> {
    return vscode.workspace.openTextDocument({ content, language });
}

suite("TrelixCodeLensProvider", () => {
    const disposables: vscode.Disposable[] = [];

    function registerSymbols(
        symbols: vscode.DocumentSymbol[],
        language = "plaintext",
    ): void {
        disposables.push(
            vscode.languages.registerDocumentSymbolProvider(
                { language },
                { provideDocumentSymbols: () => symbols },
            ),
        );
    }

    teardown(() => {
        for (const d of disposables) d.dispose();
        disposables.length = 0;
    });

    test("provideCodeLenses makes ZERO client calls (the perf contract)", async () => {
        registerSymbols([docSymbol("alpha", 0), docSymbol("beta", 3)]);
        const doc = await openDoc("alpha\n\n\nbeta\n");
        const { client, calls } = countingClient();

        const provider = new TrelixCodeLensProvider(
            async () => client,
            () => "/repo",
        );

        const lenses = await provider.provideCodeLenses(doc, NO_CANCEL);

        assert.strictEqual(
            calls(),
            0,
            "provideCodeLenses must not touch the client",
        );
        // Count-bearing lenses stay unresolved (no command) at provide time.
        const unresolved = lenses.filter((l) => l.command === undefined);
        assert.ok(unresolved.length > 0, "expected some unresolved lenses");
    });

    test("produces two lenses per top-level symbol (Find similar + unresolved count)", async () => {
        registerSymbols([docSymbol("alpha", 0), docSymbol("beta", 3)]);
        const doc = await openDoc("alpha\n\n\nbeta\n");
        const { client } = countingClient();

        const provider = new TrelixCodeLensProvider(
            async () => client,
            () => "/repo",
        );

        const lenses = (await provider.provideCodeLenses(
            doc,
            NO_CANCEL,
        )) as TrelixCodeLens[];

        assert.strictEqual(lenses.length, 4, "2 symbols x 2 lenses each");
        const findSimilar = lenses.filter(
            (l) => l.command?.command === "trelix.findSimilar",
        );
        assert.strictEqual(findSimilar.length, 2);
        assert.deepStrictEqual(
            findSimilar.map((l) => l.command!.arguments),
            [["alpha"], ["beta"]],
        );
        assert.strictEqual(
            lenses.filter((l) => l.command === undefined).length,
            2,
            "one unresolved count lens per symbol",
        );
    });

    test("resolveCodeLens populates the command with an N-dependents count", async () => {
        const { client, blastCalls } = countingClient(async () => [
            fakeEntry(),
            fakeEntry({ symbol: "other" }),
        ]);
        const provider = new TrelixCodeLensProvider(
            async () => client,
            () => "/repo",
        );

        const lens = new TrelixCodeLens(
            new vscode.Range(0, 0, 0, 0),
            "target",
            "file:///a.py",
            1,
        );
        const resolved = await provider.resolveCodeLens(lens, NO_CANCEL);

        assert.strictEqual(blastCalls(), 1);
        assert.ok(resolved.command, "command must be populated after resolve");
        assert.strictEqual(resolved.command!.command, "trelix.blastRadius");
        assert.deepStrictEqual(resolved.command!.arguments, ["target"]);
        assert.ok(
            resolved.command!.title.includes("2 dependents"),
            `title was: ${resolved.command!.title}`,
        );
    });

    test("cache hit: second resolve at same uri@version does not re-call blastRadius", async () => {
        const { client, blastCalls } = countingClient(async () => [
            fakeEntry(),
        ]);
        const provider = new TrelixCodeLensProvider(
            async () => client,
            () => "/repo",
        );

        const range = new vscode.Range(0, 0, 0, 0);
        const a = new TrelixCodeLens(range, "target", "file:///a.py", 1);
        const b = new TrelixCodeLens(range, "target", "file:///a.py", 1);

        await provider.resolveCodeLens(a, NO_CANCEL);
        await provider.resolveCodeLens(b, NO_CANCEL);

        assert.strictEqual(
            blastCalls(),
            1,
            "second resolve should hit the cache",
        );
    });

    test("cache miss: a version bump re-calls blastRadius", async () => {
        const { client, blastCalls } = countingClient(async () => [
            fakeEntry(),
        ]);
        const provider = new TrelixCodeLensProvider(
            async () => client,
            () => "/repo",
        );

        const range = new vscode.Range(0, 0, 0, 0);
        const v1 = new TrelixCodeLens(range, "target", "file:///a.py", 1);
        const v2 = new TrelixCodeLens(range, "target", "file:///a.py", 2);

        await provider.resolveCodeLens(v1, NO_CANCEL);
        await provider.resolveCodeLens(v2, NO_CANCEL);

        assert.strictEqual(
            blastCalls(),
            2,
            "version bump should miss the cache",
        );
    });

    test("no symbol provider installed -> [] (never throws)", async () => {
        // Deliberately register nothing; teardown cleared any prior provider.
        const doc = await openDoc("nothing here\n", "plaintext");
        const { client, calls } = countingClient();

        const provider = new TrelixCodeLensProvider(
            async () => client,
            () => "/repo",
        );

        const lenses = await provider.provideCodeLenses(doc, NO_CANCEL);
        assert.deepStrictEqual(lenses, []);
        assert.strictEqual(calls(), 0);
    });

    test("setting disabled -> [] without even querying symbols", async () => {
        registerSymbols([docSymbol("alpha", 0)]);
        const doc = await openDoc("alpha\n");
        const { client, calls } = countingClient();

        const config = vscode.workspace.getConfiguration("trelix");
        await config.update(
            "codeLens.enabled",
            false,
            vscode.ConfigurationTarget.Global,
        );
        try {
            const provider = new TrelixCodeLensProvider(
                async () => client,
                () => "/repo",
            );
            const lenses = await provider.provideCodeLenses(doc, NO_CANCEL);
            assert.deepStrictEqual(lenses, []);
            assert.strictEqual(calls(), 0);
        } finally {
            await config.update(
                "codeLens.enabled",
                undefined,
                vscode.ConfigurationTarget.Global,
            );
        }
    });

    test("cancellation during provide returns [] without building lenses", async () => {
        registerSymbols([docSymbol("alpha", 0)]);
        const doc = await openDoc("alpha\n");
        const { client } = countingClient();

        const provider = new TrelixCodeLensProvider(
            async () => client,
            () => "/repo",
        );

        const lenses = await provider.provideCodeLenses(doc, CANCELLED);
        assert.deepStrictEqual(lenses, []);
    });

    test("clearCache forces a re-call at the same uri@version", async () => {
        const { client, blastCalls } = countingClient(async () => [
            fakeEntry(),
        ]);
        const provider = new TrelixCodeLensProvider(
            async () => client,
            () => "/repo",
        );

        const range = new vscode.Range(0, 0, 0, 0);
        const lens = new TrelixCodeLens(range, "target", "file:///a.py", 1);

        await provider.resolveCodeLens(lens, NO_CANCEL);
        provider.clearCache();
        await provider.resolveCodeLens(lens, NO_CANCEL);

        assert.strictEqual(
            blastCalls(),
            2,
            "clearCache should drop the cached count",
        );
    });
});
