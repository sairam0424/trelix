import * as assert from "assert";
import * as vscode from "vscode";
import {
    TrelixCodeLensProvider,
    TrelixCodeLens,
    dependentsTitle,
} from "../../code-lens-provider";
import {
    TrelixMcpClient,
    BlastRadiusEntry,
    BlastRadiusResult,
} from "../../mcp-client";

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

/** A blast_radius result with `count` entries; `total` defaults to `count` (nothing cut). */
function fakeResult(count: number, total = count): BlastRadiusResult {
    return {
        entries: Array.from({ length: count }, (_, i) =>
            fakeEntry({ symbol: `caller${i}`, lineStart: i + 1 }),
        ),
        total,
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
    ) => Promise<BlastRadiusResult> = async () => fakeResult(0),
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

    test("resolveCodeLens wires the command to VS Code's native Peek References popup (Visual CodeLens), not a custom QuickPick command", async () => {
        const { client, blastCalls } = countingClient(async () =>
            fakeResult(2),
        );
        const provider = new TrelixCodeLensProvider(
            async () => client,
            () => "/repo",
        );

        const range = new vscode.Range(3, 0, 3, 0);
        const lens = new TrelixCodeLens(range, "target", "file:///a.py", 1);
        const resolved = (await provider.resolveCodeLens(
            lens,
            NO_CANCEL,
        )) as TrelixCodeLens;

        assert.strictEqual(blastCalls(), 1);
        assert.ok(resolved.command, "command must be populated after resolve");
        assert.strictEqual(
            resolved.command!.command,
            "editor.action.showReferences",
            "should invoke VS Code's built-in Peek References popup",
        );
        assert.notStrictEqual(
            resolved.command!.command,
            "trelix.blastRadius",
            "must not route through the old Invokable-CodeLens command / QuickPick",
        );
        assert.strictEqual(
            resolved.command!.title,
            "$(references) 2 dependents",
        );

        const [uri, position, locations] = resolved.command!.arguments as [
            vscode.Uri,
            vscode.Position,
            vscode.Location[],
        ];
        assert.strictEqual(uri.toString(), "file:///a.py");
        assert.deepStrictEqual(position, range.start);
        assert.strictEqual(locations.length, 2, "one Location per dependent");
        assert.ok(
            locations.every((l) => l instanceof vscode.Location),
            "each argument should be a real Location the peek popup can render",
        );
    });

    test("cache hit: second resolve at same uri@version does not re-call blastRadius", async () => {
        const { client, blastCalls } = countingClient(async () =>
            fakeResult(1),
        );
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
        const { client, blastCalls } = countingClient(async () =>
            fakeResult(1),
        );
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
        const { client, blastCalls } = countingClient(async () =>
            fakeResult(1),
        );
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

    test("a cut list shows the real total and says how many are listed", async () => {
        const { client } = countingClient(async () => fakeResult(100, 150));
        const provider = new TrelixCodeLensProvider(
            async () => client,
            () => "/repo",
        );

        const range = new vscode.Range(0, 0, 0, 0);
        const lens = new TrelixCodeLens(range, "hub", "file:///a.py", 1);
        const resolved = await provider.resolveCodeLens(lens, NO_CANCEL);

        assert.strictEqual(
            resolved.command!.title,
            "$(references) 150 dependents (showing 100)",
        );
        const locations = resolved.command!.arguments![2] as vscode.Location[];
        assert.strictEqual(
            locations.length,
            100,
            "the popup lists the entries we hold, not a made-up 150",
        );
    });

    test("a cached cut result keeps the real total", async () => {
        const { client, blastCalls } = countingClient(async () =>
            fakeResult(100, 150),
        );
        const provider = new TrelixCodeLensProvider(
            async () => client,
            () => "/repo",
        );

        const range = new vscode.Range(0, 0, 0, 0);
        await provider.resolveCodeLens(
            new TrelixCodeLens(range, "hub", "file:///a.py", 1),
            NO_CANCEL,
        );
        const second = await provider.resolveCodeLens(
            new TrelixCodeLens(range, "hub", "file:///a.py", 1),
            NO_CANCEL,
        );

        assert.strictEqual(blastCalls(), 1, "second resolve must hit the cache");
        assert.strictEqual(
            second.command!.title,
            "$(references) 150 dependents (showing 100)",
        );
    });

    test("a failing blast_radius call resolves to 0 dependents and shows no error", async () => {
        const { client } = countingClient(async () => {
            throw new Error("index missing");
        });
        const provider = new TrelixCodeLensProvider(
            async () => client,
            () => "/repo",
        );

        const range = new vscode.Range(0, 0, 0, 0);
        const resolved = await provider.resolveCodeLens(
            new TrelixCodeLens(range, "target", "file:///a.py", 1),
            NO_CANCEL,
        );

        assert.strictEqual(
            resolved.command!.title,
            "$(references) 0 dependents",
        );
    });

    test("a cancelled resolve still lists the entries, titles the lens plainly and is not cached", async () => {
        const { client, blastCalls } = countingClient(async () =>
            fakeResult(2),
        );
        const provider = new TrelixCodeLensProvider(
            async () => client,
            () => "/repo",
        );

        const range = new vscode.Range(0, 0, 0, 0);
        const cancelled = await provider.resolveCodeLens(
            new TrelixCodeLens(range, "target", "file:///a.py", 1),
            CANCELLED,
        );

        assert.strictEqual(
            cancelled.command!.title,
            "$(references) Blast radius",
        );
        const locations = cancelled.command!.arguments![2] as vscode.Location[];
        assert.strictEqual(locations.length, 2);

        await provider.resolveCodeLens(
            new TrelixCodeLens(range, "target", "file:///a.py", 1),
            NO_CANCEL,
        );
        assert.strictEqual(
            blastCalls(),
            2,
            "a cancelled result must not be cached",
        );
    });
});

suite("dependentsTitle", () => {
    const cases: Array<{
        name: string;
        total: number;
        shown: number;
        expected: string;
    }> = [
        {
            name: "0 dependents",
            total: 0,
            shown: 0,
            expected: "$(references) 0 dependents",
        },
        {
            name: "1 dependent (singular)",
            total: 1,
            shown: 1,
            expected: "$(references) 1 dependent",
        },
        {
            name: "100 dependents, nothing cut",
            total: 100,
            shown: 100,
            expected: "$(references) 100 dependents",
        },
        {
            name: "100 of 150, list was cut",
            total: 150,
            shown: 100,
            expected: "$(references) 150 dependents (showing 100)",
        },
    ];
    for (const c of cases) {
        test(c.name, () => {
            assert.strictEqual(dependentsTitle(c.total, c.shown), c.expected);
        });
    }
});
