import * as vscode from "vscode";
import { TrelixMcpClient, BlastRadiusEntry } from "./mcp-client";

/** Max symbols we annotate per document — a hard cap so huge files stay fast. */
const MAX_SYMBOLS = 200;
/**
 * Deepest nesting level we descend into. Depth 0 = top-level symbols,
 * 1 = their direct children, 2 = grandchildren. We skip nesting below
 * depth 2 (great-grandchildren and deeper) to keep the lens list legible.
 */
const MAX_NESTING_DEPTH = 2;

/**
 * A CodeLens that carries the metadata resolveCodeLens needs (which symbol,
 * and the exact document version it was produced for) without a second MCP
 * round-trip. Exported so tests can construct one directly, mirroring the way
 * hover-provider.test.ts builds a fakeSymbol.
 */
export class TrelixCodeLens extends vscode.CodeLens {
    constructor(
        range: vscode.Range,
        public readonly symbolName: string,
        public readonly docUri: string,
        public readonly docVersion: number,
        command?: vscode.Command,
    ) {
        super(range, command);
    }
}

/** Best-effort range for either a DocumentSymbol or a SymbolInformation. */
function symbolRange(
    sym: vscode.DocumentSymbol | vscode.SymbolInformation,
): vscode.Range {
    const anySym = sym as Partial<vscode.DocumentSymbol> &
        Partial<vscode.SymbolInformation>;
    if (anySym.selectionRange) return anySym.selectionRange;
    if (anySym.range) return anySym.range;
    if (anySym.location?.range) return anySym.location.range;
    return new vscode.Range(0, 0, 0, 0);
}

/**
 * Actionable code lenses for trelix.
 *
 * PERF CONTRACT (the whole point of this provider): provideCodeLenses makes
 * ZERO network/MCP calls. It derives symbol ranges purely locally via the
 * built-in vscode.executeDocumentSymbolProvider and returns cheap, mostly
 * unresolved lenses. The single MCP call (blastRadius, to fill an "N
 * dependents" count) happens lazily in resolveCodeLens — and only for lenses
 * VS Code actually paints, cached per `${uri}@${version}::${symbol}` so a
 * scroll-back never re-hits the server for the same document revision.
 *
 * Follows hover-provider.ts conventions: constructor-injected getClient /
 * getRepoPath, a version-keyed Map cache, silent no-op on any error (never an
 * error dialog), and clearCache() for tests.
 */
export class TrelixCodeLensProvider implements vscode.CodeLensProvider {
    private readonly cache = new Map<string, BlastRadiusEntry[]>();
    private readonly _onDidChangeCodeLenses = new vscode.EventEmitter<void>();
    readonly onDidChangeCodeLenses: vscode.Event<void> =
        this._onDidChangeCodeLenses.event;

    constructor(
        private readonly getClient: () => Promise<TrelixMcpClient>,
        private readonly getRepoPath: () => string,
    ) {}

    private isEnabled(): boolean {
        return vscode.workspace
            .getConfiguration("trelix")
            .get<boolean>("codeLens.enabled", true);
    }

    /**
     * ZERO MCP calls. Only asks VS Code for the document's symbols (locally
     * computed by whatever language provider is installed) and emits two lenses
     * per symbol: a fully-built "Find similar" lens (needs only a command, no
     * data) and an UNRESOLVED count-bearing lens whose title is filled in later
     * by resolveCodeLens.
     */
    async provideCodeLenses(
        document: vscode.TextDocument,
        token: vscode.CancellationToken,
    ): Promise<vscode.CodeLens[]> {
        if (!this.isEnabled()) return [];

        let symbols:
            Array<vscode.DocumentSymbol | vscode.SymbolInformation> | undefined;
        try {
            symbols = await vscode.commands.executeCommand<
                Array<vscode.DocumentSymbol | vscode.SymbolInformation>
            >("vscode.executeDocumentSymbolProvider", document.uri);
        } catch {
            // No symbol provider installed, or it errored — degrade to no lenses.
            return [];
        }
        if (token.isCancellationRequested) return [];
        if (!symbols || symbols.length === 0) return [];

        const collected: Array<
            vscode.DocumentSymbol | vscode.SymbolInformation
        > = [];
        this.collectSymbols(symbols, 0, collected);

        const uri = document.uri.toString();
        const version = document.version;
        const lenses: vscode.CodeLens[] = [];
        for (const sym of collected) {
            const start = symbolRange(sym).start;
            const lensRange = new vscode.Range(start.line, 0, start.line, 0);

            // Static-title lens — fully built here (command only, no data needed).
            lenses.push(
                new TrelixCodeLens(lensRange, sym.name, uri, version, {
                    title: "$(search) Find similar",
                    command: "trelix.findSimilar",
                    arguments: [sym.name],
                }),
            );

            // Count-bearing lens — UNRESOLVED (no command). resolveCodeLens fills it.
            lenses.push(new TrelixCodeLens(lensRange, sym.name, uri, version));
        }
        return lenses;
    }

    /**
     * The ONLY place MCP is touched. Calls blastRadius() to turn an unresolved
     * lens into an "N dependents" action. Result is cached on the exact
     * `${uri}@${version}` the lens was produced for, so re-resolves (scroll-back,
     * repeated paints) at the same revision are free; a document edit bumps the
     * version and correctly misses the cache.
     */
    async resolveCodeLens(
        codeLens: vscode.CodeLens,
        token: vscode.CancellationToken,
    ): Promise<vscode.CodeLens> {
        const lens = codeLens as TrelixCodeLens;
        const symbolName = lens.symbolName;
        const repoPath = this.getRepoPath();

        // Defensive: a lens without our metadata still gets a usable command.
        if (!symbolName) {
            lens.command = {
                title: "$(references) Blast radius",
                command: "trelix.blastRadius",
                arguments: [],
            };
            return lens;
        }

        const cacheKey = `${lens.docUri}@${lens.docVersion}::${symbolName}`;
        let entries = this.cache.get(cacheKey);
        if (!entries) {
            try {
                const client = await this.getClient();
                entries = await client.blastRadius(symbolName, repoPath);
            } catch {
                entries = []; // silent no-op — never surface an error dialog
            }
            if (token.isCancellationRequested) {
                // Don't cache a result computed for a cancelled request.
                lens.command = {
                    title: "$(references) Blast radius",
                    command: "trelix.blastRadius",
                    arguments: [symbolName],
                };
                return lens;
            }
            this.cache.set(cacheKey, entries);
        }

        const count = entries.length;
        lens.command = {
            title: `$(references) ${count} dependent${count === 1 ? "" : "s"}`,
            command: "trelix.blastRadius",
            arguments: [symbolName],
        };
        return lens;
    }

    /**
     * Flatten the symbol tree up to MAX_NESTING_DEPTH, capping the total at
     * MAX_SYMBOLS. SymbolInformation has no children, so only DocumentSymbol
     * trees recurse.
     */
    private collectSymbols(
        symbols: Array<vscode.DocumentSymbol | vscode.SymbolInformation>,
        depth: number,
        out: Array<vscode.DocumentSymbol | vscode.SymbolInformation>,
    ): void {
        for (const sym of symbols) {
            if (out.length >= MAX_SYMBOLS) return;
            out.push(sym);
            const children = (sym as vscode.DocumentSymbol).children;
            if (depth < MAX_NESTING_DEPTH && children && children.length > 0) {
                this.collectSymbols(children, depth + 1, out);
            }
        }
    }

    /** Clear the resolve cache and ask VS Code to re-query lenses. */
    refresh(): void {
        this.cache.clear();
        this._onDidChangeCodeLenses.fire();
    }

    /** Exposed for tests — same convention as the hover provider. */
    clearCache(): void {
        this.cache.clear();
    }

    dispose(): void {
        this._onDidChangeCodeLenses.dispose();
    }
}
