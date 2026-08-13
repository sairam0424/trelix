/**
 * Semantically-faithful, minimal mock of the `vscode` module for HEADLESS unit
 * tests (plain node/mocha, no Extension Host). Implements only the runtime
 * surface the SUTs and their tests exercise; type-only members are erased at
 * compile time and need no implementation.
 *
 * Faithfulness: Range/Location/Position store exactly their args; CodeLens
 * .isResolved reflects whether a command is set; executeCommand(
 * "vscode.executeDocumentSymbolProvider", uri) invokes the most-recent,
 * non-disposed DocumentSymbolProvider; getConfiguration().get/update is backed
 * by a shared store so an update is visible to a later getConfiguration().
 */
/* eslint-disable @typescript-eslint/no-explicit-any */

export class Position {
    constructor(
        readonly line: number,
        readonly character: number,
    ) {}
}

export class Range {
    readonly start: Position;
    readonly end: Position;
    constructor(
        a: number | Position,
        b: number | Position,
        c?: number,
        d?: number,
    ) {
        if (typeof a === "number") {
            this.start = new Position(a, b as number);
            this.end = new Position(c as number, d as number);
        } else {
            this.start = a;
            this.end = b as Position;
        }
    }
}

const makeUri = (fsPath: string, scheme = "file") => ({
    fsPath,
    path: fsPath,
    scheme,
    toString: () => `${scheme}://${fsPath}`,
});

export const Uri = {
    file: (p: string) => makeUri(p, "file"),
};

export class Location {
    constructor(
        readonly uri: any,
        readonly range: Range,
    ) {}
}

export class Disposable {
    constructor(private readonly _fn?: () => void) {}
    dispose(): void {
        this._fn?.();
    }
}

export class EventEmitter<T = any> {
    private listeners: Array<(e: T) => any> = [];
    event = (listener: (e: T) => any): Disposable => {
        this.listeners.push(listener);
        return new Disposable(() => {
            const i = this.listeners.indexOf(listener);
            if (i >= 0) this.listeners.splice(i, 1);
        });
    };
    fire(data: T): void {
        [...this.listeners].forEach((l) => l(data));
    }
    dispose(): void {
        this.listeners = [];
    }
}

export class CodeLens {
    range: Range;
    command?: any;
    constructor(range: Range, command?: any) {
        this.range = range;
        this.command = command;
    }
    get isResolved(): boolean {
        return !!this.command;
    }
}

export class DocumentSymbol {
    children: DocumentSymbol[] = [];
    constructor(
        public name: string,
        public detail: string,
        public kind: number,
        public range: Range,
        public selectionRange: Range,
    ) {}
}

// Real vscode numeric SymbolKind enum values.
export const SymbolKind = {
    File: 0,
    Module: 1,
    Namespace: 2,
    Package: 3,
    Class: 4,
    Method: 5,
    Property: 6,
    Field: 7,
    Constructor: 8,
    Enum: 9,
    Interface: 10,
    Function: 11,
    Variable: 12,
    Constant: 13,
    String: 14,
    Number: 15,
    Boolean: 16,
    Array: 17,
    Object: 18,
    Key: 19,
    Null: 20,
    EnumMember: 21,
    Struct: 22,
    Event: 23,
    Operator: 24,
    TypeParameter: 25,
};

export const ConfigurationTarget = {
    Global: 1,
    Workspace: 2,
    WorkspaceFolder: 3,
};

export class MarkdownString {
    value = "";
    appendCodeblock(code: string, lang?: string): this {
        this.value += "```" + (lang ?? "") + "\n" + code + "\n```";
        return this;
    }
    appendMarkdown(md: string): this {
        this.value += md;
        return this;
    }
}

export class Hover {
    contents: any[];
    constructor(
        contents: any,
        readonly range?: Range,
    ) {
        this.contents = Array.isArray(contents) ? contents : [contents];
    }
}

// --- DocumentSymbolProvider registry + open-document tracking ---
interface RegisteredProvider {
    provider: { provideDocumentSymbols: (doc: any) => any };
    disposed: boolean;
}
const symbolProviders: RegisteredProvider[] = [];
const openDocuments = new Map<string, any>();
let docCounter = 0;

export const languages = {
    registerDocumentSymbolProvider(_selector: any, provider: any): Disposable {
        const entry: RegisteredProvider = { provider, disposed: false };
        symbolProviders.push(entry);
        return new Disposable(() => {
            entry.disposed = true;
        });
    },
};

export const commands = {
    async executeCommand(command: string, ...args: any[]): Promise<any> {
        if (command !== "vscode.executeDocumentSymbolProvider")
            return undefined;
        const uri = args[0];
        const key = uri?.toString?.() ?? String(uri);
        const active = symbolProviders.filter((p) => !p.disposed);
        if (active.length === 0) return undefined;
        const doc = openDocuments.get(key) ?? uri;
        return active[active.length - 1].provider.provideDocumentSymbols(doc);
    },
};

const configStore = new Map<string, any>();

function wordRangeAt(
    lines: string[],
    line: number,
    character: number,
): Range | undefined {
    const text = lines[line] ?? "";
    const re = /\w+/g;
    let m: RegExpExecArray | null;
    while ((m = re.exec(text)) !== null) {
        const start = m.index;
        const end = start + m[0].length;
        if (character >= start && character <= end)
            return new Range(line, start, line, end);
    }
    return undefined;
}

export const workspace = {
    getConfiguration(section: string) {
        return {
            get<T>(key: string, dflt?: T): T {
                const full = `${section}.${key}`;
                return configStore.has(full)
                    ? configStore.get(full)
                    : (dflt as T);
            },
            update(key: string, value: any, _target?: any): Promise<void> {
                const full = `${section}.${key}`;
                if (value === undefined) configStore.delete(full);
                else configStore.set(full, value);
                return Promise.resolve();
            },
        };
    },
    async openTextDocument(arg: any): Promise<any> {
        const content: string = arg?.content ?? "";
        const language: string = arg?.language ?? "plaintext";
        const uri = makeUri(`/untitled-${++docCounter}`, "untitled");
        const lines = content.split("\n");
        const doc = {
            uri,
            languageId: language,
            version: 1,
            lineCount: lines.length,
            getText: (range?: Range) =>
                range
                    ? (lines[range.start.line] ?? "").substring(
                          range.start.character,
                          range.end.character,
                      )
                    : content,
            getWordRangeAtPosition: (p: Position) =>
                wordRangeAt(lines, p.line, p.character),
        };
        openDocuments.set(uri.toString(), doc);
        return doc;
    },
};

export const window: { activeTextEditor: any } = {
    activeTextEditor: undefined,
};

// Proposed/optional chat API: guarded SUT paths treat undefined as "unavailable".
export const chat: any = undefined;
