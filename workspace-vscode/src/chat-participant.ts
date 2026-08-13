import * as vscode from "vscode";
import { createTrelixChatHandler, ChatHandlerDeps } from "./chat-handler";

/** Matches the id declared in package.json contributes.chatParticipants. */
const PARTICIPANT_ID = "trelix.chat";

/**
 * Register the trelix chat participant, if this VS Code build supports the chat
 * API. Thin adapter over createTrelixChatHandler — all logic lives there.
 *
 * BACKWARD COMPAT: the chat API landed in VS Code 1.95. On 1.90–1.94
 * `vscode.chat` is undefined, so the runtime guard returns undefined and the
 * extension activates cleanly without it. engines.vscode stays "^1.90.0".
 */
export function registerTrelixChatParticipant(
    context: vscode.ExtensionContext,
    deps: ChatHandlerDeps,
): vscode.Disposable | undefined {
    if (!vscode.chat?.createChatParticipant) return undefined;

    const participant = vscode.chat.createChatParticipant(
        PARTICIPANT_ID,
        createTrelixChatHandler(deps),
    );
    context.subscriptions.push(participant);
    return participant;
}
