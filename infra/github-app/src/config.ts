/**
 * App identity/secrets, read from env only — never hardcoded, never
 * committed. See README.md for how to obtain each value at registration
 * time (GitHub issues appId/privateKey/webhookSecret when the App is
 * created from manifest.yml).
 */
export interface AppConfig {
    appId: string;
    privateKey: string;
    webhookSecret: string;
    port: number;
}

function requireEnv(name: string): string {
    const value = process.env[name];
    if (!value) {
        throw new Error(`Missing required environment variable: ${name}`);
    }
    return value;
}

export function loadConfig(): AppConfig {
    return {
        appId: requireEnv("GITHUB_APP_ID"),
        privateKey: requireEnv("GITHUB_APP_PRIVATE_KEY"),
        webhookSecret: requireEnv("GITHUB_WEBHOOK_SECRET"),
        port: Number(process.env.PORT ?? 3000),
    };
}
