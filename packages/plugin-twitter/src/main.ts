import { AgentRuntime, elizaLogger, ModelProviderName } from "@elizaos/core";
import { twitterPlugin } from "./index";
import { Scraper } from "agent-twitter-client";
import dotenv from 'dotenv';
import { IAgentRuntime } from "@elizaos/core";
import { TwitterApi } from "twitter-api-v2";
import path from 'path';
import { fileURLToPath } from 'url';
import { character } from "../../../agent/src/character";

// ESM __dirname workaround
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Load environment variables from multiple possible locations
const envPaths = [
    path.resolve(process.cwd(), '.env'),
    path.resolve(__dirname, '../.env'),
    path.resolve(process.cwd(), 'packages/plugin-twitter/.env')
];

let envLoaded = false;
for (const envPath of envPaths) {
    const result = dotenv.config({ path: envPath });
    if (!result.error) {
        elizaLogger.info(`Loaded .env from ${envPath}`);
        envLoaded = true;
        break;
    }
}

if (!envLoaded) {
    elizaLogger.error("Failed to load .env file from any location");
}

// Set log level to debug to see all logs
process.env.DEFAULT_LOG_LEVEL = "debug";

// Log environment variables for debugging
elizaLogger.info("Twitter environment variables:", {
    username: process.env.TWITTER_USERNAME,
    apiKey: process.env.TWITTER_API_KEY ? "set" : "not set",
    apiSecret: process.env.TWITTER_API_SECRET ? "set" : "not set",
    accessToken: process.env.TWITTER_ACCESS_TOKEN ? "set" : "not set",
    accessSecret: process.env.TWITTER_ACCESS_TOKEN_SECRET ? "set" : "not set"
});

async function main() {
    try {
        elizaLogger.info("Starting Twitter integration...");

        // Initialize Twitter client
        elizaLogger.info("Initializing Twitter API client...");
        const client = new TwitterApi({
            appKey: process.env.TWITTER_API_KEY || '',
            appSecret: process.env.TWITTER_API_SECRET || '',
            accessToken: process.env.TWITTER_ACCESS_TOKEN || '',
            accessSecret: process.env.TWITTER_ACCESS_TOKEN_SECRET || '',
        });

        // Initialize scraper for login
        elizaLogger.info("Initializing Twitter scraper...");
        const scraper = new Scraper();

        // Try to log in
        elizaLogger.info("Attempting to log in to Twitter...");
        try {
            elizaLogger.info(`Using credentials for username: ${process.env.TWITTER_USERNAME}`);
            await scraper.login(
                process.env.TWITTER_USERNAME || '',
                process.env.TWITTER_PASSWORD || '',
                process.env.TWITTER_EMAIL || ''
            );
            
            // Verify login status
            const isLoggedIn = await scraper.isLoggedIn();
            elizaLogger.info(`Login status check: ${isLoggedIn ? "successful" : "failed"}`);
            
            if (!isLoggedIn) {
                throw new Error("Login verification failed - isLoggedIn check returned false");
            }
            
            elizaLogger.info("Successfully logged in to Twitter");
        } catch (error) {
            elizaLogger.error("Failed to log in to Twitter:", {
                error: error.message,
                stack: error.stack,
                username: process.env.TWITTER_USERNAME,
                email: process.env.TWITTER_EMAIL
            });
            throw error;
        }

        // Verify API credentials
        elizaLogger.info("Verifying Twitter API credentials...");
        try {
            const me = await client.v2.me();
            elizaLogger.info("Successfully verified Twitter API credentials");
            elizaLogger.info(`Connected as: ${me.data.username}`);
        } catch (error) {
            elizaLogger.error("Failed to verify Twitter API credentials:", {
                error: error.message,
                stack: error.stack
            });
            throw error;
        }

        // Initialize agent runtime
        elizaLogger.info("Initializing agent runtime...");
        const runtime = new AgentRuntime({
            character: character,
            token: process.env.TWITTER_USERNAME || "",
            agentId: "123e4567-e89b-12d3-a456-426614174000" as `${string}-${string}-${string}-${string}-${string}`,
            actions: [...twitterPlugin.actions],
            modelProvider: ModelProviderName.GOOGLE,
            providers: twitterPlugin.providers || [],
            logging: true
        });

        // Start mention polling
        elizaLogger.info("Starting mention polling...");
        const pollInterval = parseInt(process.env.TWITTER_POLL_INTERVAL || "10") * 1000; // Convert to milliseconds
        const conversationContext = new Map<string, any>();
        await startMentionPolling(runtime, client, conversationContext, pollInterval);

        elizaLogger.info("Twitter integration initialized successfully");
    } catch (error) {
        elizaLogger.error("Failed to initialize Twitter integration:", {
            error: error.message,
            stack: error.stack
        });
        throw error;
    }
}

// Run the agent
main().catch(error => elizaLogger.error("Agent startup failed:", error));

// Initialize Twitter client
export async function initializeTwitter(runtime: IAgentRuntime) {
    try {
        // Initialize Twitter client
        const client = new TwitterApi({
            appKey: process.env.TWITTER_API_KEY!,
            appSecret: process.env.TWITTER_API_SECRET!,
            accessToken: process.env.TWITTER_ACCESS_TOKEN!,
            accessSecret: process.env.TWITTER_ACCESS_TOKEN_SECRET!,
        });

        // Track conversation context
        const conversationContext = new Map<string, any>();

        // Start polling for mentions and replies
        const mentionPollingInterval = await startMentionPolling(runtime, client, conversationContext);
        const replyPollingInterval = await startReplyPolling(runtime, client, conversationContext);

        return {
            stop: async () => {
                // Clean up intervals
                if (mentionPollingInterval) {
                    clearInterval(mentionPollingInterval);
                }
                if (replyPollingInterval) {
                    clearInterval(replyPollingInterval);
                }
            }
        };
    } catch (error) {
        console.error("Error initializing Twitter:", error);
    }
}

// Function to handle incoming tweets
async function handleTweet(tweet: any, runtime: IAgentRuntime, twitterClient: TwitterApi, conversationContext: Map<string, any>) {
    try {
        // Get conversation context
        const context = conversationContext.get(tweet.conversation_id) || {};
        
        // Process the tweet
        const message = {
            content: { text: tweet.text },
            source: 'twitter',
            metadata: {
                tweetId: tweet.id,
                author: tweet.author_id,
                timestamp: tweet.created_at,
                conversationId: tweet.conversation_id,
                context: context
            },
            userId: runtime.agentId,
            agentId: runtime.agentId,
            roomId: runtime.agentId
        };
        await runtime.messageManager.createMemory(message);

        if (message.content.text) {
            // Update conversation context
            conversationContext.set(tweet.conversation_id, {
                ...context,
                lastTweetId: tweet.id,
                lastResponse: message.content.text
            });

            // Reply to the tweet
            await twitterClient.v2.reply(message.content.text, tweet.id);
        }
    } catch (error) {
        console.error('Error processing tweet:', error);
    }
}

// Set up polling for mentions
export async function startMentionPolling(runtime: IAgentRuntime, twitterClient: TwitterApi, conversationContext: Map<string, any>, pollInterval: number = 60000) {
    try {
        // Get initial mentions
        const mentions = await twitterClient.v2.userMentionTimeline(process.env.TWITTER_USER_ID!);
        
        // Process each mention
        for await (const tweet of mentions) {
            await handleTweet(tweet, runtime, twitterClient, conversationContext);
        }

        // Set up polling interval
        return setInterval(async () => {
            try {
                const mentions = await twitterClient.v2.userMentionTimeline(process.env.TWITTER_USER_ID!);
                for await (const tweet of mentions) {
                    await handleTweet(tweet, runtime, twitterClient, conversationContext);
                }
            } catch (error) {
                console.error('Error polling mentions:', error);
            }
        }, pollInterval);
    } catch (error) {
        console.error('Error starting mention polling:', error);
    }
}

// Set up polling for replies
async function startReplyPolling(runtime: IAgentRuntime, twitterClient: TwitterApi, conversationContext: Map<string, any>, pollInterval: number = 60000) {
    try {
        // Get initial replies
        const replies = await twitterClient.v2.userTimeline(process.env.TWITTER_USER_ID!);
        
        // Process each reply
        for await (const tweet of replies) {
            if (tweet.in_reply_to_user_id === process.env.TWITTER_USER_ID) {
                await handleTweet(tweet, runtime, twitterClient, conversationContext);
            }
        }

        // Set up polling interval
        return setInterval(async () => {
            try {
                const replies = await twitterClient.v2.userTimeline(process.env.TWITTER_USER_ID!);
                for await (const tweet of replies) {
                    if (tweet.in_reply_to_user_id === process.env.TWITTER_USER_ID) {
                        await handleTweet(tweet, runtime, twitterClient, conversationContext);
                    }
                }
            } catch (error) {
                console.error('Error polling replies:', error);
            }
        }, pollInterval);
    } catch (error) {
        console.error('Error starting reply polling:', error);
    }
}