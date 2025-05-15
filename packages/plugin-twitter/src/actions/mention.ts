import {
    type Action,
    type IAgentRuntime,
    type Memory,
    type State,
    elizaLogger,
    type ClientInstance,
} from "@elizaos/core";
import { Scraper } from "agent-twitter-client";
import { logTweetDetection } from "../tweetLogger";
import { Tweet, TweetSchema } from "../types";

export const mentionAction: Action = {
    name: "HANDLE_MENTION",
    similes: ["MENTION", "TAG", "TAGGED"],
    description: "Handle when the agent is mentioned in a tweet",
    validate: async (
        runtime: IAgentRuntime,
        _message: Memory,
        _state?: State
    ) => {
        const username = runtime.getSetting("TWITTER_USERNAME");
        const password = runtime.getSetting("TWITTER_PASSWORD");
        const email = runtime.getSetting("TWITTER_EMAIL");
        const hasCredentials = !!username && !!password && !!email;
        elizaLogger.log(`Has credentials: ${hasCredentials}`);

        return hasCredentials;
    },
    handler: async (
        runtime: IAgentRuntime,
        message: Memory,
        state?: State
    ): Promise<boolean> => {
        try {
            const client = runtime.clients.find((c) => ((c as unknown) as { name: string; client: { twitterClient?: any } }).name === 'twitter');
            if (!client) {
                elizaLogger.error("Twitter client not found");
                return false;
            }
            const twitterClient = ((client as unknown) as { client: { twitterClient?: any } }).client.twitterClient;
            const scraper = twitterClient || new Scraper();

            if (!twitterClient) {
                const username = runtime.getSetting("TWITTER_USERNAME");
                const password = runtime.getSetting("TWITTER_PASSWORD");
                const email = runtime.getSetting("TWITTER_EMAIL");
                const twitter2faSecret = runtime.getSetting("TWITTER_2FA_SECRET");

                if (!username || !password) {
                    elizaLogger.error(
                        "Twitter credentials not configured in environment"
                    );
                    return false;
                }
                // Login with credentials
                await scraper.login(username, password, email, twitter2faSecret);
                if (!(await scraper.isLoggedIn())) {
                    elizaLogger.error("Failed to login to Twitter");
                    return false;
                }
            }

            // Log the mention
            if (message.content?.text) {
                const tweetContent = TweetSchema.parse({ text: message.content.text });
                const agentUsername = runtime.getSetting("TWITTER_USERNAME")?.toLowerCase();
                const mentions = message.content.text.match(/@(\w+)/g)?.map(m => m.slice(1).toLowerCase()) || [];

                const tweet: Tweet = {
                    id: message.content.inReplyTo || message.id,
                    text: tweetContent.text,
                    username: message.content.source || 'unknown',
                    mentions: mentions.map(username => ({
                        id: '',
                        username,
                        name: username
                    })),
                    isReply: !!message.content.inReplyTo,
                    isRetweet: false,
                    isQuoted: false,
                    thread: [],
                    hashtags: [],
                    photos: [],
                    urls: [],
                    videos: []
                };
                logTweetDetection(tweet);

                elizaLogger.info(`Processing tweet ${tweet.id}`);
                    // Check if this is a retweet request
                    if (tweetContent.text.toLowerCase().includes('retweet') || tweetContent.text.toLowerCase().includes('rt')) {
                        try {
                            await scraper.retweet(tweet.id);
                            elizaLogger.info(`Retweeted tweet ${tweet.id}`);
                        } catch (error) {
                            elizaLogger.error(`Failed to retweet ${tweet.id}:`, error);
                    }
                }
            }

            return true;
        } catch (error) {
            elizaLogger.error("Error handling mention:", error);
            return false;
        }
    },
    examples: [
        [
            {
                user: "{{user1}}",
                content: { text: "@agent_name Hey there!" },
            },
            {
                user: "{{agentName}}",
                content: {
                    text: "I noticed I was mentioned in a tweet!",
                    action: "HANDLE_MENTION",
                },
            },
        ],
        [
            {
                user: "{{user1}}",
                content: { text: "Check this out @agent_name" },
            },
            {
                user: "{{agentName}}",
                content: {
                    text: "Someone tagged me in their tweet.",
                    action: "HANDLE_MENTION",
                },
            },
        ],
    ],
}; 