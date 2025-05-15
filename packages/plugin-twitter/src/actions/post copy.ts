import {
    type Action,
    type IAgentRuntime,
    type Memory,
    type State,
    composeContext,
    elizaLogger,
    ModelClass,
    generateObject,
    truncateToCompleteSentence,
    type ClientInstance,
    ServiceType,
} from "@elizaos/core";
import { Scraper } from "agent-twitter-client";
import { tweetTemplate } from "../templates";
import { isTweetContent, TweetSchema, Tweet } from "../types";
import { logTweetDetection, logTweetResponse } from "../tweetLogger";

export const DEFAULT_MAX_TWEET_LENGTH = 280;
const AUTO_POST_INTERVAL = 5 * 60 * 1000; // 5 minutes
let lastAutoPostTime = 0;

async function checkAutoPost(runtime: IAgentRuntime): Promise<void> {
    const now = Date.now();
    if (now - lastAutoPostTime >= AUTO_POST_INTERVAL) {
        try {
            const content = await composeTweet(runtime, {
                id: '00000000-0000-0000-0000-000000000000',
                userId: '00000000-0000-0000-0000-000000000000',
                agentId: '00000000-0000-0000-0000-000000000000',
                roomId: '00000000-0000-0000-0000-000000000000',
                content: { text: '' }
            } as Memory);
            if (content) {
                await postTweet(runtime, content);
                lastAutoPostTime = now;
            }
        } catch (error) {
            elizaLogger.error("Error in auto-post:", error);
        }
    }
}

export async function composeTweet(
    runtime: IAgentRuntime,
    message: Memory,
    state?: State
): Promise<string> {
    try {
        // Log the incoming tweet if it exists
        if (message.content?.inReplyTo) {
            logTweetDetection({
                id: message.content.inReplyTo,
                text: message.content.text,
                username: message.content.source || 'unknown',
                mentions: [],
                isReply: true,
                isRetweet: false,
                isQuoted: false,
                thread: [],
                hashtags: [],
                photos: [],
                urls: [],
                videos: []
            });
        }

        const context = composeContext({
            state,
            template: tweetTemplate,
        });

        // Check if image description service is available
        try {
            const imageService = runtime.services.get(ServiceType.IMAGE_DESCRIPTION);
            if (!imageService) {
                elizaLogger.warn("Image description service not available, proceeding without image analysis");
            }
        } catch (error) {
            elizaLogger.warn("Error checking image description service:", error);
        }

        const tweetContentObject = await generateObject({
            runtime,
            context,
            modelClass: ModelClass.SMALL,
            schema: TweetSchema as any,
            stop: ["\n"],
        });

        if (!isTweetContent(tweetContentObject.object)) {
            elizaLogger.error("Invalid tweet content:", tweetContentObject.object);
            return "";
        }

        let trimmedContent = tweetContentObject.object.text.trim();
        const maxTweetLength = runtime.getSetting("MAX_TWEET_LENGTH");
        if (maxTweetLength) {
            trimmedContent = truncateToCompleteSentence(trimmedContent, Number(maxTweetLength));
        }

        // Ensure structured action output
        const actionOutput = `HANDLE_MENTION: ${trimmedContent}`;
        if (state) {
            state.response = {
                user: runtime.getSetting("TWITTER_USERNAME") || "unknown",
                content: { text: trimmedContent, action: "HANDLE_MENTION" }
            };
        }

        // Log the response
        if (message.content?.inReplyTo) {
            logTweetResponse({
                id: message.content.inReplyTo,
                text: message.content.text,
                username: message.content.source || 'unknown',
                mentions: [],
                isReply: true,
                isRetweet: false,
                isQuoted: false,
                thread: [],
                hashtags: [],
                photos: [],
                urls: [],
                videos: []
            }, trimmedContent);
        }

        return actionOutput;
    } catch (error) {
        elizaLogger.error("Error composing tweet:", error);
        throw error;
    }
}

async function sendTweet(twitterClient: Scraper, content: string) {
    const result = await twitterClient.sendTweet(content);

    const body = await result.json();
    elizaLogger.log("Tweet response:", body);

    // Check for Twitter API errors
    if (body.errors) {
        const error = body.errors[0];
        elizaLogger.error(
            `Twitter API error (${error.code}): ${error.message}`
        );
        return false;
    }

    // Check for successful tweet creation
    if (!body?.data?.create_tweet?.tweet_results?.result) {
        elizaLogger.error("Failed to post tweet: No tweet result in response");
        return false;
    }

    return true;
}

async function postTweet(
    runtime: IAgentRuntime,
    content: string
): Promise<boolean> {
    try {
        const twitterClient = ((runtime.clients.find(c => ((c as unknown) as { name: string; client: { twitterClient?: any } }).name === 'twitter') as unknown) as { client: { twitterClient?: any } })?.client.twitterClient;
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

        // Send the tweet
        elizaLogger.log("Attempting to send tweet:", content);

        try {
            if (content.length > DEFAULT_MAX_TWEET_LENGTH) {
                const noteTweetResult = await scraper.sendNoteTweet(content);
                if (noteTweetResult.errors && noteTweetResult.errors.length > 0) {
                    // Note Tweet failed due to authorization. Falling back to standard Tweet.
                    return await sendTweet(scraper, content);
                }
                return true;
            }
            return await sendTweet(scraper, content);
        } catch (error) {
            throw new Error(`Note Tweet failed: ${error}`);
        }
    } catch (error) {
        // Log the full error details
        elizaLogger.error("Error posting tweet:", {
            message: error.message,
            stack: error.stack,
            name: error.name,
            cause: error.cause,
        });
        return false;
    }
}

export const postAction: Action = {
    name: "POST_TWEET",
    similes: ["POST", "TWEET", "PUBLISH"],
    description: "Post a tweet to Twitter",
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
            // Check for auto-post
            await checkAutoPost(runtime);

            const client = runtime.clients.find(c => ((c as unknown) as { name: string; client: { twitterClient?: any } }).name === 'twitter');
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

            // Post the tweet
            if (message.content?.text) {
                const tweetContent = TweetSchema.parse({ text: message.content.text });
                const tweet: Tweet = {
                    id: message.id,
                    text: tweetContent.text,
                    username: runtime.getSetting("TWITTER_USERNAME") || 'unknown',
                    mentions: [],
                    isReply: false,
                    isRetweet: false,
                    isQuoted: false,
                    thread: [],
                    hashtags: [],
                    photos: [],
                    urls: [],
                    videos: []
                };
                await scraper.tweet(tweet.text);
                logTweetDetection(tweet);
            }

            return true;
        } catch (error) {
            elizaLogger.error("Error posting tweet:", error);
            return false;
        }
    },
    examples: [
        [
            {
                user: "{{agentName}}",
                content: { text: "I'm going to post a tweet!" },
            },
            {
                user: "{{agentName}}",
                content: {
                    text: "Just posted a new tweet!",
                    action: "POST_TWEET",
                },
            },
        ],
    ],
};
