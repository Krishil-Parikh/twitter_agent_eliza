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
                text: message.content.text || '',
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
            const maxLength = Number(maxTweetLength);
            if (!isNaN(maxLength) && maxLength > 0) {
                trimmedContent = truncateToCompleteSentence(trimmedContent, maxLength);
            } else {
                // Fallback to default if MAX_TWEET_LENGTH setting is invalid
                trimmedContent = truncateToCompleteSentence(trimmedContent, DEFAULT_MAX_TWEET_LENGTH);
            }
        } else {
            // Apply default tweet length if no setting exists
            trimmedContent = truncateToCompleteSentence(trimmedContent, DEFAULT_MAX_TWEET_LENGTH);
        }

        // Ensure structured action output
        const actionOutput = `HANDLE_MENTION: ${trimmedContent}`;
        if (state) {
            state.response = {
                user: runtime.getSetting("TWITTER_USERNAME") || "unknown",
                content: { 
                    text: trimmedContent, 
                    action: "HANDLE_MENTION",
                    inReplyTo: message.content?.inReplyTo
                }
            };
        }

        // Log the response
        if (message.content?.inReplyTo) {
            logTweetResponse({
                id: message.content.inReplyTo,
                text: message.content.text || '',
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

async function sendTweet(twitterClient: Scraper, content: string): Promise<boolean> {
    try {
    const result = await twitterClient.sendTweet(content);
        
        // Handle cases where result might not be a proper Response object
        if (!result || typeof result.json !== 'function') {
            elizaLogger.error("Invalid response from sendTweet", result);
            return false;
        }

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

        // Check for successful tweet creation with more robust validation
    if (!body?.data?.create_tweet?.tweet_results?.result) {
        elizaLogger.error("Failed to post tweet: No tweet result in response");
        return false;
    }

    return true;
    } catch (error) {
        elizaLogger.error("Error in sendTweet:", error);
        return false;
    }
}

async function postTweet(
    runtime: IAgentRuntime,
    content: string
): Promise<boolean> {
    const MAX_RETRIES = 3;
    let retryCount = 0;
    
    while (retryCount < MAX_RETRIES) {
    try {
            const client = runtime.clients.find(c => ((c as unknown) as { name: string; client: { twitterClient?: any } }).name === 'twitter');
            const twitterClient = client ? ((client as unknown) as { client: { twitterClient?: any } }).client.twitterClient : null;
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
                    if (retryCount < MAX_RETRIES - 1) {
                        retryCount++;
                        elizaLogger.info(`Retrying login attempt ${retryCount} of ${MAX_RETRIES}`);
                        await new Promise(resolve => setTimeout(resolve, 1000 * retryCount)); // Exponential backoff
                        continue;
                    }
                    return false;
                }
            } else {
                // Verify the session is still valid
                const isLoggedIn = await scraper.isLoggedIn();
                if (!isLoggedIn) {
                    elizaLogger.warn("Twitter session expired, attempting to re-login");
                    const username = runtime.getSetting("TWITTER_USERNAME");
                    const password = runtime.getSetting("TWITTER_PASSWORD");
                    const email = runtime.getSetting("TWITTER_EMAIL");
                    const twitter2faSecret = runtime.getSetting("TWITTER_2FA_SECRET");
                    
                    await scraper.login(username, password, email, twitter2faSecret);
                    if (!(await scraper.isLoggedIn())) {
                        elizaLogger.error("Failed to re-login to Twitter");
                        if (retryCount < MAX_RETRIES - 1) {
                            retryCount++;
                            elizaLogger.info(`Retrying login attempt ${retryCount} of ${MAX_RETRIES}`);
                            await new Promise(resolve => setTimeout(resolve, 1000 * retryCount));
                            continue;
                        }
                        return false;
                    }
                }
            }

            // Clean up content to ensure it's valid
            const cleanContent = content.trim();
            if (!cleanContent) {
                elizaLogger.error("Tweet content is empty after trimming");
                return false;
        }

        // Send the tweet
            elizaLogger.log("Attempting to send tweet:", cleanContent);

        try {
                if (cleanContent.length > DEFAULT_MAX_TWEET_LENGTH) {
                    elizaLogger.info(`Tweet length (${cleanContent.length}) exceeds standard limit, attempting Note Tweet`);
                    const noteTweetResult = await scraper.sendNoteTweet(cleanContent);
                    
                    // Check if Note Tweet was successful
                    if (noteTweetResult && noteTweetResult.errors && noteTweetResult.errors.length > 0) {
                    // Note Tweet failed due to authorization. Falling back to standard Tweet.
                        elizaLogger.warn("Note Tweet failed, falling back to standard tweet (may be truncated)");
                        const truncatedContent = truncateToCompleteSentence(cleanContent, DEFAULT_MAX_TWEET_LENGTH);
                        return await sendTweet(scraper, truncatedContent);
                }
                return true;
            }
                return await sendTweet(scraper, cleanContent);
        } catch (error) {
                elizaLogger.error(`Tweet posting failed: ${error.message}`);
                if (retryCount < MAX_RETRIES - 1) {
                    retryCount++;
                    elizaLogger.info(`Retrying tweet posting, attempt ${retryCount} of ${MAX_RETRIES}`);
                    await new Promise(resolve => setTimeout(resolve, 1000 * retryCount));
                    continue;
                }
                throw new Error(`Failed to post tweet after ${MAX_RETRIES} attempts: ${error.message}`);
        }
    } catch (error) {
        // Log the full error details
        elizaLogger.error("Error posting tweet:", {
            message: error.message,
            stack: error.stack,
            name: error.name,
            cause: error.cause,
        });
            
            if (retryCount < MAX_RETRIES - 1) {
                retryCount++;
                await new Promise(resolve => setTimeout(resolve, 1000 * retryCount));
                continue;
            }
        return false;
    }
    }
    
    return false; // Should not reach here, but added for safety
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

            // Validate that message has text content
            if (!message.content?.text) {
                elizaLogger.warn("No text content in message to post as tweet");
                return false;
            }

            // Parse and validate tweet content
            const tweetContent = TweetSchema.parse({ text: message.content.text });
            if (!tweetContent.text.trim()) {
                elizaLogger.warn("Tweet content is empty after trimming");
                return false;
            }

            // Get Twitter client
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
            } else {
                // Verify the session is still valid
                const isLoggedIn = await scraper.isLoggedIn();
                if (!isLoggedIn) {
                    elizaLogger.warn("Twitter session expired, attempting to re-login");
                    const username = runtime.getSetting("TWITTER_USERNAME");
                    const password = runtime.getSetting("TWITTER_PASSWORD");
                    const email = runtime.getSetting("TWITTER_EMAIL");
                    const twitter2faSecret = runtime.getSetting("TWITTER_2FA_SECRET");
                    
                    await scraper.login(username, password, email, twitter2faSecret);
                    if (!(await scraper.isLoggedIn())) {
                        elizaLogger.error("Failed to re-login to Twitter");
                        return false;
                    }
                }
            }

            // Create tweet object
                const tweet: Tweet = {
                id: message.id || Date.now().toString(),
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

            // Post the tweet with retry logic
            let success = false;
            let retryCount = 0;
            const MAX_RETRIES = 3;
            
            while (!success && retryCount < MAX_RETRIES) {
                try {
                await scraper.tweet(tweet.text);
                    success = true;
                    elizaLogger.info(`Successfully posted tweet: ${tweet.text}`);
                logTweetDetection(tweet);
                } catch (error) {
                    retryCount++;
                    elizaLogger.error(`Failed to post tweet (attempt ${retryCount}): ${error.message}`);
                    if (retryCount < MAX_RETRIES) {
                        await new Promise(resolve => setTimeout(resolve, 1000 * retryCount)); // Exponential backoff
                    }
                }
            }

            if (!success) {
                elizaLogger.error(`Failed to post tweet after ${MAX_RETRIES} attempts`);
                return false;
            }

            if (state) {
                state.response = {
                    user: runtime.getSetting("TWITTER_USERNAME") || "unknown",
                    content: {
                        text: `Successfully posted: "${tweet.text}"`,
                        action: "POST_TWEET"
                    }
                };
            }

            return true;
        } catch (error) {
            elizaLogger.error("Error posting tweet:", {
                message: error.message,
                stack: error.stack,
                name: error.name,
                cause: error.cause
            });
            return false;
        }
    },
    examples: [
        [
            {
                user: "{{user1}}",
                content: { text: "Post this update: We're excited to announce our new feature!" },
            },
            {
                user: "{{agentName}}",
                content: {
                    text: "Successfully posted the tweet about the new feature announcement!",
                    action: "POST_TWEET"
                },
            },
        ],
    ],
};