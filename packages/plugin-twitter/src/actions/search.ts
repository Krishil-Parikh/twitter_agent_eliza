import {
    type Action,
    type IAgentRuntime,
    type Memory,
    type State,
    elizaLogger,
    ServiceType,
    type IImageDescriptionService,
    type ActionExample,
    Content,
} from "@elizaos/core";
import { Scraper } from "agent-twitter-client";
import { Tweet } from "../types";

export const searchAction: Action = {
    name: "search",
    similes: ["search tweets", "find tweets", "look up tweets"],
    description: "Search for tweets matching the given query",
    examples: [
        [
            {
                user: "user1",
                content: {
                    text: "search for tweets about AI"
                }
            },
            {
                user: "user1",
                content: {
                    text: "Here are some recent tweets about AI: [tweet results]"
                }
            }
        ],
        [
            {
                user: "user2",
                content: {
                    text: "find tweets from @elonmusk"
                }
            },
            {
                user: "user2",
                content: {
                    text: "Here are recent tweets from @elonmusk: [tweet results]"
                }
            }
        ],
        [
            {
                user: "user3",
                content: {
                    text: "look up tweets about climate change"
                }
            },
            {
                user: "user3",
                content: {
                    text: "Here are some tweets about climate change: [tweet results]"
                }
            }
        ]
    ],
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

            // Get search terms from message
            const searchTerms = message.content?.text || "";
            if (!searchTerms) {
                elizaLogger.error("No search terms provided");
                return false;
            }

            // Search for tweets
            const tweets = await scraper.search(searchTerms);
            
            // Process tweets
            for (const tweet of tweets) {
                // Check if tweet has images
                if (tweet.photos && tweet.photos.length > 0) {
                    try {
                        // Try to get image description service
                        const imageDescriptionService = runtime.getService<IImageDescriptionService>(ServiceType.IMAGE_DESCRIPTION);
                        
                        if (imageDescriptionService) {
                            // If service is available, describe images
                            for (const photo of tweet.photos) {
                                try {
                                    const description = await imageDescriptionService.describeImage(photo.url);
                                    elizaLogger.info(`Image description: ${description.title}`);
                                } catch (error) {
                                    elizaLogger.warn(`Failed to describe image: ${error}`);
                                }
                            }
                        } else {
                            // If service is not available, log a warning
                            elizaLogger.warn("Image description service not available - skipping image analysis");
                        }
                    } catch (error) {
                        elizaLogger.warn(`Error processing images: ${error}`);
                    }
                }
            }

            return true;
        } catch (error) {
            elizaLogger.error("Error searching tweets:", error);
            return false;
        }
    },
}; 