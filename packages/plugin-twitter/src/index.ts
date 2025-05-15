import type { Plugin, IAgentRuntime, ClientInstance } from "@elizaos/core";
import { postAction } from "./actions/post";
import { mentionAction } from "./actions/mention";
import { searchAction } from "./actions/search";
import { TwitterApi } from "twitter-api-v2";
import { initializeTwitter, startMentionPolling } from "./main";

export { startMentionPolling };

export const twitterPlugin: Plugin = {
    name: "twitter",
    description: "Twitter integration plugin for posting tweets and handling mentions",
    actions: [postAction, mentionAction, searchAction],
    clients: [{
        name: "twitter",
        config: {
            POST_IMMEDIATELY: true,
            ENABLE_TWITTER_POST_GENERATION: true,
            ENABLE_ACTION_PROCESSING: true,
            POST_INTERVAL_MIN: 30,
            POST_INTERVAL_MAX: 120,
            ACTION_INTERVAL: 5,
            MAX_TWEET_LENGTH: 280
        },
        start: async (runtime: IAgentRuntime): Promise<ClientInstance> => {
            return await initializeTwitter(runtime);
        }
    }]
};