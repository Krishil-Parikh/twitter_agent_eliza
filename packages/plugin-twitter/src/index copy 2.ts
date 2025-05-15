import type { Plugin, IAgentRuntime } from "@elizaos/core";
import { postAction } from "./actions/post";
import { mentionAction } from "./actions/mention";
import { searchAction } from "./actions/search";
import { elizaLogger } from "@elizaos/core";

// Define the Twitter plugin
export const twitterPlugin: Plugin = {
    name: "twitter",
    description: "Twitter integration plugin for posting tweets and handling mentions",
    actions: [postAction, mentionAction, searchAction],
    evaluators: [],
    providers: [],
};

// Track polling state
let mentionPollingInterval: NodeJS.Timeout | null = null;

// Function to start polling for mentions with exponential backoff
export function startMentionPolling(runtime: IAgentRuntime, initialIntervalMs: number = 60 * 1000) {
    // Stop any existing polling
    if (mentionPollingInterval) {
        clearInterval(mentionPollingInterval);
        mentionPollingInterval = null;
    }
    
    // Track consecutive failures for exponential backoff
    let consecutiveFailures = 0;
    let currentInterval = initialIntervalMs;
    
    // Define the polling function separately so we can reference it when resetting intervals
    async function mentionPoll() {
        try {
            elizaLogger.info(`Polling for mentions (interval: ${currentInterval}ms)...`);
            
            const dummyMessage = { 
                id: "123e4567-e89b-12d3-a456-426614174000" as `${string}-${string}-${string}-${string}-${string}`,
                userId: "123e4567-e89b-12d3-a456-426614174000" as `${string}-${string}-${string}-${string}-${string}`,
                agentId: runtime.agentId,
                roomId: "123e4567-e89b-12d3-a456-426614174000" as `${string}-${string}-${string}-${string}-${string}`,
                content: { text: "" } 
            };
            
            const success = await mentionAction.handler(runtime, dummyMessage);
            
            // Reset backoff on success
            if (success) {
                consecutiveFailures = 0;
                if (currentInterval !== initialIntervalMs) {
                    currentInterval = initialIntervalMs;
                    elizaLogger.info(`Reset polling interval to ${currentInterval}ms after successful poll`);
                }
            } else {
                consecutiveFailures++;
                elizaLogger.warn(`Mention polling failed (failure #${consecutiveFailures})`);
            }
        } catch (error) {
            consecutiveFailures++;
            elizaLogger.error(`Error in mention polling (failure #${consecutiveFailures}):`, {
                message: error.message,
                stack: error.stack
            });
            
            // Implement exponential backoff for failures
            if (consecutiveFailures > 3) {
                const oldInterval = currentInterval;
                currentInterval = Math.min(currentInterval * 2, 15 * 60 * 1000); // Max 15 minutes
                
                if (oldInterval !== currentInterval) {
                    elizaLogger.warn(`Increased polling interval to ${currentInterval}ms due to consecutive failures`);
                }
            }
        }
    }
    
    // Start the polling interval
    mentionPollingInterval = setInterval(mentionPoll, currentInterval);
    elizaLogger.info(`Started mention polling with initial interval of ${initialIntervalMs}ms`);
}

// Function to stop polling
export function stopMentionPolling() {
    if (mentionPollingInterval) {
        clearInterval(mentionPollingInterval);
        mentionPollingInterval = null;
        elizaLogger.info("Stopped mention polling");
    }
}