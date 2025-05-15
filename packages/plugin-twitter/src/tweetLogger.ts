import { elizaLogger } from "@elizaos/core";
import { Tweet } from "agent-twitter-client";

export function logTweetDetection(tweet: Tweet) {
    elizaLogger.info("🔍 Tweet Detection:", {
        id: tweet.id,
        text: tweet.text,
        from: tweet.username,
        mentions: tweet.mentions,
        isReply: tweet.isReply,
        isRetweet: tweet.isRetweet,
        isQuoted: tweet.isQuoted
    });
}

export function logTweetResponse(originalTweet: Tweet, response: string) {
    elizaLogger.info("💬 Tweet Response:", {
        originalTweetId: originalTweet.id,
        originalTweet: originalTweet.text,
        from: originalTweet.username,
        response: response
    });
} 