import { TwitterApi } from 'twitter-api-v2';
import { elizaLogger } from '@elizaos/core';

export class AutoPoster {
    private client: TwitterApi;
    private interval: NodeJS.Timeout | null = null;
    private lastTweetTime: number = 0;
    private readonly MIN_INTERVAL = 60000; // 1 minute in milliseconds

    constructor(
        apiKey: string,
        apiSecret: string,
        accessToken: string,
        accessSecret: string
    ) {
        this.client = new TwitterApi({
            appKey: apiKey,
            appSecret: apiSecret,
            accessToken: accessToken,
            accessSecret: accessSecret,
        });
    }

    async start(contentGenerator: () => Promise<string>) {
        if (this.interval) {
            elizaLogger.warn('AutoPoster is already running');
            return;
        }

        // Initial tweet
        await this.postTweet(contentGenerator);

        // Set up interval
        this.interval = setInterval(async () => {
            await this.postTweet(contentGenerator);
        }, this.MIN_INTERVAL);

        elizaLogger.info('AutoPoster started');
    }

    private async postTweet(contentGenerator: () => Promise<string>) {
        try {
            const now = Date.now();
            if (now - this.lastTweetTime < this.MIN_INTERVAL) {
                elizaLogger.warn('Skipping tweet - too soon since last tweet');
                return;
            }

            const content = await contentGenerator();
            if (!content) {
                elizaLogger.warn('No content generated for tweet');
                return;
            }

            await this.client.v2.tweet(content);
            this.lastTweetTime = now;
            elizaLogger.info('Tweet posted successfully');
        } catch (error) {
            elizaLogger.error('Error posting tweet:', error);
        }
    }

    stop() {
        if (this.interval) {
            clearInterval(this.interval);
            this.interval = null;
            elizaLogger.info('AutoPoster stopped');
        }
    }
} 