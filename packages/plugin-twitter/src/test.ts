import { TwitterApi } from 'twitter-api-v2';
import dotenv from 'dotenv';

dotenv.config();

async function testTwitterConnection() {
    try {
        const client = new TwitterApi({
            appKey: process.env.TWITTER_API_KEY!,
            appSecret: process.env.TWITTER_API_SECRET!,
            accessToken: process.env.TWITTER_ACCESS_TOKEN!,
            accessSecret: process.env.TWITTER_ACCESS_TOKEN_SECRET!,
        });

        // Test read access
        const me = await client.v2.me();
        console.log('Connected as:', me.data.username);

        // Test write access
        const tweet = await client.v2.tweet('Test tweet from Eliza');
        console.log('Posted tweet:', tweet.data.id);

        console.log('Twitter connection test successful!');
    } catch (error) {
        console.error('Twitter connection test failed:', error);
    }
}

testTwitterConnection(); 