import sqlite3 from 'sqlite3';
import { open } from 'sqlite';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

export class Database {
    private db: any;

    async initialize() {
        this.db = await open({
            filename: path.join(__dirname, '../data/tweets.db'),
            driver: sqlite3.Database
        });

        // Create tweets table if it doesn't exist
        await this.db.exec(`
            CREATE TABLE IF NOT EXISTS replied_tweets (
                tweet_id TEXT PRIMARY KEY,
                replied_at INTEGER NOT NULL,
                created_at INTEGER NOT NULL
            )
        `);
    }

    async isTweetReplied(tweetId: string): Promise<boolean> {
        const result = await this.db.get(
            'SELECT tweet_id FROM replied_tweets WHERE tweet_id = ?',
            [tweetId]
        );
        return !!result;
    }

    async markTweetAsReplied(tweetId: string): Promise<void> {
        await this.db.run(
            'INSERT OR REPLACE INTO replied_tweets (tweet_id, replied_at, created_at) VALUES (?, ?, ?)',
            [tweetId, Date.now(), Date.now()]
        );
    }

    async cleanupOldTweets(maxAge: number = 30 * 24 * 60 * 60 * 1000): Promise<void> {
        // Delete tweets older than maxAge (default 30 days)
        const cutoffTime = Date.now() - maxAge;
        await this.db.run(
            'DELETE FROM replied_tweets WHERE created_at < ?',
            [cutoffTime]
        );
    }

    async close(): Promise<void> {
        if (this.db) {
            await this.db.close();
        }
    }
} 