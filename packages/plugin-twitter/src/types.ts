import { z } from "zod";

export interface TweetContent {
    text: string;
}

export const TweetSchema = z.object({
    text: z.string().describe("The text of the tweet")
});

export const isTweetContent = (obj: unknown): obj is TweetContent => {
    return TweetSchema.safeParse(obj).success;
};

export interface Mention {
    id: string;
    username: string;
    name: string;
}

export interface Tweet {
    id: string;
    text: string;
    username: string;
    mentions: Mention[];
    isReply: boolean;
    isRetweet: boolean;
    isQuoted: boolean;
    thread: any[];
    hashtags: string[];
    photos: any[];
    urls: string[];
    videos: any[];
    inReplyTo?: string;
}
