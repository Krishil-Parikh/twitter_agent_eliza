import { type IAgentRuntime, elizaLogger, type Memory, type State, stringToUuid, generateText, ModelClass, composeContext } from '@elizaos/core';
import { Scraper, SearchMode } from 'agent-twitter-client';

export class TwitterIntegration {
    private scraper: Scraper;
    private runtime: IAgentRuntime;
    private isInitialized: boolean = false;
    private pollInterval: NodeJS.Timeout | null = null;
    private lastMentionId: string | null = null;
    private isProcessing: boolean = false;
    private conversationHistory: Map<string, string[]> = new Map(); // Store conversation history by conversation ID
    private imageConversations: Map<string, { lastPrompt: string, lastImageUrl: string }> = new Map(); // Track image generation conversations

    constructor(runtime: IAgentRuntime) {
        this.runtime = runtime;
        this.scraper = new Scraper();
    }

    private async handleError(error: any, operation: string): Promise<void> {
        elizaLogger.error(`Error during ${operation}:`, error);
    }

    async initialize(): Promise<void> {
        if (this.isInitialized) {
            return;
        }

        try {
            // Check for required environment variables
            const requiredEnvVars = [
                'TWITTER_USERNAME',
                'TWITTER_PASSWORD',
                'TWITTER_EMAIL'
            ];

            for (const envVar of requiredEnvVars) {
                if (!process.env[envVar]) {
                    throw new Error(`Missing required environment variable: ${envVar}`);
                }
            }

            elizaLogger.info('Attempting to login to Twitter...');
            elizaLogger.info(`Using credentials for username: ${process.env.TWITTER_USERNAME}`);

            // Login to Twitter with retry logic and timeout
            let loginSuccess = false;
            let retryCount = 0;
            const MAX_RETRIES = 3;
            const LOGIN_TIMEOUT = 15000; // Reduced timeout to 15 seconds

            while (!loginSuccess && retryCount < MAX_RETRIES) {
                try {
                    // Create a promise that rejects after timeout
                    const timeoutPromise = new Promise((_, reject) => {
                        setTimeout(() => reject(new Error('Login timeout')), LOGIN_TIMEOUT);
                    });

                    // Race between login and timeout
                    await Promise.race([
                        this.scraper.login(
                            process.env.TWITTER_USERNAME!,
                            process.env.TWITTER_PASSWORD!,
                            process.env.TWITTER_EMAIL!,
                            process.env.TWITTER_2FA_SECRET
                        ),
                        timeoutPromise
                    ]);

                    // Verify login
                    const isLoggedIn = await this.scraper.isLoggedIn();
                    if (!isLoggedIn) {
                        throw new Error('Login verification failed - isLoggedIn check returned false');
                    }

                    loginSuccess = true;
                    elizaLogger.info('Successfully logged in to Twitter');
                } catch (error) {
                    retryCount++;
                    elizaLogger.error(`Login attempt ${retryCount} failed:`, error);
                    if (retryCount < MAX_RETRIES) {
                        const backoffTime = Math.min(1000 * Math.pow(2, retryCount), 5000); // Exponential backoff with max 5s
                        elizaLogger.info(`Retrying login in ${backoffTime}ms...`);
                        await new Promise(resolve => setTimeout(resolve, backoffTime));
                    } else {
                        throw new Error(`Failed to login after ${MAX_RETRIES} attempts: ${error.message}`);
                    }
                }
            }

            // Start polling for mentions with a shorter initial delay
            await this.startPolling();

            // Start periodic posting with a shorter initial delay
            await this.startPeriodicPosting();

            this.isInitialized = true;
        } catch (error) {
            await this.handleError(error, 'initialization');
            throw error;
        }
    }

    private async startPolling(): Promise<void> {
        const pollInterval = parseInt(process.env.TWITTER_POLL_INTERVAL || '30000'); // 30 seconds
        
        // Initial poll with a short delay
        setTimeout(async () => {
            await this.pollMentions();
        }, 1000);

        // Set up polling interval
        this.pollInterval = setInterval(async () => {
            if (!this.isProcessing) {
                await this.pollMentions();
            }
        }, pollInterval);

        elizaLogger.info(`Started polling for mentions every ${pollInterval}ms`);
    }

    private async buildConversationContext(tweet: any, conversationId: string): Promise<string> {
        // Get existing conversation history
        let history = this.conversationHistory.get(conversationId) || [];
        
        // Add current tweet to history
        const tweetText = `@${tweet.username}: ${tweet.text}`;
        history.push(tweetText);
        
        // Keep only last 5 messages to maintain context without getting too long
        if (history.length > 5) {
            history = history.slice(-5);
        }
        
        // Update conversation history
        this.conversationHistory.set(conversationId, history);
        
        // If this is a reply, get parent tweet and add to context
        if (tweet.inReplyToStatusId) {
            try {
                const parentTweet = await this.scraper.getTweet(tweet.inReplyToStatusId);
                if (parentTweet) {
                    // Add parent tweet at the beginning of history
                    history.unshift(`@${parentTweet.username}: ${parentTweet.text}`);
                }
            } catch (error) {
                elizaLogger.error(`Failed to get parent tweet ${tweet.inReplyToStatusId}:`, error);
            }
        }
        
        return history.join('\n\n');
    }

    private async generateImage(prompt: string): Promise<Buffer> {
        try {
            const encodedPrompt = encodeURIComponent(prompt);
            const imageUrl = `https://image.pollinations.ai/prompt/${encodedPrompt}?width=2560&height=2049&nologo=true`;
            
            // Fetch the image
            const response = await fetch(imageUrl);
            if (!response.ok) {
                throw new Error(`Failed to fetch image: ${response.statusText}`);
            }
            
            // Convert to buffer
            const arrayBuffer = await response.arrayBuffer();
            return Buffer.from(arrayBuffer);
        } catch (error) {
            await this.handleError(error, 'image generation');
            throw error;
        }
    }

    private async handleImageRequest(tweet: any, prompt: string): Promise<void> {
        try {
            // Generate the image
            const imageBuffer = await this.generateImage(prompt);
            
            // Create a memory for the image generation
            const memory: Memory = {
                id: stringToUuid(Date.now().toString()),
                userId: stringToUuid(tweet.userId),
                agentId: this.runtime.agentId,
                roomId: stringToUuid(tweet.conversationId),
                content: {
                    text: `Generated image for prompt: ${prompt}`,
                    action: 'GENERATE_IMAGE'
                },
                createdAt: Date.now()
            };

            // Create state for the response
            const state: State = {
                bio: Array.isArray(this.runtime.character.bio) ? this.runtime.character.bio.join(' ') : this.runtime.character.bio,
                lore: this.runtime.character.lore.join(' '),
                messageDirections: this.runtime.character.style.post.join(' '),
                postDirections: this.runtime.character.style.post.join(' '),
                roomId: stringToUuid(tweet.conversationId),
                actors: this.runtime.character.name,
                recentMessages: `Generate an image for: ${prompt}`,
                recentMessagesData: [memory]
            };

            // Process the image generation through the agent's action system
            await this.runtime.processActions(memory, [], state);

            // Post the image as a reply
            const mediaData = [{
                data: imageBuffer,
                mediaType: 'image/jpeg'
            }];

            const result = await this.scraper.sendTweet("Here's the image you requested", tweet.id, mediaData);
            const body = await result.json();
            
            // Check for Twitter API errors
            if (body.errors) {
                const error = body.errors[0];
                throw new Error(`Twitter API error (${error.code}): ${error.message}`);
            }

            // Store the image conversation context
            this.imageConversations.set(tweet.conversationId, {
                lastPrompt: prompt,
                lastImageUrl: body.data.create_tweet.tweet_results.result.legacy.entities.media[0].media_url_https
            });

            elizaLogger.info(`Successfully generated and shared image for prompt: ${prompt}`);
        } catch (error) {
            await this.handleError(error, 'handling image request');
            throw error;
        }
    }

    private isImageModificationRequest(text: string): boolean {
        const modificationKeywords = [
            'change', 'modify', 'adjust', 'update', 'edit', 'make it', 'make this',
            'different', 'another', 'new', 'more', 'less', 'brighter', 'darker',
            'larger', 'smaller', 'wider', 'taller', 'add', 'remove', 'replace'
        ];
        return modificationKeywords.some(keyword => text.toLowerCase().includes(keyword.toLowerCase()));
    }

    private async buildImagePrompt(tweet: any, text: string): Promise<string | null> {
        const conversationId = tweet.conversationId;
        const imageContext = this.imageConversations.get(conversationId);
        
        if (!imageContext) {
            // No previous image context, treat as new request
            return text;
        }

        // Check if this is a modification request
        if (this.isImageModificationRequest(text)) {
            // Combine the modification request with the previous prompt
            return `${imageContext.lastPrompt}, but ${text}`;
        }

        // Check if this is a completely new request
        const imageKeywords = ['image', 'picture', 'photo', 'draw', 'paint', 'create', 'generate', 'show me'];
        const isNewRequest = imageKeywords.some(keyword => text.toLowerCase().includes(keyword.toLowerCase()));
        
        if (isNewRequest) {
            return text;
        }

        // If it's neither a modification nor a new request, assume it's a modification
        return `${imageContext.lastPrompt}, but ${text}`;
    }

    private async pollMentions(): Promise<void> {
        if (this.isProcessing) {
            elizaLogger.warn('Previous mention processing still in progress');
            return;
        }

        this.isProcessing = true;
        try {
            const username = process.env.TWITTER_USERNAME!;
            elizaLogger.info(`Polling for mentions and interactions with @${username}`);
            
            const searchQuery = `(@${username} OR RT @${username})`;
            const tweets = this.scraper.searchTweets(searchQuery, 10, SearchMode.Latest);
            elizaLogger.info(`Searching for interactions with query: ${searchQuery}`);
            
            let interactionsCount = 0;
            for await (const tweet of tweets) {
                interactionsCount++;
                elizaLogger.info(`Checking tweet ${tweet.id}: ${tweet.text}`);
                
                if (this.lastMentionId && tweet.id <= this.lastMentionId) {
                    elizaLogger.info(`Skipping already processed tweet ${tweet.id}`);
                    continue;
                }

                if (!this.lastMentionId || tweet.id > this.lastMentionId) {
                    this.lastMentionId = tweet.id;
                }

                const isRetweet = tweet.isRetweet && tweet.text.toLowerCase().includes(`rt @${username.toLowerCase()}`);
                const isReplyToRetweet = tweet.inReplyToStatusId && tweet.text.toLowerCase().includes(`@${username.toLowerCase()}`);
                const isMention = tweet.text.toLowerCase().includes(`@${username.toLowerCase()}`);

                if (!isRetweet && !isReplyToRetweet && !isMention) {
                    elizaLogger.info(`Tweet ${tweet.id} is not a relevant interaction`);
                    continue;
                }

                elizaLogger.info(`Found interaction in tweet ${tweet.id} - Type: ${isRetweet ? 'Retweet' : isReplyToRetweet ? 'Reply to Retweet' : 'Mention'}`);

                try {
                    // Clean the tweet text
                    let cleanText = tweet.text
                        .replace(new RegExp(`@${username}`, 'i'), '')
                        .replace(/^(?:can you |please |could you |would you |will you )?/i, '')
                        .trim();

                    // Check if this is an image-related request
                    const prompt = await this.buildImagePrompt(tweet, cleanText);
                    
                    if (prompt) {
                        await this.handleImageRequest(tweet, prompt);
                        continue;
                    }

                    // Build conversation context with history
                    const conversationHistory = await this.buildConversationContext(tweet, tweet.conversationId);

                    const memory: Memory = {
                        id: stringToUuid(tweet.id + "-" + this.runtime.agentId),
                        userId: stringToUuid(tweet.userId),
                        agentId: this.runtime.agentId,
                        roomId: stringToUuid(tweet.conversationId),
                        content: {
                            text: tweet.text,
                            url: tweet.permanentUrl,
                            source: tweet.username,
                            inReplyTo: tweet.inReplyToStatusId ? stringToUuid(tweet.inReplyToStatusId + "-" + this.runtime.agentId) : undefined,
                            action: isRetweet ? 'HANDLE_RETWEET' : 'HANDLE_MENTION'
                        },
                        createdAt: tweet.timestamp * 1000
                    };

                    const state: State = {
                        bio: Array.isArray(this.runtime.character.bio) ? this.runtime.character.bio.join(' ') : this.runtime.character.bio,
                        lore: this.runtime.character.lore.join(' '),
                        messageDirections: this.runtime.character.style.post.join(' '),
                        postDirections: this.runtime.character.style.post.join(' '),
                        roomId: stringToUuid(tweet.conversationId),
                        actors: this.runtime.character.name,
                        recentMessages: conversationHistory,
                        recentMessagesData: [memory]
                    };

                    await this.runtime.processActions(memory, [], state);
                    
                    let replyTemplate = '';
                    if (isRetweet) {
                        replyTemplate = `You are ${this.runtime.character.name}. Someone retweeted your tweet. Generate a friendly response thanking them and engaging with them about the topic. Previous conversation:\n${conversationHistory}\nKeep it under 280 characters. No hashtags or emojis.`;
                    } else if (isReplyToRetweet) {
                        replyTemplate = `You are ${this.runtime.character.name}. Someone replied to a retweet of your tweet. Generate a contextual response continuing the conversation. Previous conversation:\n${conversationHistory}\nKeep it under 280 characters. No hashtags or emojis.`;
                    } else {
                        replyTemplate = `You are ${this.runtime.character.name}. Generate a reply to this tweet: "${tweet.text}". Previous conversation:\n${conversationHistory}\nKeep it under 280 characters. No hashtags or emojis.`;
                    }

                    const replyContent = await generateText({
                        runtime: this.runtime,
                        context: composeContext({
                            state,
                            template: replyTemplate
                        }),
                        modelClass: ModelClass.SMALL
                    });

                    await this.replyToTweet(tweet.id, replyContent);
                    elizaLogger.info(`Successfully processed and replied to interaction in tweet ${tweet.id}`);
                } catch (error) {
                    elizaLogger.error(`Failed to process interaction in tweet ${tweet.id}:`, error);
                }
            }
            elizaLogger.info(`Processed ${interactionsCount} interactions`);
        } catch (error) {
            elizaLogger.error('Error polling mentions:', error);
            await this.handleError(error, 'polling mentions');
        } finally {
            this.isProcessing = false;
        }
    }

    async stop(): Promise<void> {
        if (this.pollInterval) {
            clearInterval(this.pollInterval);
            this.pollInterval = null;
        }
    }

    async postTweet(content: string): Promise<void> {
        try {
            // Create a memory for the tweet
            const memory: Memory = {
                id: stringToUuid(Date.now().toString()),
                userId: this.runtime.agentId,
                agentId: this.runtime.agentId,
                roomId: stringToUuid('twitter'),
                content: {
                    text: content,
                    action: 'POST_TWEET'
                },
                createdAt: Date.now()
            };

            // Create state for the tweet
            const state: State = {
                bio: Array.isArray(this.runtime.character.bio) ? this.runtime.character.bio.join(' ') : this.runtime.character.bio,
                lore: this.runtime.character.lore.join(' '),
                messageDirections: this.runtime.character.style.post.join(' '),
                postDirections: this.runtime.character.style.post.join(' '),
                roomId: stringToUuid('twitter'),
                actors: this.runtime.character.name,
                recentMessages: content,
                recentMessagesData: [memory]
            };

            // Process the tweet through the agent's action system
            await this.runtime.processActions(memory, [], state);

            // Post the tweet using the scraper
            const result = await this.scraper.sendTweet(content);
            const body = await result.json();
            
            // Check for Twitter API errors
            if (body.errors) {
                const error = body.errors[0];
                throw new Error(`Twitter API error (${error.code}): ${error.message}`);
            }

            // Check for successful tweet creation
            if (!body?.data?.create_tweet?.tweet_results?.result) {
                throw new Error("Failed to post tweet: No tweet result in response");
            }

            elizaLogger.info(`Posted tweet: ${content}`);
        } catch (error) {
            await this.handleError(error, 'posting tweet');
            throw error;
        }
    }

    async replyToTweet(tweetId: string, content: string): Promise<void> {
        try {
            // Create a memory for the reply
            const memory: Memory = {
                id: stringToUuid(Date.now().toString()),
                userId: this.runtime.agentId,
                agentId: this.runtime.agentId,
                roomId: stringToUuid('twitter'),
                content: {
                    text: content,
                    action: 'REPLY_TO_TWEET',
                    inReplyTo: stringToUuid(tweetId + "-" + this.runtime.agentId)
                },
                createdAt: Date.now()
            };

            // Create state for the reply
            const state: State = {
                bio: Array.isArray(this.runtime.character.bio) ? this.runtime.character.bio.join(' ') : this.runtime.character.bio,
                lore: this.runtime.character.lore.join(' '),
                messageDirections: this.runtime.character.style.post.join(' '),
                postDirections: this.runtime.character.style.post.join(' '),
                roomId: stringToUuid('twitter'),
                actors: this.runtime.character.name,
                recentMessages: content,
                recentMessagesData: [memory]
            };

            // Process the reply through the agent's action system
            await this.runtime.processActions(memory, [], state);

            // Get the tweet to get the author's username
            const tweet = await this.scraper.getTweet(tweetId);
            if (!tweet) {
                throw new Error(`Failed to get tweet ${tweetId}`);
            }

            // Post the reply using the scraper's sendTweet method
            const replyContent = `@${tweet.username} ${content}`;
            const result = await this.scraper.sendTweet(replyContent, tweetId);
            const body = await result.json();
            
            // Check for Twitter API errors
            if (body.errors) {
                const error = body.errors[0];
                throw new Error(`Twitter API error (${error.code}): ${error.message}`);
            }

            // Check for successful tweet creation
            if (!body?.data?.create_tweet?.tweet_results?.result) {
                throw new Error("Failed to post reply: No tweet result in response");
            }

            elizaLogger.info(`Posted reply to tweet ${tweetId}: ${content}`);
        } catch (error) {
            await this.handleError(error, 'posting reply');
            throw error;
        }
    }

    async searchTweets(query: string) {
        try {
            return await this.scraper.searchTweets(query, 10, SearchMode.Top); // Reduced to 10 tweets
        } catch (error) {
            await this.handleError(error, "tweet search");
            throw error;
        }
    }

    async getTimeline() {
        try {
            return await this.scraper.getTweets(process.env.TWITTER_USERNAME!, 10); // Reduced to 10 tweets
        } catch (error) {
            await this.handleError(error, "timeline fetch");
            throw error;
        }
    }

    private async startPeriodicPosting(): Promise<void> {
        const POST_INTERVAL = 60000; // Fixed 1-minute interval
        const MAX_RETRIES = 3;
        
        elizaLogger.info(`Starting periodic posting every ${POST_INTERVAL/1000} seconds`);

        // Function to generate and post a tweet with retry logic
        const generateAndPostTweet = async () => {
            if (this.isProcessing) {
                elizaLogger.warn('Previous tweet generation still in progress, forcing reset');
                this.isProcessing = false;
            }

            this.isProcessing = true;
            let retryCount = 0;

            try {
                // Use the character's topics and style for generating tweets
                const topics = this.runtime.character.topics;
                const style = this.runtime.character.style.post;
                const adjectives = this.runtime.character.adjectives;
                
                // Randomly select a topic and style element
                const randomTopic = topics[Math.floor(Math.random() * topics.length)];
                const randomStyle = style[Math.floor(Math.random() * style.length)];
                const randomAdjective = adjectives[Math.floor(Math.random() * adjectives.length)];
                
                elizaLogger.info(`Generating tweet about ${randomTopic} in ${randomAdjective} style`);
                
                // Create state for the tweet generation
                const state: State = {
                    bio: Array.isArray(this.runtime.character.bio) ? this.runtime.character.bio.join(' ') : this.runtime.character.bio,
                    lore: this.runtime.character.lore.join(' '),
                    messageDirections: this.runtime.character.style.post.join(' '),
                    postDirections: this.runtime.character.style.post.join(' '),
                    roomId: this.runtime.agentId,
                    actors: this.runtime.character.name,
                    recentMessages: `Generate a tweet about ${randomTopic}`,
                    recentMessagesData: []
                };

                // Compose context for tweet generation
                const context = composeContext({
                    state,
                    template: `You are ${this.runtime.character.name}. Generate a tweet that is ${randomAdjective} about ${randomTopic}. The tweet should ${randomStyle}. Keep it under 280 characters. No hashtags or emojis.`
                });

                // Generate tweet content using Google model
                const tweetContent = await generateText({
                    runtime: this.runtime,
                    context,
                    modelClass: ModelClass.SMALL
                });

                elizaLogger.info(`Generated tweet content: ${tweetContent}`);

                // Post the tweet with retry logic
                while (retryCount < MAX_RETRIES) {
                    try {
                        await this.postTweet(tweetContent);
                        elizaLogger.info('Successfully posted tweet');
                        break;
                    } catch (error) {
                        retryCount++;
                        elizaLogger.error(`Failed to post tweet (attempt ${retryCount}/${MAX_RETRIES}):`, error);
                        if (retryCount < MAX_RETRIES) {
                            const backoffTime = Math.min(1000 * Math.pow(2, retryCount), 5000);
                            elizaLogger.info(`Retrying in ${backoffTime}ms...`);
                            await new Promise(resolve => setTimeout(resolve, backoffTime));
                        }
                    }
                }

                if (retryCount === MAX_RETRIES) {
                    elizaLogger.error('Failed to post tweet after all retries');
                }

            } catch (error) {
                elizaLogger.error('Error in generateAndPostTweet:', error);
                await this.handleError(error, 'periodic posting');
            } finally {
                this.isProcessing = false;
                elizaLogger.info('Tweet generation completed, reset isProcessing flag');
            }
        };

        // Initial post with a short delay
        setTimeout(generateAndPostTweet, 5000);

        // Set up the regular interval with proper error handling
        const scheduleNextPost = () => {
            elizaLogger.info(`Scheduling next post in ${POST_INTERVAL/1000} seconds`);
            
            setTimeout(async () => {
                try {
                    await generateAndPostTweet();
                } catch (error) {
                    elizaLogger.error('Error in scheduled post:', error);
                } finally {
                    scheduleNextPost(); // Always schedule next post, even if this one failed
                }
            }, POST_INTERVAL);
        };

        // Start the scheduling cycle
        scheduleNextPost();
    }
}