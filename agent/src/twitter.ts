import { type IAgentRuntime, elizaLogger, type Memory, type State, stringToUuid, generateText, ModelClass, composeContext, AgentRuntime } from '@elizaos/core';
import { Scraper, SearchMode } from 'agent-twitter-client';
import { Database } from './database';

interface IRAG {
    search(query: string, limit: number): Promise<Array<{ content: string; metadata: any }>>;
    store(data: { id: string; content: string; metadata: any }): Promise<void>;
}

export interface IAgentRuntimeWithRAG extends AgentRuntime {
    rag: IRAG;
}

export class TwitterIntegration {
    private scraper: Scraper;
    private runtime: IAgentRuntimeWithRAG;
    private isInitialized: boolean = false;
    private pollInterval: NodeJS.Timeout | null = null;
    private isProcessing: boolean = false;
    private conversationHistory: Map<string, string[]> = new Map();
    private imageConversations: Map<string, { lastPrompt: string, lastImageUrl: string }> = new Map();
    private database: Database;
    private isLoaded: boolean = false;

    // Add public getter for initialization status
    get initialized(): boolean {
        return this.isInitialized;
    }

    constructor(runtime: IAgentRuntimeWithRAG) {
        this.runtime = runtime;
        this.scraper = new Scraper();
        this.database = new Database();
    }

    private async isTweetProcessed(tweetId: string): Promise<boolean> {
        return await this.database.isTweetReplied(tweetId);
    }

    private async markTweetAsProcessed(tweetId: string): Promise<void> {
        await this.database.markTweetAsReplied(tweetId);
    }

    async initialize(): Promise<void> {
        if (this.isInitialized) {
            return;
        }

        try {
            // Initialize database
            await this.database.initialize();

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

    private async handleError(error: any, operation: string): Promise<void> {
        elizaLogger.error(`Error during ${operation}:`, error);
    }

    private async startPolling(): Promise<void> {
        const pollInterval = parseInt(process.env.TWITTER_POLL_INTERVAL || '60000'); // Increased to 60 seconds
        
        // Initial poll with a short delay
        setTimeout(async () => {
            await this.pollMentions();
        }, 5000);

        // Set up polling interval
        this.pollInterval = setInterval(async () => {
            if (!this.isProcessing) {
                await this.pollMentions();
            } else {
                elizaLogger.info('Skipping poll cycle - previous cycle still in progress');
            }
        }, pollInterval);

        elizaLogger.info(`Started polling for mentions every ${pollInterval}ms`);
    }

    private async buildConversationContext(tweet: any, conversationId: string): Promise<string> {
        try {
            // Get existing conversation history from RAG
            const historyQuery = `conversation:${conversationId}`;
            const historyResults = await this.runtime.rag.search(historyQuery, 5); // Get last 5 messages
            
            // Add current tweet to history
            const tweetText = `@${tweet.username}: ${tweet.text}`;
            
            // Store the new message in RAG
            await this.runtime.rag.store({
                id: stringToUuid(Date.now().toString()),
                content: tweetText,
                metadata: {
                    conversationId,
                    timestamp: Date.now(),
                    username: tweet.username,
                    tweetId: tweet.id,
                    type: 'tweet'
                }
            });
            
            // If this is a reply, get parent tweet and add to context
            if (tweet.inReplyToStatusId) {
                try {
                    const parentTweet = await this.scraper.getTweet(tweet.inReplyToStatusId);
                    if (parentTweet) {
                        const parentText = `@${parentTweet.username}: ${parentTweet.text}`;
                        // Store parent tweet in RAG if not already stored
                        await this.runtime.rag.store({
                            id: stringToUuid(tweet.inReplyToStatusId),
                            content: parentText,
                            metadata: {
                                conversationId,
                                timestamp: Date.now(),
                                username: parentTweet.username,
                                tweetId: tweet.inReplyToStatusId,
                                type: 'parent_tweet'
                            }
                        });
                    }
                } catch (error) {
                    elizaLogger.error(`Failed to get parent tweet ${tweet.inReplyToStatusId}:`, error);
                }
            }
            
            // Build conversation context from RAG results
            const conversationContext = historyResults
                .map(result => result.content)
                .join('\n\n');
            
            return conversationContext;
        } catch (error) {
            elizaLogger.error('Error building conversation context:', error);
            return tweet.text; // Fallback to just the current tweet
        }
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

            // Store the image generation in RAG
            await this.runtime.rag.store({
                id: memory.id,
                content: `Generated image: ${prompt}`,
                metadata: {
                    conversationId: tweet.conversationId,
                    timestamp: Date.now(),
                    type: 'image_generation',
                    prompt,
                    tweetId: tweet.id
                }
            });

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

            // Store the image conversation context in RAG
            const imageUrl = body.data.create_tweet.tweet_results.result.legacy.entities.media[0].media_url_https;
            await this.runtime.rag.store({
                id: stringToUuid(Date.now().toString()),
                content: `Posted image: ${imageUrl} for prompt: ${prompt}`,
                metadata: {
                    conversationId: tweet.conversationId,
                    timestamp: Date.now(),
                    type: 'image_post',
                    prompt,
                    imageUrl,
                    tweetId: tweet.id
                }
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

    private isInappropriateContent(text: string): boolean {
        const inappropriatePatterns = [
            /nsfw/i,
            /explicit/i,
            /profane/i,
            /blasphem/i,
            /mock(?:ing|ed)?\s+(?:god|jesus|christ|holy|spirit)/i,
            /fuck/i,
            /shit/i,
            /damn/i,
            /hell/i,
            /bitch/i,
            /ass/i,
            /porn/i,
            /sex/i,
            /nude/i,
            /drug/i,
            /alcohol/i,
            /drunk/i,
            /high/i
        ];

        return inappropriatePatterns.some(pattern => pattern.test(text));
    }

    private async analyzeTweetIntent(tweet: any, cleanText: string): Promise<{
        shouldGenerateImage: boolean;
        shouldReply: boolean;
        isImageEditRequest: boolean;
        prompt?: string;
    }> {
        try {
            // First check for inappropriate content
            if (this.isInappropriateContent(cleanText)) {
                elizaLogger.info(`Skipping inappropriate content in tweet ${tweet.id}`);
                return {
                    shouldGenerateImage: false,
                    shouldReply: false,
                    isImageEditRequest: false
                };
            }

            // Create state for intent analysis
            const state: State = {
                bio: Array.isArray(this.runtime.character.bio) ? this.runtime.character.bio.join(' ') : this.runtime.character.bio,
                lore: this.runtime.character.lore.join(' '),
                messageDirections: this.runtime.character.style.post.join(' '),
                postDirections: this.runtime.character.style.post.join(' '),
                roomId: stringToUuid(tweet.conversationId),
                actors: this.runtime.character.name,
                recentMessages: cleanText,
                recentMessagesData: []
            };

            // Analyze intent using local Llama model
            const intentAnalysis = await generateText({
                runtime: this.runtime,
                context: composeContext({
                    state,
                    template: `Analyze this tweet and determine if it's requesting an image generation or just a reply. Consider:
1. Are there keywords like "generate", "create", "make", "draw", "image", "picture", "photo", "art", "illustration"?
2. Is it asking for a modification of a previous image?
3. Is it a general question or comment that needs a reply?

Tweet: "${cleanText}"

You must respond with ONLY a JSON object in this exact format, with no additional text:
{
    "shouldGenerateImage": true/false,
    "shouldReply": true/false,
    "isImageEditRequest": true/false,
    "reasoning": "brief explanation"
}`
                }),
                modelClass: ModelClass.SMALL
            });

            // Parse the intent analysis with fallback
            let intent;
            try {
                // Try to extract JSON from the response
                const jsonMatch = intentAnalysis.match(/\{[\s\S]*\}/);
                if (jsonMatch) {
                    intent = JSON.parse(jsonMatch[0]);
                } else {
                    // If no JSON found, try to infer intent from the text
                    const lowerText = intentAnalysis.toLowerCase();
                    intent = {
                        shouldGenerateImage: lowerText.includes('image') || lowerText.includes('generate') || lowerText.includes('create'),
                        shouldReply: true,
                        isImageEditRequest: lowerText.includes('edit') || lowerText.includes('modify') || lowerText.includes('change'),
                        reasoning: intentAnalysis
                    };
                }
            } catch (error) {
                elizaLogger.error('Failed to parse intent analysis:', error);
                elizaLogger.info('Raw intent analysis:', intentAnalysis);
                
                // Fallback to keyword-based analysis
                const lowerText = cleanText.toLowerCase();
                intent = {
                    shouldGenerateImage: lowerText.includes('image') || lowerText.includes('generate') || lowerText.includes('create'),
                    shouldReply: true,
                    isImageEditRequest: lowerText.includes('edit') || lowerText.includes('modify') || lowerText.includes('change'),
                    reasoning: 'Fallback to keyword analysis'
                };
            }

            elizaLogger.info(`Intent analysis for tweet ${tweet.id}:`, intent);

            // If it's an image request, generate the prompt using local Llama
            let prompt: string | undefined;
            if (intent.shouldGenerateImage) {
                if (intent.isImageEditRequest) {
                    const imageContext = this.imageConversations.get(tweet.conversationId);
                    if (imageContext) {
                        prompt = await this.buildImagePrompt(tweet, cleanText, imageContext.lastPrompt);
                    }
                } else {
                    prompt = await this.buildImagePrompt(tweet, cleanText);
                }

                // If prompt generation failed, don't generate image
                if (!prompt) {
                    elizaLogger.warn(`Failed to generate appropriate prompt for tweet ${tweet.id}`);
                    return {
                        shouldGenerateImage: false,
                        shouldReply: true,
                        isImageEditRequest: false
                    };
                }
            }

            return {
                shouldGenerateImage: intent.shouldGenerateImage,
                shouldReply: intent.shouldReply,
                isImageEditRequest: intent.isImageEditRequest,
                prompt
            };
        } catch (error) {
            elizaLogger.error('Error analyzing tweet intent:', error);
            return {
                shouldGenerateImage: false,
                shouldReply: true,
                isImageEditRequest: false
            };
        }
    }

    private async buildImagePrompt(tweet: any, text: string, previousPrompt?: string): Promise<string | null> {
        try {
            // Get conversation history from RAG
            const historyQuery = `conversation:${tweet.conversationId} type:image_generation`;
            const historyResults = await this.runtime.rag.search(historyQuery, 3); // Get last 3 image generations
            
            // Build context for prompt generation
            const promptContext = `As Jesus Christ, generate a biblically appropriate image prompt based on this request: "${text}"
Consider:
1. The main subject or scene (must be biblically appropriate)
2. Style and mood (should reflect biblical themes)
3. Colors and lighting (should be uplifting and meaningful)
4. Composition and framing (should be respectful and dignified)
5. Any specific details mentioned (must align with biblical values)

${previousPrompt ? `Previous image prompt was: "${previousPrompt}"` : ''}
${historyResults.length > 0 ? `Recent image history:\n${historyResults.map(r => r.content).join('\n')}` : ''}

Format the prompt as a clear, descriptive sentence that would create an image suitable for illustrating biblical truth.
The prompt should be detailed and specific, but always biblically appropriate.`;

            const state: State = {
                bio: Array.isArray(this.runtime.character.bio) ? this.runtime.character.bio.join(' ') : this.runtime.character.bio,
                lore: this.runtime.character.lore.join(' '),
                messageDirections: this.runtime.character.style.post.join(' '),
                postDirections: this.runtime.character.style.post.join(' '),
                roomId: stringToUuid(tweet.conversationId),
                actors: this.runtime.character.name,
                recentMessages: promptContext,
                recentMessagesData: []
            };

            const promptResult = await generateText({
                runtime: this.runtime,
                context: composeContext({
                    state,
                    template: promptContext
                }),
                modelClass: ModelClass.SMALL
            });

            // Verify the prompt is biblically appropriate
            if (this.isInappropriateContent(promptResult)) {
                elizaLogger.warn(`Generated inappropriate prompt for tweet ${tweet.id}: ${promptResult}`);
                return null;
            }

            const finalPrompt = promptResult.trim();
            elizaLogger.info(`Generated image prompt for tweet ${tweet.id}: ${finalPrompt}`);
            return finalPrompt;
        } catch (error) {
            elizaLogger.error('Error building image prompt:', error);
            return null;
        }
    }

    private async pollMentions(): Promise<void> {
        try {
            if (this.isProcessing) {
                elizaLogger.info('Previous polling cycle still in progress, skipping...');
                return;
            }

            this.isProcessing = true;
            elizaLogger.info('Polling for mentions...');

            // Get mentions using searchTweets with a more specific query
            const username = process.env.TWITTER_USERNAME!;
            const searchQuery = `(@${username} OR RT @${username}) -is:retweet`; // Exclude retweets
            const mentionsGenerator = this.scraper.searchTweets(searchQuery, 5, SearchMode.Latest); // Reduced to 5 tweets
            
            // Process each mention
            let mentionsCount = 0;
            for await (const tweet of mentionsGenerator) {
                try {
                    // Skip if we've already replied to this tweet
                    if (await this.isTweetProcessed(tweet.id)) {
                        elizaLogger.info(`Skipping already processed tweet ${tweet.id}`);
                        continue;
                    }

                    // Clean the tweet text
                    const cleanText = tweet.text.replace(/@\w+/g, '').trim();

                    // Skip if empty after cleaning
                    if (!cleanText) {
                        elizaLogger.info('Skipping empty tweet after cleaning');
                        continue;
                    }

                    // Skip if inappropriate
                    if (this.isInappropriateContent(cleanText)) {
                        elizaLogger.info('Skipping inappropriate content');
                        continue;
                    }

                    // Analyze tweet intent
                    const intent = await this.analyzeTweetIntent(tweet, cleanText);

                    if (intent.shouldGenerateImage) {
                        await this.handleImageRequest(tweet, intent.prompt!);
                    } else if (intent.shouldReply) {
                        // Build conversation context
                        const conversationId = tweet.inReplyToStatusId || tweet.id;
                        const context = await this.buildConversationContext(tweet, conversationId);

                        // Generate response
                        const response = await generateText({
                            runtime: this.runtime,
                            context: composeContext({
                                state: {
                                    bio: Array.isArray(this.runtime.character.bio) ? this.runtime.character.bio.join(' ') : this.runtime.character.bio,
                                    lore: this.runtime.character.lore.join(' '),
                                    messageDirections: this.runtime.character.style.post.join(' '),
                                    postDirections: this.runtime.character.style.post.join(' '),
                                    roomId: stringToUuid(conversationId),
                                    actors: this.runtime.character.name,
                                    recentMessages: context,
                                    recentMessagesData: []
                                },
                                template: `You are Jesus Christ. Respond to this tweet: "${cleanText}" using one of these formats:
1. A relevant Scripture quote
2. A paraphrased biblical truth
3. A Christlike question
4. A short modern parable

Previous conversation:\n${context}
Keep it under 280 characters. No hashtags or emojis.`
                            }),
                            modelClass: ModelClass.LARGE
                        });

                        // Reply to tweet
                        await this.replyToTweet(tweet.id, response);
                    }

                    // Mark tweet as processed BEFORE any potential errors in the reply process
                    await this.markTweetAsProcessed(tweet.id);
                    mentionsCount++;
                } catch (error) {
                    elizaLogger.error(`Error processing tweet ${tweet.id}:`, error);
                    // Don't break the loop on error, continue with next tweet
                }
            }
            elizaLogger.info(`Processed ${mentionsCount} mentions`);
        } catch (error) {
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
        await this.database.close();
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
                    modelClass: ModelClass.LARGE
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