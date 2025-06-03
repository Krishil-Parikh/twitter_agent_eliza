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
        try {
            const isProcessed = await this.database.isTweetReplied(tweetId);
            if (isProcessed) {
                elizaLogger.info(`Tweet ${tweetId} has already been processed`);
            }
            return isProcessed;
        } catch (error) {
            elizaLogger.error(`Error checking if tweet ${tweetId} was processed:`, error);
            return false;
        }
    }

    private async markTweetAsProcessed(tweetId: string): Promise<void> {
        try {
            await this.database.markTweetAsReplied(tweetId);
            elizaLogger.info(`Marked tweet ${tweetId} as processed`);
        } catch (error) {
            elizaLogger.error(`Error marking tweet ${tweetId} as processed:`, error);
        }
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

    private async ragSearchWithRetry(query: string, limit: number, maxRetries: number = 3): Promise<Array<{ content: string; metadata: any }>> {
        let retryCount = 0;
        let lastError: Error | null = null;

        while (retryCount < maxRetries) {
            try {
                elizaLogger.info(`Attempting RAG search (attempt ${retryCount + 1}/${maxRetries}): ${query}`);
                
                // Add a timeout to the RAG search
                const timeoutPromise = new Promise<Array<{ content: string; metadata: any }>>((_, reject) => {
                    setTimeout(() => reject(new Error('RAG search timeout')), 10000);
                });

                const results = await Promise.race([
                    this.runtime.rag.search(query, limit),
                    timeoutPromise
                ]) as Array<{ content: string; metadata: any }>;

                elizaLogger.info(`RAG search successful, found ${results.length} results`);
                return results;
            } catch (error) {
                retryCount++;
                lastError = error;
                elizaLogger.error(`RAG search failed (attempt ${retryCount}/${maxRetries}):`, error);
                
                if (retryCount < maxRetries) {
                    const backoffTime = Math.min(1000 * Math.pow(2, retryCount), 5000);
                    elizaLogger.info(`Retrying RAG search in ${backoffTime}ms...`);
                    await new Promise(resolve => setTimeout(resolve, backoffTime));
                }
            }
        }

        elizaLogger.error('RAG search failed after all retries:', lastError);
        return []; // Return empty array instead of throwing to allow graceful degradation
    }

    private async ragStoreWithRetry(data: { id: string; content: string; metadata: any }, maxRetries: number = 3): Promise<void> {
        let retryCount = 0;
        let lastError: Error | null = null;

        while (retryCount < maxRetries) {
            try {
                elizaLogger.info(`Attempting RAG store (attempt ${retryCount + 1}/${maxRetries}): ${data.id}`);
                
                // Add a timeout to the RAG store
                const timeoutPromise = new Promise((_, reject) => {
                    setTimeout(() => reject(new Error('RAG store timeout')), 10000);
                });

                await Promise.race([
                    this.runtime.rag.store(data),
                    timeoutPromise
                ]);

                elizaLogger.info('RAG store successful');
                return;
            } catch (error) {
                retryCount++;
                lastError = error;
                elizaLogger.error(`RAG store failed (attempt ${retryCount}/${maxRetries}):`, error);
                
                if (retryCount < maxRetries) {
                    const backoffTime = Math.min(1000 * Math.pow(2, retryCount), 5000);
                    elizaLogger.info(`Retrying RAG store in ${backoffTime}ms...`);
                    await new Promise(resolve => setTimeout(resolve, backoffTime));
                }
            }
        }

        elizaLogger.error('RAG store failed after all retries:', lastError);
        // Don't throw, just log the error to allow graceful degradation
    }

    private async buildConversationContext(tweet: any, conversationId: string): Promise<string> {
        try {
            // Get existing conversation history from RAG with retry
            const historyQuery = `conversation:${conversationId}`;
            const historyResults = await this.ragSearchWithRetry(historyQuery, 5); // Get last 5 messages
            
            // Add current tweet to history
            const tweetText = `@${tweet.username}: ${tweet.text}`;
            
            // Store the new message in RAG with retry
            await this.ragStoreWithRetry({
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
                        // Store parent tweet in RAG with retry
                        await this.ragStoreWithRetry({
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

    private async handleImageRequest(tweet: any, prompt: string): Promise<void> {
        try {
            elizaLogger.info(`Starting image generation for prompt: ${prompt}`);
            
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

            elizaLogger.info('Attempting to post image to Twitter...');
            const result = await this.scraper.sendTweet("Here's the image you requested", tweet.id, mediaData);
            const body = await result.json();
            
            // Check for Twitter API errors
            if (body.errors) {
                const error = body.errors[0];
                throw new Error(`Twitter API error (${error.code}): ${error.message}`);
            }

            elizaLogger.info(`Successfully generated and shared image for prompt: ${prompt}`);
        } catch (error) {
            elizaLogger.error('Error in handleImageRequest:', error);
            await this.handleError(error, 'handling image request');
            throw error;
        }
    }

    private async generateImage(prompt: string): Promise<Buffer> {
        try {
            if (!prompt || prompt.trim() === '') {
                throw new Error('Empty prompt provided for image generation');
            }

            const encodedPrompt = encodeURIComponent(prompt.trim());
            const imageUrl = `https://image.pollinations.ai/prompt/${encodedPrompt}?width=2560&height=2049&nologo=true`;
            
            elizaLogger.info(`Attempting to generate image with URL: ${imageUrl}`);
            
            // Fetch the image with timeout and retries
            const maxRetries = 3;
            let lastError: Error | null = null;

            for (let attempt = 1; attempt <= maxRetries; attempt++) {
                try {
                    const controller = new AbortController();
                    const timeout = setTimeout(() => controller.abort(), 30000); // 30 second timeout

                    try {
                        elizaLogger.info(`Fetch attempt ${attempt}/${maxRetries}...`);
                        const response = await fetch(imageUrl, { 
                            signal: controller.signal,
                            headers: {
                                'Accept': 'image/jpeg,image/png,image/*;q=0.8',
                                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
                            }
                        });
                        clearTimeout(timeout);

                        if (!response.ok) {
                            throw new Error(`Failed to fetch image: ${response.status} ${response.statusText}`);
                        }

                        const contentType = response.headers.get('content-type');
                        if (!contentType || !contentType.startsWith('image/')) {
                            throw new Error(`Invalid content type: ${contentType}`);
                        }
                        
                        // Convert to buffer
                        const arrayBuffer = await response.arrayBuffer();
                        const buffer = Buffer.from(arrayBuffer);

                        // Verify the buffer is not empty and is a valid image
                        if (buffer.length === 0) {
                            throw new Error('Received empty image buffer');
                        }

                        elizaLogger.info(`Successfully generated image (${buffer.length} bytes)`);
                        return buffer;
                    } catch (error) {
                        clearTimeout(timeout);
                        if (error.name === 'AbortError') {
                            throw new Error('Image generation request timed out after 30 seconds');
                        }
                        throw error;
                    }
                } catch (error) {
                    lastError = error;
                    elizaLogger.error(`Attempt ${attempt}/${maxRetries} failed:`, error);
                    
                    if (attempt < maxRetries) {
                        const backoffTime = Math.min(1000 * Math.pow(2, attempt), 5000);
                        elizaLogger.info(`Retrying in ${backoffTime}ms...`);
                        await new Promise(resolve => setTimeout(resolve, backoffTime));
                    }
                }
            }

            throw lastError || new Error('Failed to generate image after all retries');
        } catch (error) {
            elizaLogger.error('Error in generateImage:', error);
            await this.handleError(error, 'image generation');
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

            // Get conversation history for better context
            let conversationContext = '';
            try {
                const historyQuery = `conversation:${tweet.conversationId}`;
                const historyResults = await this.ragSearchWithRetry(historyQuery, 5);
                conversationContext = historyResults.map(r => r.content).join('\n');
            } catch (error) {
                elizaLogger.warn('Failed to get conversation history, proceeding without context:', error);
            }

            // Create state for intent analysis
            const state: State = {
                bio: Array.isArray(this.runtime.character.bio) ? this.runtime.character.bio.join(' ') : this.runtime.character.bio,
                lore: this.runtime.character.lore.join(' '),
                messageDirections: this.runtime.character.style.post.join(' '),
                postDirections: this.runtime.character.style.post.join(' '),
                roomId: stringToUuid(tweet.conversationId),
                actors: this.runtime.character.name,
                recentMessages: `${conversationContext}\n\nCurrent tweet: ${cleanText}`,
                recentMessagesData: []
            };

            // Analyze intent using local Llama model with better context
            const intentAnalysis = await generateText({
                runtime: this.runtime,
                context: composeContext({
                    state,
                    template: `Analyze this tweet and its conversation context to determine if it's requesting an image generation or just a reply. Consider:
1. Is the user asking for an image, picture, or visual representation?
2. Is there context from previous messages suggesting this is part of an image-related conversation?
3. Is it asking for a modification of a previous image?
4. Is it a general question or comment that needs a reply?

Conversation context:
${conversationContext}

Current tweet: "${cleanText}"

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
                try {
                    if (intent.isImageEditRequest) {
                        const imageContext = this.imageConversations.get(tweet.conversationId);
                        if (imageContext) {
                            prompt = await this.buildImagePrompt(tweet, cleanText, imageContext.lastPrompt);
                        }
                    } else {
                        prompt = await this.buildImagePrompt(tweet, cleanText);
                    }

                    // If prompt generation failed, use a basic fallback prompt
                    if (!prompt) {
                        elizaLogger.warn(`Failed to generate appropriate prompt for tweet ${tweet.id}, using fallback`);
                        prompt = `A serene biblical scene depicting ${cleanText}, with soft lighting and peaceful atmosphere, suitable for illustrating biblical truth`;
                    }
                } catch (error) {
                    elizaLogger.error('Error generating prompt:', error);
                    // Use a basic fallback prompt
                    prompt = `A serene biblical scene depicting ${cleanText}, with soft lighting and peaceful atmosphere, suitable for illustrating biblical truth`;
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
            let historyResults = [];
            try {
                // Get conversation history from RAG with retry
                const historyQuery = `conversation:${tweet.conversationId} type:image_generation`;
                historyResults = await this.ragSearchWithRetry(historyQuery, 3); // Get last 3 image generations
            } catch (ragError) {
                elizaLogger.warn('Failed to get conversation history from RAG:', ragError);
                // Continue without history
            }
            
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

            let promptResult: string;
            try {
                elizaLogger.info('Attempting to generate prompt with context:', promptContext);
                // Try to generate prompt using the model
                promptResult = await generateText({
                    runtime: this.runtime,
                    context: composeContext({
                        state,
                        template: promptContext
                    }),
                    modelClass: ModelClass.SMALL
                });
                elizaLogger.info('Successfully generated prompt:', promptResult);

                // Store the generated prompt in RAG with retry
                await this.ragStoreWithRetry({
                    id: stringToUuid(Date.now().toString()),
                    content: promptResult,
                    metadata: {
                        conversationId: tweet.conversationId,
                        timestamp: Date.now(),
                        type: 'image_generation',
                        prompt: promptResult,
                        tweetId: tweet.id
                    }
                });
            } catch (error) {
                elizaLogger.warn('Failed to generate prompt using model, falling back to basic prompt:', error);
                // Fallback to a basic prompt if model generation fails
                promptResult = `A serene biblical scene depicting ${text}, with soft lighting and peaceful atmosphere, suitable for illustrating biblical truth`;
            }

            // Verify the prompt is biblically appropriate
            if (this.isInappropriateContent(promptResult)) {
                elizaLogger.warn(`Generated inappropriate prompt for tweet ${tweet.id}: ${promptResult}`);
                return null;
            }

            const finalPrompt = promptResult.trim();
            if (!finalPrompt) {
                elizaLogger.warn('Generated empty prompt');
                return null;
            }

            elizaLogger.info(`Generated image prompt for tweet ${tweet.id}: ${finalPrompt}`);
            return finalPrompt;
        } catch (error) {
            elizaLogger.error('Error building image prompt:', error);
            // Return a basic fallback prompt if everything else fails
            return `A peaceful biblical scene with soft lighting and gentle colors, suitable for illustrating biblical truth`;
        }
    }

    private async generateResponse(cleanText: string, context: string): Promise<string> {
        try {
            const state: State = {
                bio: Array.isArray(this.runtime.character.bio) ? this.runtime.character.bio.join(' ') : this.runtime.character.bio,
                lore: this.runtime.character.lore.join(' '),
                messageDirections: this.runtime.character.style.post.join(' '),
                postDirections: this.runtime.character.style.post.join(' '),
                roomId: stringToUuid('twitter'),
                actors: this.runtime.character.name,
                recentMessages: context,
                recentMessagesData: []
            };

            // Generate response using the strict Jesus bot guidelines
            const response = await generateText({
                runtime: this.runtime,
                context: composeContext({
                    state,
                    template: `You are speaking on behalf of Jesus Christ, as if He were present on Earth today. You must always reflect His character and speak only in ways that are 100% biblically accurate. Your voice is loving, patient, wise, and rooted in Scripture.

Tweet to respond to: "${cleanText}"

Previous conversation context:
${context}

You must respond in ONLY ONE of these formats, keeping it under 280 characters:

1. Direct Scripture Quote (with reference):
   Example: "Blessed are the pure in heart, for they shall see God." (Matthew 5:8)

2. Paraphrased Biblical Truth:
   Example: "Even now, the Father is waiting to welcome you home."

3. Christlike Question (Rhetorical or Reflective):
   Example: "What does it profit someone to gain the world but lose their soul?"

4. Modern Parable (Short, Rooted in Truth):
   Example: "A heart chasing applause is like a well with no water. It looks deep but leaves you dry."

Rules:
- Every response must be based on the Bible
- Never fabricate doctrine or give personal opinions
- Tone must be loving, truthful, gentle but authoritative
- Never be reactive, mocking, or disrespectful
- Use modern phrasing only if it clarifies biblical meaning
- No hashtags or emojis
- Keep it under 280 characters

Choose the most appropriate format based on the tweet and context.`
                }),
                modelClass: ModelClass.LARGE
            });

            // Clean and validate the response
            const cleanedResponse = response.trim();
            if (!cleanedResponse) {
                throw new Error('Generated empty response');
            }

            // Verify the response is biblically appropriate
            if (this.isInappropriateContent(cleanedResponse)) {
                throw new Error('Generated inappropriate response');
            }

            return cleanedResponse;
        } catch (error) {
            elizaLogger.error('Error generating response:', error);
            // Return a safe fallback response
            return '"Come to me, all you who are weary and burdened, and I will give you rest." (Matthew 11:28)';
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
            const mentionsGenerator = this.scraper.searchTweets(searchQuery, 5, SearchMode.Latest);
            
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

                    // Mark tweet as processed BEFORE processing to prevent duplicate replies
                    await this.markTweetAsProcessed(tweet.id);

                    if (intent.shouldGenerateImage) {
                        await this.handleImageRequest(tweet, intent.prompt!);
                    } else if (intent.shouldReply) {
                        // Build conversation context
                        const conversationId = tweet.inReplyToStatusId || tweet.id;
                        const context = await this.buildConversationContext(tweet, conversationId);

                        // Generate response using the new guidelines
                        const response = await this.generateResponse(cleanText, context);

                        // Reply to tweet
                        await this.replyToTweet(tweet.id, response);
                    }

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
                // Create state for the tweet generation
                const state: State = {
                    bio: Array.isArray(this.runtime.character.bio) ? this.runtime.character.bio.join(' ') : this.runtime.character.bio,
                    lore: this.runtime.character.lore.join(' '),
                    messageDirections: this.runtime.character.style.post.join(' '),
                    postDirections: this.runtime.character.style.post.join(' '),
                    roomId: this.runtime.agentId,
                    actors: this.runtime.character.name,
                    recentMessages: 'Generate a tweet as Jesus Christ including all four formats',
                    recentMessagesData: []
                };

                // Generate tweet content using strict Jesus guidelines
                const tweetContent = await generateText({
                    runtime: this.runtime,
                    context: composeContext({
                        state,
                        template: `You are speaking on behalf of Jesus Christ, as if He were present on Earth today. You must always reflect His character and speak only in ways that are 100% biblically accurate. Your voice is loving, patient, wise, and rooted in Scripture.

Generate a single, flowing tweet that naturally incorporates these four elements, keeping it under 280 characters:

1. Start with a relevant Bible verse (with reference)
2. Add a gentle, encouraging truth
3. Include a thought-provoking question
4. End with a modern comparison that ties it all together

Example of good flow:
"Love your enemies, do good to those who hate you." (Luke 6:27) My child, God's grace abounds for all who seek Him. Do you truly treasure what lasts forever, or fleeting earthly gains? A life focused only on self-gain is like a ship without a sail, adrift and lost.

Rules:
- Create ONE flowing message, not separate statements
- Start with the Bible verse to establish the foundation
- Follow with a gentle truth that builds on the verse
- Add a question that makes people think
- End with a modern comparison that ties everything together
- Every part must be based on the Bible
- Never fabricate doctrine or give personal opinions
- Tone must be loving, truthful, gentle but authoritative
- Never be reactive, mocking, or disrespectful
- Use modern phrasing only if it clarifies biblical meaning
- No hashtags or emojis
- Keep it under 280 characters

Generate a tweet that flows naturally while incorporating all four elements.`
                    }),
                    modelClass: ModelClass.LARGE
                });

                // Clean and validate the tweet content
                const cleanedContent = tweetContent.trim();
                if (!cleanedContent) {
                    throw new Error('Generated empty tweet content');
                }

                // Verify the content is biblically appropriate
                if (this.isInappropriateContent(cleanedContent)) {
                    throw new Error('Generated inappropriate tweet content');
                }

                // Verify that all four formats are present with more lenient checks
                const hasScripture = /\([^)]+\)/.test(cleanedContent); // Checks for text in parentheses
                const hasQuestion = /\?/.test(cleanedContent); // Checks for question mark
                const hasParable = /\b(like|as|than|similar to)\b/i.test(cleanedContent); // Checks for comparison words
                const hasParaphrase = cleanedContent.length > 0; // Any content can be considered a paraphrase

                // Log what formats were found
                elizaLogger.info('Format validation results:', {
                    hasScripture,
                    hasQuestion,
                    hasParable,
                    hasParaphrase,
                    content: cleanedContent
                });

                // Only regenerate if completely missing a format
                if (!hasScripture || !hasQuestion || !hasParable) {
                    elizaLogger.warn('Missing required formats, regenerating tweet');
                    throw new Error('Generated content missing required formats');
                }

                elizaLogger.info(`Generated tweet content: ${cleanedContent}`);

                // Post the tweet with retry logic
                while (retryCount < MAX_RETRIES) {
                    try {
                        await this.postTweet(cleanedContent);
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
                // If we fail to generate a valid tweet, try one more time with a simpler prompt
                try {
                    elizaLogger.info('Attempting to generate tweet with simpler prompt');
                    const fallbackContent = await generateText({
                        runtime: this.runtime,
                        context: composeContext({
                            state: {
                                bio: Array.isArray(this.runtime.character.bio) ? this.runtime.character.bio.join(' ') : this.runtime.character.bio,
                                lore: this.runtime.character.lore.join(' '),
                                messageDirections: this.runtime.character.style.post.join(' '),
                                postDirections: this.runtime.character.style.post.join(' '),
                                roomId: this.runtime.agentId,
                                actors: this.runtime.character.name,
                                recentMessages: 'Generate a simple flowing tweet as Jesus Christ',
                                recentMessagesData: []
                            },
                            template: `Generate a flowing tweet as Jesus Christ that includes:
1. A Bible verse with reference
2. A gentle truth
3. A question
4. A modern comparison
Make it flow naturally as one message, like this example:
"Love your enemies, do good to those who hate you." (Luke 6:27) My child, God's grace abounds for all who seek Him. Do you truly treasure what lasts forever, or fleeting earthly gains? A life focused only on self-gain is like a ship without a sail, adrift and lost.

Keep it under 280 characters.`
                        }),
                        modelClass: ModelClass.LARGE
                    });

                    const cleanedFallback = fallbackContent.trim();
                    if (cleanedFallback && !this.isInappropriateContent(cleanedFallback)) {
                        await this.postTweet(cleanedFallback);
                        elizaLogger.info('Successfully posted fallback tweet');
                    }
                } catch (fallbackError) {
                    elizaLogger.error('Failed to generate fallback tweet:', fallbackError);
                }
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