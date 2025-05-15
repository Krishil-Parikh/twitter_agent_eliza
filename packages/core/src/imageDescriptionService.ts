import { IImageDescriptionService, Service, ServiceType } from "./types";
import { IAgentRuntime } from "./types";
import { elizaLogger } from "./logger";
import OpenAI from "openai";

export class ImageDescriptionService extends Service implements IImageDescriptionService {
    private openai: OpenAI | null = null;

    get serviceType(): ServiceType {
        return ServiceType.IMAGE_DESCRIPTION;
    }

    async initialize(runtime: IAgentRuntime): Promise<void> {
        const apiKey = runtime.getSetting("OPENAI_API_KEY");
        if (!apiKey) {
            elizaLogger.warn("OPENAI_API_KEY not set - image description service will not be available");
            return;
        }
        this.openai = new OpenAI({ apiKey });
    }

    async describeImage(imageUrl: string): Promise<{ title: string; description: string }> {
        if (!this.openai) {
            throw new Error("Image description service not initialized - check OPENAI_API_KEY setting");
        }

        try {
            const response = await this.openai.chat.completions.create({
                model: "gpt-4-vision-preview",
                messages: [
                    {
                        role: "user",
                        content: [
                            { type: "text", text: "Describe this image in detail. First provide a short title, then a detailed description." },
                            { 
                                type: "image_url", 
                                image_url: {
                                    url: imageUrl
                                }
                            }
                        ]
                    }
                ],
                max_tokens: 300
            });

            const description = response.choices[0]?.message?.content || "";
            const [title, ...descriptionParts] = description.split("\n");
            
            return {
                title: title.trim(),
                description: descriptionParts.join("\n").trim()
            };
        } catch (error) {
            elizaLogger.error("Error describing image:", error);
            throw error;
        }
    }
} 