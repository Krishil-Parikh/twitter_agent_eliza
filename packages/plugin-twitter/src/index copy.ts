import type { Plugin } from "@elizaos/core";
import { postAction } from "./actions/post";
import { mentionAction } from "./actions/mention";
import { searchAction } from "./actions/search";

export const twitterPlugin: Plugin = {
    name: "twitter",
    description: "Twitter integration plugin for posting tweets and handling mentions",
    actions: [postAction, mentionAction, searchAction],
    evaluators: [],
    providers: [],
};

export default twitterPlugin;
