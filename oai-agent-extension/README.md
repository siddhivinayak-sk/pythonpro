
# OAI Chat VS Code Extension

This extension brings a Copilot Chat-like experience to VS Code, powered by Azure OpenAI and Azure AI Search. Chat with AI, search your enterprise data, and provide file context—all in a modern chat panel.

## Features

- Chat panel UI inspired by GitHub Copilot Chat
- Query Azure AI Search and Azure OpenAI in a single workflow
- Attach files to provide additional context to the AI
- Modern chat bubbles and message history
- Easily configure endpoints and keys from the panel

## Requirements

- An Azure OpenAI resource (endpoint and API key)
- An Azure AI Search resource (endpoint, API key, and index)

## Extension Settings

This extension contributes the following settings:

- `azureAIChat.openAIApiEndpoint`: Azure OpenAI API endpoint URL
- `azureAIChat.openAIApiKey`: Azure OpenAI API key
- `azureAIChat.searchApiEndpoint`: Azure AI Search endpoint URL
- `azureAIChat.searchApiKey`: Azure AI Search API key
- `azureAIChat.searchIndex`: Azure AI Search index name

You can also edit these settings directly in the chat panel.

## Usage

1. Open the command palette and run `Open OAI Chat Panel`.
2. Configure your Azure endpoints and keys in the panel.
3. Type your question and optionally attach a file for more context.
4. View responses in a chat-like interface.

## Known Issues

- File attachments are sent as plain text and may be truncated for very large files.
- Only supports text-based files for context.

## Release Notes

### 0.0.1
- Initial release with chat UI, Azure OpenAI and AI Search integration, and file attachment support.

### 1.0.1

Fixed issue #.

### 1.1.0

Added features X, Y, and Z.

---

## Following extension guidelines

Ensure that you've read through the extensions guidelines and follow the best practices for creating your extension.

* [Extension Guidelines](https://code.visualstudio.com/api/references/extension-guidelines)

## Working with Markdown

You can author your README using Visual Studio Code. Here are some useful editor keyboard shortcuts:

* Split the editor (`Cmd+\` on macOS or `Ctrl+\` on Windows and Linux).
* Toggle preview (`Shift+Cmd+V` on macOS or `Shift+Ctrl+V` on Windows and Linux).
* Press `Ctrl+Space` (Windows, Linux, macOS) to see a list of Markdown snippets.

## For more information

* [Visual Studio Code's Markdown Support](http://code.visualstudio.com/docs/languages/markdown)
* [Markdown Syntax Reference](https://help.github.com/articles/markdown-basics/)

**Enjoy!**
