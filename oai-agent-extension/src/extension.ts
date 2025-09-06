// The module 'vscode' contains the VS Code extensibility API
// Import the module and reference it with the alias vscode in your code below
import * as vscode from 'vscode';

// This method is called when your extension is activated
// Your extension is activated the very first time the command is executed

export function activate(context: vscode.ExtensionContext) {
	console.log('OAI Chat extension is now active!');

	// Simple intent recognizer for chit-chat/greetings
	function recognizeIntent(query: string): 'courtesy' | 'technical' {
		const courtesyRegex = /^(hi|hello|hey|thanks|thank you|good (morning|afternoon|evening|night)|bye|goodbye|see you|how are you|what's up|sup|yo|greetings|nice to meet you|good to see you|good day)[.!\s]*$/i;
		if (courtesyRegex.test((query || '').trim())) {
			return 'courtesy';
		}
		// Add more rules or integrate LUIS/OpenAI here
		return 'technical';
	}

	class AzureAIChatViewProvider implements vscode.WebviewViewProvider {
		public static readonly viewType = 'azureAIChatView';
		private _view?: vscode.WebviewView;

		constructor(private readonly _context: vscode.ExtensionContext) { }

		resolveWebviewView(webviewView: vscode.WebviewView) {
			this._view = webviewView;
			webviewView.webview.options = {
				enableScripts: true
			};
			webviewView.webview.html = getWebviewContent();

			// Handle messages from the webview
			const outputChannel = vscode.window.createOutputChannel('OAI Chat');
			webviewView.webview.onDidReceiveMessage(async (message) => {
				if (message.command === 'sendQuery') {
					// Recognize intent
					const intent = recognizeIntent(message.query);
					// Get config for both OpenAI and Azure Search
					const config = vscode.workspace.getConfiguration('azureAIChat');
					const endpoint = config.get('openAIApiEndpoint', '');
					const apiKey = config.get('openAIApiKey', '');
					const apiVersion = config.get('openAIApiVersion', '2024-02-15-preview');
					const deployment = config.get('openAIDeployment', '');
					const searchEndpoint = config.get('searchApiEndpoint', '');
					const searchKey = config.get('searchApiKey', '');
					const searchIndex = config.get('searchIndex', '');
					const topSearchResult = config.get('topSearchResult', 5);
					const configSystemPrompt = config.get('systemPrompt', '');
					let aiResponse = '';
					let searchSnippet = '';
					let searchError = '';
					// Change this field name to match your Azure Search index field for content
					const AZURE_SEARCH_CONTENT_FIELD = 'content'; // e.g., 'content', 'text', 'body', etc.
					outputChannel.appendLine(`[OAI] User query: ${message.query}, intent: ${intent}`);

					if (intent !== 'courtesy' && searchEndpoint && searchKey && searchIndex) {
						try {
							const searchUrl = `${searchEndpoint}/indexes/${searchIndex}/docs/search?api-version=2023-07-01-Preview`;
							outputChannel.appendLine(`[OAI] Azure Search request: ${searchUrl}`);
							const searchRes = await fetch(searchUrl, {
								method: 'POST',
								headers: {
									'Content-Type': 'application/json',
									'api-key': searchKey
								},
								body: JSON.stringify({
									search: message.query,
									top: topSearchResult
								})
							});
							if (!searchRes.ok) throw new Error(await searchRes.text());
							// Process search results
							const searchData = await searchRes.json() as any;
							if (searchData.value && searchData.value.length > 0) {
								// Concatenate up to 5 results for context
								searchSnippet = searchData.value.map((doc: any, i: number) => {
									// Use the configured field, fallback to text, then JSON
									return `Result ${i + 1}:\n` + (doc[AZURE_SEARCH_CONTENT_FIELD] || doc.text || doc.body || JSON.stringify(doc));
								}).join('\n---\n');
								outputChannel.appendLine(`[OAI] Azure Search context snippet: ${searchSnippet}`);
							} else {
								outputChannel.appendLine(`[OAI] Azure Search returned no results.`);
							}
						} catch (err: any) {
							searchError = `Azure Search error: ${err.message || err}`;
							outputChannel.appendLine(`[OAI] Azure Search error: ${searchError}`);
						}
					}
					if (endpoint && apiKey && deployment) {
						try {
							// Make the system prompt more explicit for the LLM
							let systemPrompt = configSystemPrompt;
							if (searchSnippet) {
								systemPrompt += `\n\n[BEGIN AZURE AI SEARCH CONTEXT]\n${searchSnippet}\n[END AZURE AI SEARCH CONTEXT]`;
							}
							outputChannel.appendLine(`[OAI] System prompt sent to OpenAI:\n${systemPrompt}`);
							const url = `${endpoint}/openai/deployments/${deployment}/chat/completions?api-version=${apiVersion}`;
							const res = await fetch(url, {
								method: 'POST',
								headers: {
									'Content-Type': 'application/json',
									'api-key': apiKey
								},
								body: JSON.stringify({
									messages: [
										{ role: 'system', content: systemPrompt },
										{ role: 'user', content: message.query }
									],
									max_tokens: 512
								})
							});
							if (!res.ok) throw new Error(await res.text());
							const data = await res.json() as any;
							aiResponse = data.choices?.[0]?.message?.content || '(No response)';
						} catch (err: any) {
							aiResponse = `Error: ${err.message || err}`;
						}
					} else {
						aiResponse = `Echo: ${message.query}`;
					}
					webviewView.webview.postMessage({
						command: 'response',
						text: aiResponse,
						markdown: !aiResponse.startsWith('Error:')
					});
				} else if (message.command === 'attachFile') {
					webviewView.webview.postMessage({
						command: 'fileAttached',
						text: `File attached: ${message.fileName}`
					});
				} else if (message.command === 'showHelp') {
					vscode.window.showInformationMessage('Help/Uninstall clicked.');
				}
			});
		}
	}

	// Top-level function
	function getWebviewContent(): string {
		return `<!DOCTYPE html>
	<html lang="en">
	<head>
		<meta charset="UTF-8">
		<meta name="viewport" content="width=device-width, initial-scale=1.0">
		<title>Azure AI Chat</title>
		<style>
			body { font-family: 'Segoe UI', Arial, sans-serif; margin: 0; padding: 0; background: #1e1e1e; color: #d4d4d4; }
			#container { display: flex; flex-direction: column; height: 100vh; }
			#chat { flex: 1 1 auto; overflow-y: auto; padding: 24px 20px 0 20px; background: #1e1e1e; display: flex; flex-direction: column; }
			.bubble { max-width: 70%; margin-bottom: 16px; padding: 12px 16px; border-radius: 16px; word-break: break-word; }
			.user { background: #0e639c; color: #fff; align-self: flex-end; border-bottom-right-radius: 4px; }
			.ai { background: #23272e; color: #d4d4d4; align-self: flex-start; border-bottom-left-radius: 4px; }
			#inputBar { flex-shrink: 0; display: flex; align-items: center; padding: 16px 20px; background: #23272e; border-top: 1px solid #333; position: sticky; bottom: 0; z-index: 2; }
			#query { flex: 1; padding: 10px; border-radius: 8px; border: none; background: #2d2d2d; color: #d4d4d4; font-size: 1em; }
			#send { margin-left: 10px; background: none; border: none; cursor: pointer; }
			#send svg { width: 24px; height: 24px; fill: #0e639c; }
			#file { margin-left: 10px; display: none; }
			#attachLabel { margin-left: 10px; cursor: pointer; display: flex; align-items: center; }
			#attachLabel svg { width: 22px; height: 22px; fill: #b5cea8; }
			#fileStatus { font-size: 0.9em; color: #b5cea8; margin-left: 10px; }
		</style>
	</head>
	<body>
		<div id="container">
			<div style="display: flex; align-items: center; justify-content: space-between; padding: 8px 20px 0 20px;">
				<span style="font-weight: bold; font-size: 1.1em;">OAI Chat</span>
				<div>
					<button id="helpBtn" title="Help / Uninstall" style="background: none; border: none; cursor: pointer; padding: 4px; margin-right: 8px;">
						<svg viewBox="0 0 24 24" width="22" height="22" fill="#b5cea8"><circle cx="12" cy="12" r="10" stroke="#b5cea8" stroke-width="2" fill="none"/><text x="12" y="17" text-anchor="middle" font-size="14" fill="#b5cea8">?</text></svg>
					</button>
					<button id="clearChat" title="Clear chat" style="background: none; border: none; cursor: pointer; padding: 4px;">
						<svg viewBox="0 0 24 24" width="22" height="22" fill="#b5cea8"><path d="M6 19c0 1.1.9 2 2 2h8c1.1 0 2-.9 2-2V7H6v12zM19 4h-3.5l-1-1h-5l-1 1H5v2h14V4z"/><line x1="8" y1="11" x2="16" y2="11" stroke="#b5cea8" stroke-width="2"/><line x1="10" y1="15" x2="14" y2="15" stroke="#b5cea8" stroke-width="2"/></svg>
					</button>
				</div>
			</div>
			<div id="chat"></div>
			<div id="inputBar">
				<input id="query" type="text" placeholder="Type your query here..." />
				<input id="file" type="file" />
				<label id="attachLabel" for="file" title="Attach file">
					<svg viewBox="0 0 24 24"><path d="M17.657 11.657l-5.657 5.657a4 4 0 01-5.657-5.657l7.071-7.071a2 2 0 112.828 2.828l-7.071 7.071a.5.5 0 01-.707-.707l7.071-7.071 1.414 1.414-7.071 7.071a2.5 2.5 0 003.536 3.536l5.657-5.657 1.414 1.414z"/></svg>
				</label>
				<button id="send" title="Send">
					<svg viewBox="0 0 24 24"><path d="M2 21l21-9-21-9v7l15 2-15 2z"/></svg>
				</button>
				<span id="fileStatus"></span>
			</div>
		</div>
				<script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
				<script src="https://unpkg.com/adaptivecards@2.11.0/dist/adaptivecards.min.js"></script>
				<script>
					const vscode = acquireVsCodeApi();
					let fileContent = '';
					let chatHistory = [];

					function renderChat() {
  const chat = document.getElementById('chat');
  chat.innerHTML = '';
  chatHistory.forEach(msg => {
    const div = document.createElement('div');
    div.className = 'bubble ' + (msg.role === 'user' ? 'user' : 'ai');
    if (msg.role === 'ai') {
      if (msg.adaptiveCard) {
        // Render Adaptive Card
        const ac = new AdaptiveCards.AdaptiveCard();
        ac.parse(msg.adaptiveCard);
        const renderedCard = ac.render();
        div.appendChild(renderedCard);
      } else if (msg.markdown) {
        try {
          div.innerHTML = marked.parse(msg.text || '');
        } catch (err) {
          div.innerText = msg.text || '';
        }
      } else {
        div.innerText = msg.text;
      }
    } else {
      div.innerText = msg.text;
    }
    chat.appendChild(div);
  });
  chat.scrollTop = chat.scrollHeight;
}

					function sendQuery() {
						const query = document.getElementById('query').value;
						if (!query) return;
						chatHistory.push({ role: 'user', text: query });
						renderChat();
						document.getElementById('query').value = '';
						vscode.postMessage({ command: 'sendQuery', query, fileContent });
					}

					document.getElementById('send').addEventListener('click', sendQuery);
					document.getElementById('query').addEventListener('keydown', (e) => {
						if (e.key === 'Enter' && !e.shiftKey) {
							e.preventDefault();
							sendQuery();
						}
					});
					document.getElementById('file').addEventListener('change', (event) => {
						const file = event.target.files[0];
						if (file) {
							const reader = new FileReader();
							reader.onload = function(e) {
								fileContent = e.target.result;
								vscode.postMessage({ command: 'attachFile', fileName: file.name });
								document.getElementById('fileStatus').innerText = 'File attached: ' + file.name;
							};
							reader.readAsText(file);
						}
					});
					window.addEventListener('message', event => {
						const message = event.data;
						if (message.command === 'response') {
							chatHistory.push({ role: 'ai', text: message.text, markdown: message.markdown });
							renderChat();
						}
						if (message.command === 'adaptiveCard') {
							chatHistory.push({ role: 'ai', adaptiveCard: message.card });
							renderChat();
						}
						if (message.command === 'fileAttached') {
							document.getElementById('fileStatus').innerText = message.text;
						}
					});
					document.getElementById('clearChat').addEventListener('click', () => {
						chatHistory = [];
						renderChat();
					});
					document.getElementById('helpBtn').addEventListener('click', () => {
						vscode.postMessage({ command: 'showHelp' });
					});
				</script>
				</body>


					</html>
					`;
	}

	// Register the view provider so the sidebar appears
	context.subscriptions.push(
		vscode.window.registerWebviewViewProvider(
			AzureAIChatViewProvider.viewType,
			new AzureAIChatViewProvider(context)
		)
	);
}

