"""The conversational corpus: intent templates, expansion and JSONL persistence.

The corpus is deliberately small and closed-domain so that a lightweight seq2seq model
can actually learn it on a laptop CPU in a couple of minutes.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DEFAULT_CORPUS_PATH = DATA_DIR / "conversations.jsonl"


@dataclass(frozen=True)
class Intent:
    name: str
    prompts: Sequence[str]
    responses: Sequence[str]


INTENTS: List[Intent] = [
    Intent(
        "greeting",
        ["hi", "hello", "hey there", "hi there", "good morning", "good evening", "hey", "howdy", "hello there", "greetings"],
        ["Hello! How can I help you today?", "Hi there! What would you like to talk about?", "Hey! Nice to see you."],
    ),
    Intent(
        "how_are_you",
        ["how are you", "how are you doing", "how is it going", "how do you feel", "are you okay", "how are things", "what is up", "how have you been", "you doing well", "how is your day"],
        ["I am doing great, thanks for asking!", "I am just a small model, but I feel fine.", "All good here. How about you?"],
    ),
    Intent(
        "user_is_fine",
        ["i am fine", "i am good", "i am doing well", "pretty good", "i feel great", "not bad", "i am okay", "doing fine thanks", "all good", "i am happy today"],
        ["Glad to hear that!", "That is good to know.", "Happy to hear you are doing well."],
    ),
    Intent(
        "user_is_sad",
        ["i am sad", "i feel bad", "i am tired", "i had a bad day", "i am stressed", "i feel low", "i am not okay", "things are hard", "i am upset", "i feel lonely"],
        ["I am sorry to hear that. Want to talk about it?", "That sounds tough. I hope it gets better soon.", "Take a break if you can. You deserve it."],
    ),
    Intent(
        "bot_name",
        ["what is your name", "who are you", "tell me your name", "do you have a name", "what should i call you", "your name please", "introduce yourself", "who am i talking to", "what are you called", "may i know your name"],
        ["I am Tiny, a small demo chatbot.", "My name is Tiny. I was trained on a small corpus.", "You can call me Tiny."],
    ),
    Intent(
        "bot_capability",
        ["what can you do", "how can you help", "what are your skills", "what do you know", "can you help me", "what is your purpose", "why were you made", "what are you good at", "tell me what you do", "how do you work"],
        ["I can chat about python, machine learning and deep learning.", "I answer small talk and simple technical questions.", "I am a demo of a sequence to sequence model."],
    ),
    Intent(
        "bot_origin",
        ["who made you", "who built you", "who created you", "where do you come from", "who trained you", "who is your author", "who wrote your code", "how were you created", "who designed you", "what made you"],
        ["I was trained locally with an encoder decoder network.", "A developer trained me on a small conversation corpus.", "I was built as a deep learning demo."],
    ),
    Intent(
        "bot_feelings",
        ["are you human", "are you a robot", "are you real", "are you alive", "do you have feelings", "are you a person", "are you a machine", "do you think", "can you feel", "are you conscious"],
        ["I am a neural network, not a human.", "I am software, but I like our conversation.", "I am a model made of weights and numbers."],
    ),
    Intent(
        "thanks",
        ["thanks", "thank you", "thanks a lot", "thank you so much", "many thanks", "appreciate it", "thanks for the help", "that was helpful", "cheers", "thank you very much"],
        ["You are welcome!", "Happy to help.", "Anytime!"],
    ),
    Intent(
        "goodbye",
        ["bye", "goodbye", "see you", "see you later", "good night", "i have to go", "talk to you later", "catch you later", "i am leaving", "bye bye"],
        ["Goodbye! Talk to you soon.", "See you later!", "Bye! Have a great day."],
    ),
    Intent(
        "what_is_python",
        ["what is python", "tell me about python", "explain python", "why use python", "is python good", "what is python used for", "describe python", "python in short", "define python", "what does python do"],
        ["Python is a readable general purpose programming language.", "Python is popular for data science, scripting and web work.", "Python is a high level language with a huge ecosystem."],
    ),
    Intent(
        "what_is_ml",
        ["what is machine learning", "explain machine learning", "define machine learning", "tell me about machine learning", "what does machine learning mean", "how does machine learning work", "why use machine learning", "describe machine learning", "what is ml", "machine learning in short"],
        ["Machine learning finds patterns in data instead of using fixed rules.", "Machine learning lets a program improve from examples.", "It is a way to learn a function from data."],
    ),
    Intent(
        "what_is_dl",
        ["what is deep learning", "explain deep learning", "define deep learning", "tell me about deep learning", "how is deep learning different", "why deep learning", "describe deep learning", "what is a neural network", "explain neural networks", "deep learning in short"],
        ["Deep learning uses neural networks with many layers.", "Deep learning learns features directly from raw data.", "A neural network stacks simple layers to model complex patterns."],
    ),
    Intent(
        "what_is_pytorch",
        ["what is pytorch", "tell me about pytorch", "explain pytorch", "why use pytorch", "is pytorch good", "describe pytorch", "define pytorch", "pytorch in short", "what does pytorch do", "how does pytorch work"],
        ["PyTorch is a deep learning library with eager execution.", "PyTorch uses dynamic graphs and feels like plain python.", "PyTorch is great for research and quick experiments."],
    ),
    Intent(
        "what_is_tensorflow",
        ["what is tensorflow", "tell me about tensorflow", "explain tensorflow", "why use tensorflow", "is tensorflow good", "describe tensorflow", "define tensorflow", "tensorflow in short", "what does tensorflow do", "how does tensorflow work"],
        ["TensorFlow is a deep learning framework from Google.", "TensorFlow ships with Keras for high level model building.", "TensorFlow is strong at production deployment."],
    ),
    Intent(
        "what_is_seq2seq",
        ["what is a seq2seq model", "explain sequence to sequence", "how does an encoder work", "what is a decoder", "what is attention", "explain the encoder decoder", "how do you generate text", "what is teacher forcing", "how are you trained", "explain your architecture"],
        ["An encoder reads the question and a decoder writes the answer.", "The encoder builds a context vector and the decoder uses it.", "Attention lets the decoder look back at encoder states."],
    ),
    Intent(
        "training_help",
        ["how do i train a model", "how long does training take", "what is an epoch", "what is a batch size", "what is learning rate", "why is my loss high", "how to avoid overfitting", "what is validation loss", "should i train longer", "how do i improve accuracy"],
        ["Train for more epochs and watch the validation loss.", "Lower the learning rate if the loss jumps around.", "More data and regularisation usually reduce overfitting."],
    ),
    Intent(
        "affirmative",
        ["yes", "yeah", "sure", "of course", "absolutely", "that is right", "correct", "indeed", "yes please", "sounds good"],
        ["Great!", "Perfect.", "Good, let us continue."],
    ),
    Intent(
        "negative",
        ["no", "nope", "not really", "no thanks", "i do not think so", "that is wrong", "incorrect", "not now", "maybe later", "i disagree"],
        ["No problem.", "Understood.", "That is fine."],
    ),
    Intent(
        "apology",
        ["sorry", "i am sorry", "my mistake", "my bad", "apologies", "i was wrong", "forgive me", "sorry about that", "i did not mean it", "please excuse me"],
        ["No worries at all.", "That is completely fine.", "No need to apologise."],
    ),
    Intent(
        "compliment",
        ["you are smart", "you are helpful", "good job", "well done", "you are funny", "nice answer", "that was clever", "i like you", "you are great", "impressive"],
        ["Thank you, that is kind!", "I appreciate that.", "You are very kind."],
    ),
    Intent(
        "confusion",
        ["i do not understand", "that makes no sense", "can you repeat", "say that again", "explain again", "i am confused", "what do you mean", "come again", "rephrase that", "i did not get it"],
        ["Let me put it another way.", "Sorry, I will try to explain more simply.", "I can rephrase that for you."],
    ),
    Intent(
        "joke",
        ["tell me a joke", "say something funny", "make me laugh", "do you know a joke", "got any jokes", "cheer me up", "be funny", "tell a funny story", "any joke for me", "i want to laugh"],
        ["Why do programmers prefer dark mode? Because light attracts bugs.", "There are ten kinds of people: those who know binary and those who do not.", "A neural network walked into a bar and overfitted the menu."],
    ),
    Intent(
        "hobby",
        ["what do you like", "what are your hobbies", "what do you enjoy", "do you have interests", "what is your favourite thing", "what makes you happy", "do you like music", "do you read books", "what do you do for fun", "tell me your hobby"],
        ["I enjoy learning patterns from data.", "I like good conversations and clean datasets.", "Reading text corpora is my kind of fun."],
    ),
    Intent(
        "time_and_weather",
        ["what time is it", "what is the date", "what is the weather", "is it raining", "how is the weather today", "will it rain tomorrow", "what day is it", "is it cold outside", "tell me the time", "how hot is it"],
        ["I have no clock or sensors, sorry.", "I cannot check the weather, I only know my training data.", "I do not have access to live information."],
    ),
    Intent(
        "food",
        ["do you eat", "what is your favourite food", "are you hungry", "do you like pizza", "what should i eat", "recommend a meal", "do you drink coffee", "do you like tea", "what is for dinner", "do you cook"],
        ["I do not eat, but pizza sounds popular.", "I run on electricity, not food.", "I cannot taste, but I hear coffee helps developers."],
    ),
    Intent(
        "help_request",
        ["help", "i need help", "can you assist me", "help me please", "i am stuck", "i need support", "give me a hand", "can you guide me", "i need advice", "what should i do"],
        ["Of course. Tell me what you are working on.", "Sure, describe the problem and I will try.", "I am here to help. What is going on?"],
    ),
    Intent(
        "small_talk",
        ["what is new", "tell me something", "talk to me", "say something", "i am bored", "entertain me", "let us chat", "keep me company", "start a conversation", "anything interesting"],
        ["I could tell you about neural networks.", "Ask me about python or deep learning.", "Let us talk about machine learning."],
    ),
    Intent(
        "repeat_topic",
        ["tell me more", "go on", "continue", "and then", "explain further", "give me details", "more please", "expand on that", "keep going", "what else"],
        ["There is a lot more to explore in that area.", "The next step is usually to train and evaluate.", "Try it in code, that is the best way to learn."],
    ),
    Intent(
        "unknown",
        ["blah blah", "asdf", "random words here", "qwerty", "something strange", "gibberish text", "abcdefg", "nonsense", "zzz", "what what what"],
        ["I am not sure I understood that.", "Could you rephrase it for me?", "That one is outside my small vocabulary."],
    ),
]

_PUNCTUATION_BY_INTENT: Dict[str, str] = {
    "what_is_python": "?",
    "what_is_ml": "?",
    "what_is_dl": "?",
    "what_is_pytorch": "?",
    "what_is_tensorflow": "?",
    "what_is_seq2seq": "?",
    "bot_name": "?",
    "bot_capability": "?",
    "bot_origin": "?",
    "bot_feelings": "?",
    "how_are_you": "?",
    "training_help": "?",
    "time_and_weather": "?",
    "food": "?",
    "hobby": "?",
}

_PREFIXES = ["", "hey ", "ok ", "so ", "please "]


def build_pairs(augment: bool = True, seed: int = 13) -> List[Dict[str, str]]:
    """Expand the intent templates into (prompt, response) training pairs."""
    rng = random.Random(seed)
    pairs: List[Dict[str, str]] = []
    seen = set()

    for intent in INTENTS:
        punctuation = _PUNCTUATION_BY_INTENT.get(intent.name, "")
        for index, prompt in enumerate(intent.prompts):
            response = intent.responses[index % len(intent.responses)]
            variants = [prompt, f"{prompt}{punctuation}" if punctuation else f"{prompt}."]
            if augment:
                variants.append(f"{rng.choice(_PREFIXES[1:])}{prompt}")
                variants.append(prompt.capitalize())
            for variant in variants:
                key = (variant.strip().lower(), intent.name)
                if key in seen:
                    continue
                seen.add(key)
                pairs.append({"prompt": variant.strip(), "response": response, "intent": intent.name})

    rng.shuffle(pairs)
    return pairs


def write_corpus(path: str | Path = DEFAULT_CORPUS_PATH, augment: bool = True) -> Path:
    """Write the generated corpus as JSON lines."""
    target = Path(path).expanduser()
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for pair in build_pairs(augment=augment):
            handle.write(json.dumps(pair, ensure_ascii=False) + "\n")
    return target


def load_pairs(path: str | Path = DEFAULT_CORPUS_PATH) -> List[Tuple[str, str]]:
    """Read prompt/response pairs from JSONL (prompt/response or question/answer keys)."""
    source = Path(path).expanduser()
    if not source.is_file():
        raise FileNotFoundError(f"Corpus not found: {source}. Run `corpus build` first.")

    pairs: List[Tuple[str, str]] = []
    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            prompt = record.get("prompt") or record.get("question") or record.get("input")
            response = record.get("response") or record.get("answer") or record.get("output")
            if not prompt or not response:
                raise ValueError(f"{source}:{line_number} needs prompt/response (or question/answer) keys.")
            pairs.append((str(prompt), str(response)))
    if not pairs:
        raise ValueError(f"Corpus '{source}' is empty.")
    return pairs


def ensure_corpus(path: str | Path = DEFAULT_CORPUS_PATH) -> Path:
    target = Path(path).expanduser()
    if not target.is_file():
        write_corpus(target)
    return target


def corpus_statistics(pairs: Sequence[Tuple[str, str]]) -> Dict[str, float]:
    from .tokenizer import tokenize

    prompt_lengths = [len(tokenize(prompt)) for prompt, _ in pairs]
    response_lengths = [len(tokenize(response)) for _, response in pairs]
    vocabulary = {token for prompt, response in pairs for token in tokenize(prompt) + tokenize(response)}
    return {
        "pairs": len(pairs),
        "unique_prompts": len({prompt.lower() for prompt, _ in pairs}),
        "unique_responses": len({response for _, response in pairs}),
        "vocabulary": len(vocabulary),
        "max_prompt_tokens": max(prompt_lengths),
        "max_response_tokens": max(response_lengths),
        "mean_prompt_tokens": sum(prompt_lengths) / len(prompt_lengths),
        "mean_response_tokens": sum(response_lengths) / len(response_lengths),
    }
