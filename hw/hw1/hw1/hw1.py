import reflex as rx
from typing import List, Union
from collections import Counter, defaultdict
import itertools
import pandas as pd
import email
import re


class PrefixTreeNode:
    def __init__(self):
        self.children: dict[str, PrefixTreeNode] = {}
        self.is_end_of_word = False


class PrefixTree:
    def __init__(self, vocabulary: List[str]):
        """
        vocabulary: список всех уникальных токенов в корпусе
        """
        self.root = PrefixTreeNode()
        
        for word in vocabulary:
            node = self.root
            for char in word:
                if char not in node.children:
                    node.children[char] = PrefixTreeNode()
                node = node.children[char]
            node.is_end_of_word = True

    def search_prefix(self, prefix) -> List[str]:
        """
        Возвращает все слова, начинающиеся на prefix
        prefix: str – префикс слова
        """

        node = self.root
        
        for char in prefix:
            if char not in node.children:
                return []
            node = node.children[char]
            
        results = []
        def _dfs(current_node, current_word):
            if current_node.is_end_of_word:
                results.append(current_word)
            
            for char, next_node in current_node.children.items():
                _dfs(next_node, current_word + char)
        _dfs(node, prefix)
        return results


class WordCompletor:
    def __init__(self, corpus):
        """
        corpus: list – корпус текстов
        """
        flat_tokens = list(itertools.chain.from_iterable(corpus))
        total_count = len(flat_tokens)
        counts = Counter(flat_tokens)
        self.token_probs = {
            word: count / total_count 
            for word, count in counts.items() 
        }
        self.prefix_tree = PrefixTree(self.token_probs.keys())

    def get_words_and_probs(self, prefix: str) -> (List[str], List[float]):
        """
        Возвращает список слов, начинающихся на prefix,
        с их вероятностями (нормировать ничего не нужно)
        """
        words, probs = [], []
        words = self.prefix_tree.search_prefix(prefix)
        words.sort(key=lambda x: self.token_probs[x], reverse=True)
        probs = [self.token_probs[x] for x in words]
        return words, probs


class NGramLanguageModel:
    def __init__(self, corpus, n):
        self.n = n
        self.ngrams = defaultdict(Counter)

        for sentence in corpus:
            if len(sentence) < n + 1:
                continue
            
            for i in range(len(sentence) - n):
                context = tuple(sentence[i : i + n])
                target = sentence[i + n]
                self.ngrams[context][target] += 1

    def get_next_words_and_probs(self, prefix: list) -> (List[str], List[float]):
        """
        Возвращает список слов, которые могут идти после prefix,
        а так же список вероятностей этих слов
        """

        next_words, probs = [], []
        if len(prefix) < self.n or tuple(prefix[-self.n:]) not in self.ngrams:
            return [], []

        next_word_counts = self.ngrams[tuple(prefix[-self.n:])]
        total_count = sum(next_word_counts.values())
        sorted_items = sorted(next_word_counts.items(), key=lambda item: item[1], reverse=True)

        next_words, counts = zip(*sorted_items)
        probs = [count / total_count for count in counts]
        return next_words, probs


class TextSuggestion:
    def __init__(self, word_completor, n_gram_model):
        self.word_completor = word_completor
        self.n_gram_model = n_gram_model

    def suggest_text(self, tokens: list, n_words=3, n_branching=3, complete_last_word=True) -> list[str]:
        """
        Модифицированный метод:
        Возвращает плоский список строк-кандидатов (фраз), которые можно подставить.
        n_branching: сколько вариантов автодополнения слова рассматривать.
        """
        if not tokens:
             return []

        prefix = tokens[-1]
        history = tokens[:-1]

        if complete_last_word:
            comp_words, _ = self.word_completor.get_words_and_probs(prefix)
            candidates = comp_words[:n_branching]
        else:
            candidates = [prefix]
        results = []

        for start_word in candidates:
            current_sequence = [start_word] if complete_last_word else []
            current_context = history + [start_word]
            
            for _ in range(n_words):
                next_words, _ = self.n_gram_model.get_next_words_and_probs(current_context)
                if not next_words:
                    break
                best_next = next_words[0]
                current_sequence.append(best_next)
                current_context.append(best_next)
            
            results.append(" ".join(current_sequence))
        
        return results


def get_email_body(raw_text):
    msg = email.message_from_string(raw_text)    
    payload = msg.get_payload(decode=True)
    charset = msg.get_content_charset() or 'utf-8'
    body = payload.decode(charset)
    return body.strip()

def get_body_column(messages):
    column = []
    for message in messages:
        body = get_email_body(message)
        body = re.sub(r'\t', ' ', body)
        body = re.sub(r'[^A-Za-z\n ,.]', '', body)
        body = re.sub(r'\n+', '\n', body)
        body = re.sub(r'\.', ' ', body)
        body = re.sub(r'\,', ' ', body)
        body = re.sub(r' +', ' ', body).strip()
        column.append(body)
    return column

def tokenize(text):
    return re.compile(r'\w+|,|\.').findall(text.lower())


def get_corpus():
    emails = pd.read_csv('../emails.csv')
    emails['body'] = get_body_column(emails['message'])
    email_tokens = [tokenize(mail) for mail in emails['body']]
    return email_tokens


clean_corpus = get_corpus()
print("corpus ready")
completor = WordCompletor(clean_corpus)
print("completor ready")
ngram = NGramLanguageModel(clean_corpus, n=1)
print("ngram ready")
suggester = TextSuggestion(completor, ngram)
print("suggester ready")


class TextState(rx.State):
    text_value: str = ""
    recommendations: List[str] = []

    def handle_text_change(self, val: str):
        self.text_value = val

        tokens = val.lower().split()
        if not tokens:
            self.recommendations = []
            return
        complete_last_word = not val.endswith(" ")
            
        suggestions = suggester.suggest_text(tokens, n_words=2, n_branching=4, complete_last_word=complete_last_word)
        self.recommendations = suggestions

    def accept_suggestion(self, suggestion: str):
        last_space_index = self.text_value.rfind(' ')
        
        k = len(self.text_value) - (last_space_index + 1)
        suggestion_suffix = suggestion[k:]
        
        self.handle_text_change(self.text_value + suggestion_suffix + " ")


def index():
    return rx.vstack(
        rx.heading("Autocomplete", size="8", margin_bottom="20px", color="#4287f5"),

        rx.box(
            rx.input(
                value=TextState.text_value,
                on_change=TextState.handle_text_change,
                width="100%",
                height="100%",
                font_size="1.5em",
            ),
            width="600px",
            box_shadow="lg",
        ),
        
        rx.cond(
            TextState.recommendations.length() > 0,
            rx.vstack(
                rx.flex(
                    rx.foreach(
                        TextState.recommendations,
                        lambda rec: rx.button(
                            rec,
                            on_click=lambda: TextState.accept_suggestion(rec),
                            variant="outline",
                            color_scheme="blue",
                            margin="5px",
                            cursor="pointer",
                        )
                    ),
                    wrap="wrap",
                    justify="center",
                    width="600px"
                ),
                align_items="center"
            )
        ),
        
        align_items="center",
        padding_top="20%",
        width="100%",
        height="100vh",
        background_color="#252c36"
    )


app = rx.App()
app.add_page(index, title="Autocomplete")
