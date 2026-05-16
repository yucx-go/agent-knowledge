"""Create a small synthetic LongMemEval-S-format test dataset for local testing.

This generates 10 questions with realistic session structures to validate
the benchmark script works correctly before running on the full dataset.
"""

import json
import os
import random


def make_session(topic: str, n_messages: int = 10) -> list[dict]:
    """Generate a realistic conversation session about a topic."""
    messages = []
    user_prompts = [
        f"Can you tell me about {topic}?",
        f"What are the key aspects of {topic}?",
        f"How does {topic} compare to alternatives?",
        f"I've been thinking about {topic} lately.",
        f"Let's discuss {topic} in more detail.",
    ]
    assistant_responses = [
        f"{topic} is an interesting subject. Here are some key points...",
        f"Regarding {topic}, there are several important considerations.",
        f"The main advantage of {topic} is its versatility and effectiveness.",
        f"When considering {topic}, you should keep in mind the following factors.",
        f"I'd recommend looking into {topic} from multiple perspectives.",
    ]
    for i in range(n_messages):
        if i % 2 == 0:
            messages.append({"role": "user", "content": random.choice(user_prompts)})
        else:
            messages.append({"role": "assistant", "content": random.choice(assistant_responses)})
    return messages


def make_question(qid: int, qtype: str, topic: str, n_sessions: int = 20) -> dict:
    """Generate a single LongMemEval-S format question."""
    sessions = []
    session_ids = []

    # Generate haystack sessions
    distractors = [
        "cooking recipes", "travel plans", "movie reviews", "book recommendations",
        "exercise routines", "home decoration", "gardening tips", "music preferences",
        "pet care", "career advice", "language learning", "photography tips",
        "financial planning", "meditation practices", "hobby crafts",
        "technology trends", "fashion advice", "sports analysis", "board games",
    ]

    # Place the gold session at a random position
    gold_idx = random.randint(0, n_sessions - 1)

    for i in range(n_sessions):
        sid = f"session_{i+1}"
        session_ids.append(sid)
        if i == gold_idx:
            sessions.append(make_session(topic, n_messages=12))
        else:
            sessions.append(make_session(random.choice(distractors), n_messages=8))

    return {
        "question_id": f"test_{qid:03d}",
        "question_type": qtype,
        "question": f"What did we discuss about {topic}?",
        "answer": f"We discussed {topic} in detail.",
        "haystack_sessions": sessions,
        "haystack_session_ids": session_ids,
        "answer_session_ids": [f"session_{gold_idx + 1}"],
    }


def main():
    random.seed(42)

    questions = [
        make_question(1, "single-session-user", "Python programming", 20),
        make_question(2, "single-session-user", "machine learning", 20),
        make_question(3, "multi-session", "React framework", 25),
        make_question(4, "multi-session", "database optimization", 25),
        make_question(5, "temporal-reasoning", "project deadlines", 20),
        make_question(6, "temporal-reasoning", "meeting schedules", 20),
        make_question(7, "knowledge-update", "API version changes", 20),
        make_question(8, "knowledge-update", "team structure changes", 20),
        make_question(9, "single-session-user", "Docker containers", 15),
        make_question(10, "single-session-user", "TypeScript types", 15),
        # Add 2 abstention to test filtering
        make_question(11, "abstention", "nonexistent topic xyz", 10),
        make_question(12, "abstention", "another nonexistent abc", 10),
    ]

    output_path = os.path.join(os.path.dirname(__file__), "data", "longmemeval_s_test.json")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(questions, f, indent=2, ensure_ascii=False)

    print(f"Created test dataset: {output_path}")
    print(f"  Total questions: {len(questions)}")
    print(f"  Abstention: 2 (will be skipped)")
    print(f"  Evaluatable: {len(questions) - 2}")


if __name__ == "__main__":
    main()
